from __future__ import annotations

import hashlib
import math
import time

import numpy as np

from .phase3 import slice_pack


class AnalyticallyAnchoredResidualRouter:
    """Learned residual corrections around the immutable analytical view prior.

    The model cannot replace the final predictor or preservation gate. It may
    only adjust candidate-view logits using frozen semantic embeddings and
    query-level response-bank reliability features. Both residual branches are
    zero-scaled at initialization, so the initial model is the analytical
    router up to floating-point normalization.
    """

    MODES = {
        "full", "temperature", "route_only", "reliability_only",
        "shuffle_response", "no_anchor",
    }

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        response_hidden_dim: int,
        device: str,
        seed: int,
    ):
        import torch
        from torch import nn

        torch.manual_seed(seed)
        self.torch, self.device = torch, device

        class Module(nn.Module):
            def __init__(self):
                super().__init__()
                self.task_query = nn.Linear(embedding_dim, hidden_dim)
                self.route_query = nn.Linear(embedding_dim, hidden_dim)
                self.view_key = nn.Linear(embedding_dim, hidden_dim)
                self.response_mlp = nn.Sequential(
                    nn.Linear(7, response_hidden_dim),
                    nn.GELU(),
                    nn.Linear(response_hidden_dim, 1),
                )
                self.log_temperature = nn.Parameter(torch.tensor(0.0))
                self.route_scale = nn.Parameter(torch.tensor(0.0))
                self.reliability_scale = nn.Parameter(torch.tensor(0.0))

            @staticmethod
            def _posterior_statistics(posterior):
                safe = posterior.clamp_min(1e-8)
                entropy = -(safe * safe.log()).sum(-1) / math.log(safe.shape[-1])
                top = torch.topk(posterior, 2, dim=-1).values
                return entropy, top[:, 0] - top[:, 1]

            def forward(self, batch: dict, mode: str = "full"):
                if mode not in AnalyticallyAnchoredResidualRouter.MODES:
                    raise ValueError(f"Unknown Phase 5.2 mode: {mode}")
                bank, mask = batch["view_probabilities"], batch["view_mask"]
                prior = batch["route_view_prior"].clamp_min(1e-8)
                prior = prior / prior.sum(-1, keepdim=True).clamp_min(1e-8)
                query_count = bank.shape[1]

                if mode == "no_anchor":
                    anchor = torch.zeros_like(prior)
                else:
                    anchor = prior.log()
                temperature = self.log_temperature.exp().clamp(0.25, 4.0)
                anchor = anchor[:, None, :] / temperature

                task = torch.nn.functional.normalize(self.task_query(batch["task"]), dim=-1)
                route = torch.nn.functional.normalize(self.route_query(batch["route"]), dim=-1)
                views = torch.nn.functional.normalize(self.view_key(batch["view_embeddings"]), dim=-1)
                route_delta = (
                    torch.einsum("bh,bvh->bv", task, views)
                    + torch.einsum("bh,bvh->bv", route, views)
                ) / math.sqrt(2.0)
                route_delta = route_delta[:, None, :].expand(-1, query_count, -1)

                response_bank = bank
                if mode == "shuffle_response" and len(bank) > 1:
                    response_bank = bank.roll(1, dims=0)
                bank_logit = torch.logit(response_bank.clamp(1e-5, 1 - 1e-5))
                llm_logit = torch.logit(batch["llm_probability"].clamp(1e-5, 1 - 1e-5))
                entropy, margin = self._posterior_statistics(batch["code_posterior"])
                features = torch.stack(
                    (
                        bank_logit,
                        bank_logit.abs(),
                        llm_logit[:, :, None].expand_as(bank_logit),
                        (response_bank - batch["llm_probability"][:, :, None]).abs(),
                        entropy[:, None, None].expand_as(bank_logit),
                        margin[:, None, None].expand_as(bank_logit),
                        batch["gates"][:, :, None].expand_as(bank_logit),
                    ),
                    dim=-1,
                )
                reliability_delta = self.response_mlp(features).squeeze(-1)

                correction = torch.zeros_like(bank)
                if mode in {"full", "route_only", "shuffle_response", "no_anchor"}:
                    correction = correction + torch.tanh(self.route_scale) * route_delta
                if mode in {"full", "reliability_only", "shuffle_response", "no_anchor"}:
                    correction = correction + torch.tanh(self.reliability_scale) * reliability_delta
                logits = anchor + correction
                logits = logits.masked_fill(~mask[:, None, :], torch.finfo(logits.dtype).min)
                weights = torch.softmax(logits, dim=-1)
                selected = torch.einsum("bqv,bqv->bq", weights, bank).clamp(1e-5, 1 - 1e-5)
                adapter = torch.where(batch["routed"].expand_as(selected), selected, batch["tfm_probability"])
                final = (
                    (1.0 - batch["gates"]) * batch["llm_probability"] + batch["gates"] * adapter
                ).clamp(1e-5, 1 - 1e-5)
                normalized_entropy = -(weights * weights.clamp_min(1e-8).log()).sum(-1)
                normalized_entropy = normalized_entropy / torch.log(mask.sum(-1).float().clamp_min(2))[:, None]
                return final, {
                    "weights": weights,
                    "selected_probability": selected,
                    "entropy": normalized_entropy,
                    "correction": correction,
                    "temperature": temperature.expand(len(bank)),
                    "route_scale": torch.tanh(self.route_scale).expand(len(bank)),
                    "reliability_scale": torch.tanh(self.reliability_scale).expand(len(bank)),
                }

        self.module = Module().to(device)

    @property
    def trainable_params(self) -> int:
        return sum(parameter.numel() for parameter in self.module.parameters() if parameter.requires_grad)

    def fit(
        self,
        train: dict,
        validation: dict,
        *,
        mode: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        patience: int,
        anchor_kl: float,
        correction_l2: float,
    ) -> dict:
        torch = self.torch
        if mode == "temperature":
            for name, parameter in self.module.named_parameters():
                parameter.requires_grad_(name == "log_temperature")
        optimizer = torch.optim.AdamW(
            [parameter for parameter in self.module.parameters() if parameter.requires_grad],
            lr=learning_rate,
            weight_decay=1e-4,
        )
        generator = torch.Generator(device="cpu").manual_seed(7521)
        best_loss, best_state, best_epoch, stale, max_gradient = math.inf, None, -1, 0, 0.0
        initial_probability, initial_labels, _ = self.predict(train, batch_size, mode)
        started = time.perf_counter()
        for epoch in range(epochs):
            self.module.train()
            order = torch.randperm(len(train["episode_ids"]), generator=generator).to(self.device)
            for start in range(0, len(order), batch_size):
                batch = slice_pack(train, order[start:start + batch_size])
                probability, traces = self.module(batch, mode)
                prediction_loss = torch.nn.functional.binary_cross_entropy(probability, batch["labels"])
                prior = batch["route_view_prior"].clamp_min(1e-8)
                prior = prior / prior.sum(-1, keepdim=True).clamp_min(1e-8)
                prior = prior[:, None, :].expand_as(traces["weights"])
                kl = (
                    traces["weights"]
                    * (traces["weights"].clamp_min(1e-8).log() - prior.log())
                ).sum(-1).mean()
                regularized = (
                    prediction_loss
                    + float(anchor_kl) * kl
                    + float(correction_l2) * traces["correction"].square().mean()
                )
                optimizer.zero_grad(set_to_none=True)
                regularized.backward()
                max_gradient = max(
                    max_gradient,
                    float(torch.nn.utils.clip_grad_norm_(self.module.parameters(), 1.0)),
                )
                optimizer.step()
            selection_loss = self.loss(validation, batch_size, mode)
            if selection_loss < best_loss - 1e-6:
                best_loss, best_epoch, stale = selection_loss, epoch, 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.module.state_dict().items()
                }
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_state is None:
            raise RuntimeError("Phase 5.2 AR+ did not produce a checkpoint")
        self.module.load_state_dict(best_state)
        probability, labels, traces = self.predict(train, batch_size, mode)
        return {
            "mode": mode,
            "best_epoch": best_epoch,
            "selection_loss": best_loss,
            "validation_loss": self.loss(validation, batch_size, mode),
            "initial_train_loss": _binary_loss(initial_labels, initial_probability),
            "train_loss": _binary_loss(labels, probability),
            "train_accuracy": float(np.mean((probability >= 0.5) == labels)),
            "max_gradient_norm": max_gradient,
            "mean_absolute_correction": float(np.mean(np.abs(traces["correction"]))),
            "temperature": float(np.mean(traces["temperature"])),
            "route_scale": float(np.mean(traces["route_scale"])),
            "reliability_scale": float(np.mean(traces["reliability_scale"])),
            "elapsed_seconds": time.perf_counter() - started,
        }

    def loss(self, data: dict, batch_size: int, mode: str) -> float:
        probability, labels, _ = self.predict(data, batch_size, mode)
        return _binary_loss(labels, probability)

    def predict(self, data: dict, batch_size: int, mode: str):
        torch = self.torch
        probabilities, labels = [], []
        parts = {
            key: [] for key in (
                "weights", "selected_probability", "entropy", "correction",
                "temperature", "route_scale", "reliability_scale",
            )
        }
        self.module.eval()
        with torch.inference_mode():
            for start in range(0, len(data["episode_ids"]), batch_size):
                indices = torch.arange(
                    start, min(start + batch_size, len(data["episode_ids"])), device=self.device,
                )
                batch = slice_pack(data, indices)
                probability, traces = self.module(batch, mode)
                probabilities.append(probability.cpu().numpy().reshape(-1))
                labels.append(batch["labels"].cpu().numpy().reshape(-1))
                for key, value in traces.items():
                    parts[key].append(value.cpu().numpy())
        return np.concatenate(probabilities), np.concatenate(labels), {
            key: np.concatenate(values, axis=0) for key, values in parts.items()
        }

    def state_digest(self) -> str:
        digest = hashlib.sha256()
        for key, value in sorted(self.module.state_dict().items()):
            digest.update(key.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()


def response_bank_oracle(data: dict):
    """Label-leaking episode-level ceiling diagnostic; never a model result."""
    torch = __import__("torch")
    bank, labels, mask = data["view_probabilities"], data["labels"], data["view_mask"]
    correctness = ((bank >= 0.5) == labels[:, :, None]).float().sum(1)
    correctness = correctness.masked_fill(~mask, -1)
    selected_index = correctness.argmax(-1)
    selected = bank.gather(2, selected_index[:, None, None].expand(-1, bank.shape[1], 1)).squeeze(-1)
    adapter = torch.where(data["routed"].expand_as(selected), selected, data["tfm_probability"])
    final = ((1.0 - data["gates"]) * data["llm_probability"] + data["gates"] * adapter).clamp(1e-5, 1 - 1e-5)
    weights = torch.nn.functional.one_hot(selected_index, bank.shape[-1]).to(bank.dtype)
    weights = weights[:, None, :].expand(-1, bank.shape[1], -1)
    return final.cpu().numpy().reshape(-1), labels.cpu().numpy().reshape(-1), {
        "weights": weights.cpu().numpy(),
        "selected_probability": selected.cpu().numpy(),
        "entropy": np.zeros_like(selected.cpu().numpy()),
        "correction": np.zeros_like(bank.cpu().numpy()),
    }


def _binary_loss(labels: np.ndarray, probability: np.ndarray) -> float:
    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    return float(-np.mean(labels * np.log(probability) + (1 - labels) * np.log(1 - probability)))
