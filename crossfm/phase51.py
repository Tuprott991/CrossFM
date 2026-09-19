from __future__ import annotations

import hashlib
import math
import time

import numpy as np

from .phase3 import slice_pack


class StructuredPosteriorBridge:
    """Two-beat bridge over soft code tokens without the analytical view map.

    Round one uses only the task and candidate-view embeddings. From round two,
    the bridge may consume the 16-state posterior and the corresponding frozen
    codebook sentence embeddings. It must learn their semantic relationship to
    view descriptions; ``route_view_prior`` is never read.

    There is deliberately no recurrent state mutation. Thus a skipped message
    is bitwise identical to round one, and additional rounds without new
    evidence are exact repeats of round two.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int, device: str, seed: int,
                 semantic_temperature: float = 0.10):
        import torch
        from torch import nn

        torch.manual_seed(seed)
        self.torch, self.device = torch, device

        class Module(nn.Module):
            def __init__(self):
                super().__init__()
                self.base_query = nn.Linear(embedding_dim, hidden_dim)
                self.base_key = nn.Linear(embedding_dim, hidden_dim)
                self.base_bias = nn.Linear(embedding_dim, 1)
                self.code_key = nn.Linear(embedding_dim, hidden_dim)
                self.route_view_query = nn.Linear(embedding_dim, hidden_dim)
                self.route_scale_logit = nn.Parameter(torch.tensor(0.0))

            def forward(self, batch: dict, rounds: int, mode: str = "soft"):
                if rounds not in (1, 2, 3):
                    raise ValueError(f"rounds must be 1, 2, or 3; got {rounds}")
                if mode not in {"soft", "zero", "shuffle", "hard", "uniform"}:
                    raise ValueError(f"Unknown Phase 5.1 mode: {mode}")
                views, bank, mask = batch["view_embeddings"], batch["view_probabilities"], batch["view_mask"]
                task, gates, routed = batch["task"], batch["gates"], batch["routed"]
                query_count = bank.shape[1]
                base_query = self.base_query(task)
                base_keys = self.base_key(views)
                base_logits = torch.einsum("bh,bvh->bv", base_query, base_keys) / math.sqrt(base_keys.shape[-1])
                base_logits = base_logits + self.base_bias(views).squeeze(-1)
                base_logits = base_logits[:, None, :].expand(-1, query_count, -1)

                posterior, codebook = batch["code_posterior"], batch["codebook_embeddings"]
                if mode == "shuffle" and len(posterior) > 1:
                    posterior, codebook = posterior.roll(1, 0), codebook.roll(1, 0)
                elif mode == "hard":
                    posterior = torch.nn.functional.one_hot(posterior.argmax(-1), 16).to(posterior.dtype)
                elif mode == "uniform":
                    posterior = torch.full_like(posterior, 1.0 / posterior.shape[-1])
                code_keys = torch.nn.functional.normalize(self.code_key(codebook), dim=-1)
                route_queries = torch.nn.functional.normalize(self.route_view_query(views), dim=-1)
                semantic = torch.einsum("bvh,bch->bvc", route_queries, code_keys) / semantic_temperature
                route_logits = torch.logsumexp(
                    semantic + torch.log(posterior.clamp_min(1e-8))[:, None, :], dim=-1,
                )
                route_logits = torch.nn.functional.softplus(self.route_scale_logit) * route_logits

                traces = {"weights": [], "selected_probability": [], "entropy": [], "message_applied": []}
                selected = batch["tfm_probability"]
                for round_index in range(rounds):
                    apply_message = round_index > 0 and mode != "zero"
                    logits = base_logits + (route_logits[:, None, :] if apply_message else 0.0)
                    logits = logits.masked_fill(~mask[:, None, :], torch.finfo(logits.dtype).min)
                    weights = torch.softmax(logits, dim=-1)
                    selected = torch.einsum("bqv,bqv->bq", weights, bank).clamp(1e-5, 1 - 1e-5)
                    entropy = -(weights * torch.log(weights.clamp_min(1e-8))).sum(-1)
                    entropy = entropy / torch.log(mask.sum(-1).float().clamp_min(2))[:, None]
                    traces["weights"].append(weights)
                    traces["selected_probability"].append(selected)
                    traces["entropy"].append(entropy)
                    traces["message_applied"].append(torch.full_like(selected, float(apply_message)))
                adapter = torch.where(routed.expand_as(selected), selected, batch["tfm_probability"])
                final = ((1.0 - gates) * batch["llm_probability"] + gates * adapter).clamp(1e-5, 1 - 1e-5)
                return final, traces

        self.module = Module().to(device)

    @property
    def trainable_params(self) -> int:
        return sum(parameter.numel() for parameter in self.module.parameters() if parameter.requires_grad)

    def fit(self, train: dict, validation: dict, *, epochs: int, batch_size: int,
            learning_rate: float, patience: int) -> dict:
        torch = self.torch
        optimizer = torch.optim.AdamW(self.module.parameters(), lr=learning_rate, weight_decay=1e-4)
        generator = torch.Generator(device="cpu").manual_seed(6511)
        best_loss, best_state, best_epoch, stale, max_gradient = math.inf, None, -1, 0, 0.0
        started = time.perf_counter()
        for epoch in range(epochs):
            self.module.train()
            order = torch.randperm(len(train["episode_ids"]), generator=generator).to(self.device)
            for start in range(0, len(order), batch_size):
                batch = slice_pack(train, order[start:start + batch_size])
                probability, _ = self.module(batch, 2, "soft")
                loss = torch.nn.functional.binary_cross_entropy(probability, batch["labels"])
                optimizer.zero_grad(set_to_none=True); loss.backward()
                max_gradient = max(max_gradient, float(torch.nn.utils.clip_grad_norm_(self.module.parameters(), 1.0)))
                optimizer.step()
            selection_loss = self.loss(validation, batch_size)
            if selection_loss < best_loss - 1e-6:
                best_loss, best_epoch, stale = selection_loss, epoch, 0
                best_state = {key: value.detach().cpu().clone() for key, value in self.module.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_state is None:
            raise RuntimeError("Phase 5.1 structured bridge did not produce a checkpoint")
        self.module.load_state_dict(best_state)
        probability, labels, _ = self.predict(train, 2, batch_size, "soft")
        zero, _, _ = self.predict(train, 2, batch_size, "zero")
        one, _, _ = self.predict(train, 1, batch_size, "soft")
        if not np.array_equal(zero, one):
            raise RuntimeError("Exact no-message identity was violated")
        return {"best_epoch": best_epoch, "selection_loss": best_loss,
                "validation_loss": self.loss(validation, batch_size),
                "train_loss": _binary_loss(labels, probability),
                "train_accuracy": float(np.mean((probability >= 0.5) == labels)),
                "train_message_mean_absolute_delta": float(np.mean(np.abs(probability - zero))),
                "exact_zero_identity": True, "max_gradient_norm": max_gradient,
                "elapsed_seconds": time.perf_counter() - started, "objective": "round2_only"}

    def loss(self, data: dict, batch_size: int) -> float:
        probability, labels, _ = self.predict(data, 2, batch_size, "soft")
        return _binary_loss(labels, probability)

    def predict(self, data: dict, rounds: int, batch_size: int, mode: str):
        torch = self.torch
        probabilities, labels = [], []
        parts = {key: [] for key in ("weights", "selected_probability", "entropy", "message_applied")}
        self.module.eval()
        with torch.inference_mode():
            for start in range(0, len(data["episode_ids"]), batch_size):
                indices = torch.arange(start, min(start + batch_size, len(data["episode_ids"])), device=self.device)
                batch = slice_pack(data, indices)
                probability, traces = self.module(batch, rounds, mode)
                probabilities.append(probability.cpu().numpy().reshape(-1))
                labels.append(batch["labels"].cpu().numpy().reshape(-1))
                for key, values in traces.items():
                    parts[key].append(torch.stack(values, dim=1).cpu().numpy())
        return np.concatenate(probabilities), np.concatenate(labels), {
            key: np.concatenate(values, axis=0) for key, values in parts.items()
        }

    def state_digest(self) -> str:
        digest = hashlib.sha256()
        for key, value in sorted(self.module.state_dict().items()):
            digest.update(key.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()


def _binary_loss(labels: np.ndarray, probability: np.ndarray) -> float:
    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    return float(-np.mean(labels * np.log(probability) + (1 - labels) * np.log(1 - probability)))
