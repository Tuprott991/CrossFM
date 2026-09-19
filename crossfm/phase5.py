from __future__ import annotations

import hashlib
import math
import time

import numpy as np

from .phase3 import slice_pack


class CrossFMShortcutAudit:
    """Cached bridge that cleanly separates messages from an analytical prior.

    ``message`` never reads ``route_view_prior``. Statistical routing can affect
    the prediction only after ``route_message`` updates the recurrent state.
    ``prior`` exposes the analytical posterior-to-view map before round one.
    The prediction is always the attention-weighted frozen response bank; no
    auxiliary head can create a hidden communication path.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int, device: str, seed: int):
        import torch
        from torch import nn

        torch.manual_seed(seed)
        self.torch, self.device = torch, device

        class Module(nn.Module):
            def __init__(self):
                super().__init__()
                self.state_query = nn.Linear(embedding_dim, hidden_dim)
                self.view_key = nn.Linear(embedding_dim, hidden_dim)
                self.view_bias = nn.Linear(embedding_dim, 1)
                self.route_down = nn.Linear(embedding_dim, hidden_dim)
                self.view_down = nn.Linear(embedding_dim, hidden_dim)
                self.scalar_down = nn.Linear(3, hidden_dim)
                self.evidence_up = nn.Linear(hidden_dim, embedding_dim)
                self.state_norm = nn.LayerNorm(embedding_dim)
                self.round_embedding = nn.Parameter(torch.zeros(3, hidden_dim))

            def forward(self, batch: dict, rounds: int, mode: str = "message"):
                if rounds not in (1, 2, 3):
                    raise ValueError(f"rounds must be 1, 2, or 3; got {rounds}")
                if mode not in {"message", "prior", "zero_t2l", "shuffle_t2l", "zero_l2t", "l2t_only"}:
                    raise ValueError(f"Unknown Phase 5 mode: {mode}")
                task, route = batch["task"], batch["route"]
                views, bank, mask = batch["view_embeddings"], batch["view_probabilities"], batch["view_mask"]
                gates, routed = batch["gates"], batch["routed"]
                query_count = bank.shape[1]
                state = task[:, None, :].expand(-1, query_count, -1)
                if mode == "shuffle_t2l" and len(route) > 1:
                    route = route.roll(1, dims=0)
                traces = {"weights": [], "selected_probability": [], "entropy": [], "state_delta_norm": []}
                selected = batch["tfm_probability"]
                for round_index in range(rounds):
                    if mode == "zero_l2t":
                        logits = torch.zeros_like(bank)
                    else:
                        query = self.state_query(state)
                        keys = self.view_key(views)
                        logits = torch.einsum("bqh,bvh->bqv", query, keys) / math.sqrt(keys.shape[-1])
                        logits = logits + self.view_bias(views).squeeze(-1)[:, None, :]
                        if mode == "prior":
                            logits = logits + torch.log(batch["route_view_prior"].clamp_min(1e-8))[:, None, :]
                    logits = logits.masked_fill(~mask[:, None, :], torch.finfo(logits.dtype).min)
                    weights = torch.softmax(logits, dim=-1)
                    selected = torch.einsum("bqv,bqv->bq", weights, bank).clamp(1e-5, 1 - 1e-5)
                    selected_view = torch.einsum("bqv,bvd->bqd", weights, views)
                    entropy = -(weights * torch.log(weights.clamp_min(1e-8))).sum(-1)
                    entropy = entropy / torch.log(mask.sum(-1).float().clamp_min(2))[:, None]
                    scalar = torch.stack((torch.logit(selected), entropy, gates.expand(-1, query_count)), dim=-1)
                    route_evidence = torch.zeros_like(route) if mode == "zero_t2l" else route
                    evidence = (
                        self.route_down(route_evidence)[:, None, :] + self.view_down(selected_view)
                        + self.scalar_down(scalar) + self.round_embedding[round_index]
                    )
                    delta = self.evidence_up(torch.nn.functional.gelu(evidence))
                    if mode in {"zero_t2l", "l2t_only", "prior"}:
                        delta = torch.zeros_like(delta)
                    state = self.state_norm(state + gates[:, None, :] * delta)
                    traces["weights"].append(weights)
                    traces["selected_probability"].append(selected)
                    traces["entropy"].append(entropy)
                    traces["state_delta_norm"].append(delta.norm(dim=-1))
                adapter = torch.where(routed.expand_as(selected), selected, batch["tfm_probability"])
                final = ((1.0 - gates) * batch["llm_probability"] + gates * adapter).clamp(1e-5, 1 - 1e-5)
                return final, traces

        self.module = Module().to(device)

    @property
    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.module.parameters() if p.requires_grad)

    def fit(self, train: dict, validation: dict, *, mode: str, epochs: int, batch_size: int,
            learning_rate: float, patience: int) -> dict:
        torch = self.torch
        optimizer = torch.optim.AdamW(self.module.parameters(), lr=learning_rate, weight_decay=1e-4)
        generator = torch.Generator(device="cpu").manual_seed(5911)
        best_loss, best_state, best_epoch, stale, max_gradient = math.inf, None, -1, 0, 0.0
        started = time.perf_counter()
        depth_weights = {1: 0.2, 2: 0.6, 3: 0.2}
        for epoch in range(epochs):
            self.module.train()
            order = torch.randperm(len(train["episode_ids"]), generator=generator).to(self.device)
            for start in range(0, len(order), batch_size):
                batch = slice_pack(train, order[start:start + batch_size])
                loss = sum(
                    depth_weights[depth] * torch.nn.functional.binary_cross_entropy(
                        self.module(batch, depth, mode)[0], batch["labels"],
                    ) for depth in (1, 2, 3)
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                max_gradient = max(max_gradient, float(torch.nn.utils.clip_grad_norm_(self.module.parameters(), 1.0)))
                optimizer.step()
            selection_loss = self.loss(validation, 2, batch_size, mode)
            if selection_loss < best_loss - 1e-6:
                best_loss, best_epoch, stale = selection_loss, epoch, 0
                best_state = {key: value.detach().cpu().clone() for key, value in self.module.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_state is None:
            raise RuntimeError("Phase 5 adapter did not produce a checkpoint")
        self.module.load_state_dict(best_state)
        train_probability, labels, _ = self.predict(train, 2, batch_size, mode)
        return {
            "best_epoch": best_epoch, "selection_loss": best_loss,
            "validation_loss": self.loss(validation, 2, batch_size, mode),
            "train_loss": _binary_loss(labels, train_probability),
            "train_accuracy": float(np.mean((train_probability >= 0.5) == labels)),
            "max_gradient_norm": max_gradient, "elapsed_seconds": time.perf_counter() - started,
            "mode": mode, "depth_supervision": {"r1": 0.2, "r2": 0.6, "r3": 0.2},
        }

    def loss(self, data: dict, rounds: int, batch_size: int, mode: str) -> float:
        probability, labels, _ = self.predict(data, rounds, batch_size, mode)
        return _binary_loss(labels, probability)

    def predict(self, data: dict, rounds: int, batch_size: int, mode: str):
        torch = self.torch
        probabilities, labels = [], []
        parts = {key: [] for key in ("weights", "selected_probability", "entropy", "state_delta_norm")}
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


def fixed_router_predict(data: dict, router: str, temperature: float = 0.05):
    """Zero-parameter analytical and embedding-geometry controls."""
    import torch

    mask, bank = data["view_mask"], data["view_probabilities"]
    if router == "analytic":
        weights = data["route_view_prior"]
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
    elif router == "semantic_cosine":
        route = torch.nn.functional.normalize(data["route"], dim=-1)
        views = torch.nn.functional.normalize(data["view_embeddings"], dim=-1)
        logits = torch.einsum("bd,bvd->bv", route, views) / temperature
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
    else:
        raise ValueError(router)
    selected = torch.einsum("bv,bqv->bq", weights, bank).clamp(1e-5, 1 - 1e-5)
    adapter = torch.where(data["routed"].expand_as(selected), selected, data["tfm_probability"])
    final = ((1.0 - data["gates"]) * data["llm_probability"] + data["gates"] * adapter).clamp(1e-5, 1 - 1e-5)
    entropy = -(weights * torch.log(weights.clamp_min(1e-8))).sum(-1)
    entropy = entropy / torch.log(mask.sum(-1).float().clamp_min(2))
    traces = {
        "weights": weights[:, None, None, :].expand(-1, 1, bank.shape[1], -1).cpu().numpy(),
        "selected_probability": selected[:, None, :].cpu().numpy(),
        "entropy": entropy[:, None, None].expand(-1, 1, bank.shape[1]).cpu().numpy(),
        "state_delta_norm": torch.zeros_like(selected)[:, None, :].cpu().numpy(),
    }
    return final.cpu().numpy().reshape(-1), data["labels"].cpu().numpy().reshape(-1), traces


def _binary_loss(labels: np.ndarray, probability: np.ndarray) -> float:
    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    return float(-np.mean(labels * np.log(probability) + (1 - labels) * np.log(1 - probability)))
