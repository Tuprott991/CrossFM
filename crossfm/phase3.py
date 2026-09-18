from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import time
from typing import Iterable

import numpy as np

from .baselines import QwenBinaryBaseline
from .phase2 import EpisodeEvidence, ViewEvidence, residual_gate, soft_code_posterior
from .synthetic import Episode


@dataclass
class CrossFMEpisodeCache:
    episode_id: str
    regime: str
    task_embedding: np.ndarray
    view_embeddings: np.ndarray
    route_message: np.ndarray
    view_probabilities: np.ndarray
    llm_probability: np.ndarray
    tfm_probability: np.ndarray
    gate: float
    labels: np.ndarray
    relevant_view: int
    route_view_prior: np.ndarray
    routed: bool


def _codebook_texts(ep: Episode, views: ViewEvidence) -> list[str]:
    texts = []
    for code in range(16):
        pattern = "".join("P" if code & (1 << bit) else "N" for bit in range(4))
        texts.append(f"Statistical route {pattern} selects {views.descriptions[ep.route_map[code]]}.")
    return texts


def build_crossfm_cache(
    llm: QwenBinaryBaseline,
    episodes: Iterable[Episode],
    specialist: dict[str, EpisodeEvidence],
    views: dict[str, ViewEvidence],
    *,
    score_llm: bool,
    fallback_threshold: float,
) -> list[CrossFMEpisodeCache]:
    episodes = list(episodes)
    llm_probabilities: dict[str, np.ndarray] = {}
    if score_llm:
        batch = llm.predict(episodes)
        offset = 0
        for ep in episodes:
            stop = offset + len(ep.y_query)
            llm_probabilities[ep.episode_id] = batch.probabilities[offset:stop]
            offset = stop
    codebook_cache: dict[tuple, np.ndarray] = {}
    result = []
    for ep in episodes:
        item = views[ep.episode_id]
        if item.task_embedding is None or item.view_embeddings is None:
            raise RuntimeError("Language embeddings must be attached before building Phase 3 cache")
        posterior = soft_code_posterior(ep)
        gate = residual_gate(ep, posterior, fallback_threshold)
        if ep.route_indices:
            key = (ep.feature_names, ep.route_map)
            if key not in codebook_cache:
                codebook_cache[key] = llm.encode_texts(_codebook_texts(ep, item))
            route_message = posterior @ codebook_cache[key]
            route_view_prior = np.zeros(len(item.descriptions), dtype=np.float64)
            for code, probability in enumerate(posterior):
                route_view_prior[ep.route_map[code]] += probability
        else:
            route_message = np.zeros_like(item.task_embedding)
            route_view_prior = np.full(len(item.descriptions), 1.0 / len(item.descriptions), dtype=np.float64)
        result.append(CrossFMEpisodeCache(
            episode_id=ep.episode_id,
            regime=ep.regime,
            task_embedding=item.task_embedding.astype(np.float32),
            view_embeddings=item.view_embeddings.astype(np.float32),
            route_message=np.asarray(route_message, dtype=np.float32),
            view_probabilities=item.probabilities.T.astype(np.float32),
            llm_probability=llm_probabilities.get(
                ep.episode_id, np.full(len(ep.y_query), 0.5, dtype=np.float32),
            ).astype(np.float32),
            tfm_probability=specialist[ep.episode_id].probability.astype(np.float32),
            gate=gate,
            labels=ep.y_query.astype(np.float32),
            relevant_view=next(
                (index for index, columns in enumerate(item.columns) if tuple(columns) == tuple(ep.relevant)), -1,
            ),
            route_view_prior=route_view_prior.astype(np.float32),
            routed=bool(ep.route_indices),
        ))
    return result


def pack_cache(items: list[CrossFMEpisodeCache], device: str) -> dict:
    import torch

    if not items:
        raise ValueError("Cannot pack an empty CrossFM cache")
    queries = len(items[0].labels)
    dimension = len(items[0].task_embedding)
    max_views = max(17, max(len(item.view_embeddings) for item in items))
    count = len(items)
    task = np.zeros((count, dimension), dtype=np.float32)
    route = np.zeros_like(task)
    view_embeddings = np.zeros((count, max_views, dimension), dtype=np.float32)
    view_probabilities = np.zeros((count, queries, max_views), dtype=np.float32)
    mask = np.zeros((count, max_views), dtype=bool)
    llm = np.zeros((count, queries), dtype=np.float32)
    tfm = np.zeros_like(llm)
    labels = np.zeros_like(llm)
    gates = np.zeros((count, 1), dtype=np.float32)
    relevant_views = np.zeros(count, dtype=np.int64)
    route_view_prior = np.zeros((count, max_views), dtype=np.float32)
    routed = np.zeros((count, 1), dtype=bool)
    for index, item in enumerate(items):
        if len(item.labels) != queries:
            raise ValueError("All packed episodes must have the same query count")
        width = len(item.view_embeddings)
        task[index], route[index] = item.task_embedding, item.route_message
        view_embeddings[index, :width] = item.view_embeddings
        view_probabilities[index, :, :width] = item.view_probabilities
        mask[index, :width] = True
        llm[index], tfm[index], labels[index], gates[index] = (
            item.llm_probability, item.tfm_probability, item.labels, item.gate,
        )
        relevant_views[index] = item.relevant_view
        route_view_prior[index, :width] = item.route_view_prior
        routed[index] = item.routed

    def tensor(array, dtype=None):
        value = torch.from_numpy(np.ascontiguousarray(array))
        return value.to(device=device, dtype=dtype, non_blocking=True) if dtype else value.to(device=device, non_blocking=True)

    return {
        "task": tensor(task), "route": tensor(route), "view_embeddings": tensor(view_embeddings),
        "view_probabilities": tensor(view_probabilities), "view_mask": tensor(mask),
        "llm_probability": tensor(llm), "tfm_probability": tensor(tfm),
        "labels": tensor(labels), "gates": tensor(gates),
        "relevant_views": tensor(relevant_views),
        "route_view_prior": tensor(route_view_prior), "routed": tensor(routed),
        "episode_ids": [item.episode_id for item in items],
    }


def cache_diagnostics(items: list[CrossFMEpisodeCache]) -> dict:
    relevant_correct, best_correct, total = 0, 0, 0
    for item in items:
        labels = item.labels.astype(bool)
        if item.relevant_view >= 0:
            relevant_correct += int(np.sum((item.view_probabilities[:, item.relevant_view] >= 0.5) == labels))
        view_scores = [int(np.sum((item.view_probabilities[:, index] >= 0.5) == labels)) for index in range(item.view_probabilities.shape[1])]
        best_correct += max(view_scores)
        total += len(labels)
    return {
        "relevant_view_accuracy": relevant_correct / total if total else 0.0,
        "posthoc_best_view_accuracy": best_correct / total if total else 0.0,
        "predictions": total,
    }


def slice_pack(pack: dict, indices) -> dict:
    return {
        key: value.index_select(0, indices) if hasattr(value, "index_select") else [value[int(i)] for i in indices.cpu()]
        for key, value in pack.items()
    }


class CrossFMLatentLoop:
    """Shared recurrent latent bridge over cached frozen-backbone evidence."""

    def __init__(
        self, embedding_dim: int, hidden_dim: int, max_rounds: int,
        device: str, seed: int, max_views: int = 17,
    ):
        import torch
        from torch import nn

        torch.manual_seed(seed)
        self.torch, self.device, self.max_rounds = torch, device, max_rounds

        class Module(nn.Module):
            def __init__(self):
                super().__init__()
                self.state_query = nn.Linear(embedding_dim, hidden_dim)
                self.view_key = nn.Linear(embedding_dim, hidden_dim)
                self.view_bias = nn.Linear(embedding_dim, 1)
                self.route_down = nn.Linear(embedding_dim, hidden_dim)
                self.view_down = nn.Linear(embedding_dim, hidden_dim)
                self.scalar_down = nn.Linear(3, hidden_dim)
                self.response_down = nn.Linear(max_views, hidden_dim)
                self.evidence_up = nn.Linear(hidden_dim, embedding_dim)
                self.state_norm = nn.LayerNorm(embedding_dim)
                self.round_embedding = nn.Parameter(torch.zeros(max_rounds, hidden_dim))
                self.head_state = nn.Linear(embedding_dim, hidden_dim)
                self.head_scalar = nn.Linear(2, hidden_dim)
                self.head_out = nn.Linear(hidden_dim, 1)
                nn.init.zeros_(self.head_out.weight)
                nn.init.zeros_(self.head_out.bias)

            def forward(self, batch: dict, rounds: int, message_mode: str = "normal"):
                task, route = batch["task"], batch["route"]
                route_prior, routed = batch["route_view_prior"], batch["routed"]
                views, probabilities, mask = (
                    batch["view_embeddings"], batch["view_probabilities"], batch["view_mask"],
                )
                gates = batch["gates"]
                query_count = probabilities.shape[1]
                state = task[:, None, :].expand(-1, query_count, -1)
                if message_mode == "shuffle":
                    route = route.roll(1, dims=0)
                    route_prior = route_prior.roll(1, dims=0)
                selected_probability = batch["tfm_probability"]
                weights = None
                weighted_response = torch.zeros(
                    (*probabilities.shape[:2], probabilities.shape[2]), device=probabilities.device,
                )
                for round_index in range(rounds):
                    query = self.state_query(state)
                    keys = self.view_key(views)
                    logits = torch.einsum("bqh,bvh->bqv", query, keys) / math.sqrt(keys.shape[-1])
                    logits = logits + self.view_bias(views).squeeze(-1)[:, None, :]
                    if round_index > 0 and message_mode != "zero":
                        logits = logits + torch.log(route_prior.clamp_min(1e-8))[:, None, :]
                    logits = logits.masked_fill(~mask[:, None, :], torch.finfo(logits.dtype).min)
                    weights = torch.softmax(logits, dim=-1)
                    selected_probability = torch.einsum("bqv,bqv->bq", weights, probabilities).clamp(1e-5, 1 - 1e-5)
                    selected_view = torch.einsum("bqv,bvd->bqd", weights, views)
                    weighted_response = weights * torch.logit(probabilities.clamp(1e-5, 1 - 1e-5))
                    entropy = -(weights * torch.log(weights.clamp_min(1e-8))).sum(-1)
                    entropy = entropy / torch.log(mask.sum(-1).float().clamp_min(2))[:, None]
                    scalar = torch.stack((torch.logit(selected_probability), entropy, gates.expand(-1, query_count)), dim=-1)
                    evidence = (
                        self.route_down(route)[:, None, :] + self.view_down(selected_view)
                        + self.scalar_down(scalar) + self.response_down(weighted_response)
                        + self.round_embedding[round_index]
                    )
                    delta = self.evidence_up(torch.nn.functional.gelu(evidence))
                    if message_mode == "zero":
                        delta = torch.zeros_like(delta)
                        selected_probability = torch.full_like(selected_probability, 0.5)
                        weighted_response = torch.zeros_like(weighted_response)
                    state = self.state_norm(state + gates[:, None, :] * delta)
                scalar_head = torch.stack((torch.logit(selected_probability), torch.logit(batch["tfm_probability"].clamp(1e-5, 1 - 1e-5))), dim=-1)
                correction = self.head_out(torch.nn.functional.gelu(
                    self.head_state(state) + self.head_scalar(scalar_head) + self.response_down(weighted_response)
                )).squeeze(-1)
                correction = correction * routed.expand_as(correction)
                anchor = torch.where(routed.expand_as(selected_probability), selected_probability, batch["tfm_probability"])
                adapter_logit = torch.logit(anchor.clamp(1e-5, 1 - 1e-5)) + correction
                adapter_probability = torch.sigmoid(adapter_logit)
                adapter_probability = torch.where(
                    routed.expand_as(adapter_probability), adapter_probability, batch["tfm_probability"],
                )
                final_probability = (
                    (1.0 - gates) * batch["llm_probability"] + gates * adapter_probability
                ).clamp(1e-5, 1 - 1e-5)
                return final_probability, weights

        self.module = Module().to(device)

    @property
    def trainable_params(self) -> int:
        return sum(parameter.numel() for parameter in self.module.parameters() if parameter.requires_grad)

    def fit(
        self, train: dict, validation: dict, rounds: int, epochs: int,
        batch_size: int, learning_rate: float, patience: int, checkpoint_metric: str = "validation",
    ) -> dict:
        torch = self.torch
        optimizer = torch.optim.AdamW(self.module.parameters(), lr=learning_rate, weight_decay=1e-4)
        generator = torch.Generator(device="cpu").manual_seed(3911 + rounds)
        best_loss, best_state, best_epoch, stale, max_gradient = math.inf, None, -1, 0, 0.0
        started = time.perf_counter()
        for epoch in range(epochs):
            self.module.train()
            order = torch.randperm(len(train["episode_ids"]), generator=generator, device="cpu").to(self.device)
            for start in range(0, len(order), batch_size):
                batch = slice_pack(train, order[start:start + batch_size])
                probability, _ = self.module(batch, rounds)
                loss = torch.nn.functional.binary_cross_entropy(probability, batch["labels"])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient = float(torch.nn.utils.clip_grad_norm_(self.module.parameters(), 1.0).detach().cpu())
                max_gradient = max(max_gradient, gradient)
                optimizer.step()
            selection_loss = self.loss(
                train if checkpoint_metric == "train" else validation, rounds, batch_size,
            )
            if selection_loss < best_loss - 1e-6:
                best_loss, best_epoch, stale = selection_loss, epoch, 0
                best_state = {key: value.detach().cpu().clone() for key, value in self.module.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_state is None:
            raise RuntimeError("CrossFM adapter did not produce a checkpoint")
        self.module.load_state_dict(best_state)
        train_probability, labels = self.predict(train, rounds, batch_size)
        zero_probability, _ = self.predict(train, rounds, batch_size, "zero")
        return {
            "best_epoch": best_epoch, "checkpoint_metric": checkpoint_metric,
            "selection_loss": best_loss, "validation_loss": self.loss(validation, rounds, batch_size),
            "train_loss": _binary_loss(labels, train_probability),
            "train_accuracy": float(np.mean((train_probability >= 0.5) == labels)),
            "train_zero_message_mean_absolute_delta": float(np.mean(np.abs(train_probability - zero_probability))),
            "max_gradient_norm": max_gradient,
            "elapsed_seconds": time.perf_counter() - started,
        }

    def loss(self, data: dict, rounds: int, batch_size: int) -> float:
        probability, labels = self.predict(data, rounds, batch_size)
        return _binary_loss(labels, probability)

    def predict(
        self, data: dict, rounds: int, batch_size: int, message_mode: str = "normal",
    ) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        probabilities, labels = [], []
        self.module.eval()
        with torch.inference_mode():
            for start in range(0, len(data["episode_ids"]), batch_size):
                indices = torch.arange(start, min(start + batch_size, len(data["episode_ids"])), device=self.device)
                batch = slice_pack(data, indices)
                probability, _ = self.module(batch, rounds, message_mode)
                probabilities.append(probability.cpu().numpy().reshape(-1))
                labels.append(batch["labels"].cpu().numpy().reshape(-1))
        return np.concatenate(probabilities), np.concatenate(labels)

    def state_digest(self) -> str:
        digest = hashlib.sha256()
        for key, value in sorted(self.module.state_dict().items()):
            digest.update(key.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()


def _binary_loss(labels: np.ndarray, probability: np.ndarray) -> float:
    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    return float(-np.mean(labels * np.log(probability) + (1 - labels) * np.log(1 - probability)))
