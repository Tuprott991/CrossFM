from __future__ import annotations

import hashlib
import math
import time

import numpy as np

from .phase3 import slice_pack


class CrossFMCorrectiveLoop:
    """Three-round cached CrossFM with an explicit corrective final round.

    Round 1 forms a semantic view proposal. Round 2 binds the continuous
    route posterior to the response bank. Round 3 observes the preceding
    attention entropy, LLM/specialist disagreement, probability change, and
    per-view residuals. A learned gate then decides how strongly to revise
    the state and a learned residual scorer reweights the candidate views.
    No label or query outcome is available to that gate at inference time.
    """

    def __init__(
        self, embedding_dim: int, hidden_dim: int, max_rounds: int,
        device: str, seed: int, max_views: int = 17,
    ):
        import torch
        from torch import nn

        if max_rounds != 3:
            raise ValueError("The corrective protocol is frozen to exactly three rounds")
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
                self.scalar_down = nn.Linear(5, hidden_dim)
                self.response_down = nn.Linear(max_views, hidden_dim)
                self.evidence_up = nn.Linear(hidden_dim, embedding_dim)
                self.state_norm = nn.LayerNorm(embedding_dim)
                self.round_embedding = nn.Parameter(torch.zeros(max_rounds, hidden_dim))
                self.corrective_rank = nn.Sequential(
                    nn.Linear(4, 32), nn.GELU(), nn.Linear(32, 1),
                )
                self.corrective_gate = nn.Linear(4, 1)
                self.attention_memory_logit = nn.Parameter(torch.tensor(-1.0))
                self.head_state = nn.Linear(embedding_dim, hidden_dim)
                self.head_scalar = nn.Linear(2, hidden_dim)
                self.head_out = nn.Linear(hidden_dim, 1)
                nn.init.zeros_(self.head_out.weight)
                nn.init.zeros_(self.head_out.bias)

            def forward(self, batch: dict, rounds: int, message_mode: str = "normal"):
                if rounds < 1 or rounds > 3:
                    raise ValueError(f"rounds must be in [1, 3], got {rounds}")
                valid_modes = {
                    "normal", "zero_t2l", "shuffle_t2l", "t2l_only",
                    "l2t_only", "stopgrad_l2t", "stopgrad_t2l",
                }
                if message_mode not in valid_modes:
                    raise ValueError(f"Unknown message mode: {message_mode}")

                task, route = batch["task"], batch["route"]
                route_prior, routed = batch["route_view_prior"], batch["routed"]
                views, probabilities, mask = (
                    batch["view_embeddings"], batch["view_probabilities"], batch["view_mask"],
                )
                gates = batch["gates"]
                query_count = probabilities.shape[1]
                state = task[:, None, :].expand(-1, query_count, -1)
                selected_probability = batch["tfm_probability"]
                previous_probability = selected_probability
                previous_weights = None
                weighted_response = torch.zeros_like(probabilities)
                traces = {"weights": [], "selected_probability": [], "entropy": [], "update_gate": []}

                for round_index in range(rounds):
                    if message_mode == "t2l_only":
                        logits = torch.zeros_like(probabilities)
                    else:
                        query = self.state_query(state)
                        keys = self.view_key(views)
                        logits = torch.einsum("bqh,bvh->bqv", query, keys) / math.sqrt(keys.shape[-1])
                        logits = logits + self.view_bias(views).squeeze(-1)[:, None, :]
                        if round_index > 0:
                            logits = logits + torch.log(route_prior.clamp_min(1e-8))[:, None, :]

                    if round_index == 2 and previous_weights is not None:
                        view_logits = torch.logit(probabilities.clamp(1e-5, 1 - 1e-5))
                        previous_logit = torch.logit(previous_probability.clamp(1e-5, 1 - 1e-5))
                        previous_entropy = -(previous_weights * torch.log(previous_weights.clamp_min(1e-8))).sum(-1)
                        previous_entropy = previous_entropy / torch.log(mask.sum(-1).float().clamp_min(2))[:, None]
                        disagreement = previous_logit - torch.logit(
                            batch["llm_probability"].clamp(1e-5, 1 - 1e-5),
                        )
                        residual = view_logits - previous_logit[:, :, None]
                        residual_features = torch.stack((
                            residual,
                            residual.abs(),
                            previous_entropy[:, :, None].expand_as(residual),
                            disagreement[:, :, None].expand_as(residual),
                        ), dim=-1)
                        corrective_logits = self.corrective_rank(residual_features).squeeze(-1)
                        probability_change = (
                            previous_logit - torch.logit(batch["tfm_probability"].clamp(1e-5, 1 - 1e-5))
                        ).abs()
                        gate_features = torch.stack((
                            previous_entropy, probability_change, disagreement.abs(),
                            gates.expand(-1, query_count),
                        ), dim=-1)
                        corrective_gate = torch.sigmoid(self.corrective_gate(gate_features))
                        memory = torch.sigmoid(self.attention_memory_logit)
                        logits = logits + corrective_gate * corrective_logits
                        logits = logits + memory * torch.log(previous_weights.clamp_min(1e-8))
                    else:
                        corrective_gate = torch.ones(
                            (*probabilities.shape[:2], 1), device=probabilities.device,
                            dtype=probabilities.dtype,
                        )

                    logits = logits.masked_fill(~mask[:, None, :], torch.finfo(logits.dtype).min)
                    weights = torch.softmax(logits, dim=-1)
                    evidence_weights = weights.detach() if message_mode == "stopgrad_l2t" else weights
                    current_probability = torch.einsum(
                        "bqv,bqv->bq", evidence_weights, probabilities,
                    ).clamp(1e-5, 1 - 1e-5)
                    selected_view = torch.einsum("bqv,bvd->bqd", evidence_weights, views)
                    weighted_response = evidence_weights * torch.logit(probabilities.clamp(1e-5, 1 - 1e-5))

                    if message_mode == "shuffle_t2l" and len(task) > 1:
                        current_probability = current_probability.roll(1, dims=0)
                        selected_view = selected_view.roll(1, dims=0)
                        weighted_response = weighted_response.roll(1, dims=0)
                    if message_mode == "zero_t2l":
                        current_probability = torch.full_like(current_probability, 0.5)
                        selected_view = torch.zeros_like(selected_view)
                        weighted_response = torch.zeros_like(weighted_response)

                    entropy = -(weights * torch.log(weights.clamp_min(1e-8))).sum(-1)
                    entropy = entropy / torch.log(mask.sum(-1).float().clamp_min(2))[:, None]
                    evidence_change = torch.logit(current_probability) - torch.logit(previous_probability.clamp(1e-5, 1 - 1e-5))
                    disagreement = torch.logit(current_probability) - torch.logit(
                        batch["llm_probability"].clamp(1e-5, 1 - 1e-5),
                    )
                    scalar = torch.stack((
                        torch.logit(current_probability), entropy,
                        gates.expand(-1, query_count), evidence_change, disagreement,
                    ), dim=-1)
                    route_evidence = torch.zeros_like(route) if message_mode == "t2l_only" else route
                    evidence = (
                        self.route_down(route_evidence)[:, None, :] + self.view_down(selected_view)
                        + self.scalar_down(scalar) + self.response_down(weighted_response)
                        + self.round_embedding[round_index]
                    )
                    delta = self.evidence_up(torch.nn.functional.gelu(evidence))
                    if message_mode == "stopgrad_t2l":
                        delta = delta.detach()
                    if message_mode in {"zero_t2l", "l2t_only"}:
                        delta = torch.zeros_like(delta)
                    update_gate = corrective_gate if round_index == 2 else torch.ones_like(corrective_gate)
                    state = self.state_norm(state + gates[:, None, :] * update_gate * delta)

                    traces["weights"].append(weights)
                    traces["selected_probability"].append(current_probability)
                    traces["entropy"].append(entropy)
                    traces["update_gate"].append(update_gate.squeeze(-1))
                    previous_probability, previous_weights = current_probability, weights
                    selected_probability = current_probability

                if message_mode == "l2t_only":
                    adapter_probability = selected_probability
                else:
                    scalar_head = torch.stack((
                        torch.logit(selected_probability.clamp(1e-5, 1 - 1e-5)),
                        torch.logit(batch["tfm_probability"].clamp(1e-5, 1 - 1e-5)),
                    ), dim=-1)
                    correction = self.head_out(torch.nn.functional.gelu(
                        self.head_state(state) + self.head_scalar(scalar_head)
                        + self.response_down(weighted_response)
                    )).squeeze(-1)
                    correction = correction * routed.expand_as(correction)
                    anchor = torch.where(
                        routed.expand_as(selected_probability), selected_probability, batch["tfm_probability"],
                    )
                    adapter_probability = torch.sigmoid(torch.logit(anchor.clamp(1e-5, 1 - 1e-5)) + correction)
                    adapter_probability = torch.where(
                        routed.expand_as(adapter_probability), adapter_probability, batch["tfm_probability"],
                    )
                final_probability = (
                    (1.0 - gates) * batch["llm_probability"] + gates * adapter_probability
                ).clamp(1e-5, 1 - 1e-5)
                return final_probability, traces

        self.module = Module().to(device)

    @property
    def trainable_params(self) -> int:
        return sum(parameter.numel() for parameter in self.module.parameters() if parameter.requires_grad)

    def fit(
        self, train: dict, validation: dict, *, rounds: int, epochs: int,
        batch_size: int, learning_rate: float, patience: int,
        message_mode: str = "normal", deep_supervision: bool = False,
    ) -> dict:
        torch = self.torch
        optimizer = torch.optim.AdamW(self.module.parameters(), lr=learning_rate, weight_decay=1e-4)
        generator = torch.Generator(device="cpu").manual_seed(4911 + rounds)
        best_loss, best_state, best_epoch, stale, max_gradient = math.inf, None, -1, 0, 0.0
        started = time.perf_counter()
        depth_weights = {1: 0.2, 2: 0.3, 3: 0.5}
        for epoch in range(epochs):
            self.module.train()
            order = torch.randperm(len(train["episode_ids"]), generator=generator, device="cpu").to(self.device)
            for start in range(0, len(order), batch_size):
                batch = slice_pack(train, order[start:start + batch_size])
                if deep_supervision:
                    loss = sum(
                        depth_weights[depth] * torch.nn.functional.binary_cross_entropy(
                            self.module(batch, depth, message_mode)[0], batch["labels"],
                        )
                        for depth in (1, 2, 3)
                    )
                else:
                    probability, _ = self.module(batch, rounds, message_mode)
                    loss = torch.nn.functional.binary_cross_entropy(probability, batch["labels"])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient = float(torch.nn.utils.clip_grad_norm_(self.module.parameters(), 1.0).detach().cpu())
                max_gradient = max(max_gradient, gradient)
                optimizer.step()
            selection_loss = self.loss(validation, rounds, batch_size, message_mode)
            if selection_loss < best_loss - 1e-6:
                best_loss, best_epoch, stale = selection_loss, epoch, 0
                best_state = {key: value.detach().cpu().clone() for key, value in self.module.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_state is None:
            raise RuntimeError("CrossFM corrective adapter did not produce a checkpoint")
        self.module.load_state_dict(best_state)
        train_probability, labels, _ = self.predict(train, rounds, batch_size, message_mode)
        zero_probability, _, _ = self.predict(train, rounds, batch_size, "zero_t2l")
        return {
            "best_epoch": best_epoch, "checkpoint_metric": "validation",
            "selection_loss": best_loss,
            "validation_loss": self.loss(validation, rounds, batch_size, message_mode),
            "train_loss": _binary_loss(labels, train_probability),
            "train_accuracy": float(np.mean((train_probability >= 0.5) == labels)),
            "train_zero_message_mean_absolute_delta": float(np.mean(np.abs(train_probability - zero_probability))),
            "max_gradient_norm": max_gradient, "elapsed_seconds": time.perf_counter() - started,
            "message_mode": message_mode, "deep_supervision": deep_supervision,
        }

    def loss(self, data: dict, rounds: int, batch_size: int, message_mode: str = "normal") -> float:
        probability, labels, _ = self.predict(data, rounds, batch_size, message_mode)
        return _binary_loss(labels, probability)

    def predict(
        self, data: dict, rounds: int, batch_size: int, message_mode: str = "normal",
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        torch = self.torch
        probabilities, labels = [], []
        trace_parts = {"weights": [], "selected_probability": [], "entropy": [], "update_gate": []}
        self.module.eval()
        with torch.inference_mode():
            for start in range(0, len(data["episode_ids"]), batch_size):
                indices = torch.arange(start, min(start + batch_size, len(data["episode_ids"])), device=self.device)
                batch = slice_pack(data, indices)
                probability, traces = self.module(batch, rounds, message_mode)
                probabilities.append(probability.cpu().numpy().reshape(-1))
                labels.append(batch["labels"].cpu().numpy().reshape(-1))
                for key, values in traces.items():
                    trace_parts[key].append(torch.stack(values, dim=1).cpu().numpy())
        traces = {key: np.concatenate(parts, axis=0) for key, parts in trace_parts.items()}
        return np.concatenate(probabilities), np.concatenate(labels), traces

    def state_digest(self) -> str:
        digest = hashlib.sha256()
        for key, value in sorted(self.module.state_dict().items()):
            digest.update(key.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()


def _binary_loss(labels: np.ndarray, probability: np.ndarray) -> float:
    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    return float(-np.mean(labels * np.log(probability) + (1 - labels) * np.log(1 - probability)))
