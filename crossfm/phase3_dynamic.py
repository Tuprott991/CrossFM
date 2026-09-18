from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Iterable

import numpy as np

from .baselines import QwenBinaryBaseline, TabICLBaseline
from .phase2 import candidate_views, collect_specialist_evidence, residual_gate, soft_code_posterior
from .synthetic import Episode


@dataclass
class DynamicStaticEpisode:
    episode: Episode
    prefix_text: str
    task_embedding: np.ndarray
    view_embeddings: np.ndarray
    view_columns: list[tuple[int, ...]]
    route_message: np.ndarray
    llm_probability: np.ndarray
    tfm_probability: np.ndarray
    initial_tfm_representation: np.ndarray
    gate: float


@dataclass
class DynamicObservation:
    probability: np.ndarray
    hidden_evidence: np.ndarray
    selected_view: np.ndarray
    selected_embedding: np.ndarray
    entropy: np.ndarray
    bridge_input: np.ndarray
    teacher_message: np.ndarray
    specialist_calls: int


def _route_texts(ep: Episode, descriptions: list[str]) -> list[str]:
    result = []
    for code in range(16):
        pattern = "".join("P" if code & (1 << bit) else "N" for bit in range(4))
        result.append(f"Statistical route {pattern} selects {descriptions[ep.route_map[code]]}.")
    return result


def build_dynamic_static_cache(
    llm: QwenBinaryBaseline,
    tfm: TabICLBaseline,
    episodes: Iterable[Episode],
    *,
    score_llm: bool,
    fallback_threshold: float,
) -> list[DynamicStaticEpisode]:
    """Cache only round-invariant inputs; no candidate-view TFM output is cached."""
    episodes = list(episodes)
    base = collect_specialist_evidence(tfm, episodes)
    if score_llm:
        prediction = llm.predict(episodes)
        llm_probabilities, offset = {}, 0
        for ep in episodes:
            stop = offset + len(ep.y_query)
            llm_probabilities[ep.episode_id] = prediction.probabilities[offset:stop]
            offset = stop
    else:
        llm_probabilities = {
            ep.episode_id: np.full(len(ep.y_query), 0.5, dtype=np.float32) for ep in episodes
        }

    task_texts = [f"Task: {ep.description} Schema: {', '.join(ep.feature_names)}" for ep in episodes]
    task_embeddings = llm.encode_texts(task_texts)
    descriptions_by_episode, columns_by_episode, flat_descriptions = [], [], []
    for ep in episodes:
        views = candidate_views(ep)
        descriptions_by_episode.append([description for description, _ in views])
        columns_by_episode.append([columns for _, columns in views])
        flat_descriptions.extend(description for description, _ in views)
    flat_view_embeddings = llm.encode_texts(flat_descriptions)

    view_embeddings_by_episode, offset = [], 0
    for descriptions in descriptions_by_episode:
        stop = offset + len(descriptions)
        view_embeddings_by_episode.append(flat_view_embeddings[offset:stop])
        offset = stop

    route_texts, route_slices = [], []
    for ep, descriptions in zip(episodes, descriptions_by_episode):
        start = len(route_texts)
        if ep.route_indices:
            route_texts.extend(_route_texts(ep, descriptions))
        route_slices.append((start, len(route_texts)))
    route_embeddings = llm.encode_texts(route_texts) if route_texts else np.empty((0, task_embeddings.shape[1]))

    result = []
    for index, ep in enumerate(episodes):
        posterior = soft_code_posterior(ep)
        start, stop = route_slices[index]
        route_message = (
            posterior @ route_embeddings[start:stop]
            if stop > start else np.zeros_like(task_embeddings[index])
        )
        result.append(DynamicStaticEpisode(
            episode=ep,
            prefix_text=task_texts[index],
            task_embedding=task_embeddings[index].astype(np.float32),
            view_embeddings=view_embeddings_by_episode[index].astype(np.float32),
            view_columns=columns_by_episode[index],
            route_message=np.asarray(route_message, dtype=np.float32),
            llm_probability=np.asarray(llm_probabilities[ep.episode_id], dtype=np.float32),
            tfm_probability=base[ep.episode_id].probability.astype(np.float32),
            initial_tfm_representation=base[ep.episode_id].representation.astype(np.float32),
            gate=residual_gate(ep, posterior, fallback_threshold),
        ))
    return result


def _attention(module, state, view_embeddings, round_index: int):
    keys = module.view_key(view_embeddings)
    query = module.state_query(state)
    logits = query @ keys.T / math.sqrt(keys.shape[-1])
    logits = logits + module.view_bias(view_embeddings).squeeze(-1)[None, :]
    return logits.softmax(dim=-1)


def dynamic_specialist_observation(
    tfm: TabICLBaseline,
    item: DynamicStaticEpisode,
    model,
    state,
    *,
    round_index: int,
) -> DynamicObservation:
    """Choose views from the current language state and execute fresh grouped TFM calls."""
    torch = model.torch
    module = model.module
    device = model.device
    views = torch.from_numpy(item.view_embeddings).to(device)
    weights = _attention(module, state, views, round_index)
    selected = weights.argmax(dim=-1)
    probability = np.empty(len(item.episode.y_query), dtype=np.float32)
    hidden = np.empty((len(item.episode.y_query), 512), dtype=np.float32)
    selected_np = selected.detach().cpu().numpy()
    calls = 0
    for view_index in np.unique(selected_np):
        rows = np.flatnonzero(selected_np == view_index)
        columns = item.view_columns[int(view_index)]
        dynamic_probability, dynamic_hidden = tfm.predict_arrays_with_evidence(
            item.episode.x_context[:, columns], item.episode.y_context,
            item.episode.x_query[rows][:, columns],
        )
        probability[rows] = dynamic_probability
        hidden[rows] = dynamic_hidden
        calls += 1

    probability_tensor = torch.from_numpy(probability).to(device)
    hidden_tensor = torch.from_numpy(hidden).to(device)
    selected_embedding = views.index_select(0, selected)
    entropy = -(weights * torch.log(weights.clamp_min(1e-8))).sum(-1) / math.log(max(len(item.view_columns), 2))
    response = torch.zeros((len(probability), 17), device=device)
    response.scatter_(1, selected[:, None], torch.logit(probability_tensor.clamp(1e-5, 1 - 1e-5))[:, None])
    route = torch.from_numpy(item.route_message).to(device)
    gate = torch.full((len(probability),), item.gate, device=device)
    scalar = torch.stack((torch.logit(probability_tensor.clamp(1e-5, 1 - 1e-5)), entropy, gate), dim=-1)
    teacher = module.evidence_up(torch.nn.functional.gelu(
        module.route_down(route)[None, :] + module.view_down(selected_embedding)
        + module.scalar_down(scalar) + module.response_down(response)
        + module.round_embedding[round_index]
    ))
    route_rows = route[None, :].expand(len(probability), -1)
    bridge_input = torch.cat((hidden_tensor, route_rows, selected_embedding, scalar[:, :2]), dim=-1)
    return DynamicObservation(
        probability=probability,
        hidden_evidence=hidden,
        selected_view=selected_np,
        selected_embedding=selected_embedding.detach().cpu().numpy(),
        entropy=entropy.detach().cpu().numpy(),
        bridge_input=bridge_input.detach().cpu().numpy(),
        teacher_message=teacher.detach().cpu().numpy(),
        specialist_calls=calls,
    )


class DynamicEvidenceBridge:
    """Small rich-evidence bridge distilled from the validated cached controller."""

    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int, device: str, seed: int):
        import torch
        from torch import nn

        torch.manual_seed(seed)
        self.torch, self.device = torch, device
        self.module = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        ).to(device)

    @property
    def trainable_params(self) -> int:
        return sum(parameter.numel() for parameter in self.module.parameters() if parameter.requires_grad)

    def fit(
        self, features: np.ndarray, targets: np.ndarray, *, epochs: int, batch_size: int,
        learning_rate: float, use_bfloat16: bool,
    ) -> dict:
        torch = self.torch
        x = torch.from_numpy(features).to(self.device)
        y = torch.from_numpy(targets).to(self.device)
        optimizer = torch.optim.AdamW(self.module.parameters(), lr=learning_rate, weight_decay=1e-4)
        generator = torch.Generator(device="cpu").manual_seed(7301)
        best_loss, best_state, started = math.inf, None, time.perf_counter()
        for epoch in range(epochs):
            order = torch.randperm(len(x), generator=generator, device="cpu").to(self.device)
            self.module.train()
            for start in range(0, len(order), batch_size):
                indices = order[start:start + batch_size]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
                    output = self.module(x.index_select(0, indices))
                    loss = torch.nn.functional.mse_loss(output.float(), y.index_select(0, indices))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.module.parameters(), 1.0)
                optimizer.step()
            self.module.eval()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
                value = float(torch.nn.functional.mse_loss(self.module(x).float(), y).cpu())
            if value < best_loss:
                best_loss = value
                best_state = {key: value.detach().cpu().clone() for key, value in self.module.state_dict().items()}
                best_epoch = epoch
        self.module.load_state_dict(best_state)
        return {"best_epoch": best_epoch, "train_mse": best_loss, "elapsed_seconds": time.perf_counter() - started}

    def predict(self, features: np.ndarray, use_bfloat16: bool) -> object:
        torch = self.torch
        x = torch.from_numpy(features).to(self.device)
        self.module.eval()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
            return self.module(x).float()
