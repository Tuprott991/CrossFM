from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Iterable

import numpy as np

from .baselines import QwenBinaryBaseline, TabICLBaseline, build_llm_prompts
from .synthetic import Episode


@dataclass
class EpisodeEvidence:
    probability: np.ndarray
    representation: np.ndarray


@dataclass
class ViewEvidence:
    descriptions: list[str]
    columns: list[tuple[int, ...]]
    probabilities: np.ndarray
    labels: np.ndarray
    task_embedding: np.ndarray | None = None
    view_embeddings: np.ndarray | None = None


def candidate_views(ep: Episode) -> list[tuple[str, tuple[int, ...]]]:
    """Predeclared semantic query space for the one-way L->T controls."""
    p = len(ep.feature_names)
    if ep.regime == "A":
        views = [(f"single field {ep.feature_names[i]}", (i,)) for i in range(p)]
    elif ep.regime in {"B", "C"}:
        views = [
            (f"paired fields {ep.feature_names[i]} and {ep.feature_names[i + 1]}", (i, i + 1))
            for i in range(0, p, 2)
        ]
    elif ep.regime == "C2":
        start = len(ep.route_indices)
        views = [
            (f"paired fields {ep.feature_names[i]} and {ep.feature_names[i + 1]}", (i, i + 1))
            for i in range(start, p, 2)
        ]
    else:
        views = []
    views.append(("all fields (unrestricted statistical model)", tuple(range(p))))
    return views


def collect_specialist_evidence(tfm: TabICLBaseline, episodes: Iterable[Episode]) -> dict[str, EpisodeEvidence]:
    result = {}
    for ep in episodes:
        probability, representation = tfm.predict_episode_with_evidence(ep)
        result[ep.episode_id] = EpisodeEvidence(probability, representation)
    return result


def collect_view_evidence(tfm: TabICLBaseline, episodes: Iterable[Episode]) -> dict[str, ViewEvidence]:
    result: dict[str, ViewEvidence] = {}
    for ep in episodes:
        descriptions, columns, predictions = [], [], []
        for description, indices in candidate_views(ep):
            tfm.classifier.fit(ep.x_context[:, indices], ep.y_context)
            probability = tfm.classifier.predict_proba(ep.x_query[:, indices])[:, 1]
            descriptions.append(description)
            columns.append(indices)
            predictions.append(probability)
        result[ep.episode_id] = ViewEvidence(
            descriptions=descriptions,
            columns=columns,
            probabilities=np.stack(predictions),
            labels=ep.y_query.copy(),
        )
    return result


def attach_language_embeddings(
    llm: QwenBinaryBaseline,
    episodes: Iterable[Episode],
    evidence: dict[str, ViewEvidence],
) -> None:
    episodes = list(episodes)
    task_texts = [f"Task: {ep.description} Schema: {', '.join(ep.feature_names)}" for ep in episodes]
    task_embeddings = llm.encode_texts(task_texts)
    view_texts, counts = [], []
    for ep in episodes:
        descriptions = evidence[ep.episode_id].descriptions
        view_texts.extend(descriptions)
        counts.append(len(descriptions))
    view_embeddings = llm.encode_texts(view_texts)
    offset = 0
    for ep, task_embedding, count in zip(episodes, task_embeddings, counts):
        item = evidence[ep.episode_id]
        item.task_embedding = task_embedding
        item.view_embeddings = view_embeddings[offset:offset + count]
        offset += count


def _view_features(item: ViewEvidence) -> np.ndarray:
    if item.task_embedding is None or item.view_embeddings is None:
        raise RuntimeError("Language embeddings were not attached")
    task = np.repeat(item.task_embedding[None, :], len(item.descriptions), axis=0)
    full = np.asarray([float("all fields" in text) for text in item.descriptions])[:, None]
    width = np.asarray([len(cols) / 12.0 for cols in item.columns])[:, None]
    return np.concatenate([task, item.view_embeddings, task * item.view_embeddings, full, width], axis=1)


class ViewScorer:
    """Small learned latent language-to-specialist query adapter."""

    def __init__(self, embedding_dim: int, hidden_dim: int, device: str, seed: int):
        import torch
        from torch import nn

        torch.manual_seed(seed)
        self.torch = torch
        self.device = device
        self.module = nn.Sequential(
            nn.LayerNorm(embedding_dim * 3 + 2),
            nn.Linear(embedding_dim * 3 + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        ).to(device)

    @property
    def trainable_params(self) -> int:
        return sum(parameter.numel() for parameter in self.module.parameters() if parameter.requires_grad)

    def fit(
        self,
        train: dict[str, ViewEvidence],
        validation: dict[str, ViewEvidence],
        epochs: int,
        learning_rate: float,
    ) -> dict:
        torch = self.torch
        optimizer = torch.optim.AdamW(self.module.parameters(), lr=learning_rate, weight_decay=1e-4)
        best_loss, best_state, best_epoch = math.inf, None, -1
        started = time.perf_counter()
        for epoch in range(epochs):
            self.module.train()
            for key in sorted(train):
                item = train[key]
                features = torch.from_numpy(_view_features(item)).float().to(self.device)
                view_probs = torch.from_numpy(item.probabilities).float().to(self.device)
                labels = torch.from_numpy(item.labels).float().to(self.device)
                weights = torch.softmax(self.module(features).squeeze(-1), dim=0)
                mixed = (weights[:, None] * view_probs).sum(0).clamp(1e-5, 1 - 1e-5)
                loss = torch.nn.functional.binary_cross_entropy(mixed, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            loss = self.loss(validation)
            if loss < best_loss:
                best_loss, best_epoch = loss, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in self.module.state_dict().items()}
        if best_state is None:
            raise RuntimeError("View scorer did not produce a checkpoint")
        self.module.load_state_dict(best_state)
        return {"best_epoch": best_epoch, "validation_loss": best_loss, "elapsed_seconds": time.perf_counter() - started}

    def loss(self, data: dict[str, ViewEvidence]) -> float:
        torch = self.torch
        losses = []
        self.module.eval()
        with torch.inference_mode():
            for key in sorted(data):
                item = data[key]
                features = torch.from_numpy(_view_features(item)).float().to(self.device)
                probs = torch.from_numpy(item.probabilities).float().to(self.device)
                labels = torch.from_numpy(item.labels).float().to(self.device)
                weights = torch.softmax(self.module(features).squeeze(-1), dim=0)
                mixed = (weights[:, None] * probs).sum(0).clamp(1e-5, 1 - 1e-5)
                losses.append(float(torch.nn.functional.binary_cross_entropy(mixed, labels)))
        return float(np.mean(losses))

    def predict(self, item: ViewEvidence, top_k: int) -> tuple[np.ndarray, list[int], np.ndarray]:
        torch = self.torch
        self.module.eval()
        with torch.inference_mode():
            features = torch.from_numpy(_view_features(item)).float().to(self.device)
            weights = torch.softmax(self.module(features).squeeze(-1), dim=0).cpu().numpy()
        chosen = np.argsort(weights)[-min(top_k, len(weights)):]
        selected_weights = weights[chosen] / weights[chosen].sum()
        probability = (selected_weights[:, None] * item.probabilities[chosen]).sum(0)
        return probability, chosen.tolist(), weights


def _query_prompt(ep: Episode, row: np.ndarray) -> str:
    fields = ", ".join(f"{name}={value:.3f}" for name, value in zip(ep.feature_names, row))
    return f"Task: {ep.description}\nSchema: {', '.join(ep.feature_names)}\nQuery: {fields}\nPredict label 0 or 1."


class SoftPrefixAdapter:
    """Frozen TabICL state -> learned soft prefix -> frozen Qwen -> binary head."""

    def __init__(self, llm: QwenBinaryBaseline, evidence_dim: int, bottleneck: int, prefix_tokens: int, seed: int):
        import torch
        from torch import nn

        torch.manual_seed(seed)
        self.torch, self.llm = torch, llm
        self.device = llm.device
        self.hidden_dim = int(llm.model.config.hidden_size)
        self.prefix_tokens = prefix_tokens
        self.projector = nn.Sequential(
            nn.LayerNorm(evidence_dim), nn.Linear(evidence_dim, bottleneck), nn.GELU(),
            nn.Linear(bottleneck, prefix_tokens * self.hidden_dim),
        ).to(self.device)
        self.head = nn.Sequential(nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, 1)).to(self.device)

    @property
    def trainable_params(self) -> int:
        return sum(p.numel() for module in (self.projector, self.head) for p in module.parameters() if p.requires_grad)

    def _logits(self, prompts: list[str], evidence: np.ndarray):
        torch = self.torch
        tokens = self.llm.tokenizer(
            prompts, padding=True, truncation=True, max_length=160, return_tensors="pt",
        ).to(self.device)
        token_embeddings = self.llm.model.get_input_embeddings()(tokens["input_ids"])
        evidence_tensor = torch.from_numpy(evidence).float().to(self.device)
        prefix = self.projector(evidence_tensor).reshape(-1, self.prefix_tokens, self.hidden_dim)
        prefix = prefix.to(token_embeddings.dtype)
        inputs = torch.cat([prefix, token_embeddings], dim=1)
        mask = torch.cat([
            torch.ones((len(prompts), self.prefix_tokens), dtype=tokens["attention_mask"].dtype, device=self.device),
            tokens["attention_mask"],
        ], dim=1)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device.startswith("cuda")):
            output = self.llm.model.model(inputs_embeds=inputs, attention_mask=mask, use_cache=False)
            pooled = output.last_hidden_state[:, -1, :].float()
            return self.head(pooled).squeeze(-1)

    @staticmethod
    def _flatten(episodes: Iterable[Episode], evidence: dict[str, EpisodeEvidence]):
        prompts, states, labels, ids = [], [], [], []
        for ep in episodes:
            item = evidence[ep.episode_id]
            for index, row in enumerate(ep.x_query):
                prompts.append(_query_prompt(ep, row))
                states.append(item.representation[index])
                labels.append(int(ep.y_query[index]))
                ids.append(ep.episode_id)
        return prompts, np.asarray(states), np.asarray(labels), np.asarray(ids)

    def fit(
        self,
        train_episodes: Iterable[Episode], train_evidence: dict[str, EpisodeEvidence],
        validation_episodes: Iterable[Episode], validation_evidence: dict[str, EpisodeEvidence],
        epochs: int, batch_size: int, learning_rate: float,
    ) -> dict:
        torch = self.torch
        train_prompts, train_states, train_labels, _ = self._flatten(train_episodes, train_evidence)
        val_prompts, val_states, val_labels, _ = self._flatten(validation_episodes, validation_evidence)
        parameters = list(self.projector.parameters()) + list(self.head.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-4)
        best_loss, best_state, best_epoch = math.inf, None, -1
        rng = np.random.default_rng(991)
        started = time.perf_counter()
        for epoch in range(epochs):
            order = rng.permutation(len(train_labels))
            self.projector.train(); self.head.train()
            for start in range(0, len(order), batch_size):
                idx = order[start:start + batch_size]
                logits = self._logits([train_prompts[i] for i in idx], train_states[idx])
                labels = torch.from_numpy(train_labels[idx]).float().to(self.device)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
            val_loss = self.loss(val_prompts, val_states, val_labels, batch_size)
            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch
                best_state = {
                    "projector": {k: v.detach().cpu().clone() for k, v in self.projector.state_dict().items()},
                    "head": {k: v.detach().cpu().clone() for k, v in self.head.state_dict().items()},
                }
        if best_state is None:
            raise RuntimeError("Soft-prefix adapter did not produce a checkpoint")
        self.projector.load_state_dict(best_state["projector"])
        self.head.load_state_dict(best_state["head"])
        return {"best_epoch": best_epoch, "validation_loss": best_loss, "elapsed_seconds": time.perf_counter() - started}

    def loss(self, prompts: list[str], states: np.ndarray, labels: np.ndarray, batch_size: int) -> float:
        torch = self.torch
        losses = []
        self.projector.eval(); self.head.eval()
        for start in range(0, len(labels), batch_size):
            stop = start + batch_size
            logits = self._logits(prompts[start:stop], states[start:stop])
            target = torch.from_numpy(labels[start:stop]).float().to(self.device)
            losses.append(float(torch.nn.functional.binary_cross_entropy_with_logits(logits, target).detach().cpu()))
        return float(np.mean(losses))

    def predict(self, episodes: Iterable[Episode], evidence: dict[str, EpisodeEvidence], batch_size: int):
        torch = self.torch
        prompts, states, labels, ids = self._flatten(episodes, evidence)
        probabilities = []
        self.projector.eval(); self.head.eval()
        for start in range(0, len(labels), batch_size):
            logits = self._logits(prompts[start:start + batch_size], states[start:start + batch_size])
            probabilities.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
        return np.asarray(probabilities), labels, ids


def tool_prompts(ep: Episode, specialist_probability: np.ndarray) -> list[str]:
    summaries = []
    y = ep.y_context
    for i, name in enumerate(ep.feature_names):
        corr = float(np.nan_to_num(np.corrcoef(ep.x_context[:, i], y)[0, 1]))
        summaries.append((abs(corr), f"corr({name}, label)={corr:+.2f}"))
    for i in range(0, len(ep.feature_names) - 1, 2):
        a, b = ep.feature_names[i:i + 2]
        transforms = {
            f"{b}-{a}": ep.x_context[:, i + 1] - ep.x_context[:, i],
            f"{b}+{a}": ep.x_context[:, i + 1] + ep.x_context[:, i],
            f"{b}*{a}": ep.x_context[:, i + 1] * ep.x_context[:, i],
        }
        for name, value in transforms.items():
            corr = float(np.nan_to_num(np.corrcoef(value, y)[0, 1]))
            summaries.append((abs(corr), f"corr({name}, label)={corr:+.2f}"))
    required = []
    for feature in ep.route_indices:
        corr = float(np.nan_to_num(np.corrcoef(ep.x_context[:, feature], y)[0, 1]))
        required.append(f"corr({ep.feature_names[feature]}, label)={corr:+.2f}")
    top = "; ".join(required + [text for _, text in sorted(summaries, reverse=True)[:8]])
    prompts = []
    for row, probability in zip(ep.x_query, specialist_probability):
        fields = ", ".join(f"{name}={value:.2f}" for name, value in zip(ep.feature_names, row))
        prompts.append(
            "Binary classification task. " + ep.description + "\n"
            f"A frozen statistical specialist reports P(label=1)={probability:.3f}. "
            f"Its strongest context summaries are: {top}.\nQuery: {{{fields}}}\n"
            "Reconcile domain meaning with this local evidence. Return only FINAL: LOW or FINAL: HIGH."
        )
    return prompts


def tune_ensemble_alpha(labels: np.ndarray, llm_probability: np.ndarray, tfm_probability: np.ndarray) -> tuple[float, float]:
    best = (math.inf, 0.5)
    for alpha in np.linspace(0.0, 1.0, 21):
        probability = np.clip(alpha * llm_probability + (1 - alpha) * tfm_probability, 1e-6, 1 - 1e-6)
        loss = float(-np.mean(labels * np.log(probability) + (1 - labels) * np.log(1 - probability)))
        best = min(best, (loss, float(alpha)))
    return best[1], best[0]
