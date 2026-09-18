from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import numpy as np

from .synthetic import Episode


@dataclass
class PredictionBatch:
    probabilities: np.ndarray
    labels: np.ndarray
    episode_ids: np.ndarray


def semantic_statistical_oracle(episodes: Iterable[Episode]) -> PredictionBatch:
    """Diagnostic upper bound using the declared semantic pair plus an empirical sign.

    This is not a CrossFM implementation and is never presented as a learned baseline.
    It checks whether Regime C contains recoverable complementary information.
    """
    probs, labels, ids = [], [], []
    for ep in episodes:
        if ep.regime == "A":
            score = ep.x_query[:, ep.relevant[0]]
        elif ep.regime == "B":
            score = 1.4 * ep.x_query[:, 0] * ep.x_query[:, 1] + 0.55 * ep.x_query[:, 2] - 0.25 * ep.x_query[:, 3]
        else:
            a, b = ep.relevant
            if ep.mechanism in {"pilot", "train"}:
                context_signal = ep.x_context[:, b] - ep.x_context[:, a]
                query_signal = ep.x_query[:, b] - ep.x_query[:, a]
            elif ep.mechanism == "validation":
                context_signal = ep.x_context[:, b] + ep.x_context[:, a]
                query_signal = ep.x_query[:, b] + ep.x_query[:, a]
            elif ep.mechanism == "test":
                context_signal = ep.x_context[:, b] * ep.x_context[:, a]
                query_signal = ep.x_query[:, b] * ep.x_query[:, a]
            else:
                raise ValueError(ep.mechanism)
            corr = np.corrcoef(context_signal, ep.y_context)[0, 1]
            direction = 1.0 if np.nan_to_num(corr) >= 0 else -1.0
            score = direction * query_signal
        probs.extend((1.0 / (1.0 + np.exp(-2.0 * score))).tolist())
        labels.extend(ep.y_query.tolist())
        ids.extend([ep.episode_id] * len(ep.y_query))
    return PredictionBatch(np.asarray(probs), np.asarray(labels), np.asarray(ids))


def adaptive_routing_oracle(episodes: Iterable[Episode]) -> PredictionBatch:
    """Diagnostic two-stage oracle for C2; ordinary oracle elsewhere."""
    probs, labels, ids = [], [], []
    for ep in episodes:
        if ep.regime != "C2":
            batch = semantic_statistical_oracle([ep])
            probs.extend(batch.probabilities.tolist()); labels.extend(batch.labels.tolist()); ids.extend(batch.episode_ids.tolist())
            continue
        code = 0
        for bit, feature in enumerate(ep.route_indices):
            corr = float(np.nan_to_num(np.corrcoef(ep.x_context[:, feature], ep.y_context)[0, 1]))
            if corr >= 0:
                code |= 1 << bit
        group = ep.route_map[code]
        a = len(ep.route_indices) + 2 * group
        b = a + 1
        if ep.mechanism == "train":
            context_signal = ep.x_context[:, b] - ep.x_context[:, a]
            query_signal = ep.x_query[:, b] - ep.x_query[:, a]
        elif ep.mechanism == "validation":
            context_signal = ep.x_context[:, b] + ep.x_context[:, a]
            query_signal = ep.x_query[:, b] + ep.x_query[:, a]
        elif ep.mechanism == "test":
            context_signal = ep.x_context[:, b] * ep.x_context[:, a]
            query_signal = ep.x_query[:, b] * ep.x_query[:, a]
        else:
            raise ValueError(ep.mechanism)
        direction = 1.0 if np.nan_to_num(np.corrcoef(context_signal, ep.y_context)[0, 1]) >= 0 else -1.0
        score = direction * query_signal
        probs.extend((1.0 / (1.0 + np.exp(-2.0 * score))).tolist())
        labels.extend(ep.y_query.tolist()); ids.extend([ep.episode_id] * len(ep.y_query))
    return PredictionBatch(np.asarray(probs), np.asarray(labels), np.asarray(ids))


def build_llm_prompts(ep: Episode, max_context_rows: int = 48) -> list[str]:
    rows = min(len(ep.y_context), 6 if ep.regime == "C2" else max_context_rows)
    def named_row(values: np.ndarray, indices: Iterable[int] | None = None) -> str:
        selected = range(len(ep.feature_names)) if indices is None else indices
        return ", ".join(f"{ep.feature_names[i]}={values[i]:.2f}" for i in selected)

    if ep.regime == "A":
        # The context is deliberately confounded in the semantics-dominant arm.
        # Supplying it caused a 0.6B model to follow episode-wide shortcuts rather
        # than the explicitly stable domain relation, so the preregistered v2
        # semantic baseline uses metadata plus query values only for this arm.
        prefix = (
            "Binary classification task. " + ep.description + "\n"
            "The tiny labeled context is non-identifying and omitted. Apply the stable domain relation to the query.\n"
            "Return only FINAL: LOW or FINAL: HIGH, where HIGH means label 1.\n"
        )
    else:
        examples = []
        for x, y in zip(ep.x_context[:rows], ep.y_context[:rows]):
            examples.append(f"{{{named_row(x)}}} -> {'HIGH' if y else 'LOW'}")
        prefix = (
            "Binary classification task. " + ep.description + "\n"
            "Labeled examples:\n" + "\n".join(examples) + "\n"
            "Use both the task meaning and examples. Return only FINAL: LOW or FINAL: HIGH, where HIGH means label 1.\n"
        )
    query_indices = ep.relevant if ep.regime == "A" else None
    return [prefix + "Query: {" + named_row(x, query_indices) + "}\nLabel:" for x in ep.x_query]


class QwenBinaryBaseline:
    def __init__(
        self, model_id: str, revision: str, device: str = "cuda:0", batch_size: int = 8,
        torch_dtype: str = "float16",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(torch_dtype)
        if dtype is None:
            raise ValueError(f"Unsupported Qwen dtype: {torch_dtype}")
        if dtype is torch.bfloat16 and device.startswith("cuda") and not torch.cuda.is_bf16_supported():
            raise RuntimeError("bfloat16 was requested but is not supported by this CUDA device")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, dtype=dtype, attn_implementation="sdpa"
        ).to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.device = device

    def hidden_state_preflight(self) -> tuple[int, ...]:
        inputs = self.tokenizer("schema: income premium", return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            out = self.model(**inputs, output_hidden_states=True, use_cache=False)
        if not out.hidden_states:
            raise RuntimeError("Qwen did not expose hidden states")
        return tuple(out.hidden_states[-1].shape)

    def predict(self, episodes: Iterable[Episode]) -> PredictionBatch:
        prompts, labels, ids = [], [], []
        for ep in episodes:
            raw = build_llm_prompts(ep)
            prompts.extend(raw)
            labels.extend(ep.y_query.tolist())
            ids.extend([ep.episode_id] * len(ep.y_query))
        probabilities = self.predict_prompts_likelihood(prompts)
        return PredictionBatch(np.asarray(probabilities), np.asarray(labels), np.asarray(ids))

    def predict_prompts_likelihood(self, prompts: list[str]) -> np.ndarray:
        """Score complete LOW/HIGH verbalizers without free-form parsing.

        This evaluates the joint token likelihood of each exact candidate
        sequence.  It cannot fail because the model chose to begin an
        explanation, and it does not approximate a multi-token label with only
        its first token.
        """
        candidate_texts = ("FINAL: LOW", "FINAL: HIGH")
        candidate_ids = [
            self.tokenizer(text, add_special_tokens=False)["input_ids"]
            for text in candidate_texts
        ]
        scored: list[tuple[int, int, list[int], int]] = []
        for prompt_index, prompt in enumerate(prompts):
            rendered = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True,
            )
            prompt_ids = self.tokenizer(rendered, add_special_tokens=False)["input_ids"]
            for candidate_index, suffix in enumerate(candidate_ids):
                scored.append((prompt_index, candidate_index, prompt_ids + suffix, len(suffix)))
        scores = np.empty((len(prompts), 2), dtype=np.float64)
        for start in range(0, len(scored), self.batch_size * 2):
            group = scored[start:start + self.batch_size * 2]
            batch = self.tokenizer.pad(
                {"input_ids": [item[2] for item in group]}, padding=True, return_tensors="pt",
            ).to(self.device)
            with self.torch.inference_mode():
                # Avoid materializing [batch, long_prompt, full_vocabulary]
                # logits. Only the few hidden positions that predict the
                # candidate suffix need to pass through the LM head.
                hidden = self.model.model(**batch, use_cache=False).last_hidden_state
            sequence_length = batch["input_ids"].shape[1]
            suffix_states, suffix_targets, destinations = [], [], []
            for row, (prompt_index, candidate_index, _ids, suffix_length) in enumerate(group):
                token_positions = self.torch.arange(
                    sequence_length - suffix_length - 1, sequence_length - 1, device=self.device,
                )
                suffix_states.append(hidden[row, token_positions])
                suffix_targets.append(batch["input_ids"][row, -suffix_length:])
                destinations.append((prompt_index, candidate_index, suffix_length))
            with self.torch.inference_mode():
                selected_states = self.torch.cat(suffix_states, dim=0)
                selected_targets = self.torch.cat(suffix_targets, dim=0)
                suffix_logits = self.model.lm_head(selected_states).float()
                token_log_probs = self.torch.log_softmax(suffix_logits, dim=-1)[
                    self.torch.arange(len(selected_targets), device=self.device), selected_targets
                ]
            offset = 0
            for prompt_index, candidate_index, suffix_length in destinations:
                scores[prompt_index, candidate_index] = float(token_log_probs[offset:offset + suffix_length].sum().cpu())
                offset += suffix_length
        normalized = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(normalized)
        return exp_scores[:, 1] / exp_scores.sum(axis=1)

    def predict_prompts(self, prompts: list[str]) -> np.ndarray:
        probabilities = []
        for start in range(0, len(prompts), self.batch_size):
            rendered = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in prompts[start:start + self.batch_size]
            ]
            batch = self.tokenizer(rendered, add_special_tokens=False, padding=True, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **batch, max_new_tokens=8, do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            new_tokens = generated[:, batch["input_ids"].shape[1]:]
            decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for text in decoded:
                matches = re.findall(r"\b(?:LOW|HIGH)\b", text.upper())
                if len(set(matches)) != 1:
                    raise RuntimeError(f"Invalid constrained LLM response: {text!r}")
                probabilities.append(0.999 if matches[-1] == "HIGH" else 0.001)
        return np.asarray(probabilities)

    def encode_texts(self, texts: list[str], max_length: int = 160) -> np.ndarray:
        """Mean-pool frozen final hidden states for lightweight adapters."""
        encoded: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(
                texts[start:start + self.batch_size], padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            ).to(self.device)
            with self.torch.inference_mode():
                hidden = self.model(**batch, output_hidden_states=True, use_cache=False).hidden_states[-1]
                mask = batch["attention_mask"].unsqueeze(-1)
                pooled = (hidden.float() * mask).sum(1) / mask.sum(1).clamp_min(1)
            encoded.append(pooled.cpu().numpy())
        return np.concatenate(encoded, axis=0)


class TabICLBaseline:
    def __init__(self, repo_id: str, revision: str, checkpoint: str, device: str = "cuda:0", n_estimators: int = 4):
        from huggingface_hub import hf_hub_download
        from tabicl import TabICLClassifier

        model_path = hf_hub_download(repo_id=repo_id, revision=revision, filename=checkpoint)
        self.classifier = TabICLClassifier(
            model_path=model_path,
            allow_auto_download=False,
            checkpoint_version=checkpoint,
            device=device,
            use_amp=True,
            use_fa3=False,
            n_estimators=n_estimators,
            verbose=False,
        )

    def preflight(self) -> None:
        rng = np.random.default_rng(7)
        x = rng.normal(size=(32, 3)).astype(np.float32)
        y = (x[:, 0] > 0).astype(int)
        self.classifier.fit(x[:24], y[:24])
        pred = self.classifier.predict_proba(x[24:])
        if pred.shape != (8, 2) or not np.isfinite(pred).all():
            raise RuntimeError(f"TabICL preflight failed: {pred.shape}")

    def predict(self, episodes: Iterable[Episode]) -> PredictionBatch:
        probs, labels, ids = [], [], []
        for ep in episodes:
            self.classifier.fit(ep.x_context, ep.y_context)
            pred = self.classifier.predict_proba(ep.x_query)[:, 1]
            probs.extend(pred.tolist())
            labels.extend(ep.y_query.tolist())
            ids.extend([ep.episode_id] * len(ep.y_query))
        return PredictionBatch(np.asarray(probs), np.asarray(labels), np.asarray(ids))

    def predict_episode_with_evidence(self, ep: Episode) -> tuple[np.ndarray, np.ndarray]:
        """Return probabilities and frozen 512-d TabICL ICL states.

        The hook is placed after the final ICL layer normalization and before
        TabICL's decoder.  Ensemble representations are class-shuffle invariant
        and are averaged across estimators.  No specialist parameter is trained.
        """
        import torch

        captured: list[np.ndarray] = []

        def capture(_module, _inputs, output):
            if torch.is_tensor(output):
                captured.append(output.detach().float().cpu().numpy()[:, -len(ep.x_query):, :])

        self.classifier.fit(ep.x_context, ep.y_context)
        handle = self.classifier.model_.icl_predictor.ln.register_forward_hook(capture)
        try:
            probability = self.classifier.predict_proba(ep.x_query)[:, 1]
        finally:
            handle.remove()
        if not captured:
            raise RuntimeError("TabICL evidence hook captured no ICL states")
        evidence = np.concatenate(captured, axis=0).mean(axis=0)
        if evidence.shape[0] != len(ep.x_query) or not np.isfinite(evidence).all():
            raise RuntimeError(f"Invalid TabICL evidence shape {evidence.shape}")
        return probability, evidence
