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
            context_diff = ep.x_context[:, b] - ep.x_context[:, a]
            corr = np.corrcoef(context_diff, ep.y_context)[0, 1]
            direction = 1.0 if np.nan_to_num(corr) >= 0 else -1.0
            score = direction * (ep.x_query[:, b] - ep.x_query[:, a])
        probs.extend((1.0 / (1.0 + np.exp(-2.0 * score))).tolist())
        labels.extend(ep.y_query.tolist())
        ids.extend([ep.episode_id] * len(ep.y_query))
    return PredictionBatch(np.asarray(probs), np.asarray(labels), np.asarray(ids))


def build_llm_prompts(ep: Episode, max_context_rows: int = 48) -> list[str]:
    rows = min(len(ep.y_context), max_context_rows)
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
    def __init__(self, model_id: str, revision: str, device: str = "cuda:0", batch_size: int = 8):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, torch_dtype=torch.float16, attn_implementation="sdpa"
        ).to(device).eval()
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
        return PredictionBatch(np.asarray(probabilities), np.asarray(labels), np.asarray(ids))


class TabICLBaseline:
    def __init__(self, repo_id: str, revision: str, checkpoint: str, device: str = "cuda:0"):
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
            n_estimators=4,
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
