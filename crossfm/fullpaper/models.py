from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


_FROZEN_MODEL_CACHE: dict[tuple[Any, ...], Any] = {}
_TEXT_EMBEDDING_CACHE: dict[str, np.ndarray] = {}


@dataclass(slots=True)
class PredictionResult:
    validation_probability: np.ndarray
    test_probability: np.ndarray
    metadata: dict[str, Any]


def _finite_probability(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    if not np.isfinite(value).all():
        raise RuntimeError("Model produced non-finite probabilities")
    return np.clip(value, 1e-6, 1.0 - 1e-6)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sklearn_matrix(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric = list(train.select_dtypes(include=[np.number, "bool"]).columns)
    categorical = [column for column in train if column not in numeric]
    normalized = []
    for frame in (train, validation, test):
        value = frame.copy()
        for column in numeric:
            value[column] = pd.to_numeric(value[column], errors="coerce").astype(np.float64)
        for column in categorical:
            value[column] = value[column].astype("string").fillna("__MISSING__").astype(str)
        normalized.append(value)
    train, validation, test = normalized
    transformer = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("categorical", make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore", min_frequency=2),
        ), categorical),
    ])
    x_train = transformer.fit_transform(train)
    return x_train, transformer.transform(validation), transformer.transform(test), transformer


def fit_predict_tabular(
    method: str,
    train: pd.DataFrame,
    y_train: np.ndarray,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    seed: int,
    device: str,
    params: dict[str, Any] | None = None,
) -> PredictionResult:
    params = dict(params or {})
    if train.shape[1] == 0:
        prior = float(np.mean(y_train))
        return PredictionResult(
            np.full(len(validation), prior), np.full(len(test), prior),
            {"backend": "empirical_prior", "featureless": True},
        )
    if method == "logistic":
        x_train, x_validation, x_test, _ = _sklearn_matrix(train, validation, test)
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(
            C=float(params.get("C", 1.0)), max_iter=int(params.get("max_iter", 2000)),
            class_weight=params.get("class_weight", "balanced"), random_state=seed,
        )
        model.fit(x_train, y_train)
        validation_probability = model.predict_proba(x_validation)[:, 1]
        test_probability = model.predict_proba(x_test)[:, 1]
    elif method == "tabm":
        import torch
        try:
            from tabm import TabM
        except ImportError as exc:
            raise RuntimeError("Install the pinned official tabm package") from exc
        x_train, x_validation, x_test, _ = _sklearn_matrix(train, validation, test)
        x_train = np.asarray(x_train.toarray() if hasattr(x_train, "toarray") else x_train, dtype=np.float32)
        x_validation = np.asarray(
            x_validation.toarray() if hasattr(x_validation, "toarray") else x_validation,
            dtype=np.float32,
        )
        x_test = np.asarray(x_test.toarray() if hasattr(x_test, "toarray") else x_test, dtype=np.float32)
        torch.manual_seed(seed)
        model = TabM.make(
            n_num_features=x_train.shape[1], cat_cardinalities=[], d_out=1,
            k=int(params.get("k", 32)), arch_type=params.get("arch_type", "tabm-mini"),
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(params.get("learning_rate", 0.002)),
            weight_decay=float(params.get("weight_decay", 0.0003)),
        )
        batch_size = int(params.get("batch_size", 1024))
        epochs = int(params.get("epochs", 100))
        train_tensor = torch.from_numpy(x_train)
        target_tensor = torch.as_tensor(y_train, dtype=torch.float32)
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            model.train()
            order = rng.permutation(len(x_train))
            for start in range(0, len(x_train), batch_size):
                indices = order[start:start + batch_size]
                xb = train_tensor[indices].to(device)
                yb = target_tensor[indices].to(device)[:, None, None]
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb, None)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, yb.expand_as(logits),
                )
                loss.backward(); optimizer.step()
        def predict_batches(values: np.ndarray) -> np.ndarray:
            chunks = []
            model.eval()
            with torch.inference_mode():
                for start in range(0, len(values), batch_size):
                    logits = model(torch.from_numpy(values[start:start + batch_size]).to(device), None)
                    chunks.append(logits.sigmoid().mean(1).squeeze(-1).cpu().numpy())
            return np.concatenate(chunks)
        validation_probability = predict_batches(x_validation)
        test_probability = predict_batches(x_test)
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    elif method == "autogluon":
        import tempfile
        from autogluon.tabular import TabularPredictor

        label = "__crossfm_target__"
        with tempfile.TemporaryDirectory(prefix="crossfm-autogluon-") as temporary:
            training = train.copy(); training[label] = y_train
            predictor = TabularPredictor(
                label=label, problem_type="binary", eval_metric="log_loss", path=temporary,
                verbosity=0,
            ).fit(
                training, presets=params.get("presets", "best_quality"),
                time_limit=int(params.get("time_limit", 3600)),
                num_cpus=int(params.get("num_cpus", 4)),
                num_gpus=int(params.get("num_gpus", 1 if device.startswith("cuda") else 0)),
            )
            validation_probability = predictor.predict_proba(validation, as_multiclass=False).to_numpy()
            test_probability = predictor.predict_proba(test, as_multiclass=False).to_numpy()
    elif method == "catboost":
        from catboost import CatBoostClassifier

        categorical = [column for column in train if not pd.api.types.is_numeric_dtype(train[column])]
        combined = pd.concat((train, validation, test), ignore_index=True)
        for column in categorical:
            combined[column] = combined[column].fillna("__MISSING__").astype(str)
        train2 = combined.iloc[:len(train)]
        validation2 = combined.iloc[len(train):len(train) + len(validation)]
        test2 = combined.iloc[len(train) + len(validation):]
        defaults = dict(
            iterations=800, depth=8, learning_rate=0.05, loss_function="Logloss",
            random_seed=seed, verbose=False, allow_writing_files=False,
            task_type="GPU" if device.startswith("cuda") else "CPU",
        )
        defaults.update(params)
        model = CatBoostClassifier(**defaults)
        model.fit(train2, y_train, cat_features=categorical)
        validation_probability = model.predict_proba(validation2)[:, 1]
        test_probability = model.predict_proba(test2)[:, 1]
    elif method in {"lightgbm", "xgboost"}:
        x_train, x_validation, x_test, _ = _sklearn_matrix(train, validation, test)
        if method == "lightgbm":
            from lightgbm import LGBMClassifier

            defaults = dict(
                n_estimators=800, learning_rate=0.03, num_leaves=63,
                random_state=seed, n_jobs=params.pop("n_jobs", 4), verbosity=-1,
            )
            defaults.update(params)
            model = LGBMClassifier(**defaults)
        else:
            from xgboost import XGBClassifier

            defaults = dict(
                n_estimators=800, max_depth=8, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, random_state=seed,
                tree_method="hist", device="cuda" if device.startswith("cuda") else "cpu",
            )
            defaults.update(params)
            model = XGBClassifier(**defaults)
        model.fit(x_train, y_train)
        validation_probability = model.predict_proba(x_validation)[:, 1]
        test_probability = model.predict_proba(x_test)[:, 1]
    elif method == "tabicl":
        from tabicl import TabICLClassifier

        x_train, x_validation, x_test, _ = _sklearn_matrix(train, validation, test)
        for name, matrix in (("train", x_train), ("validation", x_validation), ("test", x_test)):
            if hasattr(matrix, "toarray"):
                matrix = matrix.toarray()
            if name == "train": x_train = np.asarray(matrix, dtype=np.float32)
            elif name == "validation": x_validation = np.asarray(matrix, dtype=np.float32)
            else: x_test = np.asarray(matrix, dtype=np.float32)
        prediction_chunk_size = int(params.pop("prediction_chunk_size", 256))
        key = ("tabicl", device, seed, tuple(sorted((name, repr(value)) for name, value in params.items())))
        model = _FROZEN_MODEL_CACHE.get(key)
        if model is None:
            model = TabICLClassifier(device=device, random_state=seed, **params)
            _FROZEN_MODEL_CACHE[key] = model
        model.fit(x_train, y_train)
        def predict_chunks(values: np.ndarray) -> np.ndarray:
            return np.concatenate([
                model.predict_proba(values[start:start + prediction_chunk_size])[:, 1]
                for start in range(0, len(values), prediction_chunk_size)
            ])
        validation_probability = predict_chunks(x_validation)
        test_probability = predict_chunks(x_test)
        params["prediction_chunk_size"] = prediction_chunk_size
    elif method in {"tabpfn", "tabpfn3"}:
        try:
            from tabpfn import TabPFNClassifier
        except ImportError as exc:
            raise RuntimeError("Install the pinned official TabPFN package/checkpoint") from exc
        repo_id = params.pop("hf_repo_id", None)
        revision = params.pop("hf_revision", None)
        filename = params.pop("hf_filename", None)
        expected_sha256 = params.pop("hf_sha256", None)
        checkpoint = None
        if repo_id or revision or filename or expected_sha256:
            if not all((repo_id, revision, filename, expected_sha256)):
                raise ValueError("TabPFN Hugging Face provenance must be fully specified")
            from huggingface_hub import hf_hub_download

            checkpoint = Path(hf_hub_download(
                repo_id=str(repo_id), filename=str(filename), revision=str(revision),
            ))
            observed_sha256 = _sha256(checkpoint)
            if observed_sha256 != expected_sha256:
                raise RuntimeError(
                    f"TabPFN checkpoint SHA-256 mismatch: {observed_sha256} != {expected_sha256}"
                )
        key = (
            "tabpfn3", device, seed, str(repo_id), str(revision), str(filename),
            tuple(sorted((name, repr(value)) for name, value in params.items())),
        )
        model = _FROZEN_MODEL_CACHE.get(key)
        if model is None:
            model = TabPFNClassifier(
                device=device, random_state=seed,
                **({"model_path": str(checkpoint)} if checkpoint else {}), **params,
            )
            _FROZEN_MODEL_CACHE[key] = model
        model.fit(train, y_train)
        validation_probability = model.predict_proba(validation)[:, 1]
        test_probability = model.predict_proba(test)[:, 1]
        if checkpoint is not None:
            params.update({
                "hf_repo_id": repo_id, "hf_revision": revision,
                "hf_filename": filename, "hf_sha256": expected_sha256,
            })
    else:
        raise KeyError(f"Unsupported tabular method: {method}")
    return PredictionResult(
        _finite_probability(validation_probability), _finite_probability(test_probability),
        {"backend": method, "params": params, "features": train.shape[1]},
    )


def candidate_views(frame: pd.DataFrame, dataset_spec: dict[str, Any]) -> dict[str, list[str]]:
    views: dict[str, list[str]] = {}
    for name, view in dataset_spec.get("views", {}).items():
        explicit = [column for column in view.get("columns", []) if column in frame]
        patterns = [re.compile(pattern, re.IGNORECASE) for pattern in view.get("patterns", [])]
        matched = [column for column in frame if any(pattern.search(column) for pattern in patterns)]
        columns = list(dict.fromkeys(explicit + matched))
        if columns:
            views[name] = columns
    numeric = list(frame.select_dtypes(include=[np.number, "bool"]).columns)
    categorical = [column for column in frame if column not in numeric]
    if numeric and "numeric" not in views:
        views["numeric"] = numeric
    if categorical and "categorical" not in views:
        views["categorical"] = categorical
    views["full_table"] = list(frame.columns)
    unique = {name: columns for name, columns in views.items() if columns}
    if len(unique) < 2:
        columns = list(frame)
        midpoint = max(1, len(columns) // 2)
        unique = {"schema_group_a": columns[:midpoint], "full_table": columns}
    return unique


def constrained_label_likelihoods(
    prompts: list[str],
    *,
    model_id: str,
    revision: str,
    batch_size: int,
    device: str,
    dtype: str,
    labels: tuple[str, str] = ("LOW", "HIGH"),
) -> np.ndarray:
    """Sequence-level normalized label likelihood; no free-form parsing."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=torch_dtype,
        attn_implementation="sdpa", device_map={"": device},
    ).eval()
    scores = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        per_label = []
        for label in labels:
            prefix = tokenizer(batch, return_tensors="pt", padding=True).to(device)
            suffix = tokenizer([label] * len(batch), add_special_tokens=False, return_tensors="pt").to(device)
            input_ids = torch.cat((prefix.input_ids, suffix.input_ids), dim=1)
            attention = torch.cat((prefix.attention_mask, suffix.attention_mask), dim=1)
            with torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch_dtype, enabled=device.startswith("cuda"),
            ):
                logits = model(input_ids=input_ids, attention_mask=attention, use_cache=True).logits
            begin = prefix.input_ids.shape[1] - 1
            selected = logits[:, begin:begin + suffix.input_ids.shape[1], :]
            token_logp = torch.log_softmax(selected.float(), -1).gather(
                -1, suffix.input_ids.unsqueeze(-1),
            ).squeeze(-1)
            per_label.append(token_logp.sum(-1).cpu().numpy())
        pair = np.stack(per_label, axis=1)
        pair -= pair.max(axis=1, keepdims=True)
        normalized = np.exp(pair); normalized /= normalized.sum(axis=1, keepdims=True)
        scores.append(normalized[:, 1])
    del model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return _finite_probability(np.concatenate(scores))


def cached_llm_outputs(
    texts: list[str],
    prompts: list[str],
    *,
    model_id: str,
    revision: str,
    embedding_batch_size: int,
    likelihood_batch_size: int,
    device: str,
    dtype: str,
    labels: tuple[str, str] = ("LOW", "HIGH"),
) -> tuple[np.ndarray, np.ndarray]:
    """Compute static embeddings and constrained probabilities with one model load.

    Prompt prefixes are evaluated once per batch and their KV states are reused for
    both candidate label sequences. This is safe because the candidate sequences do
    not share mutable generation state and no free-form decoding is performed.
    """

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    model_key = ("causal_llm", model_id, revision, device, dtype)
    cached = _FROZEN_MODEL_CACHE.get(model_key)
    if cached is None:
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, torch_dtype=torch_dtype,
            attn_implementation="sdpa", device_map={"": device},
        ).eval()
        base = getattr(model, "model", getattr(model, "base_model", None))
        _FROZEN_MODEL_CACHE[model_key] = (tokenizer, model, base)
    else:
        tokenizer, model, base = cached
    if base is None:
        raise RuntimeError("Causal LM does not expose a base hidden-state model")
    embedding_key = hashlib.sha256(
        (model_id + revision + "\0" + "\0".join(texts)).encode("utf-8")
    ).hexdigest()
    embeddings = _TEXT_EMBEDDING_CACHE.get(embedding_key)
    if embeddings is None:
        embedded = []
        for start in range(0, len(texts), embedding_batch_size):
            tokens = tokenizer(
                texts[start:start + embedding_batch_size], return_tensors="pt", padding=True,
                truncation=True, max_length=1024,
            ).to(device)
            with torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch_dtype, enabled=device.startswith("cuda"),
            ):
                output = base(**tokens, use_cache=False)
                hidden = output.last_hidden_state
            mask = tokens.attention_mask.unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            embedded.append(torch.nn.functional.normalize(pooled.float(), dim=-1).cpu().numpy())
        embeddings = np.concatenate(embedded)
        _TEXT_EMBEDDING_CACHE[embedding_key] = embeddings
    probabilities = []
    for start in range(0, len(prompts), likelihood_batch_size):
        batch = prompts[start:start + likelihood_batch_size]
        prefix = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=1536,
        ).to(device)
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch_dtype, enabled=device.startswith("cuda"),
        ):
            prefix_output = model(**prefix, use_cache=True)
        per_label = []
        for label in labels:
            suffix = tokenizer(
                [label] * len(batch), add_special_tokens=False, return_tensors="pt",
            ).input_ids.to(device)
            extended_attention = torch.cat((
                prefix.attention_mask,
                torch.ones(suffix.shape, dtype=prefix.attention_mask.dtype, device=device),
            ), dim=1)
            with torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch_dtype, enabled=device.startswith("cuda"),
            ):
                continuation = model(
                    input_ids=suffix,
                    attention_mask=extended_attention,
                    past_key_values=prefix_output.past_key_values,
                    use_cache=False,
                ).logits
            first = prefix_output.logits[:, -1:, :]
            logits = torch.cat((first, continuation[:, :-1, :]), dim=1)
            score = torch.log_softmax(logits.float(), -1).gather(
                -1, suffix.unsqueeze(-1),
            ).squeeze(-1).sum(-1)
            per_label.append(score.cpu().numpy())
        pair = np.stack(per_label, axis=1)
        pair -= pair.max(axis=1, keepdims=True)
        normalized = np.exp(pair); normalized /= normalized.sum(axis=1, keepdims=True)
        probabilities.append(normalized[:, 1])
        del prefix_output
    return embeddings.copy(), _finite_probability(np.concatenate(probabilities))


def schema_embeddings(
    texts: list[str],
    *,
    model_id: str,
    revision: str,
    batch_size: int,
    device: str,
    dtype: str,
) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    model = AutoModel.from_pretrained(
        model_id, revision=revision, torch_dtype=torch_dtype,
        attn_implementation="sdpa", device_map={"": device},
    ).eval()
    output = []
    for start in range(0, len(texts), batch_size):
        tokens = tokenizer(
            texts[start:start + batch_size], return_tensors="pt", padding=True,
            truncation=True, max_length=1024,
        ).to(device)
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch_dtype, enabled=device.startswith("cuda"),
        ):
            hidden = model(**tokens).last_hidden_state
        mask = tokens.attention_mask.unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)
        output.append(pooled.cpu().numpy())
    del model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return np.concatenate(output)


def row_prompts(
    frame: pd.DataFrame,
    task_description: str,
    descriptions: dict[str, str],
    *,
    max_columns: int = 32,
    context_frame: pd.DataFrame | None = None,
    context_labels: np.ndarray | None = None,
    max_context_rows: int = 4,
) -> list[str]:
    columns = list(frame.columns)[:max_columns]
    schema = "; ".join(f"{column}: {descriptions.get(column, column)}" for column in columns)
    examples = ""
    if context_frame is not None and context_labels is not None:
        context = context_frame[columns].iloc[:max_context_rows]
        labels = np.asarray(context_labels).reshape(-1)[:len(context)]
        lines = []
        for row, label in zip(context.itertuples(index=False, name=None), labels):
            values = "; ".join(f"{column}={value}" for column, value in zip(columns, row))
            lines.append(f"Example: {values}; label={'HIGH' if int(label) else 'LOW'}")
        examples = "\n" + "\n".join(lines)
    static_prefix = f"Task: {task_description}\nSchema: {schema}{examples}\n"
    prompts = []
    for row in frame[columns].itertuples(index=False, name=None):
        values = "; ".join(f"{column}={value}" for column, value in zip(columns, row))
        prompts.append(
            f"{static_prefix}Row: {values}\n"
            "Return the more likely class label. Answer LOW or HIGH.\nLabel:"
        )
    return prompts


def cosine_semantic_logits(task_embedding: np.ndarray, view_embeddings: np.ndarray) -> np.ndarray:
    task = np.asarray(task_embedding, dtype=np.float64).reshape(-1)
    views = np.asarray(view_embeddings, dtype=np.float64)
    task /= max(np.linalg.norm(task), 1e-12)
    views /= np.maximum(np.linalg.norm(views, axis=1, keepdims=True), 1e-12)
    return views @ task


def configure_h100_math() -> None:
    import torch

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
