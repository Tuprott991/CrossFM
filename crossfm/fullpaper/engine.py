from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from .artifacts import atomic_json, digest_json, reusable_record, sha256_file, write_task_record
from .data import DatasetBundle, load_dataset, schema_condition
from .metrics import adaptive_ece, binary_metrics
from .models import (
    cached_llm_outputs, candidate_views, cosine_semantic_logits,
    fit_predict_tabular, row_prompts,
)
from .protocol import ExperimentTask
from .routing import (
    ARPlusRouter, analytical_route, corrective_route, factorized_view_posterior,
    routing_features,
)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _cache_stem(task: ExperimentTask) -> str:
    return digest_json({
        "protocol": task.protocol_id,
        "dataset": task.dataset,
        "seed": task.seed,
        "condition": task.schema_condition,
        "budget": task.context_budget,
        "config_digest": task.config_digest,
    })[:24]


def _stratified_budget(indices: np.ndarray, labels: np.ndarray, budget: int | str, seed: int) -> np.ndarray:
    if budget == "full" or int(budget) >= len(indices):
        return np.asarray(indices)
    from sklearn.model_selection import train_test_split

    selected, _ = train_test_split(
        np.asarray(indices), train_size=int(budget), random_state=seed,
        stratify=np.asarray(labels)[indices],
    )
    return np.sort(selected)


def _stratified_cap(indices: np.ndarray, labels: np.ndarray, cap: int, seed: int) -> np.ndarray:
    if cap >= len(indices):
        return np.asarray(indices)
    from sklearn.model_selection import train_test_split

    selected, _ = train_test_split(
        np.asarray(indices), train_size=cap, random_state=seed,
        stratify=np.asarray(labels)[indices],
    )
    return np.sort(selected)


def _router_split(
    indices: np.ndarray,
    labels: np.ndarray,
    seed: int,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if groups is not None:
        from sklearn.model_selection import GroupShuffleSplit

        splitter = GroupShuffleSplit(n_splits=20, test_size=0.2, random_state=seed + 991)
        for fit_pos, router_pos in splitter.split(indices, labels[indices], groups[indices]):
            fit, router = np.asarray(indices)[fit_pos], np.asarray(indices)[router_pos]
            if len(np.unique(labels[fit])) >= 2 and len(np.unique(labels[router])) >= 2:
                return np.sort(fit), np.sort(router)
        raise ValueError("Group-isolated router split must contain both classes")
    from sklearn.model_selection import train_test_split

    fit, router = train_test_split(
        np.asarray(indices), test_size=0.2, random_state=seed + 991,
        stratify=np.asarray(labels)[indices],
    )
    return np.sort(fit), np.sort(router)


def _group_cap(
    indices: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    cap: int,
    seed: int,
) -> np.ndarray:
    """Deterministically cap evaluation rows without splitting customer groups."""
    indices = np.asarray(indices)
    if cap >= len(indices):
        return indices
    rng = np.random.default_rng(seed)
    group_values = np.asarray(groups)[indices]
    unique = np.unique(group_values)
    rng.shuffle(unique)
    selected: list[int] = []
    for group in unique:
        members = indices[group_values == group]
        if selected and len(selected) + len(members) > cap:
            continue
        selected.extend(map(int, members))
        if len(selected) >= cap:
            break
    result = np.sort(np.asarray(selected, dtype=int))
    if not len(result) or len(np.unique(labels[result])) < 2:
        raise ValueError("Group-preserving evaluation cap did not retain both classes")
    return result


def _binary_loss_per_view(labels: np.ndarray, bank: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=float)[:, None]
    bank = np.clip(np.asarray(bank, dtype=float), 1e-6, 1 - 1e-6)
    return -np.mean(labels * np.log(bank) + (1 - labels) * np.log(1 - bank), axis=0)


def _posterior_from_bank(
    validation_bank: np.ndarray,
    validation_labels: np.ndarray,
    target_bank: np.ndarray,
    temperature: float,
) -> np.ndarray:
    skill = -_binary_loss_per_view(validation_labels, validation_bank)
    certainty = np.abs(target_bank - 0.5) * 2.0
    evidence = skill[None, :] + certainty
    return factorized_view_posterior(evidence, temperature=temperature)


def _tune_ensemble(labels: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    best = (float("inf"), 0.5)
    for alpha in np.linspace(0, 1, 101):
        probability = alpha * left + (1 - alpha) * right
        loss = binary_metrics(labels, probability)["log_loss"]
        best = min(best, (loss, float(alpha)))
    return best[1]


class FullPaperEngine:
    def __init__(self, config: dict[str, Any], output: Path, data_root: Path, device: str):
        self.config = config
        self.output = output
        self.data_root = data_root
        self.device = device
        self._datasets: dict[str, DatasetBundle] = {}

    def dataset(self, dataset_id: str) -> DatasetBundle:
        if dataset_id not in self._datasets:
            self._datasets[dataset_id] = load_dataset(
                dataset_id, self.config["datasets"][dataset_id], self.data_root,
            )
        return self._datasets[dataset_id]

    def cache_path(self, task: ExperimentTask, kind: str) -> Path:
        if kind == "cache_llm":
            neutral = ExperimentTask(
                task_id=task.task_id, profile=task.profile, stage=task.stage,
                dataset=task.dataset, method=task.method, seed=0,
                schema_condition=task.schema_condition, context_budget="static_four_shot",
                cost=task.cost, protocol_id=task.protocol_id, config_digest=task.config_digest,
            )
            stem = _cache_stem(neutral)
        elif kind.startswith("cache_tab"):
            neutral = ExperimentTask(
                task_id=task.task_id, profile=task.profile, stage=task.stage,
                dataset=task.dataset, method=task.method, seed=task.seed,
                schema_condition="statistical_shared", context_budget=task.context_budget,
                cost=task.cost, protocol_id=task.protocol_id, config_digest=task.config_digest,
            )
            stem = _cache_stem(neutral)
        else:
            stem = _cache_stem(task)
        return self.output / "cache" / task.dataset / stem / f"{kind}.npz"

    def _slice(self, bundle: DatasetBundle, task: ExperimentTask):
        labels = bundle.target.to_numpy()
        train_indices = _stratified_budget(
            bundle.splits["train"], labels, task.context_budget, task.seed,
        )
        sampling_seed = int(self.config["runtime"].get("evaluation_sampling_seed", 260922))
        validation_cap = int(self.config["runtime"].get(
            "max_validation_rows", len(bundle.splits["validation"]),
        ))
        test_cap = int(self.config["runtime"].get("max_test_rows", len(bundle.splits["test"])))
        if bundle.groups is not None:
            group_values = bundle.groups.to_numpy()
            validation_indices = _group_cap(
                bundle.splits["validation"], labels, group_values, validation_cap, sampling_seed,
            )
            test_indices = _group_cap(
                bundle.splits["test"], labels, group_values, test_cap, sampling_seed + 1,
            )
        else:
            validation_indices = _stratified_cap(
                bundle.splits["validation"], labels, validation_cap, sampling_seed,
            )
            test_indices = _stratified_cap(
                bundle.splits["test"], labels, test_cap, sampling_seed + 1,
            )
        return train_indices, validation_indices, test_indices

    def _static_llm_train_indices(self, bundle: DatasetBundle, task: ExperimentTask) -> np.ndarray:
        """Cover every router split that consumes the shared seed-neutral LLM cache."""
        profile = self.config["profiles"][task.profile]
        seeds = profile.get("seeds", self.config["experiment"]["seeds"])
        budgets = profile.get("context_budgets", self.config["experiment"]["context_budgets"])
        labels = bundle.target.to_numpy()
        router_indices = []
        for budget in budgets:
            for seed in seeds:
                train = _stratified_budget(bundle.splits["train"], labels, budget, int(seed))
                _, router = _router_split(
                    train, labels, int(seed),
                    None if bundle.groups is None else bundle.groups.to_numpy(),
                )
                router_indices.append(router)
        return np.unique(np.concatenate(router_indices))

    def run_response_bank(self, task: ExperimentTask, method: dict[str, Any]) -> dict[str, Any]:
        bundle = self.dataset(task.dataset)
        train_idx, validation_idx, test_idx = self._slice(bundle, task)
        fit_idx, router_idx = _router_split(
            train_idx, bundle.target.to_numpy(), task.seed,
            None if bundle.groups is None else bundle.groups.to_numpy(),
        )
        views = candidate_views(bundle.frame, self.config["datasets"][task.dataset])
        router_bank, validation_bank, test_bank, metadata = [], [], [], {}
        backend = method["backend"]
        for name, columns in views.items():
            router_result = fit_predict_tabular(
                backend,
                bundle.frame.iloc[fit_idx][columns], bundle.target.iloc[fit_idx].to_numpy(),
                bundle.frame.iloc[router_idx][columns], bundle.frame.iloc[router_idx][columns],
                seed=task.seed, device=self.device, params=method.get("params", {}),
            )
            result = fit_predict_tabular(
                backend,
                bundle.frame.iloc[train_idx][columns], bundle.target.iloc[train_idx].to_numpy(),
                bundle.frame.iloc[validation_idx][columns], bundle.frame.iloc[test_idx][columns],
                seed=task.seed, device=self.device, params=method.get("params", {}),
            )
            router_bank.append(router_result.validation_probability)
            validation_bank.append(result.validation_probability)
            test_bank.append(result.test_probability)
            metadata[name] = result.metadata
        path = self.cache_path(task, task.method)
        _atomic_npz(
            path,
            router_bank=np.stack(router_bank, axis=1),
            validation_bank=np.stack(validation_bank, axis=1),
            test_bank=np.stack(test_bank, axis=1),
            router_labels=bundle.target.iloc[router_idx].to_numpy(dtype=np.int8),
            validation_labels=bundle.target.iloc[validation_idx].to_numpy(dtype=np.int8),
            test_labels=bundle.target.iloc[test_idx].to_numpy(dtype=np.int8),
            router_indices=router_idx, validation_indices=validation_idx, test_indices=test_idx,
            view_names=np.asarray(list(views)),
        )
        return {"cache": str(path), "sha256": sha256_file(path), "views": list(views), "metadata": metadata}

    def run_llm_cache(self, task: ExperimentTask, method: dict[str, Any]) -> dict[str, Any]:
        bundle = self.dataset(task.dataset)
        train_idx, validation_idx, test_idx = self._slice(bundle, task)
        fit_idx, router_idx = _router_split(
            train_idx, bundle.target.to_numpy(), task.seed,
            None if bundle.groups is None else bundle.groups.to_numpy(),
        )
        static_llm = task.method == "cache_llm"
        llm_train_indices = self._static_llm_train_indices(bundle, task) if static_llm else router_idx
        dataset_spec = self.config["datasets"][task.dataset]
        accelerator = self.config["profiles"][task.profile]["accelerator"]
        dtype = "float16" if accelerator == "kaggle_t4x2" else method.get("dtype", "bfloat16")
        model_id = method.get("t4_model_id", method["model_id"]) if accelerator == "kaggle_t4x2" else method["model_id"]
        revision = method.get("t4_revision", method["revision"]) if accelerator == "kaggle_t4x2" else method["revision"]
        likelihood_batch_size = (
            min(2, int(method.get("likelihood_batch_size", 4)))
            if accelerator == "kaggle_t4x2" else int(method.get("likelihood_batch_size", 4))
        )
        views = candidate_views(bundle.frame, dataset_spec)
        schema_seed = int(self.config["runtime"].get("schema_perturbation_seed", 260923))
        descriptions = schema_condition(
            bundle.feature_descriptions, task.schema_condition, seed=schema_seed,
        )
        if task.schema_condition == "aliases":
            if not bundle.feature_aliases:
                raise ValueError(f"Dataset {task.dataset} lacks manually validated feature aliases")
            display_names = dict(bundle.feature_aliases)
            descriptions = {
                column: alias.replace("_", " ") for column, alias in display_names.items()
            }
        else:
            display_names = {
                column: (f"x_{index + 1}" if task.schema_condition == "anonymized" else column)
                for index, column in enumerate(bundle.frame.columns)
            }
        view_display_names = {
            name: (f"view_{index + 1}" if task.schema_condition == "anonymized" else name)
            for index, name in enumerate(views)
        }
        view_texts = [
            f"View {view_display_names[name]}: "
            + "; ".join(descriptions.get(column, display_names[column]) for column in columns)
            for name, columns in views.items()
        ]
        texts = [bundle.task_description] + view_texts
        router_prompts = row_prompts(
            bundle.frame.iloc[llm_train_indices], bundle.task_description, descriptions,
            context_frame=None if static_llm else bundle.frame.iloc[fit_idx],
            context_labels=None if static_llm else bundle.target.iloc[fit_idx].to_numpy(),
            max_context_rows=0 if static_llm else int(method.get("max_context_rows", 4)),
            display_names=display_names,
        )
        validation_prompts = row_prompts(
            bundle.frame.iloc[validation_idx], bundle.task_description, descriptions,
            context_frame=None if static_llm else bundle.frame.iloc[train_idx],
            context_labels=None if static_llm else bundle.target.iloc[train_idx].to_numpy(),
            max_context_rows=0 if static_llm else int(method.get("max_context_rows", 4)),
            display_names=display_names,
        )
        test_prompts = row_prompts(
            bundle.frame.iloc[test_idx], bundle.task_description, descriptions,
            context_frame=None if static_llm else bundle.frame.iloc[train_idx],
            context_labels=None if static_llm else bundle.target.iloc[train_idx].to_numpy(),
            max_context_rows=0 if static_llm else int(method.get("max_context_rows", 4)),
            display_names=display_names,
        )
        if method.get("tool_response_cache"):
            response = self._load_cache(task, method["tool_response_cache"])
            names = [str(value) for value in response["view_names"]]
            def add_tool(prompts, bank):
                result = []
                for prompt, values in zip(prompts, bank):
                    summary = ", ".join(
                        f"{name}={float(value):.4f}" for name, value in zip(names, values)
                    )
                    result.append(prompt.replace(
                        "\nReturn the more likely class label.",
                        f"\nStatistical tool output (candidate probabilities): {summary}"
                        "\nReturn the more likely class label.",
                    ))
                return result
            router_prompts = add_tool(router_prompts, response["router_bank"])
            validation_prompts = add_tool(validation_prompts, response["validation_bank"])
            test_prompts = add_tool(test_prompts, response["test_bank"])
        embeddings, all_probability = cached_llm_outputs(
            texts, router_prompts + validation_prompts + test_prompts,
            model_id=model_id, revision=revision,
            embedding_batch_size=int(method.get("embedding_batch_size", 8)),
            likelihood_batch_size=likelihood_batch_size, device=self.device, dtype=dtype,
        )
        router_end = len(router_prompts)
        validation_end = router_end + len(validation_prompts)
        router_probability = all_probability[:router_end]
        validation_probability = all_probability[router_end:validation_end]
        test_probability = all_probability[validation_end:]
        path = self.cache_path(task, task.method)
        _atomic_npz(
            path, task_embedding=embeddings[0], view_embeddings=embeddings[1:],
            router_probability=router_probability,
            validation_probability=validation_probability, test_probability=test_probability,
            view_names=np.asarray(list(views)), router_indices=llm_train_indices,
            validation_indices=validation_idx,
            test_indices=test_idx,
        )
        return {"cache": str(path), "sha256": sha256_file(path), "views": list(views)}

    def _load_cache(self, task: ExperimentTask, kind: str) -> dict[str, np.ndarray]:
        path = self.cache_path(task, kind)
        if not path.is_file():
            raise FileNotFoundError(f"Required immutable cache missing: {path}")
        return dict(np.load(path, allow_pickle=False))

    def _base_caches(self, task: ExperimentTask, method: dict[str, Any]):
        response_kind = method.get("response_cache", "cache_tabicl")
        llm_kind = method.get("llm_cache", "cache_llm")
        return self._load_cache(task, response_kind), self._load_cache(task, llm_kind)

    def run_evaluation(self, task: ExperimentTask, method: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        bundle = self.dataset(task.dataset)
        train_idx, validation_idx, test_idx = self._slice(bundle, task)
        kind = method["kind"]
        details: dict[str, Any] = {}
        if kind == "direct_tabular":
            result = fit_predict_tabular(
                method["backend"], bundle.frame.iloc[train_idx],
                bundle.target.iloc[train_idx].to_numpy(), bundle.frame.iloc[validation_idx],
                bundle.frame.iloc[test_idx], seed=task.seed, device=self.device,
                params=method.get("params", {}),
            )
            probability = result.test_probability
            validation_probability = result.validation_probability
            details = result.metadata
        else:
            response, llm = self._base_caches(task, method)
            router_bank = response["router_bank"]
            validation_bank = response["validation_bank"]
            test_bank = response["test_bank"]
            router_labels = response["router_labels"]
            validation_labels = response["validation_labels"]
            test_labels = response["test_labels"]
            names = [str(value) for value in response["view_names"]]
            full_index = names.index("full_table")
            tfm_router = router_bank[:, full_index]
            tfm_validation = validation_bank[:, full_index]
            tfm_test = test_bank[:, full_index]
            llm_cache_indices = np.asarray(llm["router_indices"], dtype=int)
            response_router_indices = np.asarray(response["router_indices"], dtype=int)
            if np.array_equal(llm_cache_indices, response_router_indices):
                llm_router = llm["router_probability"]
            else:
                positions = {int(index): position for position, index in enumerate(llm_cache_indices)}
                try:
                    llm_router = llm["router_probability"][
                        [positions[int(index)] for index in response_router_indices]
                    ]
                except KeyError as exc:
                    raise RuntimeError("LLM cache does not cover the router-training split") from exc
            llm_validation = llm["validation_probability"]
            llm_test = llm["test_probability"]
            if self.config["routing"]["fallback"] == "llm":
                fallback_router, fallback_validation, fallback_test = (
                    llm_router, llm_validation, llm_test,
                )
            else:
                fallback_router, fallback_validation, fallback_test = (
                    tfm_router, tfm_validation, tfm_test,
                )
            if kind == "tfm_only":
                probability = tfm_test
                validation_probability = tfm_validation
            elif kind == "llm_only":
                probability = llm_test
                validation_probability = llm_validation
            elif kind == "ensemble":
                alpha = _tune_ensemble(validation_labels, llm_validation, tfm_validation)
                probability = alpha * llm_test + (1 - alpha) * tfm_test
                validation_probability = alpha * llm_validation + (1 - alpha) * tfm_validation
                details["alpha_llm"] = alpha
            elif kind == "tfm_to_llm":
                from sklearn.linear_model import LogisticRegression

                router_features = np.column_stack((
                    llm_router, router_bank.mean(1), router_bank.std(1),
                    router_bank.max(1), router_bank.min(1),
                ))
                validation_features = np.column_stack((
                    llm_validation, validation_bank.mean(1), validation_bank.std(1),
                    validation_bank.max(1), validation_bank.min(1),
                ))
                test_features = np.column_stack((
                    llm_test, test_bank.mean(1), test_bank.std(1),
                    test_bank.max(1), test_bank.min(1),
                ))
                projector = LogisticRegression(C=1.0, max_iter=2000, random_state=task.seed)
                projector.fit(router_features, router_labels)
                validation_probability = projector.predict_proba(validation_features)[:, 1]
                probability = projector.predict_proba(test_features)[:, 1]
                details["trainable_params"] = int(projector.coef_.size + projector.intercept_.size)
            else:
                semantic = cosine_semantic_logits(llm["task_embedding"], llm["view_embeddings"])
                posterior_router = _posterior_from_bank(
                    router_bank, router_labels, router_bank,
                    float(method.get("posterior_temperature", 0.25)),
                )
                posterior_validation = _posterior_from_bank(
                    router_bank, router_labels, validation_bank,
                    float(method.get("posterior_temperature", 0.25)),
                )
                posterior_test = _posterior_from_bank(
                    router_bank, router_labels, test_bank,
                    float(method.get("posterior_temperature", 0.25)),
                )
                if kind in {"semantic_oneway", "semantic_oneway_compute_matched"}:
                    posterior_validation = np.full_like(
                        posterior_validation, 1 / posterior_validation.shape[1],
                    )
                    posterior_test = np.full_like(posterior_test, 1 / posterior_test.shape[1])
                elif kind == "no_semantic":
                    semantic = np.zeros_like(semantic)
                elif kind == "uniform_ablation":
                    posterior_validation = np.full_like(
                        posterior_validation, 1 / posterior_validation.shape[1],
                    )
                    posterior_test = np.full_like(posterior_test, 1 / posterior_test.shape[1])
                elif kind == "hard_ablation":
                    hard_validation = np.argmax(posterior_validation, axis=1)
                    posterior_validation = np.eye(posterior_validation.shape[1])[hard_validation]
                    hard = np.argmax(posterior_test, axis=1)
                    posterior_test = np.eye(posterior_test.shape[1])[hard]
                elif kind == "shuffle_message":
                    rng = np.random.default_rng(task.seed)
                    posterior_validation = posterior_validation[rng.permutation(len(posterior_validation))]
                    posterior_test = posterior_test[np.random.default_rng(task.seed).permutation(len(posterior_test))]
                elif kind == "random_bridge":
                    semantic = np.random.default_rng(task.seed).normal(size=semantic.shape)
                if kind in {"semantic_oneway", "semantic_oneway_compute_matched"}:
                    repetitions = int(method.get("logical_rounds", 1))
                    weights = None
                    for _ in range(repetitions):
                        weights = np.exp(semantic - np.max(semantic))
                        weights /= weights.sum()
                        validation_probability = validation_bank @ weights
                        probability = test_bank @ weights
                    details["semantic_weights"] = weights.tolist()
                    details["logical_rounds"] = repetitions
                elif kind == "round1":
                    probability = fallback_test.copy()
                    validation_probability = fallback_validation.copy()
                    details["rounds"] = 1
                else:
                    router_route = analytical_route(
                        router_bank, semantic, posterior_router, fallback_router,
                        temperature=float(method.get("route_temperature", 1.0)),
                        gate_floor=float(method.get("gate_floor", 0.0)),
                    )
                    validation_route = analytical_route(
                        validation_bank, semantic, posterior_validation, fallback_validation,
                        temperature=float(method.get("route_temperature", 1.0)),
                        gate_floor=float(method.get("gate_floor", 0.0)),
                    )
                    route = analytical_route(
                        test_bank, semantic, posterior_test, fallback_test,
                        temperature=float(method.get("route_temperature", 1.0)),
                        gate_floor=float(method.get("gate_floor", 0.0)),
                    )
                    if kind == "gate_disabled":
                        probability = route.routed_probability
                        validation_probability = validation_route.routed_probability
                        details["mean_gate"] = 1.0
                    elif kind == "arplus_shuffle_response":
                        permutation = np.random.default_rng(task.seed).permutation(test_bank.shape[1])
                        shuffled_bank = test_bank[:, permutation]
                        router = ARPlusRouter(
                            feature_dim=5, hidden_dim=int(method.get("hidden_dim", 32)), seed=task.seed,
                        )
                        fit = router.fit(
                            analytical_logits=router_route.analytical_logits,
                            features=routing_features(router_bank, semantic, posterior_router),
                            response_bank=router_bank, fallback_probability=fallback_router,
                            gate=router_route.gate, labels=router_labels, device=self.device,
                            epochs=int(method.get("epochs", 100)), anchor_kl=float(method.get("anchor_kl", 0.02)),
                            correction_l2=float(method.get("correction_l2", 0.001)),
                        )
                        validation_probability, _ = router.predict(
                            analytical_logits=validation_route.analytical_logits,
                            features=routing_features(validation_bank[:, permutation], semantic, posterior_validation),
                            response_bank=validation_bank[:, permutation],
                            fallback_probability=fallback_validation, gate=validation_route.gate,
                            device=self.device,
                        )
                        probability, _ = router.predict(
                            analytical_logits=route.analytical_logits,
                            features=routing_features(shuffled_bank, semantic, posterior_test),
                            response_bank=shuffled_bank, fallback_probability=fallback_test,
                            gate=route.gate, device=self.device,
                        )
                        details.update(fit)
                    elif kind in {"arplus", "arplus_no_anchor"}:
                        router = ARPlusRouter(feature_dim=5, hidden_dim=int(method.get("hidden_dim", 32)), seed=task.seed)
                        train_features = routing_features(validation_bank, semantic, posterior_validation)
                        fit = router.fit(
                            analytical_logits=router_route.analytical_logits,
                            features=routing_features(router_bank, semantic, posterior_router),
                            response_bank=router_bank,
                            fallback_probability=fallback_router, gate=router_route.gate,
                            labels=router_labels, device=self.device,
                            epochs=int(method.get("epochs", 100)),
                            anchor_kl=0.0 if kind == "arplus_no_anchor" else float(method.get("anchor_kl", 0.02)),
                            correction_l2=float(method.get("correction_l2", 0.001)),
                        )
                        validation_probability, _ = router.predict(
                            analytical_logits=validation_route.analytical_logits,
                            features=train_features, response_bank=validation_bank,
                            fallback_probability=fallback_validation, gate=validation_route.gate,
                            device=self.device,
                        )
                        probability, traces = router.predict(
                            analytical_logits=route.analytical_logits,
                            features=routing_features(test_bank, semantic, posterior_test),
                            response_bank=test_bank, fallback_probability=fallback_test,
                            gate=route.gate, device=self.device,
                        )
                        details.update(fit)
                        details["mean_abs_correction"] = float(np.mean(np.abs(traces["correction"])))
                    elif kind == "smr":
                        from sklearn.linear_model import LogisticRegression

                        x_router = np.column_stack((router_bank, llm_router))
                        x_validation = np.column_stack((validation_bank, llm_validation))
                        x_test = np.column_stack((test_bank, llm_test))
                        model = LogisticRegression(C=1.0, max_iter=2000, random_state=task.seed)
                        model.fit(x_router, router_labels)
                        validation_probability = model.predict_proba(x_validation)[:, 1]
                        probability = model.predict_proba(x_test)[:, 1]
                        details["trainable_params"] = int(model.coef_.size + model.intercept_.size)
                    elif kind == "round3":
                        validation_round3 = corrective_route(
                            router_route, router_bank, router_labels,
                            validation_route, validation_bank, fallback_validation,
                            temperature=float(method.get("route_temperature", 1.0)),
                            strength=float(method.get("correction_strength", 1.0)),
                        )
                        test_round3 = corrective_route(
                            router_route, router_bank, router_labels,
                            route, test_bank, fallback_test,
                            temperature=float(method.get("route_temperature", 1.0)),
                            strength=float(method.get("correction_strength", 1.0)),
                        )
                        validation_probability = validation_round3.probability
                        probability = test_round3.probability
                        details.update({
                            "mean_gate": float(np.mean(test_round3.gate)),
                            "preserved_fraction": float(np.mean(test_round3.gate == 0)),
                            "rounds": 3,
                            "third_beat": "confidence_scaled_router_residual_replay",
                        })
                    else:
                        probability = route.probability
                        validation_probability = validation_route.probability
                        details.update({
                            "mean_gate": float(np.mean(route.gate)),
                            "preserved_fraction": float(np.mean(route.gate == 0)),
                            "rounds": 3 if kind == "round3" else 2,
                        })
            if not np.array_equal(test_labels, bundle.target.iloc[test_idx].to_numpy(dtype=np.int8)):
                raise RuntimeError("Cache/test label mismatch")
        labels = bundle.target.iloc[test_idx].to_numpy(dtype=np.int8)
        metrics = binary_metrics(labels, probability)
        metrics["adaptive_ece"] = adaptive_ece(labels, probability)
        validation_labels = bundle.target.iloc[validation_idx].to_numpy(dtype=np.int8)
        validation_metrics = binary_metrics(validation_labels, validation_probability)
        validation_metrics["adaptive_ece"] = adaptive_ece(
            validation_labels, validation_probability,
        )
        arrays_path = self.output / "arrays" / f"{task.task_id}.npz"
        _atomic_npz(
            arrays_path, probability=np.asarray(probability), labels=labels,
            indices=np.asarray(test_idx), groups=(
                bundle.groups.iloc[test_idx].to_numpy(dtype=str)
                if bundle.groups is not None else np.asarray(test_idx).astype(str)
            ),
        )
        return {
            "metrics": metrics,
            "validation_metrics": validation_metrics,
            "details": details,
            "prediction_artifact": str(arrays_path),
            "prediction_sha256": sha256_file(arrays_path),
            "dataset_checksum": bundle.checksum,
            "split_hash": bundle.split_digest,
            "runtime_seconds": time.perf_counter() - started,
            "train_rows": len(train_idx), "test_rows": len(test_idx),
        }

    def run(self, task: ExperimentTask) -> tuple[dict[str, Any], Path | None]:
        method = self.config["methods"][task.method]
        if task.stage == "response_bank":
            result = self.run_response_bank(task, method)
            return result, Path(result["cache"])
        if task.stage == "llm_cache":
            result = self.run_llm_cache(task, method)
            return result, Path(result["cache"])
        if task.stage == "evaluate":
            result = self.run_evaluation(task, method)
            return result, Path(result["prediction_artifact"])
        raise KeyError(f"Unsupported stage: {task.stage}")


def execute_task(engine: FullPaperEngine, task: ExperimentTask, wheel_sha256: str = "local") -> str:
    record = engine.output / "records" / f"{task.task_id}.json"
    if reusable_record(
        record, task_id=task.task_id, protocol_id=task.protocol_id,
        config_digest=task.config_digest,
    ):
        return "reused"
    try:
        started = time.perf_counter()
        peak_memory_bytes = None
        if engine.device.startswith("cuda"):
            import torch
            torch.cuda.reset_peak_memory_stats()
        payload, arrays_path = engine.run(task)
        payload.setdefault("runtime_seconds", time.perf_counter() - started)
        if engine.device.startswith("cuda"):
            peak_memory_bytes = int(torch.cuda.max_memory_allocated())
        method_spec = engine.config["methods"][task.method]
        payload.update({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task": asdict(task), "wheel_sha256": wheel_sha256,
            "runtime_device": engine.device,
            "peak_accelerator_memory_bytes": peak_memory_bytes,
            "declared_inference_calls": int(method_spec.get("inference_calls", 0)),
            "logical_rounds": int(method_spec.get("logical_rounds", 1)),
            "git_commit": engine.config["source"]["git_commit"],
            "source_dirty": engine.config["source"]["dirty"],
        })
        write_task_record(
            engine.output, task_id=task.task_id, protocol_id=task.protocol_id,
            config_digest=task.config_digest, payload=payload, arrays_path=arrays_path,
        )
        return "complete"
    except Exception as exc:
        atomic_json(engine.output / "failures" / f"{task.task_id}.json", {
            "status": "failed", "task": asdict(task), "error_type": type(exc).__name__,
            "error": str(exc), "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        raise
