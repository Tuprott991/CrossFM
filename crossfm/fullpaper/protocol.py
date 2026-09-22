from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import re
from typing import Any

import yaml

from .artifacts import digest_json


_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


@dataclass(frozen=True, slots=True)
class ExperimentTask:
    task_id: str
    profile: str
    stage: str
    dataset: str
    method: str
    seed: int
    schema_condition: str
    context_budget: int | str
    cost: float
    protocol_id: str
    config_digest: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_protocol(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Protocol root must be a mapping")
    validate_protocol(value)
    return value


def validate_protocol(config: dict[str, Any]) -> None:
    for key in ("experiment", "datasets", "methods", "profiles", "runtime", "source"):
        if key not in config:
            raise ValueError(f"Missing protocol section: {key}")
    experiment = config["experiment"]
    protocol_id = str(experiment.get("protocol_id", ""))
    if not _ID.match(protocol_id):
        raise ValueError(f"Invalid protocol_id: {protocol_id!r}")
    if experiment.get("classification") != "exploratory_non_confirmatory":
        raise ValueError("Full-paper run must remain exploratory_non_confirmatory")
    if not experiment.get("seeds") or len(set(experiment["seeds"])) != len(experiment["seeds"]):
        raise ValueError("Experiment seeds must be a non-empty unique list")
    datasets = config["datasets"]
    methods = config["methods"]
    if not isinstance(datasets, dict) or not isinstance(methods, dict):
        raise ValueError("datasets and methods must be mappings")
    for dataset_id, spec in datasets.items():
        if not _ID.match(dataset_id) or spec.get("split") not in {"temporal", "grouped", "iid"}:
            raise ValueError(f"Invalid dataset specification: {dataset_id}")
        if not spec.get("checksum_required", True):
            raise ValueError(f"Dataset {dataset_id} must require a checksum")
    for name, spec in methods.items():
        if not _ID.match(name):
            raise ValueError(f"Invalid method name: {name}")
        if int(spec.get("inference_calls", 0)) < 0:
            raise ValueError(f"Invalid inference call count for {name}")
    for profile, spec in config["profiles"].items():
        unknown_datasets = set(spec.get("datasets", [])) - set(datasets)
        unknown_methods = set(spec.get("methods", [])) - set(methods)
        if unknown_datasets or unknown_methods:
            raise ValueError(
                f"Profile {profile} references unknown datasets={unknown_datasets}, "
                f"methods={unknown_methods}"
            )
        profile_methods = set(spec.get("methods", []))
        for method_name in profile_methods:
            method = methods[method_name]
            for dependency_key in ("response_cache", "llm_cache", "tool_response_cache"):
                dependency = method.get(dependency_key)
                if dependency and dependency not in profile_methods:
                    raise ValueError(
                        f"Profile {profile} schedules {method_name} without required "
                        f"{dependency_key} producer {dependency}"
                    )
        if spec.get("accelerator") not in {"h100_80gb", "kaggle_t4x2", "cpu"}:
            raise ValueError(f"Invalid accelerator for profile {profile}")
        if spec.get("owner_role") not in {"author_a", "author_b", "any_author"}:
            raise ValueError(f"Invalid owner role for profile {profile}")
        frozen_from = spec.get("selection_frozen_from")
        if frozen_from and frozen_from not in config["profiles"]:
            raise ValueError(f"Profile {profile} freezes selection from unknown profile {frozen_from}")
    routing = config.get("routing", {})
    if routing.get("residual_bypass") != "exact":
        raise ValueError("The preservation bypass must be exact")
    if routing.get("representation") != "continuous_posterior":
        raise ValueError("Routing must preserve a continuous posterior")
    if routing.get("fallback") != "llm":
        raise ValueError("Zero statistical structure must delegate exactly to the LLM")
    if config["source"].get("dirty") not in {False, "BUILD_TIME"}:
        raise ValueError("source.dirty must be false or BUILD_TIME")
    disk_requirement = config["runtime"].get("minimum_free_disk_gib", 0)
    if isinstance(disk_requirement, dict):
        expected_accelerators = {"h100_80gb", "kaggle_t4x2", "cpu"}
        if set(disk_requirement) != expected_accelerators:
            raise ValueError(
                "minimum_free_disk_gib must define h100_80gb, kaggle_t4x2, and cpu"
            )
        values = disk_requirement.values()
    else:
        values = (disk_requirement,)
    if any(int(value) < 0 for value in values):
        raise ValueError("minimum_free_disk_gib values must be non-negative")
    accelerator_versions = config["runtime"].get(
        "expected_accelerator_package_versions", {},
    )
    if accelerator_versions and set(accelerator_versions) != {
        "h100_80gb", "kaggle_t4x2", "cpu",
    }:
        raise ValueError(
            "expected_accelerator_package_versions must define h100_80gb, "
            "kaggle_t4x2, and cpu"
        )


def _task_id(parts: list[str]) -> str:
    readable = "--".join(str(value).lower().replace("_", "-") for value in parts)
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:12]
    return f"{readable[:100]}--{digest}"


def tasks_for_profile(config: dict[str, Any], profile: str) -> list[ExperimentTask]:
    validate_protocol(config)
    if profile not in config["profiles"]:
        raise KeyError(f"Unknown profile: {profile}")
    spec = config["profiles"][profile]
    protocol_id = config["experiment"]["protocol_id"]
    config_digest = digest_json(config)
    seeds = spec.get("seeds", config["experiment"]["seeds"])
    conditions = spec.get("schema_conditions", config["experiment"]["schema_conditions"])
    budgets = spec.get("context_budgets", config["experiment"]["context_budgets"])
    stages = spec.get("stages", ["evaluate"])
    tasks: list[ExperimentTask] = []
    for stage in stages:
        for dataset in spec.get("datasets", []):
            dataset_spec = config["datasets"][dataset]
            for method in spec.get("methods", []):
                method_spec = config["methods"][method]
                if stage not in method_spec.get("stages", ["evaluate"]):
                    continue
                method_conditions = conditions if method_spec.get("schema_sensitive", True) else conditions[:1]
                method_seeds = seeds if method_spec.get("seed_sensitive", True) else seeds[:1]
                method_budgets = budgets if method_spec.get("context_sensitive", True) else budgets[:1]
                for condition in method_conditions:
                    if condition not in dataset_spec.get("schema_conditions", conditions):
                        continue
                    for budget in method_budgets:
                        for seed in method_seeds:
                            parts = [profile, stage, dataset, method, seed, condition, budget]
                            cost = float(dataset_spec.get("cost", 1.0)) * float(
                                method_spec.get("cost", 1.0)
                            )
                            tasks.append(ExperimentTask(
                                task_id=_task_id(parts), profile=profile, stage=stage,
                                dataset=dataset, method=method, seed=int(seed),
                                schema_condition=str(condition), context_budget=budget,
                                cost=cost, protocol_id=protocol_id,
                                config_digest=config_digest,
                            ))
    return sorted(tasks, key=lambda item: item.task_id)


def balanced_shards(tasks: list[ExperimentTask], world_size: int) -> list[list[ExperimentTask]]:
    if world_size < 1:
        raise ValueError("world_size must be positive")
    shards: list[list[ExperimentTask]] = [[] for _ in range(world_size)]
    loads = [0.0] * world_size
    for task in sorted(tasks, key=lambda item: (-item.cost, item.task_id)):
        rank = min(range(world_size), key=lambda index: (loads[index], index))
        shards[rank].append(task)
        loads[rank] += task.cost
    for shard in shards:
        shard.sort(key=lambda item: item.task_id)
    return shards
