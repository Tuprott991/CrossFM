from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import atomic_json, reusable_record
from .metrics import paired_cluster_bootstrap
from .protocol import ExperimentTask


def _holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, key in enumerate(ordered):
        value = min(1.0, (total - index) * p_values[key])
        running = max(running, value)
        adjusted[key] = running
    return adjusted


def aggregate_run(
    *,
    config: dict[str, Any],
    profile: str,
    tasks: list[ExperimentTask],
    output: Path,
) -> dict[str, Any]:
    records = []
    missing = []
    for task in tasks:
        path = output / "records" / f"{task.task_id}.json"
        if not reusable_record(
            path, task_id=task.task_id, protocol_id=task.protocol_id,
            config_digest=task.config_digest,
        ):
            missing.append(task.task_id)
            continue
        records.append(json.loads(path.read_text(encoding="utf-8")))
    if missing:
        raise RuntimeError(f"Cannot aggregate incomplete grid; missing/invalid={missing[:20]}")
    rows = []
    for record in records:
        task = record["payload"]["task"]
        metrics = record["payload"].get("metrics", {})
        validation_metrics = record["payload"].get("validation_metrics", {})
        if not all(np.isfinite(value) for value in list(metrics.values()) + list(validation_metrics.values())):
            raise RuntimeError(f"Non-finite metric in {task['task_id']}")
        if task["stage"] == "evaluate":
            rows.append({
                **{key: task[key] for key in (
                    "task_id", "dataset", "method", "seed", "schema_condition", "context_budget",
                )},
                **metrics,
                **{f"validation_{key}": value for key, value in validation_metrics.items()},
                "runtime_seconds": record["payload"].get("runtime_seconds"),
                "dataset_checksum": record["payload"].get("dataset_checksum"),
                "split_hash": record["payload"].get("split_hash"),
                "prediction_sha256": record["payload"].get("prediction_sha256"),
            })
    csv_path = output / "results.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    selection = _select_arplus(rows, config)
    summary = {
        "status": "complete",
        "protocol_id": config["experiment"]["protocol_id"],
        "classification": config["experiment"]["classification"],
        "profile": profile,
        "config_digest": tasks[0].config_digest if tasks else None,
        "expected_tasks": len(tasks), "validated_tasks": len(records),
        "evaluation_rows": len(rows), "selection": selection,
        "results_csv": str(csv_path) if rows else None,
    }
    atomic_json(output / "summary.json", summary)
    return summary


def _select_arplus(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    gate = config.get("selection", {})
    grouped: dict[tuple, dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (row["dataset"], row["seed"], row["schema_condition"], str(row["context_budget"]))
        grouped.setdefault(key, {})[row["method"]] = row
    gains, degradations = [], []
    for values in grouped.values():
        if "crossfm_ar" in values and "crossfm_arplus" in values:
            gain = (
                values["crossfm_ar"]["validation_log_loss"]
                - values["crossfm_arplus"]["validation_log_loss"]
            )
            gains.append(gain); degradations.append(max(0.0, -gain))
    mean_gain = float(np.mean(gains)) if gains else None
    maximum_degradation = float(np.max(degradations)) if degradations else None
    selected = bool(
        gains
        and mean_gain >= float(gate.get("mean_logloss_gain", 0.01))
        and maximum_degradation <= float(gate.get("max_dataset_degradation", 0.005))
    )
    return {
        "primary": "crossfm_arplus" if selected else "crossfm_ar",
        "arplus_selected": selected,
        "mean_logloss_gain": mean_gain,
        "maximum_degradation": maximum_degradation,
        "paired_cells": len(gains),
    }


def paired_comparison(
    output: Path,
    left_task: ExperimentTask,
    right_task: ExperimentTask,
    *,
    metric: str,
    replicates: int = 2000,
    seed: int = 2718,
) -> dict[str, Any]:
    def load(task: ExperimentTask):
        record = json.loads((output / "records" / f"{task.task_id}.json").read_text())
        arrays = np.load(output / record["arrays"]["path"], allow_pickle=False)
        return arrays

    left, right = load(left_task), load(right_task)
    if not np.array_equal(left["indices"], right["indices"]) or not np.array_equal(left["labels"], right["labels"]):
        raise RuntimeError("Paired comparison received different test sets")
    return paired_cluster_bootstrap(
        left["labels"], left["probability"], right["probability"],
        groups=left["groups"], metric=metric, replicates=replicates, seed=seed,
    )
