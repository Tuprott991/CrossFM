from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from .io import atomic_json, canonical_digest, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wheel", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output, world_size = Path(args.output), int(config["runtime"]["expected_gpus"])
    digest, wheel_sha = canonical_digest(config), sha256_file(Path(args.wheel))
    methods, regimes, seeds = (
        config["experiment"]["methods"], config["experiment"]["regimes"],
        config["experiment"]["test_seeds"],
    )
    expected = {
        f"phase3-dynamic-{method}-{regime}-seed{seed}-rank{rank}"
        for method in methods for regime in regimes for seed in seeds for rank in range(world_size)
    }
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((output / "tasks").glob("*.json"))]
    actual = {record["experiment_id"] for record in records}
    if actual != expected or len(records) != len(actual):
        raise RuntimeError(f"Dynamic task grid mismatch missing={sorted(expected-actual)} unexpected={sorted(actual-expected)}")
    for record in records:
        if record["config_digest"] != digest or record["wheel_sha256"] != wheel_sha:
            raise RuntimeError(f"Provenance mismatch: {record['experiment_id']}")
        prediction = output / record["prediction_file"]
        if sha256_file(prediction) != record["prediction_sha256"]:
            raise RuntimeError(f"Prediction hash mismatch: {record['experiment_id']}")

    results = {regime: {} for regime in regimes}
    for regime in regimes:
        for method in methods:
            per_seed = []
            for seed in seeds:
                subset = [record for record in records if record["condition"] == regime and record["method"] == method and record["seed"] == seed]
                correct = sum(round(record["score"] * record["n_predictions"]) for record in subset)
                total = sum(record["n_predictions"] for record in subset)
                per_seed.append(correct / total)
            values = np.asarray(per_seed)
            results[regime][method] = {
                "mean_accuracy": float(values.mean()), "seed_accuracies": values.tolist(),
                "std_accuracy": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }

    gate = config["gate"]
    checks = {
        "A_preservation": results["A"]["dynamic_r2"]["mean_accuracy"] >= float(gate["a_accuracy_min"]),
        "B_preservation": results["B"]["dynamic_r2"]["mean_accuracy"] >= float(gate["b_accuracy_min"]),
        "C2_dynamic_target": results["C2"]["dynamic_r2"]["mean_accuracy"] >= float(gate["c2_accuracy_min"]),
        "C2_dynamic_recurrence": results["C2"]["dynamic_r2"]["mean_accuracy"]
        >= results["C2"]["dynamic_1"]["mean_accuracy"] + float(gate["recurrence_margin"]),
        "C2_zero_evidence_drop": results["C2"]["dynamic_r2"]["mean_accuracy"]
        >= results["C2"]["dynamic_r2_zero"]["mean_accuracy"] + float(gate["ablation_margin"]),
        "C2_shuffle_evidence_drop": results["C2"]["dynamic_r2"]["mean_accuracy"]
        >= results["C2"]["dynamic_r2_shuffle"]["mean_accuracy"] + float(gate["ablation_margin"]),
    }
    workers = [json.loads((output / f"worker_{rank}_summary.json").read_text(encoding="utf-8")) for rank in range(world_size)]
    summary = {
        "status": "complete", "protocol_id": config["experiment"]["protocol_id"],
        "classification": config["experiment"]["classification"], "config_digest": digest,
        "wheel_sha256": wheel_sha, "expected_tasks": len(expected), "validated_tasks": len(records),
        "results": results, "dynamic_checks": checks, "dynamic_passed": all(checks.values()),
        "controller_params": max(worker["controller_params"] for worker in workers),
        "bridge_params": max(worker["bridge_params"] for worker in workers),
        "peak_gpu_memory_bytes": max(worker["peak_gpu_memory_bytes"] for worker in workers),
        "worker_elapsed_seconds": [worker["elapsed_seconds"] for worker in workers],
        "dynamic_train_specialist_calls": [worker["dynamic_train_specialist_calls"] for worker in workers],
        "dynamic_test_specialist_calls": [worker["dynamic_test_specialist_calls"] for worker in workers],
        "static_token_cache_entries": [worker["static_token_cache_entries"] for worker in workers],
        "prefix_tokens_processed": [worker["prefix_tokens_processed"] for worker in workers],
        "bridge_training": [worker["bridge_training"] for worker in workers],
        "scientific_scope": (
            "Exploratory dynamic-transfer smoke: a cached-bank controller initializes routing, while evaluation "
            "uses fresh selected-view TabICL calls and frozen-Qwen soft-token updates with prefix KV caching."
        ),
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
