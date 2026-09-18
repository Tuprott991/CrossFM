from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import accuracy_score

from .io import atomic_json, canonical_digest, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wheel", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.output)
    methods = tuple(config["experiment"]["methods"])
    world_size = int(config["runtime"]["expected_gpus"])
    expected = {
        f"phase3-{method}-{regime}-seed{seed}-rank{rank}"
        for method in methods for regime in config["experiment"]["regimes"]
        for seed in config["experiment"]["test_seeds"] for rank in range(world_size)
    }
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((output / "tasks").glob("*.json"))]
    actual = {record["experiment_id"] for record in records}
    if actual != expected or len(records) != len(actual):
        raise RuntimeError(f"Task grid mismatch missing={sorted(expected-actual)} unexpected={sorted(actual-expected)}")
    digest, wheel_sha = canonical_digest(config), sha256_file(Path(args.wheel))
    for record in records:
        if record["status"] != "complete" or record["config_digest"] != digest or record["wheel_sha256"] != wheel_sha:
            raise RuntimeError(f"Invalid provenance for {record['experiment_id']}")
        prediction = output / record["prediction_file"]
        if sha256_file(prediction) != record["prediction_sha256"]:
            raise RuntimeError(f"Prediction checksum mismatch for {record['experiment_id']}")
        arrays = np.load(prediction, allow_pickle=False)
        if len(arrays["probability"]) != record["n_predictions"] or not np.isfinite(arrays["probability"]).all():
            raise RuntimeError(f"Invalid predictions for {record['experiment_id']}")

    workers = [json.loads((output / f"worker_{rank}_summary.json").read_text(encoding="utf-8")) for rank in range(world_size)]
    if any(worker["status"] != "complete" or worker["config_digest"] != digest for worker in workers):
        raise RuntimeError("Worker summary validation failed")
    for rounds in {str(record["rounds"]) for record in records}:
        if len({worker["state_digests"][rounds] for worker in workers}) != 1:
            raise RuntimeError(f"Replicated training diverged for {rounds} rounds")

    seed_scores = {regime: {method: [] for method in methods} for regime in config["experiment"]["regimes"]}
    combined_predictions = {}
    for regime in config["experiment"]["regimes"]:
        for method in methods:
            for seed in config["experiment"]["test_seeds"]:
                shards = sorted(
                    (record for record in records if record["condition"] == regime and record["method"] == method and record["seed"] == seed),
                    key=lambda record: record["rank"],
                )
                arrays = [np.load(output / record["prediction_file"], allow_pickle=False) for record in shards]
                probability = np.concatenate([array["probability"] for array in arrays])
                labels = np.concatenate([array["label"] for array in arrays])
                score = float(accuracy_score(labels, probability >= 0.5))
                seed_scores[regime][method].append(score)
                combined_predictions[(regime, method, int(seed))] = probability

    results = {}
    for regime in config["experiment"]["regimes"]:
        results[regime] = {}
        for method in methods:
            values = np.asarray(seed_scores[regime][method])
            standard_error = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
            results[regime][method] = {
                "mean_accuracy": float(values.mean()),
                "std_accuracy": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "ci95_t": [float(values.mean() - 4.303 * standard_error), float(values.mean() + 4.303 * standard_error)],
                "seed_accuracies": values.tolist(),
            }

    gates = config["gate"]
    checks = {
        "overfit_accuracy": min(worker["training"]["2"]["train_accuracy"] for worker in workers) >= float(gates["overfit_accuracy_min"]),
        "nonzero_bridge_gradients": min(worker["training"]["2"]["max_gradient_norm"] for worker in workers) > 0.0,
        "training_messages_change_predictions": min(
            worker["training"]["2"]["train_zero_message_mean_absolute_delta"] for worker in workers
        ) >= float(gates["message_delta_min"]),
        "A_exact_preservation": results["A"]["crossfm_r2"]["mean_accuracy"] >= float(gates["a_accuracy_min"]),
        "B_retention": results["B"]["crossfm_r2"]["mean_accuracy"] >= float(gates["b_accuracy_min"]),
        "C2_recurrent_target": results["C2"]["crossfm_r2"]["mean_accuracy"] >= float(gates["c2_accuracy_min"]),
        "C2_recurrence_gain": results["C2"]["crossfm_r2"]["mean_accuracy"]
        >= results["C2"]["crossfm_1"]["mean_accuracy"] + float(gates["recurrence_margin"]),
        "C2_zero_message_drop": results["C2"]["crossfm_r2"]["mean_accuracy"]
        >= results["C2"]["crossfm_r2_zero"]["mean_accuracy"] + float(gates["ablation_margin"]),
        "C2_shuffle_message_drop": results["C2"]["crossfm_r2"]["mean_accuracy"]
        >= results["C2"]["crossfm_r2_shuffle"]["mean_accuracy"] + float(gates["ablation_margin"]),
    }

    fields = [
        "experiment_id", "protocol_id", "git_commit", "seed", "rank", "method", "condition", "rounds",
        "score", "roc_auc", "log_loss", "trainable_params", "conceptual_specialist_calls",
        "cached_specialist_view_calls", "llm_calls", "cache_bytes_worker", "peak_gpu_memory_bytes_worker",
        "config_digest", "wheel_sha256",
    ]
    with (output / "experiments.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(records)

    summary = {
        "status": "complete", "protocol_id": config["experiment"]["protocol_id"],
        "classification": config["experiment"]["classification"], "config_digest": digest,
        "wheel_sha256": wheel_sha, "expected_tasks": len(expected), "validated_tasks": len(records),
        "results": results, "phase3_checks": checks, "phase3_passed": all(checks.values()),
        "trainable_params": max(worker["trainable_params_max"] for worker in workers),
        "cache_bytes_per_worker": [worker["cache_bytes"] for worker in workers],
        "backbone_peak_gpu_memory_bytes": max(worker["backbone_peak_gpu_memory_bytes"] for worker in workers),
        "peak_gpu_memory_bytes": max(worker["peak_gpu_memory_bytes"] for worker in workers),
        "worker_elapsed_seconds": [worker["elapsed_seconds"] for worker in workers],
        "training": [worker["training"] for worker in workers],
        "scientific_scope": "Exploratory Phase 3 cached-view CrossFM recurrence gate; frozen backbones, no novelty claim.",
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
