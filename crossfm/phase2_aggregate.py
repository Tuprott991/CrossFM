from __future__ import annotations

import argparse
import csv
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
    methods = tuple(config["experiment"]["methods"])
    prefix = config["experiment"].get("artifact_prefix", "phase2")
    output = Path(args.output)
    expected = {
        f"{prefix}-{method}-{regime}-seed{seed}"
        for method in methods
        for regime in config["experiment"]["regimes"]
        for seed in config["experiment"]["test_seeds"]
    }
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((output / "tasks").glob("*.json"))]
    actual = {record["experiment_id"] for record in records}
    if actual != expected or len(records) != len(actual):
        raise RuntimeError(f"Task grid mismatch missing={sorted(expected-actual)} unexpected={sorted(actual-expected)} duplicates={len(records)-len(actual)}")
    digest, wheel_sha = canonical_digest(config), sha256_file(Path(args.wheel))
    for record in records:
        if record["config_digest"] != digest or record["wheel_sha256"] != wheel_sha or record["status"] != "complete":
            raise RuntimeError(f"Invalid provenance for {record['experiment_id']}")
        prediction = output / record["prediction_file"]
        if sha256_file(prediction) != record["prediction_sha256"]:
            raise RuntimeError(f"Prediction checksum mismatch for {record['experiment_id']}")
        values = np.load(prediction, allow_pickle=False)
        if len(values["probability"]) != record["n_predictions"] or not np.isfinite(values["probability"]).all():
            raise RuntimeError(f"Invalid prediction artifact for {record['experiment_id']}")
        for key in ("score", "roc_auc", "log_loss", "elapsed_seconds_shared_cell"):
            if not np.isfinite(record[key]):
                raise RuntimeError(f"Non-finite {key} for {record['experiment_id']}")
        if record["checkpoint_file"]:
            checkpoint = output / record["checkpoint_file"]
            if sha256_file(checkpoint) != record["checkpoint_sha256"]:
                raise RuntimeError(f"Checkpoint checksum mismatch for {record['experiment_id']}")

    fields = [
        "experiment_id", "protocol_id", "git_commit", "seed", "method", "condition",
        "score", "roc_auc", "log_loss", "trainable_params", "specialist_calls", "llm_calls",
        "elapsed_seconds_shared_cell", "peak_gpu_memory_bytes_worker", "config_digest", "wheel_sha256",
    ]
    csv_path = output / "experiments.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(records)

    results = {}
    paired = {}
    for regime in config["experiment"]["regimes"]:
        results[regime] = {}
        per_method = {}
        for method in methods:
            values = np.asarray([r["score"] for r in records if r["condition"] == regime and r["method"] == method])
            per_method[method] = values
            standard_error = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
            results[regime][method] = {
                "mean_accuracy": float(values.mean()),
                "std_accuracy": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "ci95_t": [float(values.mean() - 4.303 * standard_error), float(values.mean() + 4.303 * standard_error)],
            }
        individual = np.maximum(per_method["llm_only"], per_method["tabicl_only"])
        paired[regime] = {
            method: {
                "mean_accuracy_delta_vs_best_individual_per_seed": float(np.mean(values - individual)),
                "seed_deltas": (values - individual).tolist(),
            }
            for method, values in per_method.items()
            if method not in {"llm_only", "tabicl_only"}
        }

    worker_summaries = [json.loads((output / f"worker_{rank}_summary.json").read_text(encoding="utf-8")) for rank in range(2)]
    if any(item["status"] != "complete" or item["config_digest"] != digest for item in worker_summaries):
        raise RuntimeError("Worker summary validation failed")
    summary = {
        "status": "complete", "protocol_id": config["experiment"]["protocol_id"],
        "classification": config["experiment"]["classification"],
        "config_digest": digest, "wheel_sha256": wheel_sha,
        "expected_tasks": len(expected), "validated_tasks": len(records),
        "results": results, "paired_differences": paired,
        "trainable_params_total": max(item["trainable_params_total"] for item in worker_summaries),
        "peak_gpu_memory_bytes": max(item["peak_gpu_memory_bytes"] for item in worker_summaries),
        "worker_elapsed_seconds": [item["elapsed_seconds"] for item in worker_summaries],
        "scientific_scope": (
            "Exploratory Phase 2.5 adaptive-complementarity gate. No CrossFM result or novelty claim."
            if "C2" in config["experiment"]["regimes"]
            else "Exploratory Phase 2 baseline calibration only. No CrossFM result or novelty claim."
        ),
    }
    if "adaptive_diagnostic_oracle" in methods and "C2" in results:
        gate = config["gate"]
        baseline_methods = [method for method in methods if method != "adaptive_diagnostic_oracle"]
        checks = {
            "A_semantics_dominant": results["A"]["llm_only"]["mean_accuracy"] >= results["A"]["tabicl_only"]["mean_accuracy"] + float(gate["a_margin"]),
            "B_statistics_dominant": results["B"]["tabicl_only"]["mean_accuracy"] >= results["B"]["llm_only"]["mean_accuracy"] + float(gate["b_margin"]),
            "C2_adaptively_recoverable": results["C2"]["adaptive_diagnostic_oracle"]["mean_accuracy"] >= float(gate["oracle_min"]),
            "C2_one_call_insufficient": results["C2"]["llm_to_tfm"]["mean_accuracy"] <= float(gate["one_call_max"]),
            "C2_three_nonadaptive_calls_insufficient": results["C2"]["llm_to_tfm_compute_matched"]["mean_accuracy"] <= float(gate["three_call_max"]),
            "C2_all_strong_baselines_insufficient": max(results["C2"][method]["mean_accuracy"] for method in baseline_methods) <= float(gate["all_baseline_max"]),
        }
        summary["adaptive_gate_checks"] = checks
        summary["adaptive_gate_passed"] = all(checks.values())
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
