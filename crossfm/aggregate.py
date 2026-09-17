from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from .io import atomic_json, canonical_digest, sha256_file
from .worker import task_grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wheel", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.output)
    expected = {f"phase1-{t['method']}-{t['regime']}-seed{t['seed']}" for t in task_grid(config)}
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((output / "tasks").glob("*.json"))]
    actual = {record["experiment_id"] for record in records}
    if actual != expected:
        raise RuntimeError(f"Task grid mismatch missing={sorted(expected-actual)} unexpected={sorted(actual-expected)}")
    digest = canonical_digest(config)
    wheel_sha = sha256_file(Path(args.wheel))
    for record in records:
        if record["config_digest"] != digest or record["wheel_sha256"] != wheel_sha or record["status"] != "complete":
            raise RuntimeError(f"Invalid provenance for {record['experiment_id']}")
        pred = output / record["prediction_file"]
        if sha256_file(pred) != record["prediction_sha256"]:
            raise RuntimeError(f"Prediction checksum mismatch for {record['experiment_id']}")
        if not all(np.isfinite(record[key]) for key in ("score", "roc_auc", "log_loss")):
            raise RuntimeError(f"Non-finite metric for {record['experiment_id']}")

    csv_path = output / "experiments.csv"
    fields = ["experiment_id", "git_commit", "seed", "method", "condition", "score", "roc_auc", "log_loss", "trainable_params", "inference_calls", "elapsed_seconds", "config_digest", "wheel_sha256"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(records)
    means = {}
    for regime in config["experiment"]["regimes"]:
        means[regime] = {}
        for method in config["experiment"]["methods"]:
            values = [r["score"] for r in records if r["condition"] == regime and r["method"] == method]
            means[regime][method] = {"mean_accuracy": float(np.mean(values)), "std_accuracy": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}
    margins = config["gate"]["margins"]
    checks = {
        "A_llm_beats_specialist": means["A"]["llm_only"]["mean_accuracy"] >= means["A"]["tabicl_only"]["mean_accuracy"] + margins["A"],
        "B_specialist_beats_llm": means["B"]["tabicl_only"]["mean_accuracy"] >= means["B"]["llm_only"]["mean_accuracy"] + margins["B"],
        "C_individuals_weak": max(means["C"]["llm_only"]["mean_accuracy"], means["C"]["tabicl_only"]["mean_accuracy"]) <= config["gate"]["c_individual_max"],
        "C_complementary_information_recoverable": means["C"]["semantic_statistical_oracle"]["mean_accuracy"] >= config["gate"]["c_oracle_min"] and means["C"]["semantic_statistical_oracle"]["mean_accuracy"] >= max(means["C"]["llm_only"]["mean_accuracy"], means["C"]["tabicl_only"]["mean_accuracy"]) + margins["C"],
    }
    summary = {
        "status": "complete", "protocol_id": config["experiment"]["protocol_id"],
        "config_digest": digest, "wheel_sha256": wheel_sha,
        "expected_tasks": len(expected), "validated_tasks": len(records),
        "results": means, "gate_checks": checks, "phase1_gate_passed": all(checks.values()),
        "interpretation": "Proceed to Phase 2 only if phase1_gate_passed is true. The oracle is diagnostic, not a learned CrossFM result.",
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

