from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from .io import atomic_json, canonical_digest, sha256_file


def _result(accuracy: list[float], losses: list[float], aucs: list[float]) -> dict:
    values = np.asarray(accuracy, dtype=np.float64)
    se = float(values.std(ddof=1) / np.sqrt(len(values)))
    return {
        "mean_accuracy": float(values.mean()),
        "std_accuracy": float(values.std(ddof=1)),
        "ci95_t": [float(values.mean() - 2.776 * se), float(values.mean() + 2.776 * se)],
        "seed_accuracies": values.tolist(),
        "mean_log_loss": float(np.mean(losses)),
        "seed_log_losses": list(map(float, losses)),
        "mean_roc_auc": float(np.mean(aucs)),
        "seed_roc_aucs": list(map(float, aucs)),
    }


def _paired(left: dict, right: dict, field: str) -> dict:
    delta = np.asarray(left[field], dtype=np.float64) - np.asarray(right[field], dtype=np.float64)
    se = float(delta.std(ddof=1) / np.sqrt(len(delta)))
    return {
        "mean_delta": float(delta.mean()),
        "ci95_t": [float(delta.mean() - 2.776 * se), float(delta.mean() + 2.776 * se)],
        "seed_deltas": delta.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wheel", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.output)
    methods = tuple(config["experiment"]["methods"])
    regimes = tuple(config["experiment"]["regimes"])
    seeds = tuple(map(int, config["experiment"]["test_seeds"]))
    world_size = int(config["runtime"]["expected_gpus"])
    expected = {
        f"phase52-{method}-{regime}-seed{seed}-rank{rank}"
        for method in methods for regime in regimes for seed in seeds for rank in range(world_size)
    }
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output / "tasks").glob("*.json"))
    ]
    actual = {record["experiment_id"] for record in records}
    if actual != expected or len(actual) != len(records):
        raise RuntimeError(
            f"Task grid mismatch missing={sorted(expected-actual)} unexpected={sorted(actual-expected)}"
        )
    digest, wheel_sha = canonical_digest(config), sha256_file(Path(args.wheel))
    required = {
        "probability", "label", "episode_id", "residual_gate", "relevant_view",
        "route_code", "code_posterior", "attention", "selected_probability",
        "attention_entropy", "logit_correction",
    }
    arrays_by_key = {}
    for record in records:
        if (
            record["status"] != "complete"
            or record["config_digest"] != digest
            or record["wheel_sha256"] != wheel_sha
            or record["artifact_schema_version"] != "5.2.0"
        ):
            raise RuntimeError(f"Invalid provenance for {record['experiment_id']}")
        prediction = output / record["prediction_file"]
        if sha256_file(prediction) != record["prediction_sha256"]:
            raise RuntimeError(f"Prediction checksum mismatch for {record['experiment_id']}")
        arrays = np.load(prediction, allow_pickle=False)
        if not required.issubset(arrays.files) or len(arrays["probability"]) != record["n_predictions"]:
            raise RuntimeError(f"Invalid arrays for {record['experiment_id']}")
        if not all(
            np.isfinite(arrays[name]).all()
            for name in ("probability", "attention", "selected_probability", "code_posterior")
        ):
            raise RuntimeError(f"Non-finite values for {record['experiment_id']}")
        arrays_by_key[
            (record["condition"], record["method"], int(record["seed"]), int(record["rank"]))
        ] = arrays
    workers = [
        json.loads((output / f"worker_{rank}_summary.json").read_text(encoding="utf-8"))
        for rank in range(world_size)
    ]
    if any(worker["status"] != "complete" or worker["config_digest"] != digest for worker in workers):
        raise RuntimeError("Worker summary validation failed")
    for model_name in workers[0]["state_digests"]:
        if len({worker["state_digests"][model_name] for worker in workers}) != 1:
            raise RuntimeError(f"Replicated training diverged for {model_name}")

    results = {regime: {} for regime in regimes}
    exact_preservation = {"A": True, "B": True}
    for regime in regimes:
        for method in methods:
            accuracies, losses, aucs = [], [], []
            for seed in seeds:
                shards = [arrays_by_key[(regime, method, seed, rank)] for rank in range(world_size)]
                probability = np.concatenate([array["probability"] for array in shards])
                labels = np.concatenate([array["label"] for array in shards])
                accuracies.append(float(accuracy_score(labels, probability >= 0.5)))
                losses.append(float(log_loss(labels, probability, labels=[0, 1])))
                aucs.append(float(roc_auc_score(labels, probability)) if len(np.unique(labels)) == 2 else 0.5)
            results[regime][method] = _result(accuracies, losses, aucs)
        if regime in exact_preservation:
            for seed in seeds:
                for rank in range(world_size):
                    reference = arrays_by_key[(regime, "analytic_router", seed, rank)]["probability"]
                    primary = arrays_by_key[(regime, "arplus_full", seed, rank)]["probability"]
                    exact_preservation[regime] &= np.array_equal(reference, primary)

    c2 = results["C2"]
    comparisons = {
        "arplus_minus_analytic_accuracy": _paired(
            c2["arplus_full"], c2["analytic_router"], "seed_accuracies",
        ),
        "arplus_minus_analytic_log_loss": _paired(
            c2["arplus_full"], c2["analytic_router"], "seed_log_losses",
        ),
        "arplus_minus_temperature_accuracy": _paired(
            c2["arplus_full"], c2["arplus_temperature"], "seed_accuracies",
        ),
        "arplus_minus_no_anchor_accuracy": _paired(
            c2["arplus_full"], c2["arplus_no_anchor"], "seed_accuracies",
        ),
        "arplus_minus_shuffle_accuracy": _paired(
            c2["arplus_full"], c2["arplus_shuffle_response"], "seed_accuracies",
        ),
        "arplus_minus_smr_accuracy": _paired(
            c2["arplus_full"], c2["structured_smr_r2"], "seed_accuracies",
        ),
        "oracle_minus_arplus_accuracy": _paired(
            c2["response_bank_oracle"], c2["arplus_full"], "seed_accuracies",
        ),
    }
    representation = {}
    for method in methods:
        route_matches, entropies, corrections = [], [], []
        for seed in seeds:
            for rank in range(world_size):
                arrays = arrays_by_key[("C2", method, seed, rank)]
                attention = arrays["attention"]
                route_matches.extend(
                    (attention.mean(axis=1).argmax(-1) == arrays["relevant_view"]).astype(float)
                )
                entropies.extend(arrays["attention_entropy"].reshape(-1).tolist())
                corrections.extend(np.abs(arrays["logit_correction"]).reshape(-1).tolist())
        representation[method] = {
            "route_attention_accuracy": float(np.mean(route_matches)),
            "attention_entropy": float(np.mean(entropies)),
            "mean_absolute_logit_correction": float(np.mean(corrections)),
        }

    thresholds = config["gate"]
    accuracy_gain = comparisons["arplus_minus_analytic_accuracy"]["mean_delta"]
    logloss_gain = -comparisons["arplus_minus_analytic_log_loss"]["mean_delta"]
    checks = {
        "A_exact_preservation": exact_preservation["A"],
        "B_exact_preservation": exact_preservation["B"],
        "A_accuracy": results["A"]["arplus_full"]["mean_accuracy"] >= float(thresholds["a_accuracy_min"]),
        "B_accuracy": results["B"]["arplus_full"]["mean_accuracy"] >= float(thresholds["b_accuracy_min"]),
        "ARplus_accuracy_or_logloss_gain": (
            accuracy_gain >= float(thresholds["accuracy_gain_min"])
            or logloss_gain >= float(thresholds["logloss_gain_min"])
        ),
        "analytical_anchor_necessary": (
            comparisons["arplus_minus_no_anchor_accuracy"]["mean_delta"]
            >= float(thresholds["anchor_margin"])
        ),
        "response_features_causal": (
            comparisons["arplus_minus_shuffle_accuracy"]["mean_delta"]
            >= float(thresholds["response_margin"])
        ),
        "ARplus_beats_SMR": (
            comparisons["arplus_minus_smr_accuracy"]["mean_delta"]
            >= float(thresholds["smr_margin"])
        ),
        "route_preserved": (
            representation["arplus_full"]["route_attention_accuracy"]
            >= float(thresholds["route_accuracy_min"])
        ),
        "within_response_bank_ceiling": (
            comparisons["oracle_minus_arplus_accuracy"]["mean_delta"]
            <= float(thresholds["oracle_gap_max"])
        ),
    }
    fieldnames = sorted({key for record in records for key in record})
    with (output / "experiments.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "status": "complete",
        "protocol_id": config["experiment"]["protocol_id"],
        "classification": config["experiment"]["classification"],
        "validated_tasks": len(records),
        "expected_tasks": len(expected),
        "config_digest": digest,
        "wheel_sha256": wheel_sha,
        "results": results,
        "paired_comparisons": comparisons,
        "representation_diagnostics": representation,
        "exact_preservation": exact_preservation,
        "phase52_checks": checks,
        "phase52_gate_passed": all(checks.values()),
        "trainable_params": max(worker["trainable_params_max"] for worker in workers),
        "training": [worker["training"] for worker in workers],
        "cache_bytes_per_worker": [worker["cache_bytes"] for worker in workers],
        "backbone_peak_gpu_memory_bytes": max(worker["backbone_peak_gpu_memory_bytes"] for worker in workers),
        "peak_gpu_memory_bytes": max(worker["peak_gpu_memory_bytes"] for worker in workers),
        "worker_elapsed_seconds": [worker["elapsed_seconds"] for worker in workers],
        "scientific_scope": (
            "Exploratory AR+ audit. The analytical posterior-to-view map is immutable; "
            "learned parameters may only calibrate temperature and add regularized view-logit residuals. "
            "The response-bank oracle uses labels and is a ceiling diagnostic, never a model result."
        ),
    }
    atomic_json(output / "summary.json", summary)


if __name__ == "__main__":
    main()
