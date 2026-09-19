from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import accuracy_score

from .io import atomic_json, canonical_digest, sha256_file


def _result(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64); se = float(array.std(ddof=1) / np.sqrt(len(array)))
    return {"mean_accuracy": float(array.mean()), "std_accuracy": float(array.std(ddof=1)),
            "ci95_t": [float(array.mean() - 2.776 * se), float(array.mean() + 2.776 * se)],
            "seed_accuracies": array.tolist()}


def _paired(left: dict, right: dict) -> dict:
    delta = np.asarray(left["seed_accuracies"]) - np.asarray(right["seed_accuracies"])
    se = float(delta.std(ddof=1) / np.sqrt(len(delta)))
    return {"mean_delta": float(delta.mean()), "ci95_t": [float(delta.mean() - 2.776 * se),
                                                            float(delta.mean() + 2.776 * se)],
            "seed_deltas": delta.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--wheel", required=True); args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); output = Path(args.output)
    methods, regimes = tuple(config["experiment"]["methods"]), tuple(config["experiment"]["regimes"])
    seeds, world_size = tuple(map(int, config["experiment"]["test_seeds"])), int(config["runtime"]["expected_gpus"])
    expected = {f"phase51-{method}-{regime}-seed{seed}-rank{rank}" for method in methods for regime in regimes
                for seed in seeds for rank in range(world_size)}
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((output / "tasks").glob("*.json"))]
    actual = {record["experiment_id"] for record in records}
    if actual != expected or len(actual) != len(records):
        raise RuntimeError(f"Task grid mismatch missing={sorted(expected-actual)} unexpected={sorted(actual-expected)}")
    digest, wheel_sha = canonical_digest(config), sha256_file(Path(args.wheel)); arrays_by_key = {}
    required = {"probability", "label", "episode_id", "round_attention", "round_probability", "round_entropy",
                "round_message_applied", "relevant_view", "route_code", "residual_gate", "code_posterior"}
    for record in records:
        if (record["status"] != "complete" or record["config_digest"] != digest or record["wheel_sha256"] != wheel_sha
                or record["artifact_schema_version"] != "5.1.0"):
            raise RuntimeError(f"Invalid provenance for {record['experiment_id']}")
        prediction = output / record["prediction_file"]
        if sha256_file(prediction) != record["prediction_sha256"]:
            raise RuntimeError(f"Prediction checksum mismatch for {record['experiment_id']}")
        arrays = np.load(prediction, allow_pickle=False)
        if not required.issubset(arrays.files) or len(arrays["probability"]) != record["n_predictions"]:
            raise RuntimeError(f"Invalid arrays for {record['experiment_id']}")
        if not all(np.isfinite(arrays[name]).all() for name in ("probability", "round_attention", "code_posterior")):
            raise RuntimeError(f"Non-finite values for {record['experiment_id']}")
        arrays_by_key[(record["condition"], record["method"], int(record["seed"]), int(record["rank"]))] = arrays
    workers = [json.loads((output / f"worker_{rank}_summary.json").read_text(encoding="utf-8")) for rank in range(world_size)]
    if any(worker["status"] != "complete" or worker["config_digest"] != digest for worker in workers):
        raise RuntimeError("Worker summary validation failed")
    for model_name in workers[0]["state_digests"]:
        if len({worker["state_digests"][model_name] for worker in workers}) != 1:
            raise RuntimeError(f"Replicated training diverged for {model_name}")

    exact_zero_identity, exact_depth_saturation = True, True
    results = {regime: {} for regime in regimes}
    for regime in regimes:
        for method in methods:
            values = []
            for seed in seeds:
                shards = [arrays_by_key[(regime, method, seed, rank)] for rank in range(world_size)]
                probability = np.concatenate([array["probability"] for array in shards])
                labels = np.concatenate([array["label"] for array in shards])
                values.append(float(accuracy_score(labels, probability >= 0.5)))
            results[regime][method] = _result(values)
        for seed in seeds:
            for rank in range(world_size):
                r1 = arrays_by_key[(regime, "structured_soft_r1", seed, rank)]["probability"]
                zero = arrays_by_key[(regime, "structured_r2_zero_t2l", seed, rank)]["probability"]
                r2 = arrays_by_key[(regime, "structured_soft_r2", seed, rank)]["probability"]
                r3 = arrays_by_key[(regime, "structured_soft_r3", seed, rank)]["probability"]
                exact_zero_identity &= np.array_equal(r1, zero)
                exact_depth_saturation &= np.array_equal(r2, r3)
    if not exact_zero_identity or not exact_depth_saturation:
        raise RuntimeError(f"Exact protocol invariant failed: zero={exact_zero_identity} depth={exact_depth_saturation}")

    c2, thresholds = results["C2"], config["gate"]
    comparisons = {
        "soft_r2_minus_r1": _paired(c2["structured_soft_r2"], c2["structured_soft_r1"]),
        "soft_r2_minus_shuffle": _paired(c2["structured_soft_r2"], c2["structured_r2_shuffle_t2l"]),
        "soft_r2_minus_uniform": _paired(c2["structured_soft_r2"], c2["structured_r2_uniform_posterior"]),
        "soft_r2_minus_random": _paired(c2["structured_soft_r2"], c2["structured_r2_random_bridge"]),
        "soft_r2_minus_hard": _paired(c2["structured_soft_r2"], c2["structured_r2_hard_posterior"]),
        "analytic_minus_soft_r2": _paired(c2["analytic_router"], c2["structured_soft_r2"]),
    }
    primary = "structured_soft_r2"
    checks = {
        "exact_zero_message_identity": exact_zero_identity,
        "exact_no_new_information_saturation": exact_depth_saturation,
        "A_exact_preservation": results["A"][primary]["mean_accuracy"] >= float(thresholds["a_accuracy_min"]),
        "B_retention": results["B"][primary]["mean_accuracy"] >= float(thresholds["b_accuracy_min"]),
        "structured_message_target": c2[primary]["mean_accuracy"] >= float(thresholds["message_accuracy_min"]),
        "two_beats_over_one": comparisons["soft_r2_minus_r1"]["mean_delta"] >= float(thresholds["recurrence_margin"]),
        "shuffle_message_drop": comparisons["soft_r2_minus_shuffle"]["mean_delta"] >= float(thresholds["ablation_margin"]),
        "uniform_posterior_drop": comparisons["soft_r2_minus_uniform"]["mean_delta"] >= float(thresholds["ablation_margin"]),
        "trained_beats_random": comparisons["soft_r2_minus_random"]["mean_delta"] >= float(thresholds["ablation_margin"]),
        "near_analytical_ceiling": comparisons["analytic_minus_soft_r2"]["mean_delta"] <= float(thresholds["analytic_gap_max"]),
        "soft_noninferior_to_hard": comparisons["soft_r2_minus_hard"]["mean_delta"] >= -float(thresholds["soft_noninferiority_tolerance"]),
    }
    representation = {}
    for method in methods:
        route_matches, entropy = [], []
        for seed in seeds:
            for rank in range(world_size):
                arrays = arrays_by_key[("C2", method, seed, rank)]
                attention = arrays["round_attention"]
                route_matches.extend((attention[:, -1].mean(axis=1).argmax(-1) == arrays["relevant_view"]).astype(float))
                entropy.extend(arrays["round_entropy"][:, -1].reshape(-1).tolist())
        representation[method] = {"final_route_attention_accuracy": float(np.mean(route_matches)),
                                  "final_attention_entropy": float(np.mean(entropy))}
    fields = ["experiment_id", "protocol_id", "git_commit", "seed", "rank", "method", "condition", "rounds",
              "message_mode", "route_view_prior_access", "score", "roc_auc", "log_loss", "trainable_params",
              "config_digest", "wheel_sha256"]
    with (output / "experiments.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(records)
    summary = {
        "status": "complete", "protocol_id": config["experiment"]["protocol_id"],
        "classification": config["experiment"]["classification"], "config_digest": digest, "wheel_sha256": wheel_sha,
        "expected_tasks": len(expected), "validated_tasks": len(records), "results": results,
        "paired_comparisons": comparisons, "representation_diagnostics": representation,
        "protocol_invariants": {"exact_zero_message_identity": exact_zero_identity,
                                "exact_no_new_information_saturation": exact_depth_saturation,
                                "learned_methods_route_view_prior_access": False},
        "phase51_checks": checks, "structured_communication_supported": all(checks.values()),
        "trainable_params": max(worker["trainable_params_max"] for worker in workers),
        "cache_bytes_per_worker": [worker["cache_bytes"] for worker in workers],
        "backbone_peak_gpu_memory_bytes": max(worker["backbone_peak_gpu_memory_bytes"] for worker in workers),
        "peak_gpu_memory_bytes": max(worker["peak_gpu_memory_bytes"] for worker in workers),
        "worker_elapsed_seconds": [worker["elapsed_seconds"] for worker in workers],
        "training": [worker["training"] for worker in workers],
        "scientific_scope": ("Exploratory cached-response structured-message audit. Learned methods receive the soft "
                             "16-state posterior and codebook sentence embeddings, never the analytical route-view prior."),
    }
    atomic_json(output / "summary.json", summary); print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
