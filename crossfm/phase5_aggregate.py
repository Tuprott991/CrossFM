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
    array = np.asarray(values, dtype=np.float64)
    standard_error = float(array.std(ddof=1) / np.sqrt(len(array)))
    return {"mean_accuracy": float(array.mean()), "std_accuracy": float(array.std(ddof=1)),
            "ci95_t": [float(array.mean() - 2.776 * standard_error), float(array.mean() + 2.776 * standard_error)],
            "seed_accuracies": array.tolist()}


def _paired(left: dict, right: dict) -> dict:
    delta = np.asarray(left["seed_accuracies"]) - np.asarray(right["seed_accuracies"])
    standard_error = float(delta.std(ddof=1) / np.sqrt(len(delta)))
    return {"mean_delta": float(delta.mean()), "ci95_t": [float(delta.mean() - 2.776 * standard_error),
                                                            float(delta.mean() + 2.776 * standard_error)],
            "seed_deltas": delta.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--wheel", required=True); args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); output = Path(args.output)
    methods, regimes = tuple(config["experiment"]["methods"]), tuple(config["experiment"]["regimes"])
    seeds, world_size = tuple(map(int, config["experiment"]["test_seeds"])), int(config["runtime"]["expected_gpus"])
    expected = {f"phase5-{method}-{regime}-seed{seed}-rank{rank}" for method in methods for regime in regimes
                for seed in seeds for rank in range(world_size)}
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((output / "tasks").glob("*.json"))]
    actual = {record["experiment_id"] for record in records}
    if actual != expected or len(actual) != len(records):
        raise RuntimeError(f"Task grid mismatch missing={sorted(expected-actual)} unexpected={sorted(actual-expected)}")
    digest, wheel_sha = canonical_digest(config), sha256_file(Path(args.wheel))
    arrays_by_key = {}
    required = {"probability", "label", "episode_id", "round_attention", "round_probability", "round_entropy",
                "round_state_delta_norm", "relevant_view", "route_code", "residual_gate"}
    for record in records:
        if (record["status"] != "complete" or record["config_digest"] != digest or
                record["wheel_sha256"] != wheel_sha or record["artifact_schema_version"] != "5.0.0"):
            raise RuntimeError(f"Invalid provenance for {record['experiment_id']}")
        prediction = output / record["prediction_file"]
        if sha256_file(prediction) != record["prediction_sha256"]:
            raise RuntimeError(f"Prediction checksum mismatch for {record['experiment_id']}")
        arrays = np.load(prediction, allow_pickle=False)
        if not required.issubset(arrays.files) or len(arrays["probability"]) != record["n_predictions"]:
            raise RuntimeError(f"Invalid arrays for {record['experiment_id']}")
        if not all(np.isfinite(arrays[name]).all() for name in ("probability", "round_attention", "round_probability")):
            raise RuntimeError(f"Non-finite values for {record['experiment_id']}")
        arrays_by_key[(record["condition"], record["method"], int(record["seed"]), int(record["rank"]))] = arrays
    workers = [json.loads((output / f"worker_{rank}_summary.json").read_text(encoding="utf-8")) for rank in range(world_size)]
    if any(worker["status"] != "complete" or worker["config_digest"] != digest for worker in workers):
        raise RuntimeError("Worker summary validation failed")
    for model_name in workers[0]["state_digests"]:
        if len({worker["state_digests"][model_name] for worker in workers}) != 1:
            raise RuntimeError(f"Replicated training diverged for {model_name}")

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

    c2, thresholds = results["C2"], config["gate"]
    comparisons = {
        "message_r2_minus_r1": _paired(c2["message_only_r2"], c2["message_only_r1"]),
        "message_r3_minus_r2": _paired(c2["message_only_r3"], c2["message_only_r2"]),
        "message_r2_minus_zero_t2l": _paired(c2["message_only_r2"], c2["message_only_r2_zero_t2l"]),
        "message_r2_minus_shuffle_t2l": _paired(c2["message_only_r2"], c2["message_only_r2_shuffle_t2l"]),
        "message_r2_minus_zero_l2t": _paired(c2["message_only_r2"], c2["message_only_r2_zero_l2t"]),
        "message_r2_minus_l2t_only": _paired(c2["message_only_r2"], c2["l2t_only_compute_matched"]),
        "message_r2_minus_random": _paired(c2["message_only_r2"], c2["random_bridge_r2"]),
        "analytic_minus_message_r2": _paired(c2["analytic_router"], c2["message_only_r2"]),
        "prior_r2_minus_r1": _paired(c2["prior_start_r2"], c2["prior_start_r1"]),
        "prior_r3_minus_r1": _paired(c2["prior_start_r3"], c2["prior_start_r1"]),
    }
    shortcut_checks = {
        "analytic_router_strong": c2["analytic_router"]["mean_accuracy"] >= float(thresholds["shortcut_accuracy_min"]),
        "prior_available_r1_strong": c2["prior_start_r1"]["mean_accuracy"] >= float(thresholds["shortcut_accuracy_min"]),
        "prior_depth_invariant_r2": abs(comparisons["prior_r2_minus_r1"]["mean_delta"]) <= float(thresholds["depth_equivalence_tolerance"]),
        "prior_depth_invariant_r3": abs(comparisons["prior_r3_minus_r1"]["mean_delta"]) <= float(thresholds["depth_equivalence_tolerance"]),
    }
    primary = "message_only_r2"
    communication_checks = {
        "A_exact_preservation": results["A"][primary]["mean_accuracy"] >= float(thresholds["a_accuracy_min"]),
        "B_retention": results["B"][primary]["mean_accuracy"] >= float(thresholds["b_accuracy_min"]),
        "message_r2_target": c2[primary]["mean_accuracy"] >= float(thresholds["message_accuracy_min"]),
        "two_beats_over_one": comparisons["message_r2_minus_r1"]["mean_delta"] >= float(thresholds["recurrence_margin"]),
        "zero_t2l_drop": comparisons["message_r2_minus_zero_t2l"]["mean_delta"] >= float(thresholds["ablation_margin"]),
        "shuffle_t2l_drop": comparisons["message_r2_minus_shuffle_t2l"]["mean_delta"] >= float(thresholds["ablation_margin"]),
        "zero_l2t_drop": comparisons["message_r2_minus_zero_l2t"]["mean_delta"] >= float(thresholds["ablation_margin"]),
        "beats_l2t_only": comparisons["message_r2_minus_l2t_only"]["mean_delta"] >= float(thresholds["one_way_margin"]),
        "beats_random_bridge": comparisons["message_r2_minus_random"]["mean_delta"] >= float(thresholds["ablation_margin"]),
    }
    representation = {}
    for method in methods:
        entropies, route_matches, deltas = [], [], []
        for seed in seeds:
            for rank in range(world_size):
                arrays = arrays_by_key[("C2", method, seed, rank)]
                attention = arrays["round_attention"]
                relevant = arrays["relevant_view"]
                route_matches.extend((attention[:, -1].mean(axis=1).argmax(-1) == relevant).astype(float).tolist())
                entropies.extend(arrays["round_entropy"].mean(axis=(0, 2)).tolist())
                deltas.extend(arrays["round_state_delta_norm"].mean(axis=(0, 2)).tolist())
        representation[method] = {"final_route_attention_accuracy": float(np.mean(route_matches)),
                                  "mean_round_entropy_observations": entropies,
                                  "mean_round_state_delta_norm_observations": deltas}
    fields = ["experiment_id", "protocol_id", "git_commit", "seed", "rank", "method", "condition", "rounds",
              "message_mode", "direct_analytical_prior", "score", "roc_auc", "log_loss", "trainable_params",
              "conceptual_specialist_calls", "incremental_backbone_calls", "config_digest", "wheel_sha256"]
    with (output / "experiments.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(records)
    summary = {
        "status": "complete", "protocol_id": config["experiment"]["protocol_id"],
        "classification": config["experiment"]["classification"], "config_digest": digest, "wheel_sha256": wheel_sha,
        "expected_tasks": len(expected), "validated_tasks": len(records), "results": results,
        "paired_comparisons": comparisons, "representation_diagnostics": representation,
        "shortcut_checks": shortcut_checks, "shortcut_hypothesis_supported": all(shortcut_checks.values()),
        "communication_checks": communication_checks,
        "learned_communication_hypothesis_supported": all(communication_checks.values()),
        "trainable_params": max(worker["trainable_params_max"] for worker in workers),
        "cache_bytes_per_worker": [worker["cache_bytes"] for worker in workers],
        "backbone_peak_gpu_memory_bytes": max(worker["backbone_peak_gpu_memory_bytes"] for worker in workers),
        "peak_gpu_memory_bytes": max(worker["peak_gpu_memory_bytes"] for worker in workers),
        "worker_elapsed_seconds": [worker["elapsed_seconds"] for worker in workers], "training": [worker["training"] for worker in workers],
        "scientific_scope": ("Exploratory cached-response shortcut audit. Analytical-prior and message-only conclusions "
                             "are reported separately; this is not evidence for online differentiable TFM conditioning or real-data transfer."),
    }
    atomic_json(output / "summary.json", summary); print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
