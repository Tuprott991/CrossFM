from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import accuracy_score

from .io import atomic_json, canonical_digest, sha256_file


def _mean(values) -> float:
    return float(np.asarray(values, dtype=np.float64).mean())


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
    seeds = tuple(int(seed) for seed in config["experiment"]["test_seeds"])
    world_size = int(config["runtime"]["expected_gpus"])
    expected = {
        f"phase4-{method}-{regime}-seed{seed}-rank{rank}"
        for method in methods for regime in regimes for seed in seeds for rank in range(world_size)
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
        required = {
            "probability", "label", "episode_id", "round_attention", "round_probability",
            "round_entropy", "round_update_gate", "relevant_view", "route_code",
        }
        if not required.issubset(arrays.files):
            raise RuntimeError(f"Missing trace arrays for {record['experiment_id']}")
        if len(arrays["probability"]) != record["n_predictions"] or not np.isfinite(arrays["probability"]).all():
            raise RuntimeError(f"Invalid predictions for {record['experiment_id']}")

    workers = [json.loads((output / f"worker_{rank}_summary.json").read_text(encoding="utf-8")) for rank in range(world_size)]
    if any(worker["status"] != "complete" or worker["config_digest"] != digest for worker in workers):
        raise RuntimeError("Worker summary validation failed")
    for model_name in workers[0]["state_digests"]:
        if len({worker["state_digests"][model_name] for worker in workers}) != 1:
            raise RuntimeError(f"Replicated training diverged for {model_name}")

    seed_scores = {regime: {method: [] for method in methods} for regime in regimes}
    traces_by_key = {}
    for regime in regimes:
        for method in methods:
            for seed in seeds:
                shards = sorted(
                    (record for record in records if record["condition"] == regime and record["method"] == method and int(record["seed"]) == seed),
                    key=lambda record: record["rank"],
                )
                arrays = [np.load(output / record["prediction_file"], allow_pickle=False) for record in shards]
                probability = np.concatenate([array["probability"] for array in arrays])
                labels = np.concatenate([array["label"] for array in arrays])
                seed_scores[regime][method].append(float(accuracy_score(labels, probability >= 0.5)))
                traces_by_key[(regime, method, seed)] = arrays

    results = {}
    for regime in regimes:
        results[regime] = {}
        for method in methods:
            values = np.asarray(seed_scores[regime][method], dtype=np.float64)
            standard_error = float(values.std(ddof=1) / np.sqrt(len(values)))
            results[regime][method] = {
                "mean_accuracy": float(values.mean()),
                "std_accuracy": float(values.std(ddof=1)),
                "ci95_t": [
                    float(values.mean() - 2.776 * standard_error),
                    float(values.mean() + 2.776 * standard_error),
                ],
                "seed_accuracies": values.tolist(),
            }

    primary = "crossfm_r3_corrective"
    representation = {"C2": {"primary_method": primary}}
    route_accuracy, entropy_by_round, confidence_by_round, correction_gates = [], [[], [], []], [[], [], []], []
    for seed in seeds:
        for arrays in traces_by_key[("C2", primary, seed)]:
            attention = arrays["round_attention"]
            round_probability = arrays["round_probability"]
            entropy = arrays["round_entropy"]
            gates = arrays["round_update_gate"]
            relevant = arrays["relevant_view"]
            final_attention = attention[:, -1].mean(axis=1).argmax(axis=-1)
            route_accuracy.extend((final_attention == relevant).astype(np.float64).tolist())
            for depth in range(3):
                entropy_by_round[depth].extend(entropy[:, depth].reshape(-1).tolist())
                confidence_by_round[depth].extend(np.abs(round_probability[:, depth].reshape(-1) - 0.5).tolist())
            correction_gates.extend(gates[:, 2].reshape(-1).tolist())
    representation["C2"].update({
        "round3_route_attention_accuracy": _mean(route_accuracy),
        "attention_entropy_by_round": [_mean(values) for values in entropy_by_round],
        "prediction_confidence_by_round": [_mean(values) for values in confidence_by_round],
        "round3_corrective_gate_mean": _mean(correction_gates),
        "round3_corrective_gate_std": float(np.std(correction_gates, ddof=1)),
    })

    c2 = results["C2"]
    r3 = c2[primary]["mean_accuracy"]
    gates = config["gate"]
    checks = {
        "A_exact_preservation": results["A"][primary]["mean_accuracy"] >= float(gates["a_accuracy_min"]),
        "B_retention": results["B"][primary]["mean_accuracy"] >= float(gates["b_accuracy_min"]),
        "C2_target": r3 >= float(gates["c2_accuracy_min"]),
        "R2_beats_R1": c2["crossfm_r2_shared"]["mean_accuracy"] >= c2["crossfm_r1_shared"]["mean_accuracy"] + float(gates["r2_margin_over_r1"]),
        "R3_beats_R2": r3 >= c2["crossfm_r2_shared"]["mean_accuracy"] + float(gates["r3_margin_over_r2"]),
        "all_seed_recurrence": all(
            later >= earlier + float(gates["per_seed_r3_margin_over_r1"])
            for later, earlier in zip(c2[primary]["seed_accuracies"], c2["crossfm_r1_shared"]["seed_accuracies"])
        ),
        "zero_evidence_drop": r3 >= c2["crossfm_r3_zero_t2l"]["mean_accuracy"] + float(gates["ablation_margin"]),
        "shuffle_evidence_drop": r3 >= c2["crossfm_r3_shuffle_t2l"]["mean_accuracy"] + float(gates["ablation_margin"]),
        "random_bridge_drop": r3 >= c2["crossfm_r3_random_bridge"]["mean_accuracy"] + float(gates["ablation_margin"]),
        "bidirectional_beats_one_way": r3 >= max(
            c2["crossfm_r3_t2l_only"]["mean_accuracy"],
            c2["crossfm_r3_l2t_only_compute_matched"]["mean_accuracy"],
        ) + float(gates["one_way_margin"]),
        "bidirectional_beats_stopgrad": r3 >= max(
            c2["crossfm_r3_stopgrad_l2t"]["mean_accuracy"],
            c2["crossfm_r3_stopgrad_t2l"]["mean_accuracy"],
        ) + float(gates["stopgrad_margin"]),
        "corrective_gate_nonconstant": representation["C2"]["round3_corrective_gate_std"] >= float(gates["corrective_gate_std_min"]),
        "messages_change_training_predictions": min(
            worker["training"]["primary"]["train_zero_message_mean_absolute_delta"] for worker in workers
        ) >= float(gates["message_delta_min"]),
    }

    fields = [
        "experiment_id", "protocol_id", "git_commit", "seed", "rank", "method", "condition", "rounds",
        "message_mode", "adapter_model", "score", "roc_auc", "log_loss", "trainable_params",
        "conceptual_specialist_calls", "incremental_backbone_calls", "shared_cached_specialist_view_calls",
        "cache_bytes_worker", "peak_gpu_memory_bytes_worker", "config_digest", "wheel_sha256",
    ]
    with (output / "experiments.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(records)

    summary = {
        "status": "complete", "protocol_id": config["experiment"]["protocol_id"],
        "classification": config["experiment"]["classification"], "config_digest": digest,
        "wheel_sha256": wheel_sha, "expected_tasks": len(expected), "validated_tasks": len(records),
        "results": results, "representation_diagnostics": representation,
        "phase4_checks": checks, "phase4_passed": all(checks.values()),
        "trainable_params": max(worker["trainable_params_max"] for worker in workers),
        "cache_bytes_per_worker": [worker["cache_bytes"] for worker in workers],
        "backbone_peak_gpu_memory_bytes": max(worker["backbone_peak_gpu_memory_bytes"] for worker in workers),
        "peak_gpu_memory_bytes": max(worker["peak_gpu_memory_bytes"] for worker in workers),
        "worker_elapsed_seconds": [worker["elapsed_seconds"] for worker in workers],
        "training": [worker["training"] for worker in workers],
        "train_cache_diagnostics": [worker["train_cache_diagnostics"] for worker in workers],
        "validation_cache_diagnostics": [worker["validation_cache_diagnostics"] for worker in workers],
        "scientific_scope": (
            "Exploratory cached-response CrossFM causal/depth experiment with a shared frozen-backbone cache; "
            "not evidence for online differentiable TFM conditioning or real-data transfer."
        ),
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
