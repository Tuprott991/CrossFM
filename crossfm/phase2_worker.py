from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from .baselines import QwenBinaryBaseline, TabICLBaseline, adaptive_routing_oracle
from .io import atomic_json, canonical_digest, sha256_file
from .phase2 import (
    SoftPrefixAdapter, ViewScorer, attach_language_embeddings, collect_specialist_evidence,
    collect_view_evidence, tool_prompts, tune_ensemble_alpha,
)
from .synthetic import Episode, make_episodes, split_hash


DEFAULT_METHODS = (
    "llm_only", "tabicl_only", "prediction_ensemble", "llm_to_tfm",
    "tfm_to_llm", "textual_tool", "llm_to_tfm_compute_matched",
)


def _episodes(config: dict, split: str, seed: int) -> list[Episode]:
    spec = config["data"][split]
    return [
        ep
        for regime in config["experiment"]["regimes"]
        for ep in make_episodes(
            regime, seed, int(spec["episodes_per_regime"]), int(spec["queries_per_episode"]),
            alias_split=spec["alias_split"], mechanism_split=spec["mechanism_split"],
        )
    ]


def _flatten_labels(episodes: list[Episode]) -> np.ndarray:
    return np.concatenate([ep.y_query for ep in episodes])


def _flatten_ids(episodes: list[Episode]) -> np.ndarray:
    return np.concatenate([[ep.episode_id] * len(ep.y_query) for ep in episodes])


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _metrics(labels: np.ndarray, probability: np.ndarray) -> dict:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "score": float(accuracy_score(labels, probability >= 0.5)),
        "roc_auc": float(roc_auc_score(labels, probability)),
        "log_loss": float(log_loss(labels, probability, labels=[0, 1])),
    }


def _save_checkpoint(path: Path, payload: dict) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp.pt")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--world-size", default=2, type=int)
    parser.add_argument("--wheel", required=True)
    args = parser.parse_args()

    import torch

    config_path, output = Path(args.config), Path(args.output)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_digest, wheel_sha = canonical_digest(config), sha256_file(Path(args.wheel))
    seeds = [int(seed) for i, seed in enumerate(config["experiment"]["test_seeds"]) if i % args.world_size == args.rank]
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    llm_cfg, tfm_cfg = config["models"]["llm"], config["models"]["specialist"]
    llm = QwenBinaryBaseline(llm_cfg["id"], llm_cfg["revision"], batch_size=int(config["runtime"]["llm_batch_size"]))
    llm.hidden_state_preflight()
    tfm = TabICLBaseline(tfm_cfg["id"], tfm_cfg["revision"], tfm_cfg["checkpoint"])
    tfm.preflight()

    train = _episodes(config, "train", int(config["data"]["train"]["seed"]))
    validation = _episodes(config, "validation", int(config["data"]["validation"]["seed"]))
    print(json.dumps({"event": "collect_train_specialist", "rank": args.rank, "episodes": len(train)}), flush=True)
    train_specialist = collect_specialist_evidence(tfm, train)
    validation_specialist = collect_specialist_evidence(tfm, validation)

    print(json.dumps({"event": "collect_view_cache", "rank": args.rank}), flush=True)
    train_views = collect_view_evidence(tfm, train)
    validation_views = collect_view_evidence(tfm, validation)
    attach_language_embeddings(llm, train, train_views)
    attach_language_embeddings(llm, validation, validation_views)
    embedding_dim = next(iter(train_views.values())).task_embedding.shape[0]
    l2t = ViewScorer(embedding_dim, int(config["training"]["l2t_hidden_dim"]), "cuda:0", int(config["training"]["seed"]))
    l2t_training = l2t.fit(
        train_views, validation_views, int(config["training"]["l2t_epochs"]), float(config["training"]["l2t_learning_rate"]),
    )

    evidence_dim = next(iter(train_specialist.values())).representation.shape[1]
    t2l = SoftPrefixAdapter(
        llm, evidence_dim, int(config["training"]["t2l_bottleneck"]),
        int(config["training"]["prefix_tokens"]), int(config["training"]["seed"]),
    )
    total_trainable = l2t.trainable_params + t2l.trainable_params
    if total_trainable >= int(config["training"]["max_trainable_params"]):
        raise RuntimeError(f"Adapter budget exceeded: {total_trainable}")
    t2l_training = t2l.fit(
        train, train_specialist, validation, validation_specialist,
        int(config["training"]["t2l_epochs"]), int(config["training"]["t2l_batch_size"]),
        float(config["training"]["t2l_learning_rate"]),
    )

    checkpoint = output / "checkpoints" / f"phase2-adapters-rank{args.rank}.pt"
    _save_checkpoint(checkpoint, {
        "protocol_id": config["experiment"]["protocol_id"], "config_digest": config_digest,
        "l2t": l2t.module.state_dict(), "t2l_projector": t2l.projector.state_dict(), "t2l_head": t2l.head.state_dict(),
        "l2t_training": l2t_training, "t2l_training": t2l_training,
    })

    # Validation-only mixture tuning. Test outcomes are never consulted.
    validation_llm = llm.predict(validation).probabilities
    validation_tfm = np.concatenate([validation_specialist[ep.episode_id].probability for ep in validation])
    ensemble_alpha, ensemble_val_loss = tune_ensemble_alpha(_flatten_labels(validation), validation_llm, validation_tfm)

    records = []
    for seed in seeds:
        test = _episodes(config, "test", seed)
        print(json.dumps({"event": "test_seed", "rank": args.rank, "seed": seed, "episodes": len(test)}), flush=True)
        test_specialist = collect_specialist_evidence(tfm, test)
        test_views = collect_view_evidence(tfm, test)
        attach_language_embeddings(llm, test, test_views)
        for regime in config["experiment"]["regimes"]:
            regime_episodes = [ep for ep in test if ep.regime == regime]
            labels, ids = _flatten_labels(regime_episodes), _flatten_ids(regime_episodes)
            task_started = time.perf_counter()
            llm_probability = llm.predict(regime_episodes).probabilities
            tfm_probability = np.concatenate([test_specialist[ep.episode_id].probability for ep in regime_episodes])
            ensemble_probability = ensemble_alpha * llm_probability + (1 - ensemble_alpha) * tfm_probability

            l2t_probabilities, l2t3_probabilities, selected_one, selected_three = [], [], [], []
            for ep in regime_episodes:
                p1, indices1, weights = l2t.predict(test_views[ep.episode_id], top_k=1)
                p3, indices3, _ = l2t.predict(test_views[ep.episode_id], top_k=3)
                l2t_probabilities.extend(p1.tolist()); l2t3_probabilities.extend(p3.tolist())
                selected_one.append(indices1); selected_three.append(indices3)
            l2t_probability = np.asarray(l2t_probabilities)
            l2t3_probability = np.asarray(l2t3_probabilities)
            t2l_probability, t2l_labels, t2l_ids = t2l.predict(
                regime_episodes, test_specialist, int(config["training"]["t2l_batch_size"]),
            )
            if not np.array_equal(labels, t2l_labels) or not np.array_equal(ids, t2l_ids):
                raise RuntimeError("T->L prediction alignment failure")
            textual_probability = np.concatenate([
                llm.predict_prompts_likelihood(tool_prompts(ep, test_specialist[ep.episode_id].probability))
                for ep in regime_episodes
            ])

            oracle_probability = adaptive_routing_oracle(regime_episodes).probabilities
            values = {
                "llm_only": (llm_probability, 0, len(labels), 0),
                "tabicl_only": (tfm_probability, len(regime_episodes), 0, 0),
                "prediction_ensemble": (ensemble_probability, len(regime_episodes), len(labels), 0),
                "llm_to_tfm": (l2t_probability, len(regime_episodes), 0, l2t.trainable_params),
                "tfm_to_llm": (t2l_probability, len(regime_episodes), math.ceil(len(labels) / int(config["training"]["t2l_batch_size"])), t2l.trainable_params),
                "textual_tool": (textual_probability, len(regime_episodes), len(labels), 0),
                "llm_to_tfm_compute_matched": (l2t3_probability, 3 * len(regime_episodes), 0, l2t.trainable_params),
                "adaptive_diagnostic_oracle": (oracle_probability, 0, 0, 0),
            }
            for method in config["experiment"]["methods"]:
                probability, specialist_calls, llm_calls, trainable_params = values[method]
                prefix = config["experiment"].get("artifact_prefix", "phase2")
                experiment_id = f"{prefix}-{method}-{regime}-seed{seed}"
                prediction_path = output / "predictions" / f"{experiment_id}.npz"
                _atomic_npz(prediction_path, probability=probability, label=labels, episode_id=ids)
                record = {
                    "experiment_id": experiment_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "protocol_id": config["experiment"]["protocol_id"],
                    "classification": config["experiment"]["classification"],
                    "git_commit": config["source"]["git_commit"], "source_dirty": bool(config["source"]["dirty"]),
                    "config_digest": config_digest, "wheel_sha256": wheel_sha,
                    "dataset": "crossfm-synthetic-v3-adaptive" if "C2" in config["experiment"]["regimes"] else "crossfm-synthetic-v2",
                    "dataset_revision": "adaptive-routing-heldout-codebook-alias-and-mechanism" if "C2" in config["experiment"]["regimes"] else "heldout-alias-and-mechanism",
                    "dataset_checksum": canonical_digest([ep.checksum() for ep in regime_episodes]),
                    "condition": regime, "seed": seed, "split_hash": split_hash(regime_episodes),
                    "model": f"{llm_cfg['id']} + {tfm_cfg['id']}",
                    "model_revision": f"{llm_cfg['revision']} + {tfm_cfg['revision']}",
                    "runtime_device": f"cuda:0 (isolated physical rank {args.rank})",
                    "metric": "accuracy", **_metrics(labels, probability),
                    "artifact_schema_version": "2.0.0", "status": "complete", "method": method,
                    "n_predictions": int(len(labels)), "specialist_calls": int(specialist_calls),
                    "llm_calls": int(llm_calls), "trainable_params": int(trainable_params),
                    "elapsed_seconds_shared_cell": time.perf_counter() - task_started,
                    "peak_gpu_memory_bytes_worker": int(torch.cuda.max_memory_allocated()),
                    "prediction_file": str(prediction_path.relative_to(output)),
                    "prediction_sha256": sha256_file(prediction_path),
                    "ensemble_alpha": ensemble_alpha if method == "prediction_ensemble" else None,
                    "validation_loss": ensemble_val_loss if method == "prediction_ensemble" else None,
                    "checkpoint_file": str(checkpoint.relative_to(output)) if trainable_params else None,
                    "checkpoint_sha256": sha256_file(checkpoint) if trainable_params else None,
                    "selected_view_indices": selected_one if method == "llm_to_tfm" else selected_three if method == "llm_to_tfm_compute_matched" else None,
                }
                atomic_json(output / "tasks" / f"{experiment_id}.json", record)
                records.append(record)
                print(json.dumps({"event": "task_complete", "id": experiment_id, "accuracy": record["score"]}), flush=True)

    atomic_json(output / f"worker_{args.rank}_summary.json", {
        "status": "complete", "rank": args.rank, "world_size": args.world_size,
        "assigned_seeds": seeds, "completed": len(records), "config_digest": config_digest,
        "wheel_sha256": wheel_sha, "trainable_params_total": total_trainable,
        "l2t_training": l2t_training, "t2l_training": t2l_training,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "elapsed_seconds": time.perf_counter() - started, "platform": platform.platform(),
    })


if __name__ == "__main__":
    main()
