from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from .baselines import QwenBinaryBaseline, TabICLBaseline, semantic_statistical_oracle
from .io import atomic_json, canonical_digest, sha256_file
from .synthetic import make_episodes, split_hash


def task_grid(config: dict) -> list[dict]:
    return [
        {"method": method, "regime": regime, "seed": seed}
        for regime in config["experiment"]["regimes"]
        for seed in config["experiment"]["seeds"]
        for method in config["experiment"]["methods"]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--wheel", required=True)
    args = parser.parse_args()

    config_path, output = Path(args.config), Path(args.output)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_digest = canonical_digest(config)
    wheel_sha = sha256_file(Path(args.wheel))
    tasks = task_grid(config)
    assigned = [task for i, task in enumerate(tasks) if i % args.world_size == args.rank]
    llm = tabicl = None
    records = []
    started = time.perf_counter()
    for task in assigned:
        method, regime, seed = task["method"], task["regime"], int(task["seed"])
        experiment_id = f"phase1-{method}-{regime}-seed{seed}"
        record_path = output / "tasks" / f"{experiment_id}.json"
        episodes = make_episodes(regime, seed, int(config["experiment"]["episodes_per_cell"]), int(config["experiment"]["queries_per_episode"]))
        task_started = time.perf_counter()
        if method == "llm_only":
            if llm is None:
                llm = QwenBinaryBaseline(
                    config["models"]["llm"]["id"], config["models"]["llm"]["revision"],
                    batch_size=int(config["runtime"]["llm_batch_size"])
                )
                llm_shape = llm.hidden_state_preflight()
            batch = llm.predict(episodes)
            model_name = config["models"]["llm"]["id"]
            model_revision = config["models"]["llm"]["revision"]
        elif method == "tabicl_only":
            if tabicl is None:
                tabicl = TabICLBaseline(
                    config["models"]["specialist"]["id"],
                    config["models"]["specialist"]["revision"],
                    config["models"]["specialist"]["checkpoint"],
                )
                tabicl.preflight()
            batch = tabicl.predict(episodes)
            model_name = config["models"]["specialist"]["id"]
            model_revision = config["models"]["specialist"]["checkpoint"]
        elif method == "semantic_statistical_oracle":
            batch = semantic_statistical_oracle(episodes)
            model_name = "diagnostic-oracle-not-learned"
            model_revision = "v1"
        else:
            raise ValueError(method)

        pred_path = output / "predictions" / f"{experiment_id}.npz"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = pred_path.with_suffix(f".rank{args.rank}.tmp.npz")
        np.savez_compressed(tmp_path, probability=batch.probabilities, label=batch.labels, episode_id=batch.episode_ids)
        os.replace(tmp_path, pred_path)
        accuracy = float(accuracy_score(batch.labels, batch.probabilities >= 0.5))
        auc = float(roc_auc_score(batch.labels, batch.probabilities))
        record = {
            "experiment_id": experiment_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "protocol_id": config["experiment"]["protocol_id"],
            "git_commit": config["source"]["git_commit"],
            "source_dirty": bool(config["source"]["dirty"]),
            "config_digest": config_digest,
            "wheel_sha256": wheel_sha,
            "dataset": "crossfm-synthetic-v1",
            "dataset_revision": "generated-from-frozen-code",
            "dataset_checksum": canonical_digest([ep.checksum() for ep in episodes]),
            "condition": regime,
            "seed": seed,
            "split_hash": split_hash(episodes),
            "model": model_name,
            "model_revision": model_revision,
            "runtime_device": f"cuda:0 (isolated physical rank {args.rank})",
            "metric": "accuracy",
            "score": accuracy,
            "roc_auc": auc,
            "log_loss": float(log_loss(batch.labels, batch.probabilities, labels=[0, 1])),
            "artifact_schema_version": "1.0.0",
            "status": "complete",
            "method": method,
            "n_predictions": int(len(batch.labels)),
            "inference_calls": int(len(batch.labels)) if method == "llm_only" else int(len(episodes)),
            "trainable_params": 0,
            "elapsed_seconds": time.perf_counter() - task_started,
            "prediction_file": str(pred_path.relative_to(output)),
            "prediction_sha256": sha256_file(pred_path),
        }
        atomic_json(record_path, record)
        records.append(record)
        print(json.dumps({"event": "task_complete", "id": experiment_id, "accuracy": accuracy}), flush=True)
    atomic_json(output / f"worker_{args.rank}_summary.json", {
        "status": "complete", "rank": args.rank, "world_size": args.world_size,
        "assigned": len(assigned), "completed": len(records), "elapsed_seconds": time.perf_counter() - started,
        "platform": platform.platform(), "config_digest": config_digest, "wheel_sha256": wheel_sha,
    })


if __name__ == "__main__":
    main()
