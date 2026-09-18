from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from .baselines import QwenBinaryBaseline, TabICLBaseline
from .io import atomic_json, canonical_digest, sha256_file
from .phase3 import CrossFMLatentLoop
from .phase3_dynamic import (
    DynamicEvidenceBridge, build_dynamic_static_cache, dynamic_specialist_observation,
)
from .synthetic import make_episodes, split_hash


METHODS = ("dynamic_1", "dynamic_r2", "dynamic_r2_zero", "dynamic_r2_shuffle")


def _episodes(config: dict, split: str, seed: int, rank: int | None = None, world_size: int = 1):
    spec, result = config["data"][split], []
    for regime in config["experiment"]["regimes"]:
        episodes = make_episodes(
            regime, seed, int(spec["episodes"][regime]), int(spec["queries_per_episode"]),
            alias_split=spec["alias_split"], mechanism_split=spec["mechanism_split"],
        )
        if rank is not None:
            episodes = [episode for index, episode in enumerate(episodes) if index % world_size == rank]
        result.extend(episodes)
    return result


def _metrics(labels: np.ndarray, probability: np.ndarray) -> dict:
    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    return {
        "score": float(accuracy_score(labels, probability >= 0.5)),
        "roc_auc": float(roc_auc_score(labels, probability)) if len(np.unique(labels)) == 2 else 0.5,
        "log_loss": float(log_loss(labels, probability, labels=[0, 1])),
    }


def _blend(item, probability: np.ndarray) -> np.ndarray:
    return ((1.0 - item.gate) * item.llm_probability + item.gate * probability).clip(1e-5, 1 - 1e-5)


def _npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{path.stat().st_ino if path.exists() else 0}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--world-size", required=True, type=int)
    parser.add_argument("--wheel", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    args = parser.parse_args()

    import torch

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output, started = Path(args.output), time.perf_counter()
    digest, wheel_sha = canonical_digest(config), sha256_file(Path(args.wheel))
    torch.manual_seed(int(config["training"]["seed"]))
    np.random.seed(int(config["training"]["seed"]))
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()

    checkpoint = Path(args.checkpoint_root) / f"phase3-adapters-rank{args.rank}.pt"
    expected_checkpoint_sha = config["source_checkpoint"]["sha256"][str(args.rank)]
    if sha256_file(checkpoint) != expected_checkpoint_sha:
        raise RuntimeError(f"Checkpoint hash mismatch for rank {args.rank}")

    llm_cfg, tfm_cfg = config["models"]["llm"], config["models"]["specialist"]
    llm = QwenBinaryBaseline(
        llm_cfg["id"], llm_cfg["revision"], batch_size=int(config["runtime"]["llm_batch_size"]),
        torch_dtype=str(config["runtime"]["torch_dtype"]),
    )
    llm.hidden_state_preflight()
    tfm = TabICLBaseline(tfm_cfg["id"], tfm_cfg["revision"], tfm_cfg["checkpoint"])
    tfm.preflight()
    threshold = float(config["routing"]["fallback_threshold"])

    train_episodes = _episodes(config, "train", int(config["data"]["train"]["seed"]))
    print(json.dumps({"event": "static_train_cache", "episodes": len(train_episodes)}), flush=True)
    train_cache = build_dynamic_static_cache(
        llm, tfm, train_episodes, score_llm=False, fallback_threshold=threshold,
    )
    test_cache = {}
    for seed in config["experiment"]["test_seeds"]:
        episodes = _episodes(config, "test", int(seed), args.rank, args.world_size)
        print(json.dumps({"event": "static_test_cache", "seed": seed, "episodes": len(episodes)}), flush=True)
        test_cache[int(seed)] = build_dynamic_static_cache(
            llm, tfm, episodes, score_llm=True, fallback_threshold=threshold,
        )

    embedding_dim = len(train_cache[0].task_embedding)
    controller = CrossFMLatentLoop(
        embedding_dim, int(config["controller"]["hidden_dim"]), 2, "cuda:0",
        int(config["training"]["seed"]),
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    controller.module.load_state_dict(saved["models"]["2"])
    controller_parameter_count = controller.trainable_params
    controller.module.eval()
    for parameter in controller.module.parameters():
        parameter.requires_grad_(False)

    bridge_features, bridge_targets, dynamic_train_calls = [], [], 0
    for index, item in enumerate(train_cache):
        state = torch.from_numpy(item.task_embedding).to("cuda:0")[None, :].repeat(len(item.episode.y_query), 1)
        observation = dynamic_specialist_observation(tfm, item, controller, state, round_index=0)
        bridge_features.append(observation.bridge_input)
        bridge_targets.append(observation.teacher_message)
        dynamic_train_calls += observation.specialist_calls
        if (index + 1) % 10 == 0:
            print(json.dumps({"event": "dynamic_bridge_cache", "complete": index + 1}), flush=True)
    features, targets = np.concatenate(bridge_features), np.concatenate(bridge_targets)
    bridge = DynamicEvidenceBridge(
        features.shape[1], embedding_dim, int(config["training"]["bridge_hidden_dim"]),
        "cuda:0", int(config["training"]["seed"]),
    )
    if bridge.trainable_params + controller_parameter_count >= int(config["training"]["max_trainable_params"]):
        raise RuntimeError("Dynamic bridge plus controller exceeds the frozen parameter budget")
    bridge_training = bridge.fit(
        features, targets, epochs=int(config["training"]["epochs"]),
        batch_size=int(config["training"]["batch_size"]),
        learning_rate=float(config["training"]["learning_rate"]), use_bfloat16=True,
    )

    records, dynamic_test_calls, prefix_tokens = [], 0, 0
    for seed, items in test_cache.items():
        by_regime = {regime: [item for item in items if item.episode.regime == regime] for regime in config["experiment"]["regimes"]}
        for regime, regime_items in by_regime.items():
            labels = np.concatenate([item.episode.y_query for item in regime_items])
            ids = np.concatenate([[item.episode.episode_id] * len(item.episode.y_query) for item in regime_items])
            predictions = {method: [] for method in METHODS}
            selected_views = {method: [] for method in METHODS}
            if regime != "C2":
                for item in regime_items:
                    preserved = item.llm_probability if regime == "A" else item.tfm_probability
                    for method in METHODS:
                        predictions[method].append(preserved)
                        selected_views[method].append(np.full(len(preserved), -1, dtype=np.int64))
            else:
                first = []
                for item in regime_items:
                    state = torch.from_numpy(item.task_embedding).to("cuda:0")[None, :].repeat(len(item.episode.y_query), 1)
                    observation = dynamic_specialist_observation(tfm, item, controller, state, round_index=0)
                    first.append(observation)
                    dynamic_test_calls += observation.specialist_calls
                    predictions["dynamic_1"].append(_blend(item, observation.probability))
                    selected_views["dynamic_1"].append(observation.selected_view)
                normal_messages = [bridge.predict(observation.bridge_input, True) for observation in first]
                mode_messages = {
                    "dynamic_r2": normal_messages,
                    "dynamic_r2_zero": [torch.zeros_like(message) for message in normal_messages],
                    "dynamic_r2_shuffle": normal_messages[1:] + normal_messages[:1],
                }
                for method, messages in mode_messages.items():
                    for item, message in zip(regime_items, messages):
                        _, kv, tokens = llm.prefix_state_and_kv(item.prefix_text, len(item.episode.y_query))
                        prefix_tokens += tokens
                        dynamic_state, _ = llm.update_from_soft_evidence(message, kv)
                        observation = dynamic_specialist_observation(
                            tfm, item, controller, dynamic_state, round_index=1,
                        )
                        dynamic_test_calls += observation.specialist_calls
                        predictions[method].append(_blend(item, observation.probability))
                        selected_views[method].append(observation.selected_view)

            for method in METHODS:
                probability = np.concatenate(predictions[method])
                views = np.concatenate(selected_views[method])
                experiment_id = f"phase3-dynamic-{method}-{regime}-seed{seed}-rank{args.rank}"
                prediction_path = output / "predictions" / f"{experiment_id}.npz"
                _npz(prediction_path, probability=probability, label=labels, episode_id=ids, selected_view=views)
                record = {
                    "experiment_id": experiment_id, "protocol_id": config["experiment"]["protocol_id"],
                    "classification": config["experiment"]["classification"],
                    "timestamp": datetime.now(timezone.utc).isoformat(), "git_commit": config["source"]["git_commit"],
                    "config_digest": digest, "wheel_sha256": wheel_sha,
                    "checkpoint_sha256": expected_checkpoint_sha, "condition": regime, "seed": seed,
                    "rank": args.rank, "method": method, "rounds": 1 if method == "dynamic_1" else 2,
                    "split_hash": split_hash([item.episode for item in regime_items]),
                    "n_predictions": len(labels), "prediction_file": str(prediction_path.relative_to(output)),
                    "prediction_sha256": sha256_file(prediction_path), **_metrics(labels, probability),
                }
                atomic_json(output / "tasks" / f"{experiment_id}.json", record)
                records.append(record)

    bridge_checkpoint = output / "checkpoints" / f"dynamic-evidence-bridge-rank{args.rank}.pt"
    bridge_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bridge.module.state_dict(), bridge_checkpoint)
    state_hash = hashlib.sha256(bridge_checkpoint.read_bytes()).hexdigest()
    atomic_json(output / f"worker_{args.rank}_summary.json", {
        "status": "complete", "rank": args.rank, "completed": len(records), "config_digest": digest,
        "wheel_sha256": wheel_sha, "source_checkpoint_sha256": expected_checkpoint_sha,
        "controller_params": controller_parameter_count, "bridge_params": bridge.trainable_params,
        "bridge_training": bridge_training, "bridge_checkpoint_sha256": state_hash,
        "static_token_cache_entries": llm.static_token_cache_entries,
        "prefix_tokens_processed": prefix_tokens, "dynamic_train_specialist_calls": dynamic_train_calls,
        "dynamic_test_specialist_calls": dynamic_test_calls,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "elapsed_seconds": time.perf_counter() - started,
    })


if __name__ == "__main__":
    main()
