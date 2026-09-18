from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from .baselines import QwenBinaryBaseline, TabICLBaseline
from .io import atomic_json, canonical_digest, sha256_file
from .phase2 import attach_language_embeddings, collect_specialist_evidence, collect_view_evidence
from .phase3 import CrossFMLatentLoop, build_crossfm_cache, pack_cache
from .synthetic import Episode, make_episodes, split_hash


METHOD_ROUNDS = {"crossfm_1": 1, "crossfm_r2": 2, "crossfm_r2_zero": 2, "crossfm_r2_shuffle": 2}


def _episodes(config: dict, split: str, seed: int, rank: int | None = None, world_size: int = 1) -> list[Episode]:
    spec = config["data"][split]
    episodes = []
    for regime in config["experiment"]["regimes"]:
        count = int(spec["episodes"][regime])
        generated = make_episodes(
            regime, seed, count, int(spec["queries_per_episode"]),
            alias_split=spec["alias_split"], mechanism_split=spec["mechanism_split"],
        )
        if rank is not None:
            generated = [episode for index, episode in enumerate(generated) if index % world_size == rank]
        episodes.extend(generated)
    return episodes


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _metrics(labels: np.ndarray, probability: np.ndarray) -> dict:
    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    return {
        "score": float(accuracy_score(labels, probability >= 0.5)),
        "roc_auc": float(roc_auc_score(labels, probability)) if len(np.unique(labels)) == 2 else 0.5,
        "log_loss": float(log_loss(labels, probability, labels=[0, 1])),
    }


def _cache_split(llm, tfm, episodes, *, score_llm, threshold):
    specialist = collect_specialist_evidence(tfm, episodes)
    views = collect_view_evidence(tfm, episodes)
    attach_language_embeddings(llm, episodes, views)
    return build_crossfm_cache(
        llm, episodes, specialist, views,
        score_llm=score_llm, fallback_threshold=threshold,
    )


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
    started = time.perf_counter()
    torch.manual_seed(int(config["training"]["seed"]))
    np.random.seed(int(config["training"]["seed"]))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")
    torch.cuda.reset_peak_memory_stats()

    llm_cfg, tfm_cfg = config["models"]["llm"], config["models"]["specialist"]
    llm = QwenBinaryBaseline(
        llm_cfg["id"], llm_cfg["revision"], batch_size=int(config["runtime"]["llm_batch_size"]),
    )
    llm.hidden_state_preflight()
    tfm = TabICLBaseline(tfm_cfg["id"], tfm_cfg["revision"], tfm_cfg["checkpoint"])
    tfm.preflight()
    threshold = float(config["routing"]["fallback_threshold"])

    train_episodes = _episodes(config, "train", int(config["data"]["train"]["seed"]))
    validation_episodes = _episodes(config, "validation", int(config["data"]["validation"]["seed"]))
    print(json.dumps({"event": "cache_train", "episodes": len(train_episodes), "rank": args.rank}), flush=True)
    train_cache = _cache_split(llm, tfm, train_episodes, score_llm=False, threshold=threshold)
    validation_cache = _cache_split(llm, tfm, validation_episodes, score_llm=False, threshold=threshold)

    test_episodes_by_seed = {}
    test_cache_by_seed = {}
    for seed in config["experiment"]["test_seeds"]:
        seed = int(seed)
        episodes = _episodes(config, "test", seed, args.rank, args.world_size)
        test_episodes_by_seed[seed] = episodes
        print(json.dumps({"event": "cache_test", "seed": seed, "episodes": len(episodes), "rank": args.rank}), flush=True)
        test_cache_by_seed[seed] = _cache_split(llm, tfm, episodes, score_llm=True, threshold=threshold)

    cache_bytes = int(sum(
        array.nbytes
        for caches in [train_cache, validation_cache, *test_cache_by_seed.values()]
        for item in caches
        for array in (
            item.task_embedding, item.view_embeddings, item.route_message,
            item.view_probabilities, item.llm_probability, item.tfm_probability, item.labels,
        )
    ))
    embedding_dim = len(train_cache[0].task_embedding)
    del llm, tfm
    gc.collect(); torch.cuda.empty_cache()
    backbone_peak = int(torch.cuda.max_memory_allocated())

    train_pack = pack_cache(train_cache, "cuda:0")
    validation_pack = pack_cache(validation_cache, "cuda:0")
    test_packs = {
        seed: {
            regime: pack_cache([item for item in cache if item.regime == regime], "cuda:0")
            for regime in config["experiment"]["regimes"]
        }
        for seed, cache in test_cache_by_seed.items()
    }

    fitted = {}
    training = {}
    for rounds in sorted(set(METHOD_ROUNDS[method] for method in config["experiment"]["methods"])):
        model = CrossFMLatentLoop(
            embedding_dim, int(config["training"]["hidden_dim"]),
            int(config["training"]["max_rounds"]), "cuda:0", int(config["training"]["seed"]),
        )
        if model.trainable_params >= int(config["training"]["max_trainable_params"]):
            raise RuntimeError(f"Adapter budget exceeded: {model.trainable_params}")
        training[str(rounds)] = model.fit(
            train_pack, validation_pack, rounds, int(config["training"]["epochs"]),
            int(config["training"]["episode_batch_size"]), float(config["training"]["learning_rate"]),
            int(config["training"]["patience"]), config["training"].get("checkpoint_metric", "validation"),
        )
        fitted[rounds] = model

    checkpoint = output / "checkpoints" / f"phase3-adapters-rank{args.rank}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    tmp_checkpoint = checkpoint.with_suffix(f".{os.getpid()}.tmp.pt")
    torch.save({
        "protocol_id": config["experiment"]["protocol_id"], "config_digest": config_digest,
        "models": {str(rounds): model.module.state_dict() for rounds, model in fitted.items()},
        "training": training,
    }, tmp_checkpoint)
    os.replace(tmp_checkpoint, checkpoint)
    state_digests = {str(rounds): model.state_digest() for rounds, model in fitted.items()}

    records = []
    batch_size = int(config["training"]["episode_batch_size"])
    for seed, episodes in test_episodes_by_seed.items():
        for regime in config["experiment"]["regimes"]:
            regime_episodes = [ep for ep in episodes if ep.regime == regime]
            pack = test_packs[seed][regime]
            labels = np.concatenate([ep.y_query for ep in regime_episodes])
            ids = np.concatenate([[ep.episode_id] * len(ep.y_query) for ep in regime_episodes])
            for method in config["experiment"]["methods"]:
                rounds = METHOD_ROUNDS[method]
                message_mode = "zero" if method.endswith("_zero") else "shuffle" if method.endswith("_shuffle") else "normal"
                probability, predicted_labels = fitted[rounds].predict(pack, rounds, batch_size, message_mode)
                if not np.array_equal(labels, predicted_labels.astype(labels.dtype)):
                    raise RuntimeError(f"Prediction alignment failed for {method} {regime} {seed} rank {args.rank}")
                experiment_id = f"phase3-{method}-{regime}-seed{seed}-rank{args.rank}"
                prediction_path = output / "predictions" / f"{experiment_id}.npz"
                gates = pack["gates"].detach().cpu().numpy().repeat(len(regime_episodes[0].y_query), axis=1).reshape(-1)
                _atomic_npz(prediction_path, probability=probability, label=labels, episode_id=ids, residual_gate=gates)
                record = {
                    "experiment_id": experiment_id, "timestamp": datetime.now(timezone.utc).isoformat(),
                    "protocol_id": config["experiment"]["protocol_id"],
                    "classification": config["experiment"]["classification"],
                    "git_commit": config["source"]["git_commit"], "source_dirty": bool(config["source"]["dirty"]),
                    "config_digest": config_digest, "wheel_sha256": wheel_sha,
                    "dataset": "crossfm-synthetic-v3-adaptive", "dataset_revision": "phase3-cached-view-bank-v1",
                    "dataset_checksum": canonical_digest([ep.checksum() for ep in regime_episodes]),
                    "condition": regime, "seed": seed, "rank": args.rank,
                    "split_hash": split_hash(regime_episodes), "method": method, "rounds": rounds,
                    "model": f"{llm_cfg['id']} + {tfm_cfg['id']} + CrossFMLatentLoop",
                    "model_revision": f"{llm_cfg['revision']} + {tfm_cfg['revision']}",
                    "runtime_device": f"cuda:0 (isolated physical rank {args.rank})",
                    "metric": "accuracy", **_metrics(labels, probability),
                    "artifact_schema_version": "3.0.0", "status": "complete",
                    "n_predictions": len(labels), "trainable_params": fitted[rounds].trainable_params,
                    "conceptual_specialist_calls": rounds * len(regime_episodes),
                    "cached_specialist_view_calls": sum(len(item.view_embeddings) for item in test_cache_by_seed[seed] if item.regime == regime),
                    "llm_calls": len(labels), "cache_bytes_worker": cache_bytes,
                    "backbone_peak_gpu_memory_bytes": backbone_peak,
                    "peak_gpu_memory_bytes_worker": int(torch.cuda.max_memory_allocated()),
                    "prediction_file": str(prediction_path.relative_to(output)),
                    "prediction_sha256": sha256_file(prediction_path),
                    "checkpoint_file": str(checkpoint.relative_to(output)),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "state_digest": state_digests[str(rounds)],
                }
                atomic_json(output / "tasks" / f"{experiment_id}.json", record)
                records.append(record)
                print(json.dumps({"event": "task_complete", "id": experiment_id, "accuracy": record["score"]}), flush=True)

    atomic_json(output / f"worker_{args.rank}_summary.json", {
        "status": "complete", "rank": args.rank, "world_size": args.world_size,
        "completed": len(records), "config_digest": config_digest, "wheel_sha256": wheel_sha,
        "training": training, "state_digests": state_digests,
        "trainable_params_max": max(model.trainable_params for model in fitted.values()),
        "cache_bytes": cache_bytes, "backbone_peak_gpu_memory_bytes": backbone_peak,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "elapsed_seconds": time.perf_counter() - started, "platform": platform.platform(),
    })


if __name__ == "__main__":
    main()
