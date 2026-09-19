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
from .phase3 import build_crossfm_cache, cache_diagnostics, pack_cache
from .phase5 import CrossFMShortcutAudit, fixed_router_predict
from .synthetic import Episode, make_episodes, split_hash


METHODS = {
    "analytic_router": ("fixed", 1, "analytic"),
    "semantic_cosine_router": ("fixed", 1, "semantic_cosine"),
    "prior_start_r1": ("prior", 1, "prior"),
    "prior_start_r2": ("prior", 2, "prior"),
    "prior_start_r3": ("prior", 3, "prior"),
    "message_only_r1": ("message", 1, "message"),
    "message_only_r2": ("message", 2, "message"),
    "message_only_r3": ("message", 3, "message"),
    "message_only_r2_zero_t2l": ("message", 2, "zero_t2l"),
    "message_only_r2_shuffle_t2l": ("message", 2, "shuffle_t2l"),
    "message_only_r2_zero_l2t": ("message", 2, "zero_l2t"),
    "l2t_only_compute_matched": ("l2t", 2, "l2t_only"),
    "random_bridge_r2": ("random", 2, "message"),
}


def _episodes(config: dict, split: str, seed: int, rank: int | None = None, world_size: int = 1) -> list[Episode]:
    spec = config["data"][split]
    episodes = []
    for regime in config["experiment"]["regimes"]:
        generated = make_episodes(
            regime, seed, int(spec["episodes"][regime]), int(spec["queries_per_episode"]),
            alias_split=spec["alias_split"], mechanism_split=spec["mechanism_split"],
        )
        if rank is not None:
            generated = [episode for index, episode in enumerate(generated) if index % world_size == rank]
        episodes.extend(generated)
    return episodes


def _cache_split(llm, tfm, episodes, *, score_llm, threshold):
    specialist = collect_specialist_evidence(tfm, episodes)
    views = collect_view_evidence(tfm, episodes)
    attach_language_embeddings(llm, episodes, views)
    return build_crossfm_cache(llm, episodes, specialist, views, score_llm=score_llm, fallback_threshold=threshold)


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _metrics(labels: np.ndarray, probability: np.ndarray) -> dict:
    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    return {
        "score": float(accuracy_score(labels, probability >= 0.5)),
        "roc_auc": float(roc_auc_score(labels, probability)) if len(np.unique(labels)) == 2 else 0.5,
        "log_loss": float(log_loss(labels, probability, labels=[0, 1])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--rank", required=True, type=int); parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--wheel", required=True)
    args = parser.parse_args()

    import torch

    config_path, output = Path(args.config), Path(args.output)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if tuple(config["experiment"]["methods"]) != tuple(METHODS):
        raise RuntimeError("Frozen Phase 5 method order does not match implementation")
    config_digest, wheel_sha = canonical_digest(config), sha256_file(Path(args.wheel))
    started, training_seed = time.perf_counter(), int(config["training"]["seed"])
    torch.manual_seed(training_seed); np.random.seed(training_seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high"); torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()

    llm_cfg, tfm_cfg = config["models"]["llm"], config["models"]["specialist"]
    llm = QwenBinaryBaseline(
        llm_cfg["id"], llm_cfg["revision"], batch_size=int(config["runtime"]["llm_batch_size"]),
        torch_dtype=str(config["runtime"]["torch_dtype"]),
    )
    llm.hidden_state_preflight()
    tfm = TabICLBaseline(tfm_cfg["id"], tfm_cfg["revision"], tfm_cfg["checkpoint"]); tfm.preflight()
    threshold = float(config["routing"]["fallback_threshold"])
    train_episodes = _episodes(config, "train", int(config["data"]["train"]["seed"]))
    validation_episodes = _episodes(config, "validation", int(config["data"]["validation"]["seed"]))
    print(json.dumps({"event": "shared_cache_train", "episodes": len(train_episodes), "rank": args.rank}), flush=True)
    train_cache = _cache_split(llm, tfm, train_episodes, score_llm=False, threshold=threshold)
    validation_cache = _cache_split(llm, tfm, validation_episodes, score_llm=False, threshold=threshold)
    test_episodes_by_seed, test_cache_by_seed = {}, {}
    for seed_value in config["experiment"]["test_seeds"]:
        seed = int(seed_value)
        episodes = _episodes(config, "test", seed, args.rank, args.world_size)
        test_episodes_by_seed[seed] = episodes
        print(json.dumps({"event": "shared_cache_test", "seed": seed, "episodes": len(episodes), "rank": args.rank}), flush=True)
        test_cache_by_seed[seed] = _cache_split(llm, tfm, episodes, score_llm=True, threshold=threshold)
    cache_bytes = int(sum(
        array.nbytes for caches in [train_cache, validation_cache, *test_cache_by_seed.values()] for item in caches
        for array in (item.task_embedding, item.view_embeddings, item.route_message, item.view_probabilities,
                      item.llm_probability, item.tfm_probability, item.labels, item.route_view_prior)
    ))
    embedding_dim = len(train_cache[0].task_embedding)
    train_diagnostics, validation_diagnostics = cache_diagnostics(train_cache), cache_diagnostics(validation_cache)
    del llm, tfm; gc.collect(); torch.cuda.empty_cache()
    backbone_peak = int(torch.cuda.max_memory_allocated())
    train_pack, validation_pack = pack_cache(train_cache, "cuda:0"), pack_cache(validation_cache, "cuda:0")
    test_packs = {
        seed: {regime: pack_cache([item for item in cache if item.regime == regime], "cuda:0")
               for regime in config["experiment"]["regimes"]}
        for seed, cache in test_cache_by_seed.items()
    }

    common = {
        "epochs": int(config["training"]["epochs"]), "batch_size": int(config["training"]["episode_batch_size"]),
        "learning_rate": float(config["training"]["learning_rate"]), "patience": int(config["training"]["patience"]),
    }
    models, training = {}, {}
    for name, mode, offset in (("message", "message", 0), ("prior", "prior", 1), ("l2t", "l2t_only", 2)):
        model = CrossFMShortcutAudit(embedding_dim, int(config["training"]["hidden_dim"]), "cuda:0", training_seed + offset)
        if model.trainable_params >= int(config["training"]["max_trainable_params"]):
            raise RuntimeError(f"Adapter budget exceeded for {name}: {model.trainable_params}")
        training[name] = model.fit(train_pack, validation_pack, mode=mode, **common)
        models[name] = model
    models["random"] = CrossFMShortcutAudit(
        embedding_dim, int(config["training"]["hidden_dim"]), "cuda:0", training_seed + 97,
    )
    state_digests = {name: model.state_digest() for name, model in models.items()}
    checkpoint = output / "checkpoints" / f"phase5-adapters-rank{args.rank}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(f".{os.getpid()}.tmp.pt")
    torch.save({"protocol_id": config["experiment"]["protocol_id"], "config_digest": config_digest,
                "models": {name: model.module.state_dict() for name, model in models.items()}, "training": training}, temporary)
    os.replace(temporary, checkpoint)

    records, batch_size = [], int(config["training"]["episode_batch_size"])
    for seed, episodes in test_episodes_by_seed.items():
        for regime in config["experiment"]["regimes"]:
            regime_episodes = [episode for episode in episodes if episode.regime == regime]
            pack = test_packs[seed][regime]
            labels = np.concatenate([episode.y_query for episode in regime_episodes])
            ids = np.concatenate([[episode.episode_id] * len(episode.y_query) for episode in regime_episodes])
            route_codes = np.asarray([episode.route_code for episode in regime_episodes], dtype=np.int64)
            for method, (model_name, rounds, mode) in METHODS.items():
                with torch.inference_mode():
                    if model_name == "fixed":
                        probability, predicted_labels, traces = fixed_router_predict(
                            pack, mode, float(config["audit"]["semantic_cosine_temperature"]),
                        )
                        trainable_params, state_digest = 0, f"fixed-{mode}-zero-parameters"
                    else:
                        probability, predicted_labels, traces = models[model_name].predict(pack, rounds, batch_size, mode)
                        trainable_params, state_digest = models[model_name].trainable_params, state_digests[model_name]
                if not np.array_equal(labels, predicted_labels.astype(labels.dtype)):
                    raise RuntimeError(f"Prediction alignment failed for {method} {regime} {seed} rank {args.rank}")
                experiment_id = f"phase5-{method}-{regime}-seed{seed}-rank{args.rank}"
                prediction_path = output / "predictions" / f"{experiment_id}.npz"
                _atomic_npz(
                    prediction_path, probability=probability, label=labels, episode_id=ids,
                    residual_gate=pack["gates"].detach().cpu().numpy(),
                    relevant_view=pack["relevant_views"].detach().cpu().numpy(), route_code=route_codes,
                    round_attention=traces["weights"], round_probability=traces["selected_probability"],
                    round_entropy=traces["entropy"], round_state_delta_norm=traces["state_delta_norm"],
                )
                record = {
                    "experiment_id": experiment_id, "timestamp": datetime.now(timezone.utc).isoformat(),
                    "protocol_id": config["experiment"]["protocol_id"], "classification": config["experiment"]["classification"],
                    "git_commit": config["source"]["git_commit"], "source_dirty": bool(config["source"]["dirty"]),
                    "config_digest": config_digest, "wheel_sha256": wheel_sha,
                    "dataset": "crossfm-synthetic-v5-shortcut-audit", "dataset_revision": "phase5-cached-view-bank-v1",
                    "dataset_checksum": canonical_digest([episode.checksum() for episode in regime_episodes]),
                    "condition": regime, "seed": seed, "rank": args.rank, "split_hash": split_hash(regime_episodes),
                    "method": method, "rounds": rounds, "message_mode": mode, "adapter_model": model_name,
                    "direct_analytical_prior": mode in {"analytic", "prior"},
                    "model": f"{llm_cfg['id']} + {tfm_cfg['id']} + CrossFMShortcutAudit",
                    "model_revision": f"{llm_cfg['revision']} + {tfm_cfg['revision']}",
                    "runtime_device": f"cuda:0 (isolated physical rank {args.rank})", "metric": "accuracy",
                    **_metrics(labels, probability), "artifact_schema_version": "5.0.0", "status": "complete",
                    "n_predictions": len(labels), "trainable_params": trainable_params,
                    "conceptual_specialist_calls": rounds * len(regime_episodes), "incremental_backbone_calls": 0,
                    "shared_cached_specialist_view_calls": sum(len(item.view_embeddings) for item in test_cache_by_seed[seed] if item.regime == regime),
                    "cache_bytes_worker": cache_bytes, "backbone_peak_gpu_memory_bytes": backbone_peak,
                    "peak_gpu_memory_bytes_worker": int(torch.cuda.max_memory_allocated()),
                    "prediction_file": str(prediction_path.relative_to(output)), "prediction_sha256": sha256_file(prediction_path),
                    "checkpoint_file": str(checkpoint.relative_to(output)), "checkpoint_sha256": sha256_file(checkpoint),
                    "state_digest": state_digest,
                }
                atomic_json(output / "tasks" / f"{experiment_id}.json", record); records.append(record)
                print(json.dumps({"event": "task_complete", "id": experiment_id, "accuracy": record["score"]}), flush=True)
    atomic_json(output / f"worker_{args.rank}_summary.json", {
        "status": "complete", "rank": args.rank, "world_size": args.world_size, "completed": len(records),
        "config_digest": config_digest, "wheel_sha256": wheel_sha, "training": training,
        "state_digests": state_digests, "train_cache_diagnostics": train_diagnostics,
        "validation_cache_diagnostics": validation_diagnostics,
        "trainable_params_max": max(model.trainable_params for model in models.values()),
        "cache_bytes": cache_bytes, "backbone_peak_gpu_memory_bytes": backbone_peak,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "elapsed_seconds": time.perf_counter() - started, "platform": platform.platform(),
    })


if __name__ == "__main__":
    main()
