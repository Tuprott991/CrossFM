from __future__ import annotations

import argparse
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback

from .aggregate import aggregate_run
from .artifacts import atomic_json, sha256_file
from .engine import FullPaperEngine, execute_task
from .models import configure_h100_math
from .protocol import balanced_shards, load_protocol, tasks_for_profile


def _materialize_source(config: dict) -> dict:
    """Replace BUILD_TIME sentinels with the exact clean source revision."""
    if config["source"].get("git_commit") != "BUILD_TIME":
        return config
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip())
    if dirty:
        raise RuntimeError(
            "Refusing an H100 research run from a dirty worktree; commit the frozen protocol first"
        )
    resolved = dict(config)
    resolved["source"] = {**config["source"], "git_commit": commit, "dirty": False}
    return resolved


def _device(accelerator: str) -> str:
    if accelerator == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("GPU profile requires PyTorch") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("GPU profile requested but CUDA is unavailable")
    return "cuda:0"


def _minimum_free_disk_gib(config: dict, accelerator: str) -> int:
    value = config["runtime"].get("minimum_free_disk_gib", 0)
    if isinstance(value, dict):
        value = value.get(accelerator, 0)
    minimum = int(value)
    if minimum < 0:
        raise ValueError("minimum_free_disk_gib must be non-negative")
    return minimum


def _model_preflight(config: dict, profile: str) -> dict:
    """Resolve pinned model metadata and the compact TabPFN checkpoint up front."""
    spec = config["profiles"][profile]
    methods = [config["methods"][name] for name in spec.get("methods", [])]
    report: dict[str, object] = {}
    if any(method.get("backend") == "tabicl" for method in methods):
        from tabicl import TabICLClassifier

        report["tabicl_import"] = TabICLClassifier.__name__
    tabpfn_methods = [method for method in methods if method.get("backend") == "tabpfn3"]
    if tabpfn_methods:
        from huggingface_hub import hf_hub_download
        from tabpfn import TabPFNClassifier

        params = tabpfn_methods[0]["params"]
        checkpoint = Path(hf_hub_download(
            repo_id=params["hf_repo_id"], filename=params["hf_filename"],
            revision=params["hf_revision"],
        ))
        observed = sha256_file(checkpoint)
        if observed != params["hf_sha256"]:
            raise RuntimeError(f"TabPFN doctor checksum mismatch: {observed}")
        report["tabpfn_import"] = TabPFNClassifier.__name__
        report["tabpfn_checkpoint_sha256"] = observed
    llm_methods = [method for method in methods if method.get("model_id")]
    if llm_methods:
        from transformers import AutoConfig, AutoTokenizer

        method = llm_methods[0]
        model_id, revision = method["model_id"], method["revision"]
        model_config = AutoConfig.from_pretrained(model_id, revision=revision)
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        report.update({
            "llm_model_id": model_id, "llm_revision": revision,
            "llm_architecture": type(model_config).__name__,
            "llm_tokenizer": type(tokenizer).__name__,
        })
    return report


def doctor(config: dict, profile: str, output: Path, data_root: Path) -> dict:
    spec = config["profiles"][profile]
    report = {
        "status": "complete", "profile": profile, "accelerator": spec["accelerator"],
        "python": sys.version, "platform": platform.platform(),
        "disk_free_bytes": shutil.disk_usage(output.parent if output.parent.exists() else Path.cwd()).free,
        "data_root": str(data_root.resolve()), "data_root_exists": data_root.exists(),
    }
    minimum_disk = _minimum_free_disk_gib(config, spec["accelerator"]) * 1024**3
    if report["disk_free_bytes"] < minimum_disk:
        raise RuntimeError(
            f"Insufficient output disk: {report['disk_free_bytes']} bytes free; "
            f"need at least {minimum_disk}"
        )
    installed = {}
    mismatched = {}
    for package, expected in config["runtime"].get("expected_package_versions", {}).items():
        try:
            observed = version(package)
        except PackageNotFoundError:
            observed = None
        installed[package] = observed
        if observed is None or observed.split("+", 1)[0] != str(expected):
            mismatched[package] = {"expected": str(expected), "observed": observed}
    report["package_versions"] = installed
    if mismatched:
        raise RuntimeError(f"Pinned package preflight failed: {mismatched}")
    missing_files = []
    for dataset_id in spec.get("datasets", []):
        dataset_spec = config["datasets"][dataset_id]
        if dataset_spec["loader"] != "beyondarena":
            for relative in dataset_spec.get("files", {}).values():
                if not (data_root / relative).is_file():
                    missing_files.append(str(data_root / relative))
    report["missing_data_files"] = missing_files
    if missing_files:
        raise FileNotFoundError(f"Missing required dataset files: {missing_files}")
    if spec["accelerator"] != "cpu":
        import torch

        if spec["accelerator"] == "h100_80gb":
            configure_h100_math()
        report.update({
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu_count_visible": torch.cuda.device_count(),
            "gpus": [{
                "name": torch.cuda.get_device_properties(index).name,
                "memory_bytes": torch.cuda.get_device_properties(index).total_memory,
            } for index in range(torch.cuda.device_count())],
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        })
        x = torch.ones((16, 16), device="cuda", dtype=(
            torch.bfloat16 if spec["accelerator"] == "h100_80gb" else torch.float16
        ))
        report["tiny_forward_sum"] = float((x @ x).float().sum().cpu())
        names = [gpu["name"].lower() for gpu in report["gpus"]]
        if spec["accelerator"] == "kaggle_t4x2" and (
            len(names) != 2 or any("t4" not in name for name in names)
        ):
            raise RuntimeError(f"Expected two visible T4 GPUs, found {report['gpus']}")
        if spec["accelerator"] == "h100_80gb" and (
            len(names) != 1 or "h100" not in names[0]
            or report["gpus"][0]["memory_bytes"] < 75 * 1024**3
        ):
            raise RuntimeError(f"Expected one visible H100 80GB, found {report['gpus']}")
        if spec["accelerator"] == "h100_80gb" and not report["bf16_supported"]:
            raise RuntimeError("Visible H100 does not report BF16 support")
        if spec["accelerator"] == "h100_80gb":
            report["model_preflight"] = _model_preflight(config, profile)
    atomic_json(output / "doctor.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="CrossFM-Align full-paper runner")
    parser.add_argument("command", choices=("plan", "doctor", "run-worker", "aggregate"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/fullpaper"))
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--stage")
    parser.add_argument("--wheel-sha256", default="local")
    args = parser.parse_args()
    config = _materialize_source(load_protocol(args.config))
    tasks = tasks_for_profile(config, args.profile)
    if args.stage:
        tasks = [task for task in tasks if task.stage == args.stage]
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "plan":
        shards = balanced_shards(tasks, args.world_size)
        value = {
            "profile": args.profile, "tasks": len(tasks),
            "stages": sorted({task.stage for task in tasks}),
            "shards": [[asdict(task) for task in shard] for shard in shards],
        }
        atomic_json(args.output / "task_plan.json", value)
        print(json.dumps({"profile": args.profile, "tasks": len(tasks)}, indent=2))
        return
    if args.command == "doctor":
        print(json.dumps(doctor(config, args.profile, args.output, args.data_root), indent=2))
        return
    if args.command == "aggregate":
        print(json.dumps(aggregate_run(
            config=config, profile=args.profile,
            tasks=tasks_for_profile(config, args.profile), output=args.output,
        ), indent=2))
        return
    shard = balanced_shards(tasks, args.world_size)[args.rank]
    assignment = {
        "rank": args.rank, "world_size": args.world_size, "stage": args.stage,
        "task_ids": [task.task_id for task in shard], "estimated_cost": sum(task.cost for task in shard),
    }
    atomic_json(args.output / "assignments" / f"{args.stage or 'all'}_rank{args.rank}.json", assignment)
    engine = FullPaperEngine(
        config, args.output, args.data_root,
        _device(config["profiles"][args.profile]["accelerator"]),
    )
    counts = {"complete": 0, "reused": 0}
    for task in shard:
        status = execute_task(engine, task, args.wheel_sha256)
        counts[status] += 1
        print(json.dumps({"task_id": task.task_id, "status": status}), flush=True)
    atomic_json(args.output / "workers" / f"{args.stage or 'all'}_rank{args.rank}.json", {
        "status": "complete", **assignment, **counts,
    })


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({
            "status": "failed", "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(),
        }, indent=2), file=sys.stderr)
        raise
