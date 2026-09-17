from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback


INPUT = Path("/kaggle/input/crossfm-phase1-bundle")
WORKING = Path("/kaggle/working")
TOP_SUMMARY = WORKING / "crossfm_summary.json"


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checked_file(name: str, expected_sha: str) -> Path:
    path = INPUT / name
    if not path.is_file():
        raise FileNotFoundError(f"Required immutable input missing: {path}")
    actual = sha256(path)
    if actual != expected_sha:
        raise RuntimeError(f"Hash mismatch for {name}: expected {expected_sha}, got {actual}")
    return path


def run_checked(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("RUN", " ".join(command), flush=True)
    return subprocess.run(command, check=True, **kwargs)


def main() -> None:
    if not INPUT.is_dir():
        raise FileNotFoundError(f"Expected exactly one package source at {INPUT}")
    manifest_path = INPUT / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    wheel = checked_file(manifest["wheel"]["name"], manifest["wheel"]["sha256"])
    config = checked_file(manifest["config"]["name"], manifest["config"]["sha256"])

    run_checked([
        sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
        "tabicl==2.2.0", "transformers==4.57.6", "huggingface-hub==0.36.0", "safetensors==0.7.0",
    ])
    run_checked([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", str(wheel)])

    import torch
    import yaml
    from huggingface_hub import hf_hub_download, snapshot_download

    gpu_info = []
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        gpu_info.append({"index": idx, "name": props.name, "memory_bytes": props.total_memory})
    if len(gpu_info) != 2 or any("T4" not in gpu["name"] for gpu in gpu_info):
        raise RuntimeError(f"Expected exactly two T4 GPUs, found {gpu_info}")
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    if cfg["runtime"]["expected_gpus"] != 2:
        raise RuntimeError("Frozen config does not request two GPUs")

    # Resolve immutable model revisions before workers start; workers then reuse the shared cache.
    snapshot_download(repo_id=cfg["models"]["llm"]["id"], revision=cfg["models"]["llm"]["revision"])
    hf_hub_download(
        repo_id=cfg["models"]["specialist"]["id"], revision=cfg["models"]["specialist"]["revision"],
        filename=cfg["models"]["specialist"]["checkpoint"]
    )

    safe_protocol = re.sub(r"[^a-zA-Z0-9_.-]+", "-", cfg["experiment"]["protocol_id"])
    run_dir = WORKING / safe_protocol
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=False)
    doctor = {
        "timestamp": datetime.now(timezone.utc).isoformat(), "gpus": gpu_info,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "disk_free_bytes": shutil.disk_usage(WORKING).free,
        "wheel_sha256": manifest["wheel"]["sha256"], "config_sha256": manifest["config"]["sha256"],
    }
    atomic_json(run_dir / "doctor.json", doctor)

    processes = []
    handles = []
    try:
        for rank in range(2):
            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES": str(rank), "RANK": str(rank), "WORLD_SIZE": "2", "LOCAL_RANK": "0",
                "OMP_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            })
            handle = (logs / f"worker_{rank}.log").open("w", encoding="utf-8")
            handles.append(handle)
            command = [
                sys.executable, "-m", "crossfm.worker", "--config", str(config), "--output", str(run_dir),
                "--rank", str(rank), "--world-size", "2", "--wheel", str(wheel),
            ]
            processes.append(subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True))
        codes = [process.wait() for process in processes]
    finally:
        for handle in handles:
            handle.close()
    if codes != [0, 0]:
        tails = {}
        for rank in range(2):
            lines = (logs / f"worker_{rank}.log").read_text(encoding="utf-8", errors="replace").splitlines()
            tails[str(rank)] = lines[-80:]
        raise RuntimeError(f"Worker failure exit_codes={codes} tails={json.dumps(tails)}")

    run_checked([
        sys.executable, "-m", "crossfm.aggregate", "--config", str(config), "--output", str(run_dir), "--wheel", str(wheel)
    ])
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    summary.update({"kernel_status": "complete", "run_directory": str(run_dir), "doctor": doctor})
    atomic_json(TOP_SUMMARY, summary)
    shutil.copy2(run_dir / "experiments.csv", WORKING / "crossfm_experiments.csv")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        failure = {
            "status": "failed", "timestamp": datetime.now(timezone.utc).isoformat(),
            "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(),
        }
        atomic_json(TOP_SUMMARY, failure)
        print(json.dumps(failure, indent=2), file=sys.stderr, flush=True)
        raise
