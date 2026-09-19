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


INPUT_CANDIDATES = ((Path(os.environ["CROSSFM_BUNDLE"]),) if os.environ.get("CROSSFM_BUNDLE") else (
    Path("/kaggle/input/crossfm-phase51-bundle"),
    Path("/kaggle/input/datasets/tuktuai/crossfm-phase51-bundle"),
))
INPUT = INPUT_CANDIDATES[0]
WORKING = Path(os.environ.get("CROSSFM_WORKING", "/kaggle/working"))
TOP_SUMMARY = WORKING / "crossfm_phase51_summary.json"


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_file(name: str, expected_sha: str) -> Path:
    path = INPUT / name
    if not path.is_file() or sha256(path) != expected_sha:
        raise RuntimeError(f"Missing or invalid immutable input: {path}")
    return path


def run_checked(command: list[str], **kwargs):
    print("RUN", " ".join(command), flush=True)
    return subprocess.run(command, check=True, **kwargs)


def main() -> None:
    global INPUT
    mounted = [path for path in INPUT_CANDIDATES if (path / "bundle_manifest.json").is_file()]
    if len(mounted) != 1:
        raise FileNotFoundError(f"Expected one Phase 5.1 package source, found {mounted}")
    INPUT = mounted[0]
    manifest = json.loads((INPUT / "bundle_manifest.json").read_text(encoding="utf-8"))
    wheel = checked_file(manifest["wheel"]["name"], manifest["wheel"]["sha256"])
    config = checked_file(manifest["config"]["name"], manifest["config"]["sha256"])
    run_checked([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
                 "numpy==1.26.4", "pandas==2.2.3", "PyYAML==6.0.2", "scikit-learn==1.6.1",
                 "tabicl==2.2.0", "transformers==4.57.6", "huggingface-hub==0.36.0", "safetensors==0.7.0"])
    run_checked([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", str(wheel)])

    import torch
    import yaml
    from huggingface_hub import hf_hub_download, snapshot_download

    gpu_info = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        gpu_info.append({"index": index, "name": properties.name, "memory_bytes": properties.total_memory})
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    expected_count, expected_family = int(cfg["runtime"]["expected_gpus"]), str(cfg["runtime"]["gpu_family"])
    normalize = lambda name: " ".join(token for token in name.lower().split() if token not in {"nvidia", "corporation"})
    if len(gpu_info) != expected_count or any(normalize(expected_family) not in normalize(gpu["name"]) for gpu in gpu_info):
        raise RuntimeError(f"Expected {expected_count} x {expected_family}, found {gpu_info}")
    snapshot_download(repo_id=cfg["models"]["llm"]["id"], revision=cfg["models"]["llm"]["revision"])
    hf_hub_download(repo_id=cfg["models"]["specialist"]["id"], revision=cfg["models"]["specialist"]["revision"],
                    filename=cfg["models"]["specialist"]["checkpoint"])
    safe_protocol = re.sub(r"[^a-zA-Z0-9_.-]+", "-", cfg["experiment"]["protocol_id"])
    run_dir, logs = WORKING / safe_protocol, WORKING / safe_protocol / "logs"; logs.mkdir(parents=True, exist_ok=False)
    doctor = {"timestamp": datetime.now(timezone.utc).isoformat(), "gpus": gpu_info, "torch": torch.__version__,
              "cuda": torch.version.cuda, "disk_free_bytes": shutil.disk_usage(WORKING).free,
              "wheel_sha256": manifest["wheel"]["sha256"], "config_sha256": manifest["config"]["sha256"],
              "execution": "two isolated workers with one shared immutable cache per worker",
              "message_contract": "soft posterior plus codebook tokens; learned methods cannot access route_view_prior",
              "torch_dtype": cfg["runtime"]["torch_dtype"]}
    atomic_json(run_dir / "doctor.json", doctor)
    processes, handles = [], []
    try:
        for rank in range(2):
            environment = os.environ.copy(); threads = str(cfg["runtime"].get("cpu_threads_per_worker", 2))
            environment.update({"CUDA_VISIBLE_DEVICES": str(rank), "RANK": str(rank), "WORLD_SIZE": "2", "LOCAL_RANK": "0",
                                "OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads, "OPENBLAS_NUM_THREADS": threads,
                                "NUMEXPR_NUM_THREADS": threads, "TOKENIZERS_PARALLELISM": "false", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:128",
                                "HF_HUB_DISABLE_PROGRESS_BARS": "1", "HF_XET_HIGH_PERFORMANCE": "1"})
            handle = (logs / f"worker_{rank}.log").open("w", encoding="utf-8"); handles.append(handle)
            command = [sys.executable, "-m", "crossfm.phase51_worker", "--config", str(config), "--output", str(run_dir),
                       "--rank", str(rank), "--world-size", "2", "--wheel", str(wheel)]
            processes.append(subprocess.Popen(command, env=environment, stdout=handle, stderr=subprocess.STDOUT, text=True))
        codes = [process.wait() for process in processes]
    finally:
        for handle in handles:
            handle.close()
    if codes != [0, 0]:
        tails = {str(rank): (logs / f"worker_{rank}.log").read_text(encoding="utf-8", errors="replace").splitlines()[-100:]
                 for rank in range(2)}
        raise RuntimeError(f"Worker failure exit_codes={codes} tails={json.dumps(tails)}")
    run_checked([sys.executable, "-m", "crossfm.phase51_aggregate", "--config", str(config),
                 "--output", str(run_dir), "--wheel", str(wheel)])
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    summary.update({"kernel_status": "complete", "run_directory": str(run_dir), "doctor": doctor})
    atomic_json(TOP_SUMMARY, summary); shutil.copy2(run_dir / "experiments.csv", WORKING / "crossfm_phase51_experiments.csv")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        failure = {"status": "failed", "timestamp": datetime.now(timezone.utc).isoformat(),
                   "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
        atomic_json(TOP_SUMMARY, failure); print(json.dumps(failure, indent=2), file=sys.stderr, flush=True); raise
