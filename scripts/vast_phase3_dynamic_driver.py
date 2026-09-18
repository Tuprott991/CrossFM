from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback


BUNDLE = Path(os.environ["CROSSFM_BUNDLE"])
WORKING = Path(os.environ["CROSSFM_WORKING"])
CHECKPOINTS = Path(os.environ["CROSSFM_SOURCE_CHECKPOINTS"])
TOP_SUMMARY = WORKING / "crossfm_phase3_dynamic_summary.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(command: list[str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    manifest = json.loads((BUNDLE / "bundle_manifest.json").read_text(encoding="utf-8"))
    wheel, config = BUNDLE / manifest["wheel"]["name"], BUNDLE / manifest["config"]["name"]
    if sha256(wheel) != manifest["wheel"]["sha256"] or sha256(config) != manifest["config"]["sha256"]:
        raise RuntimeError("Immutable dynamic bundle hash validation failed")
    run([
        sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
        "numpy==1.26.4", "pandas==2.2.3", "PyYAML==6.0.2", "scikit-learn==1.6.1",
        "tabicl==2.2.0", "transformers==4.57.6", "huggingface-hub==0.36.0", "safetensors==0.7.0",
    ])
    run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", str(wheel)])

    import torch
    import yaml
    from huggingface_hub import hf_hub_download, snapshot_download

    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    gpu_info = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        gpu_info.append({"index": index, "name": properties.name, "memory_bytes": properties.total_memory})
    if len(gpu_info) != 2 or any("RTX 5090" not in gpu["name"] for gpu in gpu_info):
        raise RuntimeError(f"Expected two RTX 5090 GPUs, found {gpu_info}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Dynamic protocol requires CUDA bfloat16 support")
    snapshot_download(repo_id=cfg["models"]["llm"]["id"], revision=cfg["models"]["llm"]["revision"])
    hf_hub_download(
        repo_id=cfg["models"]["specialist"]["id"], revision=cfg["models"]["specialist"]["revision"],
        filename=cfg["models"]["specialist"]["checkpoint"],
    )
    run_dir = WORKING / cfg["experiment"]["protocol_id"]
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=False)
    atomic_json(run_dir / "doctor.json", {
        "timestamp": datetime.now(timezone.utc).isoformat(), "gpus": gpu_info,
        "torch": torch.__version__, "cuda": torch.version.cuda, "bf16": torch.cuda.is_bf16_supported(),
        "tf32_matmul": True, "matmul_precision": "high", "disk_free_bytes": shutil.disk_usage(WORKING).free,
        "wheel_sha256": manifest["wheel"]["sha256"], "config_sha256": manifest["config"]["sha256"],
        "cache_policy": "static tokenization/embeddings/schema/initial-TFM plus per-path prefix KV; dynamic round TFM outputs uncached",
    })
    processes, handles = [], []
    try:
        for rank in range(2):
            environment = os.environ.copy()
            threads = str(cfg["runtime"]["cpu_threads_per_worker"])
            environment.update({
                "CUDA_VISIBLE_DEVICES": str(rank), "OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads,
                "OPENBLAS_NUM_THREADS": threads, "NUMEXPR_NUM_THREADS": threads,
                "TOKENIZERS_PARALLELISM": "false", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:128",
                "HF_HUB_DISABLE_PROGRESS_BARS": "1", "HF_XET_HIGH_PERFORMANCE": "1",
            })
            handle = (logs / f"worker_{rank}.log").open("w", encoding="utf-8")
            handles.append(handle)
            command = [
                sys.executable, "-m", "crossfm.phase3_dynamic_worker", "--config", str(config),
                "--output", str(run_dir), "--rank", str(rank), "--world-size", "2",
                "--wheel", str(wheel), "--checkpoint-root", str(CHECKPOINTS),
            ]
            processes.append(subprocess.Popen(command, env=environment, stdout=handle, stderr=subprocess.STDOUT))
        codes = [process.wait() for process in processes]
    finally:
        for handle in handles:
            handle.close()
    if codes != [0, 0]:
        tails = {str(rank): (logs / f"worker_{rank}.log").read_text(errors="replace").splitlines()[-100:] for rank in range(2)}
        raise RuntimeError(f"Dynamic worker failure codes={codes} tails={json.dumps(tails)}")
    run([
        sys.executable, "-m", "crossfm.phase3_dynamic_aggregate", "--config", str(config),
        "--output", str(run_dir), "--wheel", str(wheel),
    ])
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    summary.update({"run_directory": str(run_dir), "doctor": json.loads((run_dir / "doctor.json").read_text())})
    atomic_json(TOP_SUMMARY, summary)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        failure = {
            "status": "failed", "timestamp": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(),
        }
        atomic_json(TOP_SUMMARY, failure)
        raise
