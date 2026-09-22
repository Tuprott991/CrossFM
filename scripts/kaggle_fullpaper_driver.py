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


PROFILE = "__PROFILE__"
BUNDLE_OWNER = "__BUNDLE_OWNER__"
BUNDLE_SLUG = "__BUNDLE_SLUG__"
DATA_OWNER = "__DATA_OWNER__"
DATA_SLUG = "__DATA_SLUG__"
WORKING = Path(os.environ.get("CROSSFM_WORKING", "/kaggle/working"))
TOP_SUMMARY = WORKING / f"crossfm_{PROFILE}_summary.json"


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_source(owner: str, slug: str, required: str) -> Path:
    override = os.environ.get(f"CROSSFM_{required.upper()}_ROOT")
    candidates = ([Path(override)] if override else []) + [
        Path(f"/kaggle/input/{slug}"), Path(f"/kaggle/input/datasets/{owner}/{slug}"),
    ]
    matches = [path for path in candidates if (path / required).is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {owner}/{slug} source containing {required}; found {matches}")
    return matches[0]


def run_checked(command: list[str], **kwargs):
    print("RUN", " ".join(command), flush=True)
    return subprocess.run(command, check=True, **kwargs)


def launch_stage(
    stage: str, config: Path, run_dir: Path, data_root: Path, wheel_hash: str,
) -> None:
    logs = run_dir / "logs"; logs.mkdir(parents=True, exist_ok=True)
    processes, handles = [], []
    try:
        for rank in range(2):
            environment = os.environ.copy()
            environment.update({
                "CUDA_VISIBLE_DEVICES": str(rank), "LOCAL_RANK": "0", "RANK": str(rank),
                "WORLD_SIZE": "2", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                "OPENBLAS_NUM_THREADS": "2", "NUMEXPR_NUM_THREADS": "2",
                "TOKENIZERS_PARALLELISM": "false", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:128",
                "HF_HUB_DISABLE_PROGRESS_BARS": "1", "HF_XET_HIGH_PERFORMANCE": "1",
            })
            handle = (logs / f"{stage}_rank{rank}.log").open("w", encoding="utf-8")
            handles.append(handle)
            command = [
                sys.executable, "-m", "crossfm.fullpaper.cli", "run-worker",
                "--config", str(config), "--profile", PROFILE, "--output", str(run_dir),
                "--data-root", str(data_root), "--rank", str(rank), "--world-size", "2",
                "--stage", stage, "--wheel-sha256", wheel_hash,
            ]
            processes.append(subprocess.Popen(
                command, env=environment, stdout=handle, stderr=subprocess.STDOUT, text=True,
            ))
        codes = [process.wait() for process in processes]
    finally:
        for handle in handles:
            handle.close()
    if codes != [0, 0]:
        tails = {
            str(rank): (logs / f"{stage}_rank{rank}.log").read_text(
                encoding="utf-8", errors="replace",
            ).splitlines()[-120:]
            for rank in range(2)
        }
        raise RuntimeError(f"Stage {stage} failed exit_codes={codes} tails={json.dumps(tails)}")


def main() -> None:
    bundle = resolve_source(BUNDLE_OWNER, BUNDLE_SLUG, "bundle_manifest.json")
    manifest = json.loads((bundle / "bundle_manifest.json").read_text())
    wheel = bundle / manifest["wheel"]["name"]
    config = bundle / manifest["config"]["name"]
    for name, path in (("wheel", wheel), ("config", config)):
        if not path.is_file() or sha256(path) != manifest[name]["sha256"]:
            raise RuntimeError(f"Immutable {name} hash mismatch: {path}")
    data_root_override = os.environ.get("CROSSFM_DATA_ROOT")
    if data_root_override:
        data_root = Path(data_root_override)
    else:
        candidates = [
            Path(f"/kaggle/input/{DATA_SLUG}"),
            Path(f"/kaggle/input/datasets/{DATA_OWNER}/{DATA_SLUG}"),
        ]
        matches = [path for path in candidates if path.is_dir()]
        data_root = matches[0] if len(matches) == 1 else Path("/kaggle/working/empty-data-root")
        data_root.mkdir(parents=True, exist_ok=True)
    dependencies = manifest["dependencies"]
    run_checked([
        sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
        *dependencies,
    ])
    run_checked([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", str(wheel)])
    import yaml
    frozen_config = yaml.safe_load(config.read_text(encoding="utf-8"))
    profile_methods = set(frozen_config["profiles"][PROFILE]["methods"])
    if any("tabpfn" in method for method in profile_methods) and not os.environ.get("TABPFN_TOKEN"):
        try:
            from kaggle_secrets import UserSecretsClient

            os.environ["TABPFN_TOKEN"] = UserSecretsClient().get_secret("TABPFN_TOKEN")
        except Exception as exc:
            raise RuntimeError(
                "TabPFN-3.5 profile requires the private Kaggle secret TABPFN_TOKEN "
                "after accepting the official model license"
            ) from exc
    run_dir = WORKING / manifest["run_id"]
    run_dir.mkdir(parents=True, exist_ok=False)
    run_checked([
        sys.executable, "-m", "crossfm.fullpaper.cli", "plan", "--config", str(config),
        "--profile", PROFILE, "--output", str(run_dir), "--data-root", str(data_root),
        "--world-size", "2",
    ])
    run_checked([
        sys.executable, "-m", "crossfm.fullpaper.cli", "doctor", "--config", str(config),
        "--profile", PROFILE, "--output", str(run_dir), "--data-root", str(data_root),
    ])
    for stage in ("response_bank", "llm_cache", "evaluate"):
        launch_stage(stage, config, run_dir, data_root, manifest["wheel"]["sha256"])
    run_checked([
        sys.executable, "-m", "crossfm.fullpaper.cli", "aggregate", "--config", str(config),
        "--profile", PROFILE, "--output", str(run_dir), "--data-root", str(data_root),
    ])
    summary = json.loads((run_dir / "summary.json").read_text())
    summary.update({
        "kernel_status": "complete", "run_directory": str(run_dir),
        "wheel_sha256": manifest["wheel"]["sha256"],
        "config_sha256": manifest["config"]["sha256"],
    })
    atomic_json(TOP_SUMMARY, summary)
    if (run_dir / "results.csv").is_file():
        shutil.copy2(run_dir / "results.csv", WORKING / f"crossfm_{PROFILE}_results.csv")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        failure = {
            "status": "failed", "profile": PROFILE,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        atomic_json(TOP_SUMMARY, failure)
        print(json.dumps(failure, indent=2), file=sys.stderr, flush=True)
        raise
