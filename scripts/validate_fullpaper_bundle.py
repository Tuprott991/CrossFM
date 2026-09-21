from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from crossfm.fullpaper.artifacts import sha256_file
from crossfm.fullpaper.protocol import tasks_for_profile, validate_protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--profile", required=True)
    args = parser.parse_args()
    root = args.bundle.resolve()
    dataset, kernel = root / "dataset", root / "kernel"
    manifest = json.loads((dataset / "bundle_manifest.json").read_text())
    data_metadata = json.loads((dataset / "dataset-metadata.json").read_text(encoding="utf-8-sig"))
    kernel_metadata = json.loads((kernel / "kernel-metadata.json").read_text(encoding="utf-8-sig"))
    for kind in ("wheel", "config"):
        path = dataset / manifest[kind]["name"]
        assert path.is_file() and path.stat().st_size == manifest[kind]["bytes"]
        assert sha256_file(path) == manifest[kind]["sha256"]
    config = yaml.safe_load((dataset / manifest["config"]["name"]).read_text())
    validate_protocol(config)
    assert config["source"]["dirty"] is False
    assert config["source"]["git_commit"] != "BUILD_TIME"
    assert args.profile in config["profiles"]
    assert config["profiles"][args.profile]["accelerator"] == "kaggle_t4x2"
    assert data_metadata["isPrivate"] is True and kernel_metadata["is_private"] is True
    assert kernel_metadata["enable_gpu"] is True
    assert kernel_metadata["dataset_sources"][0] == data_metadata["id"]
    driver = kernel / kernel_metadata["code_file"]
    assert driver.is_file() and "__PROFILE__" not in driver.read_text()
    tasks = tasks_for_profile(config, args.profile)
    assert tasks and {task.stage for task in tasks} == {"response_bank", "llm_cache", "evaluate"}
    print(json.dumps({
        "status": "valid", "profile": args.profile, "tasks": len(tasks),
        "wheel_sha256": manifest["wheel"]["sha256"],
        "config_sha256": manifest["config"]["sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
