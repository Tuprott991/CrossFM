from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    root = args.bundle.resolve()
    dataset, kernel = root / "dataset", root / "kernel"
    manifest = json.loads((dataset / "bundle_manifest.json").read_text(encoding="utf-8"))
    dataset_metadata = json.loads((dataset / "dataset-metadata.json").read_text(encoding="utf-8-sig"))
    kernel_metadata = json.loads((kernel / "kernel-metadata.json").read_text(encoding="utf-8-sig"))
    for kind in ("wheel", "config"):
        item = manifest[kind]
        path = dataset / item["name"]
        assert path.is_file() and path.stat().st_size == item["bytes"]
        assert sha256(path) == item["sha256"]
    config = yaml.safe_load((dataset / manifest["config"]["name"]).read_text(encoding="utf-8"))
    assert dataset_metadata["isPrivate"] is True
    assert kernel_metadata["is_private"] is True and kernel_metadata["enable_gpu"] is True
    assert kernel_metadata["machine_shape"] == "NvidiaTeslaT4"
    assert kernel_metadata["dataset_sources"] == [dataset_metadata["id"]]
    assert config["runtime"]["expected_gpus"] == 2
    assert config["source"]["dirty"] is False and config["source"]["git_commit"] != "BUILD_TIME"
    if config["experiment"]["protocol_id"].startswith("crossfm-phase2"):
        assert config["experiment"]["classification"].startswith("exploratory")
        expected_methods = {
            "llm_only", "tabicl_only", "prediction_ensemble", "llm_to_tfm",
            "tfm_to_llm", "textual_tool", "llm_to_tfm_compute_matched",
        }
        if "C2" in config["experiment"]["regimes"]:
            expected_methods.add("adaptive_diagnostic_oracle")
            assert set(config["gate"]) == {
                "a_margin", "b_margin", "oracle_min", "one_call_max", "three_call_max", "all_baseline_max",
            }
        if "phase2.75" in config["experiment"]["protocol_id"]:
            expected_methods.update({"hard_routing_residual", "soft_routing_residual"})
            assert config["routing"]["states"] == 16
            assert config["routing"]["representation"] == "continuous_posterior"
            assert set(config["preservation_gate"]) == {"a_tolerance", "b_tolerance", "soft_tolerance"}
        assert set(config["experiment"]["methods"]) == expected_methods
        assert [config["data"][name]["alias_split"] for name in ("train", "validation", "test")] == [
            "train", "validation", "test",
        ]
        assert [config["data"][name]["mechanism_split"] for name in ("train", "validation", "test")] == [
            "train", "validation", "test",
        ]
        episode_queries = sum(
            int(config["data"][name]["episodes_per_regime"])
            * int(config["data"][name]["queries_per_episode"])
            * len(config["experiment"]["regimes"])
            for name in ("train", "validation")
        )
        assert episode_queries < 100_000
        assert int(config["training"]["max_trainable_params"]) <= 5_000_000
    if config["experiment"]["protocol_id"].startswith("crossfm-phase3"):
        assert config["experiment"]["classification"].startswith("exploratory")
        assert set(config["experiment"]["methods"]) == {
            "crossfm_1", "crossfm_r2", "crossfm_r2_zero", "crossfm_r2_shuffle",
        }
        assert config["routing"] == {
            "fallback_threshold": 0.10, "states": 16,
            "representation": "continuous_posterior", "residual_bypass": "exact",
        }
        assert int(config["training"]["max_trainable_params"]) <= 5_000_000
        assert int(config["training"]["max_rounds"]) == 2
        assert config["runtime"]["cache"] == "in_memory_gpu_after_backbone_unload"
    assert (kernel / kernel_metadata["code_file"]).is_file()
    print(json.dumps({
        "status": "valid", "bundle": str(root), "protocol_id": config["experiment"]["protocol_id"],
        "wheel_sha256": manifest["wheel"]["sha256"], "config_sha256": manifest["config"]["sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
