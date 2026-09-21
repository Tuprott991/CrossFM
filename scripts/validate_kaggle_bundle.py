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
        assert config["routing"] == {
            "fallback_threshold": 0.10, "states": 16,
            "representation": "continuous_posterior", "residual_bypass": "exact",
        }
        assert int(config["training"]["max_trainable_params"]) <= 5_000_000
        if "dynamic-transfer" in config["experiment"]["protocol_id"]:
            assert set(config["experiment"]["methods"]) == {
                "dynamic_1", "dynamic_r2", "dynamic_r2_zero", "dynamic_r2_shuffle",
            }
            assert config["runtime"]["torch_dtype"] == "bfloat16"
            assert config["runtime"]["dynamic_cache"] == "disabled_for_round_outputs"
            assert len(config["source_checkpoint"]["sha256"]) == 2
        else:
            assert set(config["experiment"]["methods"]) == {
                "crossfm_1", "crossfm_r2", "crossfm_r2_zero", "crossfm_r2_shuffle",
            }
            assert int(config["training"]["max_rounds"]) == 2
            assert config["runtime"]["cache"] == "in_memory_gpu_after_backbone_unload"
    if config["experiment"]["protocol_id"].startswith("crossfm-phase4"):
        assert config["experiment"]["classification"].startswith("exploratory")
        assert set(config["experiment"]["methods"]) == {
            "crossfm_r1_shared", "crossfm_r2_shared", "crossfm_r3_corrective",
            "crossfm_r3_zero_t2l", "crossfm_r3_shuffle_t2l", "crossfm_r3_t2l_only",
            "crossfm_r3_l2t_only_compute_matched", "crossfm_r3_stopgrad_l2t",
            "crossfm_r3_stopgrad_t2l", "crossfm_r3_random_bridge",
        }
        assert int(config["training"]["max_rounds"]) == 3
        assert int(config["training"]["max_trainable_params"]) <= 5_000_000
        assert config["runtime"]["cache"] == "single_shared_in_memory_gpu_backbone_cache"
        assert config["round3"]["behavior"] == "corrective_residual_update"
        assert config["round3"]["label_access_at_inference"] is False
        assert config["routing"] == {
            "fallback_threshold": 0.10, "states": 16,
            "representation": "continuous_posterior", "residual_bypass": "exact",
        }
        if "smoke" not in config["experiment"]["protocol_id"]:
            assert len(config["experiment"]["test_seeds"]) == 5
            assert sum(int(value) for value in config["data"]["test"]["episodes"].values()) == 100
    if (config["experiment"]["protocol_id"].startswith("crossfm-phase5")
            and not config["experiment"]["protocol_id"].startswith(("crossfm-phase5.1", "crossfm-phase5.2"))):
        assert config["experiment"]["classification"] == "exploratory_non_confirmatory"
        assert set(config["experiment"]["methods"]) == {
            "analytic_router", "semantic_cosine_router", "prior_start_r1", "prior_start_r2",
            "prior_start_r3", "message_only_r1", "message_only_r2", "message_only_r3",
            "message_only_r2_zero_t2l", "message_only_r2_shuffle_t2l",
            "message_only_r2_zero_l2t", "l2t_only_compute_matched", "random_bridge_r2",
        }
        assert len(config["experiment"]["test_seeds"]) == 5
        assert sum(int(value) for value in config["data"]["test"]["episodes"].values()) == 100
        assert len(set(config["experiment"]["test_seeds"]) & {10701, 10702, 10703, 10704, 10705}) == 0
        assert config["audit"]["auxiliary_prediction_head"] == "forbidden"
        assert config["audit"]["information_parity"] == "prior_available_before_round_1"
        assert int(config["training"]["max_rounds"]) == 3
        assert int(config["training"]["max_trainable_params"]) <= 5_000_000
        assert config["runtime"]["cache"] == "single_shared_in_memory_gpu_backbone_cache"
        assert config["routing"] == {
            "fallback_threshold": 0.10, "states": 16,
            "representation": "continuous_posterior", "residual_bypass": "exact",
        }
    if config["experiment"]["protocol_id"].startswith("crossfm-phase5.1"):
        assert config["experiment"]["classification"] == "exploratory_non_confirmatory"
        assert set(config["experiment"]["methods"]) == {
            "analytic_router", "structured_soft_r1", "structured_soft_r2", "structured_soft_r3",
            "structured_r2_zero_t2l", "structured_r2_shuffle_t2l", "structured_r2_hard_posterior",
            "structured_r2_uniform_posterior", "structured_r2_random_bridge",
        }
        assert len(config["experiment"]["test_seeds"]) == 5
        assert sum(int(value) for value in config["data"]["test"]["episodes"].values()) == 100
        assert not set(config["experiment"]["test_seeds"]) & {12701, 12702, 12703, 12704, 12705}
        assert config["audit"]["route_view_prior_access_for_learned_methods"] == "forbidden"
        assert config["audit"]["zero_message_identity"] == "bitwise"
        assert config["audit"]["no_new_information_depth_saturation"] == "bitwise"
        assert int(config["training"]["max_trainable_params"]) <= 5_000_000
        assert config["runtime"]["cache"] == "single_shared_in_memory_gpu_backbone_cache"
    if config["experiment"]["protocol_id"].startswith("crossfm-phase5.2"):
        assert config["experiment"]["classification"] == "exploratory_non_confirmatory"
        assert set(config["experiment"]["methods"]) == {
            "analytic_router", "arplus_temperature", "arplus_full", "arplus_route_only",
            "arplus_reliability_only", "arplus_shuffle_response", "arplus_no_anchor",
            "structured_smr_r2", "response_bank_oracle",
        }
        assert len(config["experiment"]["test_seeds"]) == 5
        assert sum(int(value) for value in config["data"]["test"]["episodes"].values()) == 100
        prior_seeds = {
            10701, 10702, 10703, 10704, 10705,
            12701, 12702, 12703, 12704, 12705,
            13701, 13702, 13703, 13704, 13705,
        }
        assert not set(config["experiment"]["test_seeds"]) & prior_seeds
        assert config["routing"] == {
            "fallback_threshold": 0.10,
            "states": 16,
            "representation": "continuous_posterior",
            "analytical_anchor": "immutable",
            "residual_bypass": "exact",
        }
        assert config["audit"]["auxiliary_prediction_head"] == "forbidden"
        assert config["audit"]["residual_initialization"] == "zero"
        assert config["audit"]["response_bank_oracle_label_access"] == "diagnostic_only"
        assert int(config["training"]["max_trainable_params"]) <= 5_000_000
        assert config["runtime"]["cache"] == "single_shared_in_memory_gpu_backbone_cache"
    assert (kernel / kernel_metadata["code_file"]).is_file()
    print(json.dumps({
        "status": "valid", "bundle": str(root), "protocol_id": config["experiment"]["protocol_id"],
        "wheel_sha256": manifest["wheel"]["sha256"], "config_sha256": manifest["config"]["sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
