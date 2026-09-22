from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from crossfm.fullpaper.artifacts import (
    atomic_json, digest_json, reusable_record, write_task_record,
)
from crossfm.fullpaper.aggregate import _select_arplus
from crossfm.fullpaper.data import build_event_cohorts, load_retailrocket
from crossfm.fullpaper.protocol import balanced_shards, load_protocol, tasks_for_profile
from crossfm.fullpaper.routing import (
    ARPlusRouter, analytical_route, factorized_view_posterior, routing_features,
)
from crossfm.fullpaper.splits import assert_asof_integrity, temporal_split
from crossfm.fullpaper.synthetic_stress import StressCell, simulate_cell


ROOT = Path(__file__).resolve().parents[1]
FLEET_SPEC = importlib.util.spec_from_file_location(
    "run_kaggle_fleet", ROOT / "scripts" / "run_kaggle_fleet.py",
)
assert FLEET_SPEC and FLEET_SPEC.loader
FLEET = importlib.util.module_from_spec(FLEET_SPEC)
sys.modules[FLEET_SPEC.name] = FLEET
FLEET_SPEC.loader.exec_module(FLEET)


def test_fullpaper_protocol_is_frozen_exploratory_and_profiles_enumerate():
    config = load_protocol(ROOT / "configs" / "fullpaper.yaml")
    assert config["experiment"]["classification"] == "exploratory_non_confirmatory"
    for profile in config["profiles"]:
        tasks = tasks_for_profile(config, profile)
        assert tasks
        assert len({task.task_id for task in tasks}) == len(tasks)
        assert all(task.config_digest == digest_json(config) for task in tasks)
        stages = {task.stage for task in tasks}
        assert stages == {"response_bank", "llm_cache", "evaluate"}


def test_cost_balanced_sharding_is_deterministic_and_complete():
    config = load_protocol(ROOT / "configs" / "fullpaper.yaml")
    tasks = tasks_for_profile(config, "author_b_kaggle_frontier_temporal")
    left = balanced_shards(tasks, 2)
    right = balanced_shards(tasks, 2)
    assert [[task.task_id for task in shard] for shard in left] == [
        [task.task_id for task in shard] for shard in right
    ]
    flattened = [task.task_id for shard in left for task in shard]
    assert sorted(flattened) == sorted(task.task_id for task in tasks)
    assert set(task.task_id for task in left[0]).isdisjoint(task.task_id for task in left[1])


def test_analytical_router_has_exact_uniform_posterior_bypass():
    bank = np.asarray([[0.1, 0.9, 0.7], [0.8, 0.3, 0.6]])
    fallback = np.asarray([0.314159, 0.271828])
    posterior = np.full_like(bank, 1 / bank.shape[1])
    result = analytical_route(bank, np.asarray([1.0, 0.0, -1.0]), posterior, fallback)
    assert np.array_equal(result.gate, np.zeros(2))
    assert np.array_equal(result.probability, fallback)


def test_factorized_posterior_is_soft_and_normalized():
    evidence = np.asarray([[0.1, 0.4, -0.2], [1.0, 1.0, 1.0]])
    posterior = factorized_view_posterior(evidence, temperature=0.5)
    assert posterior.shape == evidence.shape
    assert np.allclose(posterior.sum(1), 1.0)
    assert np.all((posterior > 0) & (posterior < 1))
    assert np.allclose(posterior[1], np.full(3, 1 / 3))


def test_arplus_zero_initialization_and_bypass_are_exact():
    bank = np.asarray([[0.1, 0.9, 0.6], [0.2, 0.7, 0.8]], dtype=np.float32)
    posterior = np.asarray([[0.7, 0.2, 0.1], [1 / 3, 1 / 3, 1 / 3]], dtype=np.float32)
    fallback = np.asarray([0.4, 0.123456], dtype=np.float32)
    semantic = np.asarray([0.3, 0.1, -0.1], dtype=np.float32)
    analytical = analytical_route(bank, semantic, posterior, fallback)
    router = ARPlusRouter(feature_dim=5, hidden_dim=8, seed=7)
    probability, traces = router.predict(
        analytical_logits=analytical.analytical_logits,
        features=routing_features(bank, semantic, posterior), response_bank=bank,
        fallback_probability=fallback, gate=analytical.gate,
    )
    assert np.allclose(probability[0], analytical.probability[0], atol=1e-6)
    assert probability[1] == fallback[1]
    assert np.array_equal(traces["correction"], np.zeros_like(traces["correction"]))
    assert router.trainable_params < 5000


def test_temporal_cohort_builder_uses_only_pre_cutoff_events():
    customers, timestamps, values = [], [], []
    start = pd.Timestamp("2024-01-01", tz="UTC")
    for customer in ("a", "b", "c"):
        for day in range(0, 180, 7):
            customers.append(customer); timestamps.append(start + pd.Timedelta(days=day)); values.append(1.0)
    events = pd.DataFrame({
        "customer": customers, "timestamp": timestamps, "value": values,
        "event": "purchase", "item": "sku",
    })
    cohorts = build_event_cohorts(
        events, customer_column="customer", timestamp_column="timestamp",
        cutoffs=[start + pd.Timedelta(days=100), start + pd.Timedelta(days=130)],
        history_days=90, horizon_days=30, value_column="value", item_column="item",
        event_type_column="event", qualifying_types={"purchase"},
    )
    assert (cohorts["max_feature_time"] < cohorts["cutoff"]).all()
    assert (cohorts["label_window_end"] == cohorts["cutoff"] + pd.Timedelta(days=30)).all()


def test_retailrocket_short_timeline_supports_embargoed_three_way_split(tmp_path: Path):
    rows = []
    start = pd.Timestamp("2024-01-01", tz="UTC")
    for customer in range(12):
        for day in range(0, 138, 3):
            rows.append({
                "visitorid": customer,
                "timestamp": int((start + pd.Timedelta(days=day)).timestamp() * 1000),
                "event": "view",
                "itemid": customer * 1000 + day,
            })
    pd.DataFrame(rows).to_csv(tmp_path / "events.csv", index=False)
    spec = {
        "files": {"events": "events.csv"}, "target_mode": "engagement",
        "history_days": 90, "cutoff_warmup_days": 14, "cutoff_frequency_days": 7,
        "train_fraction": 0.20, "validation_fraction": 0.40, "embargo_days": 30,
        "task_description": "Predict lapse.",
    }
    bundle = load_retailrocket("retailrocket", spec, tmp_path)
    assert all(len(bundle.splits[name]) for name in ("train", "validation", "test"))
    assert bundle.timestamps.iloc[bundle.splits["validation"]].min() >= (
        bundle.timestamps.iloc[bundle.splits["train"]].max() + pd.Timedelta(days=30)
    )
    assert bundle.timestamps.iloc[bundle.splits["test"]].min() >= (
        bundle.timestamps.iloc[bundle.splits["validation"]].max() + pd.Timedelta(days=30)
    )


def test_asof_audit_rejects_post_cutoff_feature():
    frame = pd.DataFrame({
        "cutoff": pd.to_datetime(["2024-01-01"], utc=True),
        "feature_time": pd.to_datetime(["2024-01-02"], utc=True),
    })
    with pytest.raises(ValueError, match="leakage"):
        assert_asof_integrity(
            frame, cutoff_column="cutoff", feature_timestamp_columns=["feature_time"],
            horizon_days=30,
        )


def test_temporal_split_has_embargo_gaps():
    timestamps = pd.date_range("2020-01-01", periods=30, freq="15D", tz="UTC")
    split = temporal_split(timestamps, embargo_days=30)
    assert split.validation_start >= split.train_end + pd.Timedelta(days=30)
    assert split.test_start >= split.validation_end + pd.Timedelta(days=30)
    assert set(split.train).isdisjoint(split.validation)
    assert set(split.validation).isdisjoint(split.test)


def test_atomic_record_reuse_requires_artifact_checksum(tmp_path: Path):
    artifact = tmp_path / "arrays" / "prediction.bin"
    artifact.parent.mkdir(); artifact.write_bytes(b"valid")
    record = write_task_record(
        tmp_path, task_id="task", protocol_id="protocol", config_digest="digest",
        payload={"metric": 1.0}, arrays_path=artifact,
    )
    assert reusable_record(
        record, task_id="task", protocol_id="protocol", config_digest="digest",
    )
    artifact.write_bytes(b"corrupt")
    assert not reusable_record(
        record, task_id="task", protocol_id="protocol", config_digest="digest",
    )


def test_synthetic_preservation_and_compositional_state_count():
    preserved = simulate_cell(StressCell(
        suite="s5", seed=3, markers=4, episodes=64, structure=0.0,
    ))
    scaled = simulate_cell(StressCell(
        suite="s3", seed=5, markers=8, episodes=64, method="soft",
    ))
    assert preserved["mean_gate"] == 0.0
    assert preserved["exact_preservation"] is True
    assert scaled["state_count"] == 256
    assert scaled["posterior_bytes"] > preserved["posterior_bytes"]


def test_arplus_global_selection_uses_validation_not_test_metrics():
    common = {
        "dataset": "d", "seed": 1, "schema_condition": "canonical",
        "context_budget": "full",
    }
    rows = [
        {**common, "method": "crossfm_ar", "validation_log_loss": 0.50, "log_loss": 0.10},
        {**common, "method": "crossfm_arplus", "validation_log_loss": 0.48, "log_loss": 0.90},
    ]
    selection = _select_arplus(rows, {
        "selection": {"mean_logloss_gain": 0.01, "max_dataset_degradation": 0.005},
    })
    assert selection["arplus_selected"] is True


def test_kaggle_fleet_has_four_distinct_accounts_profiles_and_private_token_env(monkeypatch):
    assert {lane.account for lane in FLEET.LANES} == {1, 2, 3, 4}
    assert len({lane.profile for lane in FLEET.LANES}) == 4
    assert len({lane.kernel_slug for lane in FLEET.LANES}) == 4
    assert all(6 <= len(lane.title) <= 50 for lane in FLEET.LANES)
    assert all("_" not in FLEET._bundle_slug(lane.profile) for lane in FLEET.LANES)
    monkeypatch.setenv("KAGGLE_ACCESS_TOKEN_1", "must-not-propagate")
    monkeypatch.setenv("KAGGLE_ACCESS_TOKEN_4", "must-not-propagate")
    child = FLEET._safe_env("selected-token")
    assert child["KAGGLE_API_TOKEN"] == "selected-token"
    assert all(not key.startswith("KAGGLE_ACCESS_TOKEN_") for key in child)


def test_kaggle_remote_bundle_match_excludes_control_metadata(tmp_path: Path):
    (tmp_path / "dataset-metadata.json").write_text("control")
    (tmp_path / "wheel.whl").write_bytes(b"wheel")
    (tmp_path / "config.yaml").write_bytes(b"config")
    rows = [
        {"name": "wheel.whl", "size": 5},
        {"name": "config.yaml", "size": 6},
    ]
    assert FLEET._remote_bundle_matches(rows, tmp_path)
