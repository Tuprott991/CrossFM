from __future__ import annotations

import json
import hashlib
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest

from crossfm.fullpaper.artifacts import (
    atomic_json, digest_json, reusable_record, write_task_record,
)
from crossfm.fullpaper.aggregate import _select_arplus
from crossfm.fullpaper.engine import (
    _group_cap, _latent_codebook_route, _router_split, _smr_refit_route,
)
from crossfm.fullpaper.data import (
    _binary_target, build_event_cohorts, load_beyondarena, load_manifest_table,
    load_retailrocket,
)
from crossfm.fullpaper.models import (
    _FROZEN_MODEL_CACHE, _evict_frozen_models, _sklearn_matrix, candidate_views,
    fit_predict_tabular, row_prompts,
)
from crossfm.fullpaper.protocol import (
    balanced_shards, load_protocol, tasks_for_profile, validate_protocol,
)
from crossfm.fullpaper.routing import (
    ARPlusRouter, analytical_route, corrective_route, factorized_view_posterior,
    routing_features, SoftLatentCodebookRouter,
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
    assert config["experiment"]["protocol_id"] == (
        "crossfm-align-fullpaper-exploratory-v12-runtime-aliases"
    )
    assert config["experiment"]["classification"] == "exploratory_non_confirmatory"
    assert "llm_to_tfm_compute_matched" not in config["methods"]
    assert "tfm_to_llm" not in config["methods"]
    assert "aliases" in config["profiles"]["author_a_h100_robustness"]["schema_conditions"]
    assert config["profiles"]["author_a_h100_primary"]["performs_global_model_selection"]
    assert config["profiles"]["author_a_h100_scale"]["selection_frozen_from"] == (
        "author_a_h100_primary"
    )
    assert "cache_llm_tool" in config["profiles"]["author_b_kaggle_d4"]["methods"]
    assert config["runtime"]["minimum_free_disk_gib"] == {
        "h100_80gb": 50, "kaggle_t4x2": 15, "cpu": 5,
    }
    assert "torch" not in config["runtime"]["expected_package_versions"]
    assert config["runtime"]["expected_accelerator_package_versions"] == {
        "h100_80gb": {"torch": "2.6.0"},
        "kaggle_t4x2": {"torch": "2.10.0"},
        "cpu": {},
    }
    d3 = config["profiles"]["author_b_kaggle_d3"]
    assert d3["seeds"] == [25101, 25102, 25103, 25104, 25105]
    assert {
        "crossfm_smr_refit", "crossfm_lci", "ablate_lci_single_state",
        "ablate_lci_no_semantic", "ablate_lci_hard",
        "ablate_lci_shuffle_codes", "ablate_lci_fixed16",
    }.issubset(d3["methods"])
    assert set(config["datasets"]["d4_iranian_churn"]["feature_aliases"]) == {
        "Call  Failure", "Complains", "Subscription  Length", "Charge  Amount",
        "Seconds of Use", "Frequency of use", "Frequency of SMS",
        "Distinct Called Numbers", "Age Group", "Tariff Plan", "Status", "Age",
        "Customer Value",
    }
    for profile in config["profiles"]:
        tasks = tasks_for_profile(config, profile)
        assert tasks
        assert len({task.task_id for task in tasks}) == len(tasks)
        assert all(task.config_digest == digest_json(config) for task in tasks)
        stages = {task.stage for task in tasks}
        assert stages == {"response_bank", "llm_cache", "evaluate"}


def test_protocol_rejects_missing_cache_producer():
    config = load_protocol(ROOT / "configs" / "fullpaper.yaml")
    config["profiles"]["author_b_kaggle_d4"]["methods"].remove("cache_llm_tool")
    with pytest.raises(ValueError, match="without required llm_cache producer"):
        tasks_for_profile(config, "author_b_kaggle_d4")


def test_protocol_rejects_alias_condition_without_validated_manifest_aliases():
    config = load_protocol(ROOT / "configs" / "fullpaper.yaml")
    del config["datasets"]["d4_iranian_churn"]["feature_aliases"]
    with pytest.raises(ValueError, match="without a validated feature_aliases mapping"):
        validate_protocol(config)


def test_binary_target_requires_and_preserves_explicit_positive_class():
    values = pd.Series(pd.Categorical(["No", "Yes", "No", "Yes"]))
    encoded = _binary_target(values, dataset_id="bank", positive_label="Yes")
    assert encoded.tolist() == [0, 1, 0, 1]
    with pytest.raises(ValueError, match="requires an explicit positive_label"):
        _binary_target(values, dataset_id="bank")


def test_h100_model_preflight_resolves_declared_model_families(monkeypatch, tmp_path: Path):
    from crossfm.fullpaper.cli import _model_preflight

    checkpoint = tmp_path / "tabpfn.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    class FakeTabICL: pass
    class FakeTabPFN: pass
    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(*_, **__): return types.SimpleNamespace()
    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*_, **__): return types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "tabicl", types.SimpleNamespace(TabICLClassifier=FakeTabICL))
    monkeypatch.setitem(sys.modules, "tabpfn", types.SimpleNamespace(TabPFNClassifier=FakeTabPFN))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **_: str(checkpoint))
    monkeypatch.setattr("transformers.AutoConfig", FakeAutoConfig)
    monkeypatch.setattr("transformers.AutoTokenizer", FakeAutoTokenizer)
    config = {
        "profiles": {"p": {"methods": ["t", "p", "l"]}},
        "methods": {
            "t": {"backend": "tabicl"},
            "p": {"backend": "tabpfn3", "params": {
                "hf_repo_id": "repo", "hf_filename": "file", "hf_revision": "rev",
                "hf_sha256": checksum,
            }},
            "l": {"model_id": "llm", "revision": "rev"},
        },
    }
    report = _model_preflight(config, "p")
    assert report["tabpfn_checkpoint_sha256"] == checksum
    assert report["llm_model_id"] == "llm"


def test_disk_preflight_threshold_is_accelerator_specific():
    from crossfm.fullpaper.cli import (
        _expected_package_versions, _minimum_free_disk_gib,
    )

    config = load_protocol(ROOT / "configs" / "fullpaper.yaml")
    assert _minimum_free_disk_gib(config, "kaggle_t4x2") == 15
    assert _minimum_free_disk_gib(config, "h100_80gb") == 50
    assert _minimum_free_disk_gib(config, "cpu") == 5
    assert _expected_package_versions(config, "kaggle_t4x2")["torch"] == "2.10.0"
    assert _expected_package_versions(config, "h100_80gb")["torch"] == "2.6.0"
    assert "torch" not in _expected_package_versions(config, "cpu")
    d3_packages = _expected_package_versions(
        config, "kaggle_t4x2", "author_b_kaggle_d3",
    )
    d4_packages = _expected_package_versions(
        config, "kaggle_t4x2", "author_b_kaggle_d4",
    )
    assert "tabpfn" not in d3_packages
    assert d4_packages["tabpfn"] == "9.0.0"


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


def test_grouped_router_and_evaluation_subsets_never_split_groups():
    indices = np.arange(40)
    groups = np.repeat(np.arange(10), 4)
    labels = np.tile([0, 1, 0, 1], 10)
    fit, router = _router_split(indices, labels, 7, groups)
    assert set(groups[fit]).isdisjoint(groups[router])
    capped = _group_cap(indices, labels, groups, 17, 9)
    for group in np.unique(groups[capped]):
        assert set(indices[groups == group]).issubset(set(capped))


def test_engine_caps_router_rows_without_changing_fit_context(tmp_path: Path):
    from crossfm.fullpaper.data import DatasetBundle
    from crossfm.fullpaper.engine import FullPaperEngine

    rows = 200
    bundle = DatasetBundle(
        "fixture", pd.DataFrame({"x": range(rows)}), pd.Series([0, 1] * (rows // 2)),
        {"train": np.arange(rows), "validation": np.arange(4), "test": np.arange(4)},
        {"x": "x"}, "Predict.", "checksum", "split",
    )
    engine = FullPaperEngine(
        {"runtime": {"max_router_rows": 24}}, tmp_path, tmp_path, "cpu",
    )
    fit, router = engine._fit_router_indices(bundle, np.arange(rows), 11)
    assert len(fit) == 160
    assert len(router) == 24
    assert set(fit).isdisjoint(router)


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


def test_corrective_third_beat_changes_attention_without_breaking_bypass():
    router_bank = np.asarray([[0.9, 0.2], [0.8, 0.7], [0.1, 0.8], [0.2, 0.6]])
    labels = np.asarray([1, 1, 0, 0])
    posterior = np.asarray([[0.8, 0.2]] * 4)
    fallback = np.asarray([0.5] * 4)
    round2_router = analytical_route(router_bank, np.zeros(2), posterior, fallback)
    target_bank = np.asarray([[0.9, 0.1], [0.2, 0.8]])
    target_posterior = np.asarray([[0.8, 0.2], [0.5, 0.5]])
    target_fallback = np.asarray([0.4, 0.314159])
    round2_target = analytical_route(
        target_bank, np.zeros(2), target_posterior, target_fallback,
    )
    round3 = corrective_route(
        round2_router, router_bank, labels, round2_target, target_bank, target_fallback,
    )
    assert not np.allclose(round3.weights[0], round2_target.weights[0])
    assert round3.gate[1] == 0
    assert round3.probability[1] == target_fallback[1]


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


def test_soft_latent_codebook_has_continuous_normalized_posterior():
    rng = np.random.default_rng(73)
    bank = np.clip(rng.normal(0.5, 0.2, size=(80, 3)), 0.01, 0.99)
    llm = np.clip(rng.normal(0.5, 0.2, size=80), 0.01, 0.99)
    labels = ((bank[:, 0] + bank[:, 1]) > 1.0).astype(np.int8)
    router = SoftLatentCodebookRouter(
        codebook_size=4, temperature=2.0, regularization_c=0.1, seed=7,
    )
    details = router.fit(
        response_bank=bank, llm_probability=llm, labels=labels,
        semantic_logits=np.asarray([0.4, 0.2, -0.1]),
    )
    probability, trace = router.predict(
        response_bank=bank[:12], llm_probability=llm[:12],
        semantic_logits=np.asarray([0.4, 0.2, -0.1]),
    )
    assert probability.shape == (12,)
    assert np.allclose(trace["posterior"].sum(axis=1), 1.0)
    assert np.all((trace["posterior"] > 0) & (trace["posterior"] < 1))
    assert details["code_usage_perplexity"] > 1.0


def test_single_state_lci_is_exact_refit_smr_and_preserves_residual_path():
    rng = np.random.default_rng(91)
    router_bank = rng.uniform(0.05, 0.95, size=(80, 3))
    validation_bank = rng.uniform(0.05, 0.95, size=(40, 3))
    test_bank = rng.uniform(0.05, 0.95, size=(30, 3))
    router_llm = rng.uniform(0.05, 0.95, size=80)
    validation_llm = rng.uniform(0.05, 0.95, size=40)
    test_llm = rng.uniform(0.05, 0.95, size=30)
    router_labels = (router_bank[:, 0] > 0.5).astype(np.int8)
    validation_labels = (validation_bank[:, 0] > 0.5).astype(np.int8)
    c_grid = [0.1, 1.0]
    smr_validation, smr_test, _ = _smr_refit_route(
        router_bank=router_bank, router_llm=router_llm,
        router_labels=router_labels, validation_bank=validation_bank,
        validation_llm=validation_llm, validation_labels=validation_labels,
        test_bank=test_bank, test_llm=test_llm, seed=17, c_grid=c_grid,
    )
    lci_validation, lci_test, details = _latent_codebook_route(
        router_bank=router_bank, router_llm=router_llm,
        router_labels=router_labels, validation_bank=validation_bank,
        validation_llm=validation_llm, validation_labels=validation_labels,
        test_bank=test_bank, test_llm=test_llm,
        semantic_logits=np.asarray([0.2, 0.1, -0.3]), seed=17,
        method={
            "codebook_sizes": [1], "fixed_codebook_size": 1,
            "temperatures": [1.0], "c_grid": c_grid,
        },
    )
    assert np.allclose(lci_validation, smr_validation)
    assert np.allclose(lci_test, smr_test)
    assert details["selected_codebook_size"] == 1
    assert details["residual_alpha"] == 0.0
    assert details["preserved_fraction"] == 1.0
    assert details["test_labels_used_for_selection"] is False


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


def test_sklearn_matrix_normalizes_nullable_extension_values():
    train = pd.DataFrame({
        "number": pd.Series([1, pd.NA, 3], dtype="Int64"),
        "category": pd.Series(["a", pd.NA, "b"], dtype="string"),
    })
    validation = pd.DataFrame({
        "number": pd.Series([pd.NA], dtype="Int64"),
        "category": pd.Series([pd.NA], dtype="string"),
    })
    test = pd.DataFrame({
        "number": pd.Series([2], dtype="Int64"),
        "category": pd.Series(["new"], dtype="string"),
    })
    matrices = _sklearn_matrix(train, validation, test)[:3]
    assert all(matrix.shape[0] == expected for matrix, expected in zip(matrices, (3, 1, 1)))


def test_tabicl_prediction_is_memory_bounded_by_chunks(monkeypatch):
    calls = []

    class FakeTabICL:
        def __init__(self, **_):
            pass

        def fit(self, _x, _y):
            return self

        def predict_proba(self, values):
            calls.append(len(values))
            return np.tile([0.4, 0.6], (len(values), 1))

    monkeypatch.setitem(sys.modules, "tabicl", types.SimpleNamespace(TabICLClassifier=FakeTabICL))
    _FROZEN_MODEL_CACHE.clear()
    fit_predict_tabular(
        "tabicl", pd.DataFrame({"x": range(20)}), np.asarray([0, 1] * 10),
        pd.DataFrame({"x": range(600)}), pd.DataFrame({"x": range(513)}),
        seed=7, device="cpu", params={"prediction_chunk_size": 256},
    )
    assert calls == [256, 256, 88, 256, 256, 1]


def test_tabpfn3_uses_revision_pinned_verified_huggingface_checkpoint(
    monkeypatch, tmp_path: Path,
):
    checkpoint = tmp_path / "tabpfn3.ckpt"
    checkpoint.write_bytes(b"official-checkpoint-fixture")
    expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    downloads = []
    constructors = []

    def fake_download(**kwargs):
        downloads.append(kwargs)
        return str(checkpoint)

    class FakeTabPFN:
        def __init__(self, **kwargs):
            constructors.append(kwargs)

        def fit(self, _x, _y):
            return self

        def predict_proba(self, values):
            return np.tile([0.3, 0.7], (len(values), 1))

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    monkeypatch.setitem(sys.modules, "tabpfn", types.SimpleNamespace(TabPFNClassifier=FakeTabPFN))
    _FROZEN_MODEL_CACHE.clear()
    result = fit_predict_tabular(
        "tabpfn3", pd.DataFrame({"x": range(10)}), np.asarray([0, 1] * 5),
        pd.DataFrame({"x": range(2)}), pd.DataFrame({"x": range(3)}),
        seed=11, device="cpu", params={
            "hf_repo_id": "Prior-Labs/tabpfn_3", "hf_revision": "pinned-revision",
            "hf_filename": "binary.ckpt", "hf_sha256": expected,
        },
    )
    assert downloads == [{
        "repo_id": "Prior-Labs/tabpfn_3", "filename": "binary.ckpt",
        "revision": "pinned-revision",
    }]
    assert constructors[0]["model_path"] == str(checkpoint)
    assert np.array_equal(result.test_probability, np.full(3, 0.7))


def test_heavy_model_cache_evicts_stale_seed_instances():
    _FROZEN_MODEL_CACHE.clear()
    keep = ("tabicl", "cuda:0", 2)
    _FROZEN_MODEL_CACHE[("tabicl", "cuda:0", 1)] = object()
    _FROZEN_MODEL_CACHE[keep] = object()
    _FROZEN_MODEL_CACHE[("causal_llm", "model")] = object()
    _evict_frozen_models("tabicl", keep)
    assert keep in _FROZEN_MODEL_CACHE
    assert ("tabicl", "cuda:0", 1) not in _FROZEN_MODEL_CACHE
    assert ("causal_llm", "model") in _FROZEN_MODEL_CACHE


def test_anonymized_prompts_do_not_leak_original_column_names():
    prompts = row_prompts(
        pd.DataFrame({"monthly_premium": [123], "income": [456]}),
        "Predict lapse.",
        {"monthly_premium": "x_1", "income": "x_2"},
        display_names={"monthly_premium": "x_1", "income": "x_2"},
    )
    assert "monthly_premium" not in prompts[0]
    assert "income" not in prompts[0]
    assert "x_1=123" in prompts[0] and "x_2=456" in prompts[0]


def test_all_numeric_table_does_not_duplicate_numeric_and_full_views():
    views = candidate_views(pd.DataFrame({"a": [1], "b": [2]}), {"views": {}})
    assert views == {"schema_group_a": ["a"], "full_table": ["a", "b"]}


def test_kkbox_manifest_loader_requires_audited_hash_and_customer_groups(tmp_path: Path):
    table = tmp_path / "cohort.csv"
    frame = pd.DataFrame({
        "customer_id": [f"c{i}" for i in range(6)],
        "split": ["train", "train", "validation", "validation", "test", "test"],
        "feature": range(6), "target": [0, 1, 0, 1, 0, 1],
    })
    frame.to_csv(table, index=False)
    manifest = {
        "schema_version": "crossfm-cohort-manifest-v1", "dataset_id": "d1_kkbox",
        "table_sha256": hashlib.sha256(table.read_bytes()).hexdigest(), "horizon_days": 30,
        "raw_file_sha256": {"raw.csv": "a" * 64},
        "label_reproduction_audit": "passed", "feature_asof_audit": "passed",
        "feature_aliases": {"feature": "account_signal"},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    bundle = load_manifest_table("d1_kkbox", {
        "files": {"table": "cohort.csv", "manifest": "manifest.json"},
        "audit_manifest_required": True, "target_column": "target",
        "group_column": "customer_id", "split_column": "split", "split": "temporal",
        "task_description": "Predict lapse.",
    }, tmp_path)
    assert bundle.groups.tolist() == [f"c{i}" for i in range(6)]
    assert list(bundle.frame) == ["feature"]
    assert bundle.feature_aliases == {"feature": "account_signal"}


def test_manifest_table_uses_configured_validated_aliases(tmp_path: Path):
    from crossfm.fullpaper.cli import _configured_alias_preflight

    table = tmp_path / "churn.csv"
    pd.DataFrame({
        "Call  Failure": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Complains": [0, 1] * 5,
        "Churn": [0, 1] * 5,
    }).to_csv(table, index=False)
    spec = {
        "files": {"table": "churn.csv"}, "target_column": "Churn",
        "split": "iid", "split_seed": 17, "task_description": "Predict churn.",
        "feature_aliases": {
            "Call  Failure": "unsuccessful_call_count",
            "Complains": "complaint_flag",
        },
    }
    bundle = load_manifest_table("d4", spec, tmp_path)
    assert bundle.feature_aliases == spec["feature_aliases"]
    report = _configured_alias_preflight({
        "experiment": {"schema_conditions": ["canonical"]},
        "profiles": {"p": {
            "datasets": ["d4"], "schema_conditions": ["canonical", "aliases"],
        }},
        "datasets": {"d4": {**spec, "loader": "manifest_table"}},
    }, "p", tmp_path)
    assert report["d4"] == {
        "feature_count": 2, "alias_count": 2, "validated_against_source": True,
    }
    broken = {**spec, "feature_aliases": {"Complains": "complaint_flag"}}
    with pytest.raises(ValueError, match="uniquely cover every feature column"):
        load_manifest_table("d4", broken, tmp_path)


def test_beyondarena_grouped_validation_preserves_whole_groups(monkeypatch):
    dataset = pd.DataFrame({
        "customer": ["a"] * 4 + ["b"] * 4 + ["c"] * 4,
        "feature": range(12), "target": [0, 1] * 6,
    })
    container = types.SimpleNamespace(
        dataset=dataset,
        task_metadata=types.SimpleNamespace(
            target_column_name="target", group_on="customer", time_on=None,
        ),
        experiment_metadata=types.SimpleNamespace(
            splits={0: {0: (np.arange(8), np.arange(8, 12))}},
        ),
        dataset_metadata=types.SimpleNamespace(unique_name="fixture"),
    )
    collection = types.SimpleNamespace(get_dataset=lambda _: container)
    package = types.ModuleType("data_foundry")
    collections = types.ModuleType("data_foundry.collections")
    collections.BEYOND_ARENA = collection
    monkeypatch.setitem(sys.modules, "data_foundry", package)
    monkeypatch.setitem(sys.modules, "data_foundry.collections", collections)
    bundle = load_beyondarena("grouped", {
        "unique_name": "fixture", "split": "grouped", "repeat": 0, "fold": 0,
        "task_description": "Predict.",
    }, Path("."))
    train_groups = set(bundle.groups.iloc[bundle.splits["train"]])
    validation_groups = set(bundle.groups.iloc[bundle.splits["validation"]])
    test_groups = set(bundle.groups.iloc[bundle.splits["test"]])
    assert train_groups.isdisjoint(validation_groups | test_groups)
    assert validation_groups.isdisjoint(test_groups)
    assert "customer" not in bundle.frame


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
