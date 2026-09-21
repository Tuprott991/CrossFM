import numpy as np

from crossfm.phase3 import CrossFMEpisodeCache, CrossFMLatentLoop, pack_cache
from crossfm.phase3_dynamic import DynamicEvidenceBridge
from crossfm.phase4 import CrossFMCorrectiveLoop
from crossfm.phase5 import CrossFMShortcutAudit, fixed_router_predict
from crossfm.phase51 import StructuredPosteriorBridge
from crossfm.phase52 import AnalyticallyAnchoredResidualRouter, response_bank_oracle


def _item(
    name: str, gate: float, dimension: int = 12, *, routed: bool | None = None,
) -> CrossFMEpisodeCache:
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    posterior = rng.dirichlet(np.ones(16)).astype(np.float32)
    return CrossFMEpisodeCache(
        episode_id=name,
        regime="C2" if gate else "A",
        task_embedding=rng.normal(size=dimension).astype(np.float32),
        view_embeddings=rng.normal(size=(5, dimension)).astype(np.float32),
        route_message=rng.normal(size=dimension).astype(np.float32),
        view_probabilities=rng.uniform(0.05, 0.95, size=(4, 5)).astype(np.float32),
        llm_probability=np.asarray([0.1, 0.3, 0.7, 0.9], dtype=np.float32),
        tfm_probability=rng.uniform(0.05, 0.95, size=4).astype(np.float32),
        gate=gate,
        labels=np.asarray([0, 0, 1, 1], dtype=np.float32),
        relevant_view=0,
        route_view_prior=np.full(5, 0.2, dtype=np.float32),
        routed=bool(gate) if routed is None else routed,
        code_posterior=posterior,
        codebook_embeddings=rng.normal(size=(16, dimension)).astype(np.float32),
    )


def test_crossfm_residual_bypass_is_exact():
    packed = pack_cache([_item("a", 0.0)], "cpu")
    model = CrossFMLatentLoop(12, 8, 2, "cpu", 7)
    probability, _ = model.module(packed, 2)
    assert np.array_equal(probability.detach().numpy()[0], packed["llm_probability"].numpy()[0])


def test_crossfm_nonrouted_path_preserves_specialist_exactly():
    packed = pack_cache([_item("b", 1.0, routed=False)], "cpu")
    model = CrossFMLatentLoop(12, 8, 2, "cpu", 7)
    probability, _ = model.module(packed, 2)
    assert np.array_equal(probability.detach().numpy()[0], packed["tfm_probability"].numpy()[0])


def test_soft_route_prior_only_changes_second_round_attention():
    packed = pack_cache([_item("c", 1.0, routed=True)], "cpu")
    model = CrossFMLatentLoop(12, 8, 2, "cpu", 13)
    _, round1_uniform = model.module(packed, 1)
    _, round2_uniform = model.module(packed, 2)
    packed["route_view_prior"].zero_()
    packed["route_view_prior"][0, :5] = packed["route_view_prior"].new_tensor(
        [0.80, 0.05, 0.05, 0.05, 0.05],
    )
    _, round1_routed = model.module(packed, 1)
    _, round2_routed = model.module(packed, 2)
    assert np.array_equal(round1_uniform.detach().numpy(), round1_routed.detach().numpy())
    assert not np.allclose(round2_uniform.detach().numpy(), round2_routed.detach().numpy())


def test_crossfm_rounds_share_parameters_and_messages_affect_output():
    packed = pack_cache([_item("c0", 0.9), _item("c1", 0.8)], "cpu")
    model = CrossFMLatentLoop(12, 8, 2, "cpu", 11)
    one, _ = model.module(packed, 1)
    two, weights = model.module(packed, 2)
    zero, _ = model.module(packed, 2, "zero")
    assert one.shape == two.shape == zero.shape == (2, 4)
    assert weights.shape == (2, 4, 17)
    assert np.allclose(weights.detach().numpy().sum(-1), 1.0)
    assert not np.allclose(two.detach().numpy(), zero.detach().numpy())
    assert model.trainable_params < 5_000_000


def test_crossfm_training_reports_gradient_and_message_effect():
    items = [_item(f"train-{index}", 0.9) for index in range(4)]
    packed = pack_cache(items, "cpu")
    model = CrossFMLatentLoop(12, 8, 2, "cpu", 17)
    report = model.fit(packed, packed, 2, epochs=2, batch_size=2, learning_rate=1e-3, patience=2)
    assert report["max_gradient_norm"] > 0
    assert report["train_zero_message_mean_absolute_delta"] > 0
    assert np.isfinite(report["validation_loss"])


def test_dynamic_evidence_bridge_fits_rich_hidden_messages_on_cpu():
    rng = np.random.default_rng(23)
    features = rng.normal(size=(24, 18)).astype(np.float32)
    targets = rng.normal(size=(24, 12)).astype(np.float32)
    bridge = DynamicEvidenceBridge(18, 12, 16, "cpu", 29)
    before = bridge.predict(features, False).numpy()
    report = bridge.fit(
        features, targets, epochs=4, batch_size=8, learning_rate=1e-2, use_bfloat16=False,
    )
    after = bridge.predict(features, False).numpy()
    assert bridge.trainable_params < 5_000_000
    assert report["train_mse"] < np.mean((before - targets) ** 2)
    assert not np.array_equal(before, after)


def test_corrective_round_three_uses_new_gate_and_residual_attention():
    packed = pack_cache([_item("p4-0", 0.9), _item("p4-1", 0.8)], "cpu")
    model = CrossFMCorrectiveLoop(12, 8, 3, "cpu", 31)
    two, trace_two = model.module(packed, 2)
    three, trace_three = model.module(packed, 3)
    assert two.shape == three.shape == (2, 4)
    assert len(trace_two["weights"]) == 2
    assert len(trace_three["weights"]) == 3
    assert trace_three["update_gate"][2].shape == (2, 4)
    assert not np.allclose(
        trace_three["weights"][1].detach().numpy(),
        trace_three["weights"][2].detach().numpy(),
    )
    assert model.trainable_params < 5_000_000


def test_corrective_loop_preserves_zero_gate_exactly():
    packed = pack_cache([_item("p4-a", 0.0)], "cpu")
    model = CrossFMCorrectiveLoop(12, 8, 3, "cpu", 37)
    probability, _ = model.module(packed, 3)
    assert np.array_equal(probability.detach().numpy()[0], packed["llm_probability"].numpy()[0])


def test_phase4_causal_modes_are_executable_and_distinct():
    packed = pack_cache([_item("p4-c0", 0.9), _item("p4-c1", 0.8)], "cpu")
    model = CrossFMCorrectiveLoop(12, 8, 3, "cpu", 41)
    normal, _ = model.module(packed, 3, "normal")
    zero, _ = model.module(packed, 3, "zero_t2l")
    shuffled, _ = model.module(packed, 3, "shuffle_t2l")
    t2l, _ = model.module(packed, 3, "t2l_only")
    l2t, _ = model.module(packed, 3, "l2t_only")
    assert all(value.shape == (2, 4) for value in (normal, zero, shuffled, t2l, l2t))
    assert not np.allclose(normal.detach().numpy(), zero.detach().numpy())
    assert not np.allclose(normal.detach().numpy(), shuffled.detach().numpy())


def test_corrective_loop_trains_with_shared_depth_supervision():
    items = [_item(f"p4-train-{index}", 0.9) for index in range(4)]
    packed = pack_cache(items, "cpu")
    model = CrossFMCorrectiveLoop(12, 8, 3, "cpu", 43)
    report = model.fit(
        packed, packed, rounds=3, epochs=2, batch_size=2,
        learning_rate=1e-3, patience=2, deep_supervision=True,
    )
    probability, labels, traces = model.predict(packed, 3, 2)
    assert report["max_gradient_norm"] > 0
    assert probability.shape == labels.shape == (16,)
    assert traces["weights"].shape == (4, 3, 4, 17)
    assert traces["update_gate"].shape == (4, 3, 4)


def test_phase5_message_path_cannot_read_direct_prior_in_round_one():
    import torch

    packed = pack_cache([_item("p5-c0", 0.9), _item("p5-c1", 0.8)], "cpu")
    model = CrossFMShortcutAudit(12, 8, "cpu", 47)
    message_before, _ = model.module(packed, 1, "message")
    packed["route_view_prior"].zero_(); packed["route_view_prior"][:, 0] = 1.0
    message_after, _ = model.module(packed, 1, "message")
    prior_after, _ = model.module(packed, 1, "prior")
    assert torch.equal(message_before, message_after)
    assert not torch.allclose(message_after, prior_after)


def test_phase5_exact_bypass_and_zero_parameter_analytic_router():
    import torch

    packed = pack_cache([_item("p5-a", 0.0, routed=False)], "cpu")
    with torch.inference_mode():
        probability, labels, traces = fixed_router_predict(packed, "analytic")
    assert np.array_equal(probability, packed["llm_probability"].numpy().reshape(-1))
    assert np.array_equal(labels, packed["labels"].numpy().reshape(-1))
    assert traces["weights"].shape == (1, 1, 4, 17)


def test_phase5_message_only_second_beat_uses_continuous_message():
    packed = pack_cache([_item("p5-m0", 0.9), _item("p5-m1", 0.8)], "cpu")
    model = CrossFMShortcutAudit(12, 8, "cpu", 53)
    one, trace_one = model.module(packed, 1, "message")
    two, trace_two = model.module(packed, 2, "message")
    zero, _ = model.module(packed, 2, "zero_t2l")
    assert one.shape == two.shape == zero.shape == (2, 4)
    assert len(trace_one["weights"]) == 1 and len(trace_two["weights"]) == 2
    assert not np.allclose(two.detach().numpy(), zero.detach().numpy())
    assert model.trainable_params < 5_000_000


def test_phase51_zero_message_and_no_new_information_are_bitwise_exact():
    import torch

    packed = pack_cache([_item("p51-c0", 0.9), _item("p51-c1", 0.8)], "cpu")
    model = StructuredPosteriorBridge(12, 8, "cpu", 59)
    r1, _ = model.module(packed, 1, "soft")
    zero_r2, _ = model.module(packed, 2, "zero")
    soft_r2, _ = model.module(packed, 2, "soft")
    soft_r3, _ = model.module(packed, 3, "soft")
    assert torch.equal(r1, zero_r2)
    assert torch.equal(soft_r2, soft_r3)


def test_phase51_learned_path_cannot_read_analytical_route_prior():
    import torch

    packed = pack_cache([_item("p51-prior0", 0.9), _item("p51-prior1", 0.8)], "cpu")
    model = StructuredPosteriorBridge(12, 8, "cpu", 61)
    before, _ = model.module(packed, 2, "soft")
    packed["route_view_prior"].zero_(); packed["route_view_prior"][:, 0] = 1.0
    after, _ = model.module(packed, 2, "soft")
    uniform, _ = model.module(packed, 2, "uniform")
    assert torch.equal(before, after)
    assert not torch.allclose(after, uniform)
    assert model.trainable_params < 5_000_000


def test_phase51_training_preserves_exact_identity_contract():
    items = [_item(f"p51-train-{index}", 0.9) for index in range(4)]
    packed = pack_cache(items, "cpu")
    model = StructuredPosteriorBridge(12, 8, "cpu", 67)
    report = model.fit(
        packed, packed, epochs=2, batch_size=2, learning_rate=1e-3, patience=2,
    )
    assert report["exact_zero_identity"] is True
    assert report["max_gradient_norm"] > 0
    assert np.isfinite(report["validation_loss"])


def test_phase52_zero_initialized_arplus_starts_at_analytical_router():
    import torch

    packed = pack_cache([_item("p52-c0", 0.9), _item("p52-c1", 0.8)], "cpu")
    model = AnalyticallyAnchoredResidualRouter(12, 8, 4, "cpu", 71)
    anchored, traces = model.module(packed, "full")
    with torch.inference_mode():
        analytical, _, _ = fixed_router_predict(packed, "analytic")
    assert np.allclose(anchored.detach().numpy().reshape(-1), analytical, atol=1e-6)
    assert float(traces["route_scale"][0]) == 0.0
    assert float(traces["reliability_scale"][0]) == 0.0
    assert model.trainable_params < 5_000_000


def test_phase52_exact_bypass_survives_learned_residuals():
    packed = pack_cache([_item("p52-a", 0.0, routed=False)], "cpu")
    model = AnalyticallyAnchoredResidualRouter(12, 8, 4, "cpu", 73)
    model.module.route_scale.data.fill_(0.9)
    model.module.reliability_scale.data.fill_(0.9)
    probability, _ = model.module(packed, "full")
    assert np.array_equal(probability.detach().numpy()[0], packed["llm_probability"].numpy()[0])


def test_phase52_response_features_and_anchor_are_ablatable():
    import torch

    packed = pack_cache([_item("p52-r0", 0.9), _item("p52-r1", 0.8)], "cpu")
    packed["route_view_prior"].zero_()
    packed["route_view_prior"][:, :5] = packed["route_view_prior"].new_tensor(
        [0.70, 0.10, 0.08, 0.07, 0.05],
    )
    model = AnalyticallyAnchoredResidualRouter(12, 8, 4, "cpu", 79)
    model.module.route_scale.data.fill_(0.7)
    model.module.reliability_scale.data.fill_(0.7)
    full, _ = model.module(packed, "full")
    shuffled, _ = model.module(packed, "shuffle_response")
    no_anchor, _ = model.module(packed, "no_anchor")
    assert not torch.allclose(full, shuffled)
    assert not torch.allclose(full, no_anchor)


def test_phase52_training_updates_regularized_residual_and_oracle_is_a_ceiling():
    items = [_item(f"p52-train-{index}", 0.9) for index in range(4)]
    packed = pack_cache(items, "cpu")
    model = AnalyticallyAnchoredResidualRouter(12, 8, 4, "cpu", 83)
    report = model.fit(
        packed, packed, mode="full", epochs=3, batch_size=2, learning_rate=1e-2,
        patience=3, anchor_kl=0.02, correction_l2=0.001,
    )
    probability, labels, traces = model.predict(packed, 2, "full")
    oracle, oracle_labels, oracle_traces = response_bank_oracle(packed)
    assert report["max_gradient_norm"] > 0
    assert np.isfinite(report["validation_loss"])
    assert probability.shape == labels.shape == oracle.shape == oracle_labels.shape == (16,)
    assert traces["weights"].shape == oracle_traces["weights"].shape == (4, 4, 17)
