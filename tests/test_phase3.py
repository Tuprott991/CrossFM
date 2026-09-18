import numpy as np

from crossfm.phase3 import CrossFMEpisodeCache, CrossFMLatentLoop, pack_cache


def _item(
    name: str, gate: float, dimension: int = 12, *, routed: bool | None = None,
) -> CrossFMEpisodeCache:
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
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
