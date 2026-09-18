from dataclasses import replace

import numpy as np

from crossfm.baselines import adaptive_routing_oracle, semantic_statistical_oracle
from crossfm.phase2 import (
    apply_residual_bypass, candidate_views, residual_gate, soft_code_posterior,
    tool_prompts, tune_ensemble_alpha,
)
from crossfm.synthetic import make_episode, make_episodes


def test_alias_splits_are_disjoint_and_ids_encode_protocol_split():
    train = make_episode("C", 11, 0, alias_split="train", mechanism_split="train")
    validation = make_episode("C", 11, 0, alias_split="validation", mechanism_split="validation")
    test = make_episode("C", 11, 0, alias_split="test", mechanism_split="test")
    assert train.feature_names[0] == "annual_income"
    assert validation.feature_names[0] == "yearly_earnings"
    assert test.feature_names[0] == "household_revenue"
    assert len({train.episode_id, validation.episode_id, test.episode_id}) == 3


def test_heldout_product_mechanism_remains_complementary_and_recoverable():
    episodes = make_episodes(
        "C", 19, 200, n_query=16, alias_split="test", mechanism_split="test",
    )
    batch = semantic_statistical_oracle(episodes)
    assert np.mean((batch.probabilities >= 0.5) == batch.labels) > 0.80
    assert all(episode.mechanism == "test" for episode in episodes)


def test_one_way_view_budget_and_tool_prompt_contract():
    for regime, minimum in (("A", 3), ("B", 3), ("C", 3)):
        episode = make_episode(regime, 7, 0, n_query=2)
        views = candidate_views(episode)
        assert len(views) >= minimum
        assert views[-1][1] == tuple(range(len(episode.feature_names)))
        prompts = tool_prompts(episode, np.asarray([0.25, 0.75]))
        assert len(prompts) == 2
        assert all("FINAL: LOW" in prompt and "FINAL: HIGH" in prompt for prompt in prompts)


def test_ensemble_alpha_is_validation_loss_tuned():
    labels = np.asarray([0, 0, 1, 1])
    llm = np.asarray([0.1, 0.2, 0.8, 0.9])
    tfm = 1 - llm
    alpha, loss = tune_ensemble_alpha(labels, llm, tfm)
    assert alpha == 1.0
    assert np.isfinite(loss)


def test_c2_requires_routing_and_adaptive_oracle_recovers_test_mechanism():
    episodes = make_episodes(
        "C2", 71, 256, n_query=16, alias_split="test", mechanism_split="test",
    )
    assert all(ep.x_context.shape == (32, 36) for ep in episodes)
    assert all(len(ep.route_indices) == 4 and sorted(ep.route_map) == list(range(16)) for ep in episodes)
    assert all(ep.relevant[0] >= 4 and ep.mechanism == "test" for ep in episodes)
    batch = adaptive_routing_oracle(episodes)
    assert np.mean((batch.probabilities >= 0.5) == batch.labels) > 0.80
    counts = np.bincount([ep.route_code for ep in episodes], minlength=16)
    assert counts.max() / counts.sum() < 0.12


def test_c2_codebook_and_aliases_are_held_out():
    train = make_episode("C2", 83, 0, alias_split="train", mechanism_split="train")
    validation = make_episode("C2", 83, 0, alias_split="validation", mechanism_split="validation")
    test = make_episode("C2", 83, 0, alias_split="test", mechanism_split="test")
    assert len({train.route_map, validation.route_map, test.route_map}) == 3
    assert train.feature_names[4] == "annual_income"
    assert validation.feature_names[4] == "yearly_earnings"
    assert test.feature_names[4] == "household_revenue"
    assert len(candidate_views(test)) == 17


def test_soft_route_posterior_is_continuous_and_normalized():
    episode = make_episode("C2", 91, 0, alias_split="test", mechanism_split="test")
    posterior = soft_code_posterior(episode)
    assert posterior.shape == (16,)
    assert np.isclose(posterior.sum(), 1.0)
    assert np.all((posterior > 0.0) & (posterior < 1.0))


def test_zero_routing_evidence_forces_exact_llm_bypass():
    episode = make_episode("C2", 97, 0, alias_split="test", mechanism_split="test")
    context = episode.x_context.copy()
    context[:, episode.route_indices] = 0.0
    episode = replace(episode, x_context=context)
    posterior = soft_code_posterior(episode)
    assert np.allclose(posterior, np.full(16, 1.0 / 16.0))
    assert residual_gate(episode, posterior) == 0.0
    llm = np.linspace(0.1, 0.9, len(episode.y_query))
    adapter = 1.0 - llm
    combined, gates = apply_residual_bypass([episode], llm, adapter, {episode.episode_id: 0.0})
    assert np.array_equal(gates, np.zeros_like(gates))
    assert np.array_equal(combined, llm)


def test_task_a_uses_preservation_bypass():
    episode = make_episode("A", 101, 0, alias_split="test", mechanism_split="test")
    assert residual_gate(episode, soft_code_posterior(episode)) == 0.0


def test_large_context_without_structure_also_uses_bypass():
    episode = make_episode("B", 103, 0, alias_split="test", mechanism_split="test")
    episode = replace(episode, x_context=np.zeros_like(episode.x_context))
    assert residual_gate(episode, soft_code_posterior(episode)) == 0.0
