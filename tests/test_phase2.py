import numpy as np

from crossfm.baselines import semantic_statistical_oracle
from crossfm.phase2 import candidate_views, tool_prompts, tune_ensemble_alpha
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
