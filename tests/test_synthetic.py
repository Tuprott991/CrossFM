import numpy as np

from crossfm.baselines import build_llm_prompts, semantic_statistical_oracle
from crossfm.synthetic import make_episode, make_episodes, split_hash


def test_generation_is_deterministic_and_shapes_match():
    for regime in "ABC":
        left = make_episode(regime, 17, 2, n_query=7)
        right = make_episode(regime, 17, 2, n_query=7)
        assert left.checksum() == right.checksum()
        assert left.x_query.shape[0] == 7
        assert set(np.unique(left.y_context)).issubset({0, 1})


def test_paired_methods_receive_identical_split():
    episodes = make_episodes("C", 23, 5)
    assert split_hash(episodes) == split_hash(make_episodes("C", 23, 5))


def test_c_oracle_recovers_complementary_signal():
    batch = semantic_statistical_oracle(make_episodes("C", 29, 100, n_query=16))
    accuracy = np.mean((batch.probabilities >= 0.5) == batch.labels)
    assert accuracy > 0.80


def test_a_oracle_uses_stable_semantic_feature():
    episodes = make_episodes("A", 31, 100, n_query=16)
    batch = semantic_statistical_oracle(episodes)
    accuracy = np.mean((batch.probabilities >= 0.5) == batch.labels)
    assert accuracy > 0.90
    assert all(episode.x_context.shape == (6, 12) and episode.relevant == (2,) for episode in episodes)
    assert all(np.all(np.abs(episode.x_query[:, 2]) >= 1.25) for episode in episodes)
    prompt_query = build_llm_prompts(episodes[0])[0].split("Query: ", 1)[1]
    assert episodes[0].feature_names[2] in prompt_query
    assert episodes[0].feature_names[0] not in prompt_query
