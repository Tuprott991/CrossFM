import numpy as np

from crossfm.baselines import semantic_statistical_oracle
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

