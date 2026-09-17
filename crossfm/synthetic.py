from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np


ALIASES = {
    "income": ["annual_income", "yearly_earnings", "household_revenue"],
    "premium": ["monthly_premium", "policy_fee", "recurring_insurance_cost"],
    "late": ["late_payment_count", "payment_delays", "overdue_installments"],
    "claims": ["claim_count", "reported_incidents", "insurance_claims"],
    "tenure": ["customer_tenure", "months_with_company", "policy_age"],
    "age": ["customer_age", "age_years", "client_age"],
}


@dataclass(frozen=True)
class Episode:
    episode_id: str
    regime: str
    feature_names: tuple[str, ...]
    description: str
    x_context: np.ndarray
    y_context: np.ndarray
    x_query: np.ndarray
    y_query: np.ndarray
    relevant: tuple[int, ...]

    def checksum(self) -> str:
        h = hashlib.sha256()
        for value in (self.episode_id, self.regime, *self.feature_names, self.description):
            h.update(value.encode("utf-8"))
        for array in (self.x_context, self.y_context, self.x_query, self.y_query):
            h.update(np.ascontiguousarray(array).tobytes())
        return h.hexdigest()


def _rng(seed: int, regime: str, index: int) -> np.random.Generator:
    regime_code = {"A": 11, "B": 23, "C": 37}[regime]
    return np.random.default_rng(np.random.SeedSequence([seed, regime_code, index]))


def _labels_from_context_threshold(score: np.ndarray, noise: np.ndarray, n_context: int) -> np.ndarray:
    latent = score + noise
    # Balance the small training context without using query outcomes to set the threshold.
    threshold = float(np.median(latent[:n_context]))
    return (latent > threshold).astype(np.int64)


def _semantic_names(rng: np.random.Generator) -> tuple[str, ...]:
    return tuple(rng.choice(ALIASES[key]) for key in ("income", "premium", "late", "claims", "tenure", "age"))


def make_episode(regime: str, seed: int, index: int, n_query: int = 8) -> Episode:
    rng = _rng(seed, regime, index)
    if regime == "A":
        n_context, p = 10, 6
        names = _semantic_names(rng)
        x = rng.normal(size=(n_context + n_query, p)).astype(np.float32)
        score = -0.8 * x[:, 0] + 0.8 * x[:, 2] + 0.65 * x[:, 3] - 0.45 * x[:, 4]
        y = _labels_from_context_threshold(score, rng.normal(0, 0.40, len(score)), n_context)
        desc = (
            "Predict near-term policy lapse. Higher earnings and longer customer tenure usually reduce risk; "
            "late payments and prior claims increase risk. Values are standardized: positive means above average."
        )
        relevant = (0, 2, 3, 4)
    elif regime == "B":
        n_context, p = 384, 6
        names = tuple(f"x{i + 1}" for i in range(p))
        x = rng.normal(size=(n_context + n_query, p)).astype(np.float32)
        score = 1.4 * x[:, 0] * x[:, 1] + 0.55 * x[:, 2] - 0.25 * x[:, 3]
        y = _labels_from_context_threshold(score, rng.normal(0, 0.20, len(score)), n_context)
        desc = "Predict the binary outcome from anonymized measurements. Infer the relationship only from examples."
        relevant = (0, 1, 2, 3)
    elif regime == "C":
        n_context, pairs = 28, 6
        p = pairs * 2
        semantic = list(_semantic_names(rng)[:2])
        names = tuple(semantic + [f"auxiliary_{i + 1}" for i in range(p - 2)])
        sign = int(rng.choice([-1, 1]))
        d_context = rng.normal(size=n_context)
        d_query = rng.normal(size=n_query)
        context_diffs = [d_context]
        query_diffs = [d_query]
        for _ in range(1, pairs):
            context_diffs.append(0.94 * d_context + np.sqrt(1 - 0.94**2) * rng.normal(size=n_context))
            query_diffs.append(rng.normal(size=n_query))
        def paired_matrix(diffs: list[np.ndarray]) -> np.ndarray:
            cols: list[np.ndarray] = []
            for diff in diffs:
                center = rng.normal(0, 0.7, size=len(diff))
                cols.extend([center - diff / 2, center + diff / 2])
            return np.column_stack(cols).astype(np.float32)
        xc = paired_matrix(context_diffs)
        xq = paired_matrix(query_diffs)
        sc = sign * (xc[:, 1] - xc[:, 0])
        sq = sign * (xq[:, 1] - xq[:, 0])
        yc = _labels_from_context_threshold(sc, rng.normal(0, 0.25, n_context), n_context)
        # Use the context threshold (approximately zero) rather than leaking query rank.
        yq = (sq + rng.normal(0, 0.25, n_query) > 0).astype(np.int64)
        x = np.row_stack([xc, xq])
        y = np.concatenate([yc, yq])
        desc = (
            "Predict policy lapse. Domain knowledge identifies financial strain—the relationship between recurring "
            "premium and income—as the stable mechanism, but this local portfolio's direction is not disclosed. "
            "The auxiliary columns are unstable proxies. Values are standardized."
        )
        relevant = (0, 1)
    else:
        raise ValueError(f"Unknown regime: {regime}")
    return Episode(
        episode_id=f"{regime}-s{seed}-e{index:04d}",
        regime=regime,
        feature_names=names,
        description=desc,
        x_context=x[:n_context],
        y_context=y[:n_context],
        x_query=x[n_context:],
        y_query=y[n_context:],
        relevant=relevant,
    )


def make_episodes(regime: str, seed: int, count: int, n_query: int = 8) -> list[Episode]:
    return [make_episode(regime, seed, i, n_query=n_query) for i in range(count)]


def split_hash(episodes: list[Episode]) -> str:
    payload = [(episode.episode_id, episode.checksum()) for episode in episodes]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()
