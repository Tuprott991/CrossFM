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
    mechanism: str = "pilot"

    def checksum(self) -> str:
        h = hashlib.sha256()
        for value in (self.episode_id, self.regime, self.mechanism, *self.feature_names, self.description):
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


def _semantic_names(rng: np.random.Generator, alias_split: str = "mixed") -> tuple[str, ...]:
    """Select aliases from a declared, non-overlapping split.

    ``mixed`` preserves the Phase-1 protocol.  Phase 2 uses index 0 for
    training, 1 for validation and 2 for test so that learned adapters cannot
    memorize the surface form of a feature name.
    """
    if alias_split == "mixed":
        return tuple(rng.choice(ALIASES[key]) for key in ("income", "premium", "late", "claims", "tenure", "age"))
    index = {"train": 0, "validation": 1, "test": 2}.get(alias_split)
    if index is None:
        raise ValueError(f"Unknown alias split: {alias_split}")
    return tuple(ALIASES[key][index] for key in ("income", "premium", "late", "claims", "tenure", "age"))


def make_episode(
    regime: str,
    seed: int,
    index: int,
    n_query: int = 8,
    alias_split: str = "mixed",
    mechanism_split: str = "pilot",
) -> Episode:
    rng = _rng(seed, regime, index)
    if regime == "A":
        n_context, p = 6, 12
        names = _semantic_names(rng, alias_split) + tuple(f"portfolio_proxy_{i + 1}" for i in range(p - 6))
        # Every column is a near-perfect context shortcut. Only the semantically
        # named late-payment feature remains coupled to risk at query time.
        context_risk = np.concatenate([
            rng.uniform(-2.0, -0.25, n_context // 2),
            rng.uniform(0.25, 2.0, n_context // 2),
        ])
        rng.shuffle(context_risk)
        signs = rng.choice([-1.0, 1.0], size=p)
        signs[2] = 1.0
        xc = np.column_stack([
            signs[j] * context_risk + rng.normal(0, 0.03, n_context) for j in range(p)
        ]).astype(np.float32)
        negatives = rng.uniform(-2.0, -1.25, n_query // 2)
        positives = rng.uniform(1.25, 2.0, n_query - len(negatives))
        query_risk = np.concatenate([negatives, positives])
        rng.shuffle(query_risk)
        xq = rng.normal(size=(n_query, p)).astype(np.float32)
        xq[:, 2] = query_risk
        yc = (context_risk > 0).astype(np.int64)
        yq = (query_risk > 0).astype(np.int64)
        x = np.row_stack([xc, xq])
        y = np.concatenate([yc, yq])
        desc = (
            "Predict near-term policy lapse. Repeated late payments are the stable causal warning sign: an above-average "
            "late-payment count raises risk and a below-average count lowers it. Other portfolio measurements can be "
            "unstable proxies. Values are standardized: positive means above average."
        )
        relevant = (2,)
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
        semantic = list(_semantic_names(rng, alias_split)[:2])
        names = tuple(semantic + [f"auxiliary_{i + 1}" for i in range(p - 2)])
        sign = int(rng.choice([-1, 1]))
        stable_context = rng.normal(size=n_context)
        stable_query = rng.normal(size=n_query)
        context_signals = [stable_context]
        query_signals = [stable_query]
        for _ in range(1, pairs):
            context_signals.append(0.94 * stable_context + np.sqrt(1 - 0.94**2) * rng.normal(size=n_context))
            query_signals.append(rng.normal(size=n_query))

        mechanism = "difference" if mechanism_split in {"pilot", "train"} else mechanism_split

        def paired_matrix(signals: list[np.ndarray]) -> np.ndarray:
            cols: list[np.ndarray] = []
            for signal in signals:
                if mechanism == "difference":
                    center = rng.normal(0, 0.7, size=len(signal))
                    cols.extend([center - signal / 2, center + signal / 2])
                elif mechanism == "validation":
                    delta = rng.normal(0, 0.7, size=len(signal))
                    cols.extend([signal / 2 - delta, signal / 2 + delta])
                elif mechanism == "test":
                    first = rng.choice([-1.0, 1.0], size=len(signal)) * rng.uniform(0.65, 1.35, size=len(signal))
                    cols.extend([first, signal / first])
                else:
                    raise ValueError(f"Unknown mechanism split: {mechanism_split}")
            return np.column_stack(cols).astype(np.float32)
        xc = paired_matrix(context_signals)
        xq = paired_matrix(query_signals)
        # Phase 2 deliberately holds out the functional form of the stable
        # semantic pair.  Metadata identifies the pair, but only the context
        # identifies its transformation and episode-specific direction.
        if mechanism_split in {"pilot", "train"}:
            sc = sign * stable_context
            sq = sign * stable_query
        elif mechanism_split == "validation":
            sc = sign * stable_context
            sq = sign * stable_query
        elif mechanism_split == "test":
            sc = sign * stable_context
            sq = sign * stable_query
        else:
            raise ValueError(f"Unknown mechanism split: {mechanism_split}")
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
        episode_id=(
            f"{regime}-s{seed}-e{index:04d}"
            if alias_split == "mixed" and mechanism_split == "pilot"
            else f"{regime}-{alias_split}-{mechanism_split}-s{seed}-e{index:04d}"
        ),
        regime=regime,
        feature_names=names,
        description=desc,
        x_context=x[:n_context],
        y_context=y[:n_context],
        x_query=x[n_context:],
        y_query=y[n_context:],
        relevant=relevant,
        mechanism=mechanism_split,
    )


def make_episodes(
    regime: str,
    seed: int,
    count: int,
    n_query: int = 8,
    alias_split: str = "mixed",
    mechanism_split: str = "pilot",
) -> list[Episode]:
    return [
        make_episode(
            regime, seed, i, n_query=n_query,
            alias_split=alias_split, mechanism_split=mechanism_split,
        )
        for i in range(count)
    ]


def split_hash(episodes: list[Episode]) -> str:
    payload = [(episode.episode_id, episode.checksum()) for episode in episodes]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()
