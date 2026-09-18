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

C2_GROUP_ALIASES = (
    (("annual_income", "yearly_earnings", "household_revenue"), ("monthly_premium", "policy_fee", "insurance_cost")),
    (("late_payment_count", "payment_delays", "overdue_installments"), ("claim_count", "reported_incidents", "insurance_claims")),
    (("customer_tenure", "months_with_company", "policy_age"), ("engagement_score", "activity_index", "participation_level")),
    (("customer_age", "age_years", "client_age"), ("dependent_count", "household_dependents", "covered_family_size")),
    (("visit_frequency", "annual_visits", "service_visits"), ("usage_volume", "consumption_level", "service_usage")),
    (("support_tickets", "help_requests", "service_cases"), ("escalation_count", "raised_complaints", "critical_cases")),
    (("deductible_amount", "excess_amount", "policy_deductible"), ("coverage_limit", "insured_limit", "benefit_ceiling")),
    (("account_balance", "current_balance", "ledger_balance"), ("autopay_rate", "automatic_payment_share", "recurring_payment_ratio")),
    (("body_mass_index", "bmi_measure", "weight_index"), ("medication_count", "active_prescriptions", "medicine_total")),
    (("annual_salary", "employment_income", "wage_level"), ("job_change_count", "employer_changes", "career_transitions")),
    (("monthly_rent", "rental_cost", "lease_payment"), ("mortgage_balance", "home_loan_balance", "housing_debt")),
    (("credit_score", "credit_rating", "borrower_score"), ("debt_ratio", "leverage_ratio", "debt_burden")),
    (("purchase_spend", "transaction_value", "shopping_total"), ("refund_count", "returned_orders", "reimbursement_events")),
    (("login_frequency", "account_logins", "session_count"), ("security_alerts", "risk_notifications", "fraud_warnings")),
    (("annual_mileage", "distance_traveled", "vehicle_miles"), ("trip_count", "journey_frequency", "travel_events")),
    (("outage_minutes", "service_downtime", "interruption_duration"), ("network_latency", "response_delay", "connection_lag")),
)


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
    route_indices: tuple[int, ...] = ()
    route_map: tuple[int, ...] = ()
    route_code: int = -1

    def checksum(self) -> str:
        h = hashlib.sha256()
        for value in (self.episode_id, self.regime, self.mechanism, *self.feature_names, self.description):
            h.update(value.encode("utf-8"))
        for array in (self.x_context, self.y_context, self.x_query, self.y_query):
            h.update(np.ascontiguousarray(array).tobytes())
        if self.route_indices:
            h.update(json.dumps([self.route_indices, self.route_map, self.route_code]).encode())
        return h.hexdigest()


def _rng(seed: int, regime: str, index: int) -> np.random.Generator:
    regime_code = {"A": 11, "B": 23, "C": 37, "C2": 53}[regime]
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
        route_indices, route_map, route_code = (), (), -1
    elif regime == "C2":
        n_context, branches, route_width = 32, 16, 4
        p = route_width + 2 * branches
        alias_index = {"train": 0, "validation": 1, "test": 2}.get(alias_split)
        if alias_index is None:
            raise ValueError("C2 requires a declared train/validation/test alias split")
        probe_prefix = ("audit_probe", "validation_indicator", "local_marker")[alias_index]
        probe_names = [f"{probe_prefix}_{i + 1}" for i in range(route_width)]
        pair_names = [name for pair in C2_GROUP_ALIASES for name in (pair[0][alias_index], pair[1][alias_index])]
        names = tuple(probe_names + pair_names)
        if mechanism_split == "train":
            route_map = tuple(range(branches))
            mechanism = "difference"
        elif mechanism_split == "validation":
            route_map = tuple((5 * code + 1) % branches for code in range(branches))
            mechanism = "sum"
        elif mechanism_split == "test":
            route_map = tuple((7 * code + 3) % branches for code in range(branches))
            mechanism = "product"
        else:
            raise ValueError("C2 requires a declared train/validation/test mechanism split")
        route_code = int(rng.integers(0, branches))
        active_group = route_map[route_code]
        direction = int(rng.choice([-1, 1]))
        stable_context = rng.normal(size=n_context)
        stable_query = rng.normal(size=n_query)
        bit_signs = np.asarray([1.0 if route_code & (1 << bit) else -1.0 for bit in range(route_width)])
        route_context = np.column_stack([
            bit * direction * stable_context + rng.normal(0, 0.04, n_context) for bit in bit_signs
        ])
        route_query = rng.normal(size=(n_query, route_width))
        context_signals = [
            0.96 * stable_context + np.sqrt(1 - 0.96**2) * rng.normal(size=n_context)
            for _ in range(branches)
        ]
        query_signals = [rng.normal(size=n_query) for _ in range(branches)]
        context_signals[active_group] = stable_context
        query_signals[active_group] = stable_query

        def c2_pairs(signals: list[np.ndarray]) -> np.ndarray:
            cols: list[np.ndarray] = []
            for signal in signals:
                if mechanism == "difference":
                    center = rng.normal(0, 0.7, size=len(signal))
                    cols.extend([center - signal / 2, center + signal / 2])
                elif mechanism == "sum":
                    delta = rng.normal(0, 0.7, size=len(signal))
                    cols.extend([signal / 2 - delta, signal / 2 + delta])
                else:
                    first = rng.choice([-1.0, 1.0], size=len(signal)) * rng.uniform(0.65, 1.35, size=len(signal))
                    cols.extend([first, signal / first])
            return np.column_stack(cols).astype(np.float32)

        xc = np.column_stack([route_context, c2_pairs(context_signals)]).astype(np.float32)
        xq = np.column_stack([route_query, c2_pairs(query_signals)]).astype(np.float32)
        sc, sq = direction * stable_context, direction * stable_query
        yc = _labels_from_context_threshold(sc, rng.normal(0, 0.22, n_context), n_context)
        yq = (sq + rng.normal(0, 0.22, n_query) > 0).astype(np.int64)
        x, y = np.row_stack([xc, xq]), np.concatenate([yc, yq])
        entries = []
        for code, group in enumerate(route_map):
            pattern = "".join("P" if code & (1 << bit) else "N" for bit in range(route_width))
            a, b = pair_names[2 * group:2 * group + 2]
            entries.append(f"{pattern}->{a} with {b}")
        desc = (
            "Predict the binary outcome using an adaptive local code. Estimate whether each of the four marker fields has "
            "positive (P) or negative (N) context correlation with the label, in marker order. The resulting code selects "
            "the only stable feature pair: " + "; ".join(entries) + ". Marker fields and all nonselected pairs are unstable "
            "context shortcuts. After selecting the pair, infer its local transformation and direction from labels."
        )
        pair_start = route_width + 2 * active_group
        relevant = (pair_start, pair_start + 1)
        route_indices = tuple(range(route_width))
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
        route_indices=route_indices if regime == "C2" else (),
        route_map=route_map if regime == "C2" else (),
        route_code=route_code if regime == "C2" else -1,
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
