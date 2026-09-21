from __future__ import annotations

from typing import Any

import numpy as np


def binary_metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import (
        average_precision_score, brier_score_loss, log_loss, roc_auc_score,
    )

    labels = np.asarray(labels, dtype=int).reshape(-1)
    probability = np.clip(np.asarray(probability, dtype=float).reshape(-1), 1e-8, 1 - 1e-8)
    if labels.shape != probability.shape or not np.isfinite(probability).all():
        raise ValueError("Invalid labels/probabilities")
    order = np.argsort(-probability)
    top = max(1, int(np.ceil(0.10 * len(labels))))
    prevalence = float(np.mean(labels))
    lift = float(np.mean(labels[order[:top]]) / prevalence) if prevalence else float("nan")
    predicted = (probability >= 0.5).astype(int)
    return {
        "accuracy": float(np.mean(predicted == labels)),
        "roc_auc": float(roc_auc_score(labels, probability)),
        "average_precision": float(average_precision_score(labels, probability)),
        "log_loss": float(log_loss(labels, probability, labels=[0, 1])),
        "brier": float(brier_score_loss(labels, probability)),
        "top_decile_lift": lift,
    }


def paired_cluster_bootstrap(
    labels: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    groups: np.ndarray | None,
    metric: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    labels = np.asarray(labels)
    left = np.asarray(left)
    right = np.asarray(right)
    group_values = np.arange(len(labels)) if groups is None else np.asarray(groups)
    unique = np.unique(group_values)
    rng = np.random.default_rng(seed)
    differences = []
    higher_is_better = metric not in {"log_loss", "brier"}
    for _ in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(group_values == group) for group in sampled])
        left_score = binary_metrics(labels[indices], left[indices])[metric]
        right_score = binary_metrics(labels[indices], right[indices])[metric]
        difference = left_score - right_score
        differences.append(difference if higher_is_better else -difference)
    values = np.asarray(differences)
    return {
        "metric": metric,
        "direction": "positive_favors_left",
        "mean_difference": float(np.mean(values)),
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "replicates": replicates,
    }


def adaptive_ece(labels: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    labels = np.asarray(labels)
    probability = np.asarray(probability)
    order = np.argsort(probability)
    chunks = np.array_split(order, min(bins, len(order)))
    return float(sum(
        len(chunk) / len(labels) * abs(np.mean(labels[chunk]) - np.mean(probability[chunk]))
        for chunk in chunks if len(chunk)
    ))
