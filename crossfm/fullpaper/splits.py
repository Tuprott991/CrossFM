from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class TemporalSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp


def temporal_split(
    timestamps: Iterable,
    *,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    embargo_days: int = 30,
) -> TemporalSplit:
    values = pd.to_datetime(pd.Series(timestamps), utc=True, errors="raise")
    unique = np.sort(values.unique())
    if len(unique) < 5:
        raise ValueError("At least five distinct cutoffs are required")
    train_index = max(0, min(len(unique) - 3, int(len(unique) * train_fraction) - 1))
    validation_index = max(
        train_index + 1,
        min(len(unique) - 2, int(len(unique) * (train_fraction + validation_fraction)) - 1),
    )
    train_end = pd.Timestamp(unique[train_index])
    raw_validation_start = train_end + pd.Timedelta(days=embargo_days)
    validation_start_values = unique[unique >= raw_validation_start]
    if not len(validation_start_values):
        raise ValueError("Embargo removes the validation partition")
    validation_start = pd.Timestamp(validation_start_values[0])
    validation_end = pd.Timestamp(unique[validation_index])
    if validation_end < validation_start:
        candidates = unique[unique >= validation_start]
        if len(candidates) < 2:
            raise ValueError("Insufficient validation cutoffs after embargo")
        validation_end = pd.Timestamp(candidates[max(0, len(candidates) // 2 - 1)])
    raw_test_start = validation_end + pd.Timedelta(days=embargo_days)
    test_start_values = unique[unique >= raw_test_start]
    if not len(test_start_values):
        raise ValueError("Embargo removes the test partition")
    test_start = pd.Timestamp(test_start_values[0])
    train = np.flatnonzero(values <= train_end)
    validation = np.flatnonzero((values >= validation_start) & (values <= validation_end))
    test = np.flatnonzero(values >= test_start)
    if not len(train) or not len(validation) or not len(test):
        raise ValueError("A temporal partition is empty")
    return TemporalSplit(train, validation, test, train_end, validation_start, validation_end, test_start)


def assert_asof_integrity(
    frame: pd.DataFrame,
    *,
    cutoff_column: str,
    feature_timestamp_columns: list[str],
    horizon_days: int,
    label_window_end_column: str | None = None,
) -> None:
    cutoff = pd.to_datetime(frame[cutoff_column], utc=True, errors="raise")
    for column in feature_timestamp_columns:
        observed = pd.to_datetime(frame[column], utc=True, errors="coerce")
        invalid = observed.notna() & (observed > cutoff)
        if invalid.any():
            raise ValueError(f"Post-cutoff feature leakage in {column}: {int(invalid.sum())} rows")
    if label_window_end_column:
        label_end = pd.to_datetime(frame[label_window_end_column], utc=True, errors="raise")
        expected = cutoff + pd.Timedelta(days=horizon_days)
        if not (label_end == expected).all():
            raise ValueError("Label windows do not match the frozen prediction horizon")


def split_hash(indices: dict[str, np.ndarray]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for name in sorted(indices):
        digest.update(name.encode("utf-8"))
        digest.update(np.asarray(indices[name], dtype=np.int64).tobytes())
    return digest.hexdigest()
