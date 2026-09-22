from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import sha256_file
from .splits import assert_asof_integrity, split_hash, temporal_split


@dataclass(slots=True)
class DatasetBundle:
    dataset_id: str
    frame: pd.DataFrame
    target: pd.Series
    splits: dict[str, np.ndarray]
    feature_descriptions: dict[str, str]
    task_description: str
    checksum: str
    split_digest: str
    groups: pd.Series | None = None
    timestamps: pd.Series | None = None
    official_metric: str = "roc_auc"

    def validate(self) -> None:
        if len(self.frame) != len(self.target) or not len(self.frame):
            raise ValueError(f"Invalid rows for {self.dataset_id}")
        all_indices = np.concatenate([np.asarray(value) for value in self.splits.values()])
        if len(all_indices) != len(np.unique(all_indices)):
            raise ValueError(f"Overlapping splits for {self.dataset_id}")
        if all_indices.min(initial=0) < 0 or all_indices.max(initial=-1) >= len(self.frame):
            raise ValueError(f"Out-of-range split indices for {self.dataset_id}")
        if set(np.unique(self.target.dropna())) - {0, 1, False, True}:
            raise ValueError(f"Only binary targets are supported: {self.dataset_id}")


def _frame_checksum(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return digest.hexdigest()


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def build_event_cohorts(
    events: pd.DataFrame,
    *,
    customer_column: str,
    timestamp_column: str,
    cutoffs: list[pd.Timestamp],
    horizon_days: int = 30,
    history_days: int = 180,
    value_column: str | None = None,
    item_column: str | None = None,
    event_type_column: str | None = None,
    qualifying_types: set[str] | None = None,
    minimum_events: int = 2,
    minimum_active_days: int = 2,
    recent_days: int = 90,
) -> pd.DataFrame:
    """Create leakage-safe customer/cutoff features and next-period lapse labels."""

    required = {customer_column, timestamp_column}
    missing = required - set(events)
    if missing:
        raise ValueError(f"Missing event columns: {sorted(missing)}")
    frame = events.copy()
    frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], utc=True, errors="raise")
    frame = frame.dropna(subset=[customer_column]).sort_values(timestamp_column)
    frame["__event_day"] = frame[timestamp_column].dt.floor("D")
    rows: list[pd.DataFrame] = []
    windows = (7, 14, 30, 60, 90, history_days)
    for raw_cutoff in sorted(cutoffs):
        cutoff = pd.Timestamp(raw_cutoff)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        history_start = cutoff - pd.Timedelta(days=history_days)
        horizon_end = cutoff + pd.Timedelta(days=horizon_days)
        history = frame[(frame[timestamp_column] < cutoff) & (frame[timestamp_column] >= history_start)]
        recent = history[history[timestamp_column] >= cutoff - pd.Timedelta(days=recent_days)]
        eligibility = recent.groupby(customer_column).agg(
            eligible_events=(timestamp_column, "size"),
            eligible_days=("__event_day", "nunique"),
        )
        eligible = eligibility[
            (eligibility["eligible_events"] >= minimum_events)
            & (eligibility["eligible_days"] >= minimum_active_days)
        ].index
        if not len(eligible):
            continue
        base = pd.DataFrame(index=eligible)
        last_time = history.groupby(customer_column)[timestamp_column].max().reindex(eligible)
        base["recency_days"] = (cutoff - last_time).dt.total_seconds() / 86400.0
        base["tenure_observed_days"] = (
            cutoff - history.groupby(customer_column)[timestamp_column].min().reindex(eligible)
        ).dt.total_seconds() / 86400.0
        for window in windows:
            selected = history[history[timestamp_column] >= cutoff - pd.Timedelta(days=window)]
            grouped = selected.groupby(customer_column)
            base[f"event_count_{window}d"] = grouped.size().reindex(eligible, fill_value=0)
            base[f"active_days_{window}d"] = grouped["__event_day"].nunique().reindex(
                eligible, fill_value=0,
            )
            if value_column and value_column in selected:
                values = pd.to_numeric(selected[value_column], errors="coerce").fillna(0.0)
                selected = selected.assign(__value=values)
                value_group = selected.groupby(customer_column)["__value"]
                base[f"value_sum_{window}d"] = value_group.sum().reindex(eligible, fill_value=0.0)
                base[f"value_mean_{window}d"] = value_group.mean().reindex(eligible, fill_value=0.0)
        if item_column and item_column in history:
            base["unique_items_180d"] = history.groupby(customer_column)[item_column].nunique().reindex(
                eligible, fill_value=0,
            )
        if event_type_column and event_type_column in history:
            counts = pd.crosstab(history[customer_column], history[event_type_column])
            for event_type in sorted(map(str, counts.columns)):
                base[f"event_type_{event_type}_180d"] = counts[event_type].reindex(
                    eligible, fill_value=0,
                )
        future = frame[(frame[timestamp_column] >= cutoff) & (frame[timestamp_column] < horizon_end)]
        if qualifying_types is not None and event_type_column:
            future = future[future[event_type_column].astype(str).isin(qualifying_types)]
        active_future = set(future[customer_column])
        base["target"] = [0 if customer in active_future else 1 for customer in base.index]
        base["customer_id"] = base.index.astype(str)
        base["cutoff"] = cutoff
        base["max_feature_time"] = pd.array(last_time, dtype="datetime64[ns, UTC]")
        base["label_window_end"] = horizon_end
        rows.append(base.reset_index(drop=True))
    if not rows:
        raise ValueError("No eligible customer-cutoff rows were produced")
    result = pd.concat(rows, ignore_index=True)
    assert_asof_integrity(
        result, cutoff_column="cutoff", feature_timestamp_columns=["max_feature_time"],
        horizon_days=horizon_days, label_window_end_column="label_window_end",
    )
    return result


def _bundle_from_cohorts(dataset_id: str, cohorts: pd.DataFrame, spec: dict[str, Any]) -> DatasetBundle:
    split = temporal_split(
        cohorts["cutoff"],
        train_fraction=float(spec.get("train_fraction", 0.60)),
        validation_fraction=float(spec.get("validation_fraction", 0.20)),
        embargo_days=int(spec.get("embargo_days", 30)),
    )
    indices = {"train": split.train, "validation": split.validation, "test": split.test}
    excluded = {"target", "customer_id", "cutoff", "max_feature_time", "label_window_end"}
    features = cohorts[[column for column in cohorts if column not in excluded]].copy()
    bundle = DatasetBundle(
        dataset_id=dataset_id,
        frame=features,
        target=cohorts["target"].astype(int),
        splits=indices,
        feature_descriptions={column: column.replace("_", " ") for column in features},
        task_description=spec["task_description"],
        checksum=_frame_checksum(cohorts),
        split_digest=split_hash(indices),
        groups=cohorts["customer_id"].astype(str),
        timestamps=cohorts["cutoff"],
        official_metric=spec.get("official_metric", "average_precision"),
    )
    bundle.validate()
    return bundle


def load_online_retail(dataset_id: str, spec: dict[str, Any], data_root: Path) -> DatasetBundle:
    path = data_root / spec["files"]["transactions"]
    raw = _read_table(path)
    rename = {
        "Customer ID": "customer_id", "CustomerID": "customer_id",
        "InvoiceDate": "timestamp", "StockCode": "item_id",
        "Quantity": "quantity", "Price": "price", "UnitPrice": "price",
        "Invoice": "invoice", "InvoiceNo": "invoice",
    }
    raw = raw.rename(columns=rename)
    for column in ("customer_id", "timestamp", "quantity", "price"):
        if column not in raw:
            raise ValueError(f"Online Retail file lacks {column}")
    raw["quantity"] = pd.to_numeric(raw["quantity"], errors="coerce")
    raw["price"] = pd.to_numeric(raw["price"], errors="coerce")
    if "invoice" in raw:
        raw = raw[~raw["invoice"].astype(str).str.startswith("C")]
    raw = raw[(raw["quantity"] > 0) & (raw["price"] >= 0)].copy()
    raw["value"] = raw["quantity"] * raw["price"]
    raw["event_type"] = "purchase"
    timestamps = pd.to_datetime(raw["timestamp"], utc=True)
    cutoffs = list(pd.date_range(
        timestamps.min().ceil("MS") + pd.Timedelta(days=180),
        timestamps.max().floor("D") - pd.Timedelta(days=30), freq="30D", tz="UTC",
    ))
    cohorts = build_event_cohorts(
        raw, customer_column="customer_id", timestamp_column="timestamp", cutoffs=cutoffs,
        value_column="value", item_column="item_id" if "item_id" in raw else None,
        event_type_column="event_type", qualifying_types={"purchase"},
    )
    return _bundle_from_cohorts(dataset_id, cohorts, spec)


def load_retailrocket(dataset_id: str, spec: dict[str, Any], data_root: Path) -> DatasetBundle:
    raw = _read_table(data_root / spec["files"]["events"])
    required = {"visitorid", "timestamp", "event"}
    if not required.issubset(raw):
        raise ValueError(f"RetailRocket file lacks {sorted(required - set(raw))}")
    raw = raw.rename(columns={"visitorid": "customer_id", "itemid": "item_id"})
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], unit="ms", utc=True)
    raw["value"] = (raw["event"].astype(str) == "transaction").astype(float)
    timestamps = raw["timestamp"]
    history_days = int(spec.get("history_days", 90))
    warmup_days = int(spec.get("cutoff_warmup_days", min(history_days, 30)))
    frequency_days = int(spec.get("cutoff_frequency_days", 14))
    cutoffs = list(pd.date_range(
        timestamps.min().ceil("D") + pd.Timedelta(days=warmup_days),
        timestamps.max().floor("D") - pd.Timedelta(days=30),
        freq=f"{frequency_days}D", tz="UTC",
    ))
    target_mode = spec.get("target_mode", "engagement")
    qualifying = {"transaction"} if target_mode == "purchase" else {
        "view", "addtocart", "transaction",
    }
    cohorts = build_event_cohorts(
        raw, customer_column="customer_id", timestamp_column="timestamp", cutoffs=cutoffs,
        history_days=history_days, value_column="value",
        item_column="item_id", event_type_column="event", qualifying_types=qualifying,
        minimum_events=3,
    )
    return _bundle_from_cohorts(dataset_id, cohorts, spec)


def load_manifest_table(dataset_id: str, spec: dict[str, Any], data_root: Path) -> DatasetBundle:
    path = data_root / spec["files"]["table"]
    frame = _read_table(path)
    target_column = spec["target_column"]
    if target_column not in frame:
        raise ValueError(f"Target column {target_column!r} is missing")
    target = frame.pop(target_column).astype(int)
    split_column = spec.get("split_column")
    if split_column:
        labels = frame.pop(split_column).astype(str).str.lower()
        splits = {name: np.flatnonzero(labels == name) for name in ("train", "validation", "test")}
    elif spec["split"] == "temporal":
        timestamp_column = spec["timestamp_column"]
        timestamps = pd.to_datetime(frame.pop(timestamp_column), utc=True, errors="raise")
        split = temporal_split(timestamps, embargo_days=int(spec.get("embargo_days", 30)))
        splits = {"train": split.train, "validation": split.validation, "test": split.test}
    else:
        from sklearn.model_selection import train_test_split

        all_indices = np.arange(len(frame))
        train, rest = train_test_split(
            all_indices, test_size=0.4, random_state=spec.get("split_seed", 1701),
            stratify=target,
        )
        validation, test = train_test_split(
            rest, test_size=0.5, random_state=spec.get("split_seed", 1701),
            stratify=target.iloc[rest],
        )
        splits = {"train": np.sort(train), "validation": np.sort(validation), "test": np.sort(test)}
    descriptions = spec.get("feature_descriptions", {})
    bundle = DatasetBundle(
        dataset_id, frame, target, splits,
        {column: descriptions.get(column, column.replace("_", " ")) for column in frame},
        spec["task_description"], sha256_file(path), split_hash(splits),
        official_metric=spec.get("official_metric", "roc_auc"),
    )
    bundle.validate()
    return bundle


def load_beyondarena(dataset_id: str, spec: dict[str, Any], _: Path) -> DatasetBundle:
    try:
        from data_foundry.collections import BEYOND_ARENA
    except ImportError as exc:
        raise RuntimeError("Install data-foundry to load BeyondArena") from exc
    container = BEYOND_ARENA.get_dataset(spec["unique_name"])
    frame = container.dataset.copy()
    target_column = container.task_metadata.target_column_name
    target = frame.pop(target_column).astype(int)
    repeats = container.experiment_metadata.splits
    repeat_id = sorted(repeats)[int(spec.get("repeat", 0))]
    folds = repeats[repeat_id]
    fold_id = sorted(folds)[int(spec.get("fold", 0))]
    train, test = folds[fold_id]
    train = np.asarray(train, dtype=int)
    test = np.asarray(test, dtype=int)
    # Validation is carved only from the official training fold, preserving test isolation.
    if spec["split"] in {"temporal", "grouped"}:
        # Official non-IID containers preserve source order in their outer-train indices.
        # A tail split avoids random mixing across time/group blocks. The outer test fold
        # remains untouched.
        permuted = np.sort(train)
    else:
        from sklearn.model_selection import train_test_split

        train_part, validation_part = train_test_split(
            train, test_size=0.2, random_state=int(spec.get("validation_seed", 1901)),
            stratify=target.iloc[train],
        )
        permuted = np.concatenate((np.sort(train_part), np.sort(validation_part)))
    boundary = max(1, int(0.8 * len(permuted)))
    splits = {
        "train": np.sort(permuted[:boundary]),
        "validation": np.sort(permuted[boundary:]),
        "test": np.sort(test),
    }
    metadata = container.dataset_metadata
    identity = {
        "unique_name": spec["unique_name"],
        "repeat": str(repeat_id), "fold": str(fold_id),
        "shape": frame.shape,
        "metadata": str(metadata),
    }
    descriptions = spec.get("feature_descriptions", {})
    bundle = DatasetBundle(
        dataset_id, frame, target, splits,
        {column: descriptions.get(column, column.replace("_", " ")) for column in frame},
        spec["task_description"], hashlib.sha256(repr(identity).encode()).hexdigest(),
        split_hash(splits), official_metric=spec.get("official_metric", "roc_auc"),
    )
    bundle.validate()
    return bundle


LOADERS = {
    "online_retail": load_online_retail,
    "retailrocket": load_retailrocket,
    "manifest_table": load_manifest_table,
    "beyondarena": load_beyondarena,
}


def load_dataset(dataset_id: str, spec: dict[str, Any], data_root: str | Path) -> DatasetBundle:
    loader = spec["loader"]
    if loader not in LOADERS:
        raise KeyError(f"Unsupported dataset loader: {loader}")
    return LOADERS[loader](dataset_id, spec, Path(data_root))


def schema_condition(
    descriptions: dict[str, str], condition: str, *, seed: int,
) -> dict[str, str]:
    names = list(descriptions)
    rng = np.random.default_rng(seed)
    if condition == "canonical":
        return dict(descriptions)
    if condition == "anonymized":
        return {name: f"x_{index + 1}" for index, name in enumerate(names)}
    if condition == "shuffled_descriptions":
        values = list(descriptions.values())
        rng.shuffle(values)
        return dict(zip(names, values))
    if condition.startswith("missing_"):
        fraction = float(condition.split("_", 1)[1]) / 100.0
        selected = set(rng.choice(names, size=round(len(names) * fraction), replace=False))
        return {name: "" if name in selected else value for name, value in descriptions.items()}
    if condition == "aliases":
        # Alias files can override this deterministic neutral fallback in the dataset manifest.
        return {name: name.replace("_", " ") for name in names}
    raise ValueError(f"Unknown schema condition: {condition}")
