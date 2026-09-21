from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import itertools
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import yaml

from .artifacts import atomic_json, digest_json
from .metrics import binary_metrics


@dataclass(frozen=True, slots=True)
class StressCell:
    suite: str
    seed: int
    markers: int
    episodes: int
    noise: float = 1.0
    corruption: float = 0.0
    conflict: float = 0.0
    context: int = 32
    method: str = "soft"
    structure: float = 1.0


def _states(markers: int) -> np.ndarray:
    values = np.arange(2 ** markers, dtype=np.uint16)[:, None]
    bits = ((values >> np.arange(markers)) & 1).astype(float)
    return bits * 2 - 1


def simulate_cell(cell: StressCell) -> dict[str, Any]:
    started = time.perf_counter()
    rng = np.random.default_rng(cell.seed)
    states = _states(cell.markers)
    state_count = len(states)
    true_state = rng.integers(0, state_count, size=cell.episodes)
    labels = rng.integers(0, 2, size=cell.episodes)
    # Context correlation estimates become more reliable with sqrt(n), while the
    # structure coefficient allows a true no-signal preservation condition.
    marker_evidence = (
        cell.structure * states[true_state]
        + rng.normal(0, cell.noise / np.sqrt(max(cell.context, 1) / 8),
                     size=(cell.episodes, cell.markers))
    )
    state_logits = marker_evidence @ states.T
    state_logits -= state_logits.max(1, keepdims=True)
    posterior = np.exp(state_logits); posterior /= posterior.sum(1, keepdims=True)
    if cell.structure == 0:
        posterior[:] = 1 / state_count
    mapping = np.arange(state_count)
    corrupt_count = round(cell.corruption * state_count)
    if corrupt_count:
        selected = rng.choice(state_count, corrupt_count, replace=False)
        mapping[selected] = rng.permutation(mapping[selected])
    view_prior = np.zeros((cell.episodes, state_count), dtype=float)
    view_prior[:, mapping] += posterior
    if cell.conflict:
        wrong = (true_state + 1) % state_count
        conflict_prior = np.full_like(view_prior, 1e-9)
        conflict_prior[np.arange(cell.episodes), wrong] = 1.0
        view_prior = (1 - cell.conflict) * view_prior + cell.conflict * conflict_prior
        view_prior /= view_prior.sum(1, keepdims=True)
    # Each cached specialist view is informative only in its matching regime.
    bank = rng.uniform(0.2, 0.8, size=(cell.episodes, state_count))
    correct_probability = np.where(labels == 1, 0.9, 0.1)
    bank[np.arange(cell.episodes), true_state] = correct_probability
    fallback = np.where(labels == 1, 0.75, 0.25) if cell.structure == 0 else np.full(cell.episodes, 0.5)
    if cell.method == "hard":
        weights = np.eye(state_count)[np.argmax(view_prior, axis=1)]
    elif cell.method == "uniform":
        weights = np.full_like(view_prior, 1 / state_count)
    elif cell.method == "oracle":
        weights = np.eye(state_count)[true_state]
    else:
        weights = view_prior
    routed = np.sum(weights * bank, axis=1)
    entropy = -np.sum(posterior * np.log(np.clip(posterior, 1e-12, 1)), axis=1)
    gate = np.clip(1 - entropy / np.log(state_count), 0, 1)
    gate[gate < 1e-12] = 0
    probability = fallback.copy()
    active = gate > 0
    probability[active] = (1 - gate[active]) * fallback[active] + gate[active] * routed[active]
    metrics = binary_metrics(labels, probability)
    return {
        **asdict(cell), **metrics,
        "mean_gate": float(np.mean(gate)),
        "exact_preservation": bool(np.array_equal(probability[gate == 0], fallback[gate == 0])),
        "state_count": state_count,
        "runtime_seconds": time.perf_counter() - started,
        "posterior_bytes": posterior.nbytes,
    }


def enumerate_cells(config: dict[str, Any], suite: str) -> list[StressCell]:
    common = config["experiment"]
    spec = config["suites"][suite]
    fields = {
        "seed": spec.get("seeds", common["seeds"]),
        "markers": spec.get("markers", [4]),
        "episodes": spec.get("episodes", [common["episodes"]]),
        "noise": spec.get("noise", [1.0]),
        "corruption": spec.get("corruption", [0.0]),
        "conflict": spec.get("conflict", [0.0]),
        "context": spec.get("context", [32]),
        "method": spec.get("methods", ["soft"]),
        "structure": spec.get("structure", [1.0]),
    }
    keys = list(fields)
    return [StressCell(suite=suite, **dict(zip(keys, values))) for values in itertools.product(
        *(fields[key] for key in keys)
    )]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config["experiment"]["classification"] != "exploratory_non_confirmatory":
        raise ValueError("Synthetic stress runs must remain exploratory")
    cells = enumerate_cells(config, args.suite)
    if args.aggregate:
        all_rows: list[dict[str, Any]] = []
        for rank in range(args.world_size):
            summary_path = args.output / f"{args.suite}_rank{rank}.json"
            csv_path = args.output / f"{args.suite}_rank{rank}.csv"
            if not summary_path.is_file() or not csv_path.is_file():
                raise RuntimeError(f"Missing synthetic shard {rank}")
            summary = json.loads(summary_path.read_text())
            if summary.get("status") != "complete" or summary.get("config_digest") != digest_json(config):
                raise RuntimeError(f"Invalid synthetic shard {rank}")
            with csv_path.open(newline="", encoding="utf-8") as handle:
                all_rows.extend(csv.DictReader(handle))
        if len(all_rows) != len(cells):
            raise RuntimeError(f"Expected {len(cells)} cells, found {len(all_rows)}")
        final_csv = args.output / f"{args.suite}_results.csv"
        with final_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
            writer.writeheader(); writer.writerows(all_rows)
        atomic_json(args.output / f"{args.suite}_summary.json", {
            "status": "complete", "suite": args.suite, "cells": len(all_rows),
            "config_digest": digest_json(config), "csv": str(final_csv),
        })
        return
    assigned = [cell for index, cell in enumerate(cells) if index % args.world_size == args.rank]
    rows = [simulate_cell(cell) for cell in assigned]
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"{args.suite}_rank{args.rank}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    atomic_json(args.output / f"{args.suite}_rank{args.rank}.json", {
        "status": "complete", "suite": args.suite, "rank": args.rank,
        "world_size": args.world_size, "expected_total": len(cells),
        "completed": len(rows), "config_digest": digest_json(config), "csv": str(path),
    })


if __name__ == "__main__":
    main()
