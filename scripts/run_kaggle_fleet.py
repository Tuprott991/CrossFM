from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "fullpaper.yaml"
ACTIVE_STATES = ("RUNNING", "QUEUED", "STARTING", "INITIALIZING")


@dataclass(frozen=True, slots=True)
class Lane:
    account: int
    profile: str
    kernel_slug: str
    title: str
    data_owner: str | None = None
    data_slug: str | None = None

    @property
    def token_name(self) -> str:
        return f"KAGGLE_ACCESS_TOKEN_{self.account}"


# One T4x2 notebook per legitimate collaborator initially. This consumes eight
# GPUs concurrently while preserving each account's second session for a bounded
# retry. D4 stays on the established account because TabPFN-3.5 also requires
# that account to have accepted the license and configured TABPFN_TOKEN.
LANES = (
    Lane(
        1, "author_b_kaggle_d4", "crossfm-fullpaper-d4-exploratory-v1",
        "CrossFM Full Paper D4 Exploratory V1",
        "alinoranianesfahani", "iranian-churn-dataset",
    ),
    Lane(
        2, "author_b_kaggle_frontier_temporal",
        "crossfm-fullpaper-frontier-temporal-exploratory-v1",
        "CrossFM Frontier Temporal Exploratory V1",
    ),
    Lane(
        3, "author_b_kaggle_frontier_semantic",
        "crossfm-fullpaper-frontier-semantic-exploratory-v1",
        "CrossFM Frontier Semantic Exploratory V1",
    ),
    Lane(
        4, "author_b_kaggle_d3", "crossfm-fullpaper-d3-exploratory-v1",
        "CrossFM Full Paper D3 Exploratory V1",
        "retailrocket", "ecommerce-dataset",
    ),
)


def _load_tokens() -> dict[int, str]:
    load_dotenv(ROOT / ".env", override=False)
    tokens: dict[int, str] = {}
    for lane in LANES:
        token = os.getenv(lane.token_name, "").strip()
        if not token:
            raise RuntimeError(f"Missing {lane.token_name} in {ROOT / '.env'}")
        tokens[lane.account] = token
    if len(set(tokens.values())) != len(tokens):
        raise RuntimeError("Kaggle access tokens must represent four distinct accounts")
    return tokens


def _safe_env(token: str) -> dict[str, str]:
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith("KAGGLE_ACCESS_TOKEN_")
    }
    env["KAGGLE_API_TOKEN"] = token
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _run(
    command: list[str], *, token: str | None = None, check: bool = True,
    cwd: Path = ROOT,
) -> subprocess.CompletedProcess[str]:
    env = _safe_env(token) if token else os.environ.copy()
    result = subprocess.run(
        command, cwd=cwd, env=env, text=True, capture_output=True,
        encoding="utf-8", errors="replace",
    )
    semantic_error = any(marker in result.stdout.lower() for marker in (
        "dataset creation error:", "dataset version creation error:",
        "kernel push error:", "notebook push error:",
    ))
    if check and (result.returncode or semantic_error):
        message = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"Command failed ({result.returncode}): {command[0]}: {message[-3000:]}")
    return result


def _json_output(result: subprocess.CompletedProcess[str]) -> Any:
    value = result.stdout.strip()
    if not value:
        return []
    if value.startswith("No "):
        return []
    return json.loads(value)


def _owner(token: str) -> str:
    for resource in ("kernels", "datasets"):
        result = _run(
            ["kaggle", resource, "list", "--mine", "--page-size", "1", "--format", "json"],
            token=token,
        )
        rows = _json_output(result)
        if rows:
            return str(rows[0]["ref"]).split("/", 1)[0]
    raise RuntimeError("Could not infer Kaggle owner from an otherwise valid token")


def _status(token: str, ref: str) -> str:
    result = _run(["kaggle", "kernels", "status", ref], token=token, check=False)
    return (result.stdout + result.stderr).strip()


def _active_recent(token: str) -> list[str]:
    result = _run([
        "kaggle", "kernels", "list", "--mine", "--sort-by", "dateRun",
        "--page-size", "2", "--format", "json",
    ], token=token)
    active = []
    for row in _json_output(result):
        ref = str(row["ref"])
        message = _status(token, ref).upper()
        if any(state in message for state in ACTIVE_STATES):
            active.append(ref)
    return active


def _quota(token: str) -> str:
    result = _run(["kaggle", "quota"], token=token)
    return " ".join(result.stdout.split())


def _dataset_files(token: str, ref: str) -> list[dict[str, Any]] | None:
    result = _run([
        "kaggle", "datasets", "files", ref, "--page-size", "200", "--format", "json",
    ], token=token, check=False)
    if result.returncode:
        return None
    value = _json_output(result)
    return value if isinstance(value, list) else []


def _preflight_lane(lane: Lane, token: str) -> dict[str, Any]:
    owner = _owner(token)
    active = _active_recent(token)
    if len(active) >= 2:
        raise RuntimeError(f"{owner} already has two active sessions: {active}")
    if lane.data_owner and lane.data_slug:
        source = f"{lane.data_owner}/{lane.data_slug}"
        files = _dataset_files(token, source)
        if files is None:
            raise RuntimeError(f"{owner} cannot access required dataset {source}")
    return {
        "account": lane.account, "owner": owner, "profile": lane.profile,
        "active_sessions": len(active), "quota": _quota(token),
    }


def preflight() -> list[dict[str, Any]]:
    tokens = _load_tokens()
    with ThreadPoolExecutor(max_workers=len(LANES)) as pool:
        rows = list(pool.map(lambda lane: _preflight_lane(lane, tokens[lane.account]), LANES))
    owners = [str(row["owner"]) for row in rows]
    if len(set(owners)) != len(owners):
        raise RuntimeError(f"Duplicate Kaggle owners detected: {owners}")
    return sorted(rows, key=lambda row: int(row["account"]))


def _bundle_slug(profile: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9.-]", "-", profile)
    return f"crossfm-{safe}-bundle".lower()


def build_lane(lane: Lane, owner: str) -> Path:
    command = [
        "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
        str(ROOT / "scripts" / "build_fullpaper_bundle.ps1"),
        "-Profile", lane.profile, "-KaggleOwner", owner,
        "-KernelSlug", lane.kernel_slug, "-KernelTitle", lane.title,
    ]
    if lane.data_owner and lane.data_slug:
        command += ["-DataOwner", lane.data_owner, "-DataSlug", lane.data_slug]
    _run(command)
    root = ROOT / "dist" / "fullpaper" / lane.profile
    _run([sys.executable, str(ROOT / "scripts" / "validate_fullpaper_bundle.py"), str(root), "--profile", lane.profile])
    return root


def _verify_remote_bundle(token: str, ref: str, local_dataset: Path) -> None:
    expected = {
        path.name: path.stat().st_size for path in local_dataset.iterdir() if path.is_file()
    }
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        rows = _dataset_files(token, ref)
        if rows is not None:
            observed = {
                str(row.get("name")): int(row.get("totalBytes") or row.get("size") or -1)
                for row in rows
            }
            if set(expected).issubset(observed) and all(
                observed[name] in (-1, size) for name, size in expected.items()
            ):
                return
        time.sleep(10)
    raise RuntimeError(f"Remote bundle did not become verifiable within five minutes: {ref}")


def upload_and_launch(lane: Lane, owner: str, token: str, bundle: Path) -> dict[str, str]:
    active = _active_recent(token)
    if len(active) >= 2:
        raise RuntimeError(f"Refusing launch: {owner} already has two active sessions")
    dataset_ref = f"{owner}/{_bundle_slug(lane.profile)}"
    dataset_dir = bundle / "dataset"
    exists = _dataset_files(token, dataset_ref) is not None
    if exists:
        _run([
            "kaggle", "datasets", "version", "-p", str(dataset_dir),
            "-m", f"Freeze {lane.profile} exploratory protocol", "-r", "zip",
        ], token=token)
    else:
        _run(["kaggle", "datasets", "create", "-p", str(dataset_dir), "-r", "zip"], token=token)
    _verify_remote_bundle(token, dataset_ref, dataset_dir)
    _run(["kaggle", "kernels", "push", "-p", str(bundle / "kernel")], token=token)
    kernel_ref = f"{owner}/{lane.kernel_slug}"
    return {
        "owner": owner, "profile": lane.profile, "dataset": dataset_ref,
        "kernel": kernel_ref, "status": _status(token, kernel_ref),
    }


def launch() -> list[dict[str, str]]:
    tokens = _load_tokens()
    checks = preflight()
    owners = {int(row["account"]): str(row["owner"]) for row in checks}
    bundles = {lane.account: build_lane(lane, owners[lane.account]) for lane in LANES}
    results = []
    with ThreadPoolExecutor(max_workers=len(LANES)) as pool:
        futures = {
            pool.submit(
                upload_and_launch, lane, owners[lane.account], tokens[lane.account],
                bundles[lane.account],
            ): lane for lane in LANES
        }
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda row: row["profile"])


def statuses() -> list[dict[str, str]]:
    tokens = _load_tokens()
    results = []
    for lane in LANES:
        owner = _owner(tokens[lane.account])
        ref = f"{owner}/{lane.kernel_slug}"
        results.append({"profile": lane.profile, "kernel": ref, "status": _status(tokens[lane.account], ref)})
    return results


def monitor(poll_seconds: int) -> list[dict[str, str]]:
    while True:
        rows = statuses()
        print(json.dumps(rows, indent=2), flush=True)
        combined = " ".join(row["status"].upper() for row in rows)
        if not any(state in combined for state in ACTIVE_STATES):
            return rows
        time.sleep(poll_seconds)


def download() -> list[dict[str, str]]:
    tokens = _load_tokens()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rows = []
    for lane in LANES:
        owner = _owner(tokens[lane.account])
        ref = f"{owner}/{lane.kernel_slug}"
        status = _status(tokens[lane.account], ref)
        if "COMPLETE" not in status.upper() and "ERROR" not in status.upper():
            rows.append({"kernel": ref, "status": status, "downloaded": "false"})
            continue
        target = ROOT / "outputs" / "fullpaper_fleet" / stamp / lane.profile
        target.mkdir(parents=True, exist_ok=False)
        _run(["kaggle", "kernels", "output", ref, "-p", str(target)], token=tokens[lane.account])
        rows.append({"kernel": ref, "status": status, "downloaded": str(target)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the four-author CrossFM Kaggle fleet")
    parser.add_argument("command", choices=("preflight", "launch", "status", "monitor", "download"))
    parser.add_argument("--poll-seconds", type=int, default=120)
    args = parser.parse_args()
    if args.command == "preflight":
        value = preflight()
    elif args.command == "launch":
        value = launch()
    elif args.command == "status":
        value = statuses()
    elif args.command == "monitor":
        value = monitor(args.poll_seconds)
    else:
        value = download()
    print(json.dumps(value, indent=2), flush=True)


if __name__ == "__main__":
    main()
