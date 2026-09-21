from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "crossfm-fullpaper-v1"


def canonical_json(value: Any) -> str:
    if is_dataclass(value):
        value = asdict(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(value)
    os.replace(temporary, path)


def write_task_record(
    output_dir: Path,
    *,
    task_id: str,
    protocol_id: str,
    config_digest: str,
    payload: dict[str, Any],
    arrays_path: Path | None = None,
) -> Path:
    record = {
        "artifact_schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "protocol_id": protocol_id,
        "config_digest": config_digest,
        "status": "complete",
        "payload": payload,
    }
    if arrays_path is not None:
        try:
            artifact_name = arrays_path.resolve().relative_to(output_dir.resolve()).as_posix()
        except ValueError as exc:
            raise ValueError("Referenced artifacts must live under the run output directory") from exc
        record["arrays"] = {
            "path": artifact_name,
            "sha256": sha256_file(arrays_path),
            "bytes": arrays_path.stat().st_size,
        }
    path = output_dir / "records" / f"{task_id}.json"
    atomic_json(path, record)
    return path


def reusable_record(
    path: Path,
    *,
    task_id: str,
    protocol_id: str,
    config_digest: str,
) -> bool:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record != record:  # defensive; JSON parser already rejects no values here
            return False
        expected = {
            "artifact_schema_version": SCHEMA_VERSION,
            "task_id": task_id,
            "protocol_id": protocol_id,
            "config_digest": config_digest,
            "status": "complete",
        }
        if any(record.get(key) != value for key, value in expected.items()):
            return False
        arrays = record.get("arrays")
        if arrays:
            candidate = path.parent.parent / Path(arrays["path"])
            return (
                candidate.is_file()
                and candidate.stat().st_size == int(arrays["bytes"])
                and sha256_file(candidate) == arrays["sha256"]
            )
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def validate_task_set(expected_ids: set[str], record_dir: Path) -> list[dict[str, Any]]:
    records = []
    seen: set[str] = set()
    for path in sorted(record_dir.glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        task_id = str(value["task_id"])
        if task_id in seen:
            raise RuntimeError(f"Duplicate task record: {task_id}")
        seen.add(task_id)
        records.append(value)
    missing, unexpected = expected_ids - seen, seen - expected_ids
    if missing or unexpected:
        raise RuntimeError(
            f"Incomplete task grid: missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )
    return records
