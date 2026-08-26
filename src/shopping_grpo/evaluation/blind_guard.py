"""Content- and task-ID protection for the frozen blind final test."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from importlib.resources import files
from pathlib import Path

from shopping_grpo.evaluation.artifacts import ArtifactError

BLIND_GUARD_SCHEMA = "shopping-blind-asset-guard-v1"
BLIND_TASK_IDS_SCHEMA = "shopping-blind-task-ids-v1"
_RESOURCE_PACKAGE = "shopping_grpo.resources"
_GUARD_RESOURCE = "blind_guard.json"
_EXPECTED_METADATA = {
    "asset": "shopbench_longhorizon_final_240_v2_2",
    "contract": "environment-v2.4/reward-v4/benchmark-v2.2",
    "environment_version": "shopsimulator-environment-v2.4",
    "reward_version": "shopsimulator-reward-v4",
    "evaluated": False,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_resource_object(name: str) -> dict:
    try:
        resource = files(_RESOURCE_PACKAGE).joinpath(name)
        value = json.loads(resource.read_text(encoding="utf-8"))
    except (ModuleNotFoundError, OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read packaged blind resource: {name}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"packaged blind resource must be an object: {name}")
    return value


def _row_task_id(row: Mapping) -> int | None:
    value = row.get("task_id")
    if value is None:
        extra = row.get("extra_info")
        if isinstance(extra, Mapping):
            value = extra.get("task_id")
    if value is None:
        normalized = row.get("normalized_trajectory")
        if isinstance(normalized, Mapping):
            value = normalized.get("task_id")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"invalid task_id in {row!r}") from exc


def _jsonl_task_ids(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    opener = gzip.open if path.name.endswith(".gz") else open
    task_ids = set()
    try:
        with opener(path, "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ArtifactError(
                        f"{path}:{line_number}: JSONL row must be an object"
                    )
                task_id = _row_task_id(value)
                if task_id is not None:
                    task_ids.add(task_id)
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"{path}: invalid JSONL during blind guard") from exc
    return task_ids


def validate_canonical_blind_asset() -> tuple[dict, set[int]]:
    """Validate the wheel-packaged guard contract and frozen task-ID set."""

    guard = _load_resource_object(_GUARD_RESOURCE)
    if guard.get("schema_version") != BLIND_GUARD_SCHEMA:
        raise ArtifactError("unsupported blind guard schema")
    if guard.get("manifest_version") != 1:
        raise ArtifactError("unsupported blind guard manifest version")
    if guard.get("split_role") != "blind_final_test":
        raise ArtifactError("blind guard split_role must be blind_final_test")
    required = guard.get("required_metadata")
    if not isinstance(required, Mapping):
        raise ArtifactError("blind guard required_metadata must be an object")
    mismatches = {
        key: {"required": expected, "actual": required.get(key)}
        for key, expected in _EXPECTED_METADATA.items()
        if required.get(key) != expected
    }
    if mismatches:
        raise ArtifactError(
            "packaged blind metadata contract mismatch: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    for field in ("task_sha256", "metadata_sha256"):
        digest = guard.get(field)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ArtifactError(f"invalid packaged blind {field}")

    ids_resource = guard.get("task_ids_resource")
    if not isinstance(ids_resource, str) or not ids_resource:
        raise ArtifactError("blind guard task_ids_resource is missing")
    ids_document = _load_resource_object(ids_resource)
    if ids_document.get("schema_version") != BLIND_TASK_IDS_SCHEMA:
        raise ArtifactError("unsupported blind task-ID schema")
    raw_task_ids = ids_document.get("task_ids")
    if not isinstance(raw_task_ids, list) or not all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in raw_task_ids
    ):
        raise ArtifactError("packaged blind task_ids must be integers")
    task_ids = set(raw_task_ids)
    if len(task_ids) != len(raw_task_ids):
        raise ArtifactError("packaged blind task_ids contain duplicates")
    if len(task_ids) != int(guard.get("task_count", -1)):
        raise ArtifactError("packaged blind task count mismatch")
    return guard, task_ids


def validate_canonical_benchmark_files(
    *,
    tasks_path: Path,
    metadata_path: Path,
    slices_path: Path,
) -> tuple[dict, dict[int, dict]]:
    """Validate the checked-in Final-240 files before any formal rollout."""

    guard, packaged_task_ids = validate_canonical_blind_asset()
    tasks_path = Path(tasks_path)
    metadata_path = Path(metadata_path)
    slices_path = Path(slices_path)
    for path in (tasks_path, metadata_path, slices_path):
        if not path.is_file():
            raise ArtifactError(f"missing canonical benchmark file: {path}")

    if _sha256_file(tasks_path) != guard["task_sha256"]:
        raise ArtifactError("Final-240 task file hash does not match the frozen guard")
    if _sha256_file(metadata_path) != guard["metadata_sha256"]:
        raise ArtifactError("Final-240 metadata hash does not match the frozen guard")

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError("invalid Final-240 metadata") from exc
    if not isinstance(metadata, dict):
        raise ArtifactError("Final-240 metadata must be an object")
    required_metadata = {
        "schema_version": "shopping-evaluation-dataset-v2.2",
        "asset": "shopbench_longhorizon_final_240_v2_2",
        "contract": "environment-v2.4/reward-v4/benchmark-v2.2",
        "environment": "shopsimulator-environment-v2.4",
        "reward": "shopsimulator-reward-v4",
        "tasks": 240,
        "core_tasks": 180,
        "challenge_tasks": 60,
        "evaluated": False,
    }
    mismatches = {
        key: {"expected": expected, "actual": metadata.get(key)}
        for key, expected in required_metadata.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise ArtifactError(
            "Final-240 metadata contract mismatch: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    if metadata.get("task_sha256") != guard["task_sha256"]:
        raise ArtifactError("Final-240 metadata task hash does not match the guard")
    if _sha256_file(slices_path) != metadata.get("slice_sha256"):
        raise ArtifactError("Final-240 slice file hash does not match metadata")

    task_ids = _jsonl_task_ids(tasks_path)
    if task_ids != packaged_task_ids:
        raise ArtifactError("Final-240 task IDs do not match the packaged blind asset")

    slices = {}
    try:
        with slices_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ArtifactError(
                        f"{slices_path}:{line_number}: slice row must be an object"
                    )
                task_id = _row_task_id(row)
                if task_id is None or task_id not in packaged_task_ids:
                    raise ArtifactError(
                        f"{slices_path}:{line_number}: unexpected task_id"
                    )
                if task_id in slices:
                    raise ArtifactError(
                        f"{slices_path}:{line_number}: duplicate task_id {task_id}"
                    )
                suite = row.get("suite")
                challenge_slice = row.get("challenge_slice")
                if suite not in {"core", "challenge"}:
                    raise ArtifactError(
                        f"{slices_path}:{line_number}: invalid suite {suite!r}"
                    )
                if not isinstance(row.get("domain"), str) or not row["domain"]:
                    raise ArtifactError(
                        f"{slices_path}:{line_number}: domain is required"
                    )
                if suite == "core" and challenge_slice is not None:
                    raise ArtifactError("core tasks must not declare a challenge slice")
                if suite == "challenge" and not isinstance(challenge_slice, str):
                    raise ArtifactError("challenge tasks must declare a challenge slice")
                slices[task_id] = row
    except json.JSONDecodeError as exc:
        raise ArtifactError("invalid Final-240 slice JSONL") from exc

    if set(slices) != packaged_task_ids:
        raise ArtifactError("Final-240 slices do not cover the frozen task set")
    suite_counts = Counter(row["suite"] for row in slices.values())
    challenge_counts = Counter(
        row["challenge_slice"]
        for row in slices.values()
        if row["suite"] == "challenge"
    )
    if dict(suite_counts) != {"core": 180, "challenge": 60}:
        raise ArtifactError("Final-240 suite counts are invalid")
    if sorted(challenge_counts.values()) != [10] * 6:
        raise ArtifactError("Final-240 challenge slice counts are invalid")
    return metadata, slices


def guard_blind_final(
    paths: Iterable[Path],
    *,
    allowed: bool,
) -> None:
    """Reject any artifact containing final-test task IDs, independent of name."""

    guard, final_task_ids = validate_canonical_blind_asset()
    if allowed:
        return
    blocked = {}
    canonical_sha = str(guard["task_sha256"])
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            continue
        same_content = _sha256_file(path) == canonical_sha
        overlap = sorted(_jsonl_task_ids(path) & final_task_ids)
        if same_content or overlap:
            blocked[str(path)] = {
                "same_content": same_content,
                "overlap_count": len(overlap),
                "sample_task_ids": overlap[:10],
            }
    if blocked:
        raise ArtifactError(
            "refusing to consume frozen blind-final tasks without "
            "--allow-blind-final: "
            + json.dumps(blocked, ensure_ascii=False, sort_keys=True)
        )
