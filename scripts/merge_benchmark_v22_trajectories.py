#!/usr/bin/env python3
"""Merge retained Benchmark v2.1 trajectories with the v2.2 retests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shopping_grpo.evaluation.artifacts import write_json_atomic, write_jsonl_atomic
from shopping_grpo.evaluation.blind_guard import validate_canonical_benchmark_files
from shopping_grpo.evaluation.manifest import sha256_file
from shopping_grpo.evaluation.pipeline import evaluate_trajectories


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-run-dir", type=Path, required=True)
    parser.add_argument("--retest-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, default=ROOT / "data/evaluation/tasks.jsonl")
    parser.add_argument("--slices", type=Path, default=ROOT / "data/evaluation/slices.jsonl")
    parser.add_argument("--metadata", type=Path, default=ROOT / "data/evaluation/metadata.json")
    parser.add_argument(
        "--replacement-manifest",
        type=Path,
        default=ROOT / "data/evaluation/replacement-manifest-v2.2.json",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise SystemExit(f"{path}:{line_number}: row must be an object")
            rows.append(row)
    return rows


def _task_id(row: dict) -> int:
    value = row.get("task_id")
    if value is None and isinstance(row.get("extra_info"), dict):
        value = row["extra_info"].get("task_id")
    if value is None and isinstance(row.get("normalized_trajectory"), dict):
        value = row["normalized_trajectory"].get("task_id")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit("trajectory is missing a valid task_id") from exc


def _index(rows: list[dict], label: str) -> dict[int, dict]:
    indexed = {}
    for row in rows:
        task_id = _task_id(row)
        if task_id in indexed:
            raise SystemExit(f"{label} contains duplicate task_id {task_id}")
        indexed[task_id] = row
    return indexed


def _compatible_run_contract(manifest: dict) -> dict:
    protocol = dict(manifest.get("protocol") or {})
    protocol.pop("workers", None)
    protocol.pop("missing_tasks_count_as_failures", None)
    actor = manifest.get("actor") or {}
    return {
        "actor": {key: actor.get(key) for key in ("model", "tokenizer")},
        "protocol": protocol,
        "environment": manifest.get("environment"),
        "code": manifest.get("code"),
    }


def main() -> None:
    args = parse_args()
    benchmark_metadata, slices = validate_canonical_benchmark_files(
        tasks_path=args.tasks,
        metadata_path=args.metadata,
        slices_path=args.slices,
    )
    replacement = _read_json(args.replacement_manifest)
    removed_ids = {int(value) for value in replacement.get("removed_task_ids") or []}
    added_ids = {int(value) for value in replacement.get("added_task_ids") or []}
    if not removed_ids or len(removed_ids) != len(added_ids):
        raise SystemExit("Benchmark v2.2 replacement manifest has inconsistent task sets")

    old_manifest = _read_json(args.old_run_dir / "run_manifest.json")
    retest_manifest = _read_json(args.retest_run_dir / "run_manifest.json")
    if _compatible_run_contract(old_manifest) != _compatible_run_contract(retest_manifest):
        raise SystemExit("old and retest run contracts differ; refusing to merge incomparable trajectories")

    old_rows = _index(_read_jsonl(args.old_run_dir / "trajectories.jsonl"), "old run")
    retest_rows = _index(_read_jsonl(args.retest_run_dir / "trajectories.jsonl"), "retest run")
    if set(retest_rows) != added_ids:
        raise SystemExit(
            "retest trajectories must contain exactly the replacement task IDs; "
            f"expected={sorted(added_ids)} actual={sorted(retest_rows)}"
        )
    retained_rows = {task_id: row for task_id, row in old_rows.items() if task_id not in removed_ids}
    canonical_ids = [int(row["task_id"]) for row in _read_jsonl(args.tasks)]
    merged_by_id = {**retained_rows, **retest_rows}
    missing = sorted(set(canonical_ids) - set(merged_by_id))
    unexpected = sorted(set(merged_by_id) - set(canonical_ids))
    if missing or unexpected:
        raise SystemExit(
            "cannot create complete Benchmark v2.2 result: "
            f"missing={missing} unexpected={unexpected}"
        )
    merged = [merged_by_id[task_id] for task_id in canonical_ids]
    actor = retest_manifest.get("actor") or old_manifest.get("actor") or {}
    artifacts = evaluate_trajectories(
        expected_task_ids=canonical_ids,
        trajectories=merged,
        actor=actor,
        task_slices=slices,
        rubric_bundles=[],
        judge_results=[],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectories_path = args.output_dir / "trajectories.jsonl"
    evaluations_path = args.output_dir / "evaluations.jsonl"
    summary_path = args.output_dir / "summary.json"
    write_jsonl_atomic(trajectories_path, merged)
    write_jsonl_atomic(evaluations_path, artifacts["evaluations"])
    write_json_atomic(summary_path, artifacts["summary"])
    write_json_atomic(
        args.output_dir / "merge_manifest.json",
        {
            "schema": "shopping-benchmark-v2.2-trajectory-merge-v1",
            "benchmark": benchmark_metadata["asset"],
            "old_run_dir": str(args.old_run_dir),
            "retest_run_dir": str(args.retest_run_dir),
            "retained_count": len(canonical_ids) - len(added_ids),
            "replacement_count": len(added_ids),
            "removed_task_ids": sorted(removed_ids),
            "added_task_ids": sorted(added_ids),
            "outputs": {
                "trajectories.jsonl": sha256_file(trajectories_path),
                "evaluations.jsonl": sha256_file(evaluations_path),
                "summary.json": sha256_file(summary_path),
            },
        },
    )
    print(json.dumps(artifacts["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
