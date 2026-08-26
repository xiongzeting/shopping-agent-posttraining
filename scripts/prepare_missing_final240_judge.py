#!/usr/bin/env python3
"""Prepare a small resumable Judge run using the existing frozen Final-240 Rubrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    path.write_text(payload, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-id", type=int, action="append", required=True)
    args = parser.parse_args()

    root = args.evaluation_root.resolve()
    output = args.output.resolve()
    task_ids = list(dict.fromkeys(args.task_id))
    wanted = set(task_ids)

    canonical_slices = {
        int(row["task_id"]): row
        for row in load_jsonl(root.parents[1] / "data/evaluation/slices.jsonl")
    }
    frozen_rubrics = {
        int(row["task_id"]): row for row in load_jsonl(root / "rubrics.jsonl")
    }
    missing_slices = sorted(wanted - set(canonical_slices))
    missing_rubrics = sorted(wanted - set(frozen_rubrics))
    if missing_slices or missing_rubrics:
        raise SystemExit(
            f"missing slices={missing_slices}, missing rubrics={missing_rubrics}"
        )

    write_jsonl(
        output / "tasks.jsonl",
        [{"task_id": task_id} for task_id in task_ids],
    )
    write_jsonl(
        output / "slices.jsonl",
        [canonical_slices[task_id] for task_id in task_ids],
    )
    rubric_rows = [frozen_rubrics[task_id] for task_id in task_ids]
    write_jsonl(output / "rubrics.jsonl", rubric_rows)
    write_jsonl(
        output / "checkpoints/rubrics.jsonl",
        [{"_checkpoint_key": str(row["task_id"]), **row} for row in rubric_rows],
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "task_ids": task_ids,
                "rubrics": len(rubric_rows),
                "rubric_source": str(root / "rubrics.jsonl"),
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
