#!/usr/bin/env python3
"""Combine retained canonical SFT rows with new Teacher rollouts at an exact length mix."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from shopping_grpo.collection.sft import acceptance_reasons, build_sft_row, read_jsonl


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-sft", type=Path, default=ROOT / "data/sft/all.jsonl")
    parser.add_argument("--new-raw", type=Path, action="append", required=True)
    parser.add_argument(
        "--held-out", type=Path, default=ROOT / "data/evaluation/tasks.jsonl"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--short", type=int, default=200)
    parser.add_argument("--medium", type=int, default=200)
    parser.add_argument("--long", type=int, default=100)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", default="fresh-sft-length-442-v1")
    return parser.parse_args()


def _bucket_from_sft(row: dict) -> str:
    steps = sum(message.get("role") == "tool" for message in row.get("messages") or [])
    if steps <= 10:
        return "short"
    if steps <= 20:
        return "medium"
    return "long"


def _stable(seed: str, row: dict) -> str:
    return hashlib.sha256(
        f"{seed}:{int(row['task_id'])}:{row.get('trajectory_id')}".encode()
    ).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    targets = {"short": args.short, "medium": args.medium, "long": args.long}
    if sum(targets.values()) < 1 or not 0 <= args.validation_ratio < 1:
        raise SystemExit("invalid length targets or validation ratio")
    held_out = {int(row["task_id"]) for row in read_jsonl(args.held_out)}
    base_rows = list(read_jsonl(args.base_sft))
    base_by_bucket = {name: [] for name in targets}
    for row in base_rows:
        base_by_bucket[_bucket_from_sft(row)].append(row)

    selected = []
    selected_ids = set()
    retained = {}
    for bucket in ("medium", "long"):
        target = targets[bucket]
        rows = sorted(base_by_bucket[bucket], key=lambda row: _stable(args.seed, row))
        kept = rows[:target]
        retained[bucket] = len(kept)
        selected.extend(kept)
        selected_ids.update(int(row["task_id"]) for row in kept)
    retained["short"] = 0

    additions = Counter()
    rejected = Counter()
    accepted_raw = []
    for raw_path in args.new_raw:
        for trajectory in read_jsonl(raw_path):
            accepted, reasons = acceptance_reasons(trajectory)
            if not accepted:
                rejected.update(reasons)
                continue
            task_id = int(trajectory["task_id"])
            if task_id in held_out:
                rejected["held_out_task"] += 1
                continue
            if task_id in selected_ids:
                rejected["duplicate_task"] += 1
                continue
            row = build_sft_row(trajectory)
            bucket = _bucket_from_sft(row)
            if bucket == "short":
                rejected["new_short_not_used"] += 1
                continue
            current = retained[bucket] + additions[bucket]
            if current >= targets[bucket]:
                rejected[f"{bucket}_quota_full"] += 1
                continue
            selected.append(row)
            selected_ids.add(task_id)
            additions[bucket] += 1
            accepted_raw.append(trajectory)

    available_short = sorted(
        (
            row
            for row in base_by_bucket["short"]
            if int(row["task_id"]) not in selected_ids
        ),
        key=lambda row: _stable(args.seed, row),
    )
    kept_short = available_short[: targets["short"]]
    retained["short"] = len(kept_short)
    selected.extend(kept_short)
    selected_ids.update(int(row["task_id"]) for row in kept_short)

    final_counts = Counter(_bucket_from_sft(row) for row in selected)
    missing = {name: targets[name] - final_counts[name] for name in targets}
    if any(missing.values()):
        raise SystemExit(
            json.dumps(
                {"status": "incomplete", "counts": final_counts, "missing": missing},
                ensure_ascii=False,
            )
        )

    validation = []
    train = []
    for bucket, target in targets.items():
        rows = sorted(
            (row for row in selected if _bucket_from_sft(row) == bucket),
            key=lambda row: _stable(args.seed + ":split", row),
        )
        validation_count = round(target * args.validation_ratio)
        validation.extend(rows[:validation_count])
        train.extend(rows[validation_count:])
    all_rows = sorted(selected, key=lambda row: int(row["task_id"]))
    train.sort(key=lambda row: int(row["task_id"]))
    validation.sort(key=lambda row: int(row["task_id"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "all.jsonl", all_rows)
    _write_jsonl(args.output_dir / "train.jsonl", train)
    _write_jsonl(args.output_dir / "validation.jsonl", validation)
    _write_jsonl(args.output_dir / "accepted_new_raw.jsonl", accepted_raw)
    report = {
        "schema_version": "shopping-sft-length-balanced-build-v1",
        "status": "complete",
        "targets": targets,
        "counts": dict(final_counts),
        "retained": retained,
        "added": dict(additions),
        "train_counts": dict(Counter(_bucket_from_sft(row) for row in train)),
        "validation_counts": dict(
            Counter(_bucket_from_sft(row) for row in validation)
        ),
        "rows": {"all": len(all_rows), "train": len(train), "validation": len(validation)},
        "rejected": dict(rejected),
        "seed": args.seed,
    }
    (args.output_dir / "length_balance.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
