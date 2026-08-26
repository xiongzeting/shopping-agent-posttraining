#!/usr/bin/env python3
"""Offline GRPO gate; does not load a model or require CUDA."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()

    if not args.model.is_dir() or not (args.model / "config.json").is_file():
        raise SystemExit(f"model missing config.json: {args.model}")
    if not any((args.model / n).is_file() for n in (
        "model.safetensors", "model.safetensors.index.json",
        "pytorch_model.bin", "pytorch_model.bin.index.json",
    )):
        raise SystemExit(f"model has no supported weights: {args.model}")

    train_rows = pq.read_table(args.train, columns=["extra_info"]).to_pylist()
    val_rows = pq.read_table(args.validation, columns=["extra_info"]).to_pylist()
    if not train_rows or not val_rows:
        raise SystemExit("train/validation parquet must be non-empty")
    train_ids = [int((r.get("extra_info") or {}).get("task_id")) for r in train_rows]
    val_ids = [int((r.get("extra_info") or {}).get("task_id")) for r in val_rows]
    if set(train_ids) & set(val_ids):
        raise SystemExit("train/validation task overlap detected")
    successes = [
        (r.get("extra_info") or {}).get("accepted_probe_purchase_successes")
        for r in train_rows
    ]
    if any(not isinstance(x, int) or x < 1 for x in successes):
        raise SystemExit("training rows missing accepted Probe Gold/Valid count")
    report = {
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "unique_train_task_ids": len(set(train_ids)),
        "duplicate_train_rows": len(train_ids) - len(set(train_ids)),
        "probe_success_distribution": dict(sorted(Counter(successes).items())),
        "train_validation_overlap": 0,
        "model": str(args.model.resolve()),
        "status": "passed",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
