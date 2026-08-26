#!/usr/bin/env python3
"""Compare completed Benchmark v2.1 runs on the same frozen Final-240 tasks."""

import argparse
from pathlib import Path
import sys

from shopping_grpo.evaluation.artifacts import iter_jsonl, write_json_atomic
from shopping_grpo.evaluation.blind_guard import validate_canonical_benchmark_files
from shopping_grpo.evaluation.comparison import compare_evaluation_runs


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="配对比较 Benchmark v2.1 模型结果")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "data/evaluation/tasks.jsonl",
    )
    parser.add_argument(
        "--benchmark-metadata",
        type=Path,
        default=ROOT / "data/evaluation/metadata.json",
    )
    parser.add_argument(
        "--benchmark-slices",
        type=Path,
        default=ROOT / "data/evaluation/slices.jsonl",
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=EVALUATIONS_JSONL",
        help="至少提供两次，例如 baseline=.../evaluations.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _parse_runs(values):
    runs = {}
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator or not label.strip() or not raw_path.strip():
            raise SystemExit("--run 必须使用 LABEL=EVALUATIONS_JSONL 格式")
        label = label.strip()
        if label in runs:
            raise SystemExit(f"重复的 run label: {label}")
        runs[label] = list(iter_jsonl(Path(raw_path)))
    if len(runs) < 2:
        raise SystemExit("至少需要两个 --run")
    return runs


def main():
    args = parse_args()
    _, task_slices = validate_canonical_benchmark_files(
        tasks_path=args.benchmark,
        metadata_path=args.benchmark_metadata,
        slices_path=args.benchmark_slices,
    )
    task_ids = [int(row["task_id"]) for row in iter_jsonl(args.benchmark)]
    result = compare_evaluation_runs(
        expected_task_ids=task_ids,
        runs=_parse_runs(args.run),
        task_slices=task_slices,
    )
    write_json_atomic(args.output, result)


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
    main()
