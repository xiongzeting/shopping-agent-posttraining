#!/usr/bin/env python3
"""Prepare strict, behavior-gated candidates for a final Teacher materialization."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts.collect_sft_data import behavior_gate_reasons
from shopping_grpo.collection.data_gate import action_sequence, trajectory_coverage
from shopping_grpo.collection.sft import (
    acceptance_reasons,
    read_jsonl,
    task_ids_from_jsonl,
    write_jsonl,
)

STRATEGIES = (
    "loop_recovery",
    "near_miss_rejection",
    "option_grounding",
    "terminal_tool_commit",
    "stable",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-teacher", type=Path, required=True)
    parser.add_argument("--stable-raw", type=Path, action="append", default=[])
    parser.add_argument("--loop-raw", type=Path, action="append", default=[])
    parser.add_argument("--near-raw", type=Path, action="append", default=[])
    parser.add_argument("--option-raw", type=Path, action="append", default=[])
    parser.add_argument("--terminal-raw", type=Path, action="append", default=[])
    parser.add_argument("--reused-raw", type=Path, action="append", default=[])
    parser.add_argument("--held-out-tasks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stable-limit",
        type=int,
        default=None,
        help="Optional low-memory cap applied after retaining all previous stable rows.",
    )
    parser.add_argument(
        "--stable-ranks-json",
        type=Path,
        help="Optional task_id-to-retrieval-bucket map for rank-aware stable limiting.",
    )
    parser.add_argument(
        "--stable-rank-min",
        action="append",
        default=[],
        metavar="BUCKET=COUNT",
        help="Minimum rows to retain for a retrieval bucket before general filling.",
    )
    return parser.parse_args()


def parse_rank_minimums(values: list[str]) -> dict[str, int]:
    minimums = {}
    for value in values:
        bucket, separator, count_text = value.partition("=")
        if not separator or not bucket or not count_text.isdigit():
            raise SystemExit(f"invalid --stable-rank-min value: {value!r}")
        minimums[bucket] = int(count_text)
    return minimums


def stable_key(row: dict) -> str:
    payload = f"{int(row['task_id'])}:{row.get('trajectory_id', '')}"
    return hashlib.sha256(payload.encode()).hexdigest()


def quality_score(row: dict) -> int:
    coverage = trajectory_coverage(row)
    weights = {
        "loop_recovery": 10,
        "candidate_comparison": 8,
        "multiple_options": 7,
        "search_reformulation": 6,
        "evidence_verification": 4,
        "guard_recovery": 2,
        "variant_selection": 2,
        "medium_or_long": 1,
        "long": 2,
    }
    return sum(weights.get(name, 0) for name, present in coverage.items() if present)


def strategy_reasons(row: dict, strategy: str) -> list[str]:
    if strategy == "loop_recovery":
        return behavior_gate_reasons(row, "loop_recovery")
    if strategy == "near_miss_rejection":
        return behavior_gate_reasons(row, "near_miss_rejection")
    coverage = trajectory_coverage(row)
    if strategy == "option_grounding" and not coverage["variant_selection"]:
        return ["missing_variant_selection"]
    if strategy == "terminal_tool_commit" and not coverage["explicit_terminal_buy"]:
        return ["missing_explicit_terminal_buy"]
    return []


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite candidate directory: {output}")
    output.mkdir(parents=True)
    held_out = task_ids_from_jsonl(args.held_out_tasks)

    sources: dict[str, list[Path]] = {
        "stable": list(args.stable_raw),
        "loop_recovery": list(args.loop_raw),
        "near_miss_rejection": list(args.near_raw),
        "option_grounding": list(args.option_raw),
        "terminal_tool_commit": list(args.terminal_raw),
    }
    source_rows = defaultdict(Counter)
    candidates: dict[str, dict[int, tuple[tuple, dict, str]]] = {
        strategy: {} for strategy in STRATEGIES
    }
    source_task_ids: dict[str, dict[str, set[int]]] = {
        strategy: defaultdict(set) for strategy in STRATEGIES
    }

    def consider(
        row: dict,
        strategy: str,
        source: Path,
        priority: int,
        *,
        required_existing: bool = False,
    ) -> None:
        task_id = int(row["task_id"])
        label = str(source.resolve())
        source_rows[strategy]["raw"] += 1
        ok, _ = acceptance_reasons(row)
        if not ok or task_id in held_out or not trajectory_coverage(row)["clean_critical_actions"]:
            source_rows[strategy]["strict_or_clean_rejected"] += 1
            return
        reasons = strategy_reasons(row, strategy)
        if reasons:
            source_rows[strategy]["behavior_rejected"] += 1
            return
        source_rows[strategy]["eligible"] += 1
        source_task_ids[strategy][label].add(task_id)
        selected = dict(row)
        selected["_selection_strategy"] = strategy
        if required_existing:
            selected["_required_existing"] = True
        else:
            selected.pop("_required_existing", None)
        key = (priority, -quality_score(selected), stable_key(selected))
        existing = candidates[strategy].get(task_id)
        if existing is None or key < existing[0]:
            candidates[strategy][task_id] = (key, selected, label)

    previous = args.previous_teacher.resolve()
    for row in read_jsonl(previous):
        strategy = str(row.get("_selection_strategy") or "stable")
        if strategy in STRATEGIES:
            consider(
                row,
                strategy,
                previous,
                -100,
                required_existing=True,
            )

    for strategy, paths in sources.items():
        for priority, path in enumerate(paths):
            resolved = path.resolve()
            for row in read_jsonl(resolved):
                consider(row, strategy, resolved, priority)

    for reused in args.reused_raw:
        resolved = reused.resolve()
        for row in read_jsonl(resolved):
            strategy = str(row.get("_selection_strategy") or "stable")
            if strategy in STRATEGIES:
                consider(row, strategy, resolved, 50)

    # Corrective ownership wins over stable ownership. Corrective task IDs are
    # also made unique across strategies in the declared order.
    corrective_rows = []
    corrective_profiles = []
    corrective_ids = set()
    cross_strategy_removed = []
    required_strategy_by_task = {
        task_id: strategy
        for strategy in STRATEGIES
        for task_id, (_, row, _) in candidates[strategy].items()
        if row.get("_required_existing")
    }
    for strategy in STRATEGIES[:-1]:
        ordered = sorted(
            candidates[strategy].values(),
            key=lambda item: item[0],
        )
        for _, row, source in ordered:
            task_id = int(row["task_id"])
            required_strategy = required_strategy_by_task.get(task_id)
            if required_strategy is not None and required_strategy != strategy:
                cross_strategy_removed.append(
                    {
                        "task_id": task_id,
                        "strategy": strategy,
                        "source": source,
                        "reason": f"required_existing_owned_by_{required_strategy}",
                    }
                )
                continue
            if task_id in corrective_ids:
                cross_strategy_removed.append(
                    {"task_id": task_id, "strategy": strategy, "source": source}
                )
                continue
            corrective_ids.add(task_id)
            corrective_rows.append(row)
            corrective_profiles.append(
                {"task_id": task_id, "teacher_strategy": strategy}
            )

    stable_entries = [
        item
        for item in sorted(candidates["stable"].values(), key=lambda item: item[0])
        if int(item[1]["task_id"]) not in corrective_ids
    ]
    stable_rows_before_limit = len(stable_entries)
    if args.stable_limit is not None:
        if args.stable_limit < 1:
            raise SystemExit("--stable-limit must be positive")
        previous_entries = [item for item in stable_entries if item[2] == str(previous)]
        selected_entries = list(previous_entries)
        selected_ids = {int(item[1]["task_id"]) for item in selected_entries}
        sequence_counts = Counter(action_sequence(item[1]) for item in selected_entries)
        remaining = [
            item for item in stable_entries if int(item[1]["task_id"]) not in selected_ids
        ]
        rank_minimums = parse_rank_minimums(args.stable_rank_min)
        rank_map = {}
        if args.stable_ranks_json:
            rank_map = {
                int(task_id): str(bucket)
                for task_id, bucket in json.loads(
                    args.stable_ranks_json.read_text(encoding="utf-8")
                ).items()
            }
        elif rank_minimums:
            raise SystemExit("--stable-rank-min requires --stable-ranks-json")

        retained_ranks = Counter(
            rank_map.get(int(item[1]["task_id"]), "unknown")
            for item in selected_entries
        )
        for bucket, minimum in rank_minimums.items():
            while retained_ranks[bucket] < minimum and len(selected_entries) < min(
                args.stable_limit, len(stable_entries)
            ):
                eligible = [
                    item
                    for item in remaining
                    if rank_map.get(int(item[1]["task_id"])) == bucket
                ]
                if not eligible:
                    raise SystemExit(
                        f"stable rank minimum unavailable: {bucket}="
                        f"{minimum}, retained={retained_ranks[bucket]}"
                    )
                chosen = min(
                    eligible,
                    key=lambda item: (
                        sequence_counts[action_sequence(item[1])],
                        -quality_score(item[1]),
                        item[0],
                    ),
                )
                remaining.remove(chosen)
                selected_entries.append(chosen)
                sequence_counts[action_sequence(chosen[1])] += 1
                retained_ranks[bucket] += 1
        while len(selected_entries) < min(args.stable_limit, len(stable_entries)):
            chosen = min(
                remaining,
                key=lambda item: (
                    sequence_counts[action_sequence(item[1])],
                    -quality_score(item[1]),
                    item[0],
                ),
            )
            remaining.remove(chosen)
            selected_entries.append(chosen)
            sequence_counts[action_sequence(chosen[1])] += 1
        stable_entries = selected_entries
    stable_rows = [row for _, row, _ in stable_entries]

    write_jsonl(output / "stable.jsonl", stable_rows)
    write_jsonl(output / "corrective.jsonl", corrective_rows)
    write_jsonl(output / "corrective_tasks.jsonl", corrective_profiles)

    source_overlaps = {}
    for strategy, grouped in source_task_ids.items():
        labels = list(grouped)
        overlaps = set()
        for index, left in enumerate(labels):
            for right in labels[index + 1 :]:
                overlaps.update(grouped[left] & grouped[right])
        source_overlaps[strategy] = sorted(overlaps)

    strategy_counts = Counter(row["_selection_strategy"] for row in corrective_rows)
    audit = {
        "schema_version": "shopping-teacher-final-candidates-v1",
        "previous_teacher": str(previous),
        "held_out_rows_removed": len(held_out),
        "stable_candidates": len(stable_rows),
        "stable_candidates_before_limit": stable_rows_before_limit,
        "stable_limit": args.stable_limit,
        "stable_rank_minimums": parse_rank_minimums(args.stable_rank_min),
        "corrective_candidates": len(corrective_rows),
        "corrective_strategy_counts": dict(sorted(strategy_counts.items())),
        "source_counters": {
            strategy: dict(sorted(counts.items()))
            for strategy, counts in source_rows.items()
        },
        "source_overlap_task_ids": source_overlaps,
        "cross_strategy_removed": cross_strategy_removed,
        "unique_stable_task_ids": len({int(row["task_id"]) for row in stable_rows}),
        "unique_corrective_task_ids": len(corrective_ids),
        "stable_corrective_overlap": len(
            {int(row["task_id"]) for row in stable_rows} & corrective_ids
        ),
        "corrective_action_sequences": len(
            {action_sequence(row) for row in corrective_rows}
        ),
    }
    (output / "candidate_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
