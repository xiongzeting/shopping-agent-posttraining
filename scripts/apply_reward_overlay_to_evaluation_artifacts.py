#!/usr/bin/env python3
"""Persist Reward replay classifications into evaluation JSONL and summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shopping_grpo.evaluation.results import summarize_evaluations


PURCHASE_TYPES = {
    "gold_purchase",
    "valid_alternative_purchase",
    "partial_alternative_purchase",
    "wrong_purchase",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def termination_reason(reward_type: str, source: str | None) -> str:
    if reward_type in PURCHASE_TYPES:
        return reward_type
    if reward_type == "guard_rejection":
        return "invalid_action_limit"
    return reward_type or str(source or "")


def apply_row(row: dict, override: dict, replay: dict | None) -> None:
    reward_type = str(override["reward_type"])
    final_reward = float(override["final_reward"])
    reward_valid = bool(override["reward_valid"])
    weighted_score = float(override.get("weighted_score") or 0.0)
    reason = termination_reason(reward_type, override.get("source_termination"))

    reward_and_terminal = row.setdefault("reward_and_terminal", {})
    metrics = reward_and_terminal.setdefault("metrics", {})
    metrics.update(
        {
            "final_reward": final_reward,
            "purchase_success": bool(override["purchase_success"]),
            "reward_type": reward_type,
            "reward_valid": reward_valid,
            "strict_gold_success": bool(override["strict_gold"]),
            "terminal_utility": final_reward,
            "termination_reason": reason,
            "weighted_score": weighted_score,
        }
    )
    terminal = reward_and_terminal.setdefault("terminal", {})
    terminal.update(
        {
            "reward": final_reward,
            "reward_valid": reward_valid,
            "termination_reason": reason,
        }
    )
    detail = terminal.setdefault("reward_detail", {})
    detail.update(
        {
            "reward": final_reward,
            "reward_type": reward_type,
            "reward_valid": reward_valid,
            "terminal_utility": final_reward,
            "termination_reason": reason,
            "weighted_score": weighted_score,
        }
    )

    if replay:
        if replay.get("new_constraint_results") is not None:
            detail["constraint_results"] = replay["new_constraint_results"]
        if replay.get("new_constraint_summary") is not None:
            detail["constraint_summary"] = replay["new_constraint_summary"]
        if replay.get("new_strict_purchase_contract") is not None:
            detail["strict_purchase_contract"] = replay[
                "new_strict_purchase_contract"
            ]
        rubric = row.setdefault("requirement_rubric", {})
        if replay.get("new_constraint_results") is not None:
            rubric["reward_constraint_results"] = replay[
                "new_constraint_results"
            ]
        if replay.get("new_constraint_summary") is not None:
            rubric["reward_constraint_summary"] = replay[
                "new_constraint_summary"
            ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    evaluation = root / "重点/4.评测阶段"
    overlay = load_json(args.overlay.resolve())
    task_ids = [
        int(row["task_id"])
        for row in load_jsonl(root / "data/evaluation/tasks.jsonl")
    ]
    order = {task_id: index for index, task_id in enumerate(task_ids)}
    slices = {
        int(row["task_id"]): row
        for row in load_jsonl(root / "data/evaluation/slices.jsonl")
    }
    targets = {
        label: [evaluation / f"runs/{label}/evaluations.jsonl"]
        for label in ("base", "sft", "grpo50", "grpo100", "grpo230", "qwen38_27b")
    }
    targets["grpo230"].extend(
        [
            root
            / "outputs/evaluation/final240-grpo230-harness-improved-deepseek-v4-pro-20260822/runs/grpo230/evaluations.jsonl",
            evaluation / "GRPO-step230-Final240-160gold/evaluations.jsonl",
        ]
    )
    targets["qwen38_27b"].append(
        root
        / "outputs/evaluation/final240-v24-qwen38-27b-base-nonthinking-deepseek-v4-pro-judge-r1-20260824/runs/qwen38_27b/evaluations.jsonl"
    )

    result = {}
    for label, paths in targets.items():
        overrides = {
            int(row["task_id"]): row for row in overlay["records"][label]
        }
        replay_rows = {
            int(row["task_id"]): row
            for row in load_json(args.replay_dir.resolve() / f"{label}.json")[
                "tasks"
            ]
        }
        written = []
        for path in dict.fromkeys(paths):
            if not path.is_file():
                continue
            rows = load_jsonl(path)
            for row in rows:
                task_id = int(row["task_id"])
                apply_row(row, overrides[task_id], replay_rows.get(task_id))
            rows.sort(key=lambda row: order[int(row["task_id"])])
            write_jsonl(path, rows)
            write_json(
                path.with_name("summary.json"),
                summarize_evaluations(
                    expected_task_ids=task_ids,
                    evaluations=rows,
                    task_slices=slices,
                ),
            )
            written.append(str(path))
        result[label] = written
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
