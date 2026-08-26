#!/usr/bin/env python3
"""Merge selected missing-task reruns and their Judge rows into Final-240 artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from shopping_grpo.evaluation.pipeline import evaluate_trajectories
from shopping_grpo.evaluation.results import summarize_evaluations


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    path.write_text(payload, encoding="utf-8")


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def one(path: Path) -> dict:
    rows = load_jsonl(path)
    if len(rows) != 1:
        raise ValueError(f"expected one row in {path}, found {len(rows)}")
    return rows[0]


def replace_task(rows: list[dict], replacement: dict, task_order: dict[int, int]) -> list[dict]:
    task_id = int(replacement["task_id"])
    merged = [row for row in rows if int(row["task_id"]) != task_id]
    merged.append(replacement)
    merged.sort(key=lambda row: task_order[int(row["task_id"])])
    return merged


def replace_judge(rows: list[dict], replacement: dict, task_order: dict[int, int]) -> list[dict]:
    task_id = int(replacement["task_id"])
    merged = [row for row in rows if int(row["task_id"]) != task_id]
    merged.append(replacement)
    merged.sort(key=lambda row: task_order[int(row["task_id"])])
    return merged


def replace_checkpoint(
    rows: list[dict],
    replacement: dict,
    *,
    old_trajectory_id: str | None,
) -> list[dict]:
    new_id = str(replacement.get("_checkpoint_key") or replacement.get("trajectory_id"))
    blocked = {new_id}
    if old_trajectory_id:
        blocked.add(str(old_trajectory_id))
    merged = [
        row
        for row in rows
        if str(row.get("_checkpoint_key") or row.get("trajectory_id")) not in blocked
    ]
    merged.append(replacement)
    return merged


def patch_grpo_loop(trajectory: dict) -> dict:
    row = json.loads(json.dumps(trajectory, ensure_ascii=False))
    step_count = len(row.get("steps") or [])
    if step_count != 14:
        raise ValueError(f"selected GRPO task must be the 14-step run, got {step_count}")
    terminal = row.setdefault("terminal_result", {})
    detail = terminal.setdefault("reward_detail", {})
    detail.update(
        {
            "base_terminal_utility": -0.6,
            "reward": -0.6,
            "reward_type": "repeat_loop",
            "reward_valid": True,
            "step_count": step_count,
            "step_penalty": 0.0,
            "terminal_utility": -0.6,
            "termination_reason": "repeat_loop",
            "termination_subreason": detail.get("termination_subreason")
            or "no_progress_loop",
        }
    )
    terminal.update(
        {
            "done": True,
            "over": True,
            "reward": -0.6,
            "reward_valid": True,
            "termination_reason": "repeat_loop",
        }
    )
    row["final_reward"] = -0.6
    row["done"] = True
    row["status"] = "done"
    return row


def patch_base_guard(trajectory: dict) -> dict:
    row = json.loads(json.dumps(trajectory, ensure_ascii=False))
    step_count = len(row.get("steps") or [])
    if row.get("status") != "invalid_action_limit":
        raise ValueError("selected Base task is not invalid_action_limit")
    detail = {
        "base_terminal_utility": -0.8,
        "reward": -0.8,
        "reward_type": "guard_rejection",
        "reward_valid": True,
        "step_count": step_count,
        "step_penalty": 0.0,
        "terminal_utility": -0.8,
        "termination_reason": "invalid_action_limit",
        "termination_subreason": "too_many_guard_rejections",
        "weighted_score": 0.0,
    }
    row["terminal_result"] = {
        "done": True,
        "over": True,
        "progress": {},
        "purchase": {},
        "reward": -0.8,
        "reward_detail": detail,
        "reward_valid": True,
        "termination_reason": "invalid_action_limit",
    }
    row["final_reward"] = -0.8
    row["done"] = True
    return row


def task_evaluation(
    *,
    trajectory: dict,
    actor: dict,
    rubric: dict,
    judge: dict,
    task_slice: dict,
) -> dict:
    task_id = int(trajectory["task_id"])
    artifacts = evaluate_trajectories(
        expected_task_ids=[task_id],
        trajectories=[trajectory],
        actor=actor,
        task_slices={task_id: task_slice},
        rubric_bundles=[rubric],
        judge_results=[judge],
    )
    return artifacts["evaluations"][0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--judge-output", type=Path, required=True)
    parser.add_argument("--base-trajectory", type=Path, required=True)
    parser.add_argument("--grpo-trajectory", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    evaluation = root / "重点/4.评测阶段"
    judge_output = args.judge_output.resolve()
    step230_source = (
        root
        / "outputs/evaluation/final240-grpo230-harness-improved-deepseek-v4-pro-20260822"
    )
    grpo_archive = evaluation / "GRPO-step230-Final240-160gold"

    task_rows = load_jsonl(root / "data/evaluation/tasks.jsonl")
    task_ids = [int(row["task_id"]) for row in task_rows]
    task_order = {task_id: index for index, task_id in enumerate(task_ids)}
    slices = {
        int(row["task_id"]): row
        for row in load_jsonl(root / "data/evaluation/slices.jsonl")
    }
    rubrics = {
        int(row["task_id"]): row for row in load_jsonl(evaluation / "rubrics.jsonl")
    }

    base_raw = patch_base_guard(one(args.base_trajectory.resolve()))
    grpo_raw = patch_grpo_loop(one(args.grpo_trajectory.resolve()))
    if int(base_raw["task_id"]) != 419 or int(grpo_raw["task_id"]) != 173:
        raise ValueError("unexpected selected task IDs")

    base_judge = one(judge_output / "judges-base.jsonl")
    grpo_judge = one(judge_output / "judges-grpo230.jsonl")

    base_eval_path = evaluation / "runs/base/evaluations.jsonl"
    grpo_eval_paths = [
        evaluation / "runs/grpo230/evaluations.jsonl",
        step230_source / "runs/grpo230/evaluations.jsonl",
        grpo_archive / "evaluations.jsonl",
    ]
    base_evaluations = load_jsonl(base_eval_path)
    grpo_reference_evaluations = load_jsonl(grpo_eval_paths[0])
    base_actor = dict(base_evaluations[0]["actor"])
    grpo_actor = dict(grpo_reference_evaluations[0]["actor"])

    base_evaluation = task_evaluation(
        trajectory=base_raw,
        actor=base_actor,
        rubric=rubrics[419],
        judge=base_judge,
        task_slice=slices[419],
    )
    grpo_evaluation = task_evaluation(
        trajectory=grpo_raw,
        actor=grpo_actor,
        rubric=rubrics[173],
        judge=grpo_judge,
        task_slice=slices[173],
    )

    old_grpo_row = next(
        row for row in grpo_reference_evaluations if int(row["task_id"]) == 173
    )
    old_grpo_trajectory_id = str(old_grpo_row["trajectory_id"])

    affected = [
        base_eval_path,
        evaluation / "runs/base/summary.json",
        evaluation / "judges-base.jsonl",
        evaluation / "calls/judges-base.jsonl",
        evaluation / "checkpoints/judges-base.jsonl",
        evaluation / "judges-grpo230.jsonl",
        evaluation / "calls/judges-grpo230.jsonl",
        evaluation / "checkpoints/judges-grpo230.jsonl",
        grpo_archive / "trajectories.jsonl",
        grpo_archive / "evaluations.jsonl",
        grpo_archive / "summary.json",
    ]
    for eval_path in grpo_eval_paths[:2]:
        affected.extend([eval_path, eval_path.with_name("summary.json")])
    affected.extend(
        [
            step230_source / "judges-grpo230.jsonl",
            step230_source / "calls/judges-grpo230.jsonl",
            step230_source / "checkpoints/judges-grpo230.jsonl",
        ]
    )
    backup = (
        evaluation
        / "补跑缺失任务-20260824"
        / ("merge-backup-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    )
    for path in dict.fromkeys(affected):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        destination = backup / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)

    base_evaluations = replace_task(base_evaluations, base_evaluation, task_order)
    write_jsonl(base_eval_path, base_evaluations)
    write_json(
        evaluation / "runs/base/summary.json",
        summarize_evaluations(
            expected_task_ids=task_ids,
            evaluations=base_evaluations,
            task_slices=slices,
        ),
    )

    for path in grpo_eval_paths:
        rows = replace_task(load_jsonl(path), grpo_evaluation, task_order)
        write_jsonl(path, rows)
        write_json(
            path.with_name("summary.json"),
            summarize_evaluations(
                expected_task_ids=task_ids,
                evaluations=rows,
                task_slices=slices,
            ),
        )

    full_grpo = replace_task(
        load_jsonl(grpo_archive / "trajectories.jsonl"), grpo_raw, task_order
    )
    write_jsonl(grpo_archive / "trajectories.jsonl", full_grpo)

    write_jsonl(
        evaluation / "judges-base.jsonl",
        replace_judge(
            load_jsonl(evaluation / "judges-base.jsonl"), base_judge, task_order
        ),
    )
    for path in [
        evaluation / "judges-grpo230.jsonl",
        step230_source / "judges-grpo230.jsonl",
    ]:
        write_jsonl(path, replace_judge(load_jsonl(path), grpo_judge, task_order))

    mini_base_call = one(judge_output / "calls/judges-base.jsonl")
    mini_grpo_call = one(judge_output / "calls/judges-grpo230.jsonl")
    mini_base_checkpoint = one(judge_output / "checkpoints/judges-base.jsonl")
    mini_grpo_checkpoint = one(judge_output / "checkpoints/judges-grpo230.jsonl")
    for path, replacement, old_id in [
        (evaluation / "calls/judges-base.jsonl", mini_base_call, None),
        (evaluation / "checkpoints/judges-base.jsonl", mini_base_checkpoint, None),
        (evaluation / "calls/judges-grpo230.jsonl", mini_grpo_call, old_grpo_trajectory_id),
        (
            evaluation / "checkpoints/judges-grpo230.jsonl",
            mini_grpo_checkpoint,
            old_grpo_trajectory_id,
        ),
        (step230_source / "calls/judges-grpo230.jsonl", mini_grpo_call, old_grpo_trajectory_id),
        (
            step230_source / "checkpoints/judges-grpo230.jsonl",
            mini_grpo_checkpoint,
            old_grpo_trajectory_id,
        ),
    ]:
        rows = load_jsonl(path) if path.is_file() else []
        write_jsonl(
            path,
            replace_checkpoint(rows, replacement, old_trajectory_id=old_id),
        )

    merged_dir = evaluation / "补跑缺失任务-20260824/merged-selected"
    merged_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(merged_dir / "base419-trajectories.jsonl", [base_raw])
    write_jsonl(merged_dir / "grpo230-task173-trajectories.jsonl", [grpo_raw])
    write_jsonl(merged_dir / "base419-evaluations.jsonl", [base_evaluation])
    write_jsonl(merged_dir / "grpo230-task173-evaluations.jsonl", [grpo_evaluation])
    write_json(
        merged_dir / "merge-report.json",
        {
            "backup": str(backup),
            "base": {
                "task_id": 419,
                "trajectory_id": base_raw["trajectory_id"],
                "reward_type": "guard_rejection",
                "final_reward": -0.8,
                "judge_status": base_judge["judge_status"],
            },
            "grpo230": {
                "task_id": 173,
                "trajectory_id": grpo_raw["trajectory_id"],
                "steps": 14,
                "reward_type": "repeat_loop",
                "final_reward": -0.6,
                "judge_status": grpo_judge["judge_status"],
            },
        },
    )
    print(json.dumps({"backup": str(backup), "merged": str(merged_dir)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
