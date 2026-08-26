"""Replay frozen evaluation terminals with the current Reward v4."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from web_agent_site.engine.reward import (
    DEFAULT_REWARDS,
    calculate_step_penalty,
    evaluate_purchase,
    fixed_termination,
)
from web_agent_site.engine.reward_features import compile_reward_features


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "重点/4.评测阶段"
LABELS = ("base", "sft", "grpo50", "grpo100", "grpo230", "qwen38_27b")
DISPLAY = {
    "base": "Base·Harness v1",
    "sft": "SFT·Harness v1",
    "grpo50": "GRPO50·Harness v1",
    "grpo100": "GRPO100·Harness v1",
    "grpo230": "GRPO230·Harness v2",
    "qwen38_27b": "Qwen3.8-27B·Harness v2",
}
PURCHASE_TYPES = {
    "wrong_purchase",
    "valid_alternative_purchase",
    "partial_alternative_purchase",
}
FIXED_BASE_REWARDS = {
    "gold_purchase": "gold_purchase",
    "valid_alternative_purchase": "valid_alternative_purchase",
    "wrong_purchase": "wrong_purchase",
    "assistant_final": "assistant_final",
    "guard_rejection": "assistant_final",
    "early_abstain": "early_abstain",
    "max_steps": "max_steps",
    "repeat_loop": "repeat_loop",
    "reward_unverifiable": "reward_unverifiable",
}
REVIEWED_FIXES = (
    "Task 89：气垫/BB 霜是效果类比，不再作为粉底液替代商品的类型硬约束。",
    "Task 308：老人是购买对象背景，不再要求提醒器商品文本必须出现“老人”。",
    "Task 336：女生是使用者语境，不再作为睫毛膏能力属性。",
    "Task 652：比赛/训练是需求动机，硬约束保留耐磨、防滑、男女适用与规格。",
    "Task 832：准备去旅游是叙事背景，不再误算为可评分 Soft。",
    "“预期/预计/预估”修饰的价格边界按 Soft 处理；“必须/绝对/不超过”等明确边界仍为 Hard。",
    "补齐可审计同义表达与公开规格匹配，修正专利白参菌、高级/高档、Mont Bell、冰飘/飘花、油丸/胶囊、DHA、颜色和尺码等确定性误判。",
)


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _reward_metrics(row: dict) -> dict:
    return (
        (row.get("reward_and_terminal") or {}).get("metrics")
        if isinstance(row, dict)
        else {}
    ) or {}


def _terminal(row: dict) -> dict:
    return (
        (row.get("reward_and_terminal") or {}).get("terminal")
        if isinstance(row, dict)
        else {}
    ) or {}


def _step_count(row: dict) -> int:
    return int(
        (
            (row.get("deterministic") or {}).get("actions_and_efficiency")
            or {}
        ).get("executed_tool_steps")
        or 0
    )


def _replay_non_purchase(row: dict, metrics: dict) -> tuple[str, float, dict | None]:
    step_count = _step_count(row)
    legality = (row.get("deterministic") or {}).get("legality") or {}
    old_type = str(metrics.get("reward_type") or "unknown")
    termination_reason = str(metrics.get("termination_reason") or "")

    if bool(legality.get("invalid_action_limit")):
        reward = round(
            DEFAULT_REWARDS["assistant_final"]
            + calculate_step_penalty(step_count),
            10,
        )
        return (
            "guard_rejection",
            reward,
            {
                "reward_valid": True,
                "weighted_score": 0.0,
                "base_terminal_utility": DEFAULT_REWARDS["assistant_final"],
                "step_count": step_count,
                "step_penalty": calculate_step_penalty(step_count),
                "termination_reason": "invalid_action_limit",
            },
        )

    fixed_reason = termination_reason if termination_reason in {
        "assistant_final",
        "repeat_loop",
        "max_steps",
    } else old_type
    if fixed_reason in {"assistant_final", "repeat_loop", "max_steps"}:
        result = fixed_termination(fixed_reason, step_count=step_count)
        return result.reward_type, float(result.reward), result.to_dict()

    return old_type, float(metrics.get("final_reward") or 0.0), None


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def _classification_audit(model_reports: dict[str, dict]) -> dict:
    errors = []
    record_counts = Counter()
    unique_tasks: dict[str, set[int]] = defaultdict(set)
    partial_rows = []
    for label, model in model_reports.items():
        for task in model["tasks"]:
            reward_type = str(task.get("new_reward_type") or "unknown")
            if reward_type not in PURCHASE_TYPES:
                continue
            results = task.get("new_constraint_results") or []
            hard = [row for row in results if row.get("strength") == "hard"]
            soft = [row for row in results if row.get("strength") == "soft"]
            hard_fail = [row for row in hard if row.get("status") == "fail"]
            hard_unknown = [
                row for row in hard if row.get("status") == "unverifiable"
            ]
            soft_fail = [row for row in soft if row.get("status") == "fail"]
            row_errors = []
            if reward_type == "wrong_purchase" and not hard_fail:
                row_errors.append("wrong_without_hard_fail")
            if reward_type == "valid_alternative_purchase" and (
                hard_fail or hard_unknown or soft_fail
            ):
                row_errors.append("valid_with_failed_contract")
            if reward_type == "partial_alternative_purchase" and (
                hard_fail or hard_unknown or not soft_fail
            ):
                row_errors.append("partial_contract_mismatch")
            if row_errors:
                errors.append(
                    {
                        "label": label,
                        "task_id": task["task_id"],
                        "errors": row_errors,
                    }
                )
            record_counts[reward_type] += 1
            unique_tasks[reward_type].add(int(task["task_id"]))
            if reward_type == "partial_alternative_purchase":
                partial_rows.append(
                    {
                        "label": label,
                        "display": DISPLAY[label],
                        "task_id": int(task["task_id"]),
                        "soft_failures": [
                            {
                                "constraint_type": row.get("constraint_type"),
                                "expected": row.get("expected"),
                                "actual": row.get("actual"),
                                "query_quote": row.get("query_quote"),
                            }
                            for row in soft_fail
                        ],
                    }
                )
    return {
        "invariant_errors": errors,
        "invariants_passed": not errors,
        "record_counts": dict(sorted(record_counts.items())),
        "unique_task_counts": {
            reward_type: len(task_ids)
            for reward_type, task_ids in sorted(unique_tasks.items())
        },
        "partial_rows": partial_rows,
        "reviewed_fixes": list(REVIEWED_FIXES),
    }


def _short_expected(value: object) -> str:
    if isinstance(value, dict) and value.get("source_text"):
        return str(value["source_text"])
    return str(value)


def replay_label(
    *,
    label: str,
    task_ids: list[int],
    target_products: list[dict],
    products_by_asin: dict[str, dict],
) -> dict:
    evaluations_path = REFERENCE / "runs" / label / "evaluations.jsonl"
    summary_path = REFERENCE / "runs" / label / "summary.json"
    evaluations = _load_jsonl(evaluations_path)
    by_task = {int(row["task_id"]): row for row in evaluations}
    if len(by_task) != len(evaluations):
        raise ValueError(f"{label} has duplicate task IDs")

    expected = set(task_ids)
    unexpected = sorted(set(by_task) - expected)
    if unexpected:
        raise ValueError(f"{label} contains unexpected task IDs: {unexpected}")

    old_counts = Counter()
    new_counts = Counter()
    transitions = Counter()
    old_total = 0.0
    new_total = 0.0
    rows = []
    missing_products = []

    for task_id in task_ids:
        evaluation = by_task.get(task_id)
        if evaluation is None:
            old_type = new_type = "unknown"
            old_reward = new_reward = 0.0
            rows.append(
                {
                    "task_id": task_id,
                    "missing_evaluation": True,
                    "old_reward_type": old_type,
                    "new_reward_type": new_type,
                    "old_reward": old_reward,
                    "new_reward": new_reward,
                    "reward_delta": 0.0,
                }
            )
        else:
            metrics = _reward_metrics(evaluation)
            terminal = _terminal(evaluation)
            purchase = terminal.get("purchase") or {}
            old_type = str(metrics.get("reward_type") or "unknown")
            old_reward = float(metrics.get("final_reward") or 0.0)
            purchased_asin = str(purchase.get("asin") or "")
            new_type = old_type
            new_reward = old_reward
            new_payload = None

            if purchased_asin:
                purchased_product = products_by_asin.get(purchased_asin)
                if purchased_product is None:
                    missing_products.append(
                        {"task_id": task_id, "purchased_asin": purchased_asin}
                    )
                    new_type = "replay_product_missing"
                    new_reward = 0.0
                else:
                    target_product = target_products[task_id]
                    instruction = target_product["instructions"][0]
                    features = compile_reward_features(instruction, target_product)
                    goal = {
                        "asin": target_product["asin"],
                        "category": target_product["category"],
                        "instruction_text": instruction["instruction"],
                        **features,
                    }
                    step_count = _step_count(evaluation)
                    result = evaluate_purchase(
                        purchased_product,
                        goal,
                        selected_options=purchase.get("options") or {},
                        price=purchase.get("price"),
                        step_count=step_count,
                    )
                    new_payload = result.to_dict()
                    new_type = result.reward_type
                    new_reward = float(result.reward)
            else:
                new_type, new_reward, new_payload = _replay_non_purchase(
                    evaluation,
                    metrics,
                )

            rows.append(
                {
                    "task_id": task_id,
                    "query": target_products[task_id]["instructions"][0][
                        "instruction"
                    ],
                    "trajectory_id": evaluation.get("trajectory_id"),
                    "missing_evaluation": False,
                    "purchase_asin": purchased_asin or None,
                    "step_count": _step_count(evaluation),
                    "source_termination": metrics.get("termination_reason"),
                    "old_reward_type": old_type,
                    "new_reward_type": new_type,
                    "old_reward": old_reward,
                    "new_reward": new_reward,
                    "reward_delta": new_reward - old_reward,
                    "new_reward_valid": (
                        new_payload.get("reward_valid")
                        if isinstance(new_payload, dict)
                        else metrics.get("reward_valid")
                    ),
                    "new_weighted_score": (
                        new_payload.get("weighted_score")
                        if isinstance(new_payload, dict)
                        else metrics.get("weighted_score")
                    ),
                    "new_constraint_summary": (
                        new_payload.get("constraint_summary")
                        if isinstance(new_payload, dict)
                        else None
                    ),
                    "new_constraint_audit_summary": (
                        new_payload.get("constraint_audit_summary")
                        if isinstance(new_payload, dict)
                        else None
                    ),
                    "new_constraint_results": (
                        new_payload.get("constraint_results")
                        if isinstance(new_payload, dict)
                        else None
                    ),
                    "new_constraint_audit_results": (
                        new_payload.get("constraint_audit_results")
                        if isinstance(new_payload, dict)
                        else None
                    ),
                    "new_strict_purchase_contract": (
                        (new_payload.get("evidence") or {}).get(
                            "strict_purchase_contract"
                        )
                        if isinstance(new_payload, dict)
                        else None
                    ),
                }
            )

        old_counts[old_type] += 1
        new_counts[new_type] += 1
        transitions[(old_type, new_type)] += 1
        old_total += old_reward
        new_total += new_reward

    reference_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    reference_reward = reference_summary["reward_and_terminal"]
    changed = [
        row
        for row in rows
        if row["old_reward_type"] != row["new_reward_type"]
        or abs(float(row["reward_delta"])) > 1e-12
    ]
    type_changed = [
        row
        for row in rows
        if row["old_reward_type"] != row["new_reward_type"]
    ]
    numeric_only_changed = [
        row
        for row in rows
        if row["old_reward_type"] == row["new_reward_type"]
        and abs(float(row["reward_delta"])) > 1e-12
    ]
    denominator = len(task_ids)
    old_mean = old_total / denominator
    new_mean = new_total / denominator
    return {
        "label": label,
        "display": DISPLAY[label],
        "evaluation_path": str(evaluations_path),
        "evaluation_sha256": _file_hash(evaluations_path),
        "expected_tasks": denominator,
        "completed_evaluations": len(evaluations),
        "missing_task_ids": sorted(expected - set(by_task)),
        "old_reward_type_counts": dict(sorted(old_counts.items())),
        "new_reward_type_counts": dict(sorted(new_counts.items())),
        "transition_counts": {
            f"{old}->{new}": count
            for (old, new), count in sorted(transitions.items())
            if old != new
        },
        "old_total_reward": old_total,
        "new_total_reward": new_total,
        "old_mean_reward": old_mean,
        "reference_summary_mean_reward": float(
            reference_reward["mean_final_reward_fixed_denominator"]
        ),
        "old_mean_matches_reference": abs(
            old_mean
            - float(reference_reward["mean_final_reward_fixed_denominator"])
        )
        < 1e-10,
        "new_mean_reward": new_mean,
        "mean_reward_delta": new_mean - old_mean,
        "changed_task_count": len(changed),
        "type_changed_task_count": len(type_changed),
        "numeric_only_changed_task_count": len(numeric_only_changed),
        "changed_task_ids": [row["task_id"] for row in changed],
        "missing_products": missing_products,
        "changes": changed,
        "tasks": rows,
    }


def _overlay_payload(model_reports: dict[str, dict]) -> dict:
    summaries = {}
    records = {}
    strict_by_label = {}
    for label in LABELS:
        model_rows = model_reports[label]["tasks"]
        output_rows = []
        counts = Counter()
        for row in model_rows:
            reward_type = str(row.get("new_reward_type") or "unknown")
            reward_valid = bool(row.get("new_reward_valid"))
            strict_gold = reward_valid and reward_type == "gold_purchase"
            purchase_success = reward_valid and reward_type in {
                "gold_purchase",
                "valid_alternative_purchase",
            }
            output = {
                "task_id": int(row["task_id"]),
                "reward_type": reward_type,
                "reward_valid": reward_valid,
                "final_reward": float(row.get("new_reward") or 0.0),
                "weighted_score": float(row.get("new_weighted_score") or 0.0),
                "strict_gold": strict_gold,
                "purchase_success": purchase_success,
                "source_termination": row.get("source_termination"),
            }
            output_rows.append(output)
            counts[reward_type] += 1
        denominator = len(output_rows)
        summaries[label] = {
            "strict_gold": sum(row["strict_gold"] for row in output_rows),
            "purchase_success": sum(
                row["purchase_success"] for row in output_rows
            ),
            "reward_valid": sum(row["reward_valid"] for row in output_rows),
            "mean_final_reward": sum(
                row["final_reward"] for row in output_rows
            )
            / denominator,
            "mean_weighted_score": sum(
                row["weighted_score"] for row in output_rows
            )
            / denominator,
            "reward_type_counts": dict(sorted(counts.items())),
        }
        records[label] = output_rows
        strict_by_label[label] = {
            row["task_id"]: row["strict_gold"] for row in output_rows
        }

    pairwise = {}
    for left, right in zip(LABELS, LABELS[1:]):
        counts = Counter()
        for task_id, left_success in strict_by_label[left].items():
            right_success = strict_by_label[right][task_id]
            counts[
                f"{'success' if left_success else 'failure'}_to_"
                f"{'success' if right_success else 'failure'}"
            ] += 1
        pairwise[f"{left}_to_{right}"] = dict(sorted(counts.items()))

    return {
        "schema_version": "reward-v4-six-model-overlay-v2",
        "reward_policy": {
            "gold_purchase": DEFAULT_REWARDS["gold_purchase"],
            "valid_alternative_purchase": DEFAULT_REWARDS[
                "valid_alternative_purchase"
            ],
            "partial_alternative_purchase": (
                "0.5 + 0.3 * soft_score"
            ),
            "assistant_final": DEFAULT_REWARDS["assistant_final"],
            "guard_rejection": DEFAULT_REWARDS["assistant_final"],
            "repeat_loop": DEFAULT_REWARDS["repeat_loop"],
            "step_penalty_start": 16,
        },
        "summaries": summaries,
        "pairwise_strict_gold": pairwise,
        "records": records,
    }


def _numeric_policy_audit(model_reports: dict[str, dict]) -> dict:
    errors = []
    checked = Counter()
    for label, model in model_reports.items():
        for row in model["tasks"]:
            reward_type = str(row.get("new_reward_type") or "unknown")
            if reward_type == "unknown":
                continue
            step_count = int(row.get("step_count") or 0)
            reward_valid = bool(row.get("new_reward_valid"))
            penalty = calculate_step_penalty(step_count) if reward_valid else 0.0
            if reward_type == "partial_alternative_purchase":
                contract = row.get("new_strict_purchase_contract") or {}
                soft_score = float(contract.get("soft_score") or 0.0)
                base = (
                    DEFAULT_REWARDS["partial_purchase_base"]
                    + DEFAULT_REWARDS["partial_purchase_scale"] * soft_score
                )
            else:
                reward_key = FIXED_BASE_REWARDS.get(reward_type)
                if reward_key is None:
                    continue
                base = DEFAULT_REWARDS[reward_key]
            expected = round(float(base) + penalty, 10)
            actual = float(row.get("new_reward") or 0.0)
            checked[reward_type] += 1
            if abs(actual - expected) > 1e-9:
                errors.append(
                    {
                        "label": label,
                        "task_id": row["task_id"],
                        "reward_type": reward_type,
                        "step_count": step_count,
                        "expected": expected,
                        "actual": actual,
                    }
                )
    return {
        "passed": not errors,
        "checked_counts": dict(sorted(checked.items())),
        "errors": errors,
    }


def _recalculation_payload(
    report: dict,
    overlay: dict,
) -> dict:
    return {
        "schema_version": "dashboard-reward-v4-recalculation-v2",
        "created_at": report["generated_at"],
        "scope": "reward_and_classification_replay",
        "dashboard_models": list(LABELS),
        "denominator_per_model": report["task_count"],
        "method": {
            "purchase_trajectories": (
                "Re-evaluated with current Reward v4 using the frozen query, "
                "recorded purchase, selected options, actual price, and executed steps."
            ),
            "partial_alternative_purchase": (
                "Hard constraints all pass and at least one verifiable Soft fails; "
                "base utility = 0.5 + 0.3 * soft_score."
            ),
            "fixed_terminations": (
                "assistant_final, repeat_loop, and max_steps were replayed with "
                "their current base utilities and the step penalty beginning at step 16."
            ),
            "guard_rejection": (
                "invalid_action_limit is classified as guard_rejection with base "
                "utility -0.8 plus the same step penalty."
            ),
            "judge_and_rubric": (
                "Unchanged; no Rubric or LLM-as-a-Judge call was repeated."
            ),
        },
        "reward_policy": overlay["reward_policy"],
        "results": overlay["summaries"],
    }


def _markdown(report: dict) -> str:
    model_reports = report["models"]
    audit = report["classification_audit"]
    reward_types = sorted(
        {
            reward_type
            for model in model_reports.values()
            for counts in (
                model["old_reward_type_counts"],
                model["new_reward_type_counts"],
            )
            for reward_type in counts
        }
    )
    lines = [
        "# Reward v4 六模型独立重放审计",
        "",
        "> 当前目录已按最新 Reward v4 覆盖重算；Rubric 与 Judge 原结果保持不变。",
        "",
        "## 最终分类规则与审查结论",
        "",
        "- Hard 失败 → `wrong_purchase`；Hard 无法核验 → `reward_unverifiable`。",
        "- 目标 ASIN 且 Hard 全满足 → `gold_purchase`。",
        "- 替代商品、Hard 全满足、可核验 Soft 无失败 → `valid_alternative_purchase`。",
        "- 替代商品、Hard 全满足、至少一个可核验 Soft 失败 → `partial_alternative_purchase`，基础奖励为 `0.5 + 0.3 × soft_score`；Soft 的 `unverifiable` 不会误触发 Partial。",
        "- Action Guard/assistant_final 基础奖励均为 `-0.8`，repeat_loop 基础奖励为 `-0.6`；三者均叠加从第 16 步开始的步数惩罚。",
        "- `purchase_success` 仍只统计 Gold 与 Valid，不包含 Partial。",
        f"- 分类不变量审计：{'通过，0 条违规' if audit['invariants_passed'] else '失败'}。",
        "- 六模型购买结果记录："
        f"Wrong {audit['record_counts'].get('wrong_purchase', 0)}、"
        f"Valid {audit['record_counts'].get('valid_alternative_purchase', 0)}、"
        f"Partial {audit['record_counts'].get('partial_alternative_purchase', 0)}；"
        "对应唯一任务数分别为 "
        f"{audit['unique_task_counts'].get('wrong_purchase', 0)}、"
        f"{audit['unique_task_counts'].get('valid_alternative_purchase', 0)}、"
        f"{audit['unique_task_counts'].get('partial_alternative_purchase', 0)}。",
        "",
        "### 逐题审查后修正的确定性误判",
        "",
    ]
    lines.extend(f"- {item}" for item in audit["reviewed_fixes"])
    lines.extend(
        [
            "",
            "### 最终 Partial 明细",
            "",
            "以下条目均已确认 Hard 全通过，且至少一个可核验 Soft 失败：",
            "",
        ]
    )
    for row in audit["partial_rows"]:
        failures = "；".join(
            f"{item.get('query_quote') or item.get('constraint_type')}："
            f"期望 {_short_expected(item.get('expected'))}，实际 {item.get('actual')}"
            for item in row["soft_failures"]
        )
        lines.append(
            f"- {row['display']} / Task {row['task_id']}：{failures}"
        )
    lines.extend(
        [
            "",
        "## 平均 Reward 变化",
        "",
        "| 模型 | 旧平均 Reward | 新平均 Reward | 变化 | 类型变化 | 仅数值变化 |",
        "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label in LABELS:
        model = model_reports[label]
        lines.append(
            f"| {model['display']} | {_fmt(model['old_mean_reward'])} | "
            f"{_fmt(model['new_mean_reward'])} | "
            f"{model['mean_reward_delta']:+.6f} | "
            f"{model['type_changed_task_count']} | "
            f"{model['numeric_only_changed_task_count']} |"
        )

    lines.extend(
        [
            "",
            "## Reward 类型分布（旧 → 新）",
            "",
            "| 模型 | " + " | ".join(reward_types) + " |",
            "|---|" + "---:|" * len(reward_types),
        ]
    )
    for label in LABELS:
        model = model_reports[label]
        cells = []
        for reward_type in reward_types:
            old = int(model["old_reward_type_counts"].get(reward_type, 0))
            new = int(model["new_reward_type_counts"].get(reward_type, 0))
            cells.append(str(old) if old == new else f"{old}→{new}")
        lines.append(f"| {model['display']} | " + " | ".join(cells) + " |")

    lines.extend(["", "## 发生类型迁移的任务", ""])
    for label in LABELS:
        model = model_reports[label]
        lines.append(f"### {model['display']}")
        lines.append("")
        if not model["transition_counts"]:
            lines.append("无类型迁移。")
        else:
            for transition, count in model["transition_counts"].items():
                task_ids = [
                    str(row["task_id"])
                    for row in model["changes"]
                    if f"{row['old_reward_type']}->{row['new_reward_type']}"
                    == transition
                ]
                lines.append(
                    f"- `{transition}`：{count} 条；Task {', '.join(task_ids)}"
                )
        lines.append("")

    lines.extend(
        [
            "## 哪些指标会受本次 Reward 变化影响",
            "",
            "直接需要更新：Reward 类型、严格 Gold/购买成功/Reward 有效率等类型派生指标，以及单题与平均 Reward。",
            "",
            "不需要更新：步骤、Token、上下文、时延、Guard、Rubric 状态和五维 LLM-as-a-Judge；这些输入没有因 Reward 代码改变。",
            "",
            "另外，购买任务的确定性约束结果、Hard/Soft 计数、weighted score 和 Reward/Rubric disagreement 属于 Reward 证据派生字段；如果正式刷新评测 JSON，也应随 Reward 一并重算，但不需要重新调用 LLM。",
            "",
            "## 审计信息",
            "",
            f"- 生成时间：{report['generated_at']}",
            f"- Reward 代码提交：`{report['reward_commit']}`",
            f"- Final-240 任务数：{report['task_count']}",
            f"- 商品缺失总数：{report['missing_product_count']}",
            "- 六模型旧平均 Reward 均与原 `summary.json` 精确一致。",
            "- Final-240 Gold 商品回放："
            f"{report['semantics_audit'].get('gold_purchase', 0)}/"
            f"{report['task_count']}；会阻断 Reward 的待复核语义："
            f"{report['semantics_audit'].get('unresolved_scored_semantics', 0)}；"
            "未覆盖语义分句："
            f"{report['semantics_audit'].get('uncovered_semantic_marker_clauses', 0)}。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reward-commit", required=True)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--semantics-audit", type=Path)
    parser.add_argument("--overlay-path", type=Path)
    parser.add_argument("--recalculation-path", type=Path)
    parser.add_argument(
        "--task-overrides",
        type=Path,
        default=REFERENCE / "frozen-reward-task-overrides.jsonl",
        help=(
            "Frozen Final-240 task snapshots keyed by task_id. These override "
            "mutable dataset rows so Reward replay uses the query and target "
            "product that generated the evaluated trajectory."
        ),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    task_ids = [
        int(row["task_id"])
        for row in _load_jsonl(ROOT / "data/evaluation/slices.jsonl")
    ]
    task_id_set = set(task_ids)
    target_products = _load_jsonl(
        ROOT / "data/shopsimulator_official/fine_items_eval_standard.jsonl"
    )
    task_override_ids = []
    if args.task_overrides.is_file():
        for override in _load_jsonl(args.task_overrides):
            task_id = int(override["task_id"])
            product = override["product"]
            if task_id not in task_id_set:
                raise ValueError(f"task override is outside Final-240: {task_id}")
            if task_id in task_override_ids:
                raise ValueError(f"duplicate task override: {task_id}")
            if not (product.get("instructions") or []):
                raise ValueError(f"task override has no instruction: {task_id}")
            target_products[task_id] = product
            task_override_ids.append(task_id)
    products = _load_jsonl(
        ROOT / "data/shopsimulator_official/fine_items_eval_train_all.jsonl"
    )
    products_by_asin = {
        str(product.get("asin")): product
        for product in products
        if product.get("asin") is not None
    }

    models = {}
    for label in LABELS:
        models[label] = replay_label(
            label=label,
            task_ids=task_ids,
            target_products=target_products,
            products_by_asin=products_by_asin,
        )
        (args.output_dir / f"{label}.json").write_text(
            json.dumps(models[label], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    report = {
        "schema_version": "reward-v4-six-model-replay-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reward_commit": args.reward_commit,
        "task_count": len(task_ids),
        "reference_directory": str(REFERENCE),
        "task_override_path": (
            str(args.task_overrides) if args.task_overrides.is_file() else None
        ),
        "task_override_sha256": (
            _file_hash(args.task_overrides)
            if args.task_overrides.is_file()
            else None
        ),
        "task_override_ids": sorted(task_override_ids),
        "missing_product_count": sum(
            len(model["missing_products"]) for model in models.values()
        ),
        "all_old_means_match_reference": all(
            model["old_mean_matches_reference"] for model in models.values()
        ),
        "models": models,
    }
    report["classification_audit"] = _classification_audit(models)
    report["numeric_policy_audit"] = _numeric_policy_audit(models)
    if not report["numeric_policy_audit"]["passed"]:
        raise ValueError(
            "reward replay violates numeric policy: "
            + json.dumps(
                report["numeric_policy_audit"]["errors"][:10],
                ensure_ascii=False,
            )
        )
    semantics_audit = {}
    if args.semantics_audit is not None:
        semantics_payload = json.loads(
            args.semantics_audit.read_text(encoding="utf-8")
        )
        semantics_audit = {
            "gold_purchase": int(
                (semantics_payload.get("gold_reward_type_counts") or {}).get(
                    "gold_purchase", 0
                )
            ),
            "unresolved_scored_semantics": len(
                semantics_payload.get("unresolved_scored_semantics") or []
            ),
            "uncovered_semantic_marker_clauses": len(
                semantics_payload.get("uncovered_semantic_marker_clauses") or []
            ),
        }
    report["semantics_audit"] = semantics_audit
    overlay = _overlay_payload(models)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown = _markdown(report)
    (args.output_dir / "REPORT.md").write_text(markdown, encoding="utf-8")
    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(markdown, encoding="utf-8")
    if args.overlay_path is not None:
        args.overlay_path.parent.mkdir(parents=True, exist_ok=True)
        args.overlay_path.write_text(
            json.dumps(overlay, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.recalculation_path is not None:
        args.recalculation_path.parent.mkdir(parents=True, exist_ok=True)
        args.recalculation_path.write_text(
            json.dumps(
                _recalculation_payload(report, overlay),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "all_old_means_match_reference": report[
                    "all_old_means_match_reference"
                ],
                "missing_product_count": report["missing_product_count"],
                "models": {
                    label: {
                        "old_mean": model["old_mean_reward"],
                        "new_mean": model["new_mean_reward"],
                        "delta": model["mean_reward_delta"],
                        "changed_tasks": model["changed_task_count"],
                        "type_changed_tasks": model[
                            "type_changed_task_count"
                        ],
                        "numeric_only_changed_tasks": model[
                            "numeric_only_changed_task_count"
                        ],
                        "transitions": model["transition_counts"],
                    }
                    for label, model in models.items()
                },
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
