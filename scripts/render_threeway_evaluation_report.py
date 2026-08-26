"""Render a Chinese audit report and static dashboard for Base/SFT/GRPO.

The input directory must be produced by
``evaluate_existing_trajectories_with_judges.py``. Reward outcomes and Judge
scores remain separate throughout the generated artifacts.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

LABELS = ("base", "sft", "grpo")
DISPLAY = {"base": "Base", "sft": "SFT", "grpo": "GRPO"}
DIMENSIONS = (
    "search_strategy",
    "candidate_utilization",
    "evidence_verification",
    "decision_quality",
    "termination_efficiency",
)
DIMENSION_DISPLAY = {
    "search_strategy": "Search Strategy（搜索策略）",
    "candidate_utilization": "Candidate Utilization（候选利用）",
    "evidence_verification": "Evidence Verification（证据核验）",
    "decision_quality": "Decision Quality（决策质量）",
    "termination_efficiency": "Termination Efficiency（终止效率）",
}
RUBRIC_STATUSES = ("satisfied", "violated", "unknown", "not_applicable")
RUBRIC_STATUS_DISPLAY = {
    "satisfied": "satisfied（满足）",
    "violated": "violated（违反）",
    "unknown": "unknown（证据不足）",
    "not_applicable": "not_applicable（不适用）",
}
STRATUM_DISPLAY = {
    "core": "core（核心题）",
    "challenge": "challenge（挑战题）",
    "candidate_comparison": "candidate_comparison（候选比较）",
    "evidence_verification": "evidence_verification（证据核验）",
    "long_horizon": "long_horizon（长链路）",
    "multi_option": "multi_option（多规格）",
    "price_semantics": "price_semantics（价格语义）",
    "search_reformulation": "search_reformulation（搜索改写）",
}
ERROR_DISPLAY = {
    "budget_violation": "budget_violation（违反预算）",
    "candidate_comparison_insufficient": "candidate_comparison_insufficient（候选比较不足）",
    "candidate_overexploration": "candidate_overexploration（候选探索过度）",
    "critical_evidence_missing": "critical_evidence_missing（缺少关键证据）",
    "final_price_unverified": "final_price_unverified（未核验最终价格）",
    "illegal_action": "illegal_action（非法动作）",
    "max_steps_exhaustion": "max_steps_exhaustion（耗尽最大步数）",
    "overexploration_after_convergence": "overexploration_after_convergence（收敛后过度探索）",
    "premature_abstain": "premature_abstain（过早放弃）",
    "premature_purchase": "premature_purchase（过早购买）",
    "repeat_loop": "repeat_loop（重复循环）",
    "requirement_violation": "requirement_violation（违反用户要求）",
    "search_core_requirement_missed": "search_core_requirement_missed（搜索遗漏核心要求）",
    "search_ineffective_reformulation": "search_ineffective_reformulation（无效搜索改写）",
    "search_mechanical_repeat": "search_mechanical_repeat（机械重复搜索）",
    "unreliable_evidence_used": "unreliable_evidence_used（使用不可靠证据）",
    "wrong_category": "wrong_category（商品品类错误）",
    "wrong_option": "wrong_option（商品规格选错）",
}
REWARD_TYPE_DISPLAY = {
    "gold_purchase": "gold_purchase（目标商品购买）",
    "partial_alternative_purchase": "partial_alternative_purchase（部分满足的替代商品购买）",
    "wrong_purchase": "wrong_purchase（错误商品购买）",
    "assistant_final": "assistant_final（直接文本结束）",
    "max_steps": "max_steps（达到最大步数）",
    "repeat_loop": "repeat_loop（重复循环）",
    "reward_unverifiable": "reward_unverifiable（奖励无法核验）",
    "unknown": "unknown（未知）",
}
REWARD_TYPE_ORDER = (
    "gold_purchase",
    "partial_alternative_purchase",
    "wrong_purchase",
    "assistant_final",
    "max_steps",
    "repeat_loop",
    "reward_unverifiable",
    "unknown",
)
RUBRIC_CATEGORY_DISPLAY = {
    "category": "商品品类",
    "core_function": "核心功能与属性",
    "brand": "品牌",
    "model": "型号",
    "price": "价格与预算",
    "color_option": "颜色选项",
    "size_option": "尺寸、容量与数量规格",
    "flavor_option": "口味与香味",
    "other_option": "其他购买选项",
    "other": "其他要求",
}
RUBRIC_CATEGORY_ORDER = tuple(RUBRIC_CATEGORY_DISPLAY)
RUBRIC_HARDNESS_DISPLAY = {
    "hard": "Hard（硬约束）",
    "soft": "Soft（软偏好）",
    "needs_review": "needs_review（需复核）",
}
RUBRIC_HARDNESS_ORDER = tuple(RUBRIC_HARDNESS_DISPLAY)


def rubric_status_display(status: str) -> str:
    return RUBRIC_STATUS_DISPLAY.get(status, status)


def error_display(error: str) -> str:
    return ERROR_DISPLAY.get(error, error)


def reward_type_display(reward_type: str) -> str:
    return REWARD_TYPE_DISPLAY.get(reward_type, reward_type)


def stratum_display(value: str) -> str:
    return STRATUM_DISPLAY.get(value, value)


def rubric_category(rubric: dict[str, Any]) -> str:
    constraint_type = str(rubric.get("constraint_type") or "")
    if constraint_type in {"category", "core_function", "brand", "model"}:
        return constraint_type
    if constraint_type in {"price_preference", "budget_upper", "price_range"}:
        return "price"
    if constraint_type != "option":
        return "other"

    field_path = str(rubric.get("field_path") or "").casefold()
    if "color" in field_path or "颜色" in field_path:
        return "color_option"
    if "flavor" in field_path or "口味" in field_path or "香味" in field_path:
        return "flavor_option"
    size_markers = (
        "size",
        "capacity",
        "dimensions",
        "specification",
        "net_content",
        "长度",
        "直径",
        "容量",
        "规格",
        "数量",
        "参考身高",
        "适用人数",
    )
    if any(marker in field_path for marker in size_markers):
        return "size_option"
    return "other_option"


def summarize_rubric_categories(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts = {
        category: {status: 0 for status in RUBRIC_STATUSES}
        for category in RUBRIC_CATEGORY_ORDER
    }
    for record in rows:
        requirement_rubric = record.get("requirement_rubric") or {}
        rubrics = {
            str(item.get("rubric_id")): item
            for item in requirement_rubric.get("rubrics", [])
        }
        for assessment in requirement_rubric.get("assessments", []):
            rubric = rubrics.get(str(assessment.get("rubric_id")))
            if rubric is None:
                continue
            category = rubric_category(rubric)
            status = str(assessment.get("status") or "unknown")
            if status not in RUBRIC_STATUSES:
                status = "unknown"
            counts[category][status] += 1
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "evaluation_dir",
        type=Path,
        help="Directory containing runs/, comparison.json and run_manifest.json",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} must be a JSON object")
            rows.append(value)
    return rows


def pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def num(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{float(value):,.{digits}f}"


def quantile(values: list[float], probability: float) -> float | None:
    """Return a linearly interpolated quantile for finite numeric values."""

    ordered = sorted(
        float(value)
        for value in values
        if not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )
    if not ordered:
        return None
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def enrich_deterministic_summaries(
    summaries: dict[str, dict[str, Any]],
    evaluations: dict[str, list[dict[str, Any]]],
) -> None:
    """Add report-only distribution metrics derived from saved per-task rows."""

    for label in LABELS:
        token_values = []
        duration_values = []
        context_usage_values = []
        context_hard_limit_tasks = 0
        for record in evaluations[label]:
            deterministic = record.get("deterministic") or {}
            context = deterministic.get("context") or {}
            timing = deterministic.get("timing") or {}
            validity = deterministic.get("validity") or {}
            total_tokens = context.get("total_tokens")
            if (
                not isinstance(total_tokens, bool)
                and isinstance(total_tokens, (int, float))
                and math.isfinite(float(total_tokens))
            ):
                token_values.append(float(total_tokens))
            duration = timing.get("trajectory_duration_seconds")
            if (
                not isinstance(duration, bool)
                and isinstance(duration, (int, float))
                and math.isfinite(float(duration))
            ):
                duration_values.append(float(duration))
            context_usage_ratio = context.get("max_context_usage_ratio")
            if (
                not isinstance(context_usage_ratio, bool)
                and isinstance(context_usage_ratio, (int, float))
                and math.isfinite(float(context_usage_ratio))
            ):
                context_usage_values.append(float(context_usage_ratio))
            context_hard_limit_tasks += validity.get("context_hard_limit") is True

        summary = summaries[label]["deterministic"]
        summary["provider_total_tokens_p50"] = quantile(token_values, 0.50)
        summary["provider_total_tokens_p95"] = quantile(token_values, 0.95)
        summary["trajectory_duration_seconds_p50"] = quantile(duration_values, 0.50)
        summary["trajectory_duration_seconds_p95"] = quantile(duration_values, 0.95)
        summary["context_usage_ratio_p50"] = quantile(context_usage_values, 0.50)
        summary["context_usage_ratio_p95"] = quantile(context_usage_values, 0.95)
        summary["context_hard_limit_tasks"] = context_hard_limit_tasks


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def index_evaluations(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        task_id = int(row["task_id"])
        if task_id in result:
            raise ValueError(f"duplicate task_id {task_id}")
        result[task_id] = row
    return result


def rubric_counts(record: dict[str, Any] | None) -> dict[str, int]:
    counts = Counter()
    if record is not None:
        for assessment in record["requirement_rubric"].get("assessments", []):
            counts[str(assessment.get("status") or "unknown")] += 1
    return {status: counts[status] for status in RUBRIC_STATUSES}


def hard_violation_count(record: dict[str, Any] | None) -> int | None:
    if record is None:
        return None
    rubric = record["requirement_rubric"]
    hardness = {item["rubric_id"]: item.get("hardness") for item in rubric.get("rubrics", [])}
    return sum(
        assessment.get("status") == "violated"
        and hardness.get(assessment.get("rubric_id")) == "hard"
        for assessment in rubric.get("assessments", [])
    )


def stage_fields(record: dict[str, Any] | None) -> dict[str, Any]:
    if record is None:
        return {
            "completed": False,
            "strict_success": False,
            "purchase_success": False,
            "reward_valid": False,
            "final_reward": 0.0,
            "reward_type": "missing",
            "done": False,
            "judge_status": "not_judged",
            "not_judged_reason": "missing_trajectory",
            "primary_error": "",
            "steps": 0,
            "action_attempts": 0,
            "guard_rejections": 0,
            "duplicate_actions": 0,
            "duplicate_searches": 0,
            "truncated": False,
            "context_usage_ratio": None,
            "infrastructure_invalid": False,
            "hard_violations": None,
            **{f"rubric_{status}": 0 for status in RUBRIC_STATUSES},
            **{dimension: None for dimension in DIMENSIONS},
        }
    reward = record["reward_and_terminal"]["metrics"]
    quality = record["trajectory_quality"]
    deterministic = record["deterministic"]
    actions = deterministic["actions_and_efficiency"]
    legality = deterministic["legality"]
    repetition = deterministic["repetition"]
    context = deterministic["context"]
    validity = deterministic["validity"]
    statuses = rubric_counts(record)
    values = {
        "completed": True,
        "strict_success": bool(reward.get("strict_gold_success")),
        "purchase_success": bool(reward.get("purchase_success")),
        "reward_valid": bool(reward.get("reward_valid")),
        "final_reward": float(reward.get("final_reward") or 0.0),
        "reward_type": str(reward.get("reward_type") or "unknown"),
        "done": bool(reward.get("done")),
        "judge_status": str(quality.get("judge_status") or "not_judged"),
        "not_judged_reason": str(quality.get("not_judged_reason") or ""),
        "primary_error": str(quality.get("errors", {}).get("primary") or ""),
        "steps": int(actions.get("executed_tool_steps") or 0),
        "action_attempts": int(actions.get("action_attempts") or 0),
        "guard_rejections": int(legality.get("guard_rejection_count") or 0),
        "duplicate_actions": int(repetition.get("duplicate_canonical_action_count") or 0),
        "duplicate_searches": int(repetition.get("duplicate_search_query_count") or 0),
        "truncated": bool(context.get("any_observation_truncated")),
        "context_usage_ratio": context.get("max_context_usage_ratio"),
        "infrastructure_invalid": bool(validity.get("infrastructure_invalid")),
        "hard_violations": hard_violation_count(record),
    }
    for status, count in statuses.items():
        values[f"rubric_{status}"] = count
    scores = quality.get("dimension_scores", {})
    for dimension in DIMENSIONS:
        score = scores.get(dimension, {}).get("score") if scores else None
        values[dimension] = int(score) if score is not None else None
    return values


def transition(left: bool, right: bool) -> str:
    return f"{'success' if left else 'failure'}_to_{'success' if right else 'failure'}"


def build_per_task(
    indexes: dict[str, dict[int, dict[str, Any]]],
    summaries: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    task_ids = set()
    for label in LABELS:
        task_ids.update(indexes[label])
        task_ids.update(int(value) for value in summaries[label].get("missing_task_ids", []))
    expected = int(summaries["base"]["expected_tasks"])
    if len(task_ids) != expected:
        raise ValueError(f"expected {expected} task IDs, found {len(task_ids)}")
    rows = []
    for task_id in sorted(task_ids):
        item: dict[str, Any] = {"task_id": task_id}
        stage_values = {}
        for label in LABELS:
            values = stage_fields(indexes[label].get(task_id))
            stage_values[label] = values
            for key, value in values.items():
                item[f"{label}_{key}"] = value
        item["base_to_sft_strict_transition"] = transition(
            stage_values["base"]["strict_success"],
            stage_values["sft"]["strict_success"],
        )
        item["sft_to_grpo_strict_transition"] = transition(
            stage_values["sft"]["strict_success"],
            stage_values["grpo"]["strict_success"],
        )
        item["sft_minus_base_reward"] = round(
            stage_values["sft"]["final_reward"] - stage_values["base"]["final_reward"],
            8,
        )
        item["grpo_minus_sft_reward"] = round(
            stage_values["grpo"]["final_reward"] - stage_values["sft"]["final_reward"],
            8,
        )
        rows.append(item)
    return rows


def write_per_task(output: Path, rows: list[dict[str, Any]]) -> None:
    json_path = output / "per-task-comparison.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = output / "per-task-comparison.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def top_error_names(summaries: dict[str, dict[str, Any]], limit: int = 10) -> list[str]:
    totals = Counter()
    for summary in summaries.values():
        totals.update(summary["trajectory_quality"].get("primary_error_counts", {}))
    return [name for name, _ in totals.most_common(limit)]


def strict_transition_line(pair: dict[str, Any]) -> str:
    counts = pair["reward_and_terminal"]["strict_success_transitions"]
    return (
        f"失败→成功 {counts.get('failure_to_success', 0)}；"
        f"成功→成功 {counts.get('success_to_success', 0)}；"
        f"成功→失败 {counts.get('success_to_failure', 0)}；"
        f"失败→失败 {counts.get('failure_to_failure', 0)}"
    )


def stratified_markdown(summaries: dict[str, dict[str, Any]]) -> str:
    lines = [
        "| 分层 | 子集 | 任务数 | Base | SFT | GRPO | SFT→GRPO |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in ("suite", "domain", "challenge_slice"):
        buckets = summaries["base"].get("stratified", {}).get(group, {})
        for bucket in sorted(buckets):
            values = []
            tasks = buckets[bucket]["expected_tasks"]
            for label in LABELS:
                values.append(
                    summaries[label]["stratified"][group][bucket]["reward_and_terminal"][
                        "gold_purchase_rate"
                    ]
                )
            lines.append(
                f"| {group} | {bucket} | {tasks} | {pct(values[0])} | "
                f"{pct(values[1])} | {pct(values[2])} | "
                f"{(values[2] - values[1]) * 100:+.1f} pp |"
            )
    return "\n".join(lines)


def render_markdown(
    output: Path,
    summaries: dict[str, dict[str, Any]],
    comparison: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    expected = summaries["base"]["expected_tasks"]
    errors = top_error_names(summaries)
    pair_bs = comparison["pairwise"]["base_to_sft"]
    pair_sg = comparison["pairwise"]["sft_to_grpo"]
    strict = {
        label: summaries[label]["reward_and_terminal"]["strict_gold_successes"] for label in LABELS
    }
    delta_sg = (strict["grpo"] - strict["sft"]) / expected
    lines = [
        "# Shopping Agent Benchmark · Final-240",
        "",
        "**Base → SFT → GRPO**",
        "",
        (
            "同一批 240 条未见任务、每题一次确定性 rollout。运行契约为 "
            "ShopSimulator Environment v2.4、Reward v4、Termination v3.1、"
            "observation v2、tool schema v2。终局 Reward 与 DeepSeek V4 Pro "
            "轨迹 Judge 分开报告，不合成为总分。"
        ),
        "",
        (
            "严格成功仅接受完整 `gold_purchase`、`reward_valid=true` 和合法终局。"
            "Rubric 由 DeepSeek V4 Flash 冻结生成；Flash Actor 不在本次对比中。"
        ),
        "",
        "## 核心结果",
        "",
        "| 指标 | Base | SFT | GRPO |",
        "|---|---:|---:|---:|",
    ]
    metric_rows = [
        (
            "严格成功",
            lambda s: (
                f"{s['reward_and_terminal']['strict_gold_successes']} / {expected} "
                f"({pct(s['reward_and_terminal']['gold_purchase_rate'])})"
            ),
        ),
        (
            "购买成功",
            lambda s: (
                f"{s['reward_and_terminal']['purchase_successes']} / {expected} "
                f"({pct(s['reward_and_terminal']['purchase_success_rate'])})"
            ),
        ),
        (
            "平均 Final Reward",
            lambda s: num(s["reward_and_terminal"]["mean_final_reward_fixed_denominator"], 4),
        ),
        (
            "Reward valid",
            lambda s: (
                f"{s['reward_and_terminal']['reward_valid_tasks']} / {expected} "
                f"({pct(s['reward_and_terminal']['reward_valid_rate'])})"
            ),
        ),
        (
            "Judge 覆盖率",
            lambda s: pct(s["trajectory_quality"]["judge_coverage_rate"]),
        ),
    ]
    for name, getter in metric_rows:
        lines.append(
            f"| {name} | {getter(summaries['base'])} | {getter(summaries['sft'])} | "
            f"{getter(summaries['grpo'])} |"
        )
    lines.extend(
        [
            "",
            f"SFT→GRPO 严格成功率变化：**{delta_sg * 100:+.1f} pp**。",
            "",
            "## LLM Judge 五维评分",
            "",
            "仅在有效 Judge 轨迹上计算均值，每维 0–2 分，不计算综合总分。",
            "",
            "| 维度 | Base | SFT | GRPO |",
            "|---|---:|---:|---:|",
        ]
    )
    for dimension in DIMENSIONS:
        lines.append(
            f"| {DIMENSION_DISPLAY[dimension]} | "
            + " | ".join(
                num(
                    summaries[label]["trajectory_quality"]["dimensions"][dimension][
                        "mean_score_among_valid_judges"
                    ],
                    3,
                )
                for label in LABELS
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 用户需求 Rubric",
            "",
            "| 状态 | Base | SFT | GRPO |",
            "|---|---:|---:|---:|",
        ]
    )
    for status in RUBRIC_STATUSES:
        lines.append(
            f"| {status} | "
            + " | ".join(
                num(
                    summaries[label]["requirement_rubric"]["status_counts"].get(status, 0),
                    0,
                )
                for label in LABELS
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 第四面板：确定性行为与资源效率",
            "",
            (
                "这些指标由代码直接计算，不交给 Judge 猜测。Token 与耗时的 p50/p95 "
                "基于已保存且带有对应遥测的逐题轨迹。Observation 投影压缩是正常的可见内容裁剪，"
                "不等同于上下文硬溢出。"
            ),
            "",
            "| 指标 | Base | SFT | GRPO |",
            "|---|---:|---:|---:|",
        ]
    )
    behavior = [
        ("平均执行工具调用数 / 任务", "average_executed_steps_fixed_denominator", 3),
        ("平均动作尝试数 / 任务", "average_action_attempts_fixed_denominator", 3),
        ("Guard 拒绝次数", "total_guard_rejections", 0),
        ("重复动作次数", "total_duplicate_canonical_actions", 0),
        ("重复搜索次数", "total_duplicate_search_queries", 0),
        ("Provider Token p50 / 任务", "provider_total_tokens_p50", 0),
        ("Provider Token p95 / 任务", "provider_total_tokens_p95", 0),
        ("端到端耗时 p50（秒）", "trajectory_duration_seconds_p50", 1),
        ("端到端耗时 p95（秒）", "trajectory_duration_seconds_p95", 1),
        ("Observation 投影压缩（任务数）", "tasks_with_observation_truncation", 0),
        ("上下文硬溢出（任务数）", "context_hard_limit_tasks", 0),
        ("基础设施无效任务", "infrastructure_invalid_tasks", 0),
    ]
    for name, key, digits in behavior:
        lines.append(
            f"| {name} | "
            + " | ".join(
                num(summaries[label]["deterministic"].get(key), digits) for label in LABELS
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Primary Error 对比",
            "",
            "数值顺序为 Base / SFT / GRPO。",
            "",
            "| 错误类型 | Base / SFT / GRPO |",
            "|---|---:|",
        ]
    )
    for error in errors:
        values = [
            summaries[label]["trajectory_quality"]["primary_error_counts"].get(error, 0)
            for label in LABELS
        ]
        lines.append(f"| `{error}` | {' / '.join(str(value) for value in values)} |")
    lines.extend(
        [
            "",
            "## 配对转移",
            "",
            f"- Base→SFT：{strict_transition_line(pair_bs)}。",
            f"- SFT→GRPO：{strict_transition_line(pair_sg)}。",
            "",
            "## 分层严格成功率",
            "",
            stratified_markdown(summaries),
            "",
            "## 审计说明",
            "",
            "- Base Task 419 缺少轨迹，按失败计入 240 题分母，并记为 `not_judged`。",
            "- Base Task 205 触发 Judge 服务商内容过滤，保留为 `not_judged`，未改写轨迹绕过过滤。",
            (
                "- Judge 仅看到 Query、冻结 Rubric、Actor-visible trajectory 和白名单确定性指标；"
                "看不到 Reward、Gold 私有字段或其他模型结果。"
            ),
            "- Flash Actor 因历史任务映射问题不在本次统计或结论中。",
            "- 小型 challenge slice 每组仅 10 题，只用于诊断，不单独作显著性结论。",
            "",
            "## 数据与协议",
            "",
            f"- Benchmark SHA-256：`{manifest['benchmark']['sha256']}`",
            f"- Rubric curator：`{manifest['models']['rubric_curator']}`",
            f"- Trajectory Judge：`{manifest['models']['trajectory_judge']}`",
            "- 详细逐题数据：`per-task-comparison.csv` 与 `per-task-comparison.json`。",
        ]
    )
    (output / "audit-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def stage_table_rows(summaries: dict[str, dict[str, Any]], keys: list[tuple[str, str, int]]) -> str:
    rows = []
    for title, key, digits in keys:
        cells = []
        for label in LABELS:
            value = summaries[label]["deterministic"].get(key)
            cells.append(f"<td>{esc(num(value, digits))}</td>")
        rows.append(f"<tr><th>{esc(title)}</th>{''.join(cells)}</tr>")
    return "".join(rows)


def stage_percent_rows(summaries: dict[str, dict[str, Any]], keys: list[tuple[str, str]]) -> str:
    rows = []
    for title, key in keys:
        cells = []
        for label in LABELS:
            value = summaries[label]["deterministic"].get(key)
            cells.append(f"<td>{esc(pct(value))}</td>")
        rows.append(f"<tr><th>{esc(title)}</th>{''.join(cells)}</tr>")
    return "".join(rows)


def stage_table_group(title: str, rows: str) -> str:
    return f'<tr class="group-row"><th colspan="4">{esc(title)}</th></tr>{rows}'


def rubric_status_stack_html(counts: dict[str, int]) -> str:
    return (
        '<td class="status-stack">'
        f"<span>满足 <strong>{counts.get('satisfied', 0):,}</strong></span>"
        f"<span>违反 <strong>{counts.get('violated', 0):,}</strong></span>"
        f"<span>未知 <strong>{counts.get('unknown', 0):,}</strong></span>"
        f"<span>不适用 <strong>{counts.get('not_applicable', 0):,}</strong></span>"
        "</td>"
    )


def bar(value: float, label: str, stage: str, ceiling: float = 1.0) -> str:
    width = min(100.0, max(0.0, value / ceiling * 100.0))
    return (
        f'<div class="bar-row"><span>{esc(label)}</span><div class="bar-track">'
        f'<div class="bar-fill {stage}" style="width:{width:.3f}%"></div></div>'
        f"<strong>{esc(num(value, 3) if ceiling == 2 else pct(value))}</strong></div>"
    )


def task_rows_html(rows: list[dict[str, Any]]) -> str:
    rendered = []
    for row in rows:
        transition_name = row["sft_to_grpo_strict_transition"]
        values = [
            str(row["task_id"]),
            "✓" if row["base_strict_success"] else "—",
            "✓" if row["sft_strict_success"] else "—",
            "✓" if row["grpo_strict_success"] else "—",
            num(row["base_final_reward"], 3),
            num(row["sft_final_reward"], 3),
            num(row["grpo_final_reward"], 3),
            row["base_judge_status"],
            row["sft_judge_status"],
            row["grpo_judge_status"],
            error_display(row["base_primary_error"]) if row["base_primary_error"] else "—",
            error_display(row["sft_primary_error"]) if row["sft_primary_error"] else "—",
            error_display(row["grpo_primary_error"]) if row["grpo_primary_error"] else "—",
        ]
        cells = "".join(f"<td>{esc(value)}</td>" for value in values)
        raw_search_values = [
            row["base_judge_status"],
            row["sft_judge_status"],
            row["grpo_judge_status"],
            row["base_primary_error"],
            row["sft_primary_error"],
            row["grpo_primary_error"],
        ]
        search = " ".join(str(value) for value in values + raw_search_values).casefold()
        rendered.append(
            f'<tr data-transition="{esc(transition_name)}" data-search="{esc(search)}">{cells}</tr>'
        )
    return "".join(rendered)


def render_dashboard(
    output: Path,
    summaries: dict[str, dict[str, Any]],
    comparison: dict[str, Any],
    manifest: dict[str, Any],
    per_task: list[dict[str, Any]],
    evaluations: dict[str, list[dict[str, Any]]],
) -> None:
    expected = summaries["base"]["expected_tasks"]
    strict_counts = {
        label: summaries[label]["reward_and_terminal"]["strict_gold_successes"] for label in LABELS
    }
    strict_rates = {label: strict_counts[label] / expected for label in LABELS}
    reward_means = {
        label: summaries[label]["reward_and_terminal"]["mean_final_reward_fixed_denominator"]
        for label in LABELS
    }
    delta_pp = (strict_rates["grpo"] - strict_rates["sft"]) * 100
    reward_bars = "".join(
        bar(strict_rates[label], f"{DISPLAY[label]} · Strict（严格成功）", label)
        for label in LABELS
    )
    reward_bars += "".join(
        bar(
            summaries[label]["reward_and_terminal"]["purchase_success_rate"],
            f"{DISPLAY[label]} · Purchase（购买成功）",
            label,
        )
        for label in LABELS
    )
    outcome_rows = []
    outcome_metrics = (
        (
            "严格成功",
            lambda summary: (
                f"{summary['reward_and_terminal']['strict_gold_successes']} / {expected} "
                f"({pct(summary['reward_and_terminal']['gold_purchase_rate'])})"
            ),
        ),
        (
            "完成购买",
            lambda summary: (
                f"{summary['reward_and_terminal']['purchase_successes']} / {expected} "
                f"({pct(summary['reward_and_terminal']['purchase_success_rate'])})"
            ),
        ),
        (
            "Reward 有效",
            lambda summary: (
                f"{summary['reward_and_terminal']['reward_valid_tasks']} / {expected} "
                f"({pct(summary['reward_and_terminal']['reward_valid_rate'])})"
            ),
        ),
        (
            "平均 Final Reward",
            lambda summary: num(
                summary["reward_and_terminal"]["mean_final_reward_fixed_denominator"], 4
            ),
        ),
    )
    for title, getter in outcome_metrics:
        outcome_rows.append(
            f"<tr><th>{esc(title)}</th>"
            + "".join(f"<td>{esc(getter(summaries[label]))}</td>" for label in LABELS)
            + "</tr>"
        )
    observed_reward_types = set().union(
        *(
            summaries[label]["reward_and_terminal"].get("reward_type_counts", {})
            for label in LABELS
        )
    )
    ordered_reward_types = [
        reward_type for reward_type in REWARD_TYPE_ORDER if reward_type in observed_reward_types
    ]
    ordered_reward_types.extend(sorted(observed_reward_types - set(ordered_reward_types)))
    reward_type_rows = []
    for reward_type in ordered_reward_types:
        reward_type_rows.append(
            f"<tr><th><code>{esc(reward_type_display(reward_type))}</code></th>"
            + "".join(
                f"<td>{summaries[label]['reward_and_terminal']['reward_type_counts'].get(reward_type, 0):,}</td>"
                for label in LABELS
            )
            + "</tr>"
        )
    dimension_blocks = []
    for dimension in DIMENSIONS:
        values = {
            label: summaries[label]["trajectory_quality"]["dimensions"][dimension][
                "mean_score_among_valid_judges"
            ]
            for label in LABELS
        }
        dimension_blocks.append(
            '<div class="dimension"><h3>'
            + esc(DIMENSION_DISPLAY[dimension])
            + "</h3>"
            + "".join(
                bar(values[label] or 0.0, DISPLAY[label], label, ceiling=2.0) for label in LABELS
            )
            + "</div>"
        )
    coverage_bars = "".join(
        bar(
            summaries[label]["trajectory_quality"]["judge_coverage_rate"],
            DISPLAY[label],
            label,
        )
        for label in LABELS
    )
    rubric_rows = []
    for status in RUBRIC_STATUSES:
        rubric_rows.append(
            f"<tr><th>{esc(rubric_status_display(status))}</th>"
            + "".join(
                f"<td>{summaries[label]['requirement_rubric']['status_counts'].get(status, 0):,}</td>"
                for label in LABELS
            )
            + "</tr>"
        )
    rubric_category_counts = {
        label: summarize_rubric_categories(evaluations[label]) for label in LABELS
    }
    rubric_hardness_rows = []
    for hardness in RUBRIC_HARDNESS_ORDER:
        if not any(
            sum(
                summaries[label]["requirement_rubric"]
                .get("status_counts_by_hardness", {})
                .get(hardness, {})
                .values()
            )
            for label in LABELS
        ):
            continue
        cells = []
        for label in LABELS:
            counts = (
                summaries[label]["requirement_rubric"]
                .get("status_counts_by_hardness", {})
                .get(hardness, {})
            )
            cells.append(rubric_status_stack_html(counts))
        rubric_hardness_rows.append(
            f"<tr><th>{esc(RUBRIC_HARDNESS_DISPLAY[hardness])}</th>{''.join(cells)}</tr>"
        )
    rubric_category_rows = []
    for category in RUBRIC_CATEGORY_ORDER:
        category_total = max(
            sum(rubric_category_counts[label][category].values()) for label in LABELS
        )
        if category_total == 0:
            continue
        cells = []
        for label in LABELS:
            counts = rubric_category_counts[label][category]
            cells.append(rubric_status_stack_html(counts))
        rubric_category_rows.append(
            f"<tr><th>{esc(RUBRIC_CATEGORY_DISPLAY[category])}</th>{''.join(cells)}</tr>"
        )
    error_rows = []
    for error in top_error_names(summaries, limit=12):
        values = [
            summaries[label]["trajectory_quality"]["primary_error_counts"].get(error, 0)
            for label in LABELS
        ]
        error_rows.append(
            f"<tr><th><code>{esc(error_display(error))}</code></th>"
            + "".join(f"<td>{value}</td>" for value in values)
            + "</tr>"
        )
    behavior_rows = stage_table_group(
        "动作行为",
        stage_table_rows(
            summaries,
            [
                ("平均执行工具调用数 / 任务", "average_executed_steps_fixed_denominator", 3),
                ("平均动作尝试数 / 任务", "average_action_attempts_fixed_denominator", 3),
                ("Guard（动作守卫）拒绝次数", "total_guard_rejections", 0),
                ("重复动作次数", "total_duplicate_canonical_actions", 0),
                ("重复搜索次数", "total_duplicate_search_queries", 0),
            ],
        ),
    )
    behavior_rows += stage_table_group(
        "Token（令牌）、耗时与上下文",
        stage_percent_rows(
            summaries,
            [
                ("上下文使用率 p50（中位数）", "context_usage_ratio_p50"),
                ("上下文使用率 p95（95 分位）", "context_usage_ratio_p95"),
            ],
        )
        + stage_table_rows(
            summaries,
            [
                ("Provider Token（服务商令牌）p50（中位数）/ 任务", "provider_total_tokens_p50", 0),
                ("Provider Token（服务商令牌）p95（95 分位）/ 任务", "provider_total_tokens_p95", 0),
                ("端到端耗时 p50（中位数，秒）", "trajectory_duration_seconds_p50", 1),
                ("端到端耗时 p95（95 分位，秒）", "trajectory_duration_seconds_p95", 1),
                ("Observation（观察）投影压缩（任务数）", "tasks_with_observation_truncation", 0),
                ("上下文硬溢出（任务数）", "context_hard_limit_tasks", 0),
                ("基础设施无效任务", "infrastructure_invalid_tasks", 0),
            ],
        ),
    )
    stratified_rows = []
    for group in ("suite", "domain", "challenge_slice"):
        for bucket, base_bucket in sorted(
            summaries["base"].get("stratified", {}).get(group, {}).items()
        ):
            rates = [
                summaries[label]["stratified"][group][bucket]["reward_and_terminal"][
                    "gold_purchase_rate"
                ]
                for label in LABELS
            ]
            stratified_rows.append(
                f"<tr><td>{esc(stratum_display(group))}</td>"
                f"<th>{esc(stratum_display(bucket))}</th>"
                f"<td>{base_bucket['expected_tasks']}</td>"
                + "".join(f"<td>{pct(rate)}</td>" for rate in rates)
                + f"<td>{(rates[2] - rates[1]) * 100:+.1f} pp（个百分点）</td></tr>"
            )
    pair_bs = comparison["pairwise"]["base_to_sft"]
    pair_sg = comparison["pairwise"]["sft_to_grpo"]
    css = """
:root{color-scheme:light;font-family:Inter,"Microsoft YaHei",system-ui,sans-serif;background:#f4f7fb;color:#13213b}*{box-sizing:border-box}html,body{max-width:100%;overflow-x:hidden}body{margin:0;background:#f4f7fb;color:#13213b}.wrap{max-width:1440px;margin:0 auto;padding:0 28px 64px}.hero{background:linear-gradient(135deg,#0c1936,#2e3185);color:#fff;padding:44px 48px;border-radius:0 0 32px 32px;box-shadow:0 24px 60px rgba(17,28,71,.20)}.eyebrow{letter-spacing:.18em;text-transform:uppercase;color:#aebcff;font-weight:700;font-size:13px}.hero h1{font-size:48px;line-height:1.05;margin:16px 0}.hero p{max-width:940px;color:#dbe3ff;font-size:18px;line-height:1.65}.chips{display:flex;gap:10px;flex-wrap:wrap;margin-top:22px}.chip{border:1px solid rgba(255,255,255,.22);border-radius:999px;padding:7px 12px;background:rgba(255,255,255,.08)}section{margin-top:38px}h2{font-size:27px;margin:0 0 8px}h3{font-size:17px;margin:0 0 15px}.lead{color:#60718e;margin:0 0 18px;line-height:1.6}.grid{display:grid;gap:18px}.grid>*{min-width:0}.kpis{grid-template-columns:repeat(5,minmax(0,1fr))}.two{grid-template-columns:repeat(2,minmax(0,1fr))}.three{grid-template-columns:repeat(3,minmax(0,1fr))}.card{min-width:0;background:#fff;border:1px solid #e1e8f2;border-radius:20px;padding:24px;box-shadow:0 9px 28px rgba(18,37,70,.06)}.section-card{margin-top:18px}.kpi{border-top:4px solid #64748b}.kpi.sft{border-top-color:#2563eb}.kpi.grpo{border-top-color:#7c3aed}.kpi.delta{border-top-color:#10a37f}.label{color:#687a97;font-size:14px}.value{font-size:34px;font-weight:800;margin:9px 0 5px;letter-spacing:-.03em}.context{color:#687a97;font-size:14px}.bar-row{display:grid;grid-template-columns:170px 1fr 72px;gap:12px;align-items:center;margin:13px 0}.bar-track{height:13px;background:#edf1f7;border-radius:99px;overflow:hidden}.bar-fill{height:100%;border-radius:99px;background:#64748b}.bar-fill.sft{background:#2563eb}.bar-fill.grpo{background:#7c3aed}.bar-row strong{text-align:right;font-variant-numeric:tabular-nums}.dimensions{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:28px}.dimension{padding:7px 0}.dimension .bar-row{grid-template-columns:80px 1fr 62px;margin:9px 0}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:12px 10px;border-bottom:1px solid #edf1f6;text-align:right;vertical-align:top}th:first-child,td:first-child{text-align:left}thead th{color:#61718c;font-weight:700}.group-row th{padding:14px 10px 8px;background:#f7f9fc;color:#31435f;text-align:left;font-size:13px;letter-spacing:.04em;border-bottom:1px solid #dfe7f2}.group-row:not(:first-child) th{border-top:10px solid #fff}.status-stack{min-width:150px}.status-stack span{display:flex;justify-content:space-between;gap:14px;line-height:1.75;color:#60718e}.status-stack strong{color:#13213b;font-variant-numeric:tabular-nums}code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.92em}.note{border-left:4px solid #2563eb;background:#eef5ff;padding:16px 18px;border-radius:9px;color:#36506f;line-height:1.6}.transition{font-size:16px;line-height:1.7}.transition strong{font-size:21px}.table-wrap{max-width:100%;overflow-x:auto}.filters{display:flex;gap:12px;flex-wrap:wrap;margin:0 0 16px}.filters input,.filters select{font:inherit;border:1px solid #cfd9e8;border-radius:10px;padding:10px 12px;background:#fff;color:#13213b}.filters input{min-width:260px;flex:1}.task-table{min-width:1180px}.task-table tbody tr[hidden]{display:none}.footer{margin-top:40px;color:#667793;font-size:13px;line-height:1.65}.legend{display:flex;gap:18px;flex-wrap:wrap;color:#63728c;font-size:13px}.dot{display:inline-block;width:10px;height:10px;border-radius:99px;margin-right:6px;background:#64748b}.dot.sft{background:#2563eb}.dot.grpo{background:#7c3aed}@media(max-width:1050px){.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}.two,.three{grid-template-columns:1fr}.hero h1{font-size:40px}}@media(max-width:700px){.wrap{padding:0 14px 40px}.hero{padding:34px 24px}.hero h1{font-size:30px;overflow-wrap:anywhere}.hero p{overflow-wrap:anywhere}.kpis,.dimensions{grid-template-columns:1fr}.bar-row{grid-template-columns:105px 1fr 58px;gap:8px}.card{padding:19px}.value{font-size:30px}}
"""
    js = """
const search=document.getElementById('task-search');
const filter=document.getElementById('transition-filter');
const count=document.getElementById('visible-count');
const rows=[...document.querySelectorAll('#task-body tr')];
function apply(){const q=search.value.trim().toLocaleLowerCase();const f=filter.value;let visible=0;for(const row of rows){const okSearch=!q||row.dataset.search.includes(q);const okFilter=!f||row.dataset.transition===f;row.hidden=!(okSearch&&okFilter);if(!row.hidden)visible++;}count.textContent=visible+' / '+rows.length+' tasks（任务）';}
search.addEventListener('input',apply);filter.addEventListener('change',apply);apply();
"""
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Shopping Agent Benchmark（购物智能体评测）· Final-240（最终 240 题）</title><style>{css}</style></head>
<body><main class="wrap"><header class="hero"><div class="eyebrow">Shopping Agent Benchmark（购物智能体评测）· Final-240（最终 240 题）</div><h1>Base → SFT → GRPO</h1><p>同一批 240 条未见任务、每题一次确定性 rollout（执行轨迹）。页面同时展示 Reward（奖励）v3.2 终局结果、冻结 Rubric（评分细则）和 DeepSeek V4 Pro 轨迹评测；Reward（奖励）与 Judge（评测模型）不合成为一个总分。</p><div class="chips"><span class="chip">Environment（环境）v2.4</span><span class="chip">Reward（奖励）v3.2</span><span class="chip">Termination（终止机制）v3.1</span><span class="chip">Observation（观察）v2</span><span class="chip">Tool schema（工具规范）v2</span><span class="chip">V4 Flash Rubric（评分细则）</span><span class="chip">V4 Pro Judge（评测模型）</span></div></header>
<section><h2>核心结果</h2><p class="lead">严格成功要求完整 gold_purchase（目标商品购买）、reward_valid=true（奖励有效）和合法终局。缺失、错误和 not_judged 任务仍保留在 240 题分母。</p><div class="grid kpis">
<article class="card kpi"><div class="label">Base 严格成功率</div><div class="value">{pct(strict_rates["base"])}</div><div class="context">{strict_counts["base"]} / {expected} Gold purchase（目标商品购买）</div></article>
<article class="card kpi sft"><div class="label">SFT 严格成功率</div><div class="value">{pct(strict_rates["sft"])}</div><div class="context">{strict_counts["sft"]} / {expected} Gold purchase（目标商品购买）</div></article>
<article class="card kpi grpo"><div class="label">GRPO 严格成功率</div><div class="value">{pct(strict_rates["grpo"])}</div><div class="context">{strict_counts["grpo"]} / {expected} Gold purchase（目标商品购买）</div></article>
<article class="card kpi delta"><div class="label">SFT → GRPO</div><div class="value">{delta_pp:+.1f} pp（个百分点）</div><div class="context">严格成功率增量</div></article>
<article class="card kpi grpo"><div class="label">GRPO 平均 Reward（奖励）</div><div class="value">{num(reward_means["grpo"], 4)}</div><div class="context">SFT: {num(reward_means["sft"], 4)} · Base: {num(reward_means["base"], 4)}</div></article>
</div></section>
<section><h2>1. 任务结果：最终有没有完成</h2><p class="lead">先回答最重要的问题：模型是否完成购买、是否严格命中目标商品，以及最终以哪一种 Reward 类型结束。</p><div class="grid two"><article class="card"><h3>成功率与购买结果</h3><p class="lead">Strict（严格成功）只接受 gold_purchase（目标商品购买）；Purchase（购买成功）按 Reward（奖励）v3.2 的 purchase_success（购买成功）口径。</p>{reward_bars}</article><article class="card"><h3>终局结果汇总</h3><p class="lead">同时报告成功数量、购买数量、Reward 有效数量和固定 240 题分母下的平均 Reward。</p><div class="table-wrap"><table><thead><tr><th>指标</th><th>{DISPLAY['base']}</th><th>{DISPLAY['sft']}</th><th>{DISPLAY['grpo']}</th></tr></thead><tbody>{"".join(outcome_rows)}</tbody></table></div></article></div><article class="card section-card"><h3>Reward 类型分布</h3><p class="lead">每道任务只计入一种终局 Reward 类型，用于区分目标购买、替代购买、错误购买、循环和步数耗尽等结果。</p><div class="table-wrap"><table><thead><tr><th>Reward 类型</th><th>{DISPLAY['base']}</th><th>{DISPLAY['sft']}</th><th>{DISPLAY['grpo']}</th></tr></thead><tbody>{"".join(reward_type_rows)}</tbody></table></div></article></section>
<section><h2>2. 用户需求：具体满足了哪些要求</h2><p class="lead">结果失败并不说明所有要求都没完成；这里把用户要求逐条拆开，定位满足、违反和证据不足的部分。</p><article class="card"><h3>Rubric（评分细则）总体状态</h3><p class="lead">逐条要求分别标记 satisfied（满足）、violated（违反）、unknown（证据不足）或 not_applicable（不适用）。</p><table><thead><tr><th>状态</th><th>{DISPLAY['base']}</th><th>{DISPLAY['sft']}</th><th>{DISPLAY['grpo']}</th></tr></thead><tbody>{"".join(rubric_rows)}</tbody></table></article><article class="card section-card"><h3>Hard / Soft 约束拆分</h3><p class="lead">Hard 是品类、明确品牌、型号和严格预算等必须满足的要求；Soft 是“最好”“左右”“倾向于”等偏好。少量无法可靠判定强度的要求单列为 needs_review。</p><div class="table-wrap"><table><thead><tr><th>约束强度</th><th>{DISPLAY['base']}</th><th>{DISPLAY['sft']}</th><th>{DISPLAY['grpo']}</th></tr></thead><tbody>{"".join(rubric_hardness_rows)}</tbody></table></div></article><article class="card section-card"><h3>按用户要求类型拆分</h3><p class="lead">分类直接来自冻结 Rubric 的 constraint_type 和 field_path，不由 Dashboard 临时猜测。每个模型分别显示满足、违反、未知和不适用数量。</p><div class="table-wrap"><table><thead><tr><th>要求类型</th><th>{DISPLAY['base']}</th><th>{DISPLAY['sft']}</th><th>{DISPLAY['grpo']}</th></tr></thead><tbody>{"".join(rubric_category_rows)}</tbody></table></div></article></section>
<section><h2>3. 轨迹过程：为什么成功或失败</h2><p class="lead">在终局结果和需求满足情况之后，再检查搜索、候选利用、证据核验、决策与终止过程。</p><div class="grid two"><article class="card"><h3>LLM Judge（大语言模型评测器）五维评分</h3><p class="lead">有效轨迹由 DeepSeek V4 Pro 分别打 0–2 分，不计算综合总分。</p><div class="dimensions">{"".join(dimension_blocks)}</div><div class="legend"><span><i class="dot"></i>{DISPLAY['base']}</span><span><i class="dot sft"></i>{DISPLAY['sft']}</span><span><i class="dot grpo"></i>{DISPLAY['grpo']}</span></div></article><article class="card"><h3>Primary Error（主要错误）</h3><p class="lead">Pro Judge（专业评测模型）归因的主要错误类型，用于区分搜索、核验、购买和终止等不同失败原因。</p><div class="table-wrap"><table><thead><tr><th>错误类型</th><th>{DISPLAY['base']}</th><th>{DISPLAY['sft']}</th><th>{DISPLAY['grpo']}</th></tr></thead><tbody>{"".join(error_rows)}</tbody></table></div></article></div></section>
<section><h2>4. 行为效率：完成任务付出了多少代价</h2><p class="lead">结果变好之后，还要检查是否以更多循环、工具调用、Token 或耗时换来的，避免只看成功率。</p><article class="card"><h3>确定性行为与资源效率</h3><p class="lead">这些指标由代码直接统计。Observation（观察）投影压缩是正常裁剪，不等同于上下文硬溢出。</p><div class="table-wrap"><table><thead><tr><th>指标</th><th>{DISPLAY['base']}</th><th>{DISPLAY['sft']}</th><th>{DISPLAY['grpo']}</th></tr></thead><tbody>{behavior_rows}</tbody></table></div></article></section>
<section><h2>5. 模型对比：提升发生在哪里</h2><p class="lead">前三部分解释单个模型表现，这里进一步比较 Base、SFT、GRPO 之间的增益、退化和不同任务切片。</p><div class="grid two"><article class="card transition"><strong>{DISPLAY['base']} → {DISPLAY['sft']}</strong><br>{esc(strict_transition_line(pair_bs))}</article><article class="card transition"><strong>{DISPLAY['sft']} → {DISPLAY['grpo']}</strong><br>{esc(strict_transition_line(pair_sg))}</article></div><article class="card section-card"><h3>分层严格成功率</h3><p class="lead">suite、domain 与 challenge_slice 的描述性结果；每个 challenge_slice 只有 10 题，仅用于定位差异。</p><div class="table-wrap"><table><thead><tr><th>分层</th><th>子集</th><th>任务数</th><th>{DISPLAY['base']}</th><th>{DISPLAY['sft']}</th><th>{DISPLAY['grpo']}</th><th>SFT→GRPO</th></tr></thead><tbody>{"".join(stratified_rows)}</tbody></table></div></article></section>
<section><h2>6. 评测审计：这些结论是否有完整证据</h2><p class="lead">最后检查 Judge 覆盖率并下钻到每一道题，方便复核缺失轨迹、内容过滤和具体错误归因。</p><article class="card"><h3>轨迹 Judge（评测模型）覆盖率</h3><p class="lead">Base Task（任务）419 缺失，Task（任务）205 被服务商内容过滤，均保留在固定分母。</p>{coverage_bars}<div class="note">Judge（评测模型）仅看到 Query（用户需求）、冻结 Rubric（评分细则）、Actor-visible trajectory（模型可见轨迹）和白名单行为指标；看不到 Reward（奖励）、Gold（目标商品）私有字段或其他模型结果。</div></article><article class="card section-card"><h3>逐题审计</h3><p class="lead">可按 task_id（任务编号）、Reward（奖励）类型、Judge（评测）状态或错误名称搜索；默认显示全部 240 题。完整字段见 CSV（表格）/JSON（数据）。</p><div class="filters"><input id="task-search" type="search" placeholder="搜索 task_id（任务编号）、状态或错误"><select id="transition-filter"><option value="">全部 SFT→GRPO 转移</option><option value="failure_to_success">失败→成功</option><option value="success_to_success">成功→成功</option><option value="success_to_failure">成功→失败</option><option value="failure_to_failure">失败→失败</option></select><span id="visible-count" class="chip"></span><a href="per-task-comparison.csv">下载 CSV（表格）</a><a href="per-task-comparison.json">查看 JSON（数据）</a></div><div class="table-wrap"><table class="task-table"><thead><tr><th>Task（任务）</th><th>Base strict（严格成功）</th><th>SFT strict（严格成功）</th><th>GRPO strict（严格成功）</th><th>Base reward（奖励）</th><th>SFT reward（奖励）</th><th>GRPO reward（奖励）</th><th>Base judge（评测状态）</th><th>SFT judge（评测状态）</th><th>GRPO judge（评测状态）</th><th>Base primary（主要错误）</th><th>SFT primary（主要错误）</th><th>GRPO primary（主要错误）</th></tr></thead><tbody id="task-body">{task_rows_html(per_task)}</tbody></table></div></article></section>
<footer class="footer"><strong>评估协议：</strong>Final-240（最终 240 题）held-out tasks（留出测试题）；Reward（奖励）与 Judge（评测模型）分开报告；Flash Actor（执行模型）不纳入本次对比。<br><strong>Benchmark（评测基准）SHA-256：</strong><code>{esc(manifest["benchmark"]["sha256"])}</code><br><strong>Rubric（评分细则）/ Judge（评测模型）：</strong><code>{esc(manifest["models"]["rubric_curator"])}</code> / <code>{esc(manifest["models"]["trajectory_judge"])}</code></footer></main><script>{js}</script></body></html>"""
    (output / "dashboard.html").write_text(document, encoding="utf-8")


def main() -> None:
    output = parse_args().evaluation_dir.resolve()
    summaries = {label: load_json(output / "runs" / label / "summary.json") for label in LABELS}
    evaluations = {
        label: load_jsonl(output / "runs" / label / "evaluations.jsonl") for label in LABELS
    }
    enrich_deterministic_summaries(summaries, evaluations)
    indexes = {label: index_evaluations(evaluations[label]) for label in LABELS}
    comparison = load_json(output / "comparison.json")
    manifest = load_json(output / "run_manifest.json")
    per_task = build_per_task(indexes, summaries)
    write_per_task(output, per_task)
    render_markdown(output, summaries, comparison, manifest)
    render_dashboard(output, summaries, comparison, manifest, per_task, evaluations)
    print(
        json.dumps(
            {
                "dashboard": str(output / "dashboard.html"),
                "report": str(output / "audit-report.md"),
                "per_task_csv": str(output / "per-task-comparison.csv"),
                "per_task_json": str(output / "per-task-comparison.json"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
