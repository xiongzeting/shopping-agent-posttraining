from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "重点/4.评测阶段"
STEP230_SOURCE = ROOT / "outputs/evaluation/final240-grpo230-harness-improved-deepseek-v4-pro-20260822"
QWEN_SOURCE = (
    ROOT
    / "outputs/evaluation/final240-v24-qwen38-27b-base-nonthinking-deepseek-v4-pro-judge-r1-20260824"
)
GRPO230_V3_SOURCE = (
    REFERENCE / "GRPO230-Harness-v3-r4-DeepSeek-Pro-Judge-20260826"
)
QWEN_V3_SOURCE = (
    REFERENCE / "Qwen38-27B-Harness-v3-r1-DeepSeek-Pro-Judge-20260826"
)
GRPO230_V3_TRAJECTORIES = (
    REFERENCE
    / "GRPO230-Harness-v3-Final240-4rounds-20260826"
    / "final240-v24-2b-grpo-step230-candidate-context-reset-20260826-r4"
    / "trajectories.jsonl"
)
QWEN_V3_TRAJECTORIES = (
    REFERENCE
    / "Qwen38-27B-Harness-v3-Final240-4rounds-20260826"
    / "final240-v24-qwen38-27b-harness-v3-context-reset-20260826-r1"
    / "trajectories.jsonl"
)
GRPO230_V3_CONTEXT_RECALCULATION = (
    REFERENCE / "grpo230-v3-context-usage-recalculation.json"
)
QWEN_V3_CONTEXT_RECALCULATION = (
    REFERENCE / "qwen38-27b-v3-context-usage-recalculation.json"
)
QWEN_TRAJECTORIES = (
    ROOT
    / "outputs/evaluation/final240-v24-qwen38-27b-base-step230-config-nonthinking-r1-20260823/trajectories.jsonl"
)
QWEN_CONTEXT_RECALCULATION = REFERENCE / "qwen38-27b-context-usage-recalculation.json"
STEP230_TRAJECTORIES = (
    REFERENCE / "GRPO-step230-Final240-160gold/trajectories.jsonl"
)
STEP230_CONTEXT_RECALCULATION = (
    REFERENCE
    / "GRPO-step230-Final240-160gold/context-usage-recalculation.json"
)
OUTPUT = REFERENCE
LABELS = (
    "base",
    "sft",
    "grpo100",
    "grpo230",
    "qwen38_27b",
    "grpo230_v3",
    "qwen38_27b_v3",
)
TASK_LABELS = ("base", "sft", "grpo230")
TRANSITIONS = (("sft", "grpo230"),)
DISPLAY = {
    "base": "Base v1",
    "sft": "SFT v1",
    "grpo50": "GRPO50 v1",
    "grpo100": "GRPO100 v1",
    "grpo230": "GRPO230 v2",
    "qwen38_27b": "Qwen3.8-27B v2",
    "grpo230_v3": "GRPO230 v3",
    "qwen38_27b_v3": "Qwen3.8-27B v3",
}
COLORS = {
    "base": "#64748b",
    "sft": "#2563eb",
    "grpo50": "#7c3aed",
    "grpo100": "#e0569b",
    "grpo230": "#0f9f8f",
    "qwen38_27b": "#f59e0b",
    "grpo230_v3": "#dc2626",
    "qwen38_27b_v3": "#0891b2",
}
TOOL_ORDER = (
    "search_products",
    "open_product",
    "select_option",
    "back_to_search",
    "prev_page",
    "next_page",
    "buy_now",
    "finish_without_purchase",
    "view_description",
    "view_features",
    "view_reviews",
    "view_attributes",
)
LEGACY_INFORMATION_TOOLS = frozenset(
    {"view_description", "view_features", "view_reviews", "view_attributes"}
)
CURRENT_EIGHT_TOOL_MODELS = frozenset(
    {"grpo230", "qwen38_27b", "grpo230_v3", "qwen38_27b_v3"}
)
DIMENSIONS = (
    "search_strategy",
    "candidate_utilization",
    "evidence_verification",
    "decision_quality",
    "termination_efficiency",
)
DIMENSION_DISPLAY = {
    "search_strategy": "搜索策略",
    "candidate_utilization": "候选利用",
    "evidence_verification": "证据核验",
    "decision_quality": "决策质量",
    "termination_efficiency": "终止效率",
}
RUBRIC_STATUSES = ("satisfied", "violated", "unknown", "not_applicable")
REWARD_ORDER = (
    "gold_purchase",
    "valid_alternative_purchase",
    "partial_alternative_purchase",
    "wrong_purchase",
    "assistant_final",
    "guard_rejection",
    "max_steps",
    "repeat_loop",
    "early_abstain",
    "reward_unverifiable",
    "unknown",
)
REWARD_ALWAYS_SHOW = frozenset(
    {"early_abstain", "reward_unverifiable"}
)
REWARD_DISPLAY = {
    "early_abstain": "early_abstain（过早停止）",
}

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import render_threeway_evaluation_report as common  # noqa: E402
from shopping_grpo.evaluation.comparison import compare_evaluation_runs  # noqa: E402

REWARD_OVERLAY: dict[str, dict[int, dict[str, Any]]] = {}


def build_tool_usage(
    evaluations: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for label in LABELS:
        counts: Counter[str] = Counter()
        for row in evaluations[label]:
            tool_counts = (
                row.get("deterministic", {})
                .get("actions_and_efficiency", {})
                .get("tool_counts", {})
                or {}
            )
            counts.update(
                {
                    str(tool): int(value or 0)
                    for tool, value in tool_counts.items()
                }
            )

        uses_current_schema = label in CURRENT_EIGHT_TOOL_MODELS
        displayed_counts = {
            tool: (
                None
                if uses_current_schema and tool in LEGACY_INFORMATION_TOOLS
                else counts.get(tool, 0)
            )
            for tool in TOOL_ORDER
        }
        models[label] = {
            "schema_tool_count": 8 if uses_current_schema else 12,
            "counts": displayed_counts,
            "displayed_total": sum(
                int(value) for value in displayed_counts.values() if value is not None
            ),
            "nonstandard_tool_counts": {
                tool: count
                for tool, count in sorted(counts.items())
                if tool not in TOOL_ORDER and count
            },
        }
    return {
        "source": "deterministic.actions_and_efficiency.tool_counts",
        "tool_order": list(TOOL_ORDER),
        "legacy_information_tools": sorted(LEGACY_INFORMATION_TOOLS),
        "models": models,
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_totals(path: Path) -> dict[str, float | int]:
    totals: dict[str, float | int] = {
        "results": 0,
        "provider_calls": 0,
        "latency_seconds": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "validation_repairs": 0,
    }
    if not path.is_file():
        return totals
    for row in load_jsonl(path):
        totals["results"] += 1
        totals["validation_repairs"] += int(row.get("validation_repairs", 0))
        for call in row.get("calls") or []:
            totals["provider_calls"] += 1
            totals["latency_seconds"] += float(call.get("latency_seconds", 0.0))
            usage = call.get("usage") or {}
            totals["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
            totals["completion_tokens"] += int(usage.get("completion_tokens", 0))
            totals["total_tokens"] += int(usage.get("total_tokens", 0))
    return totals


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def pct(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{float(value) * 100:.{digits}f}%"


def num(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{float(value):,.{digits}f}"


def transition(left: bool, right: bool) -> str:
    return f"{'success' if left else 'failure'}_to_{'success' if right else 'failure'}"


def step_penalty(step_count: int) -> float:
    schedule = (
        (16, 20, 0.01),
        (21, 25, 0.02),
        (26, 30, 0.03),
        (31, 35, 0.04),
        (36, 40, 0.05),
        (41, 45, 0.06),
    )
    steps = max(0, int(step_count))
    return -sum(
        max(0, min(steps, end) - start + 1) * rate
        for start, end, rate in schedule
    )


def prepare_reward_overlays(evaluations: dict[str, list[dict[str, Any]]]) -> None:
    overlay = load_json(REFERENCE / "reward-v4-per-task.json")
    for label in LABELS:
        if label in overlay["records"]:
            REWARD_OVERLAY[label] = {
                int(row["task_id"]): dict(row)
                for row in overlay["records"][label]
            }
            continue
        records: dict[int, dict[str, Any]] = {}
        for row in evaluations[label]:
            metrics = row["reward_and_terminal"]["metrics"]
            task_id = int(row["task_id"])
            records[task_id] = {
                "task_id": task_id,
                "strict_gold": bool(metrics["strict_gold_success"]),
                "purchase_success": bool(metrics["purchase_success"]),
                "reward_valid": bool(metrics["reward_valid"]),
                "final_reward": float(metrics["final_reward"]),
                "reward_type": str(metrics["reward_type"]),
                "weighted_score": float(metrics.get("weighted_score") or 0.0),
            }
        REWARD_OVERLAY[label] = records


def stage_fields(
    record: dict[str, Any] | None,
    *,
    label: str,
    task_id: int,
) -> dict[str, Any]:
    values = common.stage_fields(record)
    override = REWARD_OVERLAY.get(label, {}).get(int(task_id))
    if override:
        values.update(
            {
                "strict_success": bool(override["strict_gold"]),
                "purchase_success": bool(override["purchase_success"]),
                "reward_valid": bool(override["reward_valid"]),
                "final_reward": float(override["final_reward"]),
                "reward_type": str(override["reward_type"]),
            }
        )
    return values


def build_per_task(
    indexes: dict[str, dict[int, dict[str, Any]]], expected_ids: list[int]
) -> list[dict[str, Any]]:
    rows = []
    for task_id in expected_ids:
        item: dict[str, Any] = {"task_id": task_id}
        stages = {}
        for label in TASK_LABELS:
            values = stage_fields(indexes[label].get(task_id), label=label, task_id=task_id)
            stages[label] = values
            for key, value in values.items():
                item[f"{label}_{key}"] = value
        for source, target in TRANSITIONS:
            item[f"{source}_to_{target}_strict_transition"] = transition(
                stages[source]["strict_success"], stages[target]["strict_success"]
            )
            item[f"{source}_to_{target}_purchase_transition"] = transition(
                stages[source]["purchase_success"],
                stages[target]["purchase_success"],
            )
            item[f"{target}_minus_{source}_reward"] = round(
                stages[target]["final_reward"] - stages[source]["final_reward"], 8
            )
        rows.append(item)
    return rows


def enrich(summaries: dict[str, dict[str, Any]], evaluations: dict[str, list[dict[str, Any]]]) -> None:
    common.LABELS = LABELS
    common.enrich_deterministic_summaries(summaries, evaluations)


def apply_step230_provider_context_usage(
    evaluations: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Replace inflated local estimates with actual per-call provider prompt usage."""

    if STEP230_TRAJECTORIES.is_file():
        trajectories = {
            int(row["task_id"]): row for row in load_jsonl(STEP230_TRAJECTORIES)
        }
        records = []
        old_ratios = []
        corrected_ratios = []
        for evaluation in evaluations["grpo230"]:
            task_id = int(evaluation["task_id"])
            trajectory = trajectories[task_id]
            context = evaluation["deterministic"]["context"]
            budget = int(
                (trajectory.get("context_budget") or {}).get("max_input_tokens")
                or context.get("max_input_token_budget")
                or 0
            )
            prompt_tokens = [
                int((call.get("usage") or {}).get("prompt_tokens") or 0)
                for call in trajectory.get("model_calls") or []
            ]
            if budget <= 0 or not prompt_tokens:
                raise ValueError(
                    f"task {task_id} lacks provider prompt usage or input budget"
                )
            provider_max = max(prompt_tokens)
            old_ratio = float(context.get("max_context_usage_ratio") or 0.0)
            corrected_ratio = provider_max / budget
            old_ratios.append(old_ratio)
            corrected_ratios.append(corrected_ratio)
            records.append(
                {
                    "task_id": task_id,
                    "provider_call_count": len(prompt_tokens),
                    "max_provider_prompt_tokens": provider_max,
                    "max_input_token_budget": budget,
                    "corrected_context_usage_ratio": corrected_ratio,
                    "previous_local_estimate_ratio": old_ratio,
                }
            )

        payload = {
            "schema_version": "shopping-context-usage-recalculation-v1",
            "source": str(STEP230_TRAJECTORIES.relative_to(REFERENCE)),
            "method": (
                "For each task, take the maximum actual provider prompt_tokens "
                "across model calls and divide by that task's max input-token budget; "
                "then compute linearly interpolated quantiles across 240 tasks."
            ),
            "reason": (
                "Saved context_turn_tokens are local pre-request estimates and contain "
                "inflated values that do not match provider calls; they are unsuitable "
                "for reporting real context-window consumption."
            ),
            "tasks": len(records),
            "previous": {
                "p50": common.quantile(old_ratios, 0.50),
                "p95": common.quantile(old_ratios, 0.95),
                "max": max(old_ratios),
            },
            "corrected": {
                "p50": common.quantile(corrected_ratios, 0.50),
                "p95": common.quantile(corrected_ratios, 0.95),
                "max": max(corrected_ratios),
                "tasks_over_budget": sum(
                    ratio > 1.0 for ratio in corrected_ratios
                ),
            },
            "records": sorted(records, key=lambda row: row["task_id"]),
        }
        STEP230_CONTEXT_RECALCULATION.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        payload = load_json(STEP230_CONTEXT_RECALCULATION)

    corrected_by_task = {
        int(row["task_id"]): row for row in payload["records"]
    }
    for evaluation in evaluations["grpo230"]:
        task_id = int(evaluation["task_id"])
        corrected = corrected_by_task[task_id]
        context = evaluation["deterministic"]["context"]
        context["max_input_tokens"] = int(
            corrected["max_provider_prompt_tokens"]
        )
        context["max_context_usage_ratio"] = float(
            corrected["corrected_context_usage_ratio"]
        )
        context["context_usage_source"] = (
            "max(model_calls[].usage.prompt_tokens) / max_input_token_budget"
        )
    return payload


def apply_qwen_provider_context_usage(
    evaluations: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Use actual provider prompt usage for the deployed Qwen run."""

    trajectories = {
        int(row["task_id"]): row for row in load_jsonl(QWEN_TRAJECTORIES)
    }
    records = []
    corrected_ratios = []
    for evaluation in evaluations["qwen38_27b"]:
        task_id = int(evaluation["task_id"])
        trajectory = trajectories[task_id]
        context = evaluation["deterministic"]["context"]
        budget = int(
            (trajectory.get("context_budget") or {}).get("max_input_tokens")
            or context.get("max_input_token_budget")
            or 0
        )
        prompt_tokens = [
            int((call.get("usage") or {}).get("prompt_tokens") or 0)
            for call in trajectory.get("model_calls") or []
        ]
        if budget <= 0 or not prompt_tokens:
            raise ValueError(
                f"Qwen task {task_id} lacks provider prompt usage or input budget"
            )
        provider_max = max(prompt_tokens)
        corrected_ratio = provider_max / budget
        corrected_ratios.append(corrected_ratio)
        records.append(
            {
                "task_id": task_id,
                "provider_call_count": len(prompt_tokens),
                "max_provider_prompt_tokens": provider_max,
                "max_input_token_budget": budget,
                "corrected_context_usage_ratio": corrected_ratio,
                "previous_local_estimate_ratio": float(
                    context.get("max_context_usage_ratio") or 0.0
                ),
            }
        )
        context["max_input_tokens"] = provider_max
        context["max_context_usage_ratio"] = corrected_ratio
        context["context_usage_source"] = (
            "max(model_calls[].usage.prompt_tokens) / max_input_token_budget"
        )

    payload = {
        "schema_version": "shopping-context-usage-recalculation-v1",
        "source": str(QWEN_TRAJECTORIES.relative_to(ROOT)),
        "method": (
            "For each task, take the maximum actual provider prompt_tokens "
            "across model calls and divide by that task's max input-token budget; "
            "then compute linearly interpolated quantiles across 240 tasks."
        ),
        "tasks": len(records),
        "corrected": {
            "p50": common.quantile(corrected_ratios, 0.50),
            "p95": common.quantile(corrected_ratios, 0.95),
            "max": max(corrected_ratios),
            "tasks_over_budget": sum(ratio > 1.0 for ratio in corrected_ratios),
        },
        "records": sorted(records, key=lambda row: row["task_id"]),
    }
    QWEN_CONTEXT_RECALCULATION.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def apply_provider_context_usage_for_run(
    evaluations: dict[str, list[dict[str, Any]]],
    *,
    label: str,
    trajectories_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Correct one run using the maximum real provider prompt per task."""

    trajectories = {
        int(row["task_id"]): row for row in load_jsonl(trajectories_path)
    }
    records = []
    previous_ratios = []
    corrected_ratios = []
    for evaluation in evaluations[label]:
        task_id = int(evaluation["task_id"])
        trajectory = trajectories[task_id]
        context = evaluation["deterministic"]["context"]
        budget = int(
            (trajectory.get("context_budget") or {}).get("max_input_tokens")
            or context.get("max_input_token_budget")
            or 0
        )
        prompt_tokens = [
            int((call.get("usage") or {}).get("prompt_tokens") or 0)
            for call in trajectory.get("model_calls") or []
        ]
        if budget <= 0 or not prompt_tokens:
            raise ValueError(
                f"{label} task {task_id} lacks provider prompt usage or input budget"
            )
        provider_max = max(prompt_tokens)
        previous_ratio = float(context.get("max_context_usage_ratio") or 0.0)
        corrected_ratio = provider_max / budget
        previous_ratios.append(previous_ratio)
        corrected_ratios.append(corrected_ratio)
        records.append(
            {
                "task_id": task_id,
                "provider_call_count": len(prompt_tokens),
                "max_provider_prompt_tokens": provider_max,
                "max_input_token_budget": budget,
                "corrected_context_usage_ratio": corrected_ratio,
                "previous_local_estimate_ratio": previous_ratio,
            }
        )
        context["max_input_tokens"] = provider_max
        context["max_context_usage_ratio"] = corrected_ratio
        context["context_usage_source"] = (
            "max(model_calls[].usage.prompt_tokens) / max_input_token_budget"
        )

    payload = {
        "schema_version": "shopping-context-usage-recalculation-v1",
        "source": str(trajectories_path.relative_to(REFERENCE)),
        "method": (
            "For each task, take the maximum actual provider prompt_tokens "
            "across model calls and divide by that task's max input-token budget; "
            "then compute linearly interpolated quantiles across 240 tasks."
        ),
        "reason": (
            "context_turn_tokens are local pre-request audit estimates and can "
            "exceed the actual provider prompt; real context use must be based "
            "on model_calls[].usage.prompt_tokens."
        ),
        "tasks": len(records),
        "previous": {
            "p50": common.quantile(previous_ratios, 0.50),
            "p95": common.quantile(previous_ratios, 0.95),
            "max": max(previous_ratios),
        },
        "corrected": {
            "p50": common.quantile(corrected_ratios, 0.50),
            "p95": common.quantile(corrected_ratios, 0.95),
            "max": max(corrected_ratios),
            "tasks_over_budget": sum(ratio > 1.0 for ratio in corrected_ratios),
        },
        "records": sorted(records, key=lambda row: row["task_id"]),
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def apply_reward_summary_overlays(summaries: dict[str, dict[str, Any]]) -> None:
    for label in LABELS:
        records = list(REWARD_OVERLAY[label].values())
        counts = Counter(str(row["reward_type"]) for row in records)
        strict = sum(bool(row["strict_gold"]) for row in records)
        purchase = sum(bool(row["purchase_success"]) for row in records)
        valid = sum(bool(row["reward_valid"]) for row in records)
        total_reward = sum(float(row["final_reward"]) for row in records)
        total_weighted = sum(float(row.get("weighted_score") or 0.0) for row in records)
        summary = summaries[label]["reward_and_terminal"]
        summary.update(
            {
                "gold_purchase_rate": strict / 240,
                "strict_gold_successes": strict,
                "purchase_success_rate": purchase / 240,
                "purchase_successes": purchase,
                "reward_valid_rate": valid / 240,
                "reward_valid_tasks": valid,
                "mean_final_reward_fixed_denominator": total_reward / 240,
                "mean_terminal_utility_fixed_denominator": total_reward / 240,
                "mean_weighted_score_fixed_denominator": total_weighted / 240,
                "total_final_reward": total_reward,
                "reward_type_counts": dict(sorted(counts.items())),
                "strict_gold_task_ids": sorted(
                    int(row["task_id"]) for row in records if row["strict_gold"]
                ),
            }
        )


def apply_stratified_reward_overlays(
    summaries: dict[str, dict[str, Any]],
    task_slices: dict[int, dict[str, Any]],
) -> None:
    fields = {
        "suite": "suite",
        "challenge_slice": "challenge_slice",
        "domain": "domain",
    }
    for label in LABELS:
        for group_name, field in fields.items():
            buckets = summaries[label].get("stratified", {}).get(group_name, {})
            for bucket, bucket_summary in buckets.items():
                records = [
                    row
                    for task_id, row in REWARD_OVERLAY[label].items()
                    if str((task_slices.get(task_id) or {}).get(field)) == str(bucket)
                ]
                if not records:
                    continue
                strict = sum(bool(row["strict_gold"]) for row in records)
                purchase = sum(bool(row["purchase_success"]) for row in records)
                valid = sum(bool(row["reward_valid"]) for row in records)
                total_reward = sum(float(row["final_reward"]) for row in records)
                counts = Counter(str(row["reward_type"]) for row in records)
                reward = bucket_summary["reward_and_terminal"]
                reward.update(
                    {
                        "strict_gold_successes": strict,
                        "gold_purchase_rate": strict / len(records),
                        "purchase_successes": purchase,
                        "purchase_success_rate": purchase / len(records),
                        "reward_valid_tasks": valid,
                        "reward_valid_rate": valid / len(records),
                        "mean_final_reward_fixed_denominator": total_reward / len(records),
                        "reward_type_counts": dict(sorted(counts.items())),
                    }
                )


def apply_pairwise_reward_overlays(
    comparison: dict[str, Any],
    expected_ids: list[int],
) -> None:
    for source, target in TRANSITIONS:
        strict_counts = Counter()
        purchase_counts = Counter()
        for task_id in expected_ids:
            source_row = REWARD_OVERLAY[source][task_id]
            target_row = REWARD_OVERLAY[target][task_id]
            strict_counts[
                transition(
                    bool(source_row["strict_gold"]),
                    bool(target_row["strict_gold"]),
                )
            ] += 1
            purchase_counts[
                transition(
                    bool(source_row["purchase_success"]),
                    bool(target_row["purchase_success"]),
                )
            ] += 1
        reward = comparison["pairwise"][f"{source}_to_{target}"][
            "reward_and_terminal"
        ]
        reward["strict_success_transitions"] = dict(sorted(strict_counts.items()))
        reward["purchase_success_transitions"] = dict(
            sorted(purchase_counts.items())
        )


def rubric_category_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return common.summarize_rubric_categories(rows)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def purchase_transition_text(pair: dict[str, Any]) -> str:
    counts = pair["reward_and_terminal"]["purchase_success_transitions"]
    return (
        f"失败→成功 {counts.get('failure_to_success', 0)}；"
        f"成功→失败 {counts.get('success_to_failure', 0)}；"
        f"共同成功 {counts.get('success_to_success', 0)}；"
        f"共同失败 {counts.get('failure_to_failure', 0)}"
    )


def copy_inputs() -> None:
    (OUTPUT / "runs").mkdir(parents=True, exist_ok=True)
    (OUTPUT / "calls").mkdir(exist_ok=True)
    (OUTPUT / "checkpoints").mkdir(exist_ok=True)
    destination = OUTPUT / "runs/grpo230"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(STEP230_SOURCE / "runs/grpo230", destination)
    shutil.copy2(STEP230_SOURCE / "judges-grpo230.jsonl", OUTPUT / "judges-grpo230.jsonl")
    shutil.copy2(
        STEP230_SOURCE / "calls/judges-grpo230.jsonl",
        OUTPUT / "calls/judges-grpo230.jsonl",
    )
    shutil.copy2(
        STEP230_SOURCE / "checkpoints/judges-grpo230.jsonl",
        OUTPUT / "checkpoints/judges-grpo230.jsonl",
    )
    qwen_destination = OUTPUT / "runs/qwen38_27b"
    if qwen_destination.exists():
        shutil.rmtree(qwen_destination)
    shutil.copytree(QWEN_SOURCE / "runs/qwen38_27b", qwen_destination)
    shutil.copy2(
        QWEN_SOURCE / "judges-qwen38_27b.jsonl",
        OUTPUT / "judges-qwen38_27b.jsonl",
    )
    shutil.copy2(
        QWEN_SOURCE / "calls/judges-qwen38_27b.jsonl",
        OUTPUT / "calls/judges-qwen38_27b.jsonl",
    )
    shutil.copy2(
        QWEN_SOURCE / "checkpoints/judges-qwen38_27b.jsonl",
        OUTPUT / "checkpoints/judges-qwen38_27b.jsonl",
    )
    v3_destination = OUTPUT / "runs/grpo230_v3"
    if v3_destination.exists():
        shutil.rmtree(v3_destination)
    shutil.copytree(
        GRPO230_V3_SOURCE / "runs/grpo230_harness_v3",
        v3_destination,
    )
    shutil.copy2(
        GRPO230_V3_SOURCE / "judges-grpo230_harness_v3.jsonl",
        OUTPUT / "judges-grpo230_v3.jsonl",
    )
    shutil.copy2(
        GRPO230_V3_SOURCE / "calls/judges-grpo230_harness_v3.jsonl",
        OUTPUT / "calls/judges-grpo230_v3.jsonl",
    )
    shutil.copy2(
        GRPO230_V3_SOURCE / "checkpoints/judges-grpo230_harness_v3.jsonl",
        OUTPUT / "checkpoints/judges-grpo230_v3.jsonl",
    )
    qwen_v3_destination = OUTPUT / "runs/qwen38_27b_v3"
    if qwen_v3_destination.exists():
        shutil.rmtree(qwen_v3_destination)
    shutil.copytree(
        QWEN_V3_SOURCE / "runs/qwen38_27b_harness_v3",
        qwen_v3_destination,
    )
    shutil.copy2(
        QWEN_V3_SOURCE / "judges-qwen38_27b_harness_v3.jsonl",
        OUTPUT / "judges-qwen38_27b_v3.jsonl",
    )
    shutil.copy2(
        QWEN_V3_SOURCE / "calls/judges-qwen38_27b_harness_v3.jsonl",
        OUTPUT / "calls/judges-qwen38_27b_v3.jsonl",
    )
    shutil.copy2(
        QWEN_V3_SOURCE / "checkpoints/judges-qwen38_27b_harness_v3.jsonl",
        OUTPUT / "checkpoints/judges-qwen38_27b_v3.jsonl",
    )


def render_markdown(
    summaries: dict[str, dict[str, Any]],
    comparison: dict[str, Any],
    evaluations: dict[str, list[dict[str, Any]]],
) -> None:
    overall_rows = []
    metrics = (
        ("严格 Gold", "strict_gold_successes", "gold_purchase_rate", "rate"),
        ("购买成功", "purchase_successes", "purchase_success_rate", "rate"),
        ("Reward有效", "reward_valid_tasks", "reward_valid_rate", "rate"),
        ("平均Final Reward", "mean_final_reward_fixed_denominator", None, "value"),
        ("平均Weighted Score", "mean_weighted_score_fixed_denominator", None, "value"),
    )
    for title, key, rate_key, kind in metrics:
        row = [title]
        for label in LABELS:
            reward = summaries[label]["reward_and_terminal"]
            if kind == "rate":
                row.append(f"{reward[key]}/240 ({pct(reward[rate_key], 2)})")
            else:
                row.append(num(reward[key], 4))
        overall_rows.append(row)

    reward_rows = []
    reward_types = [
        reward_type
        for reward_type in REWARD_ORDER
        if reward_type in REWARD_ALWAYS_SHOW
        or any(
            reward_type
            in summaries[label]["reward_and_terminal"]["reward_type_counts"]
            for label in LABELS
        )
    ]
    for reward_type in reward_types:
        reward_rows.append(
            [reward_type]
            + [summaries[label]["reward_and_terminal"]["reward_type_counts"].get(reward_type, 0) for label in LABELS]
        )

    rubric_rows = []
    for status in RUBRIC_STATUSES:
        rubric_rows.append(
            [status]
            + [summaries[label]["requirement_rubric"]["status_counts"].get(status, 0) for label in LABELS]
        )

    dimension_rows = []
    for dimension in DIMENSIONS:
        dimension_rows.append(
            [DIMENSION_DISPLAY[dimension]]
            + [
                num(
                    summaries[label]["trajectory_quality"]["dimensions"][dimension][
                        "mean_score_among_valid_judges"
                    ],
                    3,
                )
                for label in LABELS
            ]
        )

    behavior_specs = (
        ("平均执行工具调用数", "average_executed_steps_fixed_denominator", 3, False),
        ("平均动作尝试数", "average_action_attempts_fixed_denominator", 3, False),
        ("Guard拒绝次数", "total_guard_rejections", 0, False),
        ("重复动作次数", "total_duplicate_canonical_actions", 0, False),
        ("重复搜索次数", "total_duplicate_search_queries", 0, False),
        ("上下文使用率p50", "context_usage_ratio_p50", 1, True),
        ("上下文使用率p95", "context_usage_ratio_p95", 1, True),
        ("Provider Token p50", "provider_total_tokens_p50", 0, False),
        ("Provider Token p95", "provider_total_tokens_p95", 0, False),
        ("端到端耗时p50(s)", "trajectory_duration_seconds_p50", 1, False),
        ("端到端耗时p95(s)", "trajectory_duration_seconds_p95", 1, False),
        ("Observation投影压缩任务", "tasks_with_observation_truncation", 0, False),
        ("上下文硬溢出任务", "context_hard_limit_tasks", 0, False),
        ("基础设施无效任务", "infrastructure_invalid_tasks", 0, False),
    )
    behavior_rows = []
    for title, key, digits, is_pct in behavior_specs:
        values = []
        for label in LABELS:
            value = summaries[label]["deterministic"].get(key)
            values.append(pct(value, digits) if is_pct else num(value, digits))
        behavior_rows.append([title] + values)

    tool_usage = comparison["tool_usage"]
    tool_rows = []
    for tool in TOOL_ORDER:
        values = []
        for label in LABELS:
            value = tool_usage["models"][label]["counts"][tool]
            values.append("未使用" if value is None else value)
        tool_rows.append([f"`{tool}`"] + values)
    tool_rows.append(
        ["**全部工具调用合计**"]
        + [tool_usage["models"][label]["displayed_total"] for label in LABELS]
    )

    pair_rows = []
    for source, target in TRANSITIONS:
        pair = comparison["pairwise"][f"{source}_to_{target}"]
        pair_rows.append(
            [
                f"{DISPLAY[source]} → {DISPLAY[target]}",
                purchase_transition_text(pair),
            ]
        )

    lines = [
        "# Final-240 七组统一评测：Harness v1 / v2 / v3",
        "",
        "## 评测合同",
        "",
        "- 七组结果使用同一240题 Final-240、同一冻结 DeepSeek V4 Flash Rubric、同一 DeepSeek V4 Pro Judge Prompt与Schema。",
        "- Base、SFT、GRPO100使用Harness v1；原GRPO230与Qwen3.8-27B使用Harness v2；新增GRPO230与Qwen3.8-27B使用Harness v3。各组均按同一盲评合同统计。",
        "- 成功率、Reward值和Reward类型按当前Reward v4聚合重算；Rubric、Judge与确定性过程指标保持冻结口径。",
        "- Judge只看Query、Rubric、Actor可见轨迹和白名单行为指标，不看Reward与Gold私有字段。",
        "",
        "## Harness版本演进",
        "",
        "- **v1 → v2：** 增加35步收敛提醒、模型输出文字但未调用工具时的拒绝纠正，以及循环/无进展提醒；删除 `view_description`、`view_features`、`view_reviews`、`view_attributes` 四种低频信息工具。",
        "- **v2 → v3：** 将搜索页、详情页和普通页面的Observation预算由 `1536 / 4096 / 768` 调整为 `2560 / 3072 / 512` Token；增加已核验候选记忆与页面/阶段级动态 Tool Schema，只向模型暴露当前状态真正可执行的工具。普通搜索首页开放搜索与放弃，搜索结果页开放商品打开、可见翻页/返回与放弃，商品详情页开放未选规格、返回、购买与放弃；进入候选收敛后进一步收紧为：候选选择阶段仅开放 `open_product`，规格阶段仅开放 `select_option`，终局阶段仅开放 `buy_now` 与 `finish_without_purchase`。当循环/无进展达到终止条件时，不再直接结束，而是强制进入候选记忆模块完成最终决策。",
        "- 本报告保留既有Harness v1/v2结果，并将GRPO230·Harness v3 r4与Qwen3.8-27B·Harness v3 r1作为独立新组接入，不覆盖历史结果。",
        "",
        "## Reward版本演进",
        "",
        "- **v3 → v4：** 将“品类为唯一 Hard Gate”升级为基于用户公开 Query 的可审计 Hard/Soft 约束合同。品类始终为 Hard；“必须、一定、绝对不要、不超过、至少、明确区间”等高置信且可确定性核验的不可妥协要求也进入 Hard；“最好、优先、尽量、大约、左右、预算”等偏好或近似表达进入 Soft；无所谓类表达忽略，复杂歧义语义进入 Needs Review / audit-only，不强行参与评分。任一可评分 Hard 失败即判 `wrong_purchase`；Hard 全通过后，目标商品为 Gold，完全满足 Soft 的替代商品为 Valid，只违反 Soft 的替代商品为 Partial。",
        "- v4 新增第16步起的分段递增步数惩罚；将 `assistant_final` 与连续 Guard 拒绝由无效样本改为 `-0.8` 的有效负样本；并重新校准部分终局分数，其中 Partial 调整为 `0.5 + 0.3 × soft_score`，Loop 调整为 `-0.6`。",
        "- **训练版本：** GRPO100使用Reward v3，GRPO230使用Reward v4。为保证横向可比，本报告中的成功率、Reward值与Reward类型仍统一按当前审计版Reward v4对冻结轨迹离线重放，不反向更新模型参数。",
        "",
        "## 总体结果",
        "",
        md_table(["指标"] + [DISPLAY[label] for label in LABELS], overall_rows),
        "",
        "## Reward类型",
        "",
        md_table(["Reward类型"] + [DISPLAY[label] for label in LABELS], reward_rows),
        "",
        "## Rubric总体状态",
        "",
        md_table(["状态"] + [DISPLAY[label] for label in LABELS], rubric_rows),
        "",
        "## LLM Judge五维评分",
        "",
        "评分规则：0分表示关键行为缺失或明显不合理；1分表示部分做到，但覆盖、证据或效率仍有不足；2分表示该维度完成充分且无明显问题。搜索策略衡量检索覆盖、有效改写与机械重复；候选利用衡量高匹配候选的利用、比较与收敛；证据核验衡量购买前对关键属性、规格和最终价格的检查；决策质量衡量商品、规格及购买/放弃决策；终止效率衡量是否过早购买/放弃、无效探索或耗尽步骤。五维独立评分，不加权、不计算总分。",
        "",
        md_table(["维度"] + [DISPLAY[label] for label in LABELS], dimension_rows),
        "",
        "## 行为、Token、耗时和上下文",
        "",
        md_table(["指标"] + [DISPLAY[label] for label in LABELS], behavior_rows),
        "",
        "## 工具调用次数",
        "",
        "逐轨迹汇总实际执行次数；`0` 表示该工具在对应 Tool Schema 中存在但没有被调用，`未使用` 表示当前 8 工具 Schema 已不再暴露该工具。",
        "",
        md_table(["工具"] + [DISPLAY[label] for label in LABELS], tool_rows),
        "",
        "注：Base 历史轨迹另有 4 次已废弃的内部 `think` 调用；它不属于上述 12 个标准购物工具，因此未计入表格。",
        "",
        "## 关键阶段迁移",
        "",
        md_table(["阶段", "成功迁移（Gold + Valid）"], pair_rows),
        "",
        "## 文件",
        "",
        "- `dashboard.html`：七组聚合前端报告。",
        "- `per-task-comparison.csv/json`：Base、SFT、Harness改善版GRPO230三模型逐题审计。",
        "- `comparison.json`：七组两两配对及分层比较；页面迁移卡仍只展示SFT→GRPO230。",
        "- `judges-*.jsonl`：各组Judge原始结构化结果。",
    ]
    (OUTPUT / "audit-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_dashboard(
    summaries: dict[str, dict[str, Any]],
    comparison: dict[str, Any],
    evaluations: dict[str, list[dict[str, Any]]],
    per_task: list[dict[str, Any]],
) -> None:
    expected = 240
    strict_rates = {
        label: summaries[label]["reward_and_terminal"]["gold_purchase_rate"] for label in LABELS
    }
    purchase_rates = {
        label: summaries[label]["reward_and_terminal"]["purchase_success_rate"] for label in LABELS
    }

    def bars(values: dict[str, float], ceiling: float = 1.0) -> str:
        output = []
        for label in LABELS:
            value = float(values.get(label) or 0.0)
            width = max(0.0, min(100.0, value / ceiling * 100))
            output.append(
                f'<div class="bar-row"><span>{DISPLAY[label]}</span><div class="bar-track"><div class="bar-fill" style="width:{width:.2f}%;background:{COLORS[label]}"></div></div><strong>{value:.3f}</strong></div>'
            )
        return "".join(output)

    outcome_specs = (
        ("严格Gold数量", lambda label: summaries[label]["reward_and_terminal"]["strict_gold_successes"]),
        ("购买成功数量", lambda label: summaries[label]["reward_and_terminal"]["purchase_successes"]),
        ("Reward有效数量", lambda label: summaries[label]["reward_and_terminal"]["reward_valid_tasks"]),
        ("平均Final Reward", lambda label: num(summaries[label]["reward_and_terminal"]["mean_final_reward_fixed_denominator"], 4)),
        ("平均Weighted Score", lambda label: num(summaries[label]["reward_and_terminal"]["mean_weighted_score_fixed_denominator"], 4)),
    )
    outcome_rows = "".join(
        "<tr><th>" + esc(title) + "</th>" + "".join(f"<td>{esc(getter(label))}</td>" for label in LABELS) + "</tr>"
        for title, getter in outcome_specs
    )

    reward_types = [
        reward_type
        for reward_type in REWARD_ORDER
        if reward_type in REWARD_ALWAYS_SHOW
        or any(
            reward_type
            in summaries[label]["reward_and_terminal"]["reward_type_counts"]
            for label in LABELS
        )
    ]
    reward_rows = "".join(
        "<tr><th>" + esc(REWARD_DISPLAY.get(reward_type, reward_type)) + "</th>"
        + "".join(
            f"<td>{summaries[label]['reward_and_terminal']['reward_type_counts'].get(reward_type, 0)}</td>"
            for label in LABELS
        )
        + "</tr>"
        for reward_type in reward_types
    )

    rubric_rows = "".join(
        "<tr><th>" + esc(status) + "</th>"
        + "".join(
            f"<td>{summaries[label]['requirement_rubric']['status_counts'].get(status, 0):,}</td>"
            for label in LABELS
        )
        + "</tr>"
        for status in RUBRIC_STATUSES
    )

    hardnesses = ("hard", "soft", "needs_review")
    hardness_rows = []
    for hardness in hardnesses:
        cells = []
        for label in LABELS:
            counts = summaries[label]["requirement_rubric"]["status_counts_by_hardness"].get(hardness, {})
            cells.append(
                "<td class='status-stack'>"
                + "".join(f"<span>{status}<strong>{counts.get(status, 0):,}</strong></span>" for status in RUBRIC_STATUSES)
                + "</td>"
            )
        hardness_rows.append(f"<tr><th>{hardness}</th>{''.join(cells)}</tr>")

    category_counts = {label: rubric_category_counts(evaluations[label]) for label in LABELS}
    category_rows = []
    for category in common.RUBRIC_CATEGORY_ORDER:
        total = sum(sum(category_counts[label][category].values()) for label in LABELS)
        if total == 0:
            continue
        cells = []
        for label in LABELS:
            counts = category_counts[label][category]
            cells.append(
                "<td class='status-stack'>"
                + "".join(f"<span>{status}<strong>{counts.get(status, 0):,}</strong></span>" for status in RUBRIC_STATUSES)
                + "</td>"
            )
        category_rows.append(
            f"<tr><th>{esc(common.RUBRIC_CATEGORY_DISPLAY[category])}</th>{''.join(cells)}</tr>"
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
            f'<div class="dimension"><h3>{DIMENSION_DISPLAY[dimension]}</h3>{bars(values, ceiling=2.0)}</div>'
        )

    error_totals = Counter()
    for label in LABELS:
        error_totals.update(summaries[label]["trajectory_quality"].get("primary_error_counts", {}))
    top_errors = [name for name, _ in error_totals.most_common(15)]
    error_rows = "".join(
        "<tr><th>" + esc(common.error_display(error_name)) + "</th>"
        + "".join(
            f"<td>{summaries[label]['trajectory_quality']['primary_error_counts'].get(error_name, 0)}</td>"
            for label in LABELS
        )
        + "</tr>"
        for error_name in top_errors
    )
    error_total_cells = "".join(
        "<td><strong>"
        + str(
            sum(
                summaries[label]["trajectory_quality"]
                ["primary_error_counts"].get(error_name, 0)
                for error_name in top_errors
            )
        )
        + "</strong></td>"
        for label in LABELS
    )
    error_rows += (
        f'<tr class="total-row"><th>上表主要错误合计</th>'
        f"{error_total_cells}</tr>"
    )

    behavior_specs = (
        ("平均执行工具调用数 / 任务", "average_executed_steps_fixed_denominator", 3, False),
        ("平均动作尝试数 / 任务", "average_action_attempts_fixed_denominator", 3, False),
        ("Guard拒绝次数", "total_guard_rejections", 0, False),
        ("重复动作次数", "total_duplicate_canonical_actions", 0, False),
        ("重复搜索次数", "total_duplicate_search_queries", 0, False),
        ("上下文使用率 p50", "context_usage_ratio_p50", 1, True),
        ("上下文使用率 p95", "context_usage_ratio_p95", 1, True),
        ("Provider Token p50 / 任务", "provider_total_tokens_p50", 0, False),
        ("Provider Token p95 / 任务", "provider_total_tokens_p95", 0, False),
        ("端到端耗时 p50（秒）", "trajectory_duration_seconds_p50", 1, False),
        ("端到端耗时 p95（秒）", "trajectory_duration_seconds_p95", 1, False),
        ("Observation投影压缩（任务数）", "tasks_with_observation_truncation", 0, False),
        ("上下文硬溢出（任务数）", "context_hard_limit_tasks", 0, False),
        ("基础设施无效任务", "infrastructure_invalid_tasks", 0, False),
    )
    behavior_rows = []
    for title, key, digits, as_pct in behavior_specs:
        cells = []
        for label in LABELS:
            value = summaries[label]["deterministic"].get(key)
            cells.append(f"<td>{pct(value, digits) if as_pct else num(value, digits)}</td>")
        behavior_rows.append(f"<tr><th>{esc(title)}</th>{''.join(cells)}</tr>")

    tool_usage = comparison["tool_usage"]
    tool_rows = []
    for tool in TOOL_ORDER:
        cells = []
        for label in LABELS:
            value = tool_usage["models"][label]["counts"][tool]
            if value is None:
                cells.append('<td class="unused">未使用</td>')
            else:
                cells.append(f"<td>{value:,}</td>")
        tool_rows.append(f"<tr><th><code>{esc(tool)}</code></th>{''.join(cells)}</tr>")
    total_cells = "".join(
        f"<td><strong>{tool_usage['models'][label]['displayed_total']:,}</strong></td>"
        for label in LABELS
    )
    tool_rows.append(
        f'<tr class="total-row"><th>全部工具调用合计</th>{total_cells}</tr>'
    )

    pair_cards = []
    for source, target in TRANSITIONS:
        pair = comparison["pairwise"][f"{source}_to_{target}"]
        pair_cards.append(
            f'<article class="card transition"><strong>{DISPLAY[source]} → {DISPLAY[target]}</strong><br>{esc(purchase_transition_text(pair))}<div class="label">成功 = Gold + Valid</div></article>'
        )

    stratified_rows = []
    for group_name in ("suite", "challenge_slice", "domain"):
        buckets = summaries["base"].get("stratified", {}).get(group_name, {})
        for bucket in sorted(buckets):
            task_count = summaries["base"]["stratified"][group_name][bucket]["expected_tasks"]
            rates = [
                summaries[label]["stratified"][group_name][bucket]["reward_and_terminal"]["purchase_success_rate"]
                for label in LABELS
            ]
            rate_by_label = dict(zip(LABELS, rates, strict=True))
            delta = (rate_by_label["grpo230"] - rate_by_label["sft"]) * 100
            stratified_rows.append(
                f"<tr><td>{esc(group_name)}</td><td>{esc(bucket)}</td><td>{task_count}</td>"
                + "".join(f"<td>{pct(rate)}</td>" for rate in rates)
                + f"<td>{delta:+.1f} pp</td></tr>"
            )

    coverage_values = {
        label: summaries[label]["trajectory_quality"]["judge_coverage_rate"] for label in LABELS
    }

    task_rows = []
    for row in per_task:
        search = " ".join(
            str(row.get(f"{label}_{field}", ""))
            for label in TASK_LABELS
            for field in ("reward_type", "judge_status", "primary_error")
        ).casefold()
        transition_value = row["sft_to_grpo230_purchase_transition"]
        cells = [f"<td>{row['task_id']}</td>"]
        for label in TASK_LABELS:
            cells.extend(
                [
                    f"<td>{'✓' if row[f'{label}_purchase_success'] else '—'}</td>",
                    f"<td>{esc(row[f'{label}_reward_type'])}</td>",
                    f"<td>{esc(row[f'{label}_judge_status'])}</td>",
                    f"<td>{esc(row[f'{label}_primary_error'])}</td>",
                ]
            )
        task_rows.append(
            f'<tr data-search="{esc(search)}" data-transition="{transition_value}">{"".join(cells)}</tr>'
        )

    headers = "".join(f"<th>{DISPLAY[label]} 成功<br><small>Gold+Valid</small></th><th>{DISPLAY[label]} reward</th><th>{DISPLAY[label]} judge</th><th>{DISPLAY[label]} primary</th>" for label in TASK_LABELS)
    th_models = "".join(f"<th>{DISPLAY[label]}</th>" for label in LABELS)
    legend = "".join(
        f'<span><i class="dot" style="background:{COLORS[label]}"></i>{DISPLAY[label]}</span>' for label in LABELS
    )
    kpis = "".join(
        f'<article class="card kpi" style="border-top-color:{COLORS[label]}"><div class="label">{DISPLAY[label]} 严格成功率</div><div class="value">{pct(strict_rates[label])}</div><div class="context">{summaries[label]["reward_and_terminal"]["strict_gold_successes"]} / {expected} Gold</div></article>'
        for label in LABELS
    )

    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Final-240 七组完整评测</title><style>
:root{{font-family:Inter,"Microsoft YaHei",system-ui,sans-serif;background:#f4f7fb;color:#13213b}}*{{box-sizing:border-box}}body{{margin:0}}.wrap{{max-width:1680px;margin:auto;padding:0 28px 64px}}.hero{{background:linear-gradient(135deg,#0c1936,#4a2674);color:white;padding:44px 48px;border-radius:0 0 32px 32px;box-shadow:0 24px 60px #111c4733}}.hero h1{{font-size:42px;margin:14px 0}}.hero p{{max-width:1200px;color:#dbe3ff;font-size:18px;line-height:1.6}}section{{margin-top:38px}}h2{{font-size:27px;margin-bottom:8px}}h3{{font-size:17px}}.lead{{color:#60718e;line-height:1.6}}.grid{{display:grid;gap:18px}}.kpis{{grid-template-columns:repeat(6,minmax(0,1fr))}}.two{{grid-template-columns:repeat(2,minmax(0,1fr))}}.three{{grid-template-columns:repeat(3,minmax(0,1fr))}}.card{{background:white;border:1px solid #e1e8f2;border-radius:20px;padding:24px;box-shadow:0 9px 28px #1225460f;min-width:0}}.section-card{{margin-top:18px}}.kpi{{border-top:4px solid}}.label,.context{{color:#687a97;font-size:14px}}.value{{font-size:32px;font-weight:800;margin:9px 0}}.bar-row{{display:grid;grid-template-columns:145px 1fr 65px;gap:12px;align-items:center;margin:11px 0}}.bar-track{{height:13px;background:#edf1f7;border-radius:99px;overflow:hidden}}.bar-fill{{height:100%;border-radius:99px}}.bar-row strong{{text-align:right}}.dimensions{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:11px 9px;border-bottom:1px solid #edf1f6;text-align:right;vertical-align:top}}th:first-child,td:first-child{{text-align:left}}thead th{{color:#61718c;position:sticky;top:0;background:white;z-index:1}}.status-stack{{min-width:150px}}.status-stack span{{display:flex;justify-content:space-between;gap:12px;line-height:1.7;color:#60718e}}.unused{{color:#94a3b8;font-style:italic}}.table-wrap{{overflow:auto;max-height:720px}}.horizontal-only{{max-height:none;overflow-x:auto;overflow-y:visible}}.task-table{{min-width:1450px}}.task-table tbody tr[hidden]{{display:none}}.transition{{line-height:1.7}}.transition strong{{font-size:20px}}.filters{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px}}.filters input,.filters select{{font:inherit;border:1px solid #cfd9e8;border-radius:10px;padding:10px 12px;background:white}}.filters input{{min-width:280px;flex:1}}.chip{{border:1px solid #ffffff38;border-radius:999px;padding:7px 12px;background:#ffffff14;display:inline-block;margin:4px}}.legend{{display:flex;gap:18px;flex-wrap:wrap;color:#dbe3ff}}.dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}}.note{{border-left:4px solid #2563eb;background:#eef5ff;padding:16px 18px;border-radius:9px;color:#36506f}}a{{color:#2563eb}}@media(max-width:1100px){{.kpis,.two,.three{{grid-template-columns:1fr 1fr}}.dimensions{{grid-template-columns:1fr}}}}@media(max-width:700px){{.wrap{{padding:0 14px 40px}}.hero{{padding:32px 22px}}.hero h1{{font-size:30px}}.kpis,.two,.three{{grid-template-columns:1fr}}}}
</style></head><body><div class="wrap"><header class="hero"><div>SHOPBENCH-LH · FINAL-240 · REWARD v4 · DEEPSEEK V4 PRO JUDGE</div><h1>Base v1 / SFT v1 / GRPO100 v1 / GRPO230 v2 / Qwen3.8-27B v2 / GRPO230 v3 / Qwen3.8-27B v3</h1><p>七组使用同一240题、同一冻结DeepSeek V4 Flash Rubric和同一盲评合同；Harness v3分别展示GRPO230 r4与Qwen3.8-27B r1，均由官网DeepSeek V4 Pro完成盲评。</p><div>{legend}</div></header>
<section><h2>总览</h2><div class="grid kpis">{kpis}</div></section>
<section><h2>Harness版本演进</h2><div class="grid two"><article class="card"><h3>v1 → v2</h3><p class="lead">增加35步收敛提醒、模型输出文字但未调用工具时的拒绝纠正，以及循环/无进展提醒；删除 <code>view_description</code>、<code>view_features</code>、<code>view_reviews</code>、<code>view_attributes</code> 四种低频信息工具。</p></article><article class="card"><h3>v2 → v3</h3><p class="lead">将搜索页、详情页和普通页面的Observation预算由 <code>1536 / 4096 / 768</code> 调整为 <code>2560 / 3072 / 512</code> Token；增加已核验候选记忆与页面/阶段级动态 Tool Schema，只暴露当前状态真正可执行的工具。当循环/无进展达到终止条件时，不再直接结束，而是强制进入候选记忆模块完成最终决策。</p><table><tbody><tr><th>普通搜索首页</th><td>搜索、放弃</td></tr><tr><th>搜索结果页</th><td>打开商品、当前可见翻页/返回、放弃</td></tr><tr><th>商品详情页</th><td>选择未选规格、返回、购买、放弃</td></tr><tr><th>候选选择阶段</th><td>仅 <code>open_product</code></td></tr><tr><th>候选规格阶段</th><td>仅 <code>select_option</code></td></tr><tr><th>候选终局阶段</th><td>仅 <code>buy_now</code>、<code>finish_without_purchase</code></td></tr></tbody></table></article></div><p class="note">GRPO230与Qwen3.8-27B的Harness v3结果作为独立新组展示，历史Harness v1/v2结果保持不变。</p></section>
<section><h2>Reward版本演进</h2><div class="grid two"><article class="card"><h3>v3 → v4：Hard / Soft语义合同</h3><p class="lead">从“品类为唯一 Hard Gate”升级为基于用户公开 Query 的可审计 Hard/Soft 约束合同。品类始终为 Hard；“必须、一定、绝对不要、不超过、至少、明确区间”等高置信且可确定性核验的不可妥协要求也进入 Hard；“最好、优先、尽量、大约、左右、预算”等偏好或近似表达进入 Soft。无所谓类表达忽略，复杂歧义语义进入 Needs Review / audit-only，不强行参与评分。</p><p class="lead">任一可评分 Hard 失败即判 <code>wrong_purchase</code>；Hard 全通过后，目标商品为 Gold，完全满足 Soft 的替代商品为 Valid，只违反 Soft 的替代商品为 Partial。</p></article><article class="card"><h3>效率与终局分数调整</h3><p class="lead">新增第16步起的分段递增步数惩罚；将 <code>assistant_final</code> 与连续 Guard 拒绝由无效样本改为 <code>-0.8</code> 的有效负样本；并重新校准部分终局分数，其中 Partial 为 <code>0.5 + 0.3 × soft_score</code>，Loop 为 <code>-0.6</code>。</p><table><thead><tr><th>Checkpoint</th><th>训练 Reward</th></tr></thead><tbody><tr><td>GRPO100</td><td>Reward v3</td></tr><tr><td>GRPO230</td><td>Reward v4</td></tr></tbody></table></article></div><p class="note">训练版本与评测口径分开记录：Final-240中的成功率、Reward值和Reward类型统一按当前审计版Reward v4对冻结轨迹离线重放，便于横向比较，不会反向更新模型参数。</p></section>
<section><h2>1. 任务结果（Reward v4聚合重算）</h2><div class="grid two"><article class="card"><h3>Strict Gold成功率</h3>{bars(strict_rates)}</article><article class="card"><h3>购买成功率</h3>{bars(purchase_rates)}</article></div><article class="card section-card"><h3>总体指标</h3><table><thead><tr><th>指标</th>{th_models}</tr></thead><tbody>{outcome_rows}</tbody></table></article><article class="card section-card"><h3>Reward类型分布</h3><table><thead><tr><th>Reward类型</th>{th_models}</tr></thead><tbody>{reward_rows}</tbody></table></article></section>
<section><h2>2. Rubric需求满足</h2><article class="card"><h3>Rubric总体状态</h3><p class="lead">每条要求由同一DeepSeek V4 Pro合同标记为satisfied、violated、unknown或not_applicable。</p><table><thead><tr><th>状态</th>{th_models}</tr></thead><tbody>{rubric_rows}</tbody></table></article><article class="card section-card"><h3>Hard / Soft约束</h3><div class="table-wrap"><table><thead><tr><th>强度</th>{th_models}</tr></thead><tbody>{''.join(hardness_rows)}</tbody></table></div></article><article class="card section-card"><h3>按要求类型拆分</h3><div class="table-wrap"><table><thead><tr><th>要求类型</th>{th_models}</tr></thead><tbody>{''.join(category_rows)}</tbody></table></div></article></section>
<section><h2>3. 轨迹质量与错误归因</h2><div class="grid two"><article class="card"><h3>LLM Judge五维评分（0–2）</h3><div class="note" style="margin-bottom:18px"><strong>分值：</strong>0 = 关键行为缺失或明显不合理；1 = 部分做到，但覆盖、证据或效率仍有不足；2 = 完成充分且无明显问题。五维独立评分，不加权、不计算总分。<br><strong>维度：</strong>搜索策略看检索覆盖、有效改写和机械重复；候选利用看高匹配候选的利用、比较与收敛；证据核验看购买前是否检查关键属性、规格和最终价格；决策质量看商品、规格及购买/放弃是否合理；终止效率看是否过早购买/放弃、无效探索或耗尽步骤。</div><div class="dimensions">{''.join(dimension_blocks)}</div></article><article class="card"><h3>Primary Error</h3><div class="table-wrap horizontal-only"><table><thead><tr><th>错误</th>{th_models}</tr></thead><tbody>{error_rows}</tbody></table></div></article></div></section>
<section><h2>4. 行为、Token、耗时与上下文</h2><article class="card"><div class="table-wrap"><table><thead><tr><th>指标</th>{th_models}</tr></thead><tbody>{''.join(behavior_rows)}</tbody></table></div></article></section>
<section><h2>5. 工具调用次数</h2><article class="card"><p class="lead">逐轨迹汇总实际执行次数。0 表示该工具在对应 Tool Schema 中存在但没有被调用；“未使用”表示当前 8 工具 Schema 已不再暴露该工具。</p><div class="table-wrap"><table><thead><tr><th>工具</th>{th_models}</tr></thead><tbody>{''.join(tool_rows)}</tbody></table></div><p class="lead">Base 历史轨迹另有 4 次已废弃的内部 <code>think</code> 调用；它不属于上述 12 个标准购物工具，因此未计入表格。</p></article></section>
<section><h2>6. 阶段迁移与分层表现</h2><div class="grid two">{''.join(pair_cards)}</div><article class="card section-card"><h3>分层成功率（Gold + Valid）</h3><div class="table-wrap"><table><thead><tr><th>分层</th><th>子集</th><th>任务数</th>{th_models}<th>GRPO230−SFT</th></tr></thead><tbody>{''.join(stratified_rows)}</tbody></table></div></article></section>
<section><h2>7. 逐题审计</h2><article class="card"><h3>240题逐题审计（仅Base / SFT / Harness改善版GRPO230；成功 = Gold + Valid）</h3><div class="filters"><input id="task-search" placeholder="搜索Reward、Judge状态或错误"><select id="transition-filter"><option value="">全部 SFT→GRPO230 转移</option><option value="failure_to_success">失败→成功</option><option value="success_to_success">成功→成功</option><option value="success_to_failure">成功→失败</option><option value="failure_to_failure">失败→失败</option></select><span id="visible-count"></span><a href="per-task-comparison.csv">CSV</a><a href="per-task-comparison.json">JSON</a></div><div class="table-wrap"><table class="task-table"><thead><tr><th>Task</th>{headers}</tr></thead><tbody id="task-body">{''.join(task_rows)}</tbody></table></div></article></section>
<footer><p><a href="audit-report.md">审计报告</a> · <a href="comparison.json">Comparison JSON</a> · <a href="run_manifest.json">Run Manifest</a></p></footer></div><script>
const search=document.getElementById('task-search'),filter=document.getElementById('transition-filter'),rows=[...document.querySelectorAll('#task-body tr')],count=document.getElementById('visible-count');function apply(){{const q=search.value.toLowerCase(),t=filter.value;let n=0;rows.forEach(r=>{{const show=(!q||r.dataset.search.includes(q))&&(!t||r.dataset.transition===t);r.hidden=!show;if(show)n++;}});count.textContent=`显示 ${{n}} / ${{rows.length}}`;}}search.addEventListener('input',apply);filter.addEventListener('change',apply);apply();
</script></body></html>"""
    document = document.replace(
        "每条要求由同一DeepSeek V4 Pro合同标记为satisfied、violated、unknown或not_applicable。",
        "七组共享同一DeepSeek V4 Flash冻结Rubric，并由同一DeepSeek V4 Pro盲评合同标记satisfied、violated、unknown或not_applicable。",
    )
    (OUTPUT / "dashboard.html").write_text(document, encoding="utf-8")


def main() -> None:
    copy_inputs()
    summaries = {
        label: load_json(OUTPUT / "runs" / label / "summary.json") for label in LABELS
    }
    evaluations = {
        label: load_jsonl(OUTPUT / "runs" / label / "evaluations.jsonl") for label in LABELS
    }
    prepare_reward_overlays(evaluations)
    step230_context_recalculation = apply_step230_provider_context_usage(evaluations)
    qwen_context_recalculation = apply_qwen_provider_context_usage(evaluations)
    grpo230_v3_context_recalculation = apply_provider_context_usage_for_run(
        evaluations,
        label="grpo230_v3",
        trajectories_path=GRPO230_V3_TRAJECTORIES,
        output_path=GRPO230_V3_CONTEXT_RECALCULATION,
    )
    qwen_v3_context_recalculation = apply_provider_context_usage_for_run(
        evaluations,
        label="qwen38_27b_v3",
        trajectories_path=QWEN_V3_TRAJECTORIES,
        output_path=QWEN_V3_CONTEXT_RECALCULATION,
    )
    enrich(summaries, evaluations)
    apply_reward_summary_overlays(summaries)
    expected_ids = [int(row["task_id"]) for row in load_jsonl(ROOT / "data/evaluation/tasks.jsonl")]
    slices = {int(row["task_id"]): row for row in load_jsonl(ROOT / "data/evaluation/slices.jsonl")}
    apply_stratified_reward_overlays(summaries, slices)
    comparison = compare_evaluation_runs(
        expected_task_ids=expected_ids, runs=evaluations, task_slices=slices
    )
    apply_pairwise_reward_overlays(comparison, expected_ids)
    comparison["tool_usage"] = build_tool_usage(evaluations)
    (OUTPUT / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    indexes = {
        label: {int(row["task_id"]): row for row in evaluations[label]} for label in LABELS
    }
    per_task = build_per_task(indexes, expected_ids)
    (OUTPUT / "per-task-comparison.json").write_text(
        json.dumps(per_task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (OUTPUT / "per-task-comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_task[0]))
        writer.writeheader()
        writer.writerows(per_task)

    render_markdown(summaries, comparison, evaluations)
    render_dashboard(summaries, comparison, evaluations, per_task)

    reference_manifest = load_json(REFERENCE / "run_manifest.json")
    source_audit = {
        "historical_four_stage_source": str(REFERENCE.relative_to(ROOT)),
        "grpo230_source": str((STEP230_SOURCE / "runs/grpo230").relative_to(ROOT)),
        "grpo230_trajectory_source": str(
            STEP230_TRAJECTORIES.relative_to(ROOT)
        ),
        "grpo230_context_usage_recalculation": str(
            STEP230_CONTEXT_RECALCULATION.relative_to(REFERENCE)
        ),
        "grpo230_context_usage_method": step230_context_recalculation["method"],
        "grpo230_context_usage_corrected": step230_context_recalculation[
            "corrected"
        ],
        "grpo230_judges_sha256": sha256(OUTPUT / "judges-grpo230.jsonl"),
        "qwen38_27b_source": str(
            (QWEN_SOURCE / "runs/qwen38_27b").relative_to(ROOT)
        ),
        "qwen38_27b_judges_sha256": sha256(OUTPUT / "judges-qwen38_27b.jsonl"),
        "grpo230_v3_source": str(GRPO230_V3_SOURCE.relative_to(ROOT)),
        "grpo230_v3_judges_sha256": sha256(
            OUTPUT / "judges-grpo230_v3.jsonl"
        ),
        "grpo230_v3_context_usage_recalculation": str(
            GRPO230_V3_CONTEXT_RECALCULATION.relative_to(REFERENCE)
        ),
        "grpo230_v3_context_usage_method": grpo230_v3_context_recalculation[
            "method"
        ],
        "grpo230_v3_context_usage_corrected": grpo230_v3_context_recalculation[
            "corrected"
        ],
        "qwen38_27b_v3_source": str(QWEN_V3_SOURCE.relative_to(ROOT)),
        "qwen38_27b_v3_judges_sha256": sha256(
            OUTPUT / "judges-qwen38_27b_v3.jsonl"
        ),
        "qwen38_27b_v3_context_usage_recalculation": str(
            QWEN_V3_CONTEXT_RECALCULATION.relative_to(REFERENCE)
        ),
        "qwen38_27b_v3_context_usage_method": qwen_v3_context_recalculation[
            "method"
        ],
        "qwen38_27b_v3_context_usage_corrected": qwen_v3_context_recalculation[
            "corrected"
        ],
        "qwen38_27b_judge_provider_split": {
            "opencode_go_completed": 120,
            "deepseek_official_completed": 120,
            "model": "deepseek-v4-pro",
            "thinking": False,
        },
        "qwen38_27b_context_usage_recalculation": str(
            QWEN_CONTEXT_RECALCULATION.relative_to(REFERENCE)
        ),
        "qwen38_27b_context_usage_method": qwen_context_recalculation["method"],
        "qwen38_27b_context_usage_corrected": qwen_context_recalculation[
            "corrected"
        ],
        "frozen_rubric_sha256": sha256(OUTPUT / "rubrics.jsonl"),
        "judge_protocol_aligned": True,
        "newly_judged_models": [
            "grpo230",
            "qwen38_27b",
            "grpo230_v3",
            "qwen38_27b_v3",
        ],
        "grpo230_display_label": DISPLAY["grpo230"],
        "grpo230_v3_display_label": DISPLAY["grpo230_v3"],
        "qwen38_27b_v3_display_label": DISPLAY["qwen38_27b_v3"],
    }
    (OUTPUT / "source_audit.json").write_text(
        json.dumps(source_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    historical_actors = reference_manifest["models"]["actors"]
    manifest = {
        "schema_version": "seven-run-trajectory-judge-report-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": reference_manifest["protocol"],
        "models": {
            "rubric_curator": reference_manifest["models"]["rubric_curator"],
            "trajectory_judge": reference_manifest["models"]["trajectory_judge"],
            "actors": {
                "base": historical_actors["base"],
                "sft": historical_actors["sft"],
                "grpo100": historical_actors["grpo100"],
                "grpo230": {
                    "label": "Harness改善版 GRPO step230",
                    "model": "qwen35-2b-grpo-step230-latest-prompt-r1",
                    "tokenizer": "qwen35-2b-grpo-v4-step230-hf-merged-sftv5-20260821",
                },
                "qwen38_27b": {
                    "label": "Qwen3.8-27B base non-thinking",
                    "model": "qwen38-27b-base-nonthinking-r1",
                    "tokenizer": "Qwen3.8-27B",
                },
                "grpo230_v3": load_json(
                    GRPO230_V3_SOURCE / "run_manifest.json"
                )["models"]["actors"]["grpo230_harness_v3"],
                "qwen38_27b_v3": load_json(
                    QWEN_V3_SOURCE / "run_manifest.json"
                )["models"]["actors"]["qwen38_27b_harness_v3"],
            },
        },
        "reuse": {
            "base": "reused",
            "sft": "reused",
            "grpo100": "reused",
            "grpo230": "new_official_deepseek_v4_pro_judge",
            "qwen38_27b": "new_deepseek_v4_pro_judge",
            "grpo230_v3": "new_official_deepseek_v4_pro_judge_r4",
            "qwen38_27b_v3": "new_official_deepseek_v4_pro_judge_r1",
            "rubrics": "reused_frozen_flash_rubrics",
        },
        "call_totals": {
            "rubrics": metadata_totals(OUTPUT / "calls/rubrics.jsonl"),
            "judges": {
                label: metadata_totals(OUTPUT / f"calls/judges-{label}.jsonl")
                for label in LABELS
            },
        },
        "summaries": summaries,
        "source_audit": source_audit,
    }
    files = [
        path
        for path in OUTPUT.rglob("*")
        if path.is_file() and path != OUTPUT / "run_manifest.json"
    ]
    manifest["outputs"] = {str(path.relative_to(OUTPUT)): sha256(path) for path in files}
    (OUTPUT / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "dashboard": str(OUTPUT / "dashboard.html"),
                "tasks": len(per_task),
                "judge_counts": {
                    label: len(load_jsonl(OUTPUT / f"judges-{label}.jsonl")) for label in LABELS
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
    main()
