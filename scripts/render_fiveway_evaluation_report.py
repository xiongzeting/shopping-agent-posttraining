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
NORMAL_SFT_SOURCE = (
    ROOT
    / "outputs/evaluation/final240-harness-v1-sft-normal1000-control-v1-30k-12tool-20way-768-context30000-maxsteps45-temp00-topp1-20260829-r2-deepseek-v4-pro-judge-20260830"
)
NORMAL_SFT_ROLLOUT_SOURCE = (
    ROOT
    / "outputs/evaluation/final240-harness-v1-sft-normal1000-control-v1-30k-12tool-20way-768-context30000-maxsteps45-temp00-topp1-20260829-r2"
)
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
    / "评测轨迹"
    / "05-GRPO230-v3-r4"
    / "trajectories.jsonl"
)
QWEN_V3_TRAJECTORIES = (
    REFERENCE
    / "评测轨迹"
    / "06-Qwen38-27B-v3-r1"
    / "trajectories.jsonl"
)
SFT_FINAL1000 = ROOT / "data/sft/all.jsonl"
SFT_TRAJECTORIES = (
    REFERENCE / "评测轨迹/02-SFT-v1/trajectories.jsonl"
)
GRPO230_V3_CONTEXT_RECALCULATION = (
    REFERENCE / "grpo230-v3-context-usage-recalculation.json"
)
QWEN_V3_CONTEXT_RECALCULATION = (
    REFERENCE / "qwen38-27b-v3-context-usage-recalculation.json"
)
FOUR_ROUND_METRICS = (
    REFERENCE / "四轮评测/pass-at-4-and-pass-power-4.json"
)
QWEN_TRAJECTORIES = (
    ROOT
    / "outputs/evaluation/final240-v24-qwen38-27b-base-step230-config-nonthinking-r1-20260823/trajectories.jsonl"
)
REWARD_V4_REPLAY_DIR = (
    REFERENCE / "reward-v4-hard-soft-v2-current-query-20260825"
)
QWEN_CONTEXT_RECALCULATION = REFERENCE / "qwen38-27b-context-usage-recalculation.json"
STEP230_TRAJECTORIES = (
    REFERENCE / "评测轨迹/04-GRPO230-v2/trajectories.jsonl"
)
STEP230_CONTEXT_RECALCULATION = (
    REFERENCE
    / "评测轨迹/04-GRPO230-v2/context-usage-recalculation.json"
)
OUTPUT = REFERENCE
LABELS = (
    "base",
    "sft_normal1000",
    "sft",
    "grpo100",
    "grpo230",
    "qwen38_27b",
    "grpo230_v3",
    "qwen38_27b_v3",
)
TASK_LABELS = LABELS
TRANSITIONS = (
    ("sft", "grpo100"),
    ("grpo100", "grpo230"),
    ("grpo230", "grpo230_v3"),
)
DISPLAY = {
    "base": "Base v1",
    "sft_normal1000": "普通 SFT v1",
    "sft": "纠错 SFT v1",
    "grpo50": "GRPO50 v1",
    "grpo100": "GRPO100 v1",
    "grpo230": "GRPO230 v2",
    "qwen38_27b": "Qwen3.8-27B v2",
    "grpo230_v3": "GRPO230 v3",
    "qwen38_27b_v3": "Qwen3.8-27B v3",
}
COLORS = {
    "base": "#64748b",
    "sft_normal1000": "#9333ea",
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
SUCCESS_STRATIFICATION_SPECS = {
    "retrieval_bucket": {
        "title": "按 Gold Recall 位置分层成功率",
        "description": (
            "Recall位置来自冻结任务元数据：表示Gold商品在ShopSimulator BM25候选列表中的排名；"
            "missing表示前150个候选未召回。成功口径为Gold + Valid。"
        ),
        "order": ("rank1", "rank2_5", "rank6_20", "rank21_150", "missing"),
        "labels": {
            "rank1": "Rank 1",
            "rank2_5": "Rank 2–5",
            "rank6_20": "Rank 6–20",
            "rank21_150": "Rank 21–150",
            "missing": "Top-150未召回",
        },
    },
    "difficulty_bucket": {
        "title": "按任务难度分层成功率",
        "description": (
            "难度分桶来自冻结任务元数据difficulty_score；分数综合属性数量、规格轴与Variant数量、"
            "Query长度以及Query与标题的词面差异。"
            "成功口径为Gold + Valid。"
        ),
        "order": ("under_10", "10_15", "15_18", "18_plus"),
        "labels": {
            "under_10": "难度 < 10",
            "10_15": "难度 10–15",
            "15_18": "难度 15–18",
            "18_plus": "难度 ≥ 18",
        },
    },
}
TEACHER_CORRECTIVE_CASE_SPECS = (
    {
        "strategy": "loop_recovery",
        "strategy_display": "循环恢复",
        "task_id": 11555,
        "accent": "#7c3aed",
        "headline": "规格闭合后发现Variant超预算，及时换候选并精确收敛",
        "correction": (
            "先排除缺少2米软管、扳手和卡扣的近似套餐；随后选择完整套餐后发现"
            "实际价格29元超出19元左右预算，没有停留或重复核验，而是立即转向新候选。"
        ),
        "learning": (
            "通过加入全铜、6分、排污、软管和扳手等区分条件，最终找到包含2米管、"
            "扳手与卡扣且实际价格正好19元的完整Variant并购买。"
        ),
    },
    {
        "strategy": "near_miss_rejection",
        "strategy_display": "近似商品拒绝",
        "task_id": 9512,
        "accent": "#dc2626",
        "headline": "标题相似不等于满足核心成分：空白贴布不是中药贴",
        "correction": (
            "对比候选虽然包含“三伏贴”关键词，但详情明确是无纺布空白贴，"
            "缺少用户要求的中药成分，因此被明确淘汰。"
        ),
        "learning": (
            "回访已经核验的中药贴候选，重新选择30贴与纯中药规格，"
            "按完整Variant价格119元完成购买。"
        ),
    },
    {
        "strategy": "option_grounding",
        "strategy_display": "精确规格落地",
        "task_id": 22436,
        "accent": "#0f9f8f",
        "headline": "三个规格轴逐项落地，避免只凭商品标题购买",
        "correction": (
            "商品本体匹配并不代表Variant匹配；轨迹只使用Observation中真实存在的选项，"
            "依次选择金色框、1.61非球面防蓝光镜片和散光定制。"
        ),
        "learning": (
            "选择完成后以最终价格105元重新核验预算，再调用终局工具，"
            "监督模型学习规格值必须页面可见且组合完整。"
        ),
    },
    {
        "strategy": "terminal_tool_commit",
        "strategy_display": "终局工具提交",
        "task_id": 15956,
        "accent": "#2563eb",
        "headline": "决策证据闭合后显式提交购买，不停在自然语言结论",
        "correction": (
            "完成品牌、型号、年龄、软毛、蓝色、买一送一和70元价格核验后，"
            "最后一轮仍必须产生合法终局工具调用。"
        ),
        "learning": (
            "Assistant先给出10项约束清单，随后同一回合调用buy_now，"
            "避免“已经分析正确但没有真正完成环境任务”。"
        ),
    },
)
TYPICAL_REWARD_TYPES = (
    "gold_purchase",
    "valid_alternative_purchase",
    "partial_alternative_purchase",
    "wrong_purchase",
    "assistant_final",
    "guard_rejection",
    "repeat_loop",
    "early_abstain",
)
TYPICAL_CASE_OVERRIDES = {
    "gold_purchase": ("grpo230_v3", 1456),
    "valid_alternative_purchase": ("grpo230_v3", 1249),
}
TYPICAL_REWARD_DESCRIPTIONS = {
    "gold_purchase": "命中目标 ASIN，且所有可评分 Hard 约束通过。",
    "valid_alternative_purchase": "替代商品所有 Hard 通过，且没有可核验 Soft 失败。",
    "partial_alternative_purchase": "替代商品所有 Hard 通过，但至少一个可核验 Soft 偏好未满足。",
    "wrong_purchase": "至少一个可评分 Hard 约束失败，购买被判为错误终局。",
    "assistant_final": "环境尚未结束时输出普通文本，没有完成合法终局工具调用。",
    "guard_rejection": "连续非法工具调用达到 Guard 拒绝终止条件。",
    "repeat_loop": "重复或无进展动作达到循环终止条件。",
    "early_abstain": "在仍有探索价值时调用 finish_without_purchase，属于过早停止。",
}
CONSTRAINT_TYPE_DISPLAY = {
    "category": "品类",
    "price_range": "价格",
    "budget_upper": "价格上限",
    "price_lower": "价格下限",
    "brand": "品牌",
    "model": "型号",
    "core_function": "功能/属性",
    "option": "规格",
}
REWARD_V3_RULES = (
    ("gold_purchase", "1.00", "命中目标 ASIN，且类目、价格固定 Hard Gate 通过"),
    ("valid_alternative_purchase", "0.80", "替代商品通过类目、价格门，且旧版四个匹配维度全部满足"),
    (
        "partial_alternative_purchase",
        "-0.30 + 0.50 × S",
        "通过类目、价格门后，按品牌、型号、核心功能、关键规格四维匹配率 S 连续给分；不是 Hard 失败后的补分",
    ),
    ("graceful_stop", "-0.15", "历史版本中充分检索后的合理停止"),
    ("early_abstain", "-0.35", "过早停止"),
    ("max_steps", "-0.50", "耗尽最大步数"),
    ("repeat_loop", "-0.65", "重复或无进展循环"),
    ("assistant_final", "-0.40（训练过滤）", "记录了终局分，但当时按采样无效/优化过滤处理"),
    ("guard_rejection", "无效", "连续非法动作终止未作为有效负样本参与优化"),
    ("wrong_purchase", "-0.85", "类目或价格固定 Hard Gate 失败"),
    ("reward_unverifiable", "0.00（无效）", "关键证据无法核验，不进入训练更新"),
)
REWARD_V4_RULES = (
    ("gold_purchase", "1.00", "所有可评分 Hard 通过，且命中目标 ASIN"),
    ("valid_alternative_purchase", "0.80", "替代商品 Hard 全通过，且没有可核验 Soft 失败"),
    (
        "partial_alternative_purchase",
        "0.50 + 0.30 × soft_score",
        "替代商品 Hard 全通过，但至少一个可核验 Soft 失败；Soft 不可核验不会误触发 Partial",
    ),
    ("early_abstain", "-0.40", "仍有探索价值时主动停止"),
    ("max_steps", "0.00", "耗尽45步的基础分；随后叠加累计步数惩罚"),
    ("repeat_loop", "-0.60", "重复或无进展循环"),
    ("assistant_final", "-0.80", "未调用合法终局工具而直接输出文字"),
    ("guard_rejection", "-0.80", "连续非法动作达到 Guard 终止条件"),
    ("wrong_purchase", "-1.00", "任一可评分 Hard 失败"),
    ("reward_unverifiable", "0.00（无效）", "Hard 证据无法核验，不进入训练更新"),
)
STEP_PENALTY_TEXT = (
    "第16步起累计扣分：16–20步每步-0.01，21–25步每步-0.02，"
    "26–30步每步-0.03，31–35步每步-0.04，36–40步每步-0.05，"
    "41–45步每步-0.06；45步累计为-1.05。惩罚只改Final Reward，不改变终局类型。"
)

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import render_threeway_evaluation_report as common  # noqa: E402
from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS  # noqa: E402
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


def build_candidate_recovery_summary(
    trajectories_path: Path = GRPO230_V3_TRAJECTORIES,
) -> dict[str, Any]:
    """Summarise trajectories rescued from the six-step no-progress Loop gate."""
    rows = load_jsonl(trajectories_path)
    entered_rows = [row for row in rows if row.get("candidate_recovery_events")]
    outcome_counts = Counter(
        str((row.get("terminal_result") or {}).get("termination_reason") or "unknown")
        for row in entered_rows
    )
    gold = outcome_counts.get("gold_purchase", 0)
    valid = outcome_counts.get("valid_alternative_purchase", 0)
    successful = gold + valid
    remaining_loops = [
        int(row["task_id"])
        for row in rows
        if (row.get("terminal_result") or {}).get("termination_reason") == "repeat_loop"
        and not row.get("candidate_recovery_events")
    ]
    return {
        "triggered": len(entered_rows),
        "entered": len(entered_rows),
        "entry_rate": 1.0 if entered_rows else 0.0,
        "gold": gold,
        "valid": valid,
        "successful": successful,
        "success_rate": successful / len(entered_rows) if entered_rows else 0.0,
        "wrong": outcome_counts.get("wrong_purchase", 0),
        "partial": outcome_counts.get("partial_alternative_purchase", 0),
        "guard_rejection": outcome_counts.get("invalid_action_limit", 0),
        "remaining_loop_tasks": remaining_loops,
    }


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


def compact_text(value: Any, limit: int = 180) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def compact_action(step: dict[str, Any]) -> str:
    tool = str(step.get("tool_name") or "unknown")
    parameters = step.get("parameters") or {}
    if not isinstance(parameters, dict) or not parameters:
        return tool
    if tool == "search_products":
        detail = parameters.get("query")
    elif tool == "open_product":
        detail = parameters.get("asin")
    elif tool == "select_option":
        detail = (
            parameters.get("option")
            or parameters.get("value")
            or parameters.get("option_value")
        )
    else:
        detail = None
    if detail is None:
        detail = json.dumps(parameters, ensure_ascii=False, sort_keys=True)
    return f"{tool}({compact_text(detail, 48)})"


def compact_action_chain(trajectory: dict[str, Any]) -> str:
    actions = [compact_action(step) for step in trajectory.get("steps") or []]
    if not actions:
        return "无已执行工具"
    runs: list[tuple[str, int]] = []
    for action in actions:
        if runs and runs[-1][0] == action:
            runs[-1] = (action, runs[-1][1] + 1)
        else:
            runs.append((action, 1))
    compacted = [f"{action} ×{count}" if count > 1 else action for action, count in runs]
    if len(compacted) > 10:
        compacted = [*compacted[:6], "…", *compacted[-3:]]
    return " → ".join(compacted)


def render_tool_schema() -> str:
    """Render the exact eight-tool public contract without dumping raw JSON."""
    rows = []
    for schema in SHOP_TOOL_SCHEMAS:
        function = schema.get("function") or {}
        parameters = function.get("parameters") or {}
        properties = parameters.get("properties") or {}
        required = set(parameters.get("required") or [])
        parameter_parts = []
        for name, specification in properties.items():
            value_type = specification.get("type") or "any"
            enum_values = specification.get("enum") or []
            enum_text = (
                "；可选值：" + "、".join(str(value) for value in enum_values)
                if enum_values
                else ""
            )
            required_text = "必填" if name in required else "可选"
            parameter_parts.append(
                f"<code>{esc(name)}</code>：{esc(value_type)}（{required_text}{esc(enum_text)}）"
            )
        parameter_text = "；".join(parameter_parts) if parameter_parts else "无参数，必须传 <code>{}</code>"
        rows.append(
            "<tr>"
            f"<th><code>{esc(function.get('name') or 'unknown')}</code></th>"
            f"<td>{esc(function.get('description') or '')}</td>"
            f"<td>{parameter_text}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap horizontal-only trajectory-schema"><table>'
        "<thead><tr><th>工具</th><th>完整用途与调用条件</th><th>参数合同</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def render_message_text(value: Any, empty_text: str) -> str:
    text = "" if value is None else str(value)
    if not text:
        return f'<p class="trajectory-empty">{esc(empty_text)}</p>'
    return f'<pre class="trajectory-text">{esc(text)}</pre>'


def render_tool_call(call: dict[str, Any], call_index: int) -> str:
    function = call.get("function") or {}
    name = str(function.get("name") or "unknown")
    arguments = function.get("arguments")
    try:
        parsed_arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
        arguments_text = json.dumps(parsed_arguments or {}, ensure_ascii=False, indent=2)
    except (TypeError, ValueError, json.JSONDecodeError):
        arguments_text = str(arguments or "{}")
    return (
        '<div class="trajectory-call">'
        f'<div class="trajectory-label">工具调用 {call_index}：<code>{esc(name)}</code></div>'
        f'<pre class="trajectory-args">{esc(arguments_text)}</pre>'
        "</div>"
    )


def render_trajectory_rounds(trajectory: dict[str, Any]) -> tuple[str, int]:
    messages = list(trajectory.get("messages") or [])[2:]
    rounds = []
    round_number = 0
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.get("role") != "assistant":
            index += 1
            continue
        round_number += 1
        response_messages = []
        next_index = index + 1
        while next_index < len(messages) and messages[next_index].get("role") != "assistant":
            response_messages.append(messages[next_index])
            next_index += 1

        reasoning = message.get("reasoning")
        content = message.get("content")
        calls = message.get("tool_calls") or []
        body = [f'<article class="trajectory-round"><h4>第 {round_number} 轮</h4>']
        if reasoning:
            body.append('<div class="trajectory-label">模型 reasoning</div>')
            body.append(render_message_text(reasoning, "本轮没有 reasoning 文本。"))
        body.append('<div class="trajectory-label">模型输出 / 本轮判断</div>')
        body.append(
            render_message_text(
                content,
                "模型没有输出说明文字，直接发起工具调用。" if calls else "模型本轮没有输出文字。",
            )
        )
        if calls:
            body.extend(render_tool_call(call, call_index) for call_index, call in enumerate(calls, 1))
        else:
            body.append('<p class="trajectory-empty">本轮没有发起工具调用。</p>')

        for response in response_messages:
            role = str(response.get("role") or "unknown")
            if role == "tool":
                tool_name = response.get("name") or "unknown"
                body.append(
                    f'<div class="trajectory-label trajectory-response">环境 / Guard 返回：<code>{esc(tool_name)}</code></div>'
                )
                body.append(render_message_text(response.get("content"), "环境返回为空。"))
            elif role == "user":
                body.append('<div class="trajectory-label trajectory-response">Harness 追加纠正或提醒</div>')
                body.append(render_message_text(response.get("content"), "Harness 提醒为空。"))
            else:
                body.append(f'<div class="trajectory-label trajectory-response">{esc(role)} 消息</div>')
                body.append(render_message_text(response.get("content"), "消息为空。"))
        body.append("</article>")
        rounds.append("".join(body))
        index = next_index
    return "".join(rounds), round_number


def render_terminal_summary(trajectory: dict[str, Any], metrics: dict[str, Any]) -> str:
    terminal = trajectory.get("terminal_result") or {}
    purchase = terminal.get("purchase") or {}
    purchase_rows = []
    for label, key in (
        ("ASIN", "asin"),
        ("商品", "name"),
        ("品类", "category"),
        ("价格", "price"),
        ("规格", "options"),
    ):
        value = purchase.get(key)
        if value in (None, "", {}, []):
            continue
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        purchase_rows.append(f"<dt>{esc(label)}</dt><dd>{esc(value)}</dd>")
    purchase_html = "".join(purchase_rows) or "<dt>购买结果</dt><dd>未购买商品</dd>"
    return (
        '<section class="trajectory-terminal"><h4>最终终局</h4><dl>'
        f'<dt>终止原因</dt><dd><code>{esc(terminal.get("termination_reason") or metrics.get("reward_type") or "unknown")}</code></dd>'
        f'<dt>Reward 类型</dt><dd><code>{esc(metrics.get("reward_type") or "unknown")}</code></dd>'
        f'<dt>Final Reward</dt><dd>{num(metrics.get("final_reward"), 4)}</dd>'
        f'<dt>Reward 有效</dt><dd>{"是" if metrics.get("reward_valid") else "否"}</dd>'
        f"{purchase_html}</dl></section>"
    )


def compact_constraint_value(value: Any, limit: int = 80) -> str:
    if isinstance(value, dict):
        source_text = value.get("source_text")
        if source_text:
            return compact_text(source_text, limit)
        value = "、".join(f"{key}={item}" for key, item in value.items())
    elif isinstance(value, list):
        value = "、".join(str(item) for item in value)
    return compact_text(value, limit)


def describe_constraint(row: dict[str, Any]) -> str:
    status = str(row.get("status") or "unverifiable")
    marker = {"pass": "✓", "fail": "✗", "unverifiable": "?"}.get(status, "?")
    constraint_type = str(row.get("constraint_type") or "constraint")
    type_label = CONSTRAINT_TYPE_DISPLAY.get(constraint_type, constraint_type)
    expected = row.get("expected")
    if constraint_type == "category" and isinstance(expected, str):
        expected_text = expected.rsplit("›", 1)[-1]
    else:
        expected_text = compact_constraint_value(expected)
    if not expected_text:
        expected_text = compact_text(row.get("query_quote"), 80) or "未命名要求"
    detail = f"{marker} {type_label}：{expected_text}"
    actual = row.get("actual")
    if constraint_type in {"price_range", "budget_upper", "price_lower"}:
        actual_text = compact_constraint_value(actual, 48)
        if actual_text:
            detail += f"（实际 {actual_text} 元）"
    elif constraint_type == "option" and isinstance(actual, dict):
        actual_text = compact_constraint_value(actual, 64)
        if actual_text:
            detail += f"（已选 {actual_text}）"
    return detail


def build_typical_trajectory_cases(
    evaluations: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    trajectory_sources = {
        "grpo230_v3": GRPO230_V3_TRAJECTORIES,
        "qwen38_27b_v3": QWEN_V3_TRAJECTORIES,
        "qwen38_27b": QWEN_TRAJECTORIES,
        "sft": SFT_TRAJECTORIES,
    }
    priority = ("grpo230_v3", "qwen38_27b_v3", "qwen38_27b", "sft")
    trajectory_indexes = {
        label: {int(row["task_id"]): row for row in load_jsonl(path)}
        for label, path in trajectory_sources.items()
    }
    evaluation_indexes = {
        label: {int(row["task_id"]): row for row in evaluations[label]}
        for label in priority
    }
    replay_indexes = {}
    for label in ("qwen38_27b", "sft"):
        replay_path = REWARD_V4_REPLAY_DIR / f"{label}.json"
        if replay_path.is_file():
            replay_indexes[label] = {
                int(row["task_id"]): row
                for row in load_json(replay_path).get("tasks") or []
            }

    def reward_snapshot(
        label: str, task_id: int, trajectory: dict[str, Any]
    ) -> dict[str, Any]:
        terminal = trajectory.get("terminal_result") or {}
        reward_detail = terminal.get("reward_detail") or {}
        replay = replay_indexes.get(label, {}).get(task_id)
        if replay:
            return {
                "reward_type": replay.get("new_reward_type"),
                "final_reward": replay.get("new_reward"),
                "constraint_rows": replay.get("new_constraint_results") or [],
                "contract": replay.get("new_strict_purchase_contract") or {},
                "reward_detail": reward_detail,
            }
        contract = (
            reward_detail.get("strict_purchase_contract")
            or (reward_detail.get("evidence") or {}).get("strict_purchase_contract")
            or {}
        )
        return {
            "reward_type": (
                REWARD_OVERLAY.get(label, {}).get(task_id, {}).get("reward_type")
                or reward_detail.get("reward_type")
            ),
            "final_reward": (
                REWARD_OVERLAY.get(label, {}).get(task_id, {}).get("final_reward")
                if task_id in REWARD_OVERLAY.get(label, {})
                else reward_detail.get("terminal_utility", reward_detail.get("reward"))
            ),
            "constraint_rows": reward_detail.get("constraint_results")
            or (reward_detail.get("evidence") or {}).get("query_constraint_results")
            or [],
            "contract": contract,
            "reward_detail": reward_detail,
        }

    def has_nonprice_only_soft(snapshot: dict[str, Any]) -> bool:
        soft_rows = [
            row
            for row in snapshot["constraint_rows"]
            if isinstance(row, dict)
            and row.get("enforcement") == "scored"
            and row.get("strength") == "soft"
        ]
        return bool(soft_rows) and all(
            row.get("constraint_type")
            not in {"price_range", "budget_upper", "price_lower"}
            for row in soft_rows
        )

    cases: list[dict[str, Any]] = []
    for reward_type in TYPICAL_REWARD_TYPES:
        selected: tuple[str, int, dict[str, Any], dict[str, Any]] | None = None
        override = TYPICAL_CASE_OVERRIDES.get(reward_type)
        if override:
            override_label, override_task_id = override
            override_trajectory = trajectory_indexes[override_label][override_task_id]
            override_snapshot = reward_snapshot(
                override_label, override_task_id, override_trajectory
            )
            if str(override_snapshot.get("reward_type")) != reward_type:
                raise ValueError(
                    f"typical case override {override_label}/{override_task_id} "
                    f"is {override_snapshot.get('reward_type')}, expected {reward_type}"
                )
            selected = (
                override_label,
                override_task_id,
                override_trajectory,
                override_snapshot,
            )
        prefer_nonprice_soft = reward_type in {
            "gold_purchase",
            "valid_alternative_purchase",
            "partial_alternative_purchase",
            "wrong_purchase",
        }
        for require_nonprice_soft in (() if selected else (
            (True, False) if prefer_nonprice_soft else (False,)
        )):
            for label in priority:
                candidates = []
                for task_id, trajectory in trajectory_indexes[label].items():
                    snapshot = reward_snapshot(label, task_id, trajectory)
                    if (
                        str(snapshot.get("reward_type")) == reward_type
                        and bool(trajectory.get("done"))
                        and not trajectory.get("error")
                        and (
                            not require_nonprice_soft
                            or has_nonprice_only_soft(snapshot)
                        )
                    ):
                        candidates.append((task_id, trajectory, snapshot))
                if not candidates:
                    continue
                step_counts = sorted(
                    len(row.get("steps") or []) for _, row, _ in candidates
                )
                typical_steps = step_counts[len(step_counts) // 2]
                task_id, trajectory, snapshot = min(
                    candidates,
                    key=lambda item: (
                        abs(len(item[1].get("steps") or []) - typical_steps),
                        int(item[0]),
                    ),
                )
                selected = (label, task_id, trajectory, snapshot)
                break
            if selected is not None:
                break
        if selected is None:
            continue
        label, task_id, trajectory, snapshot = selected
        evaluation = evaluation_indexes[label][task_id]
        stage = stage_fields(evaluation, label=label, task_id=task_id)
        terminal = trajectory.get("terminal_result") or {}
        reward_detail = snapshot["reward_detail"]
        contract = snapshot["contract"]
        purchase = terminal.get("purchase") or {}
        options = purchase.get("options") or {}
        if purchase:
            option_text = "、".join(f"{key}={value}" for key, value in options.items())
            purchase_text = compact_text(
                f"{purchase.get('name') or purchase.get('asin')}；价格 {purchase.get('price')}"
                + (f"；{option_text}" if option_text else ""),
                220,
            )
        else:
            purchase_text = "未购买"
        contract_parts = []
        if contract:
            contract_parts.append(
                f"Hard {contract.get('hard_passed', 0)}/{contract.get('hard_total', 0)}"
            )
            contract_parts.append(
                f"Soft {contract.get('soft_passed', 0)}/{contract.get('soft_total', 0)}"
            )
        constraint_rows = snapshot["constraint_rows"]
        hard_details = [
            describe_constraint(row)
            for row in constraint_rows
            if isinstance(row, dict)
            and row.get("enforcement") == "scored"
            and row.get("strength") == "hard"
        ]
        soft_details = [
            describe_constraint(row)
            for row in constraint_rows
            if isinstance(row, dict)
            and row.get("enforcement") == "scored"
            and row.get("strength") == "soft"
        ]
        cases.append(
            {
                "reward_type": reward_type,
                "description": TYPICAL_REWARD_DESCRIPTIONS[reward_type],
                "label": label,
                "model": DISPLAY[label],
                "task_id": task_id,
                "instruction": compact_text(
                    (trajectory.get("initial_result") or {}).get("instruction"), 260
                ),
                "steps": len(trajectory.get("steps") or []),
                "actions": compact_action_chain(trajectory),
                "purchase": purchase_text,
                "contract": "；".join(contract_parts) if contract_parts else "无购买约束摘要",
                "hard_details": (
                    "；".join(hard_details)
                    if hard_details
                    else ("无可评分Hard约束" if purchase else "未购买，未进入购买约束评分")
                ),
                "soft_details": (
                    "；".join(soft_details)
                    if soft_details
                    else ("无可评分Soft偏好" if purchase else "未购买，未进入购买约束评分")
                ),
                "final_reward": float(snapshot["final_reward"]),
                "judge_status": str(stage.get("judge_status") or ""),
                "primary_error": str(stage.get("primary_error") or ""),
            }
        )
    return cases


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


def build_success_stratification(
    task_slices: dict[int, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Aggregate Gold+Valid success by frozen retrieval and difficulty buckets."""

    result: dict[str, dict[str, Any]] = {}
    for field, spec in SUCCESS_STRATIFICATION_SPECS.items():
        buckets: dict[str, Any] = {}
        observed = {
            str(row.get(field))
            for row in task_slices.values()
            if row.get(field) is not None
        }
        ordered_buckets = [bucket for bucket in spec["order"] if bucket in observed]
        ordered_buckets.extend(sorted(observed - set(ordered_buckets)))
        for bucket in ordered_buckets:
            task_ids = sorted(
                task_id
                for task_id, row in task_slices.items()
                if str(row.get(field)) == bucket
            )
            models = {}
            for label in LABELS:
                success_count = sum(
                    bool(REWARD_OVERLAY[label][task_id]["purchase_success"])
                    for task_id in task_ids
                )
                models[label] = {
                    "successes": success_count,
                    "success_rate": success_count / len(task_ids),
                }
            buckets[bucket] = {
                "display": spec["labels"].get(bucket, bucket),
                "tasks": len(task_ids),
                "models": models,
            }
        result[field] = {
            "title": spec["title"],
            "description": spec["description"],
            "buckets": buckets,
        }
    return result


def _compact_teacher_tool_call(call: dict[str, Any]) -> str:
    function = call.get("function") or {}
    name = str(function.get("name") or "unknown")
    raw_arguments = function.get("arguments") or "{}"
    try:
        arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
    except json.JSONDecodeError:
        arguments = {}
    if name == "search_products" and arguments.get("query"):
        return f'{name}(“{arguments["query"]}”)'
    if name == "open_product" and arguments.get("asin"):
        return f'{name}({arguments["asin"]})'
    if name == "select_option" and arguments.get("value"):
        return f'{name}(“{arguments["value"]}”)'
    return name


def _teacher_product_detail(content: str) -> dict[str, str]:
    fields = {}
    for line in str(content or "").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key in {
            "asin",
            "title",
            "brand",
            "category",
            "price",
            "key_attributes",
            "selected_options",
        }:
            fields[key] = value.strip()
    return fields


def build_teacher_corrective_cases() -> list[dict[str, Any]]:
    wanted = {int(spec["task_id"]): spec for spec in TEACHER_CORRECTIVE_CASE_SPECS}
    rows = {
        int(row["task_id"]): row
        for row in load_jsonl(SFT_FINAL1000)
        if int(row["task_id"]) in wanted
    }
    if set(rows) != set(wanted):
        raise ValueError(
            f"missing Teacher showcase tasks: {sorted(set(wanted) - set(rows))}"
        )

    cases = []
    for spec in TEACHER_CORRECTIVE_CASE_SPECS:
        task_id = int(spec["task_id"])
        row = rows[task_id]
        user_request = ""
        actions = []
        final_detail: dict[str, str] = {}
        for message in row.get("messages") or []:
            role = message.get("role")
            if role == "user" and not user_request:
                user_request = str(message.get("content") or "")
                if user_request.startswith("Instruction:"):
                    user_request = user_request.removeprefix("Instruction:").strip()
            if role == "assistant":
                actions.extend(
                    _compact_teacher_tool_call(call)
                    for call in (message.get("tool_calls") or [])
                )
            if role == "tool" and "page_type: product_detail" in str(
                message.get("content") or ""
            ):
                detail = _teacher_product_detail(str(message.get("content") or ""))
                if detail:
                    final_detail = detail
        if not actions or actions[-1] != "buy_now":
            raise ValueError(f"Teacher showcase task {task_id} does not end with buy_now")
        required_detail = {"asin", "title", "price", "selected_options"}
        if not required_detail.issubset(final_detail):
            raise ValueError(f"Teacher showcase task {task_id} has incomplete final detail")
        cases.append(
            {
                **spec,
                "trajectory_id": str(row.get("trajectory_id") or ""),
                "instruction": user_request,
                "actions": actions,
                "steps": len(actions),
                "action_chain": " → ".join(actions),
                "final_detail": final_detail,
            }
        )
    return cases


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
    normal_sft_destination = OUTPUT / "runs/sft_normal1000"
    if normal_sft_destination.exists():
        shutil.rmtree(normal_sft_destination)
    shutil.copytree(
        NORMAL_SFT_SOURCE / "runs/sft_normal1000",
        normal_sft_destination,
    )
    shutil.copy2(
        NORMAL_SFT_SOURCE / "judges-sft_normal1000.jsonl",
        OUTPUT / "judges-sft_normal1000.jsonl",
    )
    shutil.copy2(
        NORMAL_SFT_SOURCE / "calls/judges-sft_normal1000.jsonl",
        OUTPUT / "calls/judges-sft_normal1000.jsonl",
    )
    shutil.copy2(
        NORMAL_SFT_SOURCE / "checkpoints/judges-sft_normal1000.jsonl",
        OUTPUT / "checkpoints/judges-sft_normal1000.jsonl",
    )
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
    typical_cases: list[dict[str, Any]],
) -> None:
    candidate_recovery = build_candidate_recovery_summary()
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

    reward_v3_rows = [list(row) for row in REWARD_V3_RULES]
    reward_v4_rows = [list(row) for row in REWARD_V4_RULES]
    typical_rows = [
        [
            case["reward_type"],
            case["model"],
            case["task_id"],
            case["steps"],
            num(case["final_reward"], 4),
            case["actions"],
        ]
        for case in typical_cases
    ]

    lines = [
        "# Final-240 八组统一评测：Harness v1 / v2 / v3",
        "",
        "## 评测合同",
        "",
        "- 八组结果使用同一240题 Final-240、同一冻结 DeepSeek V4 Flash Rubric、同一 DeepSeek V4 Pro Judge Prompt与Schema。",
        "- 普通 SFT v1 使用1000条普通Teacher数据；纠错 SFT v1 使用600条普通数据与400条纠错数据。两者均使用Harness v1。",
        "- Base、普通 SFT、纠错 SFT、GRPO100使用Harness v1；原GRPO230与Qwen3.8-27B使用Harness v2；新增GRPO230与Qwen3.8-27B使用Harness v3。各组均按同一盲评合同统计。",
        "- 成功率、Reward值和Reward类型按当前Reward v4聚合重算；Rubric、Judge与确定性过程指标保持冻结口径。",
        "- Judge只看Query、Rubric、Actor可见轨迹和白名单行为指标，不看Reward与Gold私有字段。",
        "",
        "## Harness版本演进",
        "",
        "- **v1 → v2：** 增加35步收敛提醒、模型输出文字但未调用工具时的拒绝纠正，以及循环/无进展提醒；删除 `view_description`、`view_features`、`view_reviews`、`view_attributes` 四种低频信息工具，并将原信息子页中的非空内容直接合并到商品详情，使模型调用 `open_product` 后即可在最新 Observation 中一次获得完整商品信息。",
        "- **v2 → v3：** 将搜索页、详情页和普通页面的Observation预算由 `1536 / 4096 / 768` 调整为 `2560 / 3072 / 512` Token，并为候选选择界面单独设置 `1024` Token预算；增加已核验候选记忆与页面/阶段级动态 Tool Schema，只向模型暴露当前状态真正可执行的工具。普通搜索首页开放搜索与放弃，搜索结果页开放商品打开、可见翻页/返回与放弃，商品详情页开放未选规格、返回、购买与放弃；进入候选收敛后进一步收紧为：候选选择阶段仅开放 `open_product`，规格阶段仅开放 `select_option`，终局阶段仅开放 `buy_now` 与 `finish_without_purchase`。当循环/无进展达到终止条件时，不再直接结束，而是强制进入候选记忆模块完成最终决策。",
        "- 本报告保留既有Harness v1/v2结果，并将GRPO230·Harness v3 r4与Qwen3.8-27B·Harness v3 r1作为独立新组接入，不覆盖历史结果。",
        "",
        "### GRPO230 v3候选强制阶段转化",
        "",
        f"- `{candidate_recovery['triggered']}`题达到连续6步无实质进展的原Loop阈值，全部进入候选强制阶段（`{pct(candidate_recovery['entry_rate'], 2)}`）。",
        f"- 其中`{candidate_recovery['successful']}/{candidate_recovery['entered']}`转化为正确购买，成功转化率为`{pct(candidate_recovery['success_rate'], 2)}`：Gold `{candidate_recovery['gold']}`题、Valid `{candidate_recovery['valid']}`题。",
        f"- 其余终局为Wrong `{candidate_recovery['wrong']}`题、Partial `{candidate_recovery['partial']}`题、Guard rejection `{candidate_recovery['guard_rejection']}`题；这{candidate_recovery['entered']}题中没有最终Repeat Loop。整轮唯一Repeat Loop为Task {', '.join(map(str, candidate_recovery['remaining_loop_tasks']))}，它未进入候选强制阶段。",
        "",
        "## Reward版本演进",
        "",
        "- **Legacy baseline（口头可简称Reward v1）：** 仓库没有正式命名的 `reward_v1.py`；这里指ShopSimulator原生Reward及早期训练适配口径。原生Loose Additive Reward为 `r_type × (匹配属性数 + 匹配规格数 + 价格是否满足) / (属性总数 + 规格总数 + 1)`；适配层同时计算 `strict = r_type × r_attribute × r_option × r_price`，再组合为 `semantic = full_success + 0.5 × strict + 0.2 × native_reward`，并附加少量步数、过长、未完成与重复动作惩罚。该口径保留了类目、属性、规格、价格四维信号，但Loose可能出现维度补偿，Strict又会在任一维度失败时乘法坍缩，且对唯一Gold、错误购买、循环、超时和主动停止的终局区分不足。",
        "- **v3 → v4：** 将v3中类目与价格等固定 Hard Gate、活跃维度匹配分数，升级为基于用户公开 Query 的可审计 Hard/Soft 约束合同。品类始终为 Hard；“必须、一定、绝对不要、不超过、至少、明确区间”等高置信且可确定性核验的不可妥协要求也进入 Hard；“最好、优先、尽量、大约、左右、预算”等偏好或近似表达进入 Soft；无所谓类表达忽略，复杂歧义语义进入 Needs Review / audit-only，不强行参与评分。任一可评分 Hard 失败即判 `wrong_purchase`；Hard 全通过后，目标商品为 Gold，完全满足 Soft 的替代商品为 Valid，只违反 Soft 的替代商品为 Partial。",
        "- v4 新增第16步起的分段递增步数惩罚；将 `assistant_final` 与连续 Guard 拒绝由无效样本改为 `-0.8` 的有效负样本；并重新校准部分终局分数，其中 Partial 调整为 `0.5 + 0.3 × soft_score`，Loop 调整为 `-0.6`。",
        "- **训练版本：** GRPO100使用Reward v3，GRPO230使用Reward v4。为保证横向可比，本报告中的成功率、Reward值与Reward类型仍统一按当前审计版Reward v4对冻结轨迹离线重放，不反向更新模型参数。",
        "",
        "### Reward v3.2历史评分",
        "",
        md_table(["终局", "基础分", "判定"], reward_v3_rows),
        "",
        "v3没有累计步数惩罚。`assistant_final`虽记录为-0.40，但在当时训练筛选中被过滤；Guard拒绝同样不作为有效负样本优化。",
        "",
        "### Reward v4当前评分",
        "",
        md_table(["终局", "基础分", "判定"], reward_v4_rows),
        "",
        STEP_PENALTY_TEXT,
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
        "## 八类典型轨迹",
        "",
        "优先从GRPO230 v3和Qwen3.8-27B v3选择；缺失类型回退到纠错 SFT v1。",
        "",
        md_table(["Reward类型", "模型", "Task", "步骤", "Final Reward", "动作链"], typical_rows),
        "",
        "## 文件",
        "",
        "- `dashboard.html`：八组聚合前端报告。",
        "- `per-task-comparison.csv/json`：八组模型逐题审计。",
        "- `comparison.json`：八组两两配对及分层比较；页面展示三段关键迁移。",
        "- GitHub 公开副本仅包含聚合报告与逐题派生比较；原始轨迹、Judge JSONL、Rubric、API calls、日志和 checkpoint 均不发布。",
    ]
    (OUTPUT / "audit-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_dashboard(
    summaries: dict[str, dict[str, Any]],
    comparison: dict[str, Any],
    evaluations: dict[str, list[dict[str, Any]]],
    per_task: list[dict[str, Any]],
    typical_cases: list[dict[str, Any]],
) -> None:
    expected = 240
    candidate_recovery = build_candidate_recovery_summary()
    teacher_corrective_cases = build_teacher_corrective_cases()
    four_round_metrics = load_json(FOUR_ROUND_METRICS)
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

    def reward_rule_rows(rules: tuple[tuple[str, str, str], ...]) -> str:
        return "".join(
            f"<tr><th><code>{esc(reward_type)}</code></th><td>{esc(score)}</td><td>{esc(rule)}</td></tr>"
            for reward_type, score, rule in rules
        )

    reward_v3_rule_rows = reward_rule_rows(REWARD_V3_RULES)
    reward_v4_rule_rows = reward_rule_rows(REWARD_V4_RULES)

    typical_cards = []
    for case in typical_cases:
        typical_cards.append(
            '<details class="card typical-card">'
            f'<summary><code>{esc(case["reward_type"])}</code><span>{esc(case["model"])} · Task {case["task_id"]}</span></summary>'
            f'<p class="lead">{esc(case["description"])}</p>'
            f'<dl><dt>Query</dt><dd>{esc(case["instruction"])}</dd>'
            f'<dt>动作链（{case["steps"]}步）</dt><dd>{esc(case["actions"])}</dd>'
            f'<dt>终局商品</dt><dd>{esc(case["purchase"])}</dd>'
            f'<dt>约束计数</dt><dd>{esc(case["contract"])}</dd>'
            f'<dt>Hard约束</dt><dd>{esc(case["hard_details"])}</dd>'
            f'<dt>Soft偏好</dt><dd>{esc(case["soft_details"])}</dd>'
            f'<dt>审计结果</dt><dd>Final Reward {num(case["final_reward"], 4)}；Judge {esc(case["judge_status"] or "—")}；Primary Error {esc(case["primary_error"] or "—")}</dd></dl>'
            '</details>'
        )

    raw_trajectory_index = {
        int(row["task_id"]): row for row in load_jsonl(GRPO230_V3_TRAJECTORIES)
    }
    grpo230_v3_evaluation_index = {
        int(row["task_id"]): row for row in evaluations["grpo230_v3"]
    }
    full_trajectory_cards = []
    full_tool_schema = render_tool_schema()
    for outcome_label, task_id, accent in (
        ("正确轨迹", 122, "#15803d"),
        ("错误轨迹", 702, "#dc2626"),
        ("强制候选收敛后选对", 1332, "#7c3aed"),
    ):
        raw_trajectory = raw_trajectory_index[task_id]
        evaluation = grpo230_v3_evaluation_index[task_id]
        metrics = evaluation["reward_and_terminal"]["metrics"]
        messages = list(raw_trajectory.get("messages") or [])
        system_prompt = messages[0].get("content") if messages else ""
        user_request = messages[1].get("content") if len(messages) > 1 else ""
        rounds_html, model_rounds = render_trajectory_rounds(raw_trajectory)
        executed_steps = len(raw_trajectory.get("steps") or [])
        full_trajectory_cards.append(
            '<details class="card typical-card full-trajectory">'
            f'<summary style="border-left:6px solid {accent}"><strong>{outcome_label} · GRPO230 v3 · Task {task_id}</strong>'
            f'<span><code>{esc(metrics["reward_type"])}</code> · {model_rounds}轮 / {executed_steps}个环境动作 · Final Reward {num(metrics["final_reward"], 4)}</span></summary>'
            '<div class="trajectory-content">'
            '<p class="lead">按模型实际接收和产生的消息顺序展示完整交互流程；Observation、Guard返回和Harness追加提醒均保留全文。仅省去Token、耗时、哈希及内部调试字段。</p>'
            '<details class="trajectory-meta" open><summary>System Prompt（完整）</summary>'
            f'{render_message_text(system_prompt, "System Prompt 为空。")}</details>'
            '<details class="trajectory-meta"><summary>基础 Tool Schema（完整8工具合同）</summary>'
            '<p class="lead">这是本轮评测使用的基础工具合同；每个模型回合会根据最新页面与候选收敛阶段动态暴露其中可执行的子集。</p>'
            f'{full_tool_schema}</details>'
            '<details class="trajectory-meta" open><summary>用户需求（完整）</summary>'
            f'{render_message_text(user_request, "用户需求为空。")}</details>'
            '<h3 class="trajectory-section-title">逐轮完整流程</h3>'
            f'{rounds_html}'
            f'{render_terminal_summary(raw_trajectory, metrics)}'
            '</div>'
            '</details>'
        )

    teacher_corrective_cards = []
    for case in teacher_corrective_cases:
        detail = case["final_detail"]
        teacher_corrective_cards.append(
            '<details class="card typical-card teacher-case">'
            f'<summary style="border-left:6px solid {case["accent"]}">'
            f'<strong>{esc(case["strategy_display"])} · Task {case["task_id"]}</strong>'
            f'<span>{case["steps"]}步 · <code>gold_purchase</code></span></summary>'
            '<div class="trajectory-content">'
            f'<h3>{esc(case["headline"])}</h3>'
            f'<p class="lead">{esc(case["instruction"])}</p>'
            '<dl>'
            f'<dt>纠错点</dt><dd>{esc(case["correction"])}</dd>'
            f'<dt>正确恢复</dt><dd>{esc(case["learning"])}</dd>'
            f'<dt>完整动作链</dt><dd>{esc(case["action_chain"])}</dd>'
            f'<dt>终局商品</dt><dd>{esc(detail["title"])}；ASIN {esc(detail["asin"])}；'
            f'价格 {esc(detail["price"])}；规格 {esc(detail["selected_options"])}</dd>'
            '<dt>质量门</dt><dd>Gold商品；完整Variant成立；显式 <code>buy_now</code>；'
            '关键动作可重放且Observation-grounded。</dd>'
            '</dl></div></details>'
        )

    th_models = "".join(f"<th>{DISPLAY[label]}</th>" for label in LABELS)
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
            delta = (rate_by_label["grpo230_v3"] - rate_by_label["sft"]) * 100
            stratified_rows.append(
                f"<tr><td>{esc(group_name)}</td><td>{esc(bucket)}</td><td>{task_count}</td>"
                + "".join(f"<td>{pct(rate)}</td>" for rate in rates)
                + f"<td>{delta:+.1f} pp</td></tr>"
            )

    success_stratification_cards = []
    for group in comparison["success_stratification"].values():
        rows = []
        for bucket in group["buckets"].values():
            sft_rate = bucket["models"]["sft"]["success_rate"]
            grpo230_v3_rate = bucket["models"]["grpo230_v3"]["success_rate"]
            cells = []
            for label in LABELS:
                model = bucket["models"][label]
                cells.append(
                    f'<td><strong>{pct(model["success_rate"])}</strong>'
                    f'<br><small>{model["successes"]} / {bucket["tasks"]}</small></td>'
                )
            rows.append(
                f'<tr><th>{esc(bucket["display"])}</th><td>{bucket["tasks"]}</td>'
                + "".join(cells)
                + f'<td>{(grpo230_v3_rate - sft_rate) * 100:+.1f} pp</td></tr>'
            )
        success_stratification_cards.append(
            '<article class="card section-card stratified-card">'
            f'<h3>{esc(group["title"])}</h3><p class="lead">{esc(group["description"])}</p>'
            '<div class="table-wrap horizontal-only"><table><thead><tr><th>分桶</th><th>任务数</th>'
            f'{th_models}<th>SFT v1→GRPO230 v3</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
            '</article>'
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
        transition_value = row["grpo230_to_grpo230_v3_purchase_transition"]
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
    legend = "".join(
        f'<span><i class="dot" style="background:{COLORS[label]}"></i>{DISPLAY[label]}</span>' for label in LABELS
    )
    kpis = "".join(
        f'<article class="card kpi" style="border-top-color:{COLORS[label]}"><div class="label">{DISPLAY[label]} 严格成功率</div><div class="value">{pct(strict_rates[label])}</div><div class="context">{summaries[label]["reward_and_terminal"]["strict_gold_successes"]} / {expected} Gold</div></article>'
        for label in LABELS
    )
    four_round_rows = []
    for metric_key, display_name, color in (
        ("GRPO230-v3", "GRPO230 v3", COLORS["grpo230_v3"]),
        ("Qwen38-27B-v3", "Qwen3.8-27B v3", COLORS["qwen38_27b_v3"]),
    ):
        metric = four_round_metrics[metric_key]
        four_round_rows.append(
            f'<tr><th><i class="dot" style="background:{color}"></i>{display_name}</th>'
            f'<td>{" / ".join(str(value) for value in metric["per_round_successes"])}</td>'
            f'<td><strong>{pct(metric["pass@4"], 2)}</strong><br><small>{metric["pass@4_count"]} / {expected}</small></td>'
            f'<td><strong>{pct(metric["pass^4"], 2)}</strong><br><small>{metric["pass^4_count"]} / {expected}</small></td></tr>'
        )

    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Final-240 八组完整评测</title><style>
:root{{font-family:Inter,"Microsoft YaHei",system-ui,sans-serif;background:#f4f7fb;color:#13213b}}*{{box-sizing:border-box}}body{{margin:0}}.wrap{{max-width:1680px;margin:auto;padding:0 28px 64px}}.hero{{background:linear-gradient(135deg,#0c1936,#4a2674);color:white;padding:44px 48px;border-radius:0 0 32px 32px;box-shadow:0 24px 60px #111c4733}}.hero h1{{font-size:42px;margin:14px 0}}.hero p{{max-width:1200px;color:#dbe3ff;font-size:18px;line-height:1.6}}section{{margin-top:38px}}h2{{font-size:27px;margin-bottom:8px}}h3{{font-size:17px}}.lead{{color:#60718e;line-height:1.6}}.grid{{display:grid;gap:18px}}.kpis{{grid-template-columns:repeat(6,minmax(0,1fr))}}.two{{grid-template-columns:repeat(2,minmax(0,1fr))}}.three{{grid-template-columns:repeat(3,minmax(0,1fr))}}.card{{background:white;border:1px solid #e1e8f2;border-radius:20px;padding:24px;box-shadow:0 9px 28px #1225460f;min-width:0}}.section-card{{margin-top:18px}}.kpi{{border-top:4px solid}}.label,.context{{color:#687a97;font-size:14px}}.value{{font-size:32px;font-weight:800;margin:9px 0}}.bar-row{{display:grid;grid-template-columns:145px 1fr 65px;gap:12px;align-items:center;margin:11px 0}}.bar-track{{height:13px;background:#edf1f7;border-radius:99px;overflow:hidden}}.bar-fill{{height:100%;border-radius:99px}}.bar-row strong{{text-align:right}}.dimensions{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:11px 9px;border-bottom:1px solid #edf1f6;text-align:right;vertical-align:top}}th:first-child,td:first-child{{text-align:left}}thead th{{color:#61718c;position:sticky;top:0;background:white;z-index:1}}.status-stack{{min-width:150px}}.status-stack span{{display:flex;justify-content:space-between;gap:12px;line-height:1.7;color:#60718e}}.unused{{color:#94a3b8;font-style:italic}}.table-wrap{{overflow:auto;max-height:720px}}.horizontal-only{{max-height:none;overflow-x:auto;overflow-y:visible}}.task-table{{min-width:3000px}}.task-table tbody tr[hidden]{{display:none}}.transition{{line-height:1.7}}.transition strong{{font-size:20px}}.filters{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px}}.filters input,.filters select{{font:inherit;border:1px solid #cfd9e8;border-radius:10px;padding:10px 12px;background:white}}.filters input{{min-width:280px;flex:1}}.chip{{border:1px solid #ffffff38;border-radius:999px;padding:7px 12px;background:#ffffff14;display:inline-block;margin:4px}}.legend{{display:flex;gap:18px;flex-wrap:wrap;color:#dbe3ff}}.dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}}.note{{border-left:4px solid #2563eb;background:#eef5ff;padding:16px 18px;border-radius:9px;color:#36506f}}.reward-evolution-grid{{align-items:stretch;margin-top:18px}}.reward-evolution-card{{display:flex;flex-direction:column;gap:14px;border-top:4px solid #64748b}}.reward-evolution-card.v4{{border-top-color:#7c3aed}}.version-tag{{align-self:flex-start;margin:0;padding:6px 10px;border-radius:999px;background:#eef2ff;color:#4f46e5;font-size:12px;font-weight:800;letter-spacing:.04em}}.reward-evolution-card h3{{font-size:20px;margin:0}}.reward-evolution-card .lead{{margin:0}}.reward-formula{{background:#f6f8fc;border:1px solid #e4eaf2;border-radius:12px;padding:12px 14px}}.reward-formula span{{display:block;color:#687a97;font-size:12px;font-weight:800;margin-bottom:6px}}.reward-formula code{{white-space:normal;overflow-wrap:anywhere;line-height:1.55;color:#243b63}}.reward-points{{display:grid;gap:9px;margin-top:auto}}.reward-point{{border-left:3px solid #c7d2fe;padding-left:10px;color:#51627d;font-size:14px;line-height:1.5}}.reward-point strong{{color:#243b63}}.reward-rules td:last-child{{text-align:left;min-width:320px}}.flow-figure{{margin:18px 0 0}}.flow-figure img{{display:block;width:100%;height:auto;border:1px solid #dbe4f0;border-radius:14px;background:white}}.flow-figure figcaption{{margin-top:10px;color:#687a97;font-size:14px;line-height:1.55}}.typical-card{{padding:0;overflow:hidden}}.typical-card summary{{cursor:pointer;padding:20px 24px;display:flex;justify-content:space-between;gap:18px;font-weight:700}}.typical-card summary span{{color:#687a97;font-weight:500}}.typical-card>p,.typical-card dl{{margin-left:24px;margin-right:24px}}.typical-card dl{{display:grid;grid-template-columns:110px 1fr;gap:10px 14px;padding-bottom:20px}}.typical-card dt{{font-weight:700;color:#435675}}.typical-card dd{{margin:0;line-height:1.55;overflow-wrap:anywhere}}.trajectory-content{{padding:0 24px 26px}}.trajectory-meta{{border:1px solid #dbe4f0;border-radius:14px;margin:16px 0;background:#f8fafc;overflow:hidden}}.trajectory-meta>summary{{padding:14px 16px;background:#eef3f9;display:block}}.trajectory-meta>.trajectory-text,.trajectory-meta>.trajectory-empty,.trajectory-meta>.lead,.trajectory-meta>.trajectory-schema{{margin:14px 16px}}.trajectory-schema td:nth-child(2),.trajectory-schema td:nth-child(3){{text-align:left;line-height:1.55;min-width:320px}}.trajectory-section-title{{margin:28px 0 14px;font-size:20px}}.trajectory-round{{border:1px solid #dbe4f0;border-left:5px solid #2563eb;border-radius:14px;padding:18px;margin:14px 0;background:#fff}}.trajectory-round h4,.trajectory-terminal h4{{margin:0 0 14px;font-size:18px}}.trajectory-label{{font-weight:700;color:#435675;margin:14px 0 7px}}.trajectory-response{{color:#0f766e}}.trajectory-text,.trajectory-args{{white-space:pre-wrap;overflow-wrap:anywhere;margin:0;background:#f7f9fc;border:1px solid #e4eaf2;border-radius:10px;padding:13px 14px;font:13px/1.62 Consolas,Monaco,"Microsoft YaHei",monospace;color:#213451}}.trajectory-args{{background:#101827;color:#dbeafe;margin-top:7px}}.trajectory-empty{{color:#7b8ba3;font-style:italic;margin:8px 0}}.trajectory-call{{margin-top:12px}}.trajectory-terminal{{margin-top:22px;border:2px solid #cbd5e1;border-radius:14px;padding:18px;background:#f8fafc}}.trajectory-terminal dl{{margin:0;display:grid;grid-template-columns:110px 1fr;gap:9px 14px;padding:0}}a{{color:#2563eb}}@media(max-width:1100px){{.kpis,.two,.three{{grid-template-columns:1fr 1fr}}.dimensions{{grid-template-columns:1fr}}}}@media(max-width:700px){{.wrap{{padding:0 14px 40px}}.hero{{padding:32px 22px}}.hero h1{{font-size:30px}}.kpis,.two,.three{{grid-template-columns:1fr}}.typical-card dl,.trajectory-terminal dl{{grid-template-columns:1fr}}}}
</style></head><body><div class="wrap"><header class="hero"><div>SHOPBENCH-LH · FINAL-240 · REWARD v4 · DEEPSEEK V4 PRO JUDGE</div><h1>Base v1 / 普通 SFT v1 / 纠错 SFT v1 / GRPO100 v1 / GRPO230 v2 / Qwen3.8-27B v2 / GRPO230 v3 / Qwen3.8-27B v3</h1><p>八组使用同一240题、同一冻结DeepSeek V4 Flash Rubric和同一盲评合同。普通SFT使用1000条普通Teacher数据，纠错SFT使用600条普通数据与400条纠错数据；Harness v3分别展示GRPO230 r4与Qwen3.8-27B r1，均由官网DeepSeek V4 Pro完成盲评。</p><div>{legend}</div></header>
<section><h2>总览</h2><div class="grid kpis">{kpis}</div></section>
<section><h2>Harness版本演进</h2><div class="grid two"><article class="card"><h3>v1 → v2</h3><p class="lead">增加35步收敛提醒、模型输出文字但未调用工具时的拒绝纠正，以及循环/无进展提醒；删除 <code>view_description</code>、<code>view_features</code>、<code>view_reviews</code>、<code>view_attributes</code> 四种低频信息工具，并将原信息子页中的非空内容直接合并到商品详情，使模型调用 <code>open_product</code> 后即可在最新 Observation 中一次获得完整商品信息。</p></article><article class="card"><h3>v2 → v3</h3><p class="lead">将搜索页、详情页和普通页面的Observation预算由 <code>1536 / 4096 / 768</code> 调整为 <code>2560 / 3072 / 512</code> Token，并为候选选择界面单独设置 <code>1024</code> Token预算；增加已核验候选记忆与页面/阶段级动态 Tool Schema，只暴露当前状态真正可执行的工具。当循环/无进展达到终止条件时，不再直接结束，而是强制进入候选记忆模块完成最终决策。</p><table><tbody><tr><th>普通搜索首页</th><td>搜索、放弃</td></tr><tr><th>搜索结果页</th><td>打开商品、当前可见翻页/返回、放弃</td></tr><tr><th>商品详情页</th><td>选择未选规格、返回、购买、放弃</td></tr><tr><th>候选选择阶段</th><td>仅 <code>open_product</code></td></tr><tr><th>候选规格阶段</th><td>仅 <code>select_option</code></td></tr><tr><th>候选终局阶段</th><td>仅 <code>buy_now</code>、<code>finish_without_purchase</code></td></tr></tbody></table></article></div><p class="note">GRPO230与Qwen3.8-27B的Harness v3结果作为独立新组展示，历史Harness v1/v2结果保持不变。</p><article class="card section-card"><h3>Final-240 不同阶段的动态工具暴露</h3><figure class="flow-figure"><img src="assets/final240-dynamic-tool-exposure.png" alt="Final-240普通页面与候选强制收敛阶段的动态工具暴露图"><figcaption>基础8工具合同保持不变；每轮实际 Tool Schema 由页面 footer、当前可见按钮、未选规格和候选收敛阶段共同裁剪，Action Guard 再校验参数是否合法。</figcaption></figure></article></section>
<section><h2>Reward版本演进</h2><div class="grid two"><article class="card"><h3>v3 → v4：Hard / Soft语义合同</h3><p class="lead">从v3的“类目与价格固定 Hard Gate + 四维匹配率”，升级为基于用户公开 Query 的可审计 Hard/Soft 约束合同。品类始终为 Hard；“必须、一定、绝对不要、不超过、至少、明确区间”等高置信且可确定性核验的不可妥协要求也进入 Hard；“最好、优先、尽量、大约、左右、预算”等偏好或近似表达进入 Soft。无所谓类表达忽略，复杂歧义语义进入 Needs Review / audit-only，不强行参与评分。</p><p class="lead">任一可评分 Hard 失败即判 <code>wrong_purchase</code>；Hard 全通过后，目标商品为 Gold，完全满足 Soft 的替代商品为 Valid，只违反 Soft 的替代商品为 Partial。</p></article><article class="card"><h3>效率与终局分数调整</h3><p class="lead">新增第16步起的分段递增步数惩罚；将 <code>assistant_final</code> 与连续 Guard 拒绝由训练过滤/无效改为 <code>-0.8</code> 的有效负样本；并重新校准部分终局分数，其中 Partial 为 <code>0.5 + 0.3 × soft_score</code>，Loop 为 <code>-0.6</code>。</p><table><thead><tr><th>Checkpoint</th><th>训练 Reward</th></tr></thead><tbody><tr><td>GRPO100</td><td>Reward v3</td></tr><tr><td>GRPO230</td><td>Reward v4</td></tr></tbody></table></article></div><div class="grid two section-card"><article class="card reward-rules"><h3>Reward v3.2历史评分细则</h3><div class="table-wrap horizontal-only"><table><thead><tr><th>终局</th><th>基础分</th><th>判定</th></tr></thead><tbody>{reward_v3_rule_rows}</tbody></table></div><p class="lead">v3没有累计步数惩罚。Partial中的S是品牌、型号、核心功能、关键规格四维匹配率，不表示Hard条件失败。</p></article><article class="card reward-rules"><h3>Reward v4当前评分细则</h3><div class="table-wrap horizontal-only"><table><thead><tr><th>终局</th><th>基础分</th><th>判定</th></tr></thead><tbody>{reward_v4_rule_rows}</tbody></table></div><p class="lead">{esc(STEP_PENALTY_TEXT)}</p></article></div><p class="note">训练版本与评测口径分开记录：Final-240中的成功率、Reward值和Reward类型统一按当前审计版Reward v4对冻结轨迹离线重放，便于横向比较，不会反向更新模型参数。</p><article class="card section-card"><h3>Harness × Reward v4 单轨迹执行、恢复与奖励流程</h3><figure class="flow-figure"><img src="assets/final240-trajectory-execution-flow.png" alt="Final-240单轨迹执行、恢复与Reward v4奖励分类流程图"><figcaption>展示模型回合、动态 Tool Schema、Action Guard、环境执行、无进展候选收敛、终局分类与第16步起步数惩罚之间的完整关系。</figcaption></figure></article></section>
<section><h2>1. 任务结果（Reward v4聚合重算）</h2><div class="grid two"><article class="card"><h3>Strict Gold成功率</h3>{bars(strict_rates)}</article><article class="card"><h3>购买成功率</h3>{bars(purchase_rates)}</article></div><article class="card section-card"><h3>总体指标</h3><table><thead><tr><th>指标</th>{th_models}</tr></thead><tbody>{outcome_rows}</tbody></table></article><article class="card section-card"><h3>Harness v3 四轮稳定性（成功 = Gold + Valid）</h3><p class="lead"><strong>pass@4</strong> 表示同一任务四轮中至少成功一次，衡量能力覆盖；<strong>pass^4</strong> 表示同一任务四轮全部成功，衡量结果稳定性。</p><div class="table-wrap horizontal-only"><table><thead><tr><th>模型</th><th>四轮单轮成功数</th><th>pass@4</th><th>pass^4</th></tr></thead><tbody>{''.join(four_round_rows)}</tbody></table></div></article><article class="card section-card"><h3>Reward类型分布</h3><table><thead><tr><th>Reward类型</th>{th_models}</tr></thead><tbody>{reward_rows}</tbody></table></article></section>
<section><h2>2. Rubric需求满足</h2><article class="card"><h3>Rubric总体状态</h3><p class="lead">每条要求由同一DeepSeek V4 Pro合同标记为satisfied、violated、unknown或not_applicable。</p><table><thead><tr><th>状态</th>{th_models}</tr></thead><tbody>{rubric_rows}</tbody></table></article><article class="card section-card"><h3>Hard / Soft约束</h3><div class="table-wrap"><table><thead><tr><th>强度</th>{th_models}</tr></thead><tbody>{''.join(hardness_rows)}</tbody></table></div></article><article class="card section-card"><h3>按要求类型拆分</h3><div class="table-wrap"><table><thead><tr><th>要求类型</th>{th_models}</tr></thead><tbody>{''.join(category_rows)}</tbody></table></div></article></section>
<section><h2>3. 轨迹质量与错误归因</h2><div class="grid two"><article class="card"><h3>LLM Judge五维评分（0–2）</h3><div class="note" style="margin-bottom:18px"><strong>分值：</strong>0 = 关键行为缺失或明显不合理；1 = 部分做到，但覆盖、证据或效率仍有不足；2 = 完成充分且无明显问题。五维独立评分，不加权、不计算总分。<br><strong>维度：</strong>搜索策略看检索覆盖、有效改写和机械重复；候选利用看高匹配候选的利用、比较与收敛；证据核验看购买前是否检查关键属性、规格和最终价格；决策质量看商品、规格及购买/放弃是否合理；终止效率看是否过早购买/放弃、无效探索或耗尽步骤。</div><div class="dimensions">{''.join(dimension_blocks)}</div></article><article class="card"><h3>Primary Error</h3><div class="table-wrap horizontal-only"><table><thead><tr><th>错误</th>{th_models}</tr></thead><tbody>{error_rows}</tbody></table></div></article></div></section>
<section><h2>4. 行为、Token、耗时与上下文</h2><article class="card"><div class="table-wrap"><table><thead><tr><th>指标</th>{th_models}</tr></thead><tbody>{''.join(behavior_rows)}</tbody></table></div></article></section>
<section><h2>5. 工具调用次数</h2><article class="card"><p class="lead">逐轨迹汇总实际执行次数。0 表示该工具在对应 Tool Schema 中存在但没有被调用；“未使用”表示当前 8 工具 Schema 已不再暴露该工具。</p><div class="table-wrap"><table><thead><tr><th>工具</th>{th_models}</tr></thead><tbody>{''.join(tool_rows)}</tbody></table></div><p class="lead">Base 历史轨迹另有 4 次已废弃的内部 <code>think</code> 调用；它不属于上述 12 个标准购物工具，因此未计入表格。</p></article></section>
<section><h2>6. 阶段迁移与分层表现</h2><div class="grid three">{''.join(pair_cards)}</div>{''.join(success_stratification_cards)}<article class="card section-card"><h3>按评测集合、Challenge类型与商品领域分层</h3><p class="lead">成功口径为Gold + Valid；用于补充观察Core/Challenge、六类Challenge与九个商品领域的差异。</p><div class="table-wrap"><table><thead><tr><th>分层</th><th>子集</th><th>任务数</th>{th_models}<th>SFT v1→GRPO230 v3</th></tr></thead><tbody>{''.join(stratified_rows)}</tbody></table></div></article></section>
<section><h2>7. 逐题审计</h2><article class="card"><h3>240题逐题审计（七组模型；成功 = Gold + Valid）</h3><div class="filters"><input id="task-search" placeholder="搜索Reward、Judge状态或错误"><select id="transition-filter"><option value="">全部 GRPO230 v2→v3 转移</option><option value="failure_to_success">失败→成功</option><option value="success_to_success">成功→成功</option><option value="success_to_failure">成功→失败</option><option value="failure_to_failure">失败→失败</option></select><span id="visible-count"></span><a href="per-task-comparison.csv">CSV</a><a href="per-task-comparison.json">JSON</a></div><div class="table-wrap"><table class="task-table"><thead><tr><th>Task</th>{headers}</tr></thead><tbody id="task-body">{''.join(task_rows)}</tbody></table></div></article></section>
<section><h2>8. 八类典型轨迹</h2><p class="lead">每类各选一条。<strong>Hard</strong> 是不可违反的要求，任一可评分Hard失败即为错误购买；<strong>Soft</strong> 是允许折中的偏好，Hard全部通过但Soft有失败时才是Partial。</p><div class="grid two">{''.join(typical_cards)}</div></section>
<section><h2>9. GRPO230 v3完整交互流程</h2><p class="lead">展示一条普通正确购买、一条错误购买，以及一条触发6步无进展后进入强制候选收敛并最终选对的轨迹。每条均包含完整System Prompt、8工具基础Schema、用户需求、全部模型回合、工具参数、Observation/Harness纠正和最终终局。</p><div class="grid">{''.join(full_trajectory_cards)}</div></section>
<footer><p><a href="audit-report.md">审计报告</a> · <a href="comparison.json">Comparison JSON</a></p></footer></div><script>
const search=document.getElementById('task-search'),filter=document.getElementById('transition-filter'),rows=[...document.querySelectorAll('#task-body tr')],count=document.getElementById('visible-count');function apply(){{const q=search.value.toLowerCase(),t=filter.value;let n=0;rows.forEach(r=>{{const show=(!q||r.dataset.search.includes(q))&&(!t||r.dataset.transition===t);r.hidden=!show;if(show)n++;}});count.textContent=`显示 ${{n}} / ${{rows.length}}`;}}search.addEventListener('input',apply);filter.addEventListener('change',apply);apply();
</script></body></html>"""
    document = document.replace(
        "240题逐题审计（七组模型；成功 = Gold + Valid）",
        "240题逐题审计（八组模型；成功 = Gold + Valid）",
    ).replace(
        "<th>SFT v1→GRPO230 v3</th>",
        "<th>纠错 SFT v1→GRPO230 v3</th>",
    ).replace(
        ".kpis{grid-template-columns:repeat(6,minmax(0,1fr))}",
        ".kpis{grid-template-columns:repeat(4,minmax(0,1fr))}",
    )
    document = document.replace(
        "</section>\n<section><h2>Harness版本演进</h2>",
        "</section>\n<section><h2>购物 Agent 后训练与评测闭环</h2><article class=\"card\"><figure class=\"flow-figure\"><img src=\"assets/shopping-agent-post-training-evaluation-loop.png\" alt=\"购物 Agent Teacher数据、SFT、在线GRPO、Final-240评测与Bad Case归因闭环\"><figcaption>从Teacher轨迹采集、三层数据清洗、LoRA SFT、在线GRPO，到冻结Final-240评测和Event-level Bad Case归因的完整后训练闭环。</figcaption></figure></article></section>\n<section><h2>Teacher 数据构成与 SFT 数据工程</h2><article class=\"card\"><figure class=\"flow-figure\"><img src=\"assets/teacher-sft-data-engineering-overview.png\" alt=\"从8708条候选轨迹筛选Final-1000 Teacher数据并完成SFT训练的数据工程总览\"><figcaption>展示8708条候选轨迹的类型构成、三层质量门、600条Stable与400条Corrective Teacher配比、Gold召回位置与轨迹长度联合配平、关键行为覆盖，以及30K上下文LoRA SFT配置和正确购买率提升。</figcaption></figure></article>"
        + '<h3 class="trajectory-section-title">四类纠错型 Teacher 代表轨迹</h3>'
        + '<p class="lead">四类各选一条Final-1000中的严格成功轨迹。Corrective不是失败样本，而是展示模型如何识别风险、完成恢复并以Gold购买结束。</p>'
        + f'<div class="grid two">{"".join(teacher_corrective_cards)}</div></section>'
        + "\n<section><h2>Harness版本演进</h2>",
        1,
    )
    document = document.replace(
        "<div class=\"grid two section-card\"><article class=\"card reward-rules\"><h3>Reward v3.2历史评分细则</h3>",
        "<article class=\"card section-card\"><h3>Reward v4 Hard / Soft判定树</h3><figure class=\"flow-figure\"><img src=\"assets/reward-v4-hard-soft-decision-tree.png\" alt=\"Reward v4从Query约束解析到Gold Valid Partial Wrong的Hard Soft判定树\"><figcaption>先按公开Query冻结Hard、Soft、Ignore与Needs Review语义，再核验当前完整Variant；任一可评分Hard失败必为Wrong，Partial只允许在全部Hard通过后由Soft失败触发。</figcaption></figure></article><div class=\"grid two section-card\"><article class=\"card reward-rules\"><h3>Reward v3.2历史评分细则</h3>",
        1,
    )
    document = document.replace(
        "<section><h2>2. Rubric需求满足</h2>",
        "<section><h2>2. Rubric需求满足</h2><article class=\"card\"><h3>Final-240评测与隔离架构</h3><figure class=\"flow-figure\"><img src=\"assets/final240-evaluation-isolation-architecture.png\" alt=\"Final-240冻结轨迹、Reward重放、Rubric校验、盲评Judge与资源审计隔离架构\"><figcaption>同一冻结轨迹分别进入确定性Reward重放、冻结Rubric校验、盲评LLM-as-a-Judge和Token/时延/工具审计；Judge不接触Reward、Gold或其他模型结果。</figcaption></figure></article>",
        1,
    )
    document = document.replace(
        "删除 <code>view_description</code>、<code>view_features</code>、<code>view_reviews</code>、<code>view_attributes</code> 四种低频信息工具。",
        "删除 <code>view_description</code>、<code>view_features</code>、<code>view_reviews</code>、<code>view_attributes</code> 四种低频信息工具，并将原信息子页中的非空内容直接合并到商品详情，使模型调用 <code>open_product</code> 后即可在最新 Observation 中一次获得完整商品信息。",
    )
    document = document.replace(
        "每条要求由同一DeepSeek V4 Pro合同标记为satisfied、violated、unknown或not_applicable。",
        "八组共享同一DeepSeek V4 Flash冻结Rubric，并由同一DeepSeek V4 Pro盲评合同标记satisfied、violated、unknown或not_applicable。",
    )
    candidate_recovery_html = (
        '<article class="card section-card"><h3>GRPO230 v3：34题进入候选强制阶段后的转化</h3>'
        '<p class="lead">这34题都达到连续6步无实质进展的原Loop阈值；Harness v3没有立即判Loop，而是全部切入候选记忆完成最终决策。</p>'
        '<div class="grid three">'
        f'<div class="card"><h3>{candidate_recovery["triggered"]}</h3><p>达到原Loop阈值</p></div>'
        f'<div class="card"><h3>{candidate_recovery["entered"]}（{pct(candidate_recovery["entry_rate"], 2)}）</h3><p>进入候选强制阶段</p></div>'
        f'<div class="card"><h3>{candidate_recovery["successful"]} / {candidate_recovery["entered"]}（{pct(candidate_recovery["success_rate"], 2)}）</h3><p>转化为Gold或Valid</p></div>'
        '</div><table><thead><tr><th>最终终局</th><th>题数</th></tr></thead><tbody>'
        f'<tr><td>Gold</td><td>{candidate_recovery["gold"]}</td></tr>'
        f'<tr><td>Valid</td><td>{candidate_recovery["valid"]}</td></tr>'
        f'<tr><td>Wrong</td><td>{candidate_recovery["wrong"]}</td></tr>'
        f'<tr><td>Partial</td><td>{candidate_recovery["partial"]}</td></tr>'
        f'<tr><td>Guard rejection</td><td>{candidate_recovery["guard_rejection"]}</td></tr>'
        '<tr><td>Repeat Loop</td><td>0</td></tr></tbody></table>'
        f'<p class="note">“Loop由34降到1”表示34条原本可能按Loop结束的无进展轨迹被候选模块接管，并不表示其中33条都购买正确：真正转成Gold/Valid的是{candidate_recovery["successful"]}条。'
        f'整轮唯一剩余Loop为Task {", ".join(map(str, candidate_recovery["remaining_loop_tasks"]))}，它没有进入候选强制阶段。</p></article>'
    )
    document = document.replace(
        '<figcaption>基础8工具合同保持不变；每轮实际 Tool Schema 由页面 footer、当前可见按钮、未选规格和候选收敛阶段共同裁剪，Action Guard 再校验参数是否合法。</figcaption></figure></article></section>',
        '<figcaption>基础8工具合同保持不变；每轮实际 Tool Schema 由页面 footer、当前可见按钮、未选规格和候选收敛阶段共同裁剪，Action Guard 再校验参数是否合法。</figcaption></figure></article>'
        + candidate_recovery_html
        + '</section>',
        1,
    )
    legacy_reward_html = (
        '<article class="card reward-evolution-card"><p class="version-tag">LEGACY BASELINE</p>'
        '<h3>ShopSimulator 原生 Reward</h3>'
        '<p class="lead">口头可简称 Reward v1；仓库中没有正式命名的 <code>reward_v1.py</code>。</p>'
        '<div class="reward-formula"><span>LOOSE · 加法平均</span><code>r_type × (属性命中 + 规格命中 + 价格命中) / 总要求数</code></div>'
        '<div class="reward-formula"><span>STRICT · 乘法约束</span><code>r_type × r_attribute × r_option × r_price</code></div>'
        '<div class="reward-points"><div class="reward-point"><strong>训练适配：</strong>full_success + 0.5 × strict + 0.2 × native</div>'
        '<div class="reward-point"><strong>局限：</strong>Loose可能维度补偿；Strict会因单项失败整体坍缩。</div>'
        '<div class="reward-point"><strong>终局：</strong>唯一Gold较死，错误购买、循环、超时和主动停止区分不足。</div></div></article>'
    )
    document = document.replace(
        '<section><h2>Reward版本演进</h2><div class="grid two">',
        '<section><h2>Reward版本演进</h2><div class="grid three reward-evolution-grid">'
        + legacy_reward_html,
        1,
    )
    document = document.replace(
        '<article class="card"><h3>v3 → v4：Hard / Soft语义合同</h3><p class="lead">从v3的“类目与价格固定 Hard Gate + 四维匹配率”，升级为基于用户公开 Query 的可审计 Hard/Soft 约束合同。品类始终为 Hard；“必须、一定、绝对不要、不超过、至少、明确区间”等高置信且可确定性核验的不可妥协要求也进入 Hard；“最好、优先、尽量、大约、左右、预算”等偏好或近似表达进入 Soft。无所谓类表达忽略，复杂歧义语义进入 Needs Review / audit-only，不强行参与评分。</p><p class="lead">任一可评分 Hard 失败即判 <code>wrong_purchase</code>；Hard 全通过后，目标商品为 Gold，完全满足 Soft 的替代商品为 Valid，只违反 Soft 的替代商品为 Partial。</p></article>',
        '<article class="card reward-evolution-card v4"><p class="version-tag">V3 → V4 · 语义合同</p><h3>从固定门槛到 Hard / Soft</h3>'
        '<p class="lead">不再只靠固定四维匹配率，而是从用户公开 Query 中冻结可审计约束。</p>'
        '<div class="reward-points"><div class="reward-point"><strong>Hard：</strong>品类，以及“必须、不超过、至少、明确区间”等不可妥协要求；任一失败即 Wrong。</div>'
        '<div class="reward-point"><strong>Soft：</strong>“最好、优先、尽量、大约、左右、预算”等可折中偏好。</div>'
        '<div class="reward-point"><strong>分流：</strong>Hard全通过后，目标商品为Gold；Soft全满足的替代为Valid；仅Soft失败为Partial。</div>'
        '<div class="reward-point"><strong>审计：</strong>无所谓表达忽略，复杂歧义进入Needs Review，不强行评分。</div></div></article>',
        1,
    )
    document = document.replace(
        '<article class="card"><h3>效率与终局分数调整</h3><p class="lead">新增第16步起的分段递增步数惩罚；将 <code>assistant_final</code> 与连续 Guard 拒绝由训练过滤/无效改为 <code>-0.8</code> 的有效负样本；并重新校准部分终局分数，其中 Partial 为 <code>0.5 + 0.3 × soft_score</code>，Loop 为 <code>-0.6</code>。</p><table><thead><tr><th>Checkpoint</th><th>训练 Reward</th></tr></thead><tbody><tr><td>GRPO100</td><td>Reward v3</td></tr><tr><td>GRPO230</td><td>Reward v4</td></tr></tbody></table></article>',
        '<article class="card reward-evolution-card v4"><p class="version-tag">V3 → V4 · 优化信号</p><h3>效率惩罚与终局重标定</h3>'
        '<p class="lead">让低效探索和非法终止成为可学习的负样本，同时拉开不同终局质量。</p>'
        '<div class="reward-points"><div class="reward-point"><strong>步数：</strong>第16步起按区间递增累计惩罚。</div>'
        '<div class="reward-point"><strong>非法终止：</strong><code>assistant_final</code>与连续Guard拒绝由无效改为<code>-0.8</code>。</div>'
        '<div class="reward-point"><strong>分数：</strong>Partial为<code>0.5 + 0.3 × soft_score</code>；Loop为<code>-0.6</code>。</div></div>'
        '<table><thead><tr><th>Checkpoint</th><th>训练Reward</th></tr></thead><tbody><tr><td>GRPO100</td><td>v3</td></tr><tr><td>GRPO230</td><td>v4</td></tr></tbody></table></article>',
        1,
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
    comparison["success_stratification"] = build_success_stratification(slices)
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

    typical_cases = build_typical_trajectory_cases(evaluations)
    render_markdown(summaries, comparison, evaluations, typical_cases)
    render_dashboard(summaries, comparison, evaluations, per_task, typical_cases)

    reference_manifest = load_json(REFERENCE / "run_manifest.json")
    source_audit = {
        "historical_four_stage_source": str(REFERENCE.relative_to(ROOT)),
        "sft_normal1000_rollout_source": str(
            NORMAL_SFT_ROLLOUT_SOURCE.relative_to(ROOT)
        ),
        "sft_normal1000_judge_source": str(NORMAL_SFT_SOURCE.relative_to(ROOT)),
        "sft_normal1000_judges_sha256": sha256(
            OUTPUT / "judges-sft_normal1000.jsonl"
        ),
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
            "sft_normal1000",
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
        "schema_version": "eight-run-trajectory-judge-report-v3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": reference_manifest["protocol"],
        "models": {
            "rubric_curator": reference_manifest["models"]["rubric_curator"],
            "trajectory_judge": reference_manifest["models"]["trajectory_judge"],
            "actors": {
                "base": historical_actors["base"],
                "sft_normal1000": load_json(
                    NORMAL_SFT_ROLLOUT_SOURCE / "run_manifest.json"
                )["actor"],
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
            "sft_normal1000": "new_official_deepseek_v4_pro_judge",
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
