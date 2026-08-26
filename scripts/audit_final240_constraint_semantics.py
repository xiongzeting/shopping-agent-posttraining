"""Audit deterministic hard/soft constraint compilation on frozen Final-240."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from web_agent_site.engine.comparators import normalize_text
from web_agent_site.engine.reward import (
    candidate_options_for_evaluation,
    evaluate_purchase,
)
from web_agent_site.engine.reward_features import (
    _query_semantic_segments,
    compile_reward_features,
)
from web_agent_site.engine.variant_price import resolve_variant_price

ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_MARKER = re.compile(
    r"绝对|必须|务必|一定|不得|不可|不能|不要|不需要|无需|不用|"
    r"没有需求|没需求|无所谓|不要求|不限|不限制|可有可无|"
    r"最好|优先|尽量|希望|偏好|倾向|预期|预计|预估|左右|上下|大概|大约|"
    r"约莫|约摸|差不多|大致|将近|接近|附近|不超过|不高于|"
    r"至少|不低于|以内|以下|以上|之间|控制在|别太|别超过|"
    r"别让|越.{0,8}越好|约(?=[零一二两三四五六七八九十百千万\d])"
)
HARD_MARKER = re.compile(
    r"绝对|必须|务必|一定|不得|不可|不能|不要(?!太)|"
    r"不超过|不高于|至少|不低于|以内|以下|以上|之间|"
    r"控制在|别超过|别让"
)
SOFT_MARKER = re.compile(
    r"最好|优先|尽量|希望|偏好|倾向|预期|预计|预估|左右|上下|大概|大约|"
    r"约莫|约摸|差不多|大致|将近|接近|附近|比较合适|"
    r"不要太|别太|越.{0,8}越好|"
    r"约(?=[零一二两三四五六七八九十百千万\d])"
)
AMBIGUOUS_NEGATION_MARKER = re.compile(r"不需要|无需|不用|没有需求|没需求")


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _clauses(text: object) -> list[str]:
    return _query_semantic_segments(str(text or ""))


def _quote_covers_clause(quote: object, clause: str) -> bool:
    normalized_quote = normalize_text(quote)
    normalized_clause = normalize_text(clause)
    return bool(normalized_clause and normalized_clause in normalized_quote)


def _surface_marker_class(clause: str) -> str:
    normalized = normalize_text(clause)
    if AMBIGUOUS_NEGATION_MARKER.search(normalized):
        return "ambiguous_negative"
    hard = bool(HARD_MARKER.search(normalized))
    soft = bool(SOFT_MARKER.search(normalized))
    if hard and soft:
        return "mixed_hard_soft"
    if hard:
        return "hard_like"
    if soft:
        return "soft_like"
    return "contextual_marker"


def audit() -> dict:
    task_ids = [
        row["task_id"]
        for row in _load_jsonl(ROOT / "data/evaluation/slices.jsonl")
    ]
    products = _load_jsonl(
        ROOT / "data/shopsimulator_official/fine_items_eval_standard.jsonl"
    )
    strength_counts = Counter()
    enforcement_counts = Counter()
    reason_counts = Counter()
    type_strength_counts: dict[str, Counter] = defaultdict(Counter)
    type_enforcement_counts: dict[str, Counter] = defaultdict(Counter)
    gold_reward_counts = Counter()
    unresolved = []
    unresolved_scored = []
    audit_only_semantic_clauses = []
    ungrounded_scored = []
    uncovered_marker_clauses = []
    task_rows = []

    for task_id in task_ids:
        product = products[task_id]
        instruction = product["instructions"][0]
        query = instruction["instruction"]
        features = compile_reward_features(instruction, product)
        constraints = features["query_constraint_contract"]["constraints"]
        for constraint in constraints:
            strength = str(constraint.get("strength") or "unknown")
            enforcement = str(constraint.get("enforcement") or "scored")
            constraint_type = str(
                constraint.get("constraint_type") or "unknown"
            )
            strength_counts[strength] += 1
            enforcement_counts[enforcement] += 1
            reason_counts[str(constraint.get("semantics_reason") or "unknown")] += 1
            type_strength_counts[constraint_type][strength] += 1
            type_enforcement_counts[constraint_type][enforcement] += 1
            row = {
                "task_id": task_id,
                "query": query,
                **constraint,
            }
            if strength == "needs_review":
                unresolved.append(row)
                if enforcement == "scored":
                    unresolved_scored.append(row)
            if (
                constraint_type == "query_clause"
                and enforcement == "audit_only"
            ):
                audit_only_semantic_clauses.append(row)
            if (
                strength in {"hard", "soft"}
                and enforcement == "scored"
                and constraint_type != "category"
                and not constraint.get("query_quote")
            ):
                ungrounded_scored.append(row)

        for clause in _clauses(query):
            if not SEMANTIC_MARKER.search(normalize_text(clause)):
                continue
            if not any(
                _quote_covers_clause(constraint.get("query_quote"), clause)
                for constraint in constraints
            ):
                uncovered_marker_clauses.append(
                    {
                        "task_id": task_id,
                        "clause": clause,
                        "query": query,
                        "surface_class": _surface_marker_class(clause),
                    }
                )

        goal = {
            "asin": product["asin"],
            "category": product["category"],
            "instruction_text": query,
            **features,
        }
        selected_options, _ = candidate_options_for_evaluation(
            product,
            goal["required_options_by_key"],
        )
        reward = evaluate_purchase(
            product,
            goal,
            selected_options=selected_options,
            price_resolution=resolve_variant_price(product, selected_options),
        )
        reward_payload = reward.to_dict()
        gold_reward_counts[reward.reward_type] += 1
        task_rows.append(
            {
                "task_id": task_id,
                "query": query,
                "constraint_counts": dict(
                    Counter(
                        str(item.get("strength") or "unknown")
                        for item in constraints
                    )
                ),
                "enforcement_counts": dict(
                    Counter(
                        str(item.get("enforcement") or "scored")
                        for item in constraints
                    )
                ),
                "gold_reward_type": reward.reward_type,
                "gold_reward_valid": reward.reward_valid,
                "gold_reward": reward.reward,
                "hard_total": reward_payload["evidence"][
                    "strict_purchase_contract"
                ]["hard_total"],
                "soft_total": reward_payload["evidence"][
                    "strict_purchase_contract"
                ]["soft_total"],
            }
        )

    uncovered_class_counts = Counter(
        item["surface_class"] for item in uncovered_marker_clauses
    )
    return {
        "schema_version": "final240-constraint-semantics-audit-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_count": len(task_ids),
        "constraint_count": sum(strength_counts.values()),
        "strength_counts": dict(strength_counts),
        "enforcement_counts": dict(enforcement_counts),
        "semantics_reason_counts": dict(reason_counts),
        "constraint_type_strength_counts": {
            key: dict(value) for key, value in sorted(type_strength_counts.items())
        },
        "constraint_type_enforcement_counts": {
            key: dict(value)
            for key, value in sorted(type_enforcement_counts.items())
        },
        "gold_reward_type_counts": dict(gold_reward_counts),
        "unresolved_semantics": unresolved,
        "unresolved_scored_semantics": unresolved_scored,
        "audit_only_semantic_clauses": audit_only_semantic_clauses,
        "ungrounded_scored_constraints": ungrounded_scored,
        "uncovered_semantic_marker_clauses": uncovered_marker_clauses,
        "uncovered_surface_class_counts": dict(uncovered_class_counts),
        "tasks": task_rows,
    }


def _markdown(report: dict) -> str:
    lines = [
        "# Final-240 硬约束/软偏好审计",
        "",
        f"- 任务数：{report['task_count']}",
        f"- 约束数：{report['constraint_count']}",
        f"- 强度分布：{json.dumps(report['strength_counts'], ensure_ascii=False)}",
        f"- 执行方式：{json.dumps(report['enforcement_counts'], ensure_ascii=False)}",
        f"- Gold 商品回放：{json.dumps(report['gold_reward_type_counts'], ensure_ascii=False)}",
        f"- 待复核语义：{len(report['unresolved_semantics'])}",
        f"- 会阻断 Reward 的待复核语义：{len(report['unresolved_scored_semantics'])}",
        f"- 补入合同但仅用于审计的 Query 分句：{len(report['audit_only_semantic_clauses'])}",
        f"- 无 Query 原文支撑但参与评分：{len(report['ungrounded_scored_constraints'])}",
        f"- 含强弱/否定标记但未进入合同的 Query 分句：{len(report['uncovered_semantic_marker_clauses'])}",
        f"- 未覆盖分句表面类型：{json.dumps(report['uncovered_surface_class_counts'], ensure_ascii=False)}",
        "",
        "## 未覆盖的语义分句",
        "",
    ]
    uncovered = report["uncovered_semantic_marker_clauses"]
    if not uncovered:
        lines.append("无。")
    else:
        for item in uncovered:
            lines.append(
                f"- Task {item['task_id']} [{item['surface_class']}]："
                f"{item['clause']}"
            )
    lines.extend(["", "## 待复核约束", ""])
    unresolved = report["unresolved_semantics"]
    if not unresolved:
        lines.append("无。")
    else:
        for item in unresolved:
            lines.append(
                f"- Task {item['task_id']} / {item.get('constraint_id')}："
                f"{item.get('query_quote')} → {item.get('expected')}"
            )
    lines.extend(["", "## 审计型 Query 语义", ""])
    audit_only = report["audit_only_semantic_clauses"]
    if not audit_only:
        lines.append("无。")
    else:
        for item in audit_only:
            lines.append(
                f"- Task {item['task_id']} / {item.get('strength')}："
                f"{item.get('query_quote')}"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/final240-constraint-semantics-audit",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = audit()
    (args.output_dir / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "README.md").write_text(
        _markdown(report),
        encoding="utf-8",
    )
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "task_count": report["task_count"],
        "strength_counts": report["strength_counts"],
        "enforcement_counts": report["enforcement_counts"],
        "gold_reward_type_counts": report["gold_reward_type_counts"],
        "unresolved_semantics": len(report["unresolved_semantics"]),
        "unresolved_scored_semantics": len(
            report["unresolved_scored_semantics"]
        ),
        "audit_only_semantic_clauses": len(
            report["audit_only_semantic_clauses"]
        ),
        "ungrounded_scored_constraints": len(
            report["ungrounded_scored_constraints"]
        ),
        "uncovered_semantic_marker_clauses": len(
            report["uncovered_semantic_marker_clauses"]
        ),
        "uncovered_surface_class_counts": report[
            "uncovered_surface_class_counts"
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
