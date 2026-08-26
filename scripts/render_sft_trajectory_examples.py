#!/usr/bin/env python3
"""Render representative Final-200 SFT trajectories as a readable Markdown report."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "outputs/evaluation/final200-vllm-parallel-20260804/sft/trajectories.jsonl"
)
DEFAULT_OUTPUT = ROOT / "results/sft/final200_lora_trajectory_examples.md"
EXAMPLES = {
    9214: "4 步短轨迹：严格成功",
    11361: "8 步中等轨迹：严格成功",
    3797: "23 步长轨迹：严格成功",
    3368: "8 步轨迹：部分替代购买",
    4925: "8 步轨迹：错误购买",
    2528: "33 步轨迹：重复循环终止",
}
EXAMPLE_NOTES = {
    4925: (
        "该轨迹保存的是修正规则前的历史结果。旧解析器把“300多元”错误解释为"
        "“不超过300元”，因此将336元目标商品标成 wrong_purchase。当前规则已将"
        "这类开放价格改为软偏好；只有重新评测后，轨迹中的 Reward 才会更新。"
    )
}
PRODUCT_ROW = re.compile(r"^\d+\|[^|]+\|")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_examples(path: Path) -> dict[int, dict]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = int(row["task_id"])
            if task_id in EXAMPLES:
                rows[task_id] = row
    missing = sorted(set(EXAMPLES) - set(rows))
    if missing:
        raise SystemExit(f"missing selected task_id values: {missing}")
    return rows


def one_line(value: object, limit: int = 520) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def observation_summary(value: object) -> list[str]:
    lines = [line.strip() for line in str(value or "").splitlines() if line.strip()]
    if not lines:
        return ["环境没有返回可显示文本。"]
    if any(line == "page_type: terminal" for line in lines):
        return ["环境进入 terminal 页面。"]

    prefixes = (
        "page_type:",
        "query:",
        "Page ",
        "asin:",
        "title:",
        "brand:",
        "category:",
        "price:",
        "key_attributes:",
        "selected_options:",
        "available_options:",
        "subpage:",
        "content:",
        "products_shown:",
        "搜索功能是否可用:",
    )
    selected = [one_line(line, 420) for line in lines if line.startswith(prefixes)]
    products = [one_line(line, 300) for line in lines if PRODUCT_ROW.match(line)][:3]
    if products:
        selected.append("搜索结果前三项：")
        selected.extend(products)
    return selected[:14] or [one_line(lines[0], 420)]


def assistant_calls(row: dict) -> list[tuple[dict, dict | None]]:
    steps_by_call_id = {}
    remaining_steps = list(row.get("steps") or [])
    for step in remaining_steps:
        call_id = ((step.get("tool_call") or {}).get("id"))
        if call_id:
            steps_by_call_id[call_id] = step

    calls = []
    fallback_index = 0
    for message in row.get("messages") or []:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            call_id = call.get("id")
            step = steps_by_call_id.get(call_id)
            if step is None and fallback_index < len(remaining_steps):
                candidate = remaining_steps[fallback_index]
                function = call.get("function") or {}
                if candidate.get("tool_name") == function.get("name"):
                    step = candidate
                    fallback_index += 1
            calls.append(({"message": message, "call": call}, step))
    return calls


def public_outcome(row: dict) -> dict:
    terminal = row.get("terminal_result") or {}
    reward = terminal.get("reward_detail") or {}
    purchase = terminal.get("purchase") or {}
    gates = reward.get("hard_gates") or {}
    return {
        "status": row.get("status"),
        "steps": len(row.get("steps") or []),
        "final_reward": row.get("final_reward"),
        "reward_type": reward.get("reward_type", "unknown"),
        "reward_valid": reward.get("reward_valid"),
        "termination_reason": reward.get("termination_reason", row.get("status")),
        "purchase_asin": purchase.get("asin"),
        "purchase_name": purchase.get("name"),
        "purchase_price": purchase.get("price"),
        "purchase_options": purchase.get("options") or {},
        "category_gate": (gates.get("category") or {}).get("status"),
        "budget_gate": (gates.get("budget") or {}).get("status"),
    }


def render_example(task_id: int, row: dict) -> list[str]:
    initial = row.get("initial_result") or {}
    outcome = public_outcome(row)
    result = [
        f"<details><summary><strong>Task {task_id}｜{EXAMPLES[task_id]}</strong></summary>",
        "",
        "### 用户 Query",
        "",
        f"> {initial.get('instruction', '未记录')}",
        "",
    ]
    if task_id in EXAMPLE_NOTES:
        result.extend([f"> **规则修正说明：** {EXAMPLE_NOTES[task_id]}", ""])
    result.extend(["### 模型与环境交互", ""])
    for index, (entry, step) in enumerate(assistant_calls(row), start=1):
        message = entry["message"]
        function = entry["call"].get("function") or {}
        content = one_line(message.get("content")) or "（没有输出显式分析文字，直接调用工具。）"
        result.extend(
            [
                f"#### 第 {index} 次决策",
                "",
                f"模型原话：{content}",
                "",
                f"工具调用：`{function.get('name')}`，参数 `{function.get('arguments', '{}')}`",
                "",
            ]
        )
        if step is None:
            result.extend(["环境结果：该调用没有形成已执行步骤，可能被 Guard 拒绝。", ""])
            continue
        result.append("环境返回的关键信息：")
        result.append("")
        for line in observation_summary(step.get("observation")):
            result.append(f"- {line}")
        result.append("")

    result.extend(
        [
            "### 最终结果",
            "",
            "| 字段 | 结果 |",
            "|---|---|",
            f"| 状态 | `{outcome['status']}` |",
            f"| 已执行工具步数 | {outcome['steps']} |",
            f"| Reward 类型 | `{outcome['reward_type']}` |",
            f"| 最终 Reward | `{outcome['final_reward']}` |",
            f"| Reward 可验证 | `{outcome['reward_valid']}` |",
            f"| 终止原因 | `{outcome['termination_reason']}` |",
            f"| 类目硬门槛 | `{outcome['category_gate']}` |",
            f"| 预算硬门槛 | `{outcome['budget_gate']}` |",
            f"| 购买 ASIN | `{outcome['purchase_asin'] or '未购买'}` |",
            f"| 购买商品 | {one_line(outcome['purchase_name']) or '未购买'} |",
            f"| 实际价格 | `{outcome['purchase_price']}` |",
            f"| 已选规格 | `{json.dumps(outcome['purchase_options'], ensure_ascii=False)}` |",
            "",
            "</details>",
            "",
        ]
    )
    return result


def main() -> None:
    args = parse_args()
    rows = load_examples(args.input)
    lines = [
        "# Qwen3.5-2B LoRA：Final-200 真实轨迹示例",
        "",
        "本报告直接从已经完成的 SFT Final-200 评测轨迹生成，没有重新调用模型。",
        "展示的是模型当时可见的 Query、模型原始输出、工具调用、可见环境信息和公共终局结果；不展示隐藏 TaskFacts。",
        "历史 Reward 按当次评测规则原样保留；后续环境规则修正不会反向改写已经完成的评测文件。",
        "",
        "## 示例索引",
        "",
        "| task_id | 类型 | 步数 | Reward 类型 |",
        "|---:|---|---:|---|",
    ]
    for task_id in EXAMPLES:
        outcome = public_outcome(rows[task_id])
        lines.append(
            f"| {task_id} | {EXAMPLES[task_id]} | {outcome['steps']} | `{outcome['reward_type']}` |"
        )
    lines.extend(
        [
            "",
            "> 提示：点击下面每个 Task 的标题即可展开完整过程。环境搜索页只保留模型当时看到的前三项商品摘要，原始轨迹文件仍保留完整页面。",
            "",
        ]
    )
    for task_id in EXAMPLES:
        lines.extend(render_example(task_id, rows[task_id]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
