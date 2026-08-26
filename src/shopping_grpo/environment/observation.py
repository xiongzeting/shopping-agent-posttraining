"""把环境的结构化状态渲染成模型可见的稳定 observation。

渲染器只输出公开商品信息、当前页面和可执行按钮；它会主动拒绝 goal、reward
等隐藏字段，防止训练和评测把答案泄露给模型。
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping

from shopping_grpo.environment.candidate_memory import (
    CANDIDATE_CONVERGENCE_NOTICE_PREFIX,
    attach_candidate_memory,
    render_candidate_memory,
    update_candidate_memory,
)
from shopping_grpo.environment.product_id import is_product_id

OBSERVATION_VERSION = "shopping-observation-v2"
HEADER = "[SHOPPING_OBSERVATION_V2]"
STEP_BUDGET_NOTICE_PREFIX = "步数提醒:"
STEP_BUDGET_NOTICE_AFTER = 34
LOOP_RECOVERY_NOTICE_PREFIX = "\u5faa\u73af\u63d0\u9192:"
LOOP_RECOVERY_NOTICE_AFTER = 3
OPTION_ID_PATTERN = re.compile(r"^opt_[0-9a-f]{16}$")


class StructuredObservationError(ValueError):
    """The environment supplied a malformed or unsafe public state."""


def _text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _list(value):
    if not isinstance(value, list):
        return []
    return [_text(item) for item in value if _text(item)]


def _footer(state):
    actions = _list(state.get("actions"))
    return [
        f"搜索功能是否可用: {bool(state.get('search_available'))}",
        "可点击的按钮: " + json.dumps(actions, ensure_ascii=False),
    ]


def add_step_budget_notice(
    observation,
    *,
    step_count,
    max_steps,
    notice_after=STEP_BUDGET_NOTICE_AFTER,
    no_progress_steps=0,
    loop_notice_after=LOOP_RECOVERY_NOTICE_AFTER,
    candidate_count=0,
    candidate_limit=4,
):
    """把确定性提醒放在页面类型之后、页面正文之前，动作 footer 仍保持最后。"""
    observation = str(observation)
    step_count = int(step_count)
    max_steps = int(max_steps)
    notice_after = int(notice_after)
    no_progress_steps = int(no_progress_steps or 0)
    loop_notice_after = int(loop_notice_after)
    if "page_type: terminal" in observation:
        return observation

    remaining_steps = max(0, max_steps - step_count)
    notices = []
    late_decision_phase = step_count >= 35 and max_steps > 0
    if step_count >= 40 and max_steps > 0:
        notices.append(
            f"{STEP_BUDGET_NOTICE_PREFIX} 已执行 {step_count}/{max_steps} 步，仅剩 {remaining_steps} 步。"
            "剩余步数很少，强烈建议立即比较当前商品与已核验商品，完成必要规格后购买，"
            "或在确无可接受商品时合理结束；若仍需探索，只能使用当前页面实际暴露的工具，"
            "并确保下一步能带来实质性新证据。"
        )
    elif step_count >= 35 and max_steps > 0:
        notices.append(
            f"{STEP_BUDGET_NOTICE_PREFIX} 已执行 {step_count}/{max_steps} 步，仅剩 {remaining_steps} 步。"
            "请开始收敛，利用已有核验信息排除不合适商品，并优先对当前可操作商品完成规格和购买决策；"
            "不要继续低价值探索。当前可调用工具仍以最新页面实际暴露的列表为准。"
        )
    elif step_count > notice_after and max_steps > 0:
        notices.append(
            f"{STEP_BUDGET_NOTICE_PREFIX} 已执行 {step_count}/{max_steps} 步，仅剩 {remaining_steps} 步。"
            "开始收敛：利用候选记忆比较已核验商品，停止低价值扩展；发现满足全部要求的当前 "
            "variant 时必须立即购买。"
        )
    if no_progress_steps >= loop_notice_after and not late_decision_phase:
        notices.append(
            f"{LOOP_RECOVERY_NOTICE_PREFIX} 已连续 {no_progress_steps} 步无实质进展，"
            "已进入 Loop 高风险状态，继续无进展将触发终止。现在必须立即改变策略："
            "立即停止重复探索，改为核验当前页新商品、完成当前规格并购买，或在没有可接受候选时合理结束；"
            "禁止继续同义搜索、重复打开商品或页面往返。"
        )
    if not notices:
        return observation

    lines = [
        line
        for line in observation.splitlines()
        if not line.startswith(
            (
                STEP_BUDGET_NOTICE_PREFIX,
                LOOP_RECOVERY_NOTICE_PREFIX,
            )
        )
    ]
    page_type_index = next(
        (index for index, line in enumerate(lines) if line.startswith("page_type:")),
        -1,
    )
    insert_index = page_type_index + 1 if page_type_index >= 0 else 0
    lines[insert_index:insert_index] = notices
    return "\n".join(lines)


def render_structured_observation(
    state: Mapping,
    *,
    candidate_memory: dict | None = None,
    step_count: int = 0,
    show_candidate_memory: bool = True,
) -> str:
    """渲染一个状态，并拒绝版本不匹配或包含隐藏字段的输入。"""
    if not isinstance(state, Mapping):
        raise StructuredObservationError("observation_state must be an object")
    if state.get("observation_version") != OBSERVATION_VERSION:
        raise StructuredObservationError("unsupported observation_state version")
    forbidden = {
        "goal",
        "reward",
        "reward_detail",
        "target_asin",
        "answer",
        "candidate_state",
        "current_candidate",
        "best_candidate",
        "satisfied_conditions",
        "missing_conditions",
        "unverified_conditions",
        "public_match_score",
        "fully_satisfied",
    }
    leaked = forbidden.intersection(state)
    if leaked:
        raise StructuredObservationError(
            "observation_state contains forbidden fields: " + ", ".join(sorted(leaked))
        )

    # 先按页面类型渲染正文，最后统一追加搜索状态和按钮 footer；守卫和投影器
    # 都依赖这个 footer 来判断下一步动作是否合法。
    page_type = str(state.get("page_type") or "unknown")
    lines = [HEADER, f"page_type: {page_type}"]
    if page_type == "search_home":
        lines.append("使用 search_products 提交简短、具有区分度的商品查询。")
    elif page_type == "search_results":
        lines.extend(_render_search_results(state))
    elif page_type in {"product_detail", "information_subpage"}:
        lines.extend(_render_product(state))
        if page_type == "information_subpage":
            lines.append(f"subpage: {_text(state.get('subpage'))}")
            lines.append("content: " + _text(state.get("content")))
    elif page_type != "terminal":
        raise StructuredObservationError(f"unsupported page_type: {page_type!r}")
    rendered = "\n".join(lines) + "\n\n" + "\n".join(_footer(state))
    if candidate_memory is not None and page_type != "terminal":
        update_candidate_memory(
            candidate_memory,
            state,
            step_count=step_count,
        )
        if show_candidate_memory:
            current_asin = ""
            if page_type == "product_detail" and isinstance(state.get("product"), Mapping):
                current_asin = _text(state["product"].get("asin"))
            rendered = attach_candidate_memory(
                rendered,
                render_candidate_memory(
                    candidate_memory,
                    current_asin=current_asin,
                ),
            )
    return rendered


def _render_search_results(state):
    """渲染搜索结果，并确保每个展示的 ASIN 都是可操作目标。"""
    products = state.get("products")
    if not isinstance(products, list):
        raise StructuredObservationError("search results must contain a products list")
    actions = set(_list(state.get("actions")))
    product_asins = []
    lines = [
        f"query: {_text(state.get('query'))}",
        f"normalized_query: {_text(state.get('normalized_query'))}",
        (
            f"Page {int(state.get('page', 1))} of {int(state.get('total_pages', 1))} "
            f"(Total results: {int(state.get('total_results', 0))}; "
            f"ranks {int(state.get('rank_start', 0))}-{int(state.get('rank_end', 0))})"
        ),
        "格式: rank|asin|price|brand|category|key_attributes|title",
    ]
    for product in products:
        if not isinstance(product, Mapping):
            raise StructuredObservationError("each product must be an object")
        asin = _text(product.get("asin"))
        if not is_product_id(asin):
            raise StructuredObservationError(f"invalid search-result ASIN: {asin!r}")
        product_asins.append(asin)
        attributes = ",".join(_list(product.get("key_attributes")))
        lines.append(
            "|".join(
                (
                    str(int(product.get("rank", 0))),
                    asin,
                    _text(product.get("price")),
                    _text(product.get("brand")),
                    _text(product.get("category")),
                    attributes,
                    _text(product.get("title")),
                )
            )
        )
    actionable_asins = {action for action in actions if is_product_id(action)}
    if set(product_asins) != actionable_asins:
        raise StructuredObservationError(
            "model-visible search ASINs differ from environment-actionable ASINs"
        )
    if len(product_asins) > 20:
        raise StructuredObservationError("search page exceeds the frozen page size of 20")
    lines.append(f"products_shown: {len(product_asins)}")
    return lines


def _render_product(state):
    """渲染商品详情和规格选择，价格优先使用当前已选 variant 的价格。"""
    product = state.get("product")
    if not isinstance(product, Mapping):
        raise StructuredObservationError("product page must contain a product object")
    asin = _text(product.get("asin"))
    if not is_product_id(asin):
        raise StructuredObservationError(f"invalid product ASIN: {asin!r}")
    available_options = state.get("available_options") or {}
    selected_options = state.get("selected_options") or {}
    if not isinstance(available_options, Mapping):
        raise StructuredObservationError("available_options must be an object")
    if not isinstance(selected_options, Mapping):
        raise StructuredObservationError("selected_options must be an object")
    option_ids = set()
    for axis, records in available_options.items():
        if not isinstance(records, list):
            raise StructuredObservationError(
                f"available option axis {axis!r} must contain a list"
            )
        for record in records:
            if not isinstance(record, Mapping):
                raise StructuredObservationError("available option must be an object")
            option_id = _text(record.get("option_id"))
            label = _text(record.get("label"))
            if not OPTION_ID_PATTERN.fullmatch(option_id) or not label:
                raise StructuredObservationError("available option has invalid ID or label")
            option_ids.add(option_id)
    actions = set(_list(state.get("actions")))
    if not option_ids.issubset(actions):
        raise StructuredObservationError(
            "model-visible option IDs differ from environment-actionable option IDs"
        )
    for axis, record in selected_options.items():
        if not isinstance(record, Mapping):
            raise StructuredObservationError(
                f"selected option axis {axis!r} must contain an object"
            )
        option_id = _text(record.get("option_id"))
        if option_id not in option_ids or not _text(record.get("label")):
            raise StructuredObservationError("selected option is not currently available")

    lines = [
        f"asin: {asin}",
        f"title: {_text(product.get('title'))}",
        f"brand: {_text(product.get('brand'))}",
        f"category: {_text(product.get('category'))}",
        f"price: {_text(state.get('selected_price', product.get('price')))}",
        "key_attributes: " + ", ".join(_list(product.get("key_attributes"))),
    ]
    selected_axis_count = len(selected_options)
    total_axis_count = len(available_options)
    if selected_axis_count == total_axis_count:
        full_price = _text(state.get("selected_price", product.get("price")))
    else:
        full_price = "待完成全部规格轴后确认"
    lines.append(
        f"规格状态: {selected_axis_count}/{total_axis_count} 个规格轴已选择；"
        f"当前完整价格: {full_price}"
    )
    features = _list(product.get("features"))
    attributes = _list(product.get("attributes"))
    if features:
        lines.append("features: " + ", ".join(features))
    if attributes:
        lines.append("attributes: " + ", ".join(attributes))
    lines.extend(
        [
            "selected_options: "
            + json.dumps(selected_options, ensure_ascii=False, sort_keys=True),
            "available_options: "
            + json.dumps(available_options, ensure_ascii=False, sort_keys=True),
        ]
    )
    return lines
