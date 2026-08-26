"""根据最新 observation 守卫模型的工具调用。

ShopSimulator 的按钮和商品只在当前页面有效。动作守卫在请求到达环境前检查
这一点，并把拒绝原因反馈给模型，避免非法点击污染环境状态。
"""

import json
import re

from shopping_grpo.environment.product_id import (
    PRODUCT_ID_CAPTURE,
    is_product_id,
)
from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS, tool_call_to_action


RUNTIME_GUARD_FIELD = "runtime_action_guard"
NAVIGATION_BUTTONS = {
    "description",
    "features",
    "reviews",
    "attributes",
    "next >",
    "< prev",
    "back to search",
    "buy now",
}
TOOL_PARAMETERS = {
    tool["function"]["name"]: tool["function"]["parameters"]
    for tool in SHOP_TOOL_SCHEMAS
}


def resolve_action_parameters(name, arguments, observation):
    """Stable option IDs are already exact environment action parameters."""
    return arguments


def action_reject_reason(
    name,
    arguments,
    observation,
    *,
    tool_schemas=None,
    candidate_memory=None,
    step_count=0,
    evaluation_extensions=False,
):
    """返回动作拒绝原因；``None`` 表示允许执行。

检查顺序很重要：先校验 schema 参数，再处理无需页面状态的动作，最后只允许点击
最新 observation 中仍然存在的目标。
"""
    schema_error = _schema_argument_error(name, arguments, tool_schemas=tool_schemas)
    if schema_error:
        return schema_error
    active_tool_names = {
        str((tool.get("function") or {}).get("name") or "")
        for tool in (tool_schemas or [])
        if isinstance(tool, dict)
    }
    if tool_schemas is not None and name not in active_tool_names:
        if name == "open_product":
            return "click_not_in_previous_observation"
        if name == "search_products":
            return "search_not_available_on_current_page"
        return "tool_not_available_on_current_page"
    if name == "think":
        return None
    if name == "finish_without_purchase":
        if arguments.get("reason") != "no_suitable_product":
            return "invalid_finish_reason"
        return None
    if name == "search_products":
        if "搜索功能是否可用: False" in observation:
            return "search_not_available_on_current_page"
        return None
    if not observation:
        return "missing_previous_observation"
    if name == "open_product":
        asin = str(arguments.get("asin", ""))
        if asin not in product_ids(observation):
            return "click_not_in_previous_observation"
        return None

    try:
        action = tool_call_to_action(name, arguments)
    except Exception:
        return "unknown_or_invalid_tool"
    if not isinstance(action, str) or not action.startswith("click[") or not action.endswith("]"):
        return "click_not_in_previous_observation"
    target = action[6:-1].casefold()
    if name == "select_option" and target in NAVIGATION_BUTTONS:
        return "select_option_is_navigation_button"
    if name == "select_option" and not re.fullmatch(r"opt_[0-9a-f]{16}", target):
        return "select_option_requires_stable_id"
    if (
        evaluation_extensions
        and name == "select_option"
        and target in selected_option_ids(observation)
    ):
        return "option_already_selected"
    if target not in {button.casefold() for button in clickable_buttons(observation)}:
        return "click_not_in_previous_observation"
    return None


def _schema_argument_error(name, arguments, *, tool_schemas=None):
    """在调用环境前拒绝未知工具、缺失参数、错误类型和空字符串。"""
    parameters_by_name = TOOL_PARAMETERS
    if tool_schemas is not None:
        active_parameters = {
            tool["function"]["name"]: tool["function"]["parameters"]
            for tool in tool_schemas
        }
        # A globally known tool can be absent from the current page schema.
        # Validate its arguments against the stable global contract, then let
        # ``action_reject_reason`` return a page-specific rejection.
        parameters_by_name = dict(TOOL_PARAMETERS)
        parameters_by_name.update(active_parameters)
    parameters = parameters_by_name.get(name)
    if parameters is None:
        return "unknown_tool"
    if not isinstance(arguments, dict):
        return "schema_arguments_not_object"
    properties = parameters.get("properties", {})
    extra_names = sorted(set(arguments) - set(properties))
    if extra_names:
        return "schema_extra_arguments:" + ",".join(extra_names)
    required_names = set(parameters.get("required", []))
    missing_names = sorted(required_names - set(arguments))
    if missing_names:
        return "schema_missing_arguments:" + ",".join(missing_names)
    for argument_name, value in arguments.items():
        expected_type = (properties.get(argument_name) or {}).get("type")
        if expected_type == "string" and not isinstance(value, str):
            return f"schema_wrong_type:{argument_name}:string"
        if argument_name in required_names and isinstance(value, str) and not value.strip():
            return f"schema_empty_string:{argument_name}"
    return None


def action_guard_tool_message(
    tool_call,
    reason,
    observation,
    *,
    candidate_memory=None,
    tool_schemas=None,
):
    """构造标准 tool error observation，让 Agent 感知调用未被执行。"""
    targets = clickable_buttons(observation)
    asins = product_ids(observation)
    selected_ids = selected_option_ids(observation)
    option_ids = [
        target
        for target in targets
        if re.fullmatch(r"opt_[0-9a-f]{16}", target.casefold())
        and target.casefold() not in selected_ids
    ]
    normalized_targets = {target.casefold() for target in targets}
    page_type = _page_type(observation)
    allowed_tool_names = {
        str((schema.get("function") or {}).get("name") or "")
        for schema in (tool_schemas or [])
        if isinstance(schema, dict)
    }
    requested_name = str((tool_call.get("function") or {}).get("name") or "")
    open_calls = ", ".join(
        f'open_product(asin="{asin}")' for asin in asins[:20]
    )
    option_calls = ", ".join(
        f'select_option(value="{option_id}")' for option_id in option_ids[:20]
    )
    terminal_calls = []
    if "buy now" in normalized_targets:
        terminal_calls.append("buy_now({})")
    terminal_calls.append(
        'finish_without_purchase(reason="no_suitable_product")'
    )
    terminal_text = ", ".join(terminal_calls)
    return_tools = []
    if "< prev" in normalized_targets:
        return_tools.append("prev_page")
    if "back to search" in normalized_targets:
        return_tools.append("back_to_search")
    only_return_buttons = bool(normalized_targets) and normalized_targets <= {"< prev", "back to search"}
    if allowed_tool_names == {"open_product"}:
        correction = (
            "当前处于候选收敛选择阶段，只能从候选页调用："
            f"{open_calls or '当前没有可打开候选'}。"
        )
    elif allowed_tool_names == {"select_option"}:
        correction = (
            "当前处于候选规格阶段，不提供搜索、返回或终局工具。"
            f"只能选择当前未选规格：{option_calls or '当前没有合法规格 ID'}。"
        )
    elif allowed_tool_names == {"buy_now", "finish_without_purchase"}:
        correction = (
            "当前处于候选终局阶段，只能执行："
            f"{terminal_text}。"
        )
    elif (
        reason == "click_not_in_previous_observation"
        and requested_name == "open_product"
        and page_type == "product_detail"
    ):
        correction = (
            "当前处于商品详情页，不能直接打开历史搜索结果中的其他商品。"
            "请先调用 back_to_search({})，再重新搜索并从最新搜索结果打开商品。"
        )
    elif reason == "click_not_in_previous_observation" and only_return_buttons:
        correction = (
            f"当前页面只允许返回，下一步只能调用 {' 或 '.join(return_tools)}，"
            "返回详情页后再依据刷新后的按钮和规格继续。"
        )
    elif reason == "option_already_selected":
        choices = option_calls or "当前没有其他可选规格 ID"
        correction = (
            "该规格已经选中，禁止再次选择。"
            f"若仍有未闭合规格，只能选择当前未选规格：{choices}；"
            f"否则立即执行 {terminal_text}。"
        )
    elif reason == "click_not_in_previous_observation":
        if requested_name == "open_product":
            choices = []
            if open_calls:
                choices.append("当前页商品只能调用：" + open_calls)
            if page_type == "search_home":
                choices.append(
                    "当前已在搜索首页，不要调用 back_to_search；需要新搜索时必须使用实质不同的 query"
                )
            choices.append("没有可接受候选时执行 " + terminal_text)
            correction = "请求的 ASIN 不在当前页面。历史候选只用于比较；若需再次查看，必须重新搜索并从最新结果页打开。" + "；".join(choices) + "。"
        elif requested_name == "select_option":
            choices = option_calls or "当前没有合法的未选规格 ID"
            correction = (
                "请求的规格 ID 已过期或不属于当前商品。"
                f"只能从最新详情选择：{choices}；完成规格后执行 {terminal_text}。"
            )
        elif requested_name == "back_to_search" and page_type == "search_home":
            choices = []
            choices.append(
                "发起实质不同的新搜索：search_products(query=\"新的品类/品牌/型号/核心功能/规格组合\")"
            )
            choices.append("或执行 " + terminal_text)
            correction = "当前已经在搜索首页，不能再次返回搜索页。下一步可执行：" + "；".join(choices) + "。"
        else:
            choices = []
            if open_calls:
                choices.append("当前页商品：" + open_calls)
            if option_calls:
                choices.append("当前规格：" + option_calls)
            if return_tools:
                choices.append("返回操作：" + ", ".join(f"{name}({{}})" for name in return_tools))
            choices.append("终局操作：" + terminal_text)
            correction = "该按钮或参数不属于最新页面。只能选择以下一种合法操作：" + "；".join(choices) + "。"
    elif reason == "search_not_available_on_current_page":
        choices = []
        if open_calls:
            choices.append("打开当前商品：" + open_calls)
        if return_tools:
            choices.append("返回后再搜索：" + ", ".join(f"{name}({{}})" for name in return_tools))
        choices.append("或执行 " + terminal_text)
        correction = "当前页面不能搜索，禁止继续调用 search_products。下一步可执行：" + "；".join(choices) + "。"
    elif only_return_buttons:
        correction = f"当前页面只允许返回，下一步只能调用 {' 或 '.join(return_tools)}。"
    else:
        correction = (
            "请根据错误原因修正工具名和参数。"
            f"当前合法商品：{open_calls or '无'}；当前合法规格：{option_calls or '无'}；"
            f"终局操作：{terminal_text}。"
        )
    return {
        "role": "tool",
        "tool_call_id": tool_call.get("id"),
        "name": (tool_call.get("function") or {}).get("name"),
        RUNTIME_GUARD_FIELD: True,
        "content": (
            f"上一工具调用被本地动作守卫拒绝（{reason}），未执行。"
            f"{correction}请只执行上述与最新 observation 一致的一种操作，"
            "不要再次提交刚才被拒绝的调用。"
        ),
    }


def _page_type(observation):
    match = re.search(r"(?m)^page_type:\s*([^\n]+)", str(observation or ""))
    return match.group(1).strip() if match else ""


def product_ids(observation):
    """提取投影格式或原始 ``[SEP]`` 格式中的当前页商品 ID。"""
    projected = re.findall(
        rf"(?m)^\d+\|({PRODUCT_ID_CAPTURE})\|",
        observation,
    )
    legacy = [
        segment.strip()
        for segment in re.split(r"\s*\[SEP\]\s*", observation)
        if is_product_id(segment.strip())
    ]
    return list(
        dict.fromkeys(
            [*projected, *legacy]
        )
    )


def clickable_buttons(observation):
    """读取 observation footer 中当前页面实际可点击的按钮。"""
    match = re.search(r"可点击的按钮:\s*(\[[^\n]*\])", observation)
    if not match:
        return []
    try:
        buttons = json.loads(match.group(1))
    except json.JSONDecodeError:
        return []
    return [button for button in buttons if isinstance(button, str)]


def selected_option_ids(observation):
    """Extract the option IDs already selected on the current detail page."""
    match = re.search(r"(?m)^selected_options:\s*(\{.*\})\s*$", str(observation))
    if not match:
        return set()
    try:
        selected = json.loads(match.group(1))
    except json.JSONDecodeError:
        return set()
    if not isinstance(selected, dict):
        return set()
    return {
        str(record.get("option_id") or "").strip().casefold()
        for record in selected.values()
        if isinstance(record, dict) and record.get("option_id")
    }
