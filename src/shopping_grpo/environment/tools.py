"""ShopSimulator 对模型公开的唯一工具定义。

这里同时保存两件必须保持同步的内容：模型看到的 JSON Schema，以及把工具调用
转换成环境 ``search[...]``/``click[...]``/``finish[...]`` 动作的映射。
"""

CLICK_TOOL_ACTIONS = {
    "open_product": ("asin", None),
    "select_option": ("value", None),
    "next_page": (None, "Next >"),
    "prev_page": (None, "< Prev"),
    "back_to_search": (None, "Back to Search"),
    "buy_now": (None, "Buy Now"),
}


def tool_call_to_action(name, parameters):
    """把一个标准 tool call 转成 ShopSimulator 能执行的字符串动作。"""
    parameters = parameters or {}
    if name == "search_products":
        return f"search[{parameters['query']}]"
    if name == "finish_without_purchase":
        return f"finish[{parameters['reason']}]"
    key, fixed_value = CLICK_TOOL_ACTIONS[name]
    value = fixed_value if fixed_value is not None else parameters[key]
    return f"click[{value}]"


def _schema(name, description, properties=None, required=None):
    """生成统一格式的 OpenAI function schema，并禁止额外参数。"""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


_INTERACTION_TOOL_SCHEMAS = [
    _schema(
        "search_products",
        "仅当最新 observation 显示“搜索功能是否可用: True”时搜索。query 应简洁，包含品类和最有区分度的品牌、型号、功能或规格；不得重复相同查询或机械复制整段需求。",
        {"query": {"type": "string"}},
        ["query"],
    ),
    _schema(
        "open_product",
        "打开最新 observation 当前搜索结果中列出的候选商品以核验价格、功能和规格；asin 必须原样取自该页面。",
        {"asin": {"type": "string"}},
        ["asin"],
    ),
    _schema(
        "select_option",
        "为当前商品选择规格。value 必须填写最新 observation 的 available_options 中与所需 label 对应的稳定 option_id（形如 opt_...），不得填写 label、导航按钮或历史页面 ID。同一规格轴只选一个值；选择后按完整 variant 的实际价格重新核验需求。",
        {"value": {"type": "string"}},
        ["value"],
    ),
    _schema("next_page", "仅当当前页面显示 Next > 且当前页没有合适候选时翻到下一页以发现新商品；无参数，必须传 {}。"),
    _schema("prev_page", "仅当当前页面显示 < Prev 按钮时返回上一页；无参数，必须传 {}。"),
    _schema("back_to_search", "仅当当前页面显示 Back to Search 按钮时返回搜索页；无参数，必须传 {}。"),
    _schema(
        "buy_now",
        "购买当前商品并结束任务。仅当最新 observation 显示 Buy Now 时可调用。当前完整 variant 的品类正确且用户明确要求均已满足时，必须立即调用，不得继续搜索、比较或重复核验。充分探索后仍无完全匹配商品时，也可购买品类正确、在已核验候选中整体最符合且可接受的候选。只依据用户需求和当前可见商品信息判断，无需寻找任何预设商品。无参数，必须传 {}。",
    ),
]

_FINISH_WITHOUT_PURCHASE_SCHEMA = _schema(
    "finish_without_purchase",
    "主动结束且不购买，这不是成功。环境不再按检索是否充分细分停止类型，调用后统一按 early_abstain 终止；仅当经过有实质差异的搜索和候选核验后仍没有可接受商品，且没有明显值得继续核验的候选时使用。",
    {
        "reason": {
            "type": "string",
            "enum": ["no_suitable_product"],
        }
    },
    ["reason"],
)
SHOP_TOOL_SCHEMAS = [
    *_INTERACTION_TOOL_SCHEMAS,
    _FINISH_WITHOUT_PURCHASE_SCHEMA,
]
