# Tool Schema v2

当前基础标识：`shopping-tools-v2`；Final-240 评测入口记录为
`shopping-evaluation-tools-v2.3`。

## 当前 8 个公开工具

| 工具 | 参数 | 作用 |
|---|---|---|
| `search_products` | `query: string` | 在搜索可用页面提交简洁且有区分度的查询 |
| `open_product` | `asin: string` | 打开最新搜索结果页中真实可见的商品 |
| `select_option` | `value: string` | 使用最新详情页的稳定 `option_id` 选择规格 |
| `next_page` | `{}` | 当前页面存在 `Next >` 时翻到下一页 |
| `prev_page` | `{}` | 当前页面存在 `< Prev` 时返回上一页 |
| `back_to_search` | `{}` | 当前页面存在 `Back to Search` 时返回搜索页 |
| `buy_now` | `{}` | 购买当前完整 variant 并终止任务 |
| `finish_without_purchase` | `reason=no_suitable_product` | 主动结束且不购买，统一进入 `early_abstain` |

`view_description`、`view_features`、`view_reviews`、`view_attributes`、`think` 和
`reopen_candidate` 均不属于当前公开工具。非空的公开商品特征和属性直接合并进商品详情
Observation；候选记忆只保存比较与重新定位线索，不提供历史商品直达工具。

所有 Schema 均设置 `additionalProperties=false`。无参数工具必须传严格 `{}`，每个 Assistant
回合只串行执行一个 Tool Call；如果模型一次生成多个调用，Harness 只保留第一个并记录其余调用。

## 页面级动态暴露

当前 Final-240 Evaluation 使用“基础工具集合 × 最新页面可执行动作”的交集生成本轮 Tool Schema：

| 最新页面状态 | 可能暴露的工具 |
|---|---|
| 搜索首页 | `search_products`、`finish_without_purchase` |
| 搜索结果 | `open_product`、当前可见翻页/返回工具、`finish_without_purchase` |
| 商品详情 | 当前未选规格、当前可见购买/返回工具、`finish_without_purchase` |
| 终局 | 不再请求下一轮工具 |

候选数、步数提醒和循环提醒不会进一步隐藏工具。历史 Observation 和候选记忆中的 ASIN、按钮、
规格 ID 也不会因此重新变成可执行目标。

## Action Guard 边界

动态 Tool Schema 负责减少模型当前能看到的错误选项；Action Guard 在执行前再次校验：

- 工具名、必填参数、参数类型和额外字段；
- `open_product` 的 ASIN 是否来自最新搜索结果页；
- `select_option` 是否使用最新详情页中未选中的稳定 `option_id`；
- 翻页、返回和购买按钮是否真实存在于最新 Observation；
- `finish_without_purchase` 是否使用唯一合法 reason。

拒绝动作不会进入环境，并会返回原因、当前合法目标和恢复方法；连续 3 次 Guard 拒绝后终止轨迹。
Guard 不判断商品是否满足用户语义，也不替模型挑选候选。

## 源码入口

- 冻结 Schema：`configs/tools.json`
- Schema 与动作映射：`src/shopping_grpo/environment/tools.py`
- 页面级动态暴露：`src/shopping_grpo/evaluation/rollout.py::_active_tool_schemas`
- 动作守卫：`src/shopping_grpo/environment/actions.py`
- 双入口一致性测试：`tests/test_shop_tools.py`、`tests/test_rollout.py`、
  `tests/test_action_validation.py`
