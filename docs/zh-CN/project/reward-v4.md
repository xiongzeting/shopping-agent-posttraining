# Reward v4

Reward v4 是只在终局执行、无需调用 LLM 的确定性购物 Reward。当前实现先将公开 Query 编译成
带原文证据的 Hard/Soft 约束合同，再用商品结构化字段、公开文本、已选规格和完整 variant 实际
价格逐项核验；Harness 和 Actor 在行动过程中都看不到 Gold、Reward 或隐藏约束结果。

## Query 约束合同

每个任务在 Rollout 前生成 `shopping-query-constraints-v1`。每项约束记录：

- `constraint_id`、类型、期望值、操作符和实际值来源；
- Query 中的连续原文证据；
- `hard / soft / ignore / needs_review` 语义强度；
- `scored / audit_only` 执行方式。

明确品类，以及“必须、一定、绝对不要、不超过”等未被软化的品牌、型号、功能、规格和价格
要求可进入 Hard；“最好、优先、尽量、左右、大约、预算、不要太”等偏好进入 Soft。语义上很硬
但缺少可靠确定性比较器的要求保留为 `audit_only`，避免制造错误负样本。

## 当前购买分类

1. 任一可评分 Hard 失败 → `wrong_purchase`；
2. 没有已知 Hard 失败，但某个可评分 Hard 无法核验 → `reward_unverifiable`；
3. Hard 全通过后，任一可评分 Soft 无法核验 → `reward_unverifiable`；
4. Hard 全通过，但任一可评分 Soft 失败 → `partial_alternative_purchase`，目标 ASIN与替代 ASIN都一样；
5. Hard、Soft 全通过且购买目标 ASIN → `gold_purchase`；
6. Hard、Soft 全通过且购买替代 ASIN → `valid_alternative_purchase`。

这意味着 Partial 仍然存在：它专门表示“不可违反的要求都满足，但可折中的偏好没有全部满足”的
购买，而不是 Hard 失败后的安慰分。目标 ASIN不是 Soft 免检通道；只有全部可评分 Soft 也通过才可得到 Gold。

## 当前基础终局效用

| Reward 类型 | 基础效用 |
|---|---:|
| `gold_purchase` | `1.00` |
| `valid_alternative_purchase` | `0.80` |
| `partial_alternative_purchase` | `0.50 + 0.30 × soft_score` |
| `max_steps` | `0.00` |
| `reward_unverifiable` | `0.00`，但 `reward_valid=false` |
| `early_abstain` | `-0.40` |
| `repeat_loop` | `-0.60` |
| `assistant_final` | `-0.80` |
| `guard_rejection` | `-0.80` |
| `wrong_purchase` | `-1.00` |

所有有效终局从第 16 个已执行工具步骤起叠加分段递增的累计步数惩罚，45 步累计 `-1.06`。
Reward 异常与基础设施无效轨迹不伪装成普通模型失败，也不参与正常 GRPO 更新。

## Harness 边界

- 三候选收敛提醒、35/40 步提醒和循环提醒只改变模型可见 Prompt，不直接加减 Reward；
- 候选记忆不保存匹配分、Hard/Soft 结果或 Gold 身份；
- Action Guard 只判断动作是否合法，不判断商品语义；
- `finish_without_purchase` 统一进入 `early_abstain`，没有单独的充分检索成功类型；
- Environment 的 6 步无实质进展和 45 步上限保持独立，不能被 Prompt 提醒替代。

## 逐约束审计

`reward_detail` 保存 `constraint_results`、Hard gate、Soft score、证据覆盖率、步数惩罚和最终分类。
GRPO Adapter 只保留训练所需的公共最小字段；Final-240 的确定性面板、冻结 Rubric 和 Blind Judge
并列展示，互不覆盖。

## 源码入口

- Reward：`environments/ShopSimulator/shop_env/web_agent_site/engine/reward.py`
- 约束与语义：`environments/ShopSimulator/shop_env/web_agent_site/engine/constraints.py`
- Reward Features：`environments/ShopSimulator/shop_env/web_agent_site/engine/reward_features.py`
- GRPO 校验与最小化：`src/shopping_grpo/training/grpo/adapter/runtime.py`
- 详细面试文档：`3.grpo阶段数据清洗及相关曲线/Reward-v4完整解析与面试问答.md`
