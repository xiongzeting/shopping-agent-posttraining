# Shopping Agent Harness

Harness 是模型与 ShopSimulator 之间的执行层。它负责组织 Prompt、工具调用、
Observation、动作守卫、上下文预算和轨迹记录；它不替代 Search、Reward 或环境终止逻辑。

## 执行流程

```text
用户任务
→ System Prompt 与 Tool Schema
→ 模型每回合选择一个工具
→ 本地 Action Guard 检查当前页面合法性
→ ShopSimulator 执行动作
→ Observation v2 投影
→ 记录轨迹、终局和基础设施事件
```

评测实现位于 `src/shopping_grpo/evaluation/rollout.py`；GRPO 使用
`src/shopping_grpo/training/grpo/adapter/` 下的 AgentLoop 与工具适配器。

## Prompt 当前规则

当前 System Prompt 固定七类规则：

1. 只依据最新页面执行动作，历史页面只用于比较；
2. 遵守信息子页返回规则和严格工具参数；
3. 使用简洁、可区分且有实质变化的搜索词；
4. 按“品类 > 预算 > 品牌 > 型号与核心功能 > 规格”比较；
5. 购买前核验证据、补齐规格轴并检查最终 variant 价格；
6. 充分探索仍无合适候选时才主动结束；
7. 避免循环、非法动作和无进展的 `think` 调用。

Prompt 是策略约束，不是硬执行保证。它不得使用 Final-240 结果反复调优，否则会污染
最终验收集。

## Observation 预算

| 页面 | 默认预算 | 规则 |
|---|---:|---|
| 搜索结果 | 2560 tokens | 最多保留当前页 20 个完整商品记录 |
| 商品详情 | 3072 tokens | 保留价格、属性、已选规格和可选规格 |
| 搜索首页、信息子页、终局与普通页 | 512 tokens | 保留状态与完整动作 footer |
| 候选记忆 | 独立附加 1024 tokens | 最近 6 个候选摘要，不挤占页面正文 |

搜索页不会为了截断只保留前 10 个商品。模型可见 ASIN 集合必须与当前可打开 ASIN
集合一致；无法在预算内安全表示整页时应记为基础设施异常，而不是静默漏掉候选。

## Action Guard 的职责边界

Action Guard 在请求到达环境前拒绝：

- Schema 未声明的额外参数；
- 未知工具、缺少必填参数、参数类型错误和必填空字符串；
- 当前页面不可搜索时调用搜索；
- 打开最新 Observation 中不存在的 ASIN；
- 点击当前页面未列出的按钮或规格；
- 把导航按钮当作商品规格；
- 非法的 `finish_without_purchase` reason。

评测侧连续 3 次被 Guard 拒绝会以动作守卫循环结束。评测入口会把当前可打开 ASIN、按钮和
恢复路径反馈给模型；GRPO Tool Adapter 当前返回简短拒绝原因。两条路径都会记录拒绝原因，
但恢复消息的详细程度仍需统一。

Guard 不判断商品类目是否正确、价格是否超预算、功能证据是否充分或候选是否最优。
这些属于 Prompt 引导、Environment Reward 和离线评测的职责。当前
`finish_without_purchase` 也只在 Guard 中检查参数；是否达到充分探索资格由环境判断。

## 上下文管理

正式评测的上下文窗口、单回合生成上限、安全余量和是否压缩由评测启动参数决定；当前 CLI
默认是 24576 token context window、512 token 生成上限、512 token 安全余量，并关闭压缩。

GRPO 当前使用 30000 token context window、768 token 单回合生成预留和 512 token 安全余量，
输入预算固定为 28720 tokens，并启用确定性完整交互组压缩。两套入口都记录上下文 token、
Observation 投影和压缩事件，不能把上下文或推理服务故障计成模型零分。

`configs/grpo.yaml` 还将 veRL 的 user/assistant turn 上限设为 40，而 Harness `max_steps` 是 45；
需要通过 AgentLoop 行为测试确认实际先触发的边界，并统一公开合同。

## 当前不包含

- 用户澄清或多轮需求修改；
- 多商品购物车；
- 无解题的结构化推荐终局；
- 基于 Final-240 结果自动改 Prompt；
- 由 Action Guard 直接判断语义正确性。

因此 Harness 当前特色是“可见状态、可执行动作和可审计轨迹的一致性”，不是扩展新的
购物任务类型。
