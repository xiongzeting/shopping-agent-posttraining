# Observation v2

当前标识：`shopping-observation-v2`；候选记忆为 `shopping-candidate-memory-v2`。

Observation v2 由环境先输出结构化公开状态，再由统一 renderer 渲染和投影给模型。支持的页面类型为：

```text
search_home
search_results
product_detail
information_subpage
terminal
```

## 当前页面预算

| 内容区域 | Token Budget | 关键保留内容 |
|---|---:|---|
| 搜索结果页 | 2,560 | 当前页最多 20 个商品的记录边界、ASIN、价格、品牌、品类、属性和完整 footer |
| 商品详情页 | 3,072 | 当前 variant 价格、关键属性、已选规格、可选规格和可执行按钮 |
| 普通页/信息子页 | 512 | 页面正文、返回路径、搜索状态和完整按钮 |
| 候选选择/候选记忆 | 2,048（独立附加） | Final-240 中稳定的 C1-C4 摘要，扩容保留标题、规格与公开证据，不挤占页面正文预算 |

模型可见顺序为：

```text
Observation 版本与页面类型
→ 可选的步数/循环提醒
→ 当前页面正文
→ 可选的候选记忆
→ 搜索状态与可点击按钮 footer
```

非空的 `features`、`attributes` 等公开证据直接并入商品详情，不再要求模型进入单独信息子页。
`goal`、`reward`、`reward_detail`、`target_asin`、`answer` 和隐藏候选判断等字段禁止渲染。

## 候选记忆

当前 Final-240 Evaluation 每条轨迹保存最先核验的 4 个商品，使用稳定编号 C1-C4，记录公开的
ASIN、标题、品牌、品类、当前价格、已选规格、简要证据及原搜索词、页码和 rank。达到 4 个后，
后续商品仍可正常核验，但不再写入或替换已有候选；同一商品重访只更新原记录。

候选记忆不等于普通阶段的可点击列表，也不包含满足度、推荐分、Gold 或 Reward。普通探索阶段不会向
模型投影候选记忆。连续6步无实质进展，或执行达到30步仍未结束时，Harness 切换候选专用 Prompt，
按 C1→C4 自动逐个 `reopen`；每次 Observation 只包含当前候选详情，候选之间硬重置上下文。

## 提醒与不变量

- 连续 3 步无实质进展时显示循环高风险提醒；35 步后进入收敛阶段，不再叠加循环提醒；
- 已执行 35 步后显示步数提醒，40 步后语气进一步加强；
- 提醒只影响下一轮模型上下文，不改变环境状态和 Reward；
- 搜索页原始 ASIN 集合、模型可见 ASIN 集合和 Guard 允许打开集合必须一致；
- 非搜索页的可执行按钮不能因投影而丢失或新增；
- 整页无法在预算内安全表示时抛出投影错误，并将轨迹标记为基础设施无效。

因此，投影只能压缩长字段或正文，不能静默删除商品、按钮、价格轴或规格动作目标。

## 训练与评测边界

C1-C4 和三候选动态 System Prompt 是当前 Final-240 Evaluation 扩展。GRPO Adapter 共享 renderer，
但 `runtime.py` 当前候选记忆默认上限仍为 6，且没有 Evaluation 的动态 System Prompt 前置逻辑；
两入口不能被描述为已经完全一致。

## 源码入口

- 环境公共状态：`environments/ShopSimulator/shop_env/web_agent_site/engine/observation.py`
- Renderer 与提醒：`src/shopping_grpo/environment/observation.py`
- 候选记忆：`src/shopping_grpo/environment/candidate_memory.py`
- 投影器：`src/shopping_grpo/environment/projection.py`
- Evaluation 动态 Prompt：`src/shopping_grpo/evaluation/rollout.py`
- 配置：`configs/agent_loop.yaml`
