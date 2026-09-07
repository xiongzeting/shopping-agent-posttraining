# 长程购物 Agent 后训练项目：完整流程、工程讲解与面试问答

> 更新时间：2026 年 8 月 26 日（Asia/Shanghai）
>
> 当前完整链路：**Qwen3.5-2B Base → Final-1000 SFT-v5 → GRPO step50/100 → Harness改善版 GRPO step230 → Final-240 六模型评测**
>
> 同协议 **Qwen3.8-27B** 作为独立 Actor 对照，不属于 SFT→GRPO230 训练迁移。
>
> 运行合同：Environment v2.4 / Reward v4 / Termination v3.1 / Observation v2 / Search v2.1 / Tool Schema v2

本文面向两个用途：

1. 完整解释项目从环境、Agent Harness、Teacher 数据、SFT、Reward、GRPO 到评测的真实实现；
2. 将项目事实转换为面试中能够清晰说明的回答，并明确哪些内容不能夸大。

本文只描述当前正式主线，不把旧 Environment、旧 Final-200、旧 SFT 或历史模型混入当前结论。

---

## 0. 一页结论

### 0.1 项目要解决什么问题

项目基于淘宝真实商品快照构建长程购物 Agent。用户给出一条自然语言需求后，模型需要自主
完成：

```text
理解用户约束
    ↓
构造和改写搜索词
    ↓
浏览搜索结果并维护候选
    ↓
打开商品并核验详情证据
    ↓
选择颜色、尺码、容量、套餐等规格
    ↓
确认完整 variant 的最终价格
    ↓
购买目标商品，或在满足停止资格后结束
```

这不是一次文本分类或单轮检索，而是一个带状态、工具、长期信用分配和终局约束的序列决策
问题。

### 0.2 我完成了哪些工作

项目包含五个相互约束的部分：

1. **Agent Harness 与长上下文工程**：把模型和 ShopSimulator 连接成稳定、可审计的多轮执行循环；
2. **Teacher 数据与 SFT**：用 1,000 条严格成功、行为可验证的 Action-only 轨迹教授工具协议；
3. **Reward v4**：从公开 Query 解析可评分 Hard 与 Soft，按当前完整 variant 核验品类、价格、品牌、型号、功能和规格，并区分 Gold、有效替代、部分满足、主动停止、Guard拒绝、循环和错误购买等终局；
4. **GRPO**：使用 Dr.GRPO 的无组内标准差归一化思路和 GAPO 风格的动态采样，优化在线决策；
5. **Final-240 评测**：通过确定性代码、冻结 Rubric 和盲 Trajectory Judge 分析结果与过程。

### 0.3 最终结果

| 指标 | Base | SFT | GRPO50 | GRPO100 | GRPO230·Harness | Qwen3.8-27B |
|---|---:|---:|---:|---:|---:|---:|
| 严格 Gold 成功 | 0/240 | 142/240 | 154/240 | 153/240 | 163/240 | 146/240 |
| Gold＋Valid 购买成功 | 0/240 | 164/240 | 167/240 | 173/240 | 180/240 | 172/240 |
| Mean Final Reward | -0.7766 | 0.3368 | 0.3766 | 0.4238 | 0.5131 | 0.4547 |
| Reward Valid | 100% | 100% | 100% | 100% | 100% | 100% |
| Guard 拒绝次数 | 969 | 82 | 56 | 41 | 25 | 40 |
| 重复搜索次数 | 71 | 49 | 40 | 26 | 45 | 0 |

严格成功要求同时满足：

- 完整 `gold_purchase`；
- `reward_valid=true`；
- 合法环境终局；
- 严格目标规格与证据完整。

有效替代商品可以获得训练 Reward，但不计入这里的严格 Gold 成功率。

### 0.4 完整系统架构

```text
用户 Query
   │
   ▼
Shopping Agent Harness
   ├── System Prompt / Tool Schema
   ├── Session 与页面状态
   ├── Action Guard
   ├── Observation 结构化投影
   ├── Context Budget
   └── Termination / Runtime Telemetry
   │
   ▼
Qwen3.5-2B Actor
   │ Tool Call
   ▼
ShopSimulator Environment v2.4
   ├── Search v2.1
   ├── Observation v2
   ├── Reward v4
   └── Termination v3.1
   │
   ├──────── Teacher Rollout → Data Gate → Action-only SFT
   │
   ├──────── Online Rollout → Dynamic Sampling → GRPO
   │
   └──────── Deterministic Checks + Rubric + Blind Judge → Final-240
```

---

# 一、项目描述

## 1.1 项目背景

通用小模型具备语言理解能力，但不天然掌握购物环境中的工具协议和页面状态机。例如模型可能：

- 点击上一页出现、当前页已经不可见的商品；
- 在商品详情子页直接发起搜索；
- 将 Description 文本误当作可以选择的规格；
- 找到目标商品但没有选择完整 variant；
- 只输出“建议购买”，却没有调用 `buy_now`；
- 在相同搜索词和相同页面之间循环；
- 根据商品基础价格购买，却忽略规格选择后的实际价格。

因此，本项目的目标不是简单让模型“推荐一个相关商品”，而是让它在冻结协议下完成可执行、
可验证、可复现的长程购物任务。

## 1.2 为什么购物任务适合研究长程 Agent

购物任务同时包含以下能力：

| 能力 | 在购物任务中的体现 |
|---|---|
| Query 理解 | 区分类目、预算、品牌、型号、功能、颜色和数量 |
| 检索策略 | 选择有区分度的关键词，并根据结果做实质改写 |
| 状态维护 | 记住已经看过的候选，同时只操作当前页面合法元素 |
| 证据核验 | 从商品详情中合并展示的关键属性、Features、规格和价格证据中寻找依据 |
| 组合决策 | 选择多个规格轴并确认完整 variant 价格 |
| 长程规划 | 决定继续搜索、比较候选、返回页面还是购买 |
| 终止策略 | 避免过早放弃、过早购买和无效循环 |

任务最多允许 45 个工具动作。SFT 和 GRPO 模型在正式评测中平均执行约 16 个工具步骤，已经
明显不同于单轮 Function Calling。

## 1.3 当前冻结合同

```text
Environment v2.4
├── Reward v4
├── Reward Features v2
├── Query Constraints v1
├── Termination v3.1
├── Observation v2
├── Search v2.1
└── Tool Schema v2
```

冻结合同的意义是：Teacher、SFT、GRPO 和 Final-240 必须运行在同一套搜索、页面、Reward、
终止和工具协议上。如果训练阶段与评测阶段使用不同搜索或 Reward，最终提升就无法解释。

主要合同文档：

- `版本内/Environment-v2.4.md`
- `版本内/Reward-v3.2.md`
- `版本内/Termination-v3.1.md`
- `版本内/Observation-v2.md`
- `版本内/Search-v2.1.md`
- `版本内/Tool-Schema-v2.md`

## 1.4 数据隔离原则

项目数据按用途分层：

```text
ShopSimulator 官方商品与任务
├── Final-240：冻结测试集
├── Final-1000：Teacher / SFT
├── GRPO Dev Probe：开发和诊断
└── GRPO Training Probe：在线训练任务候选池
```

当前正式合同中：

- Final-240 与 SFT / Teacher task ID 重叠为 0；
- Final-240 与 GRPO task ID 重叠为 0；
- GRPO 与 Final-1000 task ID 重叠为 0；
- GRPO train 与 validation task ID 重叠为 0；
- Final-240 不用于修改 Prompt、Reward、Harness、训练数据或选择 checkpoint。

## 1.5 可用于简历的项目描述

> 基于淘宝真实商品快照环境，搭建 Baseline → SFT → GRPO → Evaluation 端到端后训练流程，
> 使 Qwen3.5-2B Agent 能够根据自然语言需求完成商品搜索、候选比较、详情核验、规格选择和购买。
> 在固定且与训练数据零重叠的 Final-240 上，严格 Gold 成功率由 Base 的 0% 提升至 SFT 的
> 59.17%，Harness 改善版 GRPO230 达到 67.92%；Gold＋Valid 购买成功由 68.33% 提升至
> 75.00%，平均 Final Reward 由 0.3368 提升至 0.5131。该结果是 checkpoint 与 Harness 的
> 完整系统结果，不包装为纯训练消融。

## 1.6 面试中的 60 秒项目介绍

> 我做的是一个基于真实商品快照的长程购物 Agent 后训练项目，基座是 Qwen3.5-2B。完整流程
> 包括 Agent Harness、Teacher 数据、Action-only LoRA SFT、Reward v4、在线 GRPO 和冻结
> Final-240 评测。SFT 主要让模型学会稳定执行搜索、页面浏览、规格选择和购买协议；GRPO 再
> 优化候选选择、证据核验和终止策略。最终严格 Gold 成功率从 Base 的 0% 提升到 SFT 的
> 59.17%，Harness 改善版 GRPO230 达到 67.92%，Gold＋Valid 购买成功达到 75.00%，平均
> Final Reward 从 0.3368 提升到 0.5131。我比较强调的是，整个结果使用同一冻结环境、固定 240
> 题分母和训练零重叠协议，并且不仅统计成功率，还对轨迹过程和失败原因做了可追溯分析。

---

# 二、Agent Harness 与长上下文工程

## 2.1 Harness 到底是什么

Harness 是模型和 ShopSimulator 之间的执行控制层，可以理解为“总控程序”。

模型只负责根据当前上下文生成下一步 Tool Call；环境只负责执行动作并返回页面状态。Harness
负责让两者形成一个稳定的多轮闭环：

```text
创建环境 Session
    ↓
构造 System / User / Tool 上下文
    ↓
调用模型生成 Tool Call
    ↓
解析 Tool Call 和 JSON 参数
    ↓
执行 Schema 校验与 Action Guard
    ↓
将合法动作发送给 ShopSimulator
    ↓
接收原始 Observation
    ↓
结构化投影并更新上下文
    ↓
记录动作、Reward、Token、耗时和异常
    ↓
判断继续、购买、停止或自动终止
```

Harness 不替模型决定商品是否正确。它负责协议、状态、安全边界和记录；商品语义、证据是否
充分以及最终买什么仍属于模型策略。

## 2.2 工具集合

当前基础 Tool Schema 只公开 8 个工具：

- `search_products`
- `open_product`
- `select_option`
- `next_page`
- `prev_page`
- `back_to_search`
- `buy_now`
- `finish_without_purchase`

所有工具都设置 `additionalProperties=false`，无参数工具必须传严格的 `{}`，并且每个 Assistant
回合只允许执行一个串行工具调用。`view_description/view_features/view_reviews/view_attributes`、
`think` 与 `reopen_candidate` 已从当前公开工具中删除；非空的公开商品特征和属性直接合并进详情
Observation，候选记忆也不提供历史商品直达能力。

Final-240 Evaluation 每轮再取“基础 8 工具 × 最新页面真实可执行动作”的交集：搜索首页只暴露
搜索/结束，搜索结果只暴露打开商品、当前导航/结束，详情页只暴露未选规格、当前购买/返回/结束。
候选数、步数和循环提醒不会额外收缩工具集合。

相关实现：

- `src/shopping_grpo/environment/tools.py`
- `configs/tools.json`
- `src/shopping_grpo/training/grpo/adapter/tools.py`

## 2.3 Session 与页面状态

ShopSimulator 是有状态环境。Harness 需要维护：

- 当前 task ID；
- 环境租约和 Session；
- 当前页面类型；
- 当前页可见商品 ASIN；
- 当前可以点击的按钮；
- 当前商品和已选择规格；
- 已执行动作与历史 Observation；
- 已核验候选的轨迹内工作记忆；
- 当前工具步数和剩余预算；
- Reward、终止原因和基础设施状态。

如果每一步都把环境当成无状态 HTTP 请求，模型就可能操作已经失效的页面元素。相关代码：

- `src/shopping_grpo/environment/client.py`
- `src/shopping_grpo/training/grpo/adapter/session.py`
- `src/shopping_grpo/training/grpo/adapter/runtime.py`

## 2.4 Action Guard

Action Guard 负责拒绝确定性非法动作，例如：

- 打开当前页面不存在的 ASIN；
- 调用当前页面不存在的按钮；
- 选择 Observation 中不存在的规格值；
- 在信息子页执行不允许的搜索；
- 传入 Schema 之外的参数；
- 同时生成多个互相冲突的 Tool Call；
- 使用非法停止 reason。

Guard 被拒绝后，会将拒绝原因、当前合法目标和恢复方法重新反馈给模型。连续三次被 Guard 拒绝，Harness
会终止轨迹，避免无限纠错。

Guard 不负责：

- 判断商品语义是否满足用户要求；
- 判断哪个候选最优；
- 判断功能证据是否充分；
- 判断完整规格价格是否符合预算；
- 决定模型是否应该购买。

因此 Guard 是执行安全网，不是手写购物策略。相关实现：

- `src/shopping_grpo/environment/actions.py`
- `tests/test_action_validation.py`
- `tests/test_shop_agent_env.py`

## 2.5 Observation 投影是什么

这里的“投影”不是向量降维，而是**按照页面结构和 Token 预算，将原始页面转换成模型能够安全
消费的结构化精简页面**。

例如原始商品详情可能包含：

```text
商品标题、ASIN、基础价格、所有规格、详情长文本、卖点、数百条评论、
店铺信息、图片信息、当前规格、variant 价格、页面按钮……
```

投影后优先保留：

```text
ASIN
标题
关键价格
当前已选规格
完整可选规格轴和值
关键 Attributes / Features
当前可执行按钮
返回路径和搜索状态
```

当前预算为：

| 页面类型 | Token Budget |
|---|---:|
| 搜索结果页 | 2,560 |
| 商品详情页 | 3,072 |
| 普通页与信息子页 | 512 |
| 候选选择/候选记忆（独立附加） | 2,048 |

搜索页必须保留当前页全部 20 个商品的记录边界和 ASIN，只压缩长标题或低优先级字段。详情页
优先保留价格、规格、已选值和关键属性；非空 Features/Attributes 直接并入详情，不再依赖单独
信息工具。候选记忆使用独立预算，不挤占当前页面正文。

最重要的合同是：

> 模型可见的商品和按钮集合，必须与 Action Guard 允许操作的集合一致。

如果直接按字符截断，可能保留标题却删除 ASIN，或者保留规格值却删除规格轴名称，从而造成
“模型看得到、环境不允许操作”的状态不一致。

相关实现：

- `src/shopping_grpo/environment/observation.py`
- `src/shopping_grpo/environment/projection.py`
- `tests/test_observation_projection.py`
- `tests/test_structured_observation.py`

## 2.6 候选记忆与动态收敛提醒

当前 Final-240 Evaluation 为每条轨迹稳定保存最先核验的 4 个候选，编号为 C1-C4，记录公开的
ASIN、价格、品牌、品类、已选规格、简要证据和原搜索位置。满 4 个后不替换，重访同一商品只更新
原记录。编号不表示优劣或满足情况；普通探索阶段不向模型展示候选记忆，历史 ASIN 也不会因此变成
可点击目标。

连续6步无实质进展，或执行达到30步仍未结束时，Evaluation 会删除普通探索上下文并启用候选专用
System Prompt。Harness 按 C1→C4 自动逐个 `reopen`，每次模型只得到“候选专用 System Prompt＋
原始用户需求＋当前候选详情”，不显示候选列表，也不让模型调用 `open_product` 选候选。规格未闭合时
只允许 `select_option/back_to_search`，闭合后只允许 `buy_now/back_to_search`；返回拒绝后自动切换
并再次硬重置上下文。

该顺序候选、候选专用 Prompt、候选间硬重置和阶段 Tool Schema 已在 Evaluation 与 GRPO Adapter
两条入口对齐，并由差分合同测试覆盖。

## 2.7 长上下文管理

正式 Rollout 使用：

- 模型上下文窗口：30,000 Token；
- 单回合 Assistant 最大生成：768 Token；
- 最大工具动作：45；
- 工具返回最大长度：16,384 Token；
- 512 Token 安全余量；
- Actor 单卡 Token Budget：30,000；
- Actor / Reference Log-prob Budget：36,000。

上下文压缩不是简单删除最早消息。Tool Call 与 Tool Response 必须作为完整组保留，避免留下
无法解释的半个动作。系统优先保留：

1. System Prompt 和用户需求；
2. 最新页面；
3. 最近的完整 Assistant / Tool 交互组；
4. 最新候选记忆和规格证据；
5. 终局所需的页面状态。

相关实现：`src/shopping_grpo/environment/context.py`。

## 2.8 终止机制

Harness 和 Environment 共同处理终止：

- `gold_purchase` 等合法购买终局；
- `early_abstain`：主动停止且未购买；
- `repeat_loop`：重复动作或长期没有新证据；
- `max_steps`：达到 45 步；
- 连续 Guard 拒绝；
- 模型服务、协议或环境异常。

“有进展”不等于工具调用数量增加，而是是否获得了新页面、新候选、新规格、新价格或新的决策
证据。Evaluation 连续 3 步无实质进展时先显示循环高风险提醒，Environment 连续 6 步无进展才
终止；35 步后改用步数收敛提醒，40 步后进一步加强。提醒不改变 Reward 和自动终止阈值。

## 2.9 veRL 异步 AgentLoop 集成

GRPO 训练中，Harness 被接入 veRL 0.8 的异步多轮 AgentLoop：

- `ShoppingToolAgentLoop` 管理模型回合；
- 8 个 Agent worker 并发执行；
- vLLM 负责在线生成；
- 每个 Prompt 并行采样 4 条轨迹；
- 每条轨迹独立维护 ShopSimulator Session；
- 终局 Reward 和轨迹遥测写回训练 Batch。

主要代码：

- `src/shopping_grpo/training/grpo/adapter/agent_loop.py`
- `src/shopping_grpo/training/grpo/adapter/tools.py`
- `src/shopping_grpo/training/grpo/adapter/session.py`
- `configs/agent_loop.yaml`
- `configs/grpo.yaml`

## 2.10 Harness 解决了什么问题

Base 模型的 Final-240 结果说明 Harness 不能被忽略：

- Guard 拒绝：966 次；
- `illegal_action`：167 个任务；
- `guard_rejection`（连续三次非法动作终止）：187 个任务；
- Reward Valid：239 / 240（99.6%，其中模型失败终局仍是可训练的有效负样本）；
- 平均执行步骤：5.896。

Base 的轨迹短不是高效，而是大量任务没有真正进入购物流程。SFT 后平均步骤增加到 16.908，
但 Guard 拒绝降到 82，说明模型开始正确执行长程协议。

## 2.11 简历表述

> 实现训练与评测双入口的多轮 Shopping Agent Harness，以 System Prompt、页面级动态 Tool
> Schema 和 Action Guard 约束单工具串行执行；通过结构化 Observation 投影、候选记忆与完整
> 交互轮次裁剪，在保留全部可操作 ASIN、按钮和规格 ID 的前提下支持 30K 长上下文，并结合
> Loop/步数提醒、45 步终止及异常分层接入 veRL AgentLoop 与 vLLM Rollout。

## 2.12 面试中的 45 秒回答

> Harness 是模型和 ShopSimulator 之间的执行控制层。模型只负责生成下一步工具调用，Harness
> 负责维护 Session 和页面状态、按最新页面动态暴露工具、串行执行单个 Tool Call，并用 Guard
> 拦截过期 ASIN、规格和按钮。页面先做结构化投影，再用独立候选记忆保留 C1-C4 的公开证据；
> 三个候选后提示收敛，35/40 步提醒及时决策，历史按完整工具轮次压缩以支持 30K 上下文。
> Guard 不判断商品语义，因此 Harness 保证可执行性，买什么仍由模型决定。

---

# 三、Teacher 数据与 SFT

## 3.1 为什么需要 SFT

未经后训练的 Base 模型主要失败在协议层，而不是语言表达层：

- 不会稳定输出严格 Tool Schema；
- 不理解当前页面与历史页面的动作边界；
- 不会稳定选择完整 variant；
- 找到商品后不调用购买工具；
- 多次 Guard 拒绝后提前终止。

如果直接从 Base 开始 GRPO，同一个 Prompt 的多条 Rollout 很可能全部是非法或无效轨迹，
组内没有稳定的相对学习信号。因此先用 SFT 教模型“如何合法完成购物流程”，再用 GRPO 优化
策略质量。

## 3.2 Final-1000 数据组成

最终数据共 1,000 条：

| 类型 | 数量 | 作用 |
|---|---:|---|
| Stable Teacher | 600 | 教授稳定、完整的成功购物协议 |
| Loop Recovery | 160 | 从错误搜索或循环风险中恢复并重新收敛 |
| Near-miss Rejection | 101 | 拒绝看似相近但违反核心要求的商品 |
| Option Grounding | 99 | 选择页面真实存在且满足要求的规格 |
| Terminal Tool Commit | 40 | 完成分析后显式调用 `buy_now` |

Corrective Teacher 合计 400 条。

这套设计不是只教模型“成功轨迹长什么样”，而是针对已有模型暴露出的失败模式提供纠错示范。

## 3.3 数据来源与候选池

数据来源包含：

- 旧版 500 条 Teacher 的重新审计；
- 本地与远端双机采集的 Loop / Near 轨迹；
- Option Grounding 补充轨迹；
- Terminal Tool Commit 补充轨迹；
- 通过新合同的 Clean Reuse。

旧数据没有历史豁免。最终原样保留旧轨迹 405 条；未通过新数据门的旧样本被替换，而不是因为
“以前用过”就继续保留。

候选审计规模包括：

| 类型 | Raw 候选 | 最终进入数据集 |
|---|---:|---:|
| Stable | 4,762 | 600 |
| Loop Recovery | 1,886 | 160 |
| Near-miss Rejection | 1,746 | 101 |
| Option Grounding | 274 | 99 |
| Terminal Tool Commit | 40 | 40 |

这些 Raw 统计来自多来源合并审计，不应简化成“Teacher 恰好采集了 5,000 条”。

## 3.4 三层数据门

### 第一层：结果门

所有 SFT 样本必须满足：

- `reward_valid=true`；
- Reward Type 为 `gold_purchase`；
- 目标 ASIN 正确；
- 严格目标规格正确；
- 完整 variant 价格语义成立。

### 第二层：过程与关键动作门

- Tool Call 必须可以解析和重放；
- 动作必须基于当前 Observation；
- 规格值必须真实存在；
- Assistant / Tool 消息必须成对；
- 终局动作必须完整；
- 可证明不改变后续状态的 Guard 恢复重复，可以清洗对应消息对；
- 会污染环境状态或无法解释的非法动作直接拒绝。

### 第三层：行为与语义门

- Loop Recovery 必须真实出现循环风险并成功恢复；
- Near-miss 必须指出具体不满足的约束，而不是无证据拒绝；
- Option Grounding 必须发生真实规格选择；
- Terminal Commit 必须调用 `buy_now`；
- 已经完成决策后继续无意义搜索的轨迹被剔除。

主要代码：

- `src/shopping_grpo/collection/sft.py`
- `src/shopping_grpo/collection/data_gate.py`
- `scripts/audit_teacher_data_gate.py`

## 3.5 数据去重与双机合并

项目检查：

- task ID 重复；
- trajectory ID 重复；
- 本地和远端重复任务；
- 商品 ASIN / family 重叠；
- Query 和标题近重复；
- 动作模板过度集中。

本地和远端 Near 数据中确认有两个严格有效的重复 task，最终固定由远端版本优先，确保最终
1,000 条 task ID 唯一。

## 3.6 防止动作模板坍缩

早期 Teacher 提示词过度强调固定流程，导致大量轨迹集中在相似的 8 步模板。即使最终买对，
这种数据也可能让模型把“固定动作序列”误认为唯一正确策略。

因此数据审计不只看平均步数，而是同时检查：

- 完整动作序列 Top-1 / Top-5 占比；
- 局部动作片段；
- Search Rank × Length 联合分布；
- 搜索改写、候选比较、Guard 恢复和多规格覆盖。

最终结果：

- 唯一完整动作序列：452 种；
- Top-1 序列：7.5%；
- Top-5 序列合计：19.5%；
- 轨迹范围：4–42 步；
- Short / Medium / Long：47.7% / 40.0% / 12.3%。

Long 轨迹来自真实长链路，不通过填充无意义动作制造。

## 3.7 检索和行为覆盖

Gold Rank 分布：

| Gold Rank | 数量 | 占比 |
|---|---:|---:|
| Rank 1 | 501 | 50.1% |
| Rank 2–5 | 299 | 29.9% |
| Rank 6–20 | 100 | 10.0% |
| Rank 21–150 | 70 | 7.0% |
| Missing | 30 | 3.0% |

行为覆盖：

| 行为 | 覆盖率 |
|---|---:|
| Evidence Verification | 93.2% |
| Search Reformulation / Loop Recovery | 59.6% |
| Guard Recovery | 50.1% |
| Candidate Comparison | 47.4% |
| Multiple Options | 41.0% |
| Variant Selection | 100% |
| Explicit Terminal Buy | 100% |

## 3.8 30K Token 审计

Token 统计使用 Qwen3.5-2B 的真实 Processor 和 Chat Template，不使用字符数估算。

初次审计发现 13 条超过 30K 的轨迹。这些样本被整条删除后再回填，而不是从中间截断，因为
截断可能：

- 删除 Tool Response；
- 破坏 Assistant / Tool 配对；
- 删除规格选择证据；
- 删除严格购买终局；
- 制造无法重放的半条轨迹。

最终 900 train 和 100 validation 全部低于 30K：

| Split | Input Mean | P95 | Max |
|---|---:|---:|---:|
| Train | 11,930.78 | 22,253 | 29,533 |
| Validation | 11,951.73 | 21,244 | 29,994 |

## 3.9 Action-only SFT

训练样本保留：

- System Prompt；
- User Query；
- Assistant Tool Call；
- 环境 Observation；
- 终局动作。

但 Loss 只计算 Assistant 动作 Token：

```text
System / User / Tool Observation → label = -100
Assistant Tool Call / Terminal Action → 正常监督标签
```

这样做的目的：

1. 重点学习可执行的动作协议；
2. 不监督环境返回的长文本；
3. 不蒸馏 Teacher 私有 reasoning；
4. 不让隐藏 Gold 信息进入监督目标；
5. 减少 Observation 长文本对梯度的干扰。

相关实现：`src/shopping_grpo/training/sft/dataset.py`。

## 3.10 正式 SFT 配置

| 参数 | 配置 |
|---|---:|
| Base | Qwen3.5-2B fresh base |
| Train / Validation | 900 / 100 |
| Max Length | 30,000 |
| Epoch | 3 |
| Batch Size | 1 |
| Gradient Accumulation | 8 |
| Learning Rate | `1e-4` |
| Warmup | 3% |
| LoRA Rank / Alpha / Dropout | 16 / 32 / 0.05 |
| Dtype | BF16 |
| Attention | Flash Attention 2 |
| Liger | 开启 |
| Gradient Checkpointing | 关闭 |
| QLoRA | 关闭 |
| Loss | Assistant Action-only |

LoRA 可训练参数为 16,819,200，占总参数的 0.7542%。

## 3.11 实际训练过程

正式训练：

- GPU：单张 RTX PRO 6000 Blackwell 94.97 GiB；
- Optimizer Step：339；
- 训练时间：约 43.8 分钟；
- 峰值显存：89.64 GiB；
- 最终 Train Loss：0.302715；
- 三个 Epoch 的 Eval Loss：0.320905 → 0.302517 → 0.298500。

验证损失没有回升。训练曲线位于：

- `重点/训练过程曲线/SFT-Final1000/01-sft-training-loss.png`
- `重点/训练过程曲线/SFT-Final1000/02-sft-validation-loss.png`
- `重点/训练过程曲线/SFT-Final1000/03-sft-learning-rate.png`
- `重点/训练过程曲线/SFT-Final1000/04-sft-gradient-norm.png`

![SFT Final-1000 流程](训练流程图/01-sft-v5-final1000-process.png)

## 3.12 SFT 带来了什么

Final-240：

- 严格成功：0% → 57.5%；
- Gold＋Valid购买成功：0% → 63.33%；
- `guard_rejection`终局：187 → 7；
- Guard Rejections：966 → 82；
- `illegal_action`：167 → 5；
- Decision Quality：0.042 → 1.458；
- Termination Efficiency：0.017 → 1.325。

SFT 的核心作用是让模型从“不会稳定执行环境协议”进入“能够完成长程购物流程”。它带来了整个
项目中最大的能力跃迁。

## 3.13 SFT 的剩余问题

SFT 仍有 103 个失败任务，主要错误已经从协议错误转向策略问题：

- `critical_evidence_missing=20`；
- `candidate_comparison_insufficient=13`；
- `overexploration_after_convergence=13`；
- `premature_purchase=10`。

这构成了 GRPO 的优化空间。

## 3.14 简历表述

> 针对循环搜索、近似商品误购、规格选择不准确和未调用终局工具等 Bad Case，设计 Stable
> Teacher 与 Corrective Teacher 数据策略，并建立结果、关键动作和行为语义三层数据门。
> 最终构建 1,000 条严格成功轨迹（600 Stable + 400 Corrective），采用 Assistant
> Action-only Loss 和 LoRA 完成 30K 长上下文 SFT，使严格成功率由 0% 提升至 57.5%。

## 3.15 面试中的 45 秒回答

> Teacher 数据由 600 条 Stable 和 400 条 Corrective 组成。Corrective 专门覆盖循环恢复、
> 近似商品拒绝、规格落地和终局工具提交。我没有只按最终买对筛数据，而是设计结果门、关键
> 动作门和行为语义门，防止非法动作或碰巧成功的轨迹进入 SFT。训练采用 Action-only Loss，
> 只监督 Assistant 工具动作，不监督 Observation 和 Teacher 私有 reasoning。最终 1,000 条
> 数据让 Base 的严格成功率从 0% 提升到 57.5%，主要完成了工具协议和基本长程行为的学习。

---

# 四、Reward 与 GRPO

## 4.1 Reward v4 的设计目标

原始购物 Reward 主要评估 Category、Attributes、Options 和 Price，但对于在线强化学习仍有两个
问题：

1. 唯一 Gold 对训练信号过于僵硬，现实中可能存在满足全部用户要求的替代商品；
2. 不同失败严重程度不同，类目买错、规格选错和充分探索后放弃不应得到相同奖励。

Reward v4 因此将购买结果拆成可审计的终局档位。

## 4.2 Query Hard/Soft 与可评分约束

品类始终属于 Hard。除此之外，Reward 根据用户原话将“必须、一定、绝对不要、不超过”等未被
软化的品牌、型号、核心功能、规格和价格要求解析为 Hard；“最好、优先、尽量、左右、大约、预算、
不要太”等进入 Soft。未被 Query 支撑的 Gold 私有属性不参与评分，语义明确但无法可靠比较的要求
保留为 `audit_only`。

任一可评分 Hard 明确失败，购买终局为 `wrong_purchase`；在没有已知 Hard 失败时，若某个可评分
Hard 无法核验，则终局为 `reward_unverifiable`：

```text
reward = 0.0
reward_valid = false
sampling_invalid = true
```

这里的 0 分不是中性奖励，而是表示轨迹不能进入有效训练采样。

## 4.3 生效偏好维度

Hard 全满足后，再对 Query 中生效的 Soft 和匹配维度评估：

| 维度 | 基础权重 |
|---|---:|
| 品牌 | 0.25 |
| 型号 | 0.25 |
| 核心功能 | 0.25 |
| 关键规格 | 0.25 |

只对 Query 真正生效的维度归一化。用户没有要求品牌时，品牌不会凭空参与扣分。

## 4.4 终局奖励

| 终局 | Reward |
|---|---:|
| Gold ASIN、目标规格与全部要求满足 | `1.00` |
| 非 Gold ASIN，但全部有效要求满足 | `0.80` |
| Hard 全满足，但替代商品至少一个可核验 Soft 失败 | `0.50 + 0.30 × soft_score` |
| 主动停止且未购买 | `-0.40` |
| 达到 45 步 | `0.00`，另计步数惩罚 |
| 重复循环或长期无进展 | `-0.60`，另计步数惩罚 |
| 连续三次非法动作被Guard拒绝 | `-0.80`，另计步数惩罚 |
| Assistant未调用工具而终止 | `-0.80`，另计步数惩罚 |
| 任一可评分 Hard 约束失败 | `-1.00`，另计步数惩罚 |
| Reward 无法验证 | `0.00`，但样本无效 |

步数惩罚从第16步开始，按16–20、21–25直到41–45分段累计；因此表中的基础终局Reward与最终
`Final Reward`可能不同。

严格评测仍然只认 `gold_purchase`，不会把 `valid_alternative_purchase` 纳入当前 GRPO230
的 67.92% 严格成功率。

## 4.5 Reward 的可审计性

Rollout 前，环境将任务要求编译成 Query Constraint Contract。每条约束记录：

- 约束类型和字段路径；
- Hard Gate / Matching Dimension / Strict Target Variant 角色；
- 比较操作符和期望值；
- 数据来源；
- Query 原文证据。

终局 `reward_detail` 逐条输出：

- `pass` / `fail` / `unverifiable`；
- 实际值来源；
- comparator；
- 判断证据。

这样可以解释 Reward 为什么得到某个分数，而不是只保留不可追溯的标量。

## 4.6 Reward 的限制

Reward v4 仍然是 Episode 结束后给出的序列级 Reward。轨迹前 15 步正确、最后规格选错时，
整个序列仍共享同一个 Advantage。项目尚未使用 Step-level Reward 或 PRM。

## 4.7 GRPO 为什么放在 SFT 之后

SFT 解决的是“会不会执行协议”，GRPO 解决的是“执行策略是否更优”。

如果 Base 的 4 条 Rollout 全是非法动作或无效终局：

- 组内没有可靠的相对优势；
- Reward 可能全部相同；
- 大量计算消耗在协议错误上；
- 模型很难从稀疏终局信号学会完整工具状态机。

因此先用 SFT 建立可执行策略，再用在线 Rollout 优化决策。

## 4.8 GRPO Probe 与任务池版本

### step 1-50：两阶段在线 Probe

候选任务按两阶段执行 `n=4` Probe：

| 阶段 | 候选任务 | 在线准入 |
|---|---:|---:|
| Calibration | 200 | 65 |
| Remaining | 600 | 186 |
| 合计 | 800 | 251 |

最终从 251 个可用任务中，保留组内 Purchase Success 数为 1/4、2/4 或 3/4 的任务。这里的
Purchase Success 与 Reward v4 一致，包含 `gold_purchase` 和 `valid_alternative_purchase`：

| 组内成功数 | 最终任务数 |
|---|---:|
| 1 / 4 | 45 |
| 2 / 4 | 51 |
| 3 / 4 | 85 |
| 合计 | 181 |

这些任务处于 SFT Policy 的可学习前沿：既至少出现一次成功，也不是四条全部成功。

### 初版 200 与最终 181 的关系

仓库和最终留痕包中保留过一版 200 任务训练集：

- 144 条 Frontier；
- 56 条 Hard Exploration。

后续将训练池收紧为 `all-1to3-clean` 的 181 条 clean frontier。最终正式 step-50 的真实日志从
step 1 到 step 50 均显示训练文件为：

```text
data/grpo/training-probe-v1/all-1to3-clean/train.parquet
```

因此：

- 可以说项目迭代过两版 GRPO 任务池；
- 可以说累计对 800 个候选任务进行两阶段 Probe；
- 不能说最终 step-50 同时混合使用了 200 和 181 两版任务；
- 最终模型的正式训练池口径是 181 条。

### step 50-100：全新候选池与早停筛选

续训前重新构建了与旧 GRPO 池、SFT 数据和 Final-240 均不重叠的 2,000 题候选池，难度分布
为 20% 简单、60% 中等、20% 困难。在线 `n=4` Probe 实际覆盖 709 个唯一任务并产生 2,827
条轨迹；达到目标后早停，筛出 100 个 clean frontier、共 400 条有效轨迹：

| 组内成功数 | 任务数 |
|---|---:|
| 1 / 4 | 44 |
| 2 / 4 | 56 |
| 合计 | 100 |

该阶段使用 `reward_tolerance=0.025` 过滤低差异组，并保持动态采样与有界重采样。最终 train 和
validation 各含 100 个唯一任务，与 Final-240、SFT 数据的 task_id 重叠均为 0。准确口径是
“从 2,000 个全新隔离候选中实际 Probe 709 个任务后早停筛出 100 个 clean frontier”，不能说
2,000 题全部完成了 Probe，也不能用这组数字覆盖前 50 step 的 181 题任务池。

## 4.9 实际 GRPO 配置

| 参数 | 配置 |
|---|---:|
| 起点模型 | SFT-v5 merged |
| Group Size | `n=4` |
| Temperature / Top-p | 0.7 / 0.9 |
| Train Batch Size | 2 |
| PPO Mini-batch | 2 |
| Agent Workers | 8 |
| Rollout | Async vLLM |
| Max Environment Steps | 45 |
| Max Model Context | 30,000 |
| Max Response Length | 25,904 |
| Actor / Ref Log-prob Budget | 36,000 |
| LoRA Rank / Alpha | 16 / 32 |
| Learning Rate | `1e-6` |
| PPO Clip | 0.2 |
| KL Reward / KL Loss | 关闭 |
| GRPO Std Normalization | 关闭 |

## 4.10 Dr.GRPO 思路

当前配置：

```yaml
algorithm:
  adv_estimator: grpo
  use_kl_in_reward: false
  norm_adv_by_std_in_grpo: false
```

项目借鉴 Dr.GRPO 的关键思路：**不使用组内 Reward 标准差归一化**。

标准 GRPO 常将组内中心化 Reward 再除以组内标准差。当组内差异很小时，标准差可能放大噪声；
不同问题的 Reward 尺度也可能被重新扭曲。关闭标准差归一化后，保留更直接的组内相对差异。

Actor 还使用：

```yaml
loss_agg_mode: seq-mean-token-mean
```

先按每条轨迹的 Response Token 求均值，再按序列求均值，避免长失败循环仅因为 Token 更多而
对更新产生更大权重。

准确表述应为“借鉴 Dr.GRPO 的无标准差归一化思路”，不应声称完全复现论文中的所有设置。

## 4.11 GAPO 风格动态采样

动态采样判断一个 Group 是否值得更新：

1. 至少存在一条成功购买；
2. Group 内 Terminal Utility Range 必须大于 0.025；
3. `reward_valid=false` 的轨迹不能提供正常训练信号；
4. 初始采样无效时，最多再重试两批，即最多三批；
5. 如果整个候选 Batch 没有可用 Group，则跳过本次 Actor Update 并读取下一批任务。

这借鉴了 GAPO 的动态采样思想：将在线算力集中在有组内差异的 Prompt 上，而不是对全成功、
全失败或 Reward 几乎一致的常数组做无效更新。

正式 step-1–49 的有效训练日志统计：

| 动态采样指标 | 数量 |
|---|---:|
| 生成 Group | 272 |
| 保留 Group | 67 |
| 丢弃 Group | 205 |
| 常数 / 低差异 Group | 139 |
| 无成功购买 Group | 19 |
| 无效 Group | 3 |

这些数字说明动态采样不是装饰性配置，而是在真实训练中大量过滤低信息量 Group。

## 4.12 为什么无 KL

本次训练没有加入：

- KL Reward；
- KL Loss；
- GRPO 标准差归一化；
- 按总轨迹长度施加的连续惩罚。

这样做是为了直接观察任务 Reward 对 SFT Policy 的优化效果。为控制更新幅度，采用：

- LoRA 而非全参数更新；
- `1e-6` 小学习率；
- PPO Clip 0.2；
- 先冻结 step-50，再以同一低学习率方案续训至 step-100；
- 最终使用独立 Final-240 验收。

训练日志仍记录 `actor/ppo_kl` 作为诊断指标，但它没有作为 KL Loss 加入目标函数。

## 4.13 实际训练过程与故障处理

训练基于 veRL 0.8，过程中处理了：

- 单 GPU Actor Cache 与 vLLM Wake-up 稳定性；
- 异步 Agent worker 异常；
- 空 Group 导致 Trainer 无法正常推进；
- 动态采样连续无有效 Group；
- step 15、25、30 等 checkpoint 的断点续训；
- LoRA checkpoint 到 HF merged 模型的导出与权重差验证。

实际保存 checkpoint：

```text
step 5 / 10 / 15 / 20 / 25 / 30 / 35 / 40 / 45 / 50
```

训练日志包含 48 条带 Actor Update 的记录：step 30 是断点边界，没有单独的 Actor 指标行；
最后一条 Actor 指标为 step 49，随后训练器保存 `global_step_50`。

## 4.14 如何阅读 GRPO 曲线

![GRPO step-50 流程](训练流程图/02-grpo-step50-process.png)

曲线目录：`重点/训练过程曲线/GRPO-step50/`。

主要观察原则：

- Actor Loss 围绕 0 正负波动是正常现象，不能要求像 SFT CE Loss 一样单调下降；
- Reward 和 Purchase Success Rate 受每步采样任务难度影响，会高频波动；
- PPO KL 只作为策略漂移诊断；
- Gradient Norm 用于检查更新是否爆炸；
- Dynamic Sampling 曲线用于判断有多少 Group 被过滤；
- Response Length 和 Step Time 用于分析长轨迹成本；
- 最终必须回到冻结评测，而不是仅根据在线 Reward 选择结论。

## 4.15 为什么同时报告 step-50 和 step-100

训练先完成并冻结 `global_step_50`，随后从该 checkpoint 续训至 `global_step_100`，两者都完成了
HF 权重合并和同协议 Final-240 评测。step-100 是更完整的训练阶段结果；step-50 作为中间
checkpoint 保留，用于判断继续训练带来的收益和回退。

当前 Reward v4 聚合口径下，Final-240 单次配对结果中 step-50 的严格 Gold 为 154/240，step-100
为 153/240；但 step-100
的平均 Final Reward 更高、循环和 Guard 拒绝更少。两者共同成功 136 题，分别独占 16 和 15
题，McNemar 精确检验 `p=1.0000`，不能据此声称 step-50 的总体泛化显著更强。

## 4.16 GRPO 与 Harness 改善带来了什么

step-50 与 step-100 是 checkpoint 演进的历史对比；当前相对 SFT 的完整系统结果使用
Harness 改善版 GRPO230：

- 严格成功：59.17% → 67.92%，净增 21 题；
- Gold＋Valid 购买成功：68.33% → 75.00%，净增 16 题；
- Mean Final Reward：0.3368 → 0.5131，相对提升约 52.3%；
- 平均执行步数：16.908 → 10.071；
- Action Attempts：17.250 → 10.175；
- Guard Rejections：82 → 25；
- Rubric satisfied：1,454 → 1,565；unknown：252 → 155。

SFT→GRPO230 的逐题迁移为失败→成功 42、成功→失败 20、共同成功 118、共同失败 60。
由于 GRPO230 同时采用改善后的 Harness，不能把全部增益只归因于新增训练步数。

## 4.17 简历表述

> 设计可审计的 Reward v4，以品类作为唯一 Hard Gate，并要求购买成功时当前完整 variant 的价格、品牌、型号、功能和规格等公开需求全部满足；区分 Gold、有效替代、
> 部分满足、合理停止、循环与错误购买等终局。从 2,000 个全新隔离候选中在线 Probe 709 个
> 任务，早停筛出 100 个 clean frontier；采用无 KL LoRA GRPO，借鉴
> Dr.GRPO 关闭组内 Reward 标准差归一化，并借鉴 GAPO 动态采样过滤低差异 Group。最终
> Harness 改善版 GRPO230 较 SFT 净增 21 个严格 Gold、16 个 Gold＋Valid 购买成功任务，
> 平均 Final Reward 由 0.3368 提升至 0.5131；同时明确该结果包含 Harness 改善，不冒充纯
> checkpoint 消融。

## 4.18 面试中的 60 秒回答

> Reward v4 以品类作为唯一 Hard Gate，再根据当前完整 variant 的价格、品牌、型号、功能和关键规格区分
> Gold、有效替代、部分满足、停止、循环和错误购买。前 50 step 使用两阶段 Probe 筛出的 181
> 个 clean frontier；续训前又从 2,000 个全新隔离候选中实际 Probe 709 题，早停筛出 100 个
> 1/4 或 2/4 成功组。训练采用无 KL LoRA GRPO，关闭组内 Reward 标准差归一化，并通过 GAPO 风格动态
> 采样过滤无成功购买或 Reward 差异不足的 Group。最终 Harness 改善版 GRPO230 将严格成功率
> 从 59.17% 提升到 67.92%，Gold＋Valid 购买成功达到 75.00%，平均 Final Reward 提升约
> 52.3%，并明显减少动作尝试和 Guard
> 拒绝。这个对比是完整系统收益，不只归因于训练步数。

---

# 五、评测与归因分析

## 5.1 为什么不能只看环境 Reward

同样是失败，原因可能完全不同：

- 一开始理解错品类；
- 搜索词过宽，目标没有被召回；
- 找到候选但没有打开；
- 打开商品但没有核验关键证据；
- 商品正确但规格选错；
- 规格正确但 variant 价格超预算；
- 一直搜索，最终循环或耗尽 45 步；
- 工具协议错误，根本没有进入有效购物过程。

如果只看最终 Reward，就无法判断 SFT 和 GRPO 到底改变了什么。

## 5.2 Final-240 结构

```text
Final-240
├── Core-180：9 个一级商品领域 × 20 题
└── Challenge-60：6 个困难切片 × 10 题
```

Challenge 切片：

1. `search_reformulation`
2. `candidate_comparison`
3. `price_semantics`
4. `multi_option`
5. `evidence_verification`
6. `long_horizon`

Final-240 在训练前冻结，任务 SHA-256：

```text
1668e0693e795e2f9f7fcfacc45659da601376b020c2487853daf7dce1b4c71f
```

## 5.3 六组 Rollout 的公平协议

Base、SFT、GRPO50、GRPO100、GRPO230 和 Qwen3.8-27B 使用同一评测任务、Rubric 与
Judge 合同；其中 Qwen3.8-27B 采用自己的同协议部署，GRPO230 使用 Harness 改善版配置：

- 同一批 240 个 task ID；
- 相同 Environment / Reward / Termination / Observation / Tool Schema；
- 每题一次确定性 Rollout；
- Temperature 0、Top-p 1；
- 相同 30K Context、45 步和单回合生成预算；
- 固定 240 题分母。

缺失任务和 `not_judged` 不从分母中删除。

## 5.4 评测不是一个 Judge，而是两阶段 LLM + 确定性代码

```text
第一阶段：Rubric Curator
    代码生成候选约束
        ↓
    DeepSeek V4 Flash 选择 Query 真正表达的要求
        ↓
    冻结逐题 Rubric

第二阶段：Trajectory Judge
    Query + 冻结 Rubric + Actor 可见轨迹 + 白名单指标
        ↓
    DeepSeek V4 Pro 五维评分、逐要求判断和错误归因

并行：代码确定性检查
    Gold / Reward / 终局 / Guard / 重复 / Token / 耗时 / 异常
```

## 5.5 Rubric Curator 如何工作

直接把 Gold 商品全部字段交给 Judge 会产生答案泄漏：目标商品拥有的属性不一定都是用户要求。

当前流程：

1. 代码从 Query、Instruction 和 TaskFacts 构造候选约束；
2. 每个候选记录字段、操作符、来源和 Query Span；
3. DeepSeek V4 Flash 只负责选择 Query 真正表达的候选；
4. Curator 将约束整理为 hard、soft 或 needs-review；
5. 240 题 Rubric 冻结；
6. Base、SFT、GRPO50、GRPO100、GRPO230 和 Qwen3.8-27B 共享完全相同的 Rubric。

Curator 不读取六组 Actor 轨迹，也不会根据最终 Reward 或失败样本反向修改 Rubric。240 题
共冻结 1,769 条要求；Rubric 状态统计以要求条目为单位，不以 240 题为分母。

Rubric 样例会明确记录：

- 商品品类；
- 用户要求的功能；
- 规格和值；
- 价格是硬上限还是“左右”软偏好；
- Query 中的原文证据。

## 5.6 确定性代码面板

代码直接计算：

- 严格 Gold 成功；
- Purchase Success；
- Reward Type / Reward Valid；
- 合法终局与终止原因；
- Guard Rejection；
- 非法动作；
- 重复 Canonical Action；
- 重复搜索；
- 工具执行步数和动作尝试数；
- Observation 投影压缩；
- Context 使用率和硬溢出；
- Provider Token；
- 模型、工具和端到端耗时；
- 基础设施无效任务。

这些指标不交给 LLM 猜测，适合做稳定回归测试。

## 5.7 Blind Trajectory Judge

DeepSeek V4 Pro Judge 看到：

- 用户 Query；
- 冻结 Rubric；
- Actor 实际可见的完整轨迹；
- Action Guard 拒绝事件；
- 白名单确定性行为指标。

Judge 看不到：

- Reward 数值和 Reward Type；
- Gold ASIN；
- 隐藏目标商品字段；
- 原始未投影 Observation；
- 其他模型的结果；
- `target_asin_match` 等结论性字段。

## 5.8 Judge 五个维度

每维独立按 0–2 分：

| 维度 | 核心问题 |
|---|---|
| Search Strategy | 搜索是否有效，改写是否有实质变化 |
| Candidate Utilization | 是否打开并利用高质量候选 |
| Evidence Verification | 是否核验品类、功能、规格和价格 |
| Decision Quality | 购买或停止是否有充分依据 |
| Termination Efficiency | 是否避免过早结束、过度探索和循环 |

Judge 同时对每条 Rubric 输出：

- `satisfied`
- `violated`
- `unknown`
- `not_applicable`

所有判断必须引用真实 Event ID，例如 `e0004`、`e0011`，便于回到具体搜索、商品打开、规格选择
或购买动作检查。

## 5.9 四个独立面板

最终评测不生成综合总分，而是保留四个独立面板：

| 面板 | 内容 |
|---|---|
| Reward 与终局 | Gold、购买结果、Reward、合法终止 |
| 用户需求满足 | Reward 逐约束结果与 Rubric 状态 |
| 轨迹过程质量 | Judge 五维评分、主要与次要错误 |
| 行为与资源效率 | 步数、Guard、重复、Token、耗时、上下文和异常 |

原因是一个加权总分容易掩盖失败模式。例如成功率可能提升，但循环、Token 或错误购买也可能
同时增加。

## 5.10 同题配对和分层分析

`comparison.json` 对同一 task ID 做配对迁移：

```text
Failure → Success
Success → Success
Success → Failure
Failure → Failure
```

同时按以下层级比较：

- Core / Challenge；
- 9 个商品领域；
- 6 个困难切片。

这比四个独立成功率更有解释力，因为可以直接找到 GRPO 新解决了哪些题，又让哪些 SFT 成功题
发生回退。

## 5.11 核心结果

| 指标 | Base | SFT | GRPO50 | GRPO100 | GRPO230·Harness | Qwen3.8-27B |
|---|---:|---:|---:|---:|---:|---:|
| 严格成功 | 0.00% | 59.17% | 64.17% | 63.75% | 67.92% | 60.83% |
| Gold＋Valid购买成功 | 0.00% | 68.33% | 69.58% | 72.08% | 75.00% | 71.67% |
| Mean Final Reward | -0.7766 | 0.3368 | 0.3766 | 0.4238 | 0.5131 | 0.4547 |
| Reward Valid | 99.58% | 100% | 100% | 100% | 99.58% | 100% |
| Judge Coverage | 99.2% | 100% | 100% | 100% | 99.58% | 100% |

Judge 五维：

| 维度 | Base | SFT | GRPO50 | GRPO100 | GRPO230·Harness | Qwen3.8-27B |
|---|---:|---:|---:|---:|---:|---:|
| Search Strategy | 0.945 | 1.529 | 1.596 | 1.587 | 1.632 | 1.517 |
| Candidate Utilization | 0.613 | 1.521 | 1.592 | 1.567 | 1.632 | 1.421 |
| Evidence Verification | 0.273 | 1.304 | 1.400 | 1.429 | 1.623 | 1.188 |
| Decision Quality | 0.042 | 1.458 | 1.525 | 1.550 | 1.577 | 1.396 |
| Termination Efficiency | 0.017 | 1.325 | 1.400 | 1.408 | 1.506 | 1.446 |

## 5.12 行为与效率结果

| 指标 | Base | SFT | GRPO50 | GRPO100 | GRPO230·Harness | Qwen3.8-27B |
|---|---:|---:|---:|---:|---:|---:|
| 平均工具步骤 | 5.896 | 16.908 | 15.808 | 15.738 | 10.071 | 5.383 |
| Action Attempts / Task | 9.921 | 17.250 | 16.042 | 15.908 | 10.175 | 5.550 |
| Guard Rejections | 966 | 82 | 56 | 41 | 25 | 40 |
| Duplicate Actions | 662 | 1,518 | 1,295 | 1,238 | 527 | 64 |
| Duplicate Searches | 71 | 49 | 40 | 26 | 45 | 0 |
| 端到端耗时 P95 | 30.8s | 122.1s | 105.6s | 101.9s | 102.2s | 196.3s |
| 真实上下文使用率 P95 | 99.6% | 110.1% | 110.0% | 111.0% | 96.2% | 41.8% |
| Context Hard Overflow | 0 | 0 | 0 | 0 | 1 | 0 |
| Infrastructure Invalid | 0 | 0 | 0 | 0 | 1 | 0 |

SFT 的重复动作绝对数量高于 Base，不能简单解释成退化。Base 经常在协议错误后过早结束；SFT
真正进入更长的购物流程，因此总动作量显著增加。GRPO 则在成功率继续提升的同时降低平均步骤、
动作尝试、Guard 拒绝和重复搜索。

## 5.13 配对迁移

当前前端只展示训练主线的 `SFT → GRPO230·Harness`：

- Failure → Success：42；
- Success → Success：118；
- Success → Failure：20；
- Failure → Failure：60。

Qwen3.8-27B 是独立 Actor 对照，不进入“成功转移”统计。step50/100 的历史逐题对比仍保留在
`comparison.json`，但不作为当前前端迁移卡片。

## 5.14 分层结果

最新分层结果按 Base、SFT、GRPO50、GRPO100、GRPO230、Qwen3.8-27B 六列统一生成，详见
`重点/4.评测阶段/dashboard.html`。文档不再把 step100 的历史分层表冒充最终结果。

每个 Challenge Slice 只有 10 题，因此只用于描述性诊断，不单独宣称统计显著。

## 5.15 错误归因

主要错误 Base / SFT / GRPO50 / GRPO100 / GRPO230 / Qwen3.8-27B：

| 错误 | Base | SFT | GRPO50 | GRPO100 | GRPO230 | Qwen3.8-27B |
|---|---:|---:|---:|---:|---:|---:|
| Illegal Action | 167 | 5 | 2 | 1 | 3 | 0 |
| Critical Evidence Missing | 4 | 20 | 11 | 16 | 9 | 14 |
| Candidate Comparison Insufficient | 0 | 13 | 14 | 14 | 6 | 14 |
| Premature Purchase | 1 | 10 | 7 | 9 | 10 | 65 |
| Repeat Loop | 20 | 5 | 5 | 12 | 26 | 3 |
| Overexploration After Convergence | 0 | 13 | 15 | 8 | 2 | 0 |

模型能力变化非常清楚：

```text
Base：主要不会执行协议
    ↓
SFT：学会协议，但存在策略与证据问题
    ↓
GRPO：整体决策和执行更稳定，但候选比较、价格语义和过度探索仍是短板
```

## 5.16 固定分母与异常审计

- Base Task 419 缺失，仍按失败计入 240 题分母；
- Base Task 205 被 Judge 服务商内容过滤，保留为 `not_judged`；
- 不删除或补造 Judge 分数；
- 三组 Infrastructure Invalid 均为 0；
- Judge Coverage 与严格成功率分开统计。

## 5.17 评测产物

正式目录：

```text
重点/4.评测阶段/
```

核心文件：

- `dashboard.html`：可视化结果；
- `audit-report.md`：最终审计报告；
- `comparison.json`：六模型配对和分层比较；前端转移区只展示 SFT→GRPO230；
- `per-task-comparison.csv`：240 题逐题对比；
- `rubrics.jsonl`：冻结逐题 Rubric；
- `judges-base.jsonl` / `judges-sft.jsonl` / `judges-grpo*.jsonl` / `judges-qwen38_27b.jsonl`；
- `runs/*/evaluations.jsonl`：四面板逐题结果；
- `run_manifest.json`：模型、协议、调用和产物 SHA；
- `source_audit.json`：轨迹来源审计。

## 5.18 当前评测限制

当前主报告尚未包含：

- 多随机种子 Rollout；
- 同一模型多次采样的不确定性；
- Judge 重复评审一致性；
- 大规模人工 Rubric 抽查；
- Wilson Interval、Exact McNemar 和连续指标 Paired Bootstrap；
- Reward / Rubric 冲突样本的系统人工复核。

因此可以陈述当前固定协议下的实测提升，但不应额外宣称统计显著或最优配置。

## 5.19 简历表述

> 构建与训练数据零重叠的 Final-240 四面板评测体系：由代码硬判 Gold、Reward Valid 与合法
> 终局，DeepSeek V4 Flash 从候选约束中筛选用户真实要求并冻结逐题 Rubric，DeepSeek V4 Pro
> 仅基于 Query、共享 Rubric、Actor 可见轨迹和白名单指标进行五维盲评与 Event-level 错误
> 归因。最终在固定 240 题分母上进行同任务配对迁移、领域与困难切片分析；SFT→GRPO
> GRPO230 的 Rubric satisfied 由 SFT 的 1,454 增至 1,565、unknown 由 252 降至 155，
> 同时将 Guard 拒绝由 82 降至 25。Rubric 计数是 1,769 条要求的条目统计，不是 240 题成功率。

## 5.20 面试中的 60 秒回答

> 我的评测不是让一个 LLM 直接给总分，而是分成确定性代码、Rubric Curator 和盲 Trajectory
> Judge。代码负责 Gold、Reward Valid、合法终局、Guard、重复动作和资源指标；Flash 模型只
> 根据 Query 和候选约束整理每题用户真正要求，并冻结同一份 Rubric；V4 Pro 只看 Actor 可见
> 轨迹和白名单指标，对搜索、候选利用、证据、决策和终止五维评分，并引用 Event ID 归因。
> 最终保留四个独立面板，不合成总分，再对相同 task ID 做配对迁移和领域、困难切片分析。
> 这样既能回答买对了多少，也能解释为什么成功或失败。

---

# 六、真实面试题与参考回答

以下回答建议理解后转述，不要逐字背诵。

## 6.1 项目整体

### Q1：请用一分钟介绍项目

**参考回答：**

我基于 Qwen3.5-2B 搭建了一套长程购物 Agent 后训练系统，包括 Harness、Teacher 数据、LoRA
SFT、Reward v4、在线 GRPO 和冻结评测。Agent 需要在真实商品快照中完成搜索、候选比较、
证据核验、规格选择和购买。SFT 用 1,000 条严格成功 Action-only 轨迹教授工具协议，GRPO
再通过有组内差异的在线 Rollout 优化策略。最终 Final-240 严格成功率从 Base 的 0% 提升到
SFT 的 59.17%，Harness 改善版 GRPO230 达到 67.92%，Gold＋Valid 购买成功达到 75.00%，
平均 Final Reward 从 0.3368 提升到 0.5131；同协议 Qwen3.8-27B 严格 Gold 为 60.83%、
购买成功为 71.67%，作为独立 Actor 对照。

### Q2：你的核心贡献是什么

**参考回答：**

不是只调用训练框架，而是把环境、数据、训练和评测串成了一套一致合同。我完成了状态化
Harness、Observation 和 Action Guard、Teacher 三层数据门、Action-only SFT、可审计 Reward、
GRPO 动态采样以及四面板 Final-240 评测。项目最重要的工程价值是每个结果都有数据来源、
运行协议和逐题轨迹可以审计。

### Q3：为什么 Base 是 0%，是不是评测太难或不公平

**参考回答：**

Base 的主要瓶颈是工具和页面协议，不是自然语言能力。它有 966 次 Guard 拒绝、167 个
Illegal Action，并有187个任务因连续三次非法动作直接终止，经常在真正完成搜索和规格核验前结束。三组模型使用
相同环境和 Prompt，因此 0% 表示未经后训练的小模型无法稳定完成这套严格长程协议，不代表
它完全不理解购物语言。

### Q4：真实淘宝还是模拟环境

**参考回答：**

使用的是 ShopSimulator 中的淘宝真实商品快照，不直接操作线上淘宝。快照保留真实标题、属性、
价格和复杂规格，同时可以冻结商品库和搜索，使训练和评测可复现。线上商品持续变化，不适合
严格对比实验。

### Q5：为什么选择 2B 模型

**参考回答：**

2B 模型能够验证数据和 RL 是否真正教授了工具协议，同时训练和在线 Rollout 成本可控。如果
直接使用很强的大模型，很多工程问题会被模型能力掩盖。结果显示 1,000 条 SFT 就能带来 57.5%
严格成功，也说明固定场景的协议能力可以通过后训练明显改善。

## 6.2 Harness 与上下文

### Q6：Harness 到底做了什么

**参考回答：**

Harness 是模型和环境之间的执行控制层。它维护 Session 和页面状态，解析 Tool Call，做 Schema
和 Action Guard 校验，并按最新页面只暴露当前可执行工具，把合法动作发给环境，再将环境页面
结构化投影后放回模型上下文。同时维护候选记忆、按完整交互轮次压缩历史，记录步骤、Token、
Reward、异常和终止原因，并控制循环、无进展和最大 45 步。

### Q7：Observation 投影是什么

**参考回答：**

不是向量投影，而是按页面结构和 Token 预算整理原始页面。搜索页 2,560 Token，详情页 3,072，
普通页 512；另给候选选择/候选记忆独立 2,048 Token。搜索页保留全部 20 个商品边界与 ASIN，详情页保留
规格、已选值、公开属性和 variant 价格。核心合同是模型可见目标必须与 Guard 合法目标一致。

### Q8：为什么不能直接截断 Observation

**参考回答：**

简单截断可能留下标题却删除 ASIN，留下规格值却删除规格轴，或者删除页面底部按钮。这会造成
模型看到某个商品，但 Harness 判定它不可点击。结构化投影可以优先保留动作所需字段，再压缩
低优先级长文本；早期候选则由独立记忆保存，不依赖完整旧页面永久留在上下文。

### Q9：Action Guard 会不会替模型完成任务

**参考回答：**

不会。Guard 只拦截确定性非法动作，例如点击当前页不存在的 ASIN 或选择不存在的规格。它不
判断商品是否满足需求、不判断价格和功能证据，也不决定是否购买。动态 Tool Schema 先减少当前
页面不可能执行的工具，Guard 再做最后校验，商品决策仍由模型完成。

### Q10：如何检测循环

**参考回答：**

Environment 根据是否获得新候选、新商品信息、新规格状态或其他实质证据累计无进展步数；连续
3 步时 Evaluation 先给 Loop 高风险提醒，连续 6 步才终止为 `repeat_loop`。精确动作第三次重复
也会终止。35 步后不再叠加循环文案，而改用步数收敛提醒。

### Q11：30K 上下文不够怎么办

**参考回答：**

先对单页 Observation 做结构化投影，再对历史消息按完整 Assistant/Tool 交互组压缩，固定保留
System、Query、最新页面和最近交互。候选记忆用独立预算补偿旧页面被裁掉后的早期商品事实；若
固定内容与最新页面仍无法安全放入预算，则标记基础设施无效而不是截断 Query。

### Q12：Harness 和模型能力如何区分

**参考回答：**

Harness 保证协议合法和状态一致，模型负责语义决策。比如 Guard 可以阻止模型点击不存在的
商品，但不能告诉模型哪个候选满足需求。三个候选后的收敛提醒也只要求停止过度探索，不指定
哪个候选正确。评测同时记录 Guard 拒绝和最终策略质量，因此可以区分协议问题和决策问题。

## 6.3 Teacher 与 SFT

### Q13：1,000 条 Teacher 怎么来的

**参考回答：**

最终由 600 条 Stable 和 400 条 Corrective 组成。Corrective 分成 Loop Recovery、Near-miss
Rejection、Option Grounding 和 Terminal Tool Commit。数据来自旧轨迹重审计、本地远端双机
采集和专项补充，再通过三层数据门筛选。

### Q14：为什么不是“采集 5,000 条再留 1,000 条”

**参考回答：**

因为实际是多来源候选合并，不是单次整齐采集 5,000 条。不同类型 Raw 候选量不同，而且包含
旧数据、双机数据和 Clean Reuse。更准确的说法是从多来源大候选池中审计并筛选 1,000 条严格
成功轨迹。

### Q15：为什么只用成功轨迹做 SFT

**参考回答：**

失败轨迹可以用于偏好学习或 RL，但直接做 SFT 会监督错误动作。我的 Corrective 数据不是失败
轨迹，而是经历典型困难后最终恢复成功的示范，既覆盖错误场景，又不会把错误终局当成监督目标。

### Q16：最终买对为什么还要检查过程

**参考回答：**

模型可能经过非法动作、Guard 多次修复后碰巧买对，也可能没有核验规格而随机命中 Gold。如果
只看结果，这些错误行为会进入训练。因此还要检查动作可重放、Observation Grounding 和行为
语义。

### Q17：Action-only Loss 是什么

**参考回答：**

训练上下文包含 System、User、环境 Observation 和 Assistant Tool Call，但只有 Assistant 动作
Token 计算 Loss，其他位置 Label 设为 -100。这样重点学习工具调用，不监督环境长文本，也不
蒸馏 Teacher 私有 reasoning 和隐藏目标信息。

### Q18：为什么不训练 Teacher 的思维链

**参考回答：**

Teacher reasoning 可能包含隐藏目标、冗余推理或模型特有表达，而且当前环境真正需要的是合法
动作。保留 Observation 让模型自己形成内部判断，只监督动作可以减少泄漏，也让训练目标与
部署时输出一致。

### Q19：如何避免数据模板坍缩

**参考回答：**

不只看平均步数，而是限制完整动作序列 Top-1 / Top-5 占比，并联合约束 Search Rank 和轨迹
长度，保留真实搜索失败、候选比较和 Guard 恢复。最终有 452 种完整动作序列，Top-1 只有
7.5%。

### Q20：为什么超过 30K 的样本整条删除

**参考回答：**

Agent 轨迹中的 Assistant Tool Call 和 Tool Response 必须成对。中间截断会制造无法重放的半条
轨迹，甚至删除最终规格或 Gold 终局。因此超过训练合同的样本整条删除，再从候选池回填。

### Q21：1,000 条是不是太少

**参考回答：**

数量不大，但目标是教授固定工具协议，不是通用知识。数据经过严格门控并覆盖不同检索位置和
行为类型，SFT 也实测从 0% 提升到 57.5%。不过这不能证明 1,000 条最优，后续仍需要数据规模
和 Stable/Corrective 配比消融。

### Q22：SFT 最大的作用是什么

**参考回答：**

主要是协议学习。连续三次非法动作导致的 `guard_rejection` 终局从187降到7，Illegal Action从167降到5。模型从
经常无法进入任务，变成能够搜索、核验、选择规格和购买。剩余问题才转向候选比较和证据质量。

## 6.4 Reward 与 GRPO

### Q23：Reward 为什么需要 Hard Gate

**参考回答：**

有些错误不能被偏好得分补偿，例如买错品类或者明确超预算。Hard Gate 先保证类目和完整规格
价格，再评估品牌、型号、功能和规格，避免“其他维度得分高”掩盖严重错误。

### Q24：为什么替代商品有正奖励，但不算严格成功

**参考回答：**

训练和评测目标不同。训练时，非 Gold 商品如果满足全部有效需求，不应该得到和错误购买相同
的负奖励；但评测需要稳定、可复现的标准，所以严格成功只认 Gold ASIN、严格规格和 Reward
Valid。67.92% 没有把替代商品算进去；把有效替代计入后购买成功为 75.00%。

### Q25：为什么 Reward 无法验证时给 0

**参考回答：**

这个 0 不是中性奖励，而是配合 `reward_valid=false` 和 `sampling_invalid=true` 表示样本无效。
如果价格或关键门槛没有证据，不能把它当成正常好坏样本参与 GRPO。

### Q26：为什么 SFT 后还要做 GRPO

**参考回答：**

SFT 学的是 Teacher 动作分布，能教授协议，但不能穷举所有候选比较和终止选择。GRPO 通过同一
Prompt 的多条在线轨迹比较成功和失败，进一步优化决策。结果也显示 SFT 解决大部分协议问题，
Harness 改善版 GRPO230 再带来 8.75 个百分点、净增 21 个严格成功任务；Gold＋Valid 购买成功
净增 16 题，平均 Final Reward 相对提升约 52.3%。由于同时更新了 Harness，这一结果按完整系统
收益表述。

### Q27：GRPO 任务怎么筛选

**参考回答：**

前 50 step 对 200 加 600 个隔离候选任务分两阶段做 n=4 Probe，共有 251 个通过在线准入，
最终保留 1/4、2/4、3/4 成功的 181 个任务。step 50-100 续训前另建 2,000 题全新隔离候选
池，实际 Probe 709 题后早停，筛出 44 个 1/4 和 56 个 2/4 成功组，共 100 个 clean
frontier。两阶段都保留同题成功与失败轨迹的相对差异，但数据池互相独立。

### Q28：不是还有一版 200 任务吗

**参考回答：**

是的，项目迭代过一版 200 任务池，其中 144 个 Frontier、56 个 Hard Exploration。随后将
前 50 step 的正式池收紧为 181 个 `all-1to3-clean`；step 50-100 则改用新一轮 Probe 筛出的
100 个 clean frontier。面试时应区分早期 200 题实验池、前 50 step 的 181 题正式池，以及
续训阶段的 100 题新池，不能把三套口径混成同一批数据。

### Q29：Dr.GRPO 用在哪里

**参考回答：**

我关闭了 `norm_adv_by_std_in_grpo`，即组内 Reward 中心化后不再除以组内标准差，避免小方差
Group 放大噪声。同时使用 Sequence Mean 的 Loss 聚合，降低长失败轨迹因 Token 更多而主导
更新的风险。准确说法是借鉴 Dr.GRPO 的关键思路，不是完整复现所有论文设置。

### Q30：GAPO 用在哪里

**参考回答：**

主要是动态采样思想。Group 必须至少有一次成功购买，而且 Terminal Utility Range 要大于
0.025；否则最多再生成两批。仍无有效差异就丢弃或跳过更新，把在线算力集中在有学习信号的
Prompt 上。

### Q31：动态采样真的生效了吗

**参考回答：**

生效了。正式日志中生成 272 个 Group，只保留 67 个，丢弃 205 个，其中 139 个是常数或低
差异 Group，19 个没有成功购买，3 个无效。说明在线训练的大量计算确实经过了动态筛选。

### Q32：为什么不用 KL

**参考回答：**

我希望直接观察任务 Reward 对 SFT Policy 的作用，所以关闭 KL Reward 和 KL Loss。风险通过
LoRA、小学习率、PPO Clip 和有限 step 控制。结果有整体提升，但也有 20 个成功任务回退，
说明无 KL 并不是没有代价，后续需要做有无 KL 消融。

### Q33：日志中为什么还有 PPO KL

**参考回答：**

`actor/ppo_kl` 是诊断当前策略和旧策略偏移的监控指标，不代表它进入 Loss。配置中的
`use_kl_loss=false`、`kl_loss_coef=0` 和 `use_kl_in_reward=false` 才决定是否施加 KL 约束。

### Q34：为什么 GRPO Loss 不下降

**参考回答：**

GRPO Actor Loss 不是监督学习交叉熵，而是基于在线采样 Advantage 的策略目标。每一步任务和
轨迹都不同，Loss 围绕 0 正负波动是正常的。应结合 Reward、成功率、KL、梯度、采样保留率和
冻结评测判断训练质量。

### Q35：step-50、step-100 和 step230 应该怎么报告

**参考回答：**

step-50 和 step-100 是同训练链的中间 checkpoint，严格 Gold 分别为 63.33% 和 62.92%；
Harness 改善版 step230 是当前完整系统结果，严格 Gold 达到 67.92%，Gold＋Valid 购买成功达到
75.00%。简历可以报告 step230，但必须明确它同时包含 checkpoint 和 Harness 改善，不能把提升
全部说成“多训练 130 step”的纯增益。

### Q36：训练过程中遇到什么工程问题

**参考回答：**

主要有异步 worker 稳定性、单 GPU 下 vLLM Wake-up、空 Group 和断点续训问题。我增加了
veRL 稳定性补丁、动态采样空组处理和 checkpoint 恢复验证，最终保留 step 5 到 50 的完整
链路，完成从 step-50 到 step-100 的断点续训，并验证 LoRA 合并后的 HF 模型确实包含非零更新。

## 6.5 评测与结果

### Q37：Final-240 怎么构建

**参考回答：**

由 Core-180 和 Challenge-60 组成，Core 覆盖 9 个领域，Challenge 覆盖搜索改写、候选比较、
价格语义、多规格、证据核验和长程任务。训练前冻结 task ID、切片和哈希，并与 SFT、Teacher、
GRPO 保持零重叠。

### Q38：为什么需要 Rubric Curator

**参考回答：**

Gold 商品拥有的字段不等于用户要求。如果直接把 Gold 全部信息交给 Judge，会把目标商品的
额外属性误当成 Query 约束。代码先生成候选约束，Curator 只选择 Query 真正表达的要求，再
冻结给所有模型共享。

### Q39：Judge 如何防止答案泄漏

**参考回答：**

Judge 只看 Query、冻结 Rubric、Actor 可见轨迹、Guard 事件和白名单指标。它看不到 Reward、
Gold ASIN、隐藏商品字段、原始未投影页面和其他模型结果。严格成功也由代码判断，不由 Judge
决定。

### Q40：为什么代码和 LLM Judge 都要用

**参考回答：**

Gold 是否命中、工具是否合法和是否超预算是确定性问题，应由代码判断；搜索改写是否有效、
候选是否充分比较属于语义过程问题，适合 Judge。分开后既能稳定回归，也能解释失败原因。

### Q41：为什么不生成综合总分

**参考回答：**

综合总分需要人为设置权重，容易掩盖问题。模型可能成功率提高，但循环或 Token 成本也增加。
所以我保留 Reward 与终局、需求满足、轨迹质量、行为与资源效率四个独立面板。

### Q42：为什么固定 240 题分母

**参考回答：**

如果模型缺失轨迹或 Judge 失败就从分母删除，指标会被系统性抬高。因此 Base 缺失的 Task 419
仍按失败计算，Judge 内容过滤的 Task 205 记为 Not Judged，但不删除任务。

### Q43：Harness 改善版 GRPO230 的 8.75 个百分点算明显吗

**参考回答：**

它对应相对 SFT 净增 21 个严格成功任务、16 个购买成功任务，同时平均 Final Reward 从 0.3368
升至 0.5131，Guard
拒绝从 82 降至 25。说明完整系统有实际改善，但当前没有多随机种子训练和完整置信区间，且
Harness 同时发生变化，因此不会把 8.75 个百分点夸大为纯训练因果或统计显著。

### Q44：为什么 SFT 重复动作比 Base 多

**参考回答：**

Base 经常在非法动作后很快结束，平均只有 5.9 步；SFT 真正进入长程购物，平均约 16.9 步，
因此绝对重复动作数会增加。更合理的判断是结合轨迹长度、成功率和错误类型。GRPO 在成功率
提升的同时又减少了平均动作和重复搜索。

### Q45：项目目前最大的不足是什么

**参考回答：**

第一，几乎所有任务都发生过 Observation 投影，说明长上下文证据保留仍是瓶颈；第二，候选
比较、价格语义和收敛后过度探索仍不稳定；第三，缺少多随机种子、Judge 一致性、人工抽查和
Reward/KL/动态采样消融。

### Q46：下一步做什么

**参考回答：**

我会先建立独立 checkpoint-dev 和错误专项开发集，不接触 Final-240；重点复核 GRPO230 的
成功回退、基础设施无效样本和 Qwen3.8-27B 的部分替代购买；然后比较二值 Reward 与分段 Reward、有无动态采样、有无 KL、不同 group size
和 Observation Budget，并补充配对统计检验和 Judge 稳定性分析。

---

# 七、容易说错的边界

面试中不要将以下内容夸大：

1. **不是直接操作实时淘宝**，而是真实商品快照环境；
2. **不是简单采集 5,000 条留下 1,000 条**，而是多来源候选池审计；
3. **有效替代不计入 67.92% 严格成功率，计入后购买成功为 75.00%**；
4. **step-50、step-100 和 step230 都必须按真实模型与实际 Harness 描述**，不能把完整系统变化冒充纯 checkpoint 消融；
5. **Dr.GRPO 和 GAPO 是借鉴关键机制**，不要声称完整复现论文全部算法；
6. **PPO KL 是监控指标，不是 KL Loss**；
7. **LLM Judge 不决定严格成功率**，严格成功由代码硬判；
8. **没有综合总分**，四个面板独立报告；
9. **Challenge 每个切片只有 10 题**，不单独宣称显著性；
10. **当前没有多随机种子和完整消融**，不能声称当前配置最优。

---

# 八、材料与代码索引

## 8.1 总说明

- `重点/x.md`
- `重点/README.md`
- `重点/data.md`
- `重点/interview-final1000-data-engineering-retrospective.md`
- `重点/shopsimulator-data-cleaning-and-splitting.md`

## 8.2 Harness

- `src/shopping_grpo/environment/`
- `src/shopping_grpo/training/grpo/adapter/`
- `docs/harness.md`
- `configs/agent_loop.yaml`
- `configs/tools.json`

## 8.3 SFT

- `data/sft-final1000-convergence-v5-30k/`
- `configs/sft_convergence_repair_v5_30k.json`
- `scripts/train_lora_sft.py`
- `outputs/runs/sft/fresh-sft-convergence-repair-v5-30k/`
- `outputs/models/fresh-sft-convergence-repair-v5-30k-lora/`
- `outputs/models/fresh-sft-convergence-repair-v5-30k-merged/`

## 8.4 GRPO

- `data/grpo/training-probe-v1/all-1to3-clean/`
- `configs/grpo.yaml`
- `src/shopping_grpo/training/grpo/dynamic_sampling.py`
- `src/shopping_grpo/training/grpo/verl_dynamic_sampling.py`
- `outputs/models/grpo-all-1to3-clean-20260813-r2/global_step_50/`
- `outputs/models/grpo-step50-hf-merged/`

## 8.5 评测

- `data/evaluation/`
- `src/shopping_grpo/evaluation/`
- `重点/4.评测阶段/dashboard.html`
- `重点/4.评测阶段/audit-report.md`
- `重点/4.评测阶段/comparison.json`

## 8.6 训练图表

- `重点/训练流程图/01-sft-v5-final1000-process.png`
- `重点/训练流程图/02-grpo-step50-process.png`
- `重点/训练过程曲线/SFT-Final1000/`
- `重点/训练过程曲线/GRPO-step50/`
