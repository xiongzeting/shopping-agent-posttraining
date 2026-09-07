# Shopping Agent Harness 超级详细设计说明

> 本文只描述仓库当前唯一运行合同：ShopSimulator Environment v2.4、Reward v4、
> Termination v3.1、Observation v2、Tool Schema v2。
> 文中的“严格成功”始终指完整 `gold_purchase` 终局且 `reward_valid=true`。

## 0. 为什么需要 Harness

Shopping Agent 不是一次性问答模型，而是一个在有状态环境中连续执行动作的 Agent。模型每一轮只
负责选择“下一步做什么”，ShopSimulator 才负责维护页面、搜索结果、当前商品、已选规格、实际
variant 价格、证据进度、终止和奖励。如果直接把模型输出原样交给环境，会立即出现五类工程问题：

1. 模型可能输出未知工具、错误参数、多个工具调用或纯自然语言；
2. 模型可能点击历史页面的 ASIN、规格 ID 或按钮，污染当前环境状态；
3. 搜索页、详情页和长轨迹可能超过上下文窗口，粗暴截断又会删掉可执行目标；
4. 并发 GRPO trajectory 可能串用环境实例或运行状态；
5. 模型失败、Reward 不可验证和服务故障如果混在一起，会向训练提供错误学习信号。

Harness 就是模型和 ShopSimulator 之间的执行控制层。它不替模型挑商品，也不重新实现环境，而是
保证每条 trajectory 满足下面的工程合同：

```text
模型看见的最新页面
        ↓
模型产生一个 Tool Call
        ↓
Schema 校验 + Action Guard
        ↓ 仅合法动作可以通过
Tool Call → ShopSimulator action
        ↓
Environment step
        ↓
公共 Observation v2 渲染
        ↓
安全 Token 投影
        ↓
写回模型上下文并记录审计信息
```

最核心的不变量是：

```text
模型最新可见的动作目标集合
    = Action Guard 允许的动作目标集合
    = 当前 Environment Session 实际可执行的动作目标集合
```

只要这三个集合不一致，就会产生“模型看得见但点不了”或“模型看不见却能点”的训练污染。

## 1. Harness 的职责边界

### 1.1 Harness 必须负责

- 向模型公开唯一的 Tool Schema v2；
- 保证每个 assistant 回合最多执行一个工具调用；
- 校验工具名、参数对象、必填字段、类型、空字符串和额外字段；
- 只依据最新 observation 判断动作是否仍合法；
- 把工具调用稳定转换成 `search[...]`、`click[...]`、`finish[...]`；
- 渲染不含隐藏答案的 Observation v2；
- 为每条 trajectory 维护有界、稳定编号的公开候选证据与重新定位线索；
- 在 token 预算内投影 observation，同时完整保留动作目标；
- 管理上下文窗口，只删除完整旧交互组；
- 为每条 trajectory 创建、绑定和释放独立环境租约；
- 区分正常终局、模型行为失败、Reward 不可验证和基础设施故障；
- 记录足够的轨迹、投影、压缩、Guard、Reward 和释放诊断。

### 1.2 Harness 明确不负责

- 不根据隐藏 `goal`、`target_asin` 或答案选择商品；
- 不判断品牌、型号、功能、规格是否语义满足；
- 不替模型维护“最佳候选”或跨页面候选排序；
- 不因为预算不满足就自行阻止 `buy_now`；
- 不重新计算 Environment 的 Reward v4；
- 不把评测任务反馈用于训练数据或 Harness 调参；
- 不把基础设施错误伪装成普通零分样本。

一句话概括：Guard 判断“现在能不能做”，模型判断“现在应不应该做”，Reward 判断“最终做得对不对”。

## 2. 当前模块全景与状态所有权

| 层 | 主要文件 | 权威数据 | 核心职责 |
|---|---|---|---|
| 工具协议 | `environment/tools.py`、`configs/tools.json` | Tool Schema v2 | 定义模型可调用工具并映射环境动作 |
| 动作守卫 | `environment/actions.py` | 最新模型可见 observation | Schema 二次校验、当前页目标校验、拒绝反馈 |
| 公共状态生成 | `ShopSimulator/.../engine/observation.py` | Environment Session | 从环境内部状态构造无答案的结构化公共对象 |
| Observation 渲染 | `environment/observation.py` | 公共结构化对象 | 输出稳定文本格式，拒绝隐藏字段和结构不一致 |
| 候选记忆 | `environment/candidate_memory.py` | 已展示的公开详情与搜索位置 | Final-240 维护稳定 C1-C4，生成不可直接点击的有界摘要 |
| Observation 投影 | `environment/projection.py` | 原始公共 observation | 按页面类型压缩，并验证动作空间不变 |
| HTTP 生命周期 | `environment/client.py` | `env_idx` 租约 | `reset → step* → release` |
| 上下文预算 | `environment/context.py` | chat 或 token 轨迹 | 删除最旧完整交互组并保持数组对齐 |
| 评测循环 | `evaluation/rollout.py` | messages、steps、trajectory | OpenAI-compatible 模型调用与显式 while-loop |
| GRPO Session | `training/grpo/adapter/session.py` | coroutine-local env/state | 异步 reset、ContextVar 隔离、finally release |
| GRPO Tool | `training/grpo/adapter/tools.py` | runtime state | Guard、环境执行、终局 Reward 校验 |
| GRPO AgentLoop | `training/grpo/adapter/agent_loop.py` | veRL AgentData | 生成预算、投影、串行调用、最终奖励和诊断 |
| 终止进度 | `ShopSimulator/.../engine/termination.py` | Environment tracker | 重复、无进展、最大步数和探索证据统计 |
| Reward | `ShopSimulator/.../engine/reward.py` | goal、purchase、公共证据 | Reward v4 终局分类和 terminal utility |

### 2.1 三种不能混淆的状态

| 状态 | 保存位置 | 是否权威 | 是否可给模型 |
|---|---|---:|---:|
| 环境真实页面和内部 Session | ShopSimulator | 是 | 只能经过公共 observation 转换后给 |
| `latest_observation` | Evaluation 局部变量或 GRPO runtime state | 对“模型最后看见什么”权威 | 是 |
| `candidate_memory` | 每条 trajectory 的局部状态 | 对“模型曾看过哪些公开候选”权威 | 是，作为 Observation 摘要 |
| 轨迹诊断 | trajectory / `shopping_info` | 对审计权威 | 通常不作为下一轮模型输入 |

`latest_observation` 不是环境本身。修改这段文本不会改变环境页面，但 Guard 必须用它，因为动作合法性
合同是“模型只能点击自己最后看到的目标”。环境内部状态则用于真正执行动作和计算奖励。

## 3. 一条 trajectory 的完整生命周期

### 3.1 正常路径

```text
1. 读取 task_id 和用户完整需求
2. 创建 ShopAgentEnv
3. reset(task_id)，取得 env_idx、环境版本和 observation_state
4. 校验 Environment 必须为 shopsimulator-environment-v2.4
5. render_structured_observation(observation_state)
6. 保存 latest_observation
7. 普通 System Prompt + 用户需求 + 当前动态 Tool Schema 送入模型；普通阶段不投影候选记忆
8. 模型返回 assistant tool call
9. 强制串行：最多保留一个 tool call
10. 解析 JSON arguments
11. action_reject_reason 执行本地校验
12. 通过后 tool_call_to_action 转换环境动作
13. env.step(action)
14. 环境更新页面、进度、done、reward、reward_detail
15. 用搜索结果更新公开检索位置，或用详情页更新候选记忆
16. 把新的 observation_state 渲染为 Observation v2，并在 footer 前插入候选记忆
17. 追加循环/步数提醒
18. project_observation 按页面预算投影，并追加独立候选记忆预算
19. 验证 ASIN、option、按钮和 footer 不变量
20. 把 tool observation 写回上下文
21. 若未终止，回到第 8 步
22. 若终止，校验 Reward v4 并结算 trajectory
23. finally release env_idx
```

### 3.2 Guard 拒绝路径

```text
模型 Tool Call
  → Schema/当前页校验失败
  → 不调用 env.step
  → 环境页面不变
  → 将 Guard error 作为 tool message 返回
  → 模型在同一最新 observation 上纠正动作
```

因此“动作尝试数”和“环境执行步数”不同：Guard 拒绝增加尝试和拒绝计数，但不增加成功执行的
`steps`；合法环境动作才进入环境的 45 步合同。

### 3.3 异常路径

任何 reset、模型生成、工具执行、Observation 渲染、投影、Reward 解析或 release 异常，都必须：

1. 保留异常发生阶段和类型；
2. 尽可能释放环境租约；
3. 判断是模型失败还是 infrastructure invalid；
4. 不把未知状态的轨迹当作普通零分训练样本。

## 4. Tool Schema v2：模型唯一动作空间

当前公开 8 个工具。`configs/tools.json` 是 veRL 运行配置，
`SHOP_TOOL_SCHEMAS` 是 Python 侧定义；测试要求二者完全一致。

| 工具 | 参数 | 环境动作 | 典型可用页面 | 是否终止 |
|---|---|---|---|---:|
| `search_products` | `{"query": string}` | `search[query]` | footer 显示搜索可用的页面 | 否 |
| `open_product` | `{"asin": string}` | `click[ASIN]` | 当前搜索结果页 | 否 |
| `select_option` | `{"value": "opt_..."}` | `click[opt_...]` | 当前商品详情页 | 否 |
| `next_page` | `{}` | `click[Next >]` | 有下一页的搜索结果页 | 否 |
| `prev_page` | `{}` | `click[< Prev]` | 搜索第 2+ 页、详情页、信息子页 | 否 |
| `back_to_search` | `{}` | `click[Back to Search]` | 搜索结果、详情、信息子页 | 否 |
| `buy_now` | `{}` | `click[Buy Now]` | 当前商品详情页 | 是 |
| `finish_without_purchase` | `{"reason":"no_suitable_product"}` | `finish[no_suitable_product]` | 任何非终局阶段；统一按 `early_abstain` 终止 | 是 |

### 4.1 为什么所有 schema 都设置 `additionalProperties=false`

无参数工具必须严格传 `{}`。例如以下调用都应被拒绝：

```json
{"name":"buy_now","arguments":{"confirm":true}}
{"name":"next_page","arguments":{"page":2}}
{"name":"open_product","arguments":{"asin":"12345678","note":"best"}}
```

这样做避免模型把自然语言理由、历史候选或自定义控制字段偷偷混入环境动作。

### 4.2 本地 Schema 校验顺序

`_schema_argument_error()` 在触碰环境前依次检查：

1. 工具名是否存在；
2. `arguments` 是否为对象；
3. 是否包含 schema 未声明的字段；
4. 是否缺少 required 字段；
5. 字符串参数是否真的是字符串；
6. required 字符串去空白后是否为空。

`finish_without_purchase.reason` 还会在 Guard 中再次严格检查必须等于
`no_suitable_product`。JSON Schema 的 `enum` 是服务侧第一道约束，本地 Guard 是执行前第二道约束。

### 4.3 Tool Call 到环境动作的映射

映射函数保持非常薄：

```text
search_products(query)          → search[query]
open_product(asin)              → click[asin]
select_option(value)            → click[value]
next_page({})                   → click[Next >]
prev_page({})                   → click[< Prev]
back_to_search({})              → click[Back to Search]
buy_now({})                     → click[Buy Now]
finish_without_purchase(reason) → finish[reason]
```

`resolve_action_parameters()` 当前原样返回参数，因为 `select_option` 已使用稳定 option ID，Harness
不需要再把 label 翻译成内部值。稳定 ID 本身就是环境当前页可以识别的精确动作参数。

### 4.4 为什么没有 `view_description` 等工具

当前 Tool Schema v2 不公开 Description、Features、Reviews、Attributes 子页工具，结构化 Observation
也会从模型可见按钮中删除这些不可调用入口。为避免模型为了核验信息额外进入空子页，商品原始数据中
非空的 `Features` 与 `Attributes` 会直接并入商品详情 observation；模型依据当前详情中的标题、品牌、
品类、价格、关键属性、Features、Attributes 与规格完成核验，不能自行发明历史工具名。

### 4.5 `think` 的真实边界

部分 Guard/Adapter 代码仍能识别名为 `think` 的内部分支，但 Tool Schema v2 和 `configs/tools.json`
都没有公开 `think`，System Prompt 也明确禁止调用。因此当前运行合同中它不是模型可用工具，不能把
内部兼容分支误写成公开动作空间。

## 5. ShopSimulator 页面与实际可执行按钮

Environment 从当前 HTML 页面提取按钮、商品链接和 radio 规格，生成 `available_actions`：

```text
has_search_bar: 当前页面是否存在 search_input
clickables: 当前页面所有 .btn、product-link 和稳定 option_id
```

公开按钮名称会被规范成小写用于环境内部映射，但 Observation footer 保留环境返回的字符串。典型页面：

| 页面类型 | 典型动作目标 |
|---|---|
| `search_home` | 搜索框；通常没有商品 ASIN |
| `search_results` | 当前页最多 20 个 ASIN、Back to Search、按页码出现的 `< Prev` / `Next >` |
| `product_detail` | Back to Search、`< Prev`、Description、Features、Reviews、可选 Attributes、Buy Now、所有 `opt_...` |
| `information_subpage` | Back to Search、`< Prev` |
| `terminal` | 无后续有效动作 |

真正的动作合法性永远以 observation footer 为准，而不是依赖上表猜测。模板或环境配置变化后，只要
footer 和 Guard 仍从同一 `available_actions` 生成，合同就不会漂移。

## 6. Stable Option ID：规格选择为什么不能传 label

商品规格的原始 label 可能重复、含空格、大小写不稳定，甚至与导航按钮同名。Environment 为每个规格
生成确定性的公开 ID：

```text
identity = JSON([
  ASIN.upper(),
  axis.casefold(),
  raw_value
])
option_id = "opt_" + sha256(identity) 的前 16 位十六进制
```

例如模型会看到：

```json
"available_options": {
  "颜色分类": [
    {"option_id":"opt_4e6b4c0f2f48a310","label":"桔色"},
    {"option_id":"opt_c9cbb032fb83f9af","label":"黑色"}
  ]
}
```

正确调用：

```json
{"name":"select_option","arguments":{"value":"opt_4e6b4c0f2f48a310"}}
```

错误调用：

```json
{"name":"select_option","arguments":{"value":"桔色"}}
```

Guard 要求 `value` 匹配 `^opt_[0-9a-f]{16}$`，并且该 ID 必须出现在最新 footer 的按钮列表中。
Environment 在点击时还会重新计算预期 ID；如果公开 ID 与实际 radio 的 ASIN、axis、value 不一致，
环境会拒绝执行。这样同时防止 label 歧义、历史规格点击和跨商品规格串用。

## 7. Observation v2 的双层生成

Observation 不是直接把 HTML 或 Environment Session 全量发给模型，而是经过两层：

```text
Environment 内部状态
  → build_observation_state() 生成结构化公共对象
  → render_structured_observation() 生成稳定文本
```

第一层决定“哪些字段可以公开”，第二层决定“模型看到的格式和结构不变量”。

### 7.1 公共状态生成器的输入

`build_observation_state()` 只接收：

- `page_type`；
- 当前 `session`；
- 商品字典；
- 当前 `available_actions`。

函数签名故意不接收 task `goal`。这是一道很重要的结构性防泄漏边界：公共状态生成器无法因为调用者
疏忽而直接序列化隐藏目标。

### 7.2 公共商品摘要

搜索和详情都使用受限 `product_summary`：

| 字段 | 来源 | 限制 |
|---|---|---|
| `asin` | 商品 ID | 必须是合法目录 ID |
| `title` | title / Title | 文本 |
| `brand` | brand / shop_name | 文本 |
| `category` | category | 文本 |
| `price` | Price / pricing | 原始公共价格 |
| `key_attributes` | attribute / Attributes | 最多前 10 项 |
| `rank` | 搜索结果全局排名 | 仅搜索结果提供 |

它不公开完整 Reward 特征、目标匹配结论、候选分数或隐藏 ASIN。

### 7.3 搜索结果结构化对象

搜索页额外包含：

- 原始 `query`；
- `normalized_query`；
- 当前页码和总页数；
- 总结果数；
- 当前页全局 rank 起止；
- 当前页商品数组；
- 当前页完整动作列表。

当前页 ASIN 来自 `session.current_page_asins`，排名来自完整 `search_result_asins`。页面大小被冻结为
20，因此 renderer 遇到超过 20 个商品会失败关闭，而不是静默删掉尾部候选。

### 7.4 商品详情结构化对象

详情页包含：

- 公共商品摘要；
- `selected_options`；
- `available_options`；
- 若 variant 价格可解析，则包含 `selected_price`；
- 当前页完整动作列表。

`selected_options` 的每个轴必须引用当前 `available_options` 中仍存在的 option ID。模型看到的 `price`
优先采用 `selected_price`，否则回退商品基础价格。选择任一规格后，Environment 会重新执行
`resolve_variant_price()`，因此模型必须根据新 observation 重新核验完整 variant 的实际价格。

### 7.5 信息子页结构化对象

信息子页复用商品摘要，但主动把 `selected_options` 和 `available_options` 置空，并追加：

- `subpage`：description / features / reviews / attributes；
- `content`：对应的公开正文。

页面 footer 通常只有 Back to Search 和 `< Prev`。因此即使正文里出现商品、规格或“Buy Now”等文字，
Guard 也只认 footer 中的真实按钮。

### 7.6 Terminal 结构化对象

terminal observation 只保留 Observation header、`page_type: terminal` 和 footer，不向模型暴露
`goal`、purchase 对比或 Reward 详情。终局 Reward 进入轨迹审计和训练结算，不再作为下一轮商品决策
输入。

### 7.7 候选记忆：保存公开事实，不保存答案

每条 Evaluation 或 GRPO trajectory 都会创建独立的 `candidate_memory`。当前 Final-240 Evaluation
入口将上限设为 4，稳定保存 C1-C4；GRPO Adapter 仍通过 `runtime.py` 使用自己的默认上限，必须单独
核对，不能因为共享 renderer 就假设两条入口完全相同。候选记忆不是模型生成的摘要，也不是 Environment
的隐藏候选表，而是 Harness 对模型已经看过的公开字段做确定性、有界压缩。

每个候选保存 ASIN、标题、品牌、品类、最新价格、已选规格 label、公开属性证据，以及最近一次
打开它时对应的搜索词、页码、全局 rank 和页内位置。首次/最近查看步数和查看次数只用于审计。

模型可见格式是一行一个候选：

```text
[CANDIDATE_MEMORY_V2]
候选决策提醒: 目前已经至少有3个候选（当前已保存 3 个）。立即比较这些候选……
已核验候选：C1-C4 是候选在当前轨迹内的稳定编号；不代表优劣、满足情况或推荐。
格式：代号｜ASIN｜位置(@检索词/P页/R排名)｜价格｜品牌｜品类｜已选规格｜标题｜公开证据
C1｜12345678｜@电饭煲 黑色/P2/R24｜756｜某品牌｜电饭煲｜颜色=咖啡色｜商品标题｜66cm,防水
[/CANDIDATE_MEMORY_V2]
```

C1-C4 是稳定标识，不是“最佳候选”排名。同一 ASIN 重访时更新原记录的最新价格、规格和位置；
达到 4 个后不再写入或替换后续商品。当前详情页商品不会在候选行中重复展开，只显示当前候选编号。

位置记忆解决“模型记得 C1 好，却忘了 C1 在哪”的问题，但不会绕过动作合同，也不存在
`reopen_candidate`。若该 ASIN 已在当前搜索页，模型可调用 `open_product`；否则要按
`@检索词/P页` 重新搜索和翻页，只有 ASIN 再次出现在当前页 footer 后 Guard 才允许打开。

达到 3 个候选后会发生两件事：共享 renderer 在候选块中写入“目前已经至少有3个候选”；Evaluation
在下一轮模型请求前把同一语义的提醒置于 System Prompt 最前面。达到 3 个之前完全不出现，达到 4 个
时也不会重复注入。它不隐藏工具、不自动购买、不改变 Environment 的 6 步无进展终止规则。

防泄漏边界：候选记忆不读取 Query 目标解析、Gold、Reward、target ASIN、约束满足结果、匹配分或
其他 trajectory；也不对候选打分、排序或标记“合格”。它只是把已经出现过的公开事实变成可持续的
工作记忆。

## 8. Observation v2 文本格式

所有页面以固定 header 开始：

```text
[SHOPPING_OBSERVATION_V2]
page_type: ...
```

所有页面以固定 footer 结束：

```text
搜索功能是否可用: True|False
可点击的按钮: [JSON string array]
```

footer 不是装饰信息，而是 Guard 和 Projector 共同依赖的动作合同。

非终局页面如果存在候选记忆，记忆块插在正文与 footer 之间，因此 footer 始终保持最后。候选行中的
历史 ASIN 不使用搜索商品行格式，也不会被 `product_ids()` 或 `clickable_buttons()` 识别为当前动作。

### 8.1 搜索首页示例

```text
[SHOPPING_OBSERVATION_V2]
page_type: search_home
使用 search_products 提交简短、具有区分度的商品查询。

搜索功能是否可用: True
可点击的按钮: []
```

### 8.2 搜索结果示例

```text
[SHOPPING_OBSERVATION_V2]
page_type: search_results
query: 老人 无线 充电 助听器
normalized_query: 老人 无线 充电 助听器
Page 1 of 3 (Total results: 54; ranks 1-20)
格式: rank|asin|price|brand|category|key_attributes|title
1|12345678|218|某品牌|医疗器械›助听器|无线,充电式,耳蜗式|商品标题
...
products_shown: 20

搜索功能是否可用: False
可点击的按钮: ["back to search","next >","12345678",...]
```

Renderer 会验证“商品行中的 ASIN 集合”与“环境 actions 中属于商品 ID 的集合”完全相等。任一商品
只显示不允许点，或允许点却没显示，都会抛出 `StructuredObservationError`。

### 8.3 商品详情示例

```text
[SHOPPING_OBSERVATION_V2]
page_type: product_detail
asin: 12345678
title: 某商品
brand: 某品牌
category: 医疗器械›助听器
price: 218
key_attributes: 无线, 充电式, 耳蜗式
selected_options: {"颜色分类":{"label":"桔色","option_id":"opt_..."}}
available_options: {"颜色分类":[{"label":"桔色","option_id":"opt_..."}]}

搜索功能是否可用: False
可点击的按钮: ["back to search","< prev","description","features","reviews","buy now","opt_..."]
```

### 8.4 文本规范化

所有普通文本会把连续空白压成单空格并去掉首尾空白。列表仅保留非空项。这样减少不稳定格式和无效
token，但不会做语义改写、同义词归一或隐藏目标推断。

## 9. Observation 防泄漏与失败关闭

Renderer 顶层拒绝以下字段：

```text
goal, reward, reward_detail, target_asin, answer,
candidate_state, current_candidate, best_candidate,
satisfied_conditions, missing_conditions, unverified_conditions,
public_match_score, fully_satisfied
```

同时还执行结构约束：

- observation version 必须是 `shopping-observation-v2`；
- page type 必须是受支持页面；
- 搜索页必须有 products list；
- 商品 ID 必须满足目录 ID 规则；
- 搜索页最多 20 个商品；
- 搜索商品集合必须等于可执行 ASIN 集合；
- option ID 必须是稳定 ID；
- 公开 option ID 必须包含于环境 actions；
- selected option 必须仍属于当前 available options。

任何不一致都不应“尽量继续”，而应停止当前 trajectory 并归类为 observation/protocol 基础设施异常。
原因是继续训练会让模型在一个不可信动作空间中学习。

## 10. Action Guard 的完整校验逻辑

`action_reject_reason(name, arguments, observation)` 的顺序不能随意调整：

```text
1. 通用 Schema 参数检查
2. 内部分支 think（当前未公开）
3. finish_without_purchase reason 检查
4. search_products 的 search_available 检查
5. 其余工具要求存在 previous observation
6. open_product 校验 ASIN 属于当前页
7. 其他点击工具先转换为 click[target]
8. select_option 排除导航按钮
9. select_option 强制 stable ID 格式
10. target 必须存在于当前 footer 按钮集合
```

### 10.1 Guard 拒绝原因字典

| 原因 | 含义 | 模型应如何恢复 |
|---|---|---|
| `unknown_tool` | 工具不在 Tool Schema v2 | 选择公开的 8 个工具之一 |
| `schema_arguments_not_object` | arguments 不是 JSON object | 传对象 |
| `schema_extra_arguments:x` | 多传了字段 | 删除未声明字段 |
| `schema_missing_arguments:x` | 缺少必填字段 | 补齐正确字段 |
| `schema_wrong_type:x:string` | 字段类型错误 | 改为字符串 |
| `schema_empty_string:x` | 必填字符串为空 | 提供非空值 |
| `invalid_finish_reason` | 主动结束 reason 非法 | 只能用 `no_suitable_product` |
| `search_not_available_on_current_page` | footer 明确显示不可搜索 | 用当前按钮返回或操作当前页 |
| `missing_previous_observation` | 点击动作没有可信最新页 | 停止执行并检查上游状态 |
| `click_not_in_previous_observation` | ASIN/按钮不属于最新页 | 从最新 footer 重新选目标 |
| `select_option_is_navigation_button` | 把导航按钮传给规格工具 | 使用正确导航工具或 option ID |
| `select_option_requires_stable_id` | value 不是 `opt_...` | 从 available_options 复制 ID |
| `unknown_or_invalid_tool` | 工具无法映射成环境动作 | 使用标准工具和参数 |

### 10.2 `open_product` 如何提取当前页商品

Guard 支持两种 observation 格式：

- Observation v2 投影行：`rank|ASIN|...`；
- 旧式 `[SEP]` 分段文本中的合法商品 ID。

当前正式路径使用 Observation v2。提取后去重并保持出现顺序。模型传入的 ASIN 必须原样存在于该
列表，不能点击上一页、上一查询或历史详情中的 ASIN。

### 10.3 `clickable_buttons` 如何读取按钮

Guard 只解析 footer 中这一行：

```text
可点击的按钮: [合法 JSON 数组]
```

若 JSON 无法解析、字段缺失或元素不是字符串，对应目标不会被认为合法。正文里出现的“Buy Now”、
ASIN 或 option label 都不会自动成为按钮。

### 10.4 Guard 为什么不拦截“买错商品”

只要最新 footer 有 `Buy Now`，`buy_now({})` 在动作层就是合法的。Guard 不读取隐藏目标，也不根据
品牌、品类、预算或功能做购买准入，否则 Harness 会变成答案规则引擎，模型无法真正学习商品判断。
买得对不对由模型策略和 Reward v4 负责。

### 10.5 搜索 Guard 当前是负向检查

点击类工具要求目标明确存在于最新 footer；`search_products` 当前实现则是检查 observation 中是否明确
出现 `搜索功能是否可用: False`，出现才拒绝，否则允许。因此正式 Observation v2 必须始终可靠输出
该 footer。若上游给了缺失搜索状态的畸形 observation，搜索分支可能放行；这属于需要用 renderer 和
协议测试共同兜住的实现细节，而不是可以依赖的宽松合同。

### 10.6 Guard 拒绝反馈

Evaluation 使用 `action_guard_tool_message()` 返回可恢复信息：

- 明确说明调用未执行；
- 给出拒绝原因；
- 列出当前可打开 ASIN；
- 列出当前可点击按钮；
- 若处于只有 `< Prev` / Back to Search 的信息子页，明确下一步只能返回；
- 提醒只调用一个合法工具。

GRPO Tool Adapter 当前只返回较短文本：

```text
Error: action guard rejected this call (...); read the latest observation.
```

这是一项真实的双入口反馈差异：守卫判定相同，但模型收到的纠错信息丰富度不同。

## 11. 页面—工具合法性矩阵

下面的“可用”仍以 footer 为最终准则：

| 工具 | 搜索首页 | 搜索结果 | 商品详情 | 信息子页 | 终局 |
|---|---:|---:|---:|---:|---:|
| `search_products` | 若 search=True | 若 search=True | 若 search=True | 若 search=True | 否 |
| `open_product` | 否 | 当前页 ASIN | 否 | 否 | 否 |
| `select_option` | 否 | 否 | 当前页 `opt_...` | 否 | 否 |
| `next_page` | 否 | footer 有 Next > | 否 | 否 | 否 |
| `prev_page` | 否 | footer 有 `< Prev` | footer 有 `< Prev` | footer 有 `< Prev` | 否 |
| `back_to_search` | 通常无 | footer 有按钮 | footer 有按钮 | footer 有按钮 | 否 |
| `buy_now` | 否 | 否 | footer 有 Buy Now | 否 | 否 |
| `finish_without_purchase` | 可调用，统一早停 | 可调用，统一早停 | 可调用，统一早停 | 可调用，统一早停 | 否 |

`finish_without_purchase` 是特殊终止动作，不依赖当前按钮，但不代表调用就成功。模型主动停止且未购买时，
Environment 统一判定为 `early_abstain`。

## 12. Observation Token 投影：不能用普通截断

普通的“保留前 N token”会制造三类严重错误：

1. 搜索页尾部 ASIN 被删，但 footer 仍可点击；
2. footer 被删，Guard 认为所有按钮都不可用；
3. 详情页头部保留了商品信息，但尾部规格或导航状态丢失。

`project_observation()` 因此遵循“先识别页面，再压缩正文，最后验证动作空间”的流程。

### 12.1 当前预算

| 页面识别结果 | Token budget | 配置字段 |
|---|---:|---|
| `search_results` | 2,560 | `observation_token_budget` |
| `product_detail` | 3,072 | `observation_detail_token_budget` |
| 搜索首页、信息子页、generic | 512 | `observation_generic_token_budget` |
| 候选选择/候选记忆 | 2,048 | `observation_candidate_memory_token_budget` |

所有预算必须至少 64 token，搜索页容量必须为正数，正式配置为 20。

### 12.2 页面识别

投影器优先读取 Observation v2 的 `page_type`。为兼容输入解析，还能识别旧 `[SEP]` 搜索页、按钮中
的 Buy Now、旧“价格:”字段、search available 和仅返回按钮的信息子页。正式运行应以结构化
Observation v2 为主。

### 12.3 短 observation 保持原样

若原始 token 数不超过页面预算，投影器直接返回完全相同的字符串。这样避免每一步都改变模型输入，
也使 `truncated=false` 具有清晰含义。

### 12.4 超长搜索页投影

结构化搜索页的算法：

1. 分离正文和 footer；
2. 识别所有 `rank|ASIN|payload` 商品行；
3. 保留所有 rank 和 ASIN；
4. 仅保留当前商品 ASIN和导航按钮到 footer；
5. 对商品行中 ASIN 之后的各字段执行统一字符上限压缩；
6. 使用二分搜索找到 token 预算内能保留的最大字段长度；
7. 若发生压缩，加入 `[TRUNCATED_BY_SHOPPING_PROJECTOR]` 标记；
8. 即使字段压到空字符串仍放不下全部商品，则抛错，不删除商品。

字段压缩采用“约 2/3 头部 + 约 1/3 尾部 + 省略号”，尽量保留标题和属性两端信息。注意当前实现
会对 ASIN 后的 price、brand、category、key_attributes、title 各字段统一应用字符限制，不是只压缩
title；因此投影元数据和边界测试非常重要。

### 12.5 超长详情/信息页投影

generic 投影算法：

1. 完整分离 footer；
2. 先验证“截断标记 + 完整 footer”本身能放入预算；
3. 对正文使用二分搜索寻找可保留的最大字符数；
4. 保留正文头部约 80% 和尾部约 20%；
5. 中间插入截断标记；
6. 最后原样重建搜索状态、步数提醒、循环提醒和按钮 JSON。

头部通常包含商品标题、品牌、品类、价格和关键属性，尾部通常包含规格、子页内容尾部或其他页面
上下文；footer 始终单独完整保留。

### 12.6 投影后强制验证

投影器重新计 token，并检查：

- `visible_tokens <= effective_budget`；
- 搜索状态和按钮 footer 仍存在；
- 导航按钮集合完全一致；
- 非搜索页全部按钮集合完全一致；
- 搜索页原始 ASIN 集合等于可见 ASIN 集合；
- 搜索页可见 ASIN 集合等于 footer 中可操作商品集合；
- 当前页所有商品均保留，不能只保留 top-k 子集。

任何一项失败都会抛出 `ObservationProjectionError`，当前样本标记为基础设施无效。

存在候选记忆时，投影器会先拆出记忆块：当前页面仍独享原预算，候选选择/记忆使用独立的 2,048-token
附加预算，最后再插回 footer 前。因此搜索页仍可使用完整 2,560 tokens，不会因记忆而进一步压缩；
若记忆本身超过附加预算，则显式失败。

### 12.7 Projection 元数据

每次投影生成：

```text
tool_name, page_type, token_budget,
raw_tokens, visible_tokens, truncation_ratio, truncated,
raw_asin_count, visible_asin_count,
raw_button_count, visible_button_count,
critical_footer_preserved
```

这些字段用于判断模型失败是否与 observation 压缩有关。例如 Guard rejection 发生在
`latest_observation_truncated=true` 时，会单独计数。

## 13. 步数提醒和循环恢复提醒

Evaluation 在环境返回 observation 后、投影前调用 `add_step_budget_notice()`。

### 13.1 步数提醒

当 `step_count > 35` 时插入：

```text
步数提醒: 目前 x/45 步，仅剩 y 步……
```

当前常量为 `STEP_BUDGET_NOTICE_AFTER=34`，判断条件是 `step_count > notice_after`，所以已执行第 35 步后
首次出现。terminal 页面永不插入提醒。

### 13.2 循环提醒

当环境 `progress.no_progress_steps >= 3` 时插入循环提醒，要求下一步获取新证据、购买已核验候选，
或充分探索后结束。它不指定具体商品，不读取隐藏目标。

### 13.3 提醒为什么放在页面类型之后

提醒紧跟 `page_type`，位于当前页面正文之前；候选记忆仍位于正文之后、动作 footer 之前。这样模型先确认
当前页面，再立即看到步数或循环风险，然后阅读最新页面证据和历史候选，最后依据完整按钮集合行动。长页面
压缩不会静默删掉临近终止的恢复信号。

### 13.4 三候选动态 System Prompt

普通 `SYSTEM_PROMPT` 不包含候选收敛说明。只有连续6步无实质进展，或执行达到30步仍未结束时，
Harness 才切换独立 `CANDIDATE_SYSTEM_PROMPT`，删除普通探索历史，并按 C1→C4 自动逐个 `reopen`。
每次切换候选都把模型上下文重建为 System＋原始用户需求＋当前候选详情；Evaluation 与 GRPO 入口
使用同一合同。

## 14. Context Window 管理

Observation 投影控制单次工具返回，上下文压缩控制整条历史。两者不能互相替代：即使每页都在
2,560 token 内，45 轮累计仍可能超过 30K 上下文。

### 14.1 Chat 级压缩：Evaluation

`compact_chat_messages()` 将消息拆成：

```text
anchor = 第一条 assistant 之前的 system/user 固定消息
groups = assistant tool-call + 对应 tool response 的完整交互组
```

压缩规则：

1. 固定 anchor 永不删除；
2. 最新完整交互组永不删除；
3. 从最旧组开始整体删除；
4. 二分搜索预算内可保留的最大最近组数；
5. 如果“固定 prompt + 最新 observation”仍超预算，抛 `ContextBudgetError`；
6. 不允许出现 assistant tool call 没有 tool response，或 tool response 对不上 call ID。

Evaluation 的外部 `messages` 仍保留完整轨迹，压缩后的副本只用于本次模型请求，因此轨迹审计不会
因为请求压缩而丢失原始历史。

### 14.2 Token 级压缩：GRPO

veRL AgentLoop 已把对话编码为：

- `prompt_ids`；
- `response_mask`：assistant token 为 1，tool observation token 为 0；
- `response_logprobs`。

`compact_token_trajectory()`：

1. 识别完整的 assistant-1 区段 + tool-0 区段；
2. 至少保护最近 1 个完整组；
3. 计算需要删除的 token 数；
4. 只在完整组边界删除旧 token；
5. 同步裁剪 prompt、mask、logprobs；
6. 数组长度不对齐立即报错；
7. 固定 prompt + 受保护最近组仍超预算则基础设施无效。

如果 trajectory 带有无法随 token 安全裁剪的 `routed_experts` 状态，发生实际压缩时会终止为
`context_compaction_unsupported_routed_experts`，而不是冒险让训练张量错位。

### 14.3 正式 GRPO 上下文合同

| 参数 | 值 |
|---|---:|
| Model context window | 30,000 |
| 单回合 generation reserve | 768 |
| safety margin | 512 |
| 显式 input budget | 28,720 |
| 保留最近完整组 | 1 |
| context compaction | 开启 |

计算关系：

```text
30,000 - 768 - 512 = 28,720
```

AgentLoop 每轮还会把 sampling `max_tokens` 限制为不超过 768，防止模型用长自然语言耗尽剩余上下文。

## 15. Environment HTTP Client 与租约

`ShopAgentEnv` 是一条 trajectory 独占的最小客户端：

```text
reset(task_id) → 保存 env_idx
step(action)*  → 使用同一个 env_idx
release()      → 释放 env_idx 并清空本地状态
```

### 15.1 生命周期状态机

```text
未租用
  ├─ reset 成功 → 已租用、done=false
  └─ step → ShopEnvironmentStateError

已租用
  ├─ step 非终局 → 已租用
  ├─ step 终局 → 已租用、done=true
  ├─ reset → ShopEnvironmentStateError
  └─ release → 未租用

done=true
  ├─ step → ShopEnvironmentStateError
  └─ release → 未租用
```

重复 `release()` 是安全空操作。HTTP 层区分：

- `ShopHttpError`：请求没有取得可用响应；
- `ShopEnvironmentError`：服务返回了环境错误；
- `ShopProtocolError`：JSON 结构不符合合同；
- `ShopEnvironmentStateError`：本地生命周期调用顺序错误。

### 15.2 为什么 GRPO 要在线程中调用同步客户端

客户端使用阻塞 HTTP。GRPO Adapter 通过 `asyncio.to_thread()` 执行 reset、step、release，避免阻塞
veRL 异步事件循环。每条 coroutine 用两个 `ContextVar` 绑定当前 env 和 runtime state，防止并发
trajectory 互相读取对方的 `latest_observation` 或 env_idx。

## 16. Environment 内部进度与 Termination v3.1

Environment 使用 `EvidenceProgressTracker` 独立于 Harness 记录真实执行动作。Guard 拒绝没有调用
`env.step()`，因此不会进入 Environment tracker。

### 16.1 默认终止参数

| 参数 | 值 | 含义 |
|---|---:|---|
| `max_steps` | 45 | 环境最多记录 45 个执行动作 |
| `exact_repeat_limit` | 2 | 连续重复计数达到 2 时终止 |
| `no_progress_limit` | 6 | 连续 6 个执行动作没有运行时新证据时终止 |
| `min_new_asins_per_result_set` | 3 | 一个结果集至少带来 3 个新 ASIN 才计入 abstain 证据 |
| `product_open_progress_budget` | 5 | 最多 5 个打开商品计入受限探索证据 |
| `subpage_progress_budget` | 6 | 最多 6 个新信息子页计入受限探索证据 |
| `result_set_progress_budget` | 3 | 最多 3 个结果集计入受限探索证据 |

### 16.2 精确重复的计数语义

动作签名由规范化的动作名和参数组成：搜索 query 会走搜索归一化，点击参数会去空白并 casefold。

```text
第一次 action A：consecutive_repeats = 0
第二次 action A：consecutive_repeats = 1
第三次 action A：consecutive_repeats = 2 → repeat_loop/exact_action_repeat
```

所以 `exact_repeat_limit=2` 实际意味着连续执行同一规范化动作三次时终止。

### 16.3 什么算“运行时进展”

以下任一新证据会把 `no_progress_steps` 清零：

- 新搜索/翻页结果集合，且至少有新 ASIN；
- 第一次打开某个 ASIN；
- 第一次打开某商品的某信息子页；
- 新的商品规格轴和值组合；
- 新的候选 hard-gate 证据状态。

没有新增则 `no_progress_steps += 1`，达到 6 后终止为
`repeat_loop/no_progress_loop`。

### 16.4 Runtime progress 与 credited evidence 的区别

Environment 同时记录两套概念：

- `runtime_progress_added`：用于判断当前动作是否真的带来新信息；
- `credited_evidence_added`：受预算限制、用于主动结束资格和统计的探索证据。

例如第 6 个新商品仍然属于运行时进展，可以打断 no-progress 连续计数，但因为
`product_open_progress_budget=5`，它可能不再增加受限 abstain 证据。这个区分避免 Agent 通过无限打开
商品刷结束资格，同时也避免真实新页面被误判为循环。

### 16.5 环境自动终止优先级

每次非终局 step 后按以下优先级判断：

1. 精确重复达到阈值 → `repeat_loop/exact_action_repeat`；
2. 无进展达到阈值 → `repeat_loop/no_progress_loop`；
3. 执行步数达到上限 → `max_steps/max_steps`。

一旦命中，Environment 自己生成固定终局 Reward；Harness 不再允许模型继续调用工具。

## 17. 主动结束 `finish_without_purchase`

Tool Schema 只允许：

```json
{"name":"finish_without_purchase","arguments":{"reason":"no_suitable_product"}}
```

Environment 在执行时读取：

- `effective_result_sets`：满足最少新 ASIN 要求、且在受限预算内的结果集数；
- `opened_candidates`：轨迹真实打开过的不同 ASIN 数；
- `known_acceptable_candidates`：Environment 已知满足候选资格的商品数；
- 当前 step count。

Reward v4 会把主动停止统一分类为 `early_abstain`，不再根据检索是否充分细分停止类型。
`finish_without_purchase` 只是一个合法终止请求，不是 Harness 保证的成功动作。

## 18. Reward v4 终局合同

Environment v2.4 的正式 Reward version 是 `shopsimulator-reward-v4`。基础终局类型：

| `reward_type` | 基础 utility | `purchase_success` | 严格成功 |
|---|---:|---:|---:|
| `gold_purchase` | 1.0 | 是 | 是 |
| `valid_alternative_purchase` | 0.8 | 是 | 否 |
| `partial_alternative_purchase` | `-0.3 + 0.5 × 匹配分` | 否 | 否 |
| `early_abstain` | -0.4 | 否 | 否 |
| `wrong_purchase` | -1.0 | 否 | 否 |
| `repeat_loop` | -0.6 | 否 | 否 |
| `max_steps` | 0.0 | 否 | 否 |
| `reward_unverifiable` | 0.0 | 否 | 否，且采样无效 |

Harness 侧另外定义两种模型行为负样本：

| 本地类型 | 基础 utility | 来源 |
|---|---:|---|
| `assistant_final` | -0.8 | 环境未终止时模型连续不调用工具 |
| `guard_rejection` | -0.8 | GRPO 中连续 3 次 Guard 拒绝 |

这两种不是 Environment `validate_reward()` 接受的原生 Reward v4 类型，而是 Harness 在确认属于模型
行为失败后构造的可学习负信号。

### 18.1 累计步数惩罚

终局 utility 还包含累计 step penalty：

| 执行步区间 | 每步 penalty |
|---|---:|
| 1–15 | 0 |
| 16–20 | -0.01 |
| 21–25 | -0.02 |
| 26–30 | -0.03 |
| 31–35 | -0.04 |
| 36–40 | -0.05 |
| 41–45 | -0.06 |

`terminal_utility = base_terminal_utility + step_penalty`。Harness 对正常环境终局不额外塑形，直接使用
Environment 已包含步数惩罚的 terminal utility。`assistant_final` 和 `guard_rejection` 则用同一份
累计 schedule 构造本地负奖励。

### 18.2 Reward 结构校验

GRPO Adapter 对 Environment Reward detail 验证：

- reward version 必须是 Reward v4；
- `reward_type` 必须属于已知集合；
- `termination_reason == reward_type`；
- `reward_valid` 必须是 boolean；
- 只有 `reward_unverifiable` 可以 `reward_valid=false`；
- `terminal_utility`、base utility、step count、step penalty 必须数值有效；
- `terminal_utility == base + penalty`；
- `sampling_invalid == not reward_valid`；
- hard gate 状态只能是 pass/fail/unverifiable；
- weighted score、evidence coverage、各维度分数在 `[0,1]`；
- query constraint version 必须匹配；
- constraint result 的状态和字段结构合法。

校验后只保留公共、最小化的诊断，不把完整隐藏 goal 或内部 evaluator 状态写进 GRPO runtime。

### 18.3 严格成功判定

任何统计、动态采样、评测汇总或面试说明都必须使用：

```text
done = true
AND terminal_result 完整
AND reward_type = gold_purchase
AND reward_valid = true
```

`purchase_success=true` 还包括 `valid_alternative_purchase`，不能替代 strict gold 指标。

## 19. Evaluation Harness：显式 while-loop

Evaluation 的核心入口是 `collect_for_task()`。它适合 Baseline/SFT/GRPO 模型的独立评测和轨迹采集。

### 19.1 初始消息

若 task 自带 prompt，则复制它；否则加入默认 System Prompt。若 prompt 中没有 system 消息，会在首位
补上默认 System Prompt。然后把 reset 返回的用户完整 instruction 作为 user 消息追加。此时不会出现
“目前已经至少有3个候选”，因为尚未打开任何商品。

初始结构化 observation 会保存为 `latest_observation` 供 Guard 使用；初始用户需求本身告诉模型应先
搜索，搜索首页 observation 不需要重复塞进 user 文本。

### 19.2 每轮模型调用

Evaluation client 可配置：

- `tool_choice=auto|required`；
- `max_tokens`；
- temperature / top-p / seed；
- context window 和安全余量；
- context compaction；
- observation 各页面预算；
- transient 网络重试；
- required tool call 缺失重试。

模型请求统一携带标准 `tools` 数组。特定模型端点可显式禁止 parallel tool calls，但 Harness 仍在
返回后执行串行化，不能只信任服务参数。

每轮调用前，Evaluation 根据最新页面和候选阶段计算动态 Schema。普通阶段不展示候选；进入强制阶段
后由 Harness 自动打开当前候选，规格未闭合仅暴露 `select_option/back_to_search`，闭合后仅暴露
`buy_now/back_to_search`，仍不跳过 Guard，也不替模型判断 Hard。

### 19.3 多 Tool Call

当前 Evaluation 行为是：

1. 保留第一个 tool call；
2. 丢弃其余调用；
3. 在 `tool_call_truncations` 中记录 kept call ID 和 dropped calls；
4. 环境最多执行一个动作。

当前 GRPO AgentLoop 已采用相同策略：保留第一个并累计 `tool_call_truncation_count`。这是为了避免多个
动作在同一个旧 observation 上批量执行。

### 19.4 模型没有 Tool Call

Evaluation 不会立刻终止，而是最多插入 2 次纠正 user 消息：

```text
上一回复触发768token截断，同时未包含工具调用，未执行。
禁止继续输出分析；现在必须且只能调用一个合法工具。
```

纠正不消耗环境 step。两次纠正后仍没有工具调用，轨迹终止为 `assistant_final`，基础 utility -0.8，
再加已执行步数的累计 penalty。

### 19.5 Guard 拒绝

Evaluation 把 assistant 消息和 Guard tool message 都写进 messages。连续拒绝达到 3 次时，当前轨迹
状态为 `invalid_action_limit` 并停止。若中间成功执行一个合法环境动作，连续拒绝计数归零。

需要注意：Evaluation 当前没有像 GRPO `run()` 那样把 `invalid_action_limit` 统一规范化为带完整
Reward detail 的 `guard_rejection` 终局；分析评测轨迹时必须查看 status、blocked calls 和 terminal
result，不能只看 `final_reward`。这是当前实现中的双入口终局封装差异。

### 19.6 合法工具执行

只有 Guard 通过后才：

1. 生成环境动作；
2. 调用 `env.step()`；
3. 记录 tool latency；
4. 渲染结构化 observation；
5. 添加步数和循环提醒；
6. 投影为模型可见 observation；
7. 追加 step 和 tool message；
8. 更新 `latest_observation`。

若 step `done=true`，立即保存环境 terminal result、final reward 并返回，不再请求模型总结。

### 19.7 Evaluation trajectory 主要字段

```text
trajectory_id, task_id, attempt_index, created_at, status,
messages, steps, blocked_tool_calls,
missing_tool_call_corrections, tool_call_truncations,
context_compactions, context_turn_tokens, model_calls,
initial_result, terminal_result, final_reward, done,
candidate_memory, error, release_error, timing, context_budget
```

`initial_result` 和 `terminal_result` 是评测审计对象，可能包含服务端返回的更多字段；这些字段不得被
自动追加到模型消息。模型只接收用户 instruction 和经过公共 renderer/projector 的 observation。

### 19.8 Evaluation 的基础设施失败

批量采集会在以下情况停止继续任务：

- release 失败；
- 模型 API 超时、断开或无法连接；
- ShopSimulator HTTP 失败；
- 环境资源池不可用。

普通任务失败仍应落盘并继续，避免把单个模型失误误判成全局基础设施中断。

## 20. GRPO Harness：veRL AgentLoop 适配

GRPO 不重写 veRL 的生成状态机，而是在父类 `ToolAgentLoop` 三个边界插入 Shopping 合同。

### 20.1 Session start

`ShoppingToolAgentLoop.run()` 从 parquet `extra_info.task_id` 读取任务号，创建
`ShopSimulatorSession`，异步 reset，并校验环境版本。reset 失败会尝试 release；版本不匹配也会先
释放再抛错。

### 20.2 生成前边界

`_handle_generating_state()`：

1. 记录历史最大输入 token；
2. 若启用压缩，执行 token trajectory compaction；
3. ContextBudgetError → infrastructure invalid；
4. 不启用压缩时，硬上限超出 → infrastructure invalid；
5. 若发生压缩且存在 routed experts → infrastructure invalid；
6. 同步替换 prompt ids、mask、logprobs；
7. 将本轮最大生成限制为 768；
8. 调用父类生成；
9. 若生成没有让 prompt 变长，区分 vLLM timeout 和 aborted，标记 infrastructure invalid。

### 20.3 Tool Adapter 执行边界

`ShopSimulatorTool.execute()`：

1. 从 ContextVar 取得当前 env/state；
2. 已终局则拒绝任何新工具；
3. 已达到 Harness max steps 则终止；
4. 记录动作尝试和最近 3 次签名；
5. 执行 Action Guard；
6. Guard 拒绝时累计计数，不触碰环境；
7. 连续 3 次拒绝 → `too_many_guard_rejections`；
8. 合法动作映射后在线程中执行 env.step；
9. 渲染新的公共 observation；
10. 追加可审计 step；
11. 环境终局时校验 terminal result 和 Reward v4；
12. 非终局时把原始 observation 放到 `_pending_raw_observation`，等待 AgentLoop 投影。

每次合法环境 step 会把连续 Guard 拒绝计数清零。

### 20.4 Tool observation 投影边界

父类 `_call_tool()` 返回后，Shopping AgentLoop 取出 `_pending_raw_observation`，使用真实 tokenizer
计数并执行 `project_observation()`。投影失败时：

- `termination_reason=observation_projection_failed`；
- `infrastructure_invalid=true`；
- tool response 替换为错误文本；
- 当前样本不产生正常学习奖励。

投影成功后，才更新模型真正可见的 `latest_observation` 和聚合统计。

### 20.5 Tool call 串行化边界

`_handle_processing_tools_state()` 若发现多个 tool calls，保留第一个，其余计入
`tool_call_truncation_count`，然后让父类只执行一个工具。工具执行后如果 runtime state 已标记
terminate，则直接进入 TERMINATED。

### 20.6 AgentLoop 结束后的模型失败规范化

父类循环结束后：

- 若原因是连续 Guard 拒绝，转成 `reward_type=guard_rejection`；
- 若没有环境 done、没有 infrastructure error，视为模型直接结束，转成
  `reward_type=assistant_final`；
- 两者都是 `reward_valid=true` 的模型行为负样本；
- 正常环境终局使用 Environment Reward；
- infrastructure invalid 总奖励为 0，并标 sampling invalid。

### 20.7 输出给 veRL 的 `shopping_info`

GRPO 输出至少包含：

```text
task_id, steps, done, termination_reason, error,
infrastructure_invalid, action_attempts, repeat_actions,
reward_mode, reward_version, reward_type, reward_valid,
reward_unverifiable, reward breakdown,
context compactions/tokens/max input,
observation projection/truncation/token/button/ASIN stats,
guard rejection stats, tool call truncations,
candidate memory version/entries/updates/search updates/evictions
```

它同时写入 `extra_fields.shopping` 和 `reward_extra_info.shopping`，供动态采样、奖励提取和训练日志使用。

### 20.8 finally release

无论父类正常返回、投影失败、生成失败还是 Reward 解析失败，`run()` 最终都会调用
`session.close()`。close 即使 release 抛错，也会在 `finally` 恢复 ContextVar 并清理本地绑定。

## 21. Evaluation 与 GRPO 的一致和差异

| 行为 | Evaluation | GRPO |
|---|---|---|
| 模型接口 | OpenAI-compatible chat completion | veRL + vLLM async AgentLoop |
| Tool Schema | `SHOP_TOOL_SCHEMAS` | `configs/tools.json` |
| Guard 判定 | 共享 `action_reject_reason` | 共享 `action_reject_reason` |
| Guard 反馈 | 详细恢复目标 | 简短错误文本 |
| 多 Tool Call | 保留第一个并记录其余 | 保留第一个并计数其余 |
| 缺少 Tool Call | 最多纠正 2 次 | 由父类结束后规范化为 assistant_final |
| Observation renderer | 共享 | 共享 |
| Observation projector | client 可配置 | AgentLoop 强制配置 |
| 上下文压缩 | chat message 组 | token/mask/logprob 组 |
| 连续 Guard 终止 | `invalid_action_limit` | `guard_rejection` 负样本 |
| Reward 结算 | 保存环境结果 | 严格 validate + breakdown |
| 租约释放 | 显式 try/finally | Session + try/finally |

共享函数只能证明局部合同相同，不能证明整个轨迹行为相同。每次 Harness 改动都要在两个入口分别测试。

## 22. 45 steps、42 history 和 40 turns 的多重边界

仓库当前存在三层相关限制：

1. ShopSimulator `MAX_HISTORY_LENGTH=42`，服务返回 `over = history > 42 or done`；
2. Environment Termination v3.1 `max_steps=45`；
3. GRPO `max_user_turns=40`、`max_assistant_turns=40`；
4. Shopping AgentLoop 自己也配置 `max_steps=45`。

这些数字的计数单位不同：环境 history 会记录 assistant actions，veRL turn 由父类状态机定义，Harness
steps 只记录通过 Guard 的执行动作。实际最早终止层可能不是文档口头所说的 45 步。

因此设计和排障时必须同时检查：

- 模型回合数；
- action attempts；
- Guard rejections；
- Harness executed steps；
- Environment tracker steps；
- 服务端 `over`；
- veRL user/assistant turn limit。

在没有专门行为测试前，不能假设这些计数天然相等。

## 23. 动作尝试、重复动作和步骤的统计口径

### 23.1 GRPO Harness 尝试签名

每个非 `think` 尝试使用：

```text
(tool_name, canonical JSON parameters, latest_observation sha256)
```

最近只保留 3 个签名。相同工具、相同参数、同一模型可见页面在最近三次内再次出现，就增加
`repeat_action_count`。如果页面 observation 已变化，即使工具参数相同，也不计 Harness 重复。

### 23.2 Environment 重复签名

Environment 使用动作名和规范化动作参数，不包含 observation fingerprint，并只看连续相同动作。
所以 Harness 的“最近三次相同页面重复率”和 Environment 的“连续精确动作循环”不是一个指标。

### 23.3 推荐同时报告

- `action_attempt_count`；
- executed step count；
- Guard rejection count/rate；
- repeat action count/rate；
- Environment consecutive repeats；
- Environment no-progress steps；
- tool call truncation count。

只看 steps 会漏掉大量非法尝试，只看 attempts 又会把未执行动作误当成环境成本。

## 24. 终局分类决策树

```text
trajectory 结束
├─ release/reset/model service/projection/context/protocol 异常？
│   └─ 是 → infrastructure_invalid，训练总奖励 0，采样无效
├─ Environment done 且 terminal result 完整？
│   ├─ reward_valid=false → reward_unverifiable，采样无效
│   ├─ gold_purchase → strict success
│   ├─ valid_alternative_purchase → purchase success，非 strict
│   └─ 其他 Reward v4 类型 → 有效失败/停止样本
├─ 连续 3 次 Guard 拒绝？
│   └─ GRPO → guard_rejection 有效负样本
├─ 模型未调用终止工具就结束？
│   └─ assistant_final 有效负样本
└─ 原因不完整或结构不可信
    └─ infrastructure_invalid
```

“奖励为 0”不能直接说明失败类型：`max_steps` 的原生 utility 可以是 0，`reward_unverifiable` 也是 0，
infrastructure invalid 也会输出 0，但三者是否可用于训练完全不同。

## 25. 典型端到端示例

### 25.1 正常搜索—打开—选规格—购买

```text
用户：要橙色、无线、充电、200–220 元的助听器

模型 → search_products({"query":"无线 充电 助听器"})
Guard → search 可用，允许
Env → search[...]，返回 20 个结果
Renderer → 搜索 Observation v2
Projector → 保留全部 20 个 ASIN

模型 → open_product({"asin":"12345678"})
Guard → ASIN 在最新搜索页，允许
Env → click[12345678]，返回详情

模型 → select_option({"value":"opt_4e6b4c0f2f48a310"})
Guard → stable ID 且在当前 footer，允许
Env → 更新 selected_options 和 selected_price=218

模型 → buy_now({})
Guard → 当前 footer 有 Buy Now，允许
Env → Reward v4 计算终局
Harness → 校验 reward_detail，若 gold_purchase 且 reward_valid=true，严格成功
finally → release
```

### 25.2 点击历史 ASIN 后恢复

```text
当前模型可见搜索页：ASIN A1–A20
模型却调用 open_product(A37)

Guard → click_not_in_previous_observation
Env → 完全不执行
Evaluation tool message → 列出当前 A1–A20 和导航按钮
模型 → open_product(A08)
Guard → 允许
连续拒绝计数 → 清零
```

### 25.3 在信息子页错误购买

```text
当前 page_type=information_subpage
footer buttons=["back to search","< prev"]
模型 → buy_now({})

Guard → click_not_in_previous_observation
Evaluation 反馈 → 明确当前只能 prev_page 或 back_to_search
Env → 页面不变
```

### 25.4 错把规格 label 当 ID

```text
available_options 显示 label="桔色", option_id="opt_..."
模型 → select_option({"value":"桔色"})
Guard → select_option_requires_stable_id
Env → 不执行
```

### 25.5 超长搜索页仍保留第 20 个商品

```text
raw observation > 2560 tokens
Projector → 二分压缩每个商品字段
验证 → 20 个原始 ASIN = 20 个可见 ASIN = footer 中 20 个商品目标
模型 → open_product(第20个ASIN)
Guard → 允许
```

如果无法在预算内保留全部 20 个商品，投影器报错并让 trajectory 无效，不会只保留前 10 个。

### 25.6 模型输出长分析不调用工具

Evaluation 最多纠正两次；仍无工具调用后生成 `assistant_final` 负奖励。GRPO 父类循环结束后也会把
无正常环境终局、无基础设施错误的情况规范化为 `assistant_final`。这类轨迹属于模型行为失败，应该
保留为学习信号。

## 26. 基础设施无效场景清单

以下场景不能当普通模型零分：

- Environment version 不匹配；
- reset/step/release HTTP 或协议错误；
- 公共 observation 带隐藏字段；
- 搜索 ASIN 与环境 action 不一致；
- option ID 与当前 action 不一致；
- 长 observation 缺少完整 footer；
- 无法在预算内保留全部动作目标；
- 投影后按钮/ASIN/footer 不变量失败；
- 固定 prompt + 最新 observation 超过上下文预算；
- token trajectory 的 mask/logprob 不对齐；
- vLLM 生成超时或无 token 推进；
- terminal result 没有 `done=true`、`over=true` 或 finite reward；
- Reward detail 类型、版本、utility 或有效性结构不可信；
- release 失败导致租约状态未知。

基础设施无效轨迹可以保留诊断，但 `reward_breakdown.total` 必须为 0，且标记
`sampling_invalid/infrastructure_invalid`，避免 GRPO 把平台故障学成模型行为。

## 27. 当前实现中需要特别警惕的细节

### 27.1 Reward detail 宽松回退

`adapter/tools.py` 在服务端 terminal detail 缺失或部分旧字段不完整时，存在按 top-level
termination type 和 scalar reward 构造最小公共 detail 的回退，还存在对“已知类型 + utility 匹配”的
兼容接受分支。它的目的不是改变公开 Reward v4 合同，而是避免服务 worker 少字段时把所有购买都误判
为 infrastructure invalid。

排障时必须记录实际 worker 返回结构；长期设计应优先让所有服务端严格返回完整 Reward v4，而不是
继续扩大训练侧回退范围。

### 27.2 Observation 投影会压缩多个商品字段

结构化搜索投影对 ASIN 后的每个 `|` 字段统一压缩。动作空间安全不等于语义信息完全无损。若模型在
投影后更容易误判价格、品类或关键属性，应按 `truncated` 切片评测，而不是只看 Guard 是否通过。

### 27.3 初始和终局审计对象可能比模型可见内容更丰富

Evaluation trajectory 会保存 server reset/terminal result，服务端其中可能带 goal options、purchase
或 Reward 诊断。这些只能用于受控审计，不能被拼回下一轮 messages，也不能进入训练 prompt。

### 27.4 双入口 Guard 反馈不同

Evaluation 能列出恢复目标，GRPO 只给短错误。若 GRPO 在非法动作后恢复率明显更低，应首先检查反馈
差异，而不是误以为模型策略本身退化。

### 27.5 多层步数上限可能提前结束

veRL 40 turns、服务 history 42、Environment/Harness 45 steps 需要定向行为测试。没有证据时，不应把
所有“未到 45 步就结束”都归因于模型 assistant final。

## 28. Harness 改动的最小测试矩阵

| 场景 | 必须断言 |
|---|---|
| Tool schema 同步 | Python schema 与 `configs/tools.json` 完全一致 |
| 合法搜索 | 正确映射 `search[...]`，环境执行一次 |
| 未知工具/错误参数 | Guard 拒绝，环境不执行 |
| 当前页 ASIN | 最后一个可见 ASIN 也能打开 |
| 历史 ASIN | 被拒绝且能恢复 |
| stable option ID | ID 允许、label 拒绝、导航按钮拒绝 |
| 搜索 20 商品 | renderer 和 projector 均完整保留 |
| 搜索 action 不一致 | renderer 失败关闭 |
| 超长详情 | footer 和全部按钮集合不变 |
| 长页无 footer | projection 失败关闭 |
| 步数/循环提醒 | 阈值正确、terminal 不插入、投影后仍存在 |
| chat 压缩 | 只删完整旧组、保留固定 prompt 和最新组 |
| token 压缩 | prompt/mask/logprobs 同步对齐 |
| 多 Tool Call | 只执行第一个并记录丢弃数 |
| 连续 Guard 拒绝 | 第 3 次终止；GRPO 为有效负样本 |
| 分隔的 Guard 拒绝 | 合法动作后连续计数归零 |
| assistant final | 有效负样本，带累计 step penalty |
| Gold purchase | Reward v4 utility 原样进入 GRPO，strict=1 |
| Valid alternative | purchase success=1，strict=0 |
| Reward unverifiable | sampling invalid，不产生学习信号 |
| 任意执行异常 | trajectory 留痕并 finally release |
| release 异常 | 明确标记生命周期故障 |

相关定向测试：

```powershell
python -m pytest `
  tests/test_shop_tools.py `
  tests/test_action_validation.py `
  tests/test_structured_observation.py `
  tests/test_observation_projection.py `
  tests/test_context_window.py `
  tests/test_rollout.py `
  tests/test_grpo_agent_loop_serialization.py `
  tests/test_shopping_reward.py -q
```

这组测试不启动训练、不合并模型，也不执行 240-task evaluation。

## 29. 可观测性指标建议

### 29.1 协议和动作

- unknown tool rate；
- schema error rate 和字段分布；
- Guard rejection rate；
- Guard reason 分布；
- 连续 3 次拒绝终止率；
- 多 tool call truncation rate；
- missing tool call correction rate；
- assistant final rate。

### 29.2 Observation

- 各 page type raw/visible token；
- projection trigger rate；
- truncation ratio 分布；
- footer failure；
- visible ASIN/button count；
- projection 后 Guard rejection rate；
- 截断与错误购买、循环、成功率的关联。

### 29.3 Context

- 每轮 input token；
- trajectory 最大 input token；
- compaction 次数；
- 删除 token、groups、messages；
- context budget exhausted rate；
- 压缩后重复搜索和重复打开率。

### 29.4 Environment 和生命周期

- reset/step/release latency；
- HTTP/protocol/environment error rate；
- 租约申请失败率；
- release error rate；
- Environment termination reason/subreason；
- no-progress 和 exact-repeat 触发分布。

### 29.5 Reward 和训练有效性

- `gold_purchase`；
- `valid_alternative_purchase`；
- partial/wrong purchase；
- graceful/early abstain；
- reward unverifiable；
- infrastructure invalid；
- terminal utility 和 step penalty 分布；
- 动态采样中可用 group、重采样和 drop 原因。

所有结果至少按“是否发生 observation truncation、context compaction、Guard rejection”切片，否则无法
区分模型能力变化和 Harness 输入变化。

## 30. 设计评审清单

每次修改 tools、observation、guard、projection、AgentLoop 或终局逻辑，逐项回答：

1. 模型可见字段是否仍然完全来自公共状态？
2. 是否可能泄漏 goal、target ASIN、Reward 判断或最佳候选？
3. Tool Schema Python 定义与 veRL JSON 是否完全同步？
4. 每个参数是否在环境执行前校验？
5. 最新可见目标、Guard 目标和环境可执行目标是否仍相等？
6. 搜索页是否保留当前页全部商品，而不是自行 top-k？
7. 详情页规格 ID、已选规格和 variant 价格是否一致？
8. 长 observation 是否完整保留 footer 和按钮？
9. 上下文压缩是否只删完整旧组并保持训练数组对齐？
10. Evaluation 与 GRPO 是否都有相同行为测试？
11. Guard 拒绝是否绝不触碰环境？
12. 多 tool call 是否最多执行一个动作？
13. 模型失败与 infrastructure invalid 是否无歧义？
14. Reward v4 是否由 Environment 权威返回并被严格校验？
15. 严格成功是否仍要求 `gold_purchase && reward_valid`？
16. reset、任意中间异常和 terminal 后是否都能 release？
17. 配置中的 turn、step、history、token 上限是否互相对齐？
18. 新指标能否证明改动没有制造隐性盲区？

## 31. 推荐源码阅读顺序

第一次完整学习建议按“协议 → 可见状态 → 合法性 → 压缩 → 生命周期 → 终局”阅读：

```text
1. src/shopping_grpo/environment/tools.py
2. configs/tools.json
3. environments/ShopSimulator/shop_env/web_agent_site/engine/observation.py
4. src/shopping_grpo/environment/observation.py
5. src/shopping_grpo/environment/actions.py
6. src/shopping_grpo/environment/projection.py
7. src/shopping_grpo/environment/context.py
8. src/shopping_grpo/environment/client.py
9. src/shopping_grpo/evaluation/rollout.py
10. src/shopping_grpo/training/grpo/adapter/runtime.py
11. src/shopping_grpo/training/grpo/adapter/session.py
12. src/shopping_grpo/training/grpo/adapter/tools.py
13. src/shopping_grpo/training/grpo/adapter/agent_loop.py
14. environments/ShopSimulator/shop_env/web_agent_site/engine/termination.py
15. environments/ShopSimulator/shop_env/web_agent_site/engine/reward.py
16. configs/agent_loop.yaml
17. configs/grpo.yaml
```

读完后应能独立回答：

- 模型这一轮具体看到了哪些字段和动作目标？
- 一个 Tool Call 在哪一层、因为什么被拒绝？
- Guard 拒绝后 Environment 是否发生任何变化？
- Observation 为什么可以压缩但不能删 ASIN/按钮？
- 长轨迹压缩如何保证 logprob 对齐？
- 一条 trajectory 为什么结束，是否能用于训练？
- strict gold、purchase success 和 reward valid 有什么区别？
- 任意异常后 env_idx 在哪里释放？

## 32. 最终心智模型

Harness 可以理解为六道串联的门：

```text
第 1 道：Tool Schema
  输出是不是一个受支持、参数合法的动作？

第 2 道：Action Guard
  这个动作是不是模型根据最新页面真实可执行的目标？

第 3 道：Environment Session
  动作是否在独立、有状态、未终局的商店实例中执行？

第 4 道：Observation Contract
  返回给模型的信息是否公开、结构稳定、没有答案泄漏？

第 5 道：Token Contract
  observation 和长历史是否在预算内，同时没有破坏动作空间或训练数组？

第 6 道：Terminal Contract
  结束原因、Reward、有效性和基础设施状态是否可信并可审计？
```

这六道门全部成立，模型学习到的才是“在真实页面状态上进行长程购物决策”；任何一道门放松，都可能
让训练结果变成对协议漏洞、历史按钮、截断盲区或基础设施噪声的拟合。
