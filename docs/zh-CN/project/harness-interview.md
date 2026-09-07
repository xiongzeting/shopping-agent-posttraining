# Harness、ShopSimulator 与轨迹变化面试大全

> 适用合同：ShopSimulator Environment v2.4、Observation v2、Tool Schema v2、
> Termination v3.1、Reward v4。本文只讨论当前 `Baseline → SFT → GRPO → Evaluation`
> 工作流，不引入历史兼容实现。

## 0. 这份文档怎么用

这份文档解决三类面试问题：

1. 模型、Harness、ShopSimulator 分别负责什么；
2. 一条真实 trajectory 如何从 System Prompt 变化到 Gold Purchase；
3. Prompt Token、Chat Template、Observation 投影、Guard、Termination 和 Reward 为什么这样设计。

建议按以下顺序准备：

```text
先背 30 秒回答
→ 再背 1 分钟回答
→ 能画出完整调用链
→ 能用 Task 927 解释每一步状态变化
→ 最后准备高频追问
```

---

# 第一部分：面试开场回答

## 1. 30 秒背诵版

> 整个系统分成模型、Harness 和 ShopSimulator 三层。模型是策略层，负责理解用户需求、比较候选并决定下一步调用哪个工具；Harness 是控制与编排层，负责 Prompt、Tool Schema、模型调用、Action Guard、动作转换、Observation 投影、上下文和轨迹管理；ShopSimulator 是有状态的环境层，负责加载商品和任务数据，真正执行搜索、点击、规格选择和购买，维护页面与 Session，并计算 Termination 和 Reward。简单说就是：模型负责想，Harness 负责管，Simulator 负责执行。

## 2. 1 分钟标准版

> 模型不会直接操作商城。它根据用户需求和最新 Observation，生成一个结构化 Tool Call，例如 `open_product` 或 `buy_now`。Harness 接收 Tool Call 后，先按 Schema 和最新页面状态做 Guard 校验，再把工具转换成 ShopSimulator 能识别的 `search[...]` 或 `click[...]` 动作。ShopSimulator 执行动作、更新 Session 和页面状态，返回原始 Observation；Harness 再根据页面预算做安全投影，把可见 Observation 加入消息历史并再次调用模型。购买后 Reward v4 由 ShopSimulator 根据品类、实际 variant 价格、功能、规格、目标商品和公开 Query Constraint 计算，Harness 负责记录结果并传给 GRPO 或评测系统。

## 3. 一句话记忆

```text
模型：决定做什么
Harness：保证这一步合法、紧凑、可记录
ShopSimulator：真正执行并返回世界变化
```

## 4. 一个好记的类比

```text
模型          = 司机
Harness       = 导航、交通规则、行车记录仪
ShopSimulator = 车辆、道路和真实交通环境
```

- 司机决定去哪里；
- 导航和规则系统检查动作、转换指令并记录过程；
- 环境决定执行后真正到了哪里、发生了什么。

---

# 第二部分：整体架构与边界

## 5. 完整闭环

```text
用户需求
   ↓
Harness 构造 System Prompt + User Message + Tool Schema
   ↓
模型 Provider 生成一个 Tool Call
   ↓
Harness Action Guard
   ↓
Harness 将 Tool Call 转换成环境动作
   ↓
ShopSimulator 执行 search[...] / click[...] / finish[...]
   ↓
ShopSimulator 更新页面、Session、规格、进度和终局
   ↓
ShopSimulator 返回 Raw Observation
   ↓
Harness 渲染并投影 Visible Observation
   ↓
追加 assistant/tool message 和 step
   ↓
未结束：重新调用模型
已结束：记录 Reward，释放环境租约
```

## 6. 三层职责总表

| 能力 | 模型 | Harness | ShopSimulator |
|---|---:|---:|---:|
| 理解用户要买什么 | 是 | 否 | 否 |
| 设计搜索关键词 | 是 | 否 | 否 |
| 比较候选商品 | 是 | 只保留证据 | 否 |
| 决定购买或放弃 | 是 | 否 | 否 |
| 公开 Tool Schema | 否 | 是 | 否 |
| 调用模型 Provider | 否 | 是 | 否 |
| 校验 Tool 参数 | 否 | 是 | 执行时兜底 |
| 阻止历史页面非法点击 | 否 | 是 | 当前页面兜底 |
| Tool Call 转环境动作 | 否 | 是 | 否 |
| 管理 Prompt 和 Messages | 否 | 是 | 否 |
| 统计 Provider Token | 否 | 是 | 否 |
| 投影 Observation | 否 | 是 | 否 |
| 保存完整 trajectory | 否 | 是 | 否 |
| reset/release 环境租约 | 否 | 是 | 提供服务端租约池 |
| 加载商品和任务数据 | 否 | 否 | 是 |
| 建立商品搜索索引 | 否 | 否 | 是 |
| 真正执行搜索和点击 | 否 | 否 | 是 |
| 维护页面与 Session | 否 | 否 | 是 |
| 计算 variant 价格 | 否 | 否 | 是 |
| 生成 Raw Observation | 否 | 否 | 是 |
| 记录探索进度 | 否 | 否 | 是 |
| 环境内 Termination | 否 | 否 | 是 |
| Reward v4 | 否 | 记录并转发 | 是 |

## 7. 最重要的状态所有权

| 状态 | 所有者 | 原因 |
|---|---|---|
| 用户购物需求 | Harness messages | 模型每轮都需要看到 |
| 模型策略判断 | 模型 | 不能由环境泄露答案 |
| 当前 `env_idx` | Harness Client + Simulator 租约池 | 一条轨迹独占一个环境槽位 |
| 当前页面 | ShopSimulator Session | 页面由环境动作决定 |
| 当前可点击目标 | ShopSimulator | 来自真实页面状态 |
| `latest_observation` | Harness | Guard 只允许最新可见目标 |
| Prompt 历史 | Harness | 每轮重新构造模型上下文 |
| 当前商品规格 | ShopSimulator | 选择 option 后由环境更新 |
| 当前 variant 价格 | ShopSimulator | 必须依据完整规格组合计算 |
| Reward evidence | ShopSimulator | 正常购买终局由 Reward v4 产生 |
| trajectory 落盘 | Harness | 供训练、评测和审计使用 |

## 8. 最重要的不变量

```text
模型最新可见 Observation 中的动作目标
        = Harness Guard 认为合法的动作目标
        = ShopSimulator 当前 Session 能够执行的动作目标
```

如果三者不一致，会出现严重问题：

- 模型看得到但点不了；
- Guard 允许了模型没看到的目标；
- 投影删掉 ASIN，但 Simulator 仍保留点击；
- 模型使用历史页面 ID 污染当前状态。

---

# 第三部分：模型负责什么

## 9. 模型的核心职责

模型负责策略，不负责基础设施。

具体包括：

1. 从用户原话提取品类、价格、品牌、型号、功能、规格和地区；
2. 设计具有区分度的搜索关键词；
3. 从搜索摘要中选择值得打开的候选；
4. 在详情页核验标题、品牌、品类、价格、关键属性和规格；
5. 选择正确的 stable option ID；
6. 规格变化后重新核验实际价格；
7. 判断继续搜索、返回、购买还是主动结束；
8. 避免重复查询、重复点击和无效循环。

## 10. 模型不应该做什么

模型不应该：

- 猜测隐藏 target ASIN；
- 读取 Reward 内部字段；
- 直接操作数据库；
- 绕过 Tool Schema 输出 `click[...]`；
- 点击历史页面 ASIN 或 option ID；
- 把 label 当作 stable option ID；
- 在品类不符时购买；
- 把“50 元左右”错误理解为“不超过 50 元”；
- 已经完全满足时继续无效搜索。

## 11. 模型和 Harness 的判断边界

```text
模型判断：这个商品是否满足用户需求
Harness 判断：这个动作是否符合协议并能在当前页面执行
```

Harness 可以检查：

```text
ASIN 是否出现在最新 Observation
```

但 Harness 不应检查：

```text
这个 ASIN 是不是最佳商品
```

否则 Harness 会从控制层变成答案层，破坏训练公平性。

---

# 第四部分：Harness 负责什么

## 12. Harness 的定位

Harness 是模型与 ShopSimulator 之间的 Agent Runtime。

它同时承担：

```text
协议适配
模型调用
安全守卫
上下文管理
Observation 投影
轨迹记录
生命周期管理
训练与评测接入
```

## 13. Harness 的输入层

第一次模型调用前，Harness 需要构造：

```text
System Prompt
User instruction
8 个 Tool Schema
Chat Template
Assistant generation prefix
```

System Prompt 规定：

- 只能使用工具与商店交互；
- 每轮只能调用一个工具；
- 当前页面是动作合法性的唯一依据；
- 规格选择后重新核验价格；
- 完全满足且 Buy Now 可用时立即购买；
- 不允许无效循环和历史点击。

静态 Prompt 不会提前声称已有候选。Final-240 Evaluation 只有在候选记忆实际达到 3 个后，才在下一轮
System Prompt 最前面动态加入“目前已经至少有3个候选”的收敛提醒；候选少于 3 个时完全不出现。

## 14. Tool Schema v2

当前公开 8 个工具：

| Tool | 参数 | 用途 |
|---|---|---|
| `search_products` | `query:string` | 搜索商品 |
| `open_product` | `asin:string` | 打开当前结果页商品 |
| `select_option` | `value:string` | 选择 stable option ID |
| `next_page` | `{}` | 下一页 |
| `prev_page` | `{}` | 上一页或返回详情 |
| `back_to_search` | `{}` | 返回搜索首页 |
| `buy_now` | `{}` | 购买并结束 |
| `finish_without_purchase` | `reason:enum` | 无可接受商品时主动结束 |

模型看到的是 JSON Schema，ShopSimulator 实际接收的是字符串动作。

## 15. Tool Call 到环境动作的转换

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

模型不需要学习 ShopSimulator 的原始字符串协议，Simulator 也不需要理解 OpenAI Tool Call JSON。

## 16. Action Guard

Guard 在动作发送到 Simulator 前执行。

检查顺序：

```text
工具是否存在
→ arguments 是否为 object
→ 必需参数是否齐全
→ 是否包含额外参数
→ 参数类型是否正确
→ 必需字符串是否为空
→ 当前页面是否允许搜索
→ ASIN 是否来自最新 Observation
→ option 是否为 stable ID
→ 按钮是否存在于最新 footer
```

常见拒绝原因：

```text
unknown_tool
schema_missing_arguments
schema_extra_arguments
schema_wrong_type
schema_empty_string
search_not_available_on_current_page
click_not_in_previous_observation
select_option_requires_stable_id
select_option_is_navigation_button
```

Guard 拒绝后不会调用 ShopSimulator，而是产生标准 tool error message，让模型下一轮纠正。

## 17. 为什么 Guard 只信最新 Observation

因为 ShopSimulator 是页面状态机：

```text
搜索结果页的 ASIN
商品详情页的 option ID
信息子页的返回按钮
```

都只在当前页面有效。

历史 Observation 可以用于比较候选，但不能直接作为动作来源。

## 18. Observation 投影

ShopSimulator 返回 Raw Observation 后，Harness 会：

```text
识别页面类型
→ 统计 Token
→ 未超预算则原样保留
→ 超预算则压缩正文
→ 再次验证 ASIN、按钮和关键 footer
```

页面预算：

| Page type | Token budget |
|---|---:|
| `search_results` | 2,560 |
| `product_detail` | 3,072 |
| `search_home` | 512 |
| `information_subpage` | 512 |
| `terminal`/generic | 512 |
| Candidate Memory（附加） | 1,024 |

Candidate Memory 使用独立附加预算：例如搜索页最多是 `2,560 + 1,024 = 3,584` tokens，记忆不会挤占
当前搜索页原有预算。

投影不是简单截断，必须保证：

```text
搜索页所有 ASIN 完整保留
所有可点击按钮完整保留
搜索状态 footer 保留
详情页 stable option ID 保留
投影后 Token 不超预算
```

## 19. 上下文管理

Harness 每轮会重新构造完整消息：

```text
System Prompt
+ User instruction
+ Tool Schema
+ 历史 Assistant Tool Call
+ 历史 Tool Observation
+ Assistant generation prefix
```

模型调用与工具执行的关系：

```text
模型调用
→ 生成一个 Tool Call
→ Harness 执行工具
→ Simulator 返回 Observation
→ 下一次模型调用
```

工具执行本身不是一次模型调用。

## 20. 轨迹记录

Harness 需要同时记录：

- `messages`：模型真正看到的对话序列；
- `steps`：工具执行和环境结果；
- `model_calls`：Provider latency、attempt 和 usage；
- `projection`：raw/visible Token 和动作不变量；
- `blocked_tool_calls`：Guard 拒绝；
- `context_compactions`：上下文压缩事件；
- `terminal_result`：Reward 和购买结果；
- `release_error`：环境释放异常。

## 21. 生命周期管理

```text
ShopAgentEnv 创建
→ reset(task_id)
→ 保存 env_idx
→ 多轮 step(action)
→ done=true 或 Harness 终止
→ release_one(env_idx)
```

关键要求：

- 一条 trajectory 独占一个环境租约；
- `reset` 前不能已有租约；
- `done` 后不能继续 `step`；
- 异常路径也必须尽力 `release`；
- 重复 release 是安全空操作。

---

# 第五部分：ShopSimulator 负责什么

## 22. ShopSimulator 不是数据仓库

ShopSimulator 包含两类能力。

静态资源：

```text
商品数据
任务数据
商品属性
规格和价格
搜索索引
```

动态交互：

```text
Session
页面状态
搜索
分页
打开商品
选择规格
Variant 价格
购买
Observation
Progress
Termination
Reward v4
```

所以更准确的定义是：

> ShopSimulator 是以商品和任务数据为基础的、有状态、可交互购物仿真环境。

## 23. Simulator 的页面状态机

```text
search_home
   ↓ search[keywords]
search_results
   ↓ click[ASIN]
product_detail
   ↓ click[option_id]
product_detail（新 variant）
   ↓ click[Buy Now]
terminal
```

还存在内部信息子页：

```text
product_detail
   ↓ click[Description | Features | Reviews | Attributes]
information_subpage
   ↓ click[< Prev]
product_detail
```

## 24. Simulator 维护的 Session 状态

包括但不限于：

```text
session_id
task index
current URL
current page type
current search query
current page number
current ASIN
selected options
selected variant price
current clickables
progress tracker
done/reward
```

## 25. Simulator 的动作执行

Simulator 支持的底层动作主要是：

```text
search[keywords]
click[value]
finish[reason]
```

它会解析动作，检查目标是否存在于当前页面，然后更新页面状态。

即使 Harness Guard 出现缺陷，Simulator 仍会在执行层拒绝不存在的当前页面目标。

## 26. Raw Observation

Simulator 负责生成结构化状态，例如搜索页：

```text
page_type: search_results
query
normalized_query
page / total_pages
products
actions
search_available
```

商品详情：

```text
page_type: product_detail
asin
title
brand
category
price
key_attributes
selected_options
available_options
actions
```

## 27. Description、Features、Reviews、Attributes

ShopSimulator 内部支持：

```text
click[Description]
click[Features]
click[Reviews]
click[Attributes]
```

字段映射：

| 子页 | 数据字段 |
|---|---|
| Description | `Description` |
| Features | `BulletPoints` |
| Reviews | `Reviews` |
| Attributes | `Attributes` |

Tool Schema v2 不公开：

```text
view_description
view_features
view_reviews
view_attributes
```

当前结构化 Observation 会删除这些不可调用按钮，并把商品数据中非空的 `BulletPoints`（Features）
和 `Attributes` 直接合并进商品详情。因此：

```text
Simulator 内部仍有信息子页
≠ 当前模型需要额外进入子页核验
```

## 28. Termination v3.1

Simulator 每步记录探索进度：

- 搜索结果集合；
- 新发现 ASIN 数；
- 打开候选数；
- option 证据；
- 重复动作；
- 无进展步骤；
- 是否达到环境强制终止条件。

环境内终止负责：

```text
购买终局
主动结束资格
重复/无进展终止
环境步数和证据进度
```

Harness 运行级终止负责：

```text
模型没有 Tool Call
多 Tool Call 截断
Provider 异常
上下文失败
Guard 失败策略
模型轮数限制
基础设施异常
```

## 29. Reward v4

正常购买终局由 Simulator 计算 Reward。

检查内容包括：

- 品类硬门槛；
- 实际 variant 价格；
- 核心功能；
- 品牌和型号；
- 规格选项；
- Query Constraint；
- target ASIN；
- 证据是否可验证。

严格 Gold 要求：

```text
完整购买终局
reward_type = gold_purchase
reward_valid = true
target_asin_match = true
公开 Query Constraint 满足
```

Harness 不重新计算正常 Reward，只负责接收、记录并传给训练或评测系统。

---

# 第六部分：真实 Task 927 完整变化

## 30. 样本身份

| 字段 | 值 |
|---|---|
| `task_id` | 927 |
| `trajectory_id` | `5e770d45-2910-4b3f-adda-c509498c0cd8` |
| Environment | `shopsimulator-environment-v2.4` |
| Observation | `shopping-observation-v2` |
| Termination | `shopping-termination-v3.1` |
| Reward | `shopsimulator-reward-v4` |
| 工具步骤 | 4 |
| Messages | 10 |
| Guard 拒绝 | 0 |
| Context compaction | 0 |
| 终局 | `gold_purchase` |
| Reward | 1.0 |
| `reward_valid` | true |

用户需求：

```text
坐标上海，求安利一款能贴在实木桌面上的保护膜，
需要有上门贴膜服务，价格在50元左右。
```

## 31. 总时间线

```text
Prompt assembly
→ reset(task_id=927)
→ search_products
→ open_product(rank 11)
→ select_option(stable option ID)
→ buy_now
→ Reward v4 gold_purchase
→ release_one
```

| Step | 模型 Tool | Environment action | 页面变化 | Reward |
|---:|---|---|---|---:|
| 0 | `search_products` | `search[实木桌面保护膜 上门贴膜]` | 首页→搜索结果 | 0 |
| 1 | `open_product` | `click[897757572316]` | 搜索结果→详情 | 0 |
| 2 | `select_option` | `click[opt_7a27ad05fc0c593c]` | 未选规格→实木家具 | 0 |
| 3 | `buy_now` | `click[Buy Now]` | 详情→terminal | 1.0 |

## 32. 阶段 0：Prompt 装配

第一次调用模型前包含：

```text
System Prompt                         1,233 token
User instruction                        26 token
8 个 Tool Schema JSON                  650 token
Chat Template 等                        328 token
──────────────────────────────────────────
Provider Prompt                       2,237 token
```

注意：`328` 是差值口径，包括：

- 角色特殊标记；
- Tools 固定英文说明；
- `<tools>`、`<tool_call>` 等协议标签；
- Assistant generation prefix；
- 空 thinking block；
- 拼接后的分词边界变化。

Provider 返回：

```json
{
  "prompt_tokens": 2237,
  "completion_tokens": 30,
  "total_tokens": 2267
}
```

## 33. 阶段 1：搜索

模型生成：

```json
{
  "name": "search_products",
  "arguments": {
    "query": "实木桌面保护膜 上门贴膜"
  }
}
```

Harness：

```text
Schema 校验通过
当前搜索可用
映射为 search[实木桌面保护膜 上门贴膜]
```

Simulator：

```text
执行搜索
返回第 1/8 页
总结果 150
当前展示 ranks 1-20
```

Observation：

> 以下 Token 数来自预算调整前的历史 Task 927 轨迹；按当前 2,560 搜索页预算，1,863-token 原页不会触发压缩。

```text
raw tokens       1,863
visible tokens   1,514
预算             1,536
ASIN             20 → 20
buttons          22 → 22
```

关键候选：

```text
rank 11
ASIN: 897757572316
price: 50.0
brand: 哈灵家具贴膜
attributes: 上海, 上门, 实木
```

## 34. 阶段 2：打开商品

下一次 Prompt：

```text
3,798 = 2,237 + 30 + 1,514 + 17
```

模型调用：

```json
{
  "name": "open_product",
  "arguments": {
    "asin": "897757572316"
  }
}
```

Guard 检查：

```text
ASIN 格式合法
ASIN 存在于最新搜索 Observation
不是历史页面目标
```

Simulator 返回详情：

```text
asin: 897757572316
title: 上海上门实木家具贴膜……
brand: 哈灵家具贴膜
category: 家居饰品›贴饰›家居保护膜
price: 50.0
key_attributes: 上海, 上门, 实木
selected_options: {}
available_options: 8 个 stable option ID
```

该详情只有 492 token，低于当时的 4,096 预算，也低于当前的 3,072 预算：

```text
492 → 492
```

## 35. 阶段 3：选择规格

下一次 Prompt：

```text
4,400 = 3,798 + 93 + 492 + 17
```

可选规格包括：

```text
桌面     → opt_b669eb30ba19716c
实木家具 → opt_7a27ad05fc0c593c
```

模型调用：

```json
{
  "name": "select_option",
  "arguments": {
    "value": "opt_7a27ad05fc0c593c"
  }
}
```

Guard 检查：

```text
ID 匹配 opt_[0-9a-f]{16}
ID 存在于最新 available_options
ID 不是导航按钮
```

Simulator 更新：

```text
selected_options.颜色分类 = 实木家具
variant price = 50.0
Buy Now 仍然可用
```

新详情 524 token：

```text
524 → 524
```

## 36. 阶段 4：购买

下一次 Prompt：

```text
5,055 = 4,400 + 114 + 524 + 17
```

模型重新确认：

```text
上海 ✓
上门 ✓
实木 ✓
家居保护膜品类 ✓
价格 50 元左右 ✓
```

模型调用：

```json
{
  "name": "buy_now",
  "arguments": {}
}
```

Harness：

```text
参数严格为 {}
最新页面存在 Buy Now
映射为 click[Buy Now]
```

Simulator：

```text
完成购买
进入 terminal
返回 Reward v4
动作空间变为 []
```

Terminal Observation：

```text
28 → 28 token
```

## 37. 阶段 5：Reward 与 release

Reward v4 的 6 个 Query Constraint：

| ID | 约束 | 实际证据 | 状态 |
|---|---|---|---|
| q0001 | 家居保护膜品类 | category | pass |
| q0002 | 50 元左右 | 50 ∈ [40, 60] | pass |
| q0003 | 上海 | attributes | pass |
| q0004 | 上门 | attributes | pass |
| q0005 | 实木 | attributes | pass |
| q0006 | 桌面 | 公开可见语义证据 | pass |

最终：

```text
reward_type = gold_purchase
reward = 1.0
reward_valid = true
target_asin_match = true
constraint pass = 6/6
```

一个重要细节：

```text
exact_target_variant_match = false
```

因为模型选择“实木家具”，不是 exact“桌面”。但 Reward v4 使用公开 Query Option 语义证据判定 q0006 pass，
且目标 ASIN 和其他约束全部满足，因此仍为 Gold。

最后 Harness：

```text
冻结 terminal_result
保存 trajectory
release_one(env_idx=1)
status = done
```

---

# 第七部分：Prompt、Chat Template 与 Token

## 38. 什么是 Provider Prompt

Provider Prompt 是模型在一次生成前实际读入的完整 Token 序列，不是用户新发的一句话。

```text
Provider Prompt
= System Prompt
+ User Messages
+ Assistant Messages
+ Tool Observations
+ Tool Schema
+ Chat Template
+ generation prefix
```

## 39. 什么是 Chat Template

API 输入通常是结构化对象：

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "tools": ["..."]
}
```

模型需要的是一维 Token 序列，因此 Qwen3.5 Chat Template 会渲染成类似：

```text
<|im_start|>system
# Tools
<tools>
...
</tools>
工具调用固定说明
System Prompt
<|im_end|>

<|im_start|>user
User instruction
<|im_end|>

<|im_start|>assistant
<think>

</think>
```

## 40. 为什么每次调用都重新带历史

标准模型请求本身没有天然跨请求记忆。

普通聊天也是：

```text
Turn 1: S + U1 → A1
Turn 2: S + U1 + A1 + U2 → A2
Turn 3: S + U1 + A1 + U2 + A2 + U3 → A3
```

购物 Agent 是：

```text
Call 0: S + Tools + U → Assistant Tool Call 0
Call 1: S + Tools + U + A0 + O0 → A1
Call 2: S + Tools + U + A0 + O0 + A1 + O1 → A2
```

聊天产品只是把历史拼接隐藏在后台，Harness 把它显式记录出来。

## 41. 15 步是不是注入 15 轮

在“一轮一个 Tool Call”的合同下：

```text
15 个工具步骤 ≈ 15 次模型调用
```

每次都逻辑上包含：

```text
System Prompt
User instruction
Tool Schema
历史消息
```

但不是每次固定 2,237 token，而是越来越长。

如果基础 Prompt 为 `B`，每步平均新增 `D`，调用次数为 `N`：

```text
第 i 次 Prompt ≈ B + iD

累计 Prompt
= N × B + D × N(N-1)/2
```

因此长轨迹的累计处理量可能接近二次增长。

## 42. 上下文占用和累计处理量

Task 927：

| Call | Prompt | Completion | Total |
|---:|---:|---:|---:|
| 0 | 2,237 | 30 | 2,267 |
| 1 | 3,798 | 93 | 3,891 |
| 2 | 4,400 | 114 | 4,514 |
| 3 | 5,055 | 82 | 5,137 |
| 合计 | 15,490 | 319 | 15,809 |

两个不同指标：

```text
上下文窗口安全性 = 最大单次 Prompt = 5,055
累计计算/API 量  = 所有调用相加 = 15,809
```

不能说模型某一时刻需要容纳 15,809 token。

## 43. Prefix/KV Cache

即使 Provider 使用 Prefix 或 KV Cache：

```text
逻辑上下文仍然包含完整历史
Prompt usage 通常仍报告完整 Token 数
但相同前缀可能不必从头计算
```

需要区分：

```text
逻辑上下文
网络传输
实际计算
Token 记账
```

---

# 第八部分：Observation 面试重点

## 44. 三种 Observation 口径

```text
Structured State
→ Simulator 内部结构化页面状态

Raw Observation
→ Harness 渲染后的完整文本

Visible Observation
→ 模型最终看到的投影文本
```

## 45. 只有 open_product 才有详细信息吗

不是。

```text
open_product
→ 第一次返回商品完整结构化详情

select_option
→ 返回规格变化后的新完整详情和实际价格
```

所有成功工具返回都进入 `project_observation()`，只是未超预算时原样返回。

## 46. 为什么搜索页更容易压缩

搜索页有 20 个商品，每个商品包含：

```text
rank
asin
price
brand
category
key_attributes
title
```

因此即使单个商品只是摘要，20 个商品合并后仍可能超过 2,560 token。

详情页通常只有一个商品，预算为 3,072，所以绝大多数页面仍不会压缩。

## 47. 投影为什么不能直接截断

如果从末尾硬截断，可能删掉：

- rank 11～20 的 ASIN；
- Next/Prev；
- Buy Now；
- stable option ID；
- 搜索是否可用；
- Guard 依赖的 footer。

Task 927 正确商品位于 rank 11，说明只保留 top-10 会直接破坏成功轨迹。

## 48. 投影失败如何处理

如果投影后出现：

```text
ASIN 数量变化
按钮集合变化
关键 footer 丢失
Token 仍超预算
```

应将轨迹视为 infrastructure invalid，而不是继续让模型在错误动作空间中运行。

---

# 第九部分：Guard、Simulator 校验和双层防线

## 49. 为什么两边都检查动作

Harness Guard：

```text
模型协议层检查
参数 Schema 检查
最新可见 Observation 检查
向模型返回明确纠错信息
```

Simulator：

```text
环境状态机最终检查
目标必须真实属于当前页面
非法动作不得改变内部状态
```

总结：

```text
Harness 防止错误请求到达环境
Simulator 保证环境即使收到错误请求也不被污染
```

## 50. 历史证据和当前动作的区别

```text
历史 Observation：可以用于记忆和比较
最新 Observation：决定下一步是否合法
```

例如模型以前看到商品 A，但已经进入商品 B 详情页：

```text
模型可以记得 A 的价格
但不能直接用历史 option ID 操作 A
```

---

# 第十部分：训练、评测和 Reward 边界

## 51. Harness 在 SFT 中的作用

SFT 数据来自规范化轨迹：

```text
System/User
Assistant Tool Call
Tool Observation
Assistant Tool Call
...
```

需要过滤：

- Guard 拒绝污染；
- 多 Tool Call；
- 错误参数；
- 无效终局；
- infrastructure invalid；
- 与评测任务重叠的数据。

## 52. Harness 在 GRPO 中的作用

GRPO Rollout 中 Harness 负责：

```text
模型生成
→ Tool 执行
→ Observation 投影
→ 下一轮生成
→ Terminal Reward
```

Reward 由 Simulator 返回，Harness 将其转换为训练框架需要的 reward score 和 extra info。

## 53. Harness 在 Evaluation 中的作用

评测与训练应共享：

```text
同一 Tool Schema
同一 Action Guard
同一 Observation v2
同一 Projector
同一 Environment v2.4
同一 Termination v3.1
同一 Reward v4
```

否则训练学到的行为和评测动作空间不一致。

## 54. `reward_valid` 为什么重要

```text
reward = 0
```

不一定表示模型失败，也可能是 Reward 或基础设施不可验证。

严格区分：

```text
reward_valid=true
→ 可以用于训练和正式成功率统计

reward_valid=false
→ Reward 不可信，应按 sampling/infrastructure invalid 处理
```

---

# 第十一部分：已落地优化与下一步功能

## 55. 已实现：Candidate Memory + Context Compaction

当前 Harness 会从模型已经看过的公开详情和搜索页提取：

```text
asin
title
brand
category
price
key_attributes
selected_options
key_attributes / features / attributes
source_query / source_page / source_rank
```

形成：

```text
[CANDIDATE_MEMORY_V2]
```

当前 Final-240 Evaluation 每条 trajectory 稳定保存最先核验的 4 个候选，编号为 C1-C4。同一 ASIN
重访时更新原记录；达到 4 个后不替换已有候选。普通探索阶段模型看不到候选记忆。连续6步无实质
进展，或执行达到30步仍未结束时，Harness 切换候选专用 System Prompt，并按 C1→C4 自动逐个
`reopen`；每次只向模型发送当前一个候选的完整详情，候选之间硬重置上下文。

安全边界：

- 只保存模型已看到的公开字段；
- 不保存隐藏 target、Reward 和答案；
- 只用于比较，不作为点击合法性来源；
- Guard 仍只信最新 Observation。

位置字段让模型知道候选来自哪个检索词和第几页，但不会自动跨页点击。若 ASIN 已在当前页可直接
打开；否则模型必须按位置重新搜索、翻页，待 ASIN 出现在当前页后再调用 `open_product`。

## 56. P0：Tool-only Canonicalization

Task 927 的模型在 Tool Call 前输出了自然语言，共约 193 plain token。

建议：

```text
保留 raw assistant response 用于诊断
但进入下一轮 messages 的 canonical content 置空
只保留一个 Tool Call
```

收益：

- 减少后续 Prompt；
- 保持严格 Tool-only；
- 净化 SFT 数据；
- 降低长轨迹啰嗦和漂移。

## 57. P1：受控 `inspect_product_info`

不建议直接开放四个 `view_*` 工具。

更适合增加一个：

```json
{
  "name": "inspect_product_info",
  "arguments": {
    "section": "description | features | attributes"
  }
}
```

第一版不开放 Reviews，并增加：

- 只允许当前详情页调用；
- 同一商品同一 section 只能查看一次；
- 已能从核心字段确认时拒绝；
- 子页使用 768-token 预算；
- 防止信息子页循环。

## 58. P1：Observation Diff

规格选择后可返回：

```text
selected_options: {} → 实木家具
price: 50.0 → 50.0
```

配合 Candidate Memory，减少重复详情。

## 59. P2：动态 Context Budget

根据剩余上下文动态调整：

```text
搜索页预算
详情页预算
旧 Observation compaction
Candidate Memory 大小
下一轮 Completion 预留
```

但动作不变量永远不能牺牲。

---

# 第十二部分：高频面试问答

## 60. Q：ShopSimulator 是不是数据库？

不是。它包含商品和任务数据，但核心是有状态交互环境：有搜索、页面、Session、规格、价格、
Termination 和 Reward。

## 61. Q：为什么模型不直接调用 `click[...]`？

因为模型侧需要稳定、类型明确、可校验的 Tool Schema；环境侧继续使用简单字符串动作。Harness 负责协议转换。

## 62. Q：为什么要 Action Guard？

模型可能产生错误参数、历史 ASIN、历史 option ID 或当前页不存在的按钮。Guard 在污染环境前拒绝，并给模型纠错信息。

## 63. Q：Simulator 已经校验动作，为什么还需要 Guard？

Simulator 的校验保护环境状态；Harness Guard 保护模型协议和轨迹质量，并能返回更明确的错误原因。

## 64. Q：为什么只允许一个 Tool Call？

多个 Tool Call 基于同一旧页面并行生成，第一个执行后页面可能已经变化，后面的动作会成为 stale action。

## 65. Q：Observation 为什么要投影？

搜索页和信息页可能很长。如果完整历史每轮重复携带，会快速消耗上下文和计算量。

## 66. Q：投影会不会改变答案？

正确投影只能压缩文本字段，不能删除当前页 ASIN、按钮、option ID 和关键 footer。否则应视为基础设施错误。

## 67. Q：为什么详情页预算比搜索页高？

详情页是购买决策的核心证据，需要完整保留品牌、品类、价格、属性和规格；搜索页主要用于候选发现，可以压缩文本摘要。

## 68. Q：`open_product` 一定会压缩吗？

不会。它只是返回详情页；只有详情超过 3,072 token 才压缩。Task 927 为 492→492。

## 69. Q：`select_option` 为什么也返回详情？

规格会影响 `selected_options` 和实际 variant 价格，模型必须重新核验，不能只得到“选择成功”。

## 70. Q：ShopSimulator 有 Attributes 页面吗？

内部仍有 `click[Attributes]`，但 Tool Schema v2 不公开 `view_attributes`，模型可见按钮中也会删除该入口；
非空 Attributes 已直接并入商品详情 observation。

## 71. Q：为什么不公开 Reviews？

Reviews 长、噪声大、主观性强，容易增加无效步骤，与当前结构化 Reward 证据关系较弱。

## 72. Q：Prompt Token 为什么每轮增长？

每轮重新携带固定 Prompt 和之前所有 Assistant/Tool 历史，所以 Observation 和输出会持续进入后续 Prompt。

## 73. Q：普通聊天也重新发历史吗？

逻辑上是。聊天产品可能发送完整 messages，也可能通过 conversation ID 在服务端恢复，还可能用 KV Cache 减少重复计算。

## 74. Q：15 个工具步骤是不是 15 次模型调用？

通常是。流程是“模型生成一个工具→执行工具→下一次模型生成”，终局工具执行后不再额外调用模型。

## 75. Q：15 步是不是上下文占用等于 15 次 Token 总和？

不是。上下文占用看最大单次 Prompt；计算/API 量看所有请求之和。

## 76. Q：Reward 是 Harness 算的吗？

正常购物终局不是。Reward v4 由 ShopSimulator 计算，Harness 负责记录和传给训练或评测。

## 77. Q：模型是否知道 target ASIN？

不知道。模型只能依据用户需求和公开商品信息判断；target ASIN 只用于 Reward 判定。

## 78. Q：为什么 exact option false 还能 Gold？

Task 927 选择“实木家具”而不是 exact“桌面”，但公开可见语义证据仍满足 q0006，其他 5 个约束和 target ASIN 也满足，因此 Reward v4 判 Gold。

## 79. Q：`reward_valid=false` 怎么处理？

不能当普通模型失败样本使用，应归为 Reward/infrastructure unverifiable，并从严格成功统计或训练采样中排除。

## 80. Q：Harness 最值得优化什么？

当前已完成安全 Candidate Memory 和 Context Compaction；下一优先级是 Tool-only Canonicalization，再考虑受控信息核验工具。

## 81. Q：Candidate Memory 会不会泄露答案？

当前实现只保存模型已经看到的公开字段和公开检索位置，不保存 target、Reward 或满足判断，也不把
历史 ASIN 不会在普通阶段加入动作合法集合，因此不是答案泄露。C1-C4 只是轨迹内稳定编号，不是候选排名；强制阶段
只要求模型收敛，不告诉模型哪个候选正确。

## 82. Q：为什么一定要 release？

环境槽位是有限资源。不 release 会造成租约泄漏，后续轨迹无法获得可用 env_idx。

## 83. Q：训练和评测 Harness 为什么要统一？

如果 Tool、Guard、Observation 或 Reward 合同不一致，模型训练时学到的策略在评测时可能无法执行，指标也无法公平比较。

## 84. Q：多 Tool Call 为什么危险？

模型可能同时输出“打开商品”和“购买”，但打开后页面已经改变，购买是基于旧页面生成的，不再可靠。

## 85. Q：模型为什么不能直接使用历史 option ID？

option ID 只属于具体商品和具体页面状态，离开页面后可能失效或指向错误规格。

---

# 第十三部分：白板画法

## 86. 最简白板图

```text
┌────────────┐
│   Model    │
│  决策策略   │
└─────┬──────┘
      │ Tool Call
      ▼
┌────────────────────────┐
│        Harness         │
│ Prompt / Schema / Guard│
│ Mapping / Projection   │
│ Context / Trajectory   │
└──────────┬─────────────┘
           │ search[] / click[]
           ▼
┌────────────────────────┐
│     ShopSimulator      │
│ Data / Search / Session│
│ Page / Variant / Reward│
└──────────┬─────────────┘
           │ Raw Observation
           └──────────────→ Harness → Model
```

## 87. 面试官追问时补充

在 Harness 和 Simulator 之间写：

```text
POST /api/shop_agent
reset / interact / release_one
```

在 Harness 内部写：

```text
latest_observation
Action Guard
project_observation
messages / steps / model_calls
```

在 Simulator 内部写：

```text
env_idx
session
current_url
selected_options
progress
reward_detail
```

---

# 第十四部分：常见错误回答

## 88. 错误：Harness 负责推荐商品

正确说法：

```text
模型负责判断哪个商品满足需求；
Harness 只负责动作协议、安全和运行控制。
```

## 89. 错误：ShopSimulator 只是存商品数据

正确说法：

```text
Simulator 同时维护 Session、页面、规格、价格、Termination 和 Reward，
是可交互状态机，不是静态数据库。
```

## 90. 错误：搜索结果压缩就是保留 top-k

正确说法：

```text
当前页所有可操作 ASIN 必须保留；
Task 927 的正确商品在 rank 11，只保留 top-10 会破坏答案。
```

## 91. 错误：累计 15,809 Token 就是上下文长度

正确说法：

```text
15,809 是 4 次请求累计处理量；
上下文峰值是 Call 3 的 5,055。
```

## 92. 错误：有 Prefix Cache 就不用携带历史

正确说法：

```text
缓存可以减少重复计算，但逻辑上下文仍然包含历史，
模型仍需要对这些 Token 建立注意力关系。
```

## 93. 错误：Reward 1.0 就一定可信

正确说法：

```text
还必须检查 reward_valid=true、完整 terminal result 和 Reward 证据。
```

---

# 第十五部分：源码跟读索引

## 94. Harness

| 问题 | 文件 |
|---|---|
| Tool Schema 和动作映射 | `src/shopping_grpo/environment/tools.py` |
| Action Guard | `src/shopping_grpo/environment/actions.py` |
| Structured Observation 渲染 | `src/shopping_grpo/environment/observation.py` |
| Observation 投影 | `src/shopping_grpo/environment/projection.py` |
| HTTP Client 和租约 | `src/shopping_grpo/environment/client.py` |
| 评测 Agent Loop | `src/shopping_grpo/evaluation/rollout.py` |
| GRPO Agent Loop | `src/shopping_grpo/training/grpo/adapter/agent_loop.py` |

## 95. ShopSimulator

| 问题 | 文件 |
|---|---|
| Shop Agent API 处理 | `environments/ShopSimulator/shop_env/shop_env/shop_agent.py` |
| 页面动作与 Gym step | `environments/ShopSimulator/shop_env/web_agent_site/envs/web_agent_text_env.py` |
| 页面模板与内部按钮 | `environments/ShopSimulator/shop_env/web_agent_site/engine/engine.py` |
| Structured State | `environments/ShopSimulator/shop_env/web_agent_site/engine/observation.py` |
| Termination | `environments/ShopSimulator/shop_env/web_agent_site/engine/termination.py` |
| Reward v4 | `environments/ShopSimulator/shop_env/web_agent_site/engine/reward.py` |

## 96. 真实数据

| 内容 | 路径 |
|---|---|
| Task 927 完整轨迹 | `重点/4.评测阶段/GRPO-step230-Final240-160gold/trajectories.jsonl` |
| Task 927 教学解剖 | `重点/5.harness设计/真实Gold轨迹逐步与Token解剖-task927.md` |
| Harness 完整设计 | `重点/5.harness设计/Harness超级详细设计说明.md` |

---

# 第十六部分：最终背诵模板

## 97. 终极 90 秒回答

> 这个项目的 Agent Runtime 分成模型、Harness 和 ShopSimulator 三层。模型是策略层，根据用户需求和最新公开 Observation 决定搜索、打开商品、选择规格还是购买。Harness 是控制与编排层，它负责构造 System Prompt 和 Tool Schema、调用模型、限制每轮一个 Tool Call、用 Action Guard 拒绝非法参数和历史页面目标，再把结构化工具转换成 Simulator 的 `search[...]` 或 `click[...]`。Simulator 是有状态购物环境，维护商品数据、搜索索引、Session、当前页面、规格和 variant 价格，真正执行动作，并生成 Raw Observation、Termination progress 和 Reward v4。
>
> 每次工具执行后，Harness 会按页面类型对 Observation 做 Token 投影，但必须完整保留当前页 ASIN、按钮和 stable option ID，然后把 Assistant Tool Call 与 Tool Observation 写入 trajectory，再进行下一次模型调用。模型请求逻辑上会重新携带固定 Prompt 和历史消息，所以长轨迹 Prompt 会持续增长；上下文安全看最大单次 Prompt，计算量看所有调用累计。
>
> 真实 Task 927 走了 `search_products → open_product → select_option → buy_now` 四步。搜索页从 1,863 压到 1,514 token，但 20 个 ASIN 和 22 个按钮全部保留；模型打开 rank 11 的正确商品，选择 stable option ID，最终以 50 元购买上海上门实木家具保护膜。Reward v4 判 6/6 Query Constraint pass、target ASIN match、reward_valid=true，因此得到 Gold Purchase 1.0。这个案例完整展示了模型负责决策、Harness 负责安全编排、Simulator 负责环境执行和 Reward 的边界。

## 98. 最后一句收尾

```text
模型提供智能，Harness 提供可靠性，ShopSimulator 提供可交互世界；
trajectory 则把三者每一次状态变化保存为可训练、可评测、可审计的数据。
```
