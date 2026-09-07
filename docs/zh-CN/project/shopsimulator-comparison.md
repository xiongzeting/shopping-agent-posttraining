# ShopSimulator 原版与当前项目改造对照

本文记录 ShopSimulator 原版能力、当前项目的改造内容以及两者的归属边界，主要用于项目复盘、简历撰写和面试说明。

## 1. 对照基线

- 上游仓库：`ShopAgent-Team/ShopSimulator`
- 对照提交：`51bb26012cee31aea7ac26177c5ffe807026ac07`
- 当前项目中的环境：`environments/ShopSimulator`
- 当前项目嵌入版本信息：`environments/ShopSimulator/EMBEDDED_SOURCE.json`

ShopSimulator 原版提供商品快照、网页式购物环境、搜索系统、任务目标、基础 Reward 和评测入口。当前项目没有重新发明完整购物环境，而是在其基础上改造检索、价格、并发会话、工具协议、Observation、终止控制、Reward 以及训练评测闭环。

## 2. 总体能力边界

| 模块 | ShopSimulator 原版 | 当前项目的主要改造 |
|---|---|---|
| 商品环境 | 冻结淘宝商品快照和网页式购物流程 | 保留商品环境，补充版本、数据和索引身份校验 |
| 搜索 | Pyserini/Lucene BM25 | 可复现的 SQLite FTS5 多字段 BM25 |
| 原始动作 | `search[keywords]`、`click[value]` | 结构化 8 工具 Tool Schema |
| 页面信息 | HTML/Text 页面与四个信息子页 | 结构化 Observation 投影，将非空子页信息合并进详情 |
| 非法动作 | 非法动作通常 no-op | Action Guard 拒绝并返回明确纠正方法 |
| 上下文 | 原始页面与交互历史 | 页面预算、完整 Assistant–Tool 组裁剪、30K 上下文控制 |
| 候选记忆 | 无 | C1–C4 已核验候选记忆与强制候选收敛阶段 |
| Reward | 环境返回 Loose Reward，脚本统计 Strict/Success | Reward v3/v4 终局分档、Hard/Soft 语义合同和步数惩罚 |
| 并发 | 20 环境池，但缺少完整锁和会话身份隔离 | 环境池锁、单环境锁、独立 rollout session 和异常回收 |
| 终止 | Buy Now、历史/轮数上限 | Loop、Max-step、Assistant Final、Guard rejection、主动放弃等终局 |
| 训练评测 | 提供基础环境与单轮/多轮评测 | Teacher 数据、LoRA SFT、在线 GRPO、Final-240 和归因审计 |

## 3. 原版任务与购物流程

原版任务由具体商品构造，每条任务通常包含：

- 目标 `asin`
- 用户需求 `instruction_text`
- 简化需求 `instruction_simple`
- 商品品类 `category`
- 内部检索类别 `query`
- 目标属性 `attributes`
- 目标规格 `goal_options`
- 价格上限 `price_upper`
- 可选 Persona 信息

一次典型轨迹为：

```text
读取需求
→ 搜索商品
→ 浏览搜索结果
→ 打开商品详情
→ 查看描述、卖点、评论或属性
→ 选择商品规格
→ Buy Now
→ 计算终局 Reward
```

原版支持两种评测形态：

1. 单轮需求模式：用户在开头提供完整需求，Agent 多步操作购物环境。
2. 多轮 Shopper 模式：Shopper 模型先透露部分需求，Agent 需要追问，再与环境交互。

当前项目采用第一种形态：完整需求只在开头给出，后续是 Agent 与环境之间的长程工具交互，不提供追问用户的工具。

## 4. 搜索系统变化

### 4.1 原版搜索

原版使用：

```python
from pyserini.search.lucene import LuceneSearcher
```

检索链路为：

```text
搜索词 → Pyserini/Lucene BM25 → Top 150 → 每页展示20个商品
```

因此 BM25 是 ShopSimulator 原版能力，不是当前项目首次加入的算法。

原版的主要不足是：

- Lucene 索引构建配置没有完整进入项目的可审计合同。
- 中文切词和依赖版本可能随机器环境变化。
- 没有清晰记录各商品字段的检索权重。
- 没有固定的同分排序规则、商品数据 SHA 和索引 Manifest。

### 4.2 当前项目搜索

当前项目将其改造成 Search v2.1：

- 使用 SQLite FTS5 BM25，降低外部运行依赖。
- 分字段索引标题、品牌、品类、型号、属性、规格和卖点。
- 默认字段权重：标题 3.0、型号 2.5、品牌 2.0、品类 2.0、属性 1.5、规格 1.2、卖点 0.8。
- 中文使用确定性二元切词，并单独保留英文、数字和型号Token。
- 对查询执行 NFKC、大小写、标点和价格单位标准化。
- BM25 同分时按 ASIN 排序。
- 记录搜索版本、字段权重、商品数据 SHA、SQLite 版本和索引Schema。

准确表述应为：

> 在 ShopSimulator 原有 Lucene BM25 基础上，将搜索后端重构为面向中文商品的可复现多字段 BM25，增加字段加权、确定性中文切词、索引身份校验和稳定排序。

BM25 属于 ShopSimulator 环境搜索层，不属于 Agent Harness。Harness 负责把模型搜索词传给环境，并将返回页面投影成模型可见的 Observation。

## 5. 动作协议与 Tool Schema 变化

### 5.1 原版动作

原版环境只识别两类自由文本动作：

```text
search[关键词]
click[页面中的值]
```

点击值可能是：

- 商品 ASIN
- 规格值
- 上一页或下一页
- 返回搜索
- Description、Features、Reviews、Attributes
- Buy Now

原版要求模型输出 `Thought` 和 `Action` 文本，再通过字符串规则提取动作。格式变化、多个动作或非法按钮都可能导致解析失败。

### 5.2 当前项目动作

当前项目将原始 `search/click` 动作映射成8个结构化工具：

1. `search_products`
2. `open_product`
3. `select_option`
4. `back_to_search`
5. `prev_page`
6. `next_page`
7. `buy_now`
8. `finish_without_purchase`

同时实现：

- 单回合只执行一个工具。
- 页面级动态 Tool Schema，只暴露最新页面可执行的工具。
- 候选收敛阶段继续按照阶段缩小动作空间。
- Guard 校验工具名、参数、ASIN、按钮和规格ID。
- 非法调用不会静默 no-op，而是返回对应的修正方法。

## 6. 页面与 Observation 变化

### 6.1 原版页面

原版搜索页主要展示：

- 用户需求
- 页码和结果总数
- 商品 ASIN
- 标题
- 价格
- 翻页和返回按钮

原版详情页展示：

- 标题
- 当前价格
- 店铺
- 规格按钮
- Buy Now
- 四个信息子页入口

四个信息子页分别为：

- Description
- Features
- Reviews
- Attributes

部分商品对应字段为空，但入口仍可能显示，模型打开空页面仍会消耗动作并增加循环风险。

### 6.2 当前项目 Observation

当前项目删除四种低频信息工具，将非空的描述、卖点、属性等直接并入 `open_product` 后的商品详情 Observation。

当前投影预算为：

- 搜索页：2560 Token
- 商品详情：3072 Token
- 普通页面：512 Token
- 候选选择页面：1024 Token

投影必须优先保留：

- 可操作商品 ASIN
- 当前可点击按钮
- 规格轴与规格ID
- 当前已选规格
- 当前实际价格
- 关键商品属性
- Harness提醒和终局状态

旧历史按完整 Assistant–Tool 交互组裁剪，避免只保留工具调用或只保留结果，破坏动作与 Observation 的对应关系。

## 7. 候选记忆与Loop恢复

原版没有跨页候选记忆。模型返回搜索页后，只能依赖长上下文自行记住之前核验过的商品。

当前项目保存最多4个已核验候选，候选摘要主要包含：

- 候选编号 C1–C4
- ASIN
- 商品标题
- 当前核验价格
- 已选择或已确认规格
- 已核验的关键属性与证据
- 候选是否仍存在 Hard 违反

正常探索阶段不持续展示候选页，避免挤占每轮搜索和详情页面的Token。连续6步无实质进展达到原Loop阈值时，Harness不再直接判Loop，而是清理旧交互上下文并进入候选强制阶段：

```text
候选摘要比较
→ 打开一个候选
→ 补全规格
→ buy_now 或 finish_without_purchase
```

GRPO230 v3 r4冻结轨迹中：

- 34题达到6步无实质进展阈值。
- 34题全部被候选收敛状态机接管，但这不表示它们原本必然都会以Loop终止。
- 按当前Reward v4重新统计：8题最终Gold。
- 3题最终Valid。
- 候选阶段Gold+Valid为11/34，正确购买率32.35%。
- 其余为Wrong 15题、Partial 6题、Guard rejection 2题。
- 这34题中没有最终Repeat Loop。
- 整轮唯一Repeat Loop为Task 788，它没有进入候选收敛阶段。

这组数据属于旧候选列表版冻结轨迹：34条达到无进展阈值，其中11条最终完成Gold/Valid。当前顺序候选实现已改为 C1→C4 自动逐个展示和候选间上下文硬重置，应单独报告r4的6/33与r5的7/38，不能继续用旧34题结果代表当前Harness。

## 8. 原版 Reward

### 8.1 四个维度

原版主要检查：

- Type/Category
- Attributes
- Options
- Price

#### Type Reward

原版综合判断：

- 内部 `query` 是否相同
- 品类路径是否至少有两个公共节点
- 标题中的名词是否有一定重合

满足任一匹配条件时 `r_type=1.0`，否则仍为 `0.5`，因此错误品类也可能获得部分正分。

#### Attribute Reward

目标属性与商品属性使用 `token_set_ratio > 85` 模糊匹配；若未匹配，还会检查属性文本是否出现在标题、卖点或描述中。

```text
r_attribute = 匹配属性数 / 目标属性总数
```

#### Option Reward

规格先做颜色标准化，再通过模糊字符串匹配：

```text
r_option = 匹配规格数 / 目标规格总数
```

#### Price Reward

原版只支持简单价格上限：

```text
r_price = actual_price <= price_upper
```

它不能区分明确区间、严格低于、至少、大约、左右、多少元以上等自然语言价格语义。

### 8.2 Loose Reward

环境购买后返回：

```text
r_type ×
(匹配属性数 + 匹配规格数 + 价格是否满足)
/
(属性总数 + 规格总数 + 1)
```

Loose Reward提供连续信号，但不同维度可以互相补偿，而且错误品类仍可能获得正奖励。

### 8.3 Strict Reward

原版统计脚本另外计算：

```text
r_hard = r_type × r_attribute × r_option × r_price
```

Strict避免简单加法补偿，但任一维度为0后整体坍缩，差一个次要规格与完全错误的轨迹可能得到相同分数。

需要注意：在该上游代码快照中，环境 `step()` 直接返回的是Loose Reward；Strict主要由 `get_score.py`根据Reward明细离线计算。

### 8.4 Success与目标ASIN

原版存在两个不同指标：

- `r_success=1`：`r_type`、`r_attribute`、`r_option`和`r_price`全部为1。
- 选对商品率：最终购买ASIN与目标ASIN完全相同。

所以另一个商品如果满足全部评分维度，可以取得Success，但不会计入目标ASIN命中。Reward v3把这两个概念显式改造成 `gold_purchase` 和 `valid_alternative_purchase`。

## 9. Reward v3/v4变化

### 9.1 Reward v3

Reward v3将不同购买和未购买终局映射到不同奖励档位，主要解决：

- 唯一Gold过死。
- Loose补偿问题。
- Strict乘法坍缩。
- Loop、Max-step、主动停止和错误购买缺少区分。
- 同一Prompt的多个GRPO Candidate容易出现相同Reward。

它区分Gold、有效替代、部分替代、合理停止、过早放弃、Loop、Max-step和错误购买等终局。

### 9.2 Reward v4

Reward v4进一步根据用户Query冻结Hard/Soft语义：

- 品类始终为Hard。
- 明确不可妥协且可确定性核验的要求进入Hard。
- 偏好、近似表达和可折中需求进入Soft。
- 无所谓表达忽略。
- 高歧义内容进入Needs Review或仅审计。

判定原则：

```text
任一可评分Hard失败 → wrong_purchase
Hard全通过且命中目标ASIN → gold_purchase
Hard全通过且替代商品没有Soft失败 → valid_alternative_purchase
Hard全通过但存在Soft失败 → partial_alternative_purchase
```

另外增加：

- 第16步起的分段累计步数惩罚。
- `assistant_final`作为-0.8有效负样本。
- 连续Guard rejection作为-0.8有效负样本。
- 重新校准Partial、Loop及其他终局分数。

## 10. 价格与规格变化

原版商品价格可能从价格范围中调用全局随机状态采样，任务 `price_upper`也会随机生成，因此同一任务在不同进程和不同启动中可能产生不同价格条件。

此外，原版详情页选择规格后可能展示规格价格，但购买Reward读取的是商品级 `product_prices[asin]`，不一定对应完整已选Variant的实际价格。

当前项目改为：

- 商品价格根据ASIN确定性生成。
- 任务价格约束根据ASIN和Instruction确定性生成。
- Reward按照完整已选Variant的实际价格判断。
- 显式处理价格上限、明确区间、严格边界、近似价格和开放方向表达。

## 11. 并发与会话隔离变化

原版API预创建20个环境，并通过 `free_env_index`分配环境。但原版缺少：

- 环境池锁
- 单环境锁
- 独立rollout session身份
- interact会话所有权校验
- 完整异常回收

并发采样时可能出现环境重复分配、状态覆盖、旧请求操作新会话和异常后的资源泄漏。

当前项目增加：

- Flask线程化服务。
- 环境池锁和单环境锁。
- 独立 `rollout_session_id`，与任务 `idx`和环境 `env_idx`分离。
- reset分配空闲环境并绑定会话。
- interact同时校验环境编号和会话身份。
- 任务结束后自动释放。
- 分配后发生异常时自动回收。
- 完善 `release_one`、`release_all`和 `status`。

## 12. 终止与Action Guard变化

原版主要通过Buy Now、历史长度超过42条或评测脚本最大轮数结束。非法动作通常只导致页面不变化，没有明确错误分类。

当前项目补充：

- 45步环境上限。
- 35步和40步收敛提醒。
- 连续无进展提醒。
- 候选强制收敛。
- Assistant输出文字但未调用工具时的纠正。
- 工具名、按钮、ASIN、规格ID和页面合法性校验。
- 连续非法调用后的Guard rejection终局。
- `finish_without_purchase`主动结束合同。

Action Guard主要负责安全性和协议正确性，不再代替模型判断普通搜索是否“足够不同”。搜索质量与候选收敛主要交给Prompt、Observation提醒和候选状态机。

## 13. 训练与评测变化

原版仓库提供单轮和多轮评测入口，并统计：

- 完成率
- Loose Reward
- Strict Reward
- Success Reward
- 各维度分数
- 目标ASIN命中率

当前项目进一步形成：

```text
Teacher轨迹采集
→ 三层数据质量门
→ Assistant-only LoRA SFT
→ 在线LoRA GRPO
→ Final-240冻结评测
→ Reward重放、Rubric、LLM Judge和资源审计
```

评测将同一冻结轨迹分别送入：

- 确定性Reward v4重放
- 1,769条冻结Rubric检查
- 隔离Reward和Gold的LLM-as-a-Judge盲评
- Tool、Token、上下文、时延和终止行为审计

训练集与Final-240保持零重叠。

## 14. 面试时最容易说错的地方

1. 不能说“我为ShopSimulator加入了BM25”；原版已经使用Lucene BM25。
2. 可以说“我将原版BM25重构为可复现的中文多字段BM25”。
3. 不能说原版已经有8工具Tool Schema；原版只有 `search/click`自由文本动作。
4. 不能说原版有Action Guard、候选记忆或动态工具暴露。
5. 不能把原版Success与目标ASIN命中混为一谈。
6. 不能说原版环境直接返回Strict Reward；该快照环境主要返回Loose，Strict由统计脚本重算。
7. 不能把SQLite FTS5、中文二元切词、确定性价格和会话锁归到原版。
8. 不要简化为“候选模块将Loop从34降到1”；旧候选列表版和当前顺序候选版必须分开报告，当前r4/r5候选阶段Gold+Valid分别为6/33和7/38。
9. Harness贡献和模型训练贡献必须分开比较：固定Harness比较SFT与GRPO，固定模型比较Harness v2与v3。

## 15. 推荐项目表述

> 基于ShopSimulator冻结淘宝商品环境搭建长程购物Agent后训练闭环。原环境采用Lucene BM25和自由文本`search/click`交互；项目将其重构为可复现的中文多字段BM25，并修复随机价格、规格价格和并发会话隔离问题。在Agent执行层设计8工具动态Tool Schema、结构化Observation投影、Action Guard及C1–C4候选记忆，通过候选强制收敛降低长程Loop；进一步构建Reward v4 Hard/Soft语义合同、Teacher数据、LoRA SFT、在线GRPO和Final-240多维评测体系。

## 16. 一句话区分原版与当前项目

> 原版ShopSimulator负责提供“购物世界”；当前项目主要完成“可复现环境改造、Agent执行控制、训练Reward设计和SFT-GRPO后训练评测闭环”。
