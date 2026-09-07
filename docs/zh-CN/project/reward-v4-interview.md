# Reward v4 完整解析与面试问答

> 本文以仓库当前实现和最新 Final-240 Reward 重放结果为准，更新时间为 2026-08-31。
> 当前总环境合同为 Environment v2.4，Reward 为 `shopsimulator-reward-v4`。

## 1. 一分钟说明

Reward v4 是一个不调用 LLM、只在终局执行的确定性购物 Reward。它先把用户 Query 编译成带来源证据的约束合同，再用商品结构化字段、可见文本、所选规格和实际成交价格逐项核验。

当前购买分类的核心规则是：

1. 任一可评分 Hard 约束失败，记为 `wrong_purchase`；
2. 没有已知 Hard 失败，但任一可评分 Hard 或 Soft 无法核验，记为 `reward_unverifiable`，该轨迹不进入 GRPO 更新；
3. Hard 全通过、但至少一个可评分 Soft 失败，记为 `partial_alternative_purchase`，目标 ASIN和替代 ASIN都使用该规则，奖励为 `0.5 + 0.3 × soft_score`；
4. Hard、Soft 全通过且购买目标 ASIN，记为 `gold_purchase`；
5. Hard、Soft 全通过且购买替代 ASIN，记为 `valid_alternative_purchase`；
6. 所有有效终局从第 16 个已执行工具步骤起叠加分段递增的累计步数惩罚。

它解决的不是“买没买”这一个二分类问题，而是同时回答四件事：

- 是否违反用户不可妥协的要求；
- 替代商品对可折中偏好的满足程度；
- 模型是否正确结束任务；
- 是否用过长轨迹完成本可更快完成的任务。

## 2. 版本边界：面试时最不能说错的地方

### 2.1 当前冻结口径

当前代码中的基础终局效用如下：

| Reward 类型 | 基础效用 |
|---|---:|
| `gold_purchase` | `1.00` |
| `valid_alternative_purchase` | `0.80` |
| `partial_alternative_purchase` | `0.50 + 0.30 × soft_score` |
| `max_steps` | `0.00` |
| `early_abstain` | `-0.40` |
| `repeat_loop` | `-0.60` |
| `assistant_final` | `-0.80` |
| `guard_rejection` | `-0.80` |
| `wrong_purchase` | `-1.00` |
| `reward_unverifiable` | `0.00`，但 `reward_valid=false` |

当前实现还包含：

- Reward Features v2；
- Query Constraints v1；
- Constraint Semantics v3；
- Price Semantics v4；
- Step Penalty v1；
- Termination v3.1。

### 2.2 GRPO230 训练时与当前评测时不是完全相同的 Reward 快照

GRPO230 的 2026-08-22 训练日志已经使用 Reward v4，并且已经包含第 16 步开始的累计步数惩罚。日志中可以直接看到：

- 18 步 Gold：`1.00 - 0.04 = 0.96`；
- 24 步 Gold：`1.00 - 0.14 = 0.86`；
- 32 步 assistant final：`-0.80 - 0.39 = -1.19`。

但训练时使用的是较早的 Reward v4 快照：

- Partial 仍是 `-0.30 + 0.50 × match_score`；
- `repeat_loop` 基础值仍是 `-0.80`；
- Hard/Soft 还没有扩展为当前这套完整 Query 语义合同。

2026-08-24 的最终审计版进行了三类修正：

- 将“必须、一定、绝对不要、不超过”等高置信语义纳入 Hard/Soft 解析；
- 将 Partial 调整为 `0.50 + 0.30 × soft_score`；
- 将 `repeat_loop` 调整为 `-0.60`。

因此，严谨表述应是：

> GRPO230 在 Reward v4 的终局分段与效率惩罚框架下完成训练；训练结束后，我又对 Hard/Soft 语义和少数终局效用做了可审计修订，并在冻结轨迹上离线重放当前 Reward，用于统一 Final-240 结果口径，没有把新的评测标签反向用于训练该 checkpoint。

这不是“改分作弊”，原因是：

- 模型参数没有根据 Final-240 新结果继续更新；
- 六个模型使用相同重放程序和相同任务定义；
- Rubric、LLM-as-a-Judge、步骤、Token、时延等非 Reward 指标没有跟着改；
- Gold 商品回放必须逐题审计；在“Gold 也必须满足 Soft”的当前口径下，238题为Gold、2题因Soft价格失败降为Partial；
- 旧结果和新结果均保留，可逐任务查看类型迁移。

## 3. 为什么需要 Reward v4

### 3.1 单一二元 Reward 不够

购物 Agent 的失败并不只有一种：

- 买错品类；
- 品类正确但违反明确预算、品牌、型号、功能或规格；
- 没买到 Gold，但买了满足公开需求的合理替代；
- Hard 满足，但只违反“最好、左右、尽量”等软偏好；
- 搜索后合理停止；
- 没充分检索就停止；
- 循环、超步数、非法动作或直接文本结束。

如果只使用 `成功=1，失败=0`，这些行为会被压成同一个分数，GRPO 无法区分“接近正确的替代购买”和“明显错误购买”，也无法学习何时继续搜索、何时停止。

### 3.2 Reward v4 的设计目标

Reward v4 的目标包括：

- **正确性：** Hard 失败必须比不购买更差，防止为了得到购买奖励而抢购；
- **可学习性：** 给合理替代和部分满足提供分层信号；
- **可审计性：** 每项约束都保留 Query 原文、比较器、实际值来源和结果；
- **无泄漏：** 不把 Gold 商品所有私有属性自动解释成用户要求；
- **稳定性：** Reward 不调用 LLM，在线训练时不会受模型服务波动影响；
- **效率：** 对过长轨迹统一扣分；
- **异常隔离：** 环境证据不可验证和基础设施失败不能伪装成模型失败。

## 4. 从 Query 到 Reward 的完整链路

```text
公开 Query + 任务标注 + 目标商品
            │
            ▼
compile_reward_features
            │
            ├─ 提取公开品牌、型号、功能、价格和规格
            ├─ 记录 Query evidence 与来源
            ├─ 区分 Hard / Soft / Ignore / Needs Review
            └─ 冻结 shopping-query-constraints-v1
            │
            ▼
Agent 执行搜索、浏览、选规格、购买或停止
            │
            ▼
终局商品 + selected_options + 实际 variant price + step_count
            │
            ▼
逐约束确定性比较
            │
            ├─ Hard fail → Wrong
            ├─ Hard / Soft unverifiable → Invalid
            ├─ Hard pass + Soft fail → Partial（目标/替代均可）
            ├─ Hard / Soft pass + Gold ASIN → Gold
            └─ Hard / Soft pass + Alternative → Valid
            │
            ▼
基础终局效用 + 累计步数惩罚
            │
            ▼
GRPO adapter 校验、最小化并写入训练轨迹
```

Reward 只在终局计算。Agent 行动前不会看到目标商品的私有评分事实，也不能在交互过程中修改评分合同。

## 5. Query 约束合同

每个任务在 Rollout 前生成 `shopping-query-constraints-v1`。每项约束至少包含：

- `constraint_id`：任务内稳定编号，如 `q0001`；
- `constraint_type`：类别、价格、品牌、型号、功能、规格等；
- `role`：Hard Gate、Matching Dimension、Strict Query Variant、Gold Reference 等；
- `expected`：期望值；
- `source`：数据来自公开 Query、任务类目合同还是 Gold 标注；
- `query_evidence`：允许影响 Reward 的 Query 原文证据；
- `optional_query_evidence`：只表示可选或偏好的证据；
- `strength`：`hard / soft / ignore / needs_review`；
- `enforcement`：`scored / audit_only`；
- `polarity`：要求、禁止、偏好、无所谓或未知；
- `semantics_reason`：为什么得到该语义标签。

同时保存 Query 的 SHA-256。环境可以检查 Reward Features 与当前任务文本是否来自同一条指令，防止任务文本和评分合同错位。

### 5.1 为什么不能直接使用 Gold 商品全部字段

目标商品中可能包含用户没有说过的型号、颜色、赠品或隐藏规格。如果把这些字段全部当成硬要求，会出现两种问题：

- 合理替代商品被错误判为失败；
- Reward 实际优化的是隐藏标注，而不是用户公开需求。

因此当前合同遵循：

- 类目来自任务类目合同；
- 品牌、型号、功能和规格只有得到 Query 原文支撑才允许参与购买判定；
- 仅用于 Gold 对照、但没有公开 Query 支撑的字段标为 `ignore`；
- 无法安全实现的复杂语义保留为 `audit_only`，交给 Rubric/Judge 审计，而不制造确定性误判。

## 6. Hard、Soft、Ignore 和 Needs Review

### 6.1 四种语义强度

| 强度 | 含义 | 对购买分类的影响 |
|---|---|---|
| `hard` | 用户不可妥协的要求 | Fail→Wrong；Unverifiable→Reward Invalid |
| `soft` | 可折中的偏好 | 替代商品失败时降为 Partial |
| `ignore` | 无所谓、让步或只属于 Gold 私有参考 | 不参与 Reward |
| `needs_review` | 规则无法可靠确定语义 | 当前新合同中只审计，不阻断 Reward |

### 6.2 `strength` 与 `enforcement` 不是一回事

某条 Query 分句可以被识别为 Hard，但如果当前没有可靠的确定性比较器，它仍会设置为：

```text
strength = hard
enforcement = audit_only
```

例如“必须是原厂家店铺的真货”语义上显然很硬，但快照商品字段未必足以可靠判断真伪。此时强行打分容易产生假阴性，所以保留在审计合同中，由 Rubric/轨迹 Judge 衡量，不直接把购买判成 Wrong。

只有 `enforcement=scored` 的约束会进入 Reward 决策。

### 6.3 当前确定性语义规则

高置信 Hard 标记包括：

- `必须、务必、一定要、一定得、非……不可、仅限`；
- `绝对不要、坚决不要、严禁、禁止、不得、不可`；
- `不超过、不能超过、不高于、至少、不低于、不少于、以内、以下、以上、之间、控制在`；
- 普通的明确“不要……”，但“不要太……”按 Soft 处理。

Soft 标记包括：

- `最好、优先、尽量、希望、偏好、倾向`；
- `左右、上下、大概、大约、约、接近、附近、差不多`；
- `越……越好、不要太、别太`；
- 被“最好、尽量、预期、预计、预估”等修饰的价格边界。

Ignore/让步标记包括：

- `无所谓、不要求、不限、任意、随便、都行、可有可无`；
- `也可以、也行、没有……也可以`。

`不需要、无需、不用`具有歧义：

- “不需要售后”通常是无需求或无所谓；
- “不需要内搭短裤”可能是在禁止某种商品规格；
- “不用工具也能拆装”中的“不用”又是能力描述。

因此无法由结构化目标证据消歧时标为 `needs_review + audit_only`，不偷偷转成 Hard 或 Soft。

### 6.4 背景叙述不应成为商品要求

规则会排除“准备、打算、计划去……”等叙事背景。例如：

- “准备去旅游，想买白色相机”中，“旅游”是购买背景，不要求商品标题必须出现旅游；
- “给老人买提醒器”中的“老人”是使用对象，不应自动变成产品功能；
- “用于比赛训练”可能是动机，真正可评分的是耐磨、防滑、规格等商品要求。

## 7. 支持的约束类型与比较器

### 7.1 类目

类目始终是 Hard。比较器不是简单子串匹配，而是：

- 叶子类目必须一致；
- 若双方都有祖先链，还需要祖先存在交集。

这样可减少“同名叶子但属于不同大类”的误匹配。

### 7.2 品牌

品牌证据按以下顺序检查：

- 结构化 `brand` 或 `shop_name`；
- 商品标题中的安全品牌提及；
- 属性字段中与品牌别名精确一致的证据。

同时防止兼容配件攻击。以下文本不能单独证明品牌：

```text
兼容飞利浦
适用苹果
适配小米
用于华为设备
```

“兼容飞利浦的第三方刷头”不能因此被判成飞利浦品牌。

### 7.3 型号

型号使用 Token Boundary，而不是任意子串。例如 Query 中的 `A20` 不能被商品价格 `2020` 或更长型号的一部分误匹配。

### 7.4 核心功能

优先核验结构化属性，必要时回退到标题、短描述、Bullet 和完整描述，并考虑否定前缀。

例如商品文本“没有热洗功能”不能因为出现“热洗”二字就判通过。

### 7.5 规格与 Option

规格比较包括：

- 规格轴别名归一化，如颜色、颜色分类；
- 规格值精确匹配；
- 数量、颜色等显式冲突检测；
- Query 公开数值范围的语义回退。

例如 Query 要“15cm 以下”，Gold 标注是 12cm，而模型选择同一商品的 9cm：

- 9cm 满足用户公开范围；
- 可以通过公开规格约束；
- `exact_target_variant_match=false` 仅作为诊断；
- 如果所有可评分 Soft 也通过，则目标 ASIN可为 Gold、替代 ASIN可为 Valid。

选择 16cm 则违反 Hard，判 Wrong。

商品标题同时写“3罐/5罐可选”，但模型实际选择 3 罐时，不能用标题中的 5 罐覆盖已选规格的显式冲突。

### 7.6 价格

价格来自所选 variant 的实际可核验价格，而不是只取商品展示的最低价。

支持：

- 明确上限：`不超过300元、300元以内、最高300元`；
- 明确下限：`至少300元、不低于300元`；
- 明确区间：`300到400元之间`；
- 近似价格：`300元左右、接近300元`；
- 中文数字与混合表达；
- 每包、每平方米等单位价格。

近似价格使用固定 ±20% 区间，例如：

```text
300 元左右 → [240, 360]
20 元左右  → [16, 24]
```

`300多元、三百来块、大概300元出头`属于开放表达，不会被误解析成 `<=300` 的硬上限。

若 Query 为“50包，每包约1.5元”，Reward 会从已选规格中解析数量，用 variant 总价除以包数后比较单价。

### 7.7 四个匹配维度、覆盖率和当前分类分数

实现中还保留四个等权诊断维度：

| 维度 | 权重 |
|---|---:|
| 品牌 | `0.25` |
| 型号 | `0.25` |
| 核心功能 | `0.25` |
| 关键规格 | `0.25` |

只有任务实际激活的维度进入分母：

```text
match_score = Σ(weight_i × dimension_score_i) / Σ(active_weight_i)
evidence_coverage = Σ(weight_i × coverage_i) / Σ(active_weight_i)
```

需要区分三个容易混淆的量：

- `dimension_scores / match_score`：品牌、型号、功能、规格的诊断分数，也用于轻量候选可接受性判断；
- `weighted_score`：当前购买结果中记录的 Hard Pass 比例；
- `soft_score`：可核验 Soft 中的通过比例，当前 Partial 公式只使用这个分数。

旧 Reward 快照曾直接用匹配维度分数计算 Partial；当前 Hard/Soft 合同已经成为购买分类主路径，四维分数主要用于诊断、过程审计和停止资格的候选判断。

## 8. 购买终局决策树

设：

- `H` 为所有可评分 Hard 约束；
- `S` 为所有可评分 Soft 约束；
- `soft_score = Soft Pass 数 / 可核验 Soft 数`；
- Soft 的 `unverifiable` 不会被当作通过，而会使该购买成为 `reward_unverifiable`；
- 没有可评分 Soft 时，`soft_score=1`。

决策顺序为：

```text
存在 Hard fail？
├─ 是 → wrong_purchase
└─ 否
   └─ 存在 Hard / Soft unverifiable 或阻断型 needs_review？
      ├─ 是 → reward_unverifiable，reward_valid=false
      └─ 否
         └─ Hard 全通过？
            ├─ 否 → wrong_purchase
            └─ 是
               └─ 存在 Soft fail？
                  ├─ 是 → partial_alternative_purchase（目标/替代均可）
                  └─ 否
                     └─ target ASIN match？
                        ├─ 是 → gold_purchase
                        └─ 否 → valid_alternative_purchase
```

### 8.1 Gold 的准确含义

当前代码中的 Gold 是：

```text
目标 ASIN + 所有可评分公开 Hard 与 Soft 约束通过
```

它不要求模型命中用户没有公开表达的隐藏 Gold 规格。隐藏规格是否精确命中通过 `exact_target_variant_match` 单独审计。

这避免用私有标注惩罚满足用户真实 Query 的行为。

### 8.2 Valid Alternative

Valid 是：

```text
替代 ASIN + 所有可评分 Hard 与 Soft 全通过
```

Gold 和 Valid 都计入 `purchase_success`，但只有 Gold 计入严格成功。

### 8.3 Partial 为什么仍然存在

Partial 不是“Hard 差一点也可以”。只要 Hard 失败，就必须是 Wrong。

Partial 专门表示：

```text
替代 ASIN + Hard 全满足 + 至少一个可核验 Soft 不满足
```

例如：

- 用户要求商品必须防滑，最好 100 元左右；
- 模型买了 150 元、但确实防滑的替代商品；
- Hard 防滑通过，Soft 价格失败；
- 应判 Partial，而不是 Wrong，也不是 Valid。

### 8.4 Partial 公式

```text
R_partial = 0.50 + 0.30 × soft_score
```

若有 3 个可核验 Soft，其中 2 个通过、1 个失败：

```text
soft_score = 2 / 3
R_partial = 0.50 + 0.30 × 2/3 = 0.70
```

因为进入 Partial 的前提是至少一个 Soft 失败，所以实际基础效用小于 `0.80`，不会超过 Valid。

## 9. 非购买终局

### 9.1 主动停止

模型调用 `finish_without_purchase` 且没有购买时，统一记为：

```text
early_abstain = -0.40
```

“已知可接受候选”的环境判定阈值为：

- 轻量类目 Gate 通过；
- 匹配分数至少 `0.70`；
- 证据覆盖率至少 `0.75`。

它只用于阻止模型明明已经看到可接受商品却提前退出，不直接代替最终购买分类。

### 9.2 Repeat Loop

Termination v3.1 有两个循环子原因：

- `exact_action_repeat`：同一动作连续第三次执行时终止；
- `no_progress_loop`：连续 6 个环境步骤没有新增运行证据。

基础效用为 `-0.60`，再叠加步数惩罚。

运行进展和“充分探索资格”是分开的：

- 新发现至少 1 个 ASIN，就重置无进展计数；
- 一次结果集至少带来 3 个新 ASIN，才增加 `effective_result_sets`；
- 因此少量新商品可以维持运行，但不能被用于刷合理停止资格。

### 9.3 Max Steps

完成第 45 个环境动作后终止：

```text
base(max_steps) = 0.00
step_penalty(45) = -1.05
final = -1.05
```

基础分设为 0，是为了把“任务没完成”和“买错商品”分开；最终仍会通过累计效率惩罚得到明显负分。

### 9.4 Assistant Final

模型直接输出普通文本、没有调用规定终局工具时，属于可归因的模型失败：

```text
assistant_final = -0.80 + step_penalty
```

它不是基础设施异常，因此 `reward_valid=true`，保留为训练负样本。

### 9.5 Action Guard Rejection

Action Guard 检查动作是否能在最新 Observation 中执行。连续 3 次 Guard 拒绝后终止：

```text
guard_rejection = -0.80 + step_penalty
```

合法动作会重置连续拒绝计数，避免把分散的历史错误累计成错误终止。

Guard 只判断动作合法性，不判断商品语义是否正确：

- 点了页面上不存在的 ASIN → Guard；
- 买了可执行但违反用户要求的商品 → Reward Wrong。

## 10. 累计步数惩罚

### 10.1 公式

设执行工具步骤数为 `T`，累计惩罚为：

```text
P(T) = -Σ max(0, min(T, end)-start+1) × rate
```

分段表：

| 步数区间 | 每步新增惩罚 |
|---|---:|
| 1–15 | `0` |
| 16–20 | `-0.01` |
| 21–25 | `-0.02` |
| 26–30 | `-0.03` |
| 31–35 | `-0.04` |
| 36–40 | `-0.05` |
| 41–45 | `-0.06` |

关键边界：

| T | 累计惩罚 |
|---:|---:|
| 15 | `0.00` |
| 16 | `-0.01` |
| 20 | `-0.05` |
| 21 | `-0.07` |
| 25 | `-0.15` |
| 30 | `-0.30` |
| 35 | `-0.50` |
| 40 | `-0.75` |
| 45 | `-1.05` |

最终 Reward：

```text
terminal_utility = base_terminal_utility + step_penalty
```

### 10.2 为什么从第 16 步开始

前 15 步作为正常检索、比较、核验和选规格的免罚区，避免鼓励模型跳过必要证据直接购买。超过 15 步后，惩罚逐段增大，重点抑制过度翻页、反复搜索和迟迟不收敛。

### 10.3 为什么所有有效终局都扣

如果只扣成功轨迹，模型可能倾向于失败来逃避效率成本；如果只扣失败轨迹，又无法优化成功路径长度。因此 Gold、Valid、Partial、Wrong、停止、循环、文本结束等所有 `reward_valid=true` 的终局使用同一张表。

`reward_unverifiable` 和基础设施无效轨迹不扣，因为它们本来就不进入学习，保持 0 能避免人为制造错误梯度。

### 10.4 示例

```text
10 步 Gold   = 1.00
21 步 Gold   = 1.00 - 0.07 = 0.93
26 步 Partial，soft_score=2/3
             = 0.50 + 0.30×2/3 - 0.18
             = 0.52
21 步 Loop   = -0.60 - 0.07 = -0.67
45 步 Timeout= 0.00 - 1.05 = -1.05
```

## 11. Reward 有效性与异常隔离

### 11.1 `reward_valid=false` 不等于普通失败

`reward_unverifiable` 表示环境无法可靠判断某个可评分 Hard 条件，例如：

- 明确价格上限存在，但所选规格价格无法唯一解析；
- 类目或关键结构化证据缺失；
- 评分合同出现必须人工确认的阻断语义。

这类样本：

```text
reward = 0
reward_valid = false
sampling_invalid = true
```

它不会进入 GRPO 组内优势计算。

### 11.2 模型失败与基础设施失败必须分开

以下属于模型可学习失败，保留负 Reward：

- assistant final；
- 连续非法动作；
- 循环；
- 超步数；
- 错误购买。

以下属于基础设施无效：

- 服务断连；
- 环境异常；
- 非有限数 Reward；
- 轨迹没有形成合法正常终局且无法归因给模型。

基础设施无效轨迹返回 0 并丢弃，不能让模型因为服务故障得到负优势。

### 11.3 Adapter 的二次校验

veRL Adapter 会验证：

- `reward_version` 必须是 Reward v4；
- Reward 类型必须在白名单中；
- `termination_reason == reward_type`；
- 只有 `reward_unverifiable` 可以设置 `reward_valid=false`；
- `terminal_utility = base + step_penalty`；
- `weighted_score`、覆盖率和各维度分数必须在 `[0,1]`；
- 约束结果状态只能是 Pass、Fail 或 Unverifiable。

随后只保留公共诊断字段，不把完整 Gold 私有事实写回训练侧。

### 11.4 `r_type`、`r_att` 等不是额外相加的 Reward

训练日志中还能看到：

```text
r_type, r_att, r_option, r_price,
brand_score, model_score, core_function_score, option_score
```

这些字段是可观测诊断分解，不会再次加到环境终局分数上。当前真正进入 GRPO 的是：

```text
total = terminal_utility = base_terminal_utility + step_penalty
```

因此不能把 Reward v4 讲成“类别分 + 属性分 + 规格分 + 价格分的线性加权和”。当前版本的分类由约束决策树决定，维度分数用于解释和审计。

最终数值也没有裁剪到 `[-1,1]`。例如 45 步 Wrong 为 `-1.00-1.05=-2.05`。因为同一步数使用相同效率惩罚，终局类型的相对顺序仍然保持。

## 12. Reward 与 GRPO 动态采样的关系

GRPO 需要同一 Prompt 的多条轨迹形成相对优势。当前训练采用每组 `n=4`：

- `reward_valid=false` 或基础设施无效轨迹不进入更新；
- 四条 Reward 完全相同，或极差 `<=0.025`，视为低信息组；
- 一个组最多生成 3 批，即初始采样加两次重试；
- 最后仍没有有效差异则整组丢弃；
- 训练使用无 KL Reward；
- `norm_adv_by_std_in_grpo=false`，避免小方差噪声被组内标准差放大。

Reward 分层的意义就在这里：Gold、Valid、Partial、停止、循环和 Wrong 形成有序差异，使同一任务的四条轨迹更容易产生可学习的排序信号。

步数惩罚还能区分“同样买对，但一个 10 步、一个 30 步”的轨迹；不过它不会替代终局正确性，因为基础终局类型仍决定主要排序。

训练前还对在线 Probe 做 Reward 质量审计，剔除了 4 组零成功或 Reward 异常的 Probe，避免把无相对学习信号或评分链路异常的数据放进正式训练池。训练中联合监控：

- `reward/strict_mean`；
- `reward/shaped_mean`；
- `reward/purchase_success_rate`；
- `reward/partial_purchase_rate`；
- 组内 Reward 极差与无效 Reward 数；
- 动态采样保留、重采样和丢弃组数量。

不能只看平均 Reward 上升。如果 Purchase Success 不升、Partial 激增或大量组因常数 Reward 被丢弃，说明可能出现 Reward 分布漂移或策略钻空子。

## 13. 当前 Final-240 审计结果

当前语义审计覆盖 240 个任务：

- 共 1,564 条约束；
- Hard 1,156 条；
- Soft 226 条；
- Ignore 182 条；
- Needs Review 0 条；
- Scored 1,303 条；
- Audit Only 261 条；
- 目标商品回放：238题为Gold，2题因Soft价格失败为Partial；
- 会阻断 Reward 的未解决语义为 0；
- 含语义标记但未进入合同的 Query 分句为 0。

六模型购买记录合计：

- Wrong 123 条；
- Valid 98 条；
- Partial 10 条；
- 对应唯一任务数分别为 70、44、8。

当前 Harness 改善版 GRPO230 在 Final-240 上为：

| 指标 | 数值 |
|---|---:|
| Strict Gold | `163/240` |
| Gold + Valid 购买成功 | `180/240` |
| Valid | `17` |
| Partial | `1` |
| Wrong | `19` |
| Repeat Loop | `33` |
| Guard Rejection | `3` |
| Max Steps | `3` |
| Reward 无效/Unknown | `1` |
| 平均 Final Reward | `0.515583` |

这组数字是当前 Reward 的离线重放结果。简历若继续使用训练完成时的旧口径数字，需要明确口径；面试深挖时应优先给出当前重放结果并说明版本演进。

## 14. 如何防止 Reward Hacking

### 14.1 防止隐藏标注泄漏

- 只有 Query 有证据的品牌、型号、功能和规格才参与评分；
- Gold 私有字段只作对照或审计；
- Query 和合同都带 SHA-256；
- Actor 在行动前看不到 Gold、Reward 或其他模型结果。

### 14.2 防止低价展示页攻击

- 使用已选 variant 的可核验价格；
- 规格未选全、价格无法唯一解析时不猜最低价；
- 支持每单位价格，避免总价和单价混淆。

### 14.3 防止文本子串攻击

- 品牌排除“兼容、适配、适用”等前缀；
- 型号使用 Token Boundary；
- 功能识别否定语义；
- 规格优先使用实际 selected options，标题不能覆盖显式冲突。

### 14.4 防止抢购

- Hard Fail 直接 `-1.0`；
- 合理停止比错误购买更高；
- Partial 只允许 Soft 失败，不能用来容忍 Hard 失败；
- 同一步数下，错误购买始终比循环或主动停止更差。

### 14.5 防止用长轨迹刷证据

- 45 步硬上限；
- 精确动作重复和无进展终止；
- 第 16 步起递增惩罚；
- 用运行进展和充分探索资格两套状态，避免翻页或少量新商品刷停止资格。

## 15. 为什么 Reward 不调用 LLM

优点：

- 在线训练吞吐稳定；
- 成本低；
- 结果可复现；
- 不依赖外部 API；
- 每个分数可追溯到商品字段和比较器；
- 不会出现 Judge 模型版本漂移直接改变训练梯度。

缺点：

- 很难覆盖真实性、审美、复杂否定、场景适配等开放语义；
- 同义词、商品脏数据和隐含常识仍可能导致误判；
- 规则越复杂，维护成本越高。

因此项目采用分工：

- Reward：高精度、可执行、可复现的训练信号；
- Frozen Rubric：逐条用户要求满足情况；
- LLM-as-a-Judge：过程质量、证据核验和决策合理性；
- 确定性过程指标：步骤、Token、Guard、循环和时延。

Reward 不需要覆盖所有评测维度，只需保证训练信号高精度、方向正确。

## 16. 当前设计的局限与可改进点

### 16.1 Audit-only 语义不会直接训练模型

真实性、审美和复杂否定等要求虽然进入合同，但若没有可靠比较器，只能由 Rubric/Judge 评估。这意味着 Reward 对这些能力的训练信号仍不充分。

改进方向：增加离线人工标注的结构化属性、商品证书字段或小型高精度分类器，而不是直接把开放式 LLM Judge 接进每一步在线 Reward。

### 16.2 Soft Unverifiable 不会被当成满足

当前策略是“没有证据不等于满足”：Soft Unverifiable 不会被误判为 Soft Pass，也不会直接算作 Soft Fail；整条购买轨迹进入 `reward_unverifiable` 并从正常 GRPO 学习信号中排除。这样既不虚高 Gold/Valid，也不把字段缺失武断解释为偏好违反。

### 16.3 Target ASIN 也必须满足 Soft

目标 ASIN 只决定“全满足后是 Gold 还是 Valid”，不再绕过用户 Soft。目标商品 Hard 全通过但 Soft 失败时降为 Partial；Soft 无法核验时进入 Reward Unverifiable。隐藏 Gold-only variant 仍只作审计，不会凭空成为用户条件。

### 16.4 手工规则仍有长尾

当前通过语义别名、任务标注修复和逐题审计修正了一批确定性误判，但规则系统不可能完美覆盖所有中文表达。因此必须：

- 版本化语义规则；
- 保存旧结果；
- 对 Gold 全量回放；
- 对分类迁移逐题审计；
- 不在 Final-240 上改完 Reward 后继续训练并直接汇报同一测试集。

## 17. 高频面试问题与参考回答

### Q1：为什么不用简单的 0/1 Reward？

因为购物任务的错误严重程度不同。合理替代、违反软偏好、过早停止、循环和错误购买如果都记 0，GRPO 的组内排序信号太粗。分层 Reward 能让模型学到“先满足 Hard，再优化 Soft，最后减少无效步骤”的偏序关系。

### Q2：Hard 和 Soft 是怎么判断的？需要 LLM 吗？

不需要 LLM。Rollout 前用确定性中文规则结合 Query 原文证据分类。必须、一定、明确上下限等为 Hard；最好、尽量、左右等为 Soft；无所谓、也可以等为 Ignore；不需要、无需等歧义表达进入 Needs Review。只有有可靠比较器的约束才 `scored`，其他保留为 `audit_only`。

### Q3：所有带“不要”的都是 Hard 吗？

不是。“绝对不要、不要某功能”通常是 Hard；“不要太贵、别太大”是程度偏好，属于 Soft；“不需要、无需、不用”可能表示禁止、无所谓或能力表达，不能一刀切，歧义时只审计。

### Q4：Partial 到底是什么？

Partial 可用于目标商品或替代商品，并且要求 Hard 全通过、至少一个可评分 Soft 失败。Hard 失败绝不能进入 Partial，否则会鼓励模型违反用户底线。

### Q5：为什么 Partial 是 `0.5+0.3×score`？

它把 Partial 放在失败终局和 Valid 之间：基础 0.5 表示已经满足全部 Hard，额外 0.3 根据 Soft 满足比例连续插值，上限不超过 Valid 的 0.8。这样有可学习梯度，但不会让 Partial 超过完全满足的替代商品。

### Q6：Soft 有一项无法核验怎么办？

Unverifiable 不算 Fail，也绝不能当作 Pass；整条购买轨迹记为 `reward_unverifiable`，不进入正常 GRPO 更新。这样不会把字段缺失误解释为偏好违反，也不会虚高 Gold/Valid。

### Q7：Gold 和 Valid 有什么区别？

Gold 要命中目标 ASIN并通过所有可评分 Hard 与 Soft；Valid 是不同 ASIN，但同样满足所有可评分 Hard 与 Soft。两者都算购买成功，但严格成功只统计 Gold。

### Q8：为什么不要求精确隐藏规格才能 Gold？

因为用户没说出的隐藏规格不应成为约束。当前把精确 Gold variant match 作为诊断字段，真正的 Gold 判定使用目标 ASIN和公开 Query 支撑的 Hard/Soft。这减少了对数据集私有标注的过拟合。

### Q9：如果目标 ASIN违反 Soft，会怎样？

不会得到 Gold。Hard 全通过但 Soft 失败时降为 Partial；Soft 无法核验时为 Reward Unverifiable。目标 ASIN只在 Hard/Soft 全通过后决定终局是 Gold，而不是 Valid。

### Q10：为什么 Wrong 是 -1，而不购买只有 -0.2 到 -0.4？

错误购买会真实损害用户利益，严重性高于安全停止。这个排序能防止模型为了拿购买奖励而抢购。

### Q11：Max Steps 基础分为什么是 0？

基础分 0 用于区分“没有完成”和“明确买错”。但它固定发生在第 45 步，会叠加 -1.05 的步数惩罚，所以最终仍是显著负分。

### Q12：为什么 Loop 从 -0.8 调成 -0.6？

Loop 是模型失败，但通常比明确违反 Hard 的 Wrong 更轻，也比早停更严重。设为 -0.6 后形成 `early_abstain(-0.4) > loop(-0.6) > assistant/guard(-0.8) > wrong(-1.0)` 的基础排序；长循环还会继续受到步数惩罚。

### Q13：步数惩罚为什么不是固定每步扣分？

前 15 步是正常检索免罚区，之后分段递增。固定从第一步扣分容易鼓励模型跳过核验，递增设计更符合“允许必要探索、重点惩罚拖沓”的目标。

### Q14：为什么 Reward Invalid 给 0？会不会被模型喜欢？

不会，因为它设置 `reward_valid=false` 和 `sampling_invalid=true`，不进入 GRPO 更新。0 只是中性的诊断占位，不是有效学习奖励。

### Q15：Assistant Final 为什么不是 Invalid？

它是模型没有遵循必须调用终局工具的可归因失败，应保留负信号。如果直接丢弃，模型无法学会正确结束任务。

### Q16：Guard Rejection 和 Wrong Purchase 有何区别？

Guard 判断动作能否执行，例如 ASIN、按钮或规格 ID 是否存在于最新 Observation；Reward 判断执行成功后的商品是否满足用户需求。前者是动作合法性，后者是任务语义正确性。

### Q17：价格怎么防止最低展示价作弊？

Reward 解析已选择的 variant 价格；规格没选全或存在多个可能价格时不猜最低值，而是 Unverifiable。每单位预算还会根据所选数量换算。

### Q18：为什么不用 LLM 做在线 Reward？

在线 GRPO 需要大量终局评分，LLM Reward 会增加成本、延迟、服务波动和不可复现性。确定性 Reward 负责高精度可执行约束，开放语义由离线 Rubric/Judge 补充。

### Q19：如何证明 Reward 没有泄漏 Gold？

评分合同记录每项约束来源；无 Query 证据的 Gold 属性不参与评分；Actor 行动前看不到 Reward；训练 Adapter 只保留公共终局诊断。还可以审计“无 Query 支撑但参与评分”的数量，当前 Final-240 为 0。

### Q20：如何验证 Reward 改动没有把 240 题改坏？

先对240个目标商品全量回放，要求每题都符合“Hard/Soft逐项判定后再按ASIN分流”的不变量，而不是预设240题必须全为Gold；当前结果为238题Gold、2题Soft价格失败后降为Partial。随后检查阻断型未解决语义和未覆盖语义分句为0，再逐题审查Wrong、Valid、Partial迁移。

### Q21：Reward 改了，Rubric 和 Judge 要重跑吗？

只改 Reward 分类和数值时不需要。Rubric 和 Judge 的输入轨迹没有变化。需要更新的是 Reward 类型、平均 Reward、购买成功、Hard/Soft 约束结果和由 Reward 派生的对齐统计；LLM 结果继续复用。

### Q22：训练后改 Reward 再重算评测算作弊吗？

只要不拿 Final-240 新标签继续训练该 checkpoint、所有模型统一重放、保留旧结果并披露版本，就属于评测口径修正。若根据 Final-240 反复调 Reward 后继续训练并仍把它当独立测试集，才会形成测试集泄漏。

### Q23：当前 Reward 最大的不足是什么？

复杂真实性、审美、隐含场景和歧义否定仍难以由无状态规则准确判断。当前选择高精度优先：无法可靠评分的要求进入 audit-only，由 Rubric/Judge 衡量，而不是强行制造训练噪声。

### Q24：为什么 Final-240 还需要 Rubric 和 LLM Judge？

Reward 是训练目标，不应同时充当唯一评测裁判。Rubric 检查逐条约束，LLM Judge 评估轨迹质量，确定性指标评估结果和资源成本，四者相互独立才能发现 Reward Hacking 和过拟合。

### Q25：如果重新训练，你会怎么做？

会冻结当前语义合同后重新构建 Probe，重新检查组内 Reward 方差，再用新 Reward 训练新的 checkpoint；同时保留原 GRPO230 作为旧 Reward 基线，并使用新的零重叠测试集验收，避免在 Final-240 上继续闭环调参。

### Q26：Reward 是多个分量相加的吗？

不是。`r_type、r_att、r_option、r_price`是诊断项，真正进入 GRPO 的 `total`就是环境终局效用加步数惩罚。这样可以避免训练侧和环境侧各加一次分数，也保证在线训练与离线重放一致。

### Q27：步数惩罚会改变 Gold、Valid、Partial 的分类吗？

不会。先决定 `reward_type`，再叠加步数惩罚。它只改变数值，不改变严格成功和购买成功标签。因此可以分别分析任务正确性与效率。

### Q28：怎么发现 Reward 异常？

先做结构校验和目标商品回放，再看 Probe 的有效性、有限数、终局类型一致性、组内极差和成功分布。项目中剔除了4组零成功或Reward异常Probe；正式评测要求六模型分类不变量0违规，并逐题解释目标回放中的238个Gold与2个Partial。

### Q29：Reward 数值超出 `[-1,1]`会不会不稳定？

基础终局效用控制在 `[-1,1]`，只有长轨迹累计惩罚会继续向下。GRPO 使用组内相对优势，训练又关闭按组内标准差归一化，避免小方差噪声被放大；同时通过梯度、KL、Reward 方差和长度曲线监控异常。若未来出现极端长上限，可以再考虑裁剪或归一化。

### Q30：为什么训练使用无 KL Reward？

项目希望让小模型有足够空间从 SFT 策略继续探索，因此没有把 KL 作为 Reward 惩罚项；但 PPO KL 仍作为监控指标。稳定性主要依靠 LoRA、小学习率、动态采样、Dr.GRPO 去标准差和冻结 Final-240 验收。

## 18. 三种面试回答长度

### 18.1 30 秒版

> 我设计的是确定性终局 Reward v4。Rollout 前把公开 Query 编译成可审计的 Hard/Soft 约束合同，购买时核验类目、价格、品牌、型号、功能和实际所选规格。Hard 失败为 Wrong；Hard 全满足但 Soft 失败时，无论目标还是替代商品都按 `0.5+0.3×soft_score`给 Partial；Hard/Soft 全满足后，目标商品为 Gold、替代商品为 Valid。循环、早停、非法动作和超步数也有独立分段，并从第16步起递增扣分。不可核验和基础设施异常会被丢弃，不制造错误梯度。

### 18.2 两分钟版

> 购物 Agent 的难点是失败类型很多，二元 Reward 无法区分合理替代、违反软偏好、错误购买和循环。我先把用户 Query 编译成带原文证据的约束合同，用规则区分 Hard、Soft、Ignore 和 Needs Review。Hard 只允许高置信且有确定性比较器的约束参与评分，复杂语义保留为 audit-only。终局时使用商品结构化信息、可见文本、selected options 和 variant price 逐条比较：Hard fail 直接 Wrong；Hard 或 Soft unverifiable 丢弃；Hard 全通过但 Soft fail 时，目标或替代商品都为 Partial，公式为 `0.5+0.3×soft_score`；Hard/Soft 全通过后再按是否命中目标 ASIN分为 Gold 与 Valid。所有有效终局从第16步开始按区间累计扣分，另外区分合理停止、过早停止、循环、assistant final和连续三次 Guard 拒绝。训练侧会再次校验 Reward 结构，并通过动态采样过滤无效或无组内差异的轨迹。

### 18.3 深挖版的主线

深挖时按以下顺序回答：

1. 为什么二元 Reward 不够；
2. Query 约束如何冻结且不泄漏 Gold；
3. Hard/Soft/Ignore/Review 与 scored/audit-only；
4. 各比较器如何抵抗子串、最低价和规格冲突；
5. 购买决策树与 Partial 公式；
6. 停止、循环、Guard 和步数惩罚；
7. Reward Invalid 与基础设施异常隔离；
8. 与 GRPO 动态采样的关系；
9. Final-240 全量回放和多面板独立评测；
10. 版本演进、局限和下一版计划。

## 19. 面试前速记：旧材料中需要纠正的表述

当前代码与部分旧文档存在版本差异，面试时以本节为准：

| 容易说错的旧口径 | 当前口径 |
|---|---|
| Wrong 只由品类 Hard Gate 触发 | 所有可评分公开 Hard Fail 都会触发 Wrong |
| Partial=`-0.3+0.5×S` | Partial=`0.5+0.3×soft_score` |
| Repeat Loop=`-0.8` | Repeat Loop=`-0.6` |
| 步数惩罚从第15步开始 | 第1–15步免罚，第16步开始 |
| 45步累计惩罚 `-1.06` | 当前为 `-1.05` |
| 目标 ASIN在 Hard 通过后直接 Gold | 当前为目标 ASIN + 公开 Hard/Soft 全通过；Soft 失败降 Partial |
| Gold 必须精确命中全部隐藏 variant | 精确隐藏 variant 单独审计；用户未表达的规格不进入 Hard/Soft |
| Hard/Soft 全部都直接评分 | 只有 `scored` 参与 Reward，复杂语义可为 `audit_only` |
| Reward 改动后必须重跑 LLM Judge | 轨迹未变时不需要，只重算 Reward 派生指标 |

## 20. 源码与审计入口

核心实现：

- `environments/ShopSimulator/shop_env/web_agent_site/engine/reward.py`
- `environments/ShopSimulator/shop_env/web_agent_site/engine/reward_features.py`
- `environments/ShopSimulator/shop_env/web_agent_site/engine/comparators.py`
- `environments/ShopSimulator/shop_env/web_agent_site/engine/variant_price.py`
- `environments/ShopSimulator/shop_env/web_agent_site/engine/termination.py`
- `src/shopping_grpo/price_semantics.py`
- `src/shopping_grpo/training/grpo/adapter/runtime.py`

冻结配置与测试：

- `environments/ShopSimulator/shop_env/configs/environment.json`
- `data/environment.json`
- `environments/ShopSimulator/shop_env/tests/test_reward.py`
- `tests/test_final240_reward_optional_semantics.py`
- `tests/test_price_semantics.py`
- `environments/ShopSimulator/shop_env/tests/test_termination.py`

当前 Final-240 结果：

- `重点/4.评测阶段/reward-v4-recalculation.json`
- `重点/4.评测阶段/reward-v4-per-task.json`
- `重点/4.评测阶段/reward-v4-semantics-rerun-20260824-2f7c337/REPORT.md`
- `重点/4.评测阶段/dashboard.html`
