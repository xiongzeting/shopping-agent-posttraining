# Reward v4

Reward v4 是一个面向约束购物任务的确定性终局
Reward。它只根据环境中的商品、
规格、价格和终局证据评分，不调用另一个语言模型判断结果。

Reward v4继承v3.2的两项审计能力：

1. 为每个任务冻结`shopping-query-constraints-v1`约束来源合同，区分公开Query、
   指令标注、严格目标variant和任务类目合同；
2. 在终局`reward_detail`中逐条输出`constraint_results`，明确每项约束的
   `pass / fail / unverifiable`、比较器和证据来源。

这样Environment Reward、离线Rubric和轨迹Judge可以互相审计，但仍保持四个面板
彼此独立。

## 1. 任务要求编译

Rollout 开始前，环境会把任务、公开Query和目标商品元数据编译成固定特征：

- 目标类目和用户预算；
- Query 明确提及的品牌别名；
- Query 与目标商品共同出现的型号；
- 核心功能；
- 颜色、尺寸、容量、数量、套餐等关键规格。

这些评分特征在策略行动前已经冻结，模型无法在执行过程中修改评分标准。每项特征还会
记录`source`和`query_evidence`，避免把目标商品全部私有属性自动当成用户需求。

## 2. 唯一错误购买硬门槛

`wrong_purchase`只由类目门槛触发：购买商品不属于用户要求的品类时为：

```text
wrong_purchase = -1.0
```

价格比较仍写入逐约束审计结果，但价格超限或价格不可验证都不再触发
`wrong_purchase`或`reward_unverifiable`。

如果唯一的类目门槛无法验证，则为：

```text
reward_unverifiable
reward = 0.0
reward_valid = false
```

这个零分不代表中性成功，也不能作为有效 GRPO 学习样本。

训练侧可验证的模型终止错误不会伪装成环境购买结果：`assistant_final`
与`guard_rejection`（连续三次动作守卫拒绝）的基础奖励均为 `-0.8`。
二者都继续叠加统一的累计步数惩罚，且
`reward_valid=true`、`sampling_invalid=false`，因此可参与GRPO组内学习。

## 3. 活跃偏好维度

只有任务实际激活的维度才进入分母：

| 维度 | 权重 |
|---|---:|
| 品牌 | 0.25 |
| 型号 | 0.25 |
| 核心功能 | 0.25 |
| 关键规格 | 0.25 |

加权匹配分数为：

```text
S = Σ(weight_i × score_i) / Σ(active weight_i)
```

证据覆盖率使用相同权重。只有匹配分数和证据覆盖率都为1，才算完整满足所有
活跃偏好。

## 4. 终局奖励

| 结果 | Reward |
|---|---:|
| 精确目标ASIN与精确目标规格 | `1.00` |
| 完整满足用户约束的有效替代 | `0.80` |
| 部分满足的替代购买 | `-0.30 + 0.50 × S` |
| 充分探索后的合理停止 | `-0.20` |
| 过早停止 | `-0.40` |
| 达到45步上限 | `0.00`基础分，再叠加累计步数惩罚 |
| 普通 assistant 文本结束 | `-0.80` |
| 重复动作或无进展循环 | `-0.80` |
| 错误类目 | `-1.00` |
| 硬门槛证据不可验证 | `0.00`，无效样本 |

从第15个已执行工具步骤开始，所有有效终局平等叠加累计步数惩罚：15–20每步
`-0.01`，21–25每步`-0.02`，26–30每步`-0.03`，31–35每步`-0.04`，
36–40每步`-0.05`，41–45每步`-0.06`。45步累计为`-1.06`；Reward异常样本
不施加该惩罚。

严格成功 `gold_purchase` 必须同时命中目标 ASIN 和精确标注的目标规格。

不同 ASIN，或者同一 ASIN 下不同但满足用户公开约束的规格，只能得到
`valid_alternative_purchase`，不能冒充严格 Gold。这既保留严格成功指标，也
避免把一个没有出现在 Query 中的隐藏规格当成唯一合理答案。

## 5. 价格语义

Reward v4 区分明确上限、近似价格和开放价格，并把结果用于审计：

- `不超过300元`、`300元以内`：硬上限300元；
- `170元左右`、`170元上下`：采用冻结的确定性容差；
- `300多元`、`三百多元`、`4100多元`、`两千来块`：开放价格偏好，不生成
  `<=300`、`<=4100`、`<=2000`等虚假硬上限；
- 明确区间如`300元到400元`：保留完整区间并输出确定性价格比较结果，但不作为
  `wrong_purchase`硬门槛。

Rubric extractor v4 使用相同语义：明确上限和明确区间是硬约束，开放表达和
“左右/上下”是软价格偏好。

## 6. 规格的严格命中与语义替代

目标数据中的精确 option 继续决定严格 Gold，但环境同时检查 Query 是否只声明
了一个公开数值范围。

例如用户要求：

```text
枕头高度15厘米以下
```

隐藏目标规格是12cm，而模型选择9cm时：

- 9cm确实满足公开的“15cm以下”；
- 不再判成规格错误；
- 因为没有精确命中12cm，只能得到`valid_alternative_purchase`；
- 选择16cm仍然是规格失败。

这个回退只适用于 Query 明确表达的数值上限、下限、区间或近似范围。明确颜色、
明确型号、明确数值规格和超范围选择不会被放宽。

## 7. 品牌证据

品牌比较按以下证据核验：

1. 结构化商品品牌和店铺字段；
2. 商品标题或名称中的安全品牌提及；
3. 与品牌别名完全匹配的结构化属性。

为了避免把兼容配件当成原厂商品，以下表达不能单独证明商品品牌：

```text
兼容飞利浦
适用苹果
适配小米
用于华为设备
```

也就是说，“飞利浦电动牙刷头”可以作为品牌/产品证据，而“兼容飞利浦的第三方
刷头”不能仅凭这一句话判为飞利浦品牌。

## 8. 主动停止和循环终止

只有模型至少检查两个有效搜索结果集、打开两个候选，并且没有已知可接受候选时，
`finish_without_purchase`才属于合理停止。

Termination v3 的上限保持：

```text
连续精确重复限制：2
连续无进展限制：4
最大环境步骤：45
```

当前运行时会进一步记录终止子原因：

```text
repeat_loop
├── exact_action_repeat  连续重复完全相同动作
└── no_progress_loop     连续6步没有新增运行证据

max_steps
└── max_steps            执行完第45个环境动作后终止
```

运行存活判断与“充分探索证据”已经分离：

- 新搜索只要发现至少1个未见过的ASIN，就会重置`no_progress_steps`；
- 只有一次结果集带来至少3个新ASIN时，才计入`effective_result_sets`；
- 因此1～2个新商品不会再被误判成完全无进展，但也不能用于刷高合理停止资格。

## 9. 当前组件版本

组件版本保持：

```text
Environment v2.4
Reward v4
Termination v3.1
Observation v2
Tool Schema v2
```

Reward v3.1的价格语义、公开范围规格和品牌证据规则保持不变；v3.2新增Query约束
来源合同和逐约束终局结果，当时总环境合同升级为Environment v2.3。随后Search v2.1
修正BM25字段权重错位并清理规格索引，当前总合同为Environment v2.4。Termination、
Observation和Tool Schema没有变化。历史Final-200仍属于Environment v2.1 /
Reward v3，不会被静默改写；新的Final-240使用Environment v2.4 / Reward v4。

## 10. 源码入口

- Reward实现：`environments/ShopSimulator/shop_env/web_agent_site/engine/reward.py`
- 价格语义：`environments/ShopSimulator/shop_env/web_agent_site/engine/constraints.py`
- 循环与进展：`environments/ShopSimulator/shop_env/web_agent_site/engine/termination.py`
- 环境冻结配置：`environments/ShopSimulator/shop_env/configs/environment.json`
- 仓库环境清单：`data/environment.json`
