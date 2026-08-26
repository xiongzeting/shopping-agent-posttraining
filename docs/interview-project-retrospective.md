# Shopping Agent 项目迭代复盘（面试版）

更新时间：2026-08-08。

这份文档记录本项目从旧版 SFT 基线、Final-240 升级、Teacher-500 构建、上下文与训练
问题排查，到三层数据门合同落地的主要迭代。它用于面试复盘，不替代当前状态文档；正式
状态仍以 `PROJECT_STATUS.md`、数据 metadata 和代码为准。

本文有意省略 API Key、SSH 地址、密码、机器绝对路径、账户和临时运维凭据。远程机器
迁移、GPU 中断和服务重启只保留对工程设计有价值的部分。

## 一句话项目介绍

这是一个基于 ShopSimulator 的长程购物 Agent 后训练项目，目标是建立可复现的：

```text
Baseline → SFT → GRPO → Evaluation
```

Agent 需要通过搜索、候选比较、属性和规格核验、价格确认与购买工具完成单商品任务。
项目重点不只是训练模型，还包括环境终局、Reward、轨迹质量、数据隔离、上下文管理和
评测合同的一致性。

## 面试时最值得讲的三个问题

1. 为什么旧 SFT 模型在新版 Final-240 上大幅下降，根因究竟是模型退化、Reward 变严，
   还是评测分布发生了变化？
2. 为什么加入“过程质量门”后，Teacher 数据反而集中在 8 步左右，长轨迹与恢复能力被
   削弱？
3. 为什么 Teacher 训练轨迹不长，但 Final-240 推理会触及 24K 上下文；上下文压缩会不会
   改变模型能力结论？

最终得到的核心认识是：这三个问题都不能只靠调一个超参数解决，必须同时管理任务分布、
逐轨迹正确性、数据集覆盖、推理上下文和正式评测合同。

## 项目演进时间线

| 阶段 | 主要工作 | 关键结果或教训 |
|---|---|---|
| 项目接管与仓库化 | 扫描代码、初始化 Git、冻结工作流和版本合同 | 训练、评测、数据与本地机器状态分离，关键修改原子提交 |
| Benchmark v2 | 统一四面板评测和配对比较 | Environment Reward 是主判定，Rubric/Judge 只做诊断，不合成黑盒总分 |
| Final-240 | 从 eval-tag 池构建 Core-180 + Challenge-60 | 新评测集与训练池零 task 重叠、零 family 重复，难度明显高于旧 Final-200 |
| Environment / Reward 升级 | 固定 Environment v2.4、Reward v3.2、Termination v3.1、Observation/Tool v2、Search v2.1 | Reward 增加逐约束审计；BM25 字段权重和规格索引错位得到修复 |
| 第一版 Teacher 双门 | 严格结果门 + 过程门 | 结果正确性提高，但过程门错误删除 Guard 恢复轨迹 |
| Teacher-500 | 939 raw 筛到 500，切分 450/50 | 249/500 恰好 8 步，Short 占 92.6%，数据高度同质化 |
| 长度平衡尝试 | 目标 Short/Medium/Long 约 4:4:2，补采中长轨迹 | 强行指定步数容易产生“为了长度而探索”，不能把步数当作难度本身 |
| 难度比例采样 | 按源任务难度四档抽样，而不是强行拉长 | 500 条目标配额固定为 121/298/64/17 |
| 自然策略多样化 | 搜索改写、候选比较、证据核验、价格语义、多规格等策略混采 | 从“长度驱动”转为“任务难度和能力覆盖驱动” |
| Final-240 API 诊断 | 多个 API 模型与本地 Base/SFT 运行入口分离 | 暴露 24K 上下文、观察历史累积和 API 网关兼容问题 |
| SFT 训练工程 | 固定 BF16 LoRA 配方、tokenizer audit、运行留痕和恢复合同 | 解释了新数据下显存随 batch 长度波动，而不是简单的显存泄漏 |
| 三门合同 | 保留结果门、放宽过程门、增加数据门 | 旧 Teacher-500 被标记为 legacy，不能再通过新 canonical 预检 |

## 1. 先冻结“什么叫同一个实验”

项目早期最大的风险不是模型代码，而是不同版本结果被混在一起解释。最终冻结的当前运行
合同为：

```text
Environment v2.4
├── Reward v3.2
├── Reward Features v2
├── Query Constraints v1
├── Termination v3.1
├── Observation v2
├── Search v2.1
└── Tool Schema v2
```

严格成功必须同时满足：

```text
reward_type = gold_purchase
reward_valid = true
purchase_success = true
termination_reason = gold_purchase
done = true
over = true
```

Reward v3.2 的重要变化不是把满分从 1 改成别的数，而是把用户 Query 拆成可审计约束，
逐项输出 `pass / fail / unverifiable`。这让错误能够被定位为预算、类目、功能、型号或规格
问题，而不是只看到一个最终 Reward。

Search v2.1 继续使用可复现的多字段 BM25，但修复了 FTS5 列位置与字段权重错位，并从
规格索引中删除图片 URL、variant ASIN 和内部状态字段。这个修复说明：检索质量问题可能
来自数据管线，而不一定来自模型不会搜索。

面试表达：我先把环境、奖励、终局、搜索、工具和数据哈希冻结，确保后续模型差异不是
协议漂移造成的。

## 2. Final-240 为什么比旧 Final-200 难很多

### 2.1 数据池发生了结构性变化

排查中发现：

| 数据集 | train tag | eval tag |
|---|---:|---:|
| 旧 Final-200 | 196 | 4 |
| 新 Final-240 | 0 | 240 |
| 第一版 Teacher-500 | 500 | 0 |
| 历史 Teacher-525 | 508 | 17 |

旧 Final-200 几乎来自训练分布，而新 Final-240 完全来自 eval-tag 池。旧模型在新评测上
下降，首先是 train→eval 分布迁移，不能直接归因于 Reward v3.2 变严格或新 SFT 训练失败。

### 2.2 完整 Query 检索排名给出了更直接的证据

| 数据 | Gold@1 | Gold@5 | Gold@20 |
|---|---:|---:|---:|
| 第一版 Teacher-500 | 81.8% | 92.4% | 97.4% |
| 旧 Final-200 | 73.5% | 88.5% | 94.0% |
| 新 Core-180 | 54.4% | 79.4% | 88.9% |
| Challenge-60 | 46.7% | 73.3% | 85.0% |

Teacher 数据中 Gold 商品经常排第一，模型很容易学成“搜索一次、打开第一个、核验后购买”。
新评测中 Gold@1 只有约一半，模型必须真正具备搜索改写和候选比较能力。

在一组相同的新评测题上，旧模型面对 Gold Rank1 任务成功 12/26，而面对非 Rank1 任务
只成功 3/23。这个对照进一步支持“检索排名和候选密度是主要难度来源”。

### 2.3 Final-240 的结构

```text
Final-240
├── Core-180：9 个一级领域 × 20 题
└── Challenge-60：6 个困难切片 × 10 题
```

困难切片的冻结难度均值为：

| 困难切片 | 数量 | 难度均值 | 主要过程特点 |
|---|---:|---:|---|
| candidate comparison | 10 | 19.88 | 多候选搜索、打开和比较 |
| long horizon | 10 | 19.45 | 综合长链路，容易循环 |
| evidence verification | 10 | 19.43 | 多属性、多页面证据核验 |
| price semantics | 10 | 16.11 | 预算边界与最终规格价格语义 |
| search reformulation | 10 | 15.99 | 需要多次实质性改写搜索 |
| multi option | 10 | 12.98 | 多规格轴与 variant 选择 |

### 2.4 评测方法也被重新约束

正式管线统一为：

```text
rollout
→ normalize
→ deterministic metrics
→ rubric / judge（可选离线输入）
→ per-task four-panel evaluation
→ summary
```

四个面板分别是：

- 环境 Reward 和终局结果；
- 用户需求逐约束满足情况；
- 轨迹过程质量；
- 工具、重复动作、Guard、Token、上下文和耗时等确定性指标。

缺失任务固定计入 240 分母；Judge 不覆盖环境 Reward；不同模型必须在同一 Final-240 上
按 task ID 配对报告 failure→success 和 success→failure。项目主动删除了显著性检验和
不可解释的综合总分，让结果更容易审计。

## 3. Teacher-500 为什么会集中在 8 步

### 3.1 第一版双门筛选

第一版 Teacher 收集结果为：

```text
939 raw
→ 262 条结果门拒绝
→ 177 条过程门拒绝
→ 500 条通过
```

其中主要过程拒绝原因是：

- `process_guard_rejection`：169；
- `process_invalid_action`：14；
- 连续重复动作：7。

这套门的初衷是避免模型模仿非法动作、循环和无效探索，但它把“曾经被 Guard 拦截，后来
正确恢复并完成 Gold 购买”的整条轨迹也删掉了。困难任务更容易出现恢复动作，因此过程门
对困难轨迹产生了选择性淘汰。

### 3.2 通过数据的实际长度不是“全部 8 步”，但高度集中

第一版 500 条的执行步数分布中：

- 恰好 8 步：249 条，49.8%；
- Short（≤10 步）：463 条，92.6%；
- Medium（11～20 步）：35 条，7.0%；
- Long（>20 步）：2 条，0.4%。

所以用户看到的“几乎全是 8 步”是准确的直觉，只是严格来说不是 500 条全部等于 8。
最常见工具序列也高度重复，模型存在过拟合固定操作模板的风险。

第一版数据中的工具调用总量还显示：

| 工具 | 调用数 |
|---|---:|
| prev_page | 886 |
| select_option | 706 |
| search_products | 549 |
| open_product | 548 |
| buy_now | 500 |
| view_features | 451 |
| view_description | 421 |
| back_to_search | 49 |
| view_reviews | 23 |
| next_page | 2 |
| view_attributes / finish_without_purchase / think | 0 |

这说明数据覆盖了“搜索→打开→规格→详情→购买”的主路径，但分页、候选比较、无结果终止和
部分证据页覆盖不足。

### 3.3 旧 Teacher 为什么没有同样严重的问题

旧增量数据的大致流程是：

```text
145 raw → 100 条严格 Gold 成功 → 97 条通过长度/覆盖检查
```

旧流程允许 Guard 错误后恢复，并在写入 SFT 时清理被 Guard 拦截的 assistant/tool 消息。
100 条严格成功中约 37 条经历过 Guard recovery。它还维护数据集级能力最低覆盖，例如：

- query recovery ≥10；
- multiple candidates ≥10；
- multiple option axes ≥5；
- variant price recheck ≥10；
- subpage return ≥20；
- wrong category ≥3；
- next_page ≥1。

旧流程的关键不是“门更松”，而是把逐轨迹可清理错误与数据集整体能力覆盖分开处理。

### 3.4 过拟合风险来自模板集中，不只来自步数相同

单纯 8 步并不会自动导致过拟合。真正危险的是：

- 输入任务本身过于容易，Gold 经常排第一；
- 工具序列和页面访问顺序高度重复；
- 成功筛选只保留最顺利的路径；
- 候选比较、搜索恢复、分页和长链路样本不足。

这类“成功生存者偏差”会让训练 loss 很好看，但模型面对 eval-tag 难题时缺少恢复策略。

## 4. 从强行 4:4:2 到按源任务难度采样

### 4.1 最初的长度目标

Fresh SFT 一度按工具步数定义：

```text
Short：≤10
Medium：11～20
Long：>20
```

目标比例为 4:4:2。为减少重新生成，先尝试保留已有 Medium/Long、删除部分 Short，再补采
新的中长轨迹；同时为 Medium 和 Long Teacher 写了带停止条件的过程提示，避免无限探索到
`max_steps=35`。

### 4.2 为什么放弃“强行拉长”

实践发现，要求模型必须达到某个步数容易产生两个副作用：

- 为满足长度而重复打开页面或做低信息增益动作；
- Long 提示使 Teacher 过晚收敛，反而降低 Gold 购买成功率。

因此后续原则改为：步数是任务难度和必要证据链的结果，而不是优化目标。长度桶在全部
轨迹生成后按实际分布定义，追求近似 4:4:2，而不是把每条轨迹填充到指定长度。

### 4.3 按原始任务难度比例抽样

源任务难度分布为：

| 难度区间 | 源任务数 | 500 条目标配额 |
|---|---:|---:|
| <10 | 5,321 | 121 |
| 10～15 | 13,077 | 298 |
| 15～18 | 2,811 | 64 |
| ≥18 | 753 | 17 |

对应脚本会先复用 Final-240 的冻结难度特征计算方式，再按比例和固定 seed 构建无泄漏的
Teacher 任务池。这样训练数据仍接近真实源分布，同时确保高难任务不会在成功筛选中完全
消失。

### 4.4 自然策略混采

Teacher 提示从单一固定路径扩展为：

- focused verification；
- search reformulation；
- candidate comparison；
- evidence verification；
- price semantics；
- multi option。

收集器还优先遍历唯一 task，再考虑 retry，避免少数容易任务重复占满目标；选择器支持多
来源流式读取，避免一次性加载大量 raw 轨迹。

## 5. 最终的数据门设计

最后采用三层合同，而不是继续调一个越来越复杂的过程门。

### 5.1 第一道门：结果门保持不变

必须完整 Gold 购买、Reward v3.2 有效、环境 `done/over`，不接受替代购买、部分购买、错误
购买、不可验证或未完成轨迹。

### 5.2 第二道门：过程门只拒绝不可恢复缺陷

继续拒绝：

- 工具调用截断或单轮多工具；
- 实际执行 step error；
- 参数、工具名或环境动作不一致；
- 缺少 `search → open_product → buy_now` 基本路径；
- 连续完全相同动作。

不再整条拒绝：

- Runtime Guard 拦截后成功恢复；
- 非连续、具有任务意义的搜索改写。

Guard 拦截不是已执行的环境 step。SFT 序列化会删除被拒绝的 assistant/tool 对，只保留
后续合法恢复与成功购买过程。

### 5.3 第三道门：数据集级覆盖

500 条 canonical 目标：

| 维度 | 合同 |
|---|---:|
| Gold@1 | 最多 60% |
| Gold@2-5 | 至少 20% |
| Gold@6-20 | 至少 12% |
| Gold@21-150 | 至少 6% |
| 完整 Query Top-150 不命中 | 至少 2% |
| search reformulation | 至少 20% |
| candidate comparison | 至少 25% |
| multiple options | 至少 20% |
| Guard recovery | 至少 5% |
| >10 步 | 至少 35% |
| >20 步 | 至少 10% |
| 单一完整工具序列 | 最多 12% |
| 恰好 8 步 | 最多 30% |

数据门生成独立报告，记录输入、商品数据和搜索索引 SHA-256。SFT 预检要求 metadata 指向
该报告并校验版本、完整 policy、500 个唯一 task、通过状态和报告 SHA-256。

### 5.4 旧 939 raw 在放宽过程门后的真实审计

放宽过程门后，原始 939 条中有 656 条满足严格结果和新过程门，其中：

- Guard recovery：156；
- search reformulation：96；
- candidate comparison：69；
- multiple options：217；
- >10 步：112；
- >20 步：16。

检索排名为：

- Gold@1：513；
- Gold@2-5：74；
- Gold@6-20：43；
- Gold@21-150：21；
- Top-150 不命中：5。

另外，8 步轨迹有 269 条，最大单一工具序列出现 191 次。即使成功轨迹数量超过 500，也
无法仅靠重排旧数据通过新门；必须补采真正困难的候选比较、低检索排名和长链路任务。

这也是为什么现有 Teacher-500 被标记为 `legacy_pending_replacement`：保留它作为来源和
诊断证据，但不能修改 metadata 冒充新 canonical。

## 6. 为什么保留三个低频工具

曾经因为 Teacher-500 中调用次数为零，考虑删除：

- `view_attributes`；
- `finish_without_purchase`；
- `think`。

随后恢复并保留，原因分别是：

- `view_attributes`：某些任务的关键属性只在 Attributes 页可核验，训练集中为零不代表
  正式评测永远不会需要；
- `finish_without_purchase`：如果多轮实质搜索后确实没有可接受商品，评测模型必须有显式
  终止动作，不能被迫购买错误商品或循环到上限；
- `think`：保留工具 schema 和运行兼容性，但 Prompt 明确要求不要调用，因为它不与环境
  交互、不会增加证据并浪费有限步数。

这个决策体现了一个重要原则：不能仅根据训练频率删除运行时能力，训练分布与评测状态空间
并不相同。

## 7. 24K 上下文溢出到底是什么

### 7.1 `24576` 的准确含义

在评测中，`24576` 是一次模型请求允许的总上下文窗口，不是 Teacher 单条 completion 的
上限，也不是“模型总共只能调用多少次工具”。

当前默认还预留：

- 单轮最大生成 `max_tokens=512`；
- 安全余量 512。

因此可用输入历史预算大约是：

```text
24576 - 512 - 512 = 23552 tokens
```

每次工具调用后，下一次请求会携带 System Prompt、用户任务、之前的 assistant tool call、
工具 observation 和后续历史。轨迹越长、搜索结果和商品详情越多，输入历史越大。

### 7.2 为什么 Teacher 短而 Final 长

这不是因为 Final 的“答案文字”更长，而是：

- Teacher 主要来自 train-tag 容易任务，Gold@1 很高；
- Teacher 只保留成功生存路径，失败探索被结果门过滤；
- Final-240 完全来自 eval-tag，候选更密、Gold 排名更低；
- 失败任务平均执行步数显著高于成功任务，更容易循环并累积 observation。

一次 DeepSeek API 诊断运行中，严格成功任务平均执行约 12.7 步，失败任务约 20.45 步。
这解释了为什么上下文问题主要出现在长链路失败任务。

### 7.3 评测为什么也需要管理上下文

评测不做反向传播，因此不会产生训练 activation/optimizer OOM，但它仍然需要：

- 模型权重显存；
- KV cache；
- 服务端或 API 的上下文长度配额；
- 请求和历史消息序列化内存。

超过上下文窗口会导致请求被拒绝、输入截断或轨迹被标为 infrastructure-invalid。它会影响
评测有效性，即使没有训练 OOM。

### 7.4 压缩如何实现，是否会伤害能力结论

项目采用两级控制：

1. Observation projection：对单次搜索或详情 observation 做结构化裁剪，优先保留当前页
   商品、关键按钮、页面状态和可执行动作；
2. Context compaction：只有历史超过输入预算时，删除或压缩较旧的完整交互，同时保留最近
   的完整 assistant/tool 对，避免把一条工具交互切成半条。

压缩过度当然可能影响能力判断，所以不能只看“请求不溢出”。评测同时记录原始/可见
Observation Token、压缩比例、截断任务、上下文使用率和 infrastructure-invalid 数量。

一次非正式 DeepSeek Final-240 诊断完成 240 条，得到 113 条严格 Gold 成功，但同时存在：

- 44 条 infrastructure-invalid；
- 237 条出现 observation truncation；
- 最大上下文使用率达到 1.0。

因此这个结果只能用于定位问题，不能当作正式 Final-240 指标；仓库的正式 metadata 仍是
`evaluated=false`。后续还试验过把 Base/SFT 评测上下文提高到约 30K，以区分“模型能力差”
和“24K 基础设施预算不足”。

## 8. SFT 最大长度和评测上下文不是同一个概念

SFT 的 `max_length=24576` 是 tokenizer 渲染后的单条训练样本长度门限。旧 Teacher-500 的
token audit 中：

- train 最大输入长度：18,783；
- validation 最大输入长度：17,641；
- 450 train + 50 validation 全部低于 24,576。

评测上下文则在每一个 Agent step 动态增长。二者数值可以相同，但作用位置不同：

- 训练长度决定单个 batch 的张量和 activation 大小；
- 评测上下文决定每次自回归请求能携带多少历史和 KV cache。

## 9. 为什么新版 SFT 显存上下波动更明显

新版训练中显存振幅较大，主要原因是动态 padding 和样本长度差异：

- batch size 为 1，但不同轨迹 token 长度不同；
- collator 只把当前 batch pad 到当前最长样本；
- 长样本需要更多 attention/activation 内存，短样本会释放部分临时张量；
- gradient checkpointing、CUDA caching allocator、评估 batch 和保存节点会叠加波动。

旧数据更同质，长度与工具模板更集中，因此更容易表现为显存先爬升到缓存高水位后稳定。
新版如果包含更多 Medium/Long 样本，显存随 batch 顺序上下变化是合理现象，不等于泄漏。

Canonical SFT 配方固定为：

| 参数 | 值 |
|---|---:|
| Base | Qwen3.5-2B |
| max_length | 24576 |
| epochs | 3 |
| train/eval batch | 1 / 1 |
| gradient accumulation | 8 |
| learning rate | 1e-4 |
| warmup ratio | 0.03 |
| LoRA r / alpha / dropout | 16 / 32 / 0.05 |
| dtype | BF16 |
| gradient checkpointing | true |
| attention | SDPA |

训练入口还记录 planned recipe、命令、环境变量白名单、数据和模型哈希、tokenization、GPU
采样、metrics、checkpoint、恢复合同与失败原因。这样即使远程 GPU 因资源或余额中断，也能
从完整 checkpoint 继续，而不是靠人工回忆参数。

## 10. 历史模型、新模型和新评测应该怎么比较

仓库内可正式陈述的历史基线属于 Environment v2.1 / Reward v3：

- 原始数据 525 条；
- 473 train / 52 validation；
- token gate 后 468 train / 52 validation；
- 3 epochs，177 optimizer steps；
- train loss 0.323135；
- validation loss 0.339526；
- 历史 Final-200：125/200，62.5%。

这些结果不能冒充 Environment v2.4 / Reward v3.2 下的新指标。旧 SFT 模型在新 Final-240
上同样大幅下降，反而说明新版难度上升主要来自评测分布和长链路，而不是只有新 SFT 数据
或新 Reward 有问题。

模型比较策略最终统一为：

- API Teacher/Actor 模型用于外部能力参照；
- Qwen Base 和同一 Base 的 SFT 模型做配对对照；
- 旧 Reward v3 SFT 作为历史迁移诊断；
- 所有模型使用同一冻结任务、同一环境、同一上下文策略和同一分母；
- 重点看逐题迁移和错误类型，不只看总成功率。

API Teacher、Judge 和 Actor 的模型配置被拆分，避免 Judge 凭据或模型参数意外覆盖 Teacher
收集。API 网关还修复了 Actor 评测时错误走本地模式的问题。

## 11. 最有价值的工程教训

### 教训一：更严格的逐轨迹过滤可能让数据集更差

过程门看起来提高了质量，但它选择性删除困难恢复轨迹，使整体数据更短、更容易、更模板化。
必须同时审计逐轨迹正确性和数据集级覆盖。

### 教训二：成功数据也会有生存者偏差

只收集 Teacher 成功轨迹，会自然保留 Gold@1 和短路径任务。结果门没有错，但还需要按源
难度抽样和检索排名配额。

### 教训三：步数不是难度的可靠代理

强制 Long 会诱发无效点击；真实 Long 应来自低检索排名、多候选、多规格和多证据要求。

### 教训四：评测变难不能直接解释成模型退化

先比较任务来源、Gold 排名、约束数量和轨迹长度，再讨论训练是否有效。

### 教训五：上下文截断和训练 OOM 是两类问题

训练 OOM 主要是 activation、optimizer 和 batch tensor；评测溢出主要是历史输入和 KV
cache。两者都叫“长度问题”，但修复手段不同。

### 教训六：零频工具不等于无用工具

训练数据覆盖不到的合法终局和证据页，可能在评测中出现。删除 schema 会缩小模型运行时
可表达能力。

### 教训七：实验结果必须带版本和有效性标签

正式结果、诊断性部分运行、含 infrastructure-invalid 的运行和历史协议结果必须分开。
否则成功率数字很容易被误用。

## 12. 面试可直接使用的 STAR 讲法

### 案例 A：定位新版评测准确率暴跌

- Situation：旧 SFT 和新 SFT 在 Final-240 上都明显低于旧 Final-200。
- Task：确认是 Reward 变严、训练失败还是评测分布变化。
- Action：对比 train/eval tag、Gold@1/@5/@20、同题 Rank1 与非 Rank1 成功率，并检查新旧
  环境合同。
- Result：确认主要根因是 train→eval 分布迁移和候选检索难度，而不是 Reward 数值变化；
  后续 Teacher 数据门加入低排名任务配额。

### 案例 B：修复 Teacher 数据“全是 8 步”

- Situation：500 条中 249 条恰好 8 步，Short 占 92.6%。
- Task：提升搜索恢复、候选比较和长链路覆盖，又不能靠无效动作凑长度。
- Action：复盘新旧门差异，发现 169 条因 Guard 被过程门整条删除；放宽可恢复过程门，改用
  难度比例采样和数据集级覆盖门。
- Result：旧 raw 可恢复成功轨迹增加到 656，其中 156 条含 Guard recovery；同时数据门仍
  明确指出候选比较和 Long 不足，阻止伪 canonical 晋升。

### 案例 C：解释并治理 24K 上下文溢出

- Situation：Teacher 轨迹较短，但 Final-240 长链路任务触及 24K。
- Task：判断是模型输出太长、训练 max length 设置错误，还是 Agent 历史累积。
- Action：拆分 context window、单轮 completion 和训练 max_length；增加 observation 投影、
  可选 compaction 和上下文使用率指标。
- Result：确认失败任务平均步数更高，历史 observation 是主要增长来源；诊断运行中的 44 条
  infrastructure-invalid 被单独标记，不再混入模型能力结论。

### 案例 D：构建可恢复的 SFT 运行合同

- Situation：远程 GPU 任务可能因资源抢占或费用中断，手工重跑容易丢参数。
- Task：让每次训练可审计、可恢复、不可误覆盖 canonical。
- Action：冻结版本化 recipe，记录命令、数据/模型/源码/依赖哈希、token audit、GPU 指标、
  checkpoint 和首次运行合同。
- Result：中断可以从最高完整 checkpoint 继续；非 canonical 变体必须写到独立目录。

## 13. 关键量化结果速查

| 指标 | 数值 |
|---|---:|
| Final-240 | 240 = Core 180 + Challenge 60 |
| 第一版 Teacher raw / accepted | 939 / 500 |
| 第一版结果门 / 过程门拒绝 | 262 / 177 |
| 第一版 8 步轨迹 | 249/500，49.8% |
| 第一版 Short / Medium / Long | 92.6% / 7.0% / 0.4% |
| 放宽过程门后的严格成功结构合格轨迹 | 656 |
| 其中 Guard recovery | 156 |
| 放宽后 candidate comparison | 69，目标至少 125 |
| 放宽后 >20 步 | 16，目标至少 50 |
| 放宽后 Gold@1 | 513，500 条 canonical 上限为 300 |
| 历史 SFT Final-200 | 125/200，62.5% |
| 非正式 DeepSeek Final-240 诊断 | 113/240 strict，但有 44 infrastructure-invalid |
| 旧 Teacher-500 token max | train 18,783；validation 17,641 |
| Canonical SFT max_length | 24,576 |

## 14. 当前状态和下一步

当前已经完成：

- Final-240 冻结和盲测保护；
- Environment v2.4 / Reward v3.2 合同；
- 四面板评测和配对比较；
- Teacher 难度抽样、自然策略、多来源选择；
- 可恢复过程门；
- 数据门审计 CLI；
- canonical SFT v3 预检合同。

当前尚未完成：

- 补采满足数据门的最终 500 条；
- 新数据 tokenizer audit 和 canonical 晋升；
- 基于新三门数据的正式 SFT；
- 同合同 Base/SFT/GRPO Final-240 正式配对评测。

下一步不应直接启动 SFT，而应重点补采：

- Gold 排名 2～150 或 Top-150 不命中的任务；
- 至少打开两个真实候选的比较轨迹；
- 搜索改写和分页轨迹；
- 自然产生的 Medium/Long 证据链；
- 保留成功恢复但清洗 Guard 拒绝消息的轨迹。

## 15. 对应的关键 Git 提交

| Commit | 内容 |
|---|---|
| `4b74831` | 统一 Benchmark v2 四面板评测管线 |
| `8bbd559` | 分离 Teacher 与 Judge 模型配置 |
| `df1d1c4` | 第一版 Teacher 双门筛选 |
| `2fefebb` | 修复 BM25 字段权重和规格索引 |
| `fabec23` | 工具执行前参数与动作校验 |
| `6275005` | 晋升第一版 Environment v2.4 / Reward v3.2 SFT 数据 |
| `ae036ad` | 完成第一版 canonical token audit |
| `a2d8437` | 中长 Teacher 补采与长度平衡工具 |
| `fe1c034` | 按源难度比例构建 Teacher 任务池 |
| `026abca` | 固定 canonical SFT run plan 和留痕合同 |
| `c3f9990` | 自然策略多样化和多样性选择 |
| `30dfd63` | 优先采集唯一任务再 retry |
| `8c4bcfd` | 多来源 Teacher 流式选择 |
| `2393ea2` | 修复 API 网关 Actor 评测模式 |
| `2464394` | 可恢复过程门、数据门和 SFT v3 预检 |

## 16. 面试时应避免的表述

- 不要说“新 SFT 已经正式在 Final-240 达到某成功率”；当前正式 metadata 仍是
  `evaluated=false`。
- 不要把 DeepSeek 113/240 当作纯模型指标；该诊断运行存在 44 条基础设施无效轨迹。
- 不要说 500 条 Teacher“全部是 8 步”；准确说法是 249/500 恰好 8 步，Short 占 92.6%。
- 不要把旧 Final-200 的 62.5% 与新 Final-240 直接横向比较；协议和任务分布不同。
- 不要把 observation 压缩描述成无损；它是受约束的结构化裁剪，必须通过压缩率和无效任务
  数量审计其影响。
- 不要声称数据门已经通过；当前审计的价值正是证明旧数据不能仅靠重排晋升。
