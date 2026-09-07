# SFT 数据清洗与质量门

## 1. 数据目标

SFT 不是简单收集“最终买对”的轨迹，而是把 Final-240 暴露的四类主要问题转化成可监督行为：

- 循环搜索后能恢复并收敛；
- 能拒绝违反关键要求的近似商品；
- 规格值必须来自当前 Observation；
- 证据闭合后使用 `buy_now`，不用自然语言代替终局动作。

最终 1,000 条由 600 条 Stable 和 400 条 Corrective 组成：

| Teacher 类型 | 数量 | 训练作用 |
|---|---:|---|
| stable | 600 | 保持常规严格成功能力 |
| loop recovery | 160 | 从弱检索、错误候选或循环风险中恢复 |
| near-miss rejection | 101 | 拒绝核心条件不满足的相似候选 |
| option grounding | 99 | 精确选择页面可见规格 |
| terminal tool commit | 40 | 决策完成后显式购买 |

## 2. 候选来源与旧数据再验证

旧版 500 条 Teacher 不享受历史豁免，必须重新通过当前 Environment v2.4、Reward v3.2、
Termination v3.1、Observation v2 和 Tools v2 合同。最终原样保留旧轨迹 405 条，其余由新采集
且通过新版行为门的轨迹替换。

Corrective 候选的真实筛选结果：

| 类型 | Raw | 严格/关键动作拒绝 | 行为门拒绝 | 有效候选 |
|---|---:|---:|---:|---:|
| stable | 4,762 | 2,218 | - | 2,544（去重前） |
| loop recovery | 1,886 | 907 | 805 | 174 |
| near-miss rejection | 1,746 | 720 | 911 | 115 |
| option grounding | 274 | 174 | - | 100 |
| terminal tool commit | 40 | 0 | - | 40 |

30K 审计后可用 Corrective 恰好支持 160/101/99/40，共 400 条；Stable 从 1,625 条唯一严格
候选中先构建 1,100 条低内存候选池，再按检索档位和轨迹长度联合选择 600 条。

## 3. 三层数据门

### 3.1 结果门

- `reward_valid=true`；
- 终局为完整 `gold_purchase`；
- Gold 商品、必要规格和最终价格可验证；
- 不接受 valid alternative、partial、wrong、loop、max steps 或自然语言终局。

### 3.2 过程与关键动作门

- assistant tool call 与 tool return 必须完整配对；
- 每回合最多调用一个工具；
- 工具名称和参数必须符合统一 Tool Schema v2；
- ASIN、按钮和规格值必须由对应 Observation 支持；
- 最终完整动作必须是 `buy_now`；
- 不将 Gold、Reward 等审计私有字段写入训练序列。

对于可证明不改变后续状态的 Guard 重复调用，只清洗对应 assistant/tool 对；无法证明安全的
重复仍整条拒绝。最终 `guard_recovery` 覆盖 501 条，但 `clean_critical_actions` 为 1,000/1,000。

### 3.3 行为与语义门

- loop recovery 必须真实经历不确定性并成功恢复，不能只贴标签；
- near-miss 必须指出具体缺失约束，不能用模糊“不合适”代替；
- option grounding 必须发生真实 variant 选择；
- terminal commit 必须通过终局工具完成；
- 已证据闭合仍无意义搜索、证据外推和选择“勿拍”规格的轨迹会被剔除。

## 4. 去重和防泄漏

- task_id、trajectory_id 内部唯一；
- train / validation task 与 trajectory 零交叉；
- Final-240 task、ASIN、family、标题/Query 近似语义重叠为 0；
- 本地和远端 near 候选只在过质量门后求交集，最终删除 task 6500、16404 的本地重复版本；
- 完整动作序列不强制唯一，但单一模板占比上限为 12%。

最终共有 452 种完整动作序列，Top-1 占 7.5%，Top-5 合计 19.5%，避免 SFT 退化成固定
“搜索→打开→Features→Description→购买”模板。

## 5. 分布配平

检索排名最终为 Rank 1 / Rank 2-5 / Rank 6-20 / Rank 21-150 / Missing =
501 / 299 / 100 / 70 / 30。工具步数定义为 Short ≤10、Medium 11-20、Long >20，最终分布为
477 / 400 / 123，范围 4-42 步。

分布门还要求候选比较、证据核验、搜索改写、多规格选择、Guard 恢复和长轨迹覆盖。冻结
`data/data_gate.json` 中所有覆盖项均超过最低要求，没有通过删除高质量轨迹去强行凑整齐比例。

## 6. 30K Token 合同

Token 使用 Qwen3.5-2B 的真实 chat template 和 tokenizer 统计。策略固定为：

```text
超过 30,000 tokens → 整条轨迹删除 → 禁止截断 assistant/tool 链
```

初次审计剔除 13 条超长轨迹，随后用 Medium 和 Rank 2-5 样本回填并重新执行完整审计。最终：

| Split | 数量 | Input mean | P95 | Max | Assistant target mean |
|---|---:|---:|---:|---:|---:|
| train | 900 | 11,930.78 | 22,253 | 29,533 | 1,282.41 |
| validation | 100 | 11,951.73 | 21,244 | 29,994 | 1,253.88 |

## 7. 冻结产物

本目录的 `data/` 保存元数据、数据门、Token 审计、预检和训练指标。实际 1,000 条 JSONL
仍保留在仓库规范路径 `data/sft-final1000-convergence-v5-30k/`，此目录不重复复制大训练文件。
