# Benchmark v2.2 Final-240 题目分布

## 1. 版本与数据合同

| 项目 | 值 |
|---|---|
| Schema | `shopping-evaluation-dataset-v2.2` |
| Asset | `shopbench_longhorizon_final_240_v2_2` |
| Contract | `environment-v2.4/reward-v3.2/benchmark-v2.2` |
| 总题数 | 240 |
| Core / Challenge | 180 / 60 |
| 从v2.1保留 | 217 |
| 异常任务替换 | 23 |
| 训练任务重叠 | 0 |
| Task / family重复 | 0 / 0 |
| 难度均值 | 12.752122 |
| 难度最大值 | 19.038889 |

Benchmark v2.2已移除23条无法稳定通过离线数据门的题目，并用同一官方eval商品池中的有效任务替换。新版240题全部可进入正式评测合同。

## 2. Suite分布

| Suite | 题数 | 占比 |
|---|---:|---:|
| Core | 180 | 75.0% |
| Challenge | 60 | 25.0% |
| 合计 | 240 | 100.0% |

## 3. 领域分布

| 领域 | Core | Challenge | 总计 |
|---|---:|---:|---:|
| 服饰鞋包饰品 | 20 | 10 | 30 |
| 家用电器数码 | 20 | 9 | 29 |
| 生产材料农用品 | 20 | 9 | 29 |
| 美妆个护健康 | 20 | 9 | 29 |
| 家居家装 | 20 | 7 | 27 |
| 休闲娱乐文教 | 20 | 6 | 26 |
| 母婴儿童 | 20 | 5 | 25 |
| 运动户外交通 | 20 | 3 | 23 |
| 食品饮品 | 20 | 2 | 22 |
| 合计 | 180 | 60 | 240 |

Core保持每个领域20题；Challenge按实际高难候选分布补充。

## 4. 难度分布

| Difficulty bucket | Core | Challenge | 总计 |
|---|---:|---:|---:|
| under_10 | 41 | 2 | 43 |
| 10_15 | 115 | 35 | 150 |
| 15_18 | 23 | 21 | 44 |
| 18_plus | 1 | 2 | 3 |
| 合计 | 180 | 60 | 240 |

主分布集中在10–15难度段；15–18段共44题。18以上只有3题，相关评测比例应谨慎解读。

## 5. Gold检索位置分布

| Retrieval bucket | Core | Challenge | 总计 |
|---|---:|---:|---:|
| rank1 | 108 | 36 | 144 |
| rank2_5 | 46 | 15 | 61 |
| rank6_20 | 18 | 6 | 24 |
| rank21_150 | 7 | 2 | 9 |
| missing | 1 | 1 | 2 |
| 合计 | 180 | 60 | 240 |

96题的Gold不在首位，其中35题位于Rank 6之后或默认检索缺失，用于检验搜索改写、翻页和候选比较能力。

## 6. Challenge切片

| Challenge类型 | 题数 |
|---|---:|
| search_reformulation | 10 |
| candidate_comparison | 10 |
| price_semantics | 10 |
| multi_option | 10 |
| evidence_verification | 10 |
| long_horizon | 10 |
| 合计 | 60 |

六类Challenge互斥计数，每类固定10题。

## 7. 任务特征标签

以下特征允许重叠，因此合计会超过240：

| 特征 | 题数 |
|---|---:|
| approximate_price | 160 |
| multi_option | 86 |
| hard_budget | 69 |
| negation | 29 |
| compatibility | 16 |
| 无上述标签 | 9 |

## 8. 完整性与哈希

| 项目 | SHA-256 |
|---|---|
| `data/evaluation/tasks.jsonl` | `65537215966095c42240bd4e47931661f7f321d75c17beb44ad22ca7d54dbd6f` |
| `data/evaluation/slices.jsonl` | `d498bbf195aeda7ce63279b8ccb33498cd97c64f81833a5cc6ea2725cf4d4c83` |

数据源与详细替换映射见：

- `data/evaluation/metadata.json`
- `data/evaluation/replacement-manifest-v2.2.json`

该分布是当前Benchmark v2.2 Final-240的冻结基准。后续Base、SFT和GRPO的正式比较必须使用相同Task哈希与评测合同。
