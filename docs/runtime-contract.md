# 当前运行合同

本文只描述新运行必须满足的合同，不把历史 SFT 结果重新标注为当前版本。

## 组件组合

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

对应的机器可读版本为：

| 组件 | 标识 |
|---|---|
| Environment | `shopsimulator-environment-v2.4` |
| Reward | `shopsimulator-reward-v4` |
| Reward Features | `shopping-reward-features-v2` |
| Query Constraints | `shopping-query-constraints-v1` |
| Termination | `shopping-termination-v3.2` |
| Observation | `shopping-observation-v2` |
| Search | `shopsimulator-multifield-bm25-v2.1` |
| Tool Schema | `shopping-tools-v2` |
| Final-240 Evaluation Tool Schema | `shopping-evaluation-tools-v2.1` |

权威机器清单是 `data/environment.json`；环境内部配置是
`environments/ShopSimulator/shop_env/configs/environment.json`。新运行必须通过版本、
商品数据哈希和运行文件哈希检查，不能只在文档或 metadata 中改版本号。

## 严格成功

项目把“环境完成了一次购买”和“严格命中 Gold”分开。严格成功要求完整终局同时满足：

```text
reward_type = gold_purchase
reward_valid = true
purchase_success = true
termination_reason = gold_purchase
done = true
over = true
```

其中：

- `purchase_success=true` 只说明购买动作形成了购买终局；
- `gold_purchase` 说明购买的 ASIN 和严格目标规格均命中；
- `reward_valid=true` 说明 Reward 所需的终局证据可验证。

因此“买了东西”不等于严格成功；有效替代购买、部分满足和错误购买都可能有购买终局，
但不能计入 strict success。

## 固定运行参数

Final-240 的默认协议为：

| 参数 | 值 |
|---|---:|
| `temperature` | 0 |
| `top_p` | 1 |
| `max_steps` | 45 |
| 单回合 `max_tokens` | 768 |
| context window | 30000 |
| 输入预算 | 28720 tokens（另留 768 generation + 512 safety margin） |
| 搜索页 Observation | 2560 tokens，最多 20 个商品 |
| 商品详情 Observation | 3072 tokens |
| 搜索首页、信息子页、终局与普通页 Observation | 512 tokens |
| 候选记忆 | 独立附加 1024 tokens，不挤占页面正文 |

正式比较 Baseline、SFT 和 GRPO 时，任务文件、环境合同、模型推理参数、Prompt、工具
Schema 和 Harness 参数必须一致。

## 当前数据与结果边界

- Final-240 已冻结并完成八组独立评测；历史 Harness v1/v2 结果保持原版本标记，v3 结果使用当前合同；
- 当前公开报告统一用 Reward v4 离线重放，并使用冻结 Rubric 与轨迹盲评审计；
- 新训练、Teacher 采集、推理和评测入口默认使用 Harness v3；历史数据和 checkpoint 不会被重新标注；
- 聚合结果位于 `reports/final240/`，公开仓库不包含原始轨迹、Judge JSONL、训练数据或 checkpoint；
- 特殊无解、澄清、多商品购物和相似推荐不属于当前任务合同。

评测结构与发布边界见 `docs/evaluation.md` 和 `reports/final240/audit-report.md`。
