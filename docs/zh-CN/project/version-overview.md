# 当前版本总览

本目录主要保存当前有效组件的版本快照。Reward v1/v2/v3 的设计演进记录作为历史
资料保留在 [`rewardv1,v2,v3.md`](rewardv1,v2,v3.md)，它不是当前运行合同。
历史实验版本以 `results/sft/` 中的结构化记录为准。

| 组件 | 当前版本 | 文档 |
|---|---|---|
| Benchmark | `shopping-evaluation-dataset-v2.1` | [Benchmark-v2.md](Benchmark-v2.md) |
| Environment | `shopsimulator-environment-v2.4` | [Environment-v2.4.md](Environment-v2.4.md) |
| Reward | `shopsimulator-reward-v4` | [Reward-v4.md](Reward-v4.md) |
| Observation | `shopping-observation-v2` | [Observation-v2.md](Observation-v2.md) |
| Search | `shopsimulator-multifield-bm25-v2.1` | [Search-v2.1.md](Search-v2.1.md) |
| Termination | `shopping-termination-v3.1` | [Termination-v3.1.md](Termination-v3.1.md) |
| Tool Schema | `shopping-tools-v2` | [Tool-Schema-v2.md](Tool-Schema-v2.md) |
| Evaluation Tool Extension | `shopping-evaluation-tools-v2.3` | [Tool-Schema-v2.md](Tool-Schema-v2.md) |
| Candidate Memory | `shopping-candidate-memory-v2` | [Observation-v2.md](Observation-v2.md) |
| Evaluation Termination Extension | `shopping-termination-v3.2` | [Termination-v3.1.md](Termination-v3.1.md) |
| Evaluation | `Final-240 v2.2 / six-model report` | [评测审计报告](4.评测阶段/audit-report.md) |

当前兼容组合：

```text
Environment v2.4
├── Reward v4
├── Reward Features v2
├── Query Constraints v1
├── Termination v3.1
├── Observation v2
├── Search v2.1
├── Tool Schema v2（8 个基础工具）
└── Final-240 Harness extensions
    ├── 页面级动态 Tool Schema
    ├── Candidate Memory v2（C1-C4）
    ├── 候选专用 System Prompt 与候选间硬重置
    └── 6步无进展 / 30步预算触发的顺序候选收敛
```

版本号独立演进。Environment v2.4 表示这组组件和运行文件哈希被共同冻结，不代表
所有子组件都升级为 v2.4。

上述 C1-C4、候选专用 System Prompt、C1→C4 自动逐个 `reopen` 和候选间上下文硬重置已经在
Final-240 Evaluation 与 GRPO Adapter 两条入口对齐，并由差分合同测试校验。普通探索阶段不展示候选
记忆；强制阶段每次只向模型发送当前一个候选。

当前 Final-240 已完成 Base、SFT、GRPO50、GRPO100、Harness 改善版 GRPO230 和
Qwen3.8-27B 六模型统一评测。六组共享同一 240 题、冻结 Rubric 和 DeepSeek V4 Pro
盲评合同；当前 Reward v4 聚合口径下严格 Gold 分别为 `0/240`、`142/240`、`154/240`、
`153/240`、`163/240` 和 `146/240`，Gold＋Valid 购买成功分别为 `0/240`、`164/240`、
`167/240`、`173/240`、`180/240` 和 `172/240`。历史 SFT、Final-200 和中间 checkpoint
结果仍按原合同保留，不会被静默改写。
