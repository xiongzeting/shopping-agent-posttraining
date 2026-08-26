# 数据状态

| 阶段 | 入口 | 状态 |
|---|---|---|
| Environment | `environment.json` | Environment v2.4 |
| SFT | `sft/` | canonical v5，1000条 |
| GRPO | `grpo/` | 尚未构建 |
| Evaluation | `evaluation/` | Final-240 已冻结，未执行 |

## SFT

- `all.jsonl`：1000条；
- `train.jsonl`：900条；
- `validation.jsonl`：100条；
- `data_gate.json`：当前 Teacher 数据门不可变审计；
- `token_audit.json`：Qwen3.5-2B、30K上下文 tokenizer 审计；
- `metadata.json`：数据哈希、运行合同、分布和零重叠声明。

当前数据通过 `shopping-teacher-recoverable-process-v4` 与
`shopping-teacher-data-gate-v1`，训练参数以 `configs/sft_canonical.json` 为准。

## Evaluation

Final-240 包含 Core-180 和 Challenge-60。`metadata.json` 中 `evaluated=false`，表示尚未
运行当前正式 Baseline、SFT 或 GRPO 评测。
