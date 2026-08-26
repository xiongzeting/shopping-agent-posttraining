# Canonical SFT 数据与运行合同

`data/sft/` 是唯一保留的 SFT 数据集，来源版本为
`final1000-convergence-v5-30k`，数据 schema 为 `shopping-sft-dataset-v3`。

## 数据

- 1000 条严格成功 Teacher 轨迹，900 train / 100 validation；
- Teacher selection：`shopping-teacher-recoverable-process-v4`；
- 结果门：完整 `gold_purchase` 且 `reward_valid=true`；
- 数据门：`data/sft/data_gate.json`，状态为 `passed`；
- 与 Final-240 的 task ID、ASIN、family 和 semantic overlap 均为 0；
- Rank 1 / 2–5 / 6–20 / 21–150 / missing：501 / 299 / 100 / 70 / 30；
- Short / Medium / Long：477 / 400 / 123；
- 唯一动作序列 452 种，Top-1 占 7.5%。

## Tokenizer 审计

Qwen3.5-2B 在 `max_length=30000` 下：

- train：900/900 保留，最大 29533 tokens；
- validation：100/100 保留，最大 29994 tokens；
- 3 epochs、batch size 1、gradient accumulation 8，共 339 个 optimizer steps。

完整记录见 `data/sft/token_audit.json`。

## 固定训练参数

- Base：Qwen3.5-2B；
- BF16 LoRA：r=16，alpha=32，dropout=0.05；
- max length：30000；
- epochs：3；
- train/eval batch size：1；
- gradient accumulation：8；
- learning rate：1e-4，warmup ratio：0.03；
- gradient checkpointing：关闭；
- attention：Flash Attention 2；
- Liger：开启；
- QLoRA、自动 merge：默认关闭；
- seed：42。

唯一权威参数文件为 `configs/sft_canonical.json`。

## 启动前检查

无卡主机可执行：

```text
bash scripts/sft.sh --preflight-only --skip-gpu-check
```

正式训练机必须重新执行：

```text
bash scripts/sft.sh --preflight-only
```

只有用户明确要求开始 SFT 后，才能执行 `bash scripts/sft.sh`。
