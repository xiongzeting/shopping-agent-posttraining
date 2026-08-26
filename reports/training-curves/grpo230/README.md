# GRPO step 1–230 连续训练曲线

本目录只参考`../GRPO-step50/`中11张图片的版式与指标组合，不读取或复用其中的step-50 CSV数据。

## 最终有效训练链路

- step 1–90：`raw-remote-20260822/grpo-step1-to90-training.log`
- step 91–150：`raw-remote-20260822/grpo-step90-to150-training.log`
- step 151–230：`raw-remote-20260822/grpo-step150-to230-supervisor.log`

step 150–230阶段曾在step 190停止并从最新checkpoint恢复，因此`training.log`只含后半段；`supervisor.log`保留完整续训链路。脚本按step区间选择最终有效记录，并对恢复时重复出现的step 191–195采用最后一条Actor更新记录。

## 数据口径

- `grpo-step230-actor-updates.csv`和`grpo-step230-sampling-by-step.csv`均严格包含step 1–230共230行。
- 动态采样把若干step标记为`sampling/skipped_actor_update=1`。这些step有真实Reward、成功率、长度、采样与丢弃指标，但没有Actor loss、梯度、KL等更新指标；CSV保持为空，图片只连接相邻真实Actor观测，不伪造更新值。
- 所有图共用连续的global step横轴1–230，不按三次训练运行拆图。

运行：

```text
python build_step230_curves.py
```

会重新生成2份CSV和11张PNG。
