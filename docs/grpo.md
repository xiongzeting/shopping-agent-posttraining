# GRPO 状态

项目已完成 GRPO100 与 GRPO230 历史训练及 Final-240 评测。公开仓库提供运行代码、配置、
聚合结果和审计材料，但不发布训练 Parquet、原始 rollout、checkpoint 或模型权重。

## 已具备

- veRL 0.8 多轮 AgentLoop；
- ShopSimulator Tool adapter；
- Environment v2.4 / Reward v4 公共终局字段适配；
- 过滤低差异 Reward 组和不可验证组的有界动态采样；
- Observation 投影、Action Guard 和上下文预算；
- `configs/grpo.yaml`、`configs/agent_loop.yaml`、`configs/tools.json`；
- 预检、训练和 checkpoint 导出入口。

当前动态采样以同一 prompt 的4条轨迹为一组。terminal utility范围不超过`0.025`
视为低差异组；初次采样失败后最多再采两次，第三次仍无有效信号则丢弃该组。
动态采样不包含难度课程或基于Final评测的任务路由。

## 已有结果

- GRPO100 v1：Strict Gold 63.75%，Gold + Valid 72.08%；
- GRPO230 v2：Strict Gold 67.92%，Gold + Valid 75.00%；
- GRPO230 v3：Strict Gold 73.33%，Gold + Valid 81.25%；
- 完整分组、Reward v4 重放和盲评审计见 `reports/final240/`。

## 新一轮训练前

1. 用`data/grpo/training-probe-v1/calibration-200.jsonl`完成首轮在线probe；
2. 校验 task ID、商品 family、环境 manifest 和数据 SHA；
3. 完成 CPU 预检和 GPU 一步训练 smoke；
4. 冻结模型、Prompt、Harness、Reward 和训练配置；
5. 用户明确授权后才启动正式训练。

不得使用 Final-240 结果选择 GRPO 训练数据或调整超参数。
