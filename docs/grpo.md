# GRPO 状态

GRPO 尚未开始。仓库只包含运行代码和基础配置，不包含可声明的训练结果。

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

## 尚未具备

- GRPO train / validation Parquet；
- 正式 rollout；
- checkpoint 或合并模型；
- 训练曲线和稳定性结果；
- Final-240 评测；
- 相比 SFT 的提升结论。

## 开始训练前

1. 用`data/grpo/training-probe-v1/calibration-200.jsonl`完成首轮在线probe；
2. 校验 task ID、商品 family、环境 manifest 和数据 SHA；
3. 完成 CPU 预检和 GPU 一步训练 smoke；
4. 冻结模型、Prompt、Harness、Reward 和训练配置；
5. 用户明确授权后才启动正式训练。

不得使用 Final-240 结果选择 GRPO 训练数据或调整超参数。
