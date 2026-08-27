# Long-Horizon Shopping Agent: SFT + GRPO

本项目基于淘宝商品快照环境构建长程购物 Agent 的后训练闭环，覆盖：

```text
Teacher Data → LoRA SFT → Online GRPO → Final-240 Evaluation
```

Agent 需要在最多 45 步内完成搜索、候选比较、详情核验、规格选择以及购买或合理停止。核心模型为 Qwen3.5-2B，训练与推理栈包括 LoRA、veRL、vLLM 和 ShopSimulator。

## 主要结果

Final-240 与训练数据零重叠，包含 Core-180 和 Challenge-60。当前报告统一使用 Reward v4 离线重放，并通过冻结 Rubric 与轨迹盲评进行独立审计。

| 模型 | 严格 Gold | 成功购买（Gold + Valid） |
|---|---:|---:|
| Base v1 | 0.00% | 0.00% |
| SFT v1 | 59.17% | 68.33% |
| GRPO100 v1 | 63.75% | 72.08% |
| GRPO230 v2 | 67.92% | 75.00% |
| Qwen3.8-27B v2 | 60.83% | 71.67% |
| GRPO230 v3 | **73.33%** | **81.25%** |
| Qwen3.8-27B v3 | 67.92% | 75.42% |

完整聚合结果见 [Final-240 Dashboard](reports/final240/dashboard.html) 和 [审计报告](reports/final240/audit-report.md)。

## 核心设计

- **Agent Harness：** System Prompt、页面级动态 Tool Schema、结构化 Observation、Action Guard 和单工具串行执行共同约束多轮交互。
- **长上下文：** 按完整 Assistant–Tool 交互组裁剪历史，使用搜索页、详情页、普通页面和候选记忆的独立预算支撑 30K 上下文。
- **候选记忆：** 保存最多四个已核验候选；循环或无进展达到阈值时进入候选收敛流程，完成最终选择、规格闭合与终局动作。
- **Teacher + SFT：** 针对循环恢复、近似商品拒绝、规格精确选择和终局工具提交等 Bad Case 构建纠错轨迹，使用 Assistant-only Loss 进行 LoRA SFT。
- **Reward v4：** 将公开 Query 编译为可审计 Hard/Soft 约束合同，区分 Gold、Valid、Partial、错误购买、循环、非法动作与停止，并从第 16 步起施加递增步数惩罚。
- **GRPO：** 使用无 KL LoRA GRPO、Dr.GRPO 去标准差思路和 GAPO 动态重采样，提高组内有效学习信号比例。

## 目录

```text
src/shopping_grpo/          Agent、训练与评测核心代码
environments/ShopSimulator/ ShopSimulator 集成快照与 Reward v4
scripts/                    数据、SFT、GRPO、评测和报告入口
configs/                    训练、Rollout 与工具配置
tests/                      单元测试和合同测试
data/                       小型任务合同、Final-240 定义与审计元数据
docs/                       Harness、Reward、训练和评测文档
reports/                    聚合评测页面与派生审计结果
patches/                    veRL 0.8.0 集成补丁
```

## 安装与检查

建议使用 Linux、Python 3.10+ 和 NVIDIA GPU。仅运行代码检查与 CPU 测试时不需要下载模型权重。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

完整环境准备：

```bash
bash scripts/setup.sh
bash scripts/start_environment.sh
```

训练入口：

```bash
bash scripts/sft.sh --preflight-only
bash scripts/grpo.sh --preflight-only
```

评测入口：

```bash
bash scripts/evaluate.sh
```

首次运行前请将 `.env.example` 复制为 `.env`，再通过环境变量提供模型服务地址和 API Key。不要将 `.env` 或真实密钥提交到 Git。

## 发布边界

本公开副本不包含：

- Base、SFT 或 GRPO 模型权重与 checkpoint；
- 私有 Teacher 原始轨迹、完整 SFT 样本和 GRPO 训练 Parquet；
- 原始 Rollout、逐条 LLM Judge 响应、API 调用缓存和运行日志；
- 虚拟环境、搜索索引、模型缓存、临时审计目录、简历/PPT和个人面试材料。

因此，仓库可以审查核心实现、运行合同与聚合结果，但若没有对应训练数据和历史 checkpoint，不承诺逐项复现历史数值。第三方来源和许可边界见 [THIRD_PARTY.md](THIRD_PARTY.md)。
