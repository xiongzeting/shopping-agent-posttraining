# 当前保留的项目主线

本目录突出当前有效的完整主线：

```text
Final-1000 Teacher → SFT-v5 → GRPO step50/100 → Harness改善版 GRPO step230
→ Final-240 六模型评测（另含同协议 Qwen3.8-27B Actor）
```

## 0. 完整项目与面试总文档

- [长程购物 Agent 项目完整流程、工程讲解与面试问答](0.项目全流程/项目全流程与面试问答.md)
- 该文档将项目描述、Agent Harness、Teacher/SFT、Reward/GRPO、Final-240 评测和真实面试问答集中到一处。

## 1. Final-1000 Teacher 轨迹

- [Final-1000 Teacher / SFT 数据说明](2.sft阶段数据清洗及相关曲线/data.md)
- [Final-1000 Teacher 数据工程详解](2.sft阶段数据清洗及相关曲线/interview-final1000-data-engineering-retrospective.md)
- 原始保留目录：`outputs/teacher-v3.2/final1000-convergence-v4-20260809/`
- 最终训练数据：`data/sft-final1000-convergence-v5-30k/`

## 2. Final-1000 SFT 过程

- [SFT / GRPO 流程图说明](训练流程图/README.md)
- [SFT Final-1000 流程图](训练流程图/01-sft-v5-final1000-process.png)
- [真实训练曲线说明](训练过程曲线/README.md)
- [SFT Final-1000 训练曲线](训练过程曲线/SFT-Final1000/)
- 训练记录：`outputs/runs/sft/fresh-sft-convergence-repair-v5-30k/`
- 模型：`outputs/models/fresh-sft-convergence-repair-v5-30k-lora/` 与
  `outputs/models/fresh-sft-convergence-repair-v5-30k-merged/`

## 3. GRPO 训练与 Harness 改善

- [GRPO step-50 流程图](训练流程图/02-grpo-step50-process.png)
- [GRPO step-50 训练曲线](训练过程曲线/GRPO-step50/)
- 原始 checkpoint：`outputs/models/grpo-all-1to3-clean-20260813-r2/global_step_50/`
- HF merged：`outputs/models/grpo-step50-hf-merged/`
- 留痕包：`artifacts/latest-sftv5-grpo-step50-20260814/`

step-50 与 step-100 用于训练过程和中间 checkpoint 对比；当前最终对外结论使用
Harness 改善版 GRPO step230。同协议 Qwen3.8-27B 是独立 Actor 对照，不属于 SFT→GRPO
训练迁移链。

当前 Final-240 Evaluation Harness 的关键合同为：8 个公开工具、依据最新页面动态暴露可执行
Tool Schema、结构化 Observation 投影、2,560/3,072/512 页面预算与独立 2,048 Token 候选选择/记忆预算；
候选稳定保存为 C1-C4，达到 3 个后动态前置收敛提醒。候选记忆不提供直达工具，历史商品需要
重新搜索后从最新结果页打开；连续 3 步无进展先提醒、6 步由环境终止，35/40 步显示收敛提醒。
完整设计与训练/评测边界见 [Harness 设计目录](5.harness设计/README.md)。

## 4. 最新 Final-240 六模型评测

- [最终 HTML Dashboard](4.评测阶段/dashboard.html)
- [最终审计报告](4.评测阶段/audit-report.md)
- 完整目录：`重点/4.评测阶段/`

最新严格 Gold：Base `0/240`、SFT `142/240`、GRPO50 `154/240`、GRPO100
`153/240`、Harness 改善版 GRPO230 `163/240`、Qwen3.8-27B `146/240`。
当前训练主线最佳结果是 GRPO230 的 `67.92%`；Qwen3.8-27B 为同协议外部 Actor
对照，严格 Gold 为 `60.83%`。Gold＋Valid 购买成功分别为 `180/240` 与 `172/240`。

最终评测的原始 SFT 与 GRPO rollout 来源目录继续保留在 `outputs/evaluation/`，
用于审计和复现；其余历史评测版本已归档到本地 `old/`。

## 历史归档

`old/` 被 Git 忽略，只作为本机可恢复归档。没有执行删除；移动清单记录在
`old/ARCHIVE_MANIFEST.json`。
