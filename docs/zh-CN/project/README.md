# 项目设计与面试资料

这组文档记录长程购物 Agent 从环境、数据清洗、SFT、Reward/GRPO、Harness 到 Final-240 评测的完整设计。内容来自当前项目的关键说明文件，已排除模型权重、原始轨迹、虚拟环境、缓存和临时实验产物。

## 推荐阅读顺序

1. **项目总览**：[`project-overview.md`](project-overview.md)、[`version-overview.md`](version-overview.md)
2. **环境与执行合同**：[`environment-v2.4.md`](environment-v2.4.md)、[`observation-v2.md`](observation-v2.md)、[`search-v2.1.md`](search-v2.1.md)、[`tool-schema-v2.md`](tool-schema-v2.md)、[`termination-v3.1.md`](termination-v3.1.md)
3. **Reward 设计**：[`reward-v3.2.md`](reward-v3.2.md)、[`reward-v4.md`](reward-v4.md)、[`reward-v4-interview.md`](reward-v4-interview.md)
4. **数据与训练**：[`sft-data-quality-gates.md`](sft-data-quality-gates.md)、[`grpo-data-filtering-sampling.md`](grpo-data-filtering-sampling.md)
5. **Harness**：[`harness-design.md`](harness-design.md)、[`harness-detailed.md`](harness-detailed.md)、[`harness-interview.md`](harness-interview.md)
6. **评测与归因**：[`evaluation-audit.md`](evaluation-audit.md)
7. **背景对照**：[`shopsimulator-comparison.md`](shopsimulator-comparison.md)
8. **面试串讲**：[`project-full-flow-interview.md`](project-full-flow-interview.md)

## 文档说明

- 版本文档保留历史演进，阅读时以文档标注的版本为准，不将历史实验数字与当前运行合同混用。
- 评测文档只描述聚合指标、审计规则和可复现的分析方法，不包含大规模原始轨迹。
- 训练和评测所需的密钥、内部主机信息及生成的模型文件不在仓库中。
