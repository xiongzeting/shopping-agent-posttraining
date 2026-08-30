# ShopBench-LH Final-240

## 当前协议

- Benchmark：ShopBench-LH Final-240
- Environment：ShopSimulator v2.4
- Reward：Reward v4
- Termination：v3.2
- Observation：v2；Evaluation Tool Schema：v2.1；Search：v2.1
- temperature：0
- top_p：1
- max_steps：45
- max_tokens：768
- context window：30000

Final-240 已冻结并完成八组评测。历史 Harness v1/v2 与最新 Harness v3 分组展示，当前
公开报告统一使用 Reward v4 离线重放，不把旧实验重新标注为 v3。

Benchmark 名称仍为 **ShopBench-LH Benchmark v2**。本次升级只修改正式评测管线、
统计方法和运行产物，不修改 Final-240 题目、切片、metadata 或已冻结 SHA-256。

## 任务结构

```text
Final-240
├── Core-180：9 个一级领域 × 20 题
└── Challenge-60：6 个困难切片 × 10 题
```

六个困难切片为：

1. `search_reformulation`：目标标题与用户表达词面差异较大；
2. `candidate_comparison`：同类候选密集，需要比较而非看到相似商品就购买；
3. `price_semantics`：硬预算、左右、多元、出头等价格边界；
4. `multi_option`：多个规格轴、较多 variant 或规格价格变化；
5. `evidence_verification`：要求较多，需要核验详情、属性或功能证据；
6. `long_horizon`：综合难度较高，容易出现无效探索或循环。

特殊无解、矛盾需求和推荐相似商品任务暂不进入本版，因为它们需要新的任务成功定义、
Reward 档位和结构化推荐终局。该方向仅作为后续版本展望。

## 数据隔离

Final-240 只从产品源数据的 `tag=eval` 池选择，并满足：

- 与当前 SFT task ID 重叠为 0；
- 与历史 Final-200 task ID 重叠为 0；
- 最终集合内部 `family_id` 不重复；
- 对同类目标题和 Query 做近重复检查；
- 只在所有选择规则冻结后生成一次正式任务清单。

公开的 `tasks.jsonl` 仍只包含 `task_id`。领域、难度和切片标签保存在
`data/evaluation/slices.jsonl`，不包含用户 Query 或 Gold 商品私有字段。

## 四面板结果

### A. 环境 Reward 和终局结果

- strict gold success；
- purchase success；
- Reward type、valid 和 terminal utility；
- done、over、termination reason；
- 缺失任务固定计入 Final-240 分母。

严格成功仍要求完整 `gold_purchase` 终局并且 `reward_valid=true`。

### B. 用户需求满足情况

- Reward v4逐约束结果：`pass / fail / unverifiable`；
- 按`hard_gate / matching_dimension / strict_target_variant`和约束类型汇总；
- 每条 Rubric 的 `satisfied / violated / unknown / not_applicable`；
- 按 hard、soft 和 needs-review 分组；
- Reward 与 Rubric 冲突单独记录，不互相覆盖。

### C. 轨迹过程质量

- search strategy；
- candidate utilization；
- evidence verification；
- decision quality；
- termination efficiency。

五个维度分别使用0、1、2分，并要求给出原因和轨迹证据。LLM Judge只用于诊断，
不覆盖Environment Reward，也不生成综合总分。Judge失败或轨迹基础设施无效时保留为空，
不能伪造成0分。

### D. 确定性行为指标

- 工具调用和执行步数；
- 重复搜索与重复动作；
- Action Guard、非法调用和 step error；
- Observation 投影和上下文使用；
- 输入、Completion 和总 Token（服务端提供 usage 时）；
- Observation 可见/原始 Token 比例；
- 最大上下文预算使用率；
- 模型耗时、工具耗时和轨迹总耗时；
- infrastructure-invalid 任务。

步数不再只报告一个混合平均值，同时分别报告严格成功任务和失败任务的均值与中位数。
Observation 不再只报告“是否截断”，还报告压缩比例和最大上下文预算使用率。

## 正式评测管线

唯一正式路径为：

```text
rollout
→ normalize
→ deterministic metrics
→ rubric / judge（可选离线输入）
→ per-task four-panel evaluation
→ summary
```

`scripts/evaluate_shop_benchmark.py` 不再使用旧 `summary.py` 作为正式结果入口。未提供
冻结 Rubric 时，需求面板仍报告 Reward v4 的逐约束结果，但不凭空生成 LLM Rubric；
未提供 Judge 时，轨迹质量面板明确记为 `not_judged`，不伪造 0 分。Judge 始终只用于
诊断，不参与严格成功判定或模型主排名。

每次运行统一写入：

```text
RUN_DIR/
  run_manifest.json
  trajectories.jsonl
  evaluations.jsonl
  summary.json
```

`run_manifest.json` 保存数据和环境哈希、运行协议、seed、System Prompt 与 Tool Schema
哈希以及所有结果文件哈希；不保存 API Key 和机器绝对路径。

## 比较方式

Baseline、SFT 和 GRPO 必须使用同一份 Final-240、同一环境和同一推理参数，并按
`task_id` 配对报告：

- failure → success；
- success → failure；
- Reward type 迁移；
- Rubric hard violation 变化；
- Reward约束`fail / unverifiable`数量变化；
- 五个轨迹维度变化；
- 步数、Guard 和重复动作变化。

缺失任务在严格成功比较中按失败处理，仍使用完整 Final-240 分母；连续指标只在两次
运行都存在有效逐题结果时配对。六个 Challenge 切片每个只有 10 题，只作为描述性
诊断，不单独宣称显著性。

模型比较只保留容易解释的内容：严格成功率、failure→success、success→failure、
Reward 类型变化，以及步数、Guard、重复动作和各诊断维度的平均变化。不做显著性检验，
也不生成综合总分。

不把四个面板合成一个不可解释的总分。

汇总同时按`suite`、`domain`和`challenge_slice`输出，缺失任务继续计入各自分层
分母。正式运行入口会在Rollout前核验任务、切片和metadata哈希、当前SFT零重叠，
以及Environment v2.4 / Reward v4合同；任一不一致都会停止运行。

## 数据与构建入口

```text
data/evaluation/tasks.jsonl
data/evaluation/slices.jsonl
data/evaluation/metadata.json
scripts/build_evaluation_benchmark.py
```

构建脚本用于冻结新版本，不得在看到 Final-240 模型结果后重新抽题。

正式运行：

```bash
python scripts/evaluate_shop_benchmark.py \
  --benchmark data/evaluation/tasks.jsonl \
  --run-dir outputs/evaluation/baseline \
  --model shopping-agent \
  --llm-base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY
```

配对比较：

```bash
python scripts/compare_shop_benchmark.py \
  --run baseline=outputs/evaluation/baseline/evaluations.jsonl \
  --run sft=outputs/evaluation/sft/evaluations.jsonl \
  --run grpo=outputs/evaluation/grpo/evaluations.jsonl \
  --output outputs/evaluation/paired_comparison.json
```
