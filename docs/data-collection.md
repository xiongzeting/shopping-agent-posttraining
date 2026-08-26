# SFT 数据与采集合同

## 当前保留数据

`data/sft/` 保存唯一的 canonical SFT v5 数据，按 task ID 划分为
900 train / 100 validation。数据门和 tokenizer audit 均已通过。

| 文件 | 行数 | SHA-256 |
|---|---:|---|
| `data/sft/all.jsonl` | 1000 | `a4a8919dd6c51e72cb1bfaac08b0d048ccf523bf6c521d5bc9a3833df8e7750e` |
| `data/sft/train.jsonl` | 900 | `f11c5507caef84aadcdb08eecbe37ccda06fb2ec6ff93737e374938b502cfeaf` |
| `data/sft/validation.jsonl` | 100 | `e36cda3671587896c77bfed55dde9cf2740353aac9fe5819a543f68e84a30649` |

train / validation 重叠为 0；与 Final-240 task ID、ASIN、family 和语义近似重叠均为 0。

## 当前采集合同

新收集的 Teacher 轨迹必须通过当前合同：

### Teacher Prompt v2

Teacher 采集使用 `shopping-teacher-prompt-v2`，目标是“最短的充分证据轨迹”：不按预设
步数生成 Short、Medium 或 Long，也不把最短路径本身当成质量目标。容易任务在证据闭合后
及时购买；困难任务不能省略任务本身需要的搜索改写、候选比较、关键属性核验、多规格选择
和最终 variant 价格确认。

每个任务可携带 `focused_verification`、`search_reformulation`、
`candidate_comparison`、`evidence_verification`、`price_semantics` 或 `multi_option`
策略标签。标签只规定该题需要完成的能力过程，不允许通过重复搜索、无关页面或错误候选
人为增加长度。采集 metadata 必须保存 Prompt 版本、基础 Prompt SHA-256 和六种完整策略
Prompt 的 SHA-256。

### 第一道门：结果正确

```text
Environment v2.4
Reward v3.2
最终动作 buy_now
reward_type = gold_purchase
reward_valid = true
terminal done = true
terminal over = true
轨迹中存在实际 buy_now
轨迹没有执行错误
```

顶层 `status`、顶层 `done`、`purchase_success`、Reward 内的 `termination_reason` 和环境
释放错误不再重复参与硬拒绝。它们作为一致性告警写入采集统计：前四项用于发现终局字段
复制或协议异常；`release_error` 用于暂停和修复环境资源，但不会删除已经完整完成的 Gold
购买轨迹。

### 第二道门：过程合格

过程门只排除不可恢复或不适合模仿的结构缺陷：

- 出现工具调用截断、多工具调用、实际执行的非法动作或 step error；
- 没有形成 `search → open_product → buy_now` 的基本路径；
- 连续重复完全相同的动作，且无法证明第二次调用没有改变可见环境状态。

Action Guard 拒绝本身不是已执行的环境 step。只要轨迹随后满足严格结果门，就保留恢复
后的成功过程，并在 SFT 序列化时删除被拒绝的 assistant/tool 消息。非连续的搜索改写也
不再被过程门误删。连续重复动作只有在两次动作签名一致、第二次未终局且两次返回的结构化
`observation_state` 完全一致时才视为可清洗 no-op；此时删除第二个 assistant/tool 对并保留
整条成功轨迹。缺少结构化状态、页面发生变化或涉及购买终局时仍然硬拒绝。

### 第三道门：数据集覆盖

最终 1000 条必须通过 `shopping-teacher-data-gate-v1`：

- Gold@1 最多 60%，Gold@2-5 至少 20%，Gold@6-20 至少 12%；
- Gold@21-150 至少 6%，完整 Query 的 Top-150 不命中至少 2%；
- 搜索改写至少 20%，候选比较至少 25%，多规格选择至少 20%；
- Guard 恢复至少 5%，11 步以上至少 35%，20 步以上至少 10%；
- 任一完整工具序列最多 12%，恰好 8 步最多 30%。

审计报告必须与 canonical metadata 一起保存并记录 SHA-256。预检会核对报告版本、完整
阈值、500 个唯一 task、通过状态与文件哈希，不能只修改 metadata 冒充通过。

当前收集器不接受有效替代品、部分购买、主动停止、循环、不可验证或错误购买轨迹。
同一任务有多个通过轨迹时仍按首次通过的轨迹入库，不额外使用复杂打分或“越短越好”
规则。拒绝统计分别记录 `result_gate_rejected` 和 `process_gate_rejected`。

## 数据隔离

新的 SFT 或 GRPO 数据必须同时检查：

- train / validation task ID 不重叠；
- 与 Final-240 task ID 不重叠；
- SFT 与 GRPO task ID 不重叠；
- 必要时按商品 family 去重；
- metadata 记录环境、Reward、Observation、Tool Schema 和数据 SHA。

当前 `data/grpo/` 尚未构建，不存在可用于训练的 GRPO 数据。
