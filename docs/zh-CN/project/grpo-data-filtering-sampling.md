# GRPO 数据筛选与动态采样

## 1. 为什么需要在线 Probe

GRPO 需要同一 Prompt 的多条轨迹形成有效相对优势。若四条轨迹全部失败、全部成功或 Reward
几乎相同，组内信号很弱，继续更新只会消耗显存和采样时间。因此任务进入训练前需要同时通过：

1. 离线商品与规格数据门；
2. 与 SFT、Final-240、Dev Probe 的隔离门；
3. 在线 n=4 Rollout 的 Reward 有效性和差异门。

## 2. 离线候选数据门

第一轮从 train-tag 商品中排除已用 task/family 和语义近重复后，进入候选审计。主要拒绝原因：

| 原因 | 次数 |
|---|---:|
| pricing option range mismatch | 367 |
| no verifiable variant price | 340 |
| required options unresolved | 291 |
| normalized option value collision | 41 |

同一商品可能命中多个原因，所以原因数之和不等于拒绝行数。最终按难度、领域和策略配额选出
800 个候选，单领域上限为 25%，且与排除集合 task/family 重叠为 0。

## 3. step 1-50 任务池

第一轮按 Calibration 200 + Remaining 600 执行 n=4 Probe：

| 阶段 | 候选 | 在线准入 |
|---|---:|---:|
| Calibration | 200 | 65 |
| Remaining | 600 | 186 |
| 合计 | 800 | 251 |

从 251 个准入任务中选择具有混合成功/失败轨迹的组：

| 组内 Purchase Success | 任务数 |
|---|---:|
| 1 / 4 | 45 |
| 2 / 4 | 51 |
| 3 / 4 | 85 |
| 合计 | 181 |

正式 step 1-50 使用这 181 个 `all-1to3-clean` frontier，而不是早期试验过的 200 题混合池。

## 4. step 50-100 新任务池

续训前构建与旧 GRPO、SFT 和 Final-240 隔离的 2,000 题新候选池，难度配额为 20% 简单、
60% 中等、20% 困难。实际在线 Probe 产生 2,827 条轨迹，覆盖 709 个唯一 task_id；达到 100
个有效 frontier 后早停：

| 组内成功数 | 任务数 |
|---|---:|
| 1 / 4 | 44 |
| 2 / 4 | 56 |
| 合计 | 100 |

准确表述是“从 2,000 个新候选中实际 Probe 709 题后早停筛出 100 题”，不能说 2,000 题全部
完成了在线 Probe。

## 5. 动态采样与有界重采样

当前训练配置：

- 每个任务组 `n=4`；train batch size 为 2，一次更新最多使用 8 条有效轨迹；
- `reward_valid=false` 或基础设施无效的轨迹不进入更新；
- 四条 Reward 相同，或组内最大值与最小值之差 `<=0.025`，视为低信息组；
- 一个 n=4 组最多生成 3 批，即初始采样加最多两次重试；
- 第三批仍不合格则整组丢弃；
- 一个训练 step 可得到 0、4 或 8 条有效轨迹；0 组可用时允许跳过 Actor 更新；
- 保持 GAPO 风格动态采样，不关闭 dynamic sampling。

训练使用无 KL Reward，`norm_adv_by_std_in_grpo=false`，借鉴 Dr.GRPO 不用组内标准差放大小方差
噪声；Loss 使用 `seq-mean-token-mean`，避免长失败轨迹仅因 Token 更多而主导更新。

## 6. 数据隔离

两轮正式训练池内部 task_id 唯一，train / validation 零重叠，并对 Final-240、SFT-1000、旧 SFT
执行显式 task overlap 检查。第二轮 train 与 validation 各 100 个唯一任务，Final-240 和 SFT
重叠均为 0。
