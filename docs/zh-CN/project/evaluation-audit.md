# Final-240 八组统一评测：Harness v1 / v2 / v3

## 评测合同

- 八组结果使用同一240题 Final-240、同一冻结 DeepSeek V4 Flash Rubric、同一 DeepSeek V4 Pro Judge Prompt与Schema。
- 普通 SFT v1 使用1000条普通Teacher数据；纠错 SFT v1 使用600条普通数据与400条纠错数据。两者均使用Harness v1。
- Base、普通 SFT、纠错 SFT、GRPO100使用Harness v1；原GRPO230与Qwen3.8-27B使用Harness v2；新增GRPO230与Qwen3.8-27B使用Harness v3。各组均按同一盲评合同统计。
- 成功率、Reward值和Reward类型按当前Reward v4聚合重算；Rubric、Judge与确定性过程指标保持冻结口径。
- Judge只看Query、Rubric、Actor可见轨迹和白名单行为指标，不看Reward与Gold私有字段。

## Harness版本演进

- **v1 → v2：** 增加35步收敛提醒、模型输出文字但未调用工具时的拒绝纠正，以及循环/无进展提醒；删除 `view_description`、`view_features`、`view_reviews`、`view_attributes` 四种低频信息工具，并将原信息子页中的非空内容直接合并到商品详情，使模型调用 `open_product` 后即可在最新 Observation 中一次获得完整商品信息。
- **v2 → v3（当前顺序候选实现）：** 将搜索页、详情页和普通页面的Observation预算由 `1536 / 4096 / 768` 调整为 `2560 / 3072 / 512` Token，候选详情与候选记忆单独使用 `2048` Token；增加 C1-C4 已核验候选记忆与页面/阶段级动态 Tool Schema。连续6步无实质进展，或执行达到30步仍未结束时，Harness 删除普通探索上下文并切换候选专用 System Prompt，随后按 C1→C4 固定顺序自动逐个 `reopen`，不再展示候选列表，也不让模型调用 `open_product` 选择候选。每个新候选的上下文严格重建为“候选专用 System Prompt＋原始用户需求＋当前候选完整详情”，上一个候选不会残留；同一候选从选择规格到终局保持连续。规格未闭合时仅暴露 `select_option/back_to_search`，规格闭合后仅暴露 `buy_now/back_to_search`。候选阶段只判断 Hard：全部 Hard 满足必须购买，Hard 失败才可返回并自动进入下一候选，Soft 不作为拒绝理由。候选全部拒绝后，6步触发记为Loop，30步触发记为Max Steps。
- 本报告保留既有Harness v1/v2结果，并将GRPO230·Harness v3 r4与Qwen3.8-27B·Harness v3 r1作为独立新组接入，不覆盖历史结果。

### 旧候选列表版冻结评测：34条无进展轨迹的候选收敛结果

- 冻结轨迹中，`34`题达到连续6步无实质进展阈值，全部被候选收敛状态机接管（`100.00%`）。这表示轨迹进入了原Loop终止点的替代决策流程，不表示它们原本必然都会以Loop终止。
- 候选收敛后`11/34`最终完成正确购买，候选阶段正确购买率为`32.35%`：Gold `8`题、Valid `3`题。
- 其余终局为Wrong `15`题、Partial `6`题、Guard rejection `2`题。它们虽然未在阈值处直接判Loop，但也不能解释为“被成功挽救”。整轮最终仅Task 788为Repeat Loop，且该题未进入候选收敛阶段。以上是旧候选列表版的冻结轨迹，不代表当前顺序候选实现。

### 当前顺序候选实现实跑结果

- r4：`33`题进入候选阶段，Gold `4`、Valid `2`，正式转化 `6/33=18.18%`；另有Partial `4`、Wrong `15`、Repeat Loop `5`、Max Steps `3`。
- r5：`38`题进入候选阶段，Gold `6`、Valid `1`，正式转化 `7/38=18.42%`；另有Partial `7`、Wrong `12`、Repeat Loop `8`、Guard rejection `2`、Max Steps `2`。
- 若只按候选 Prompt 的 Hard-only 决策目标，将Partial也视为Hard通过购买，则r4为 `10/33=30.30%`，r5为 `14/38=36.84%`。正式正确率仍只计算Gold+Valid。

## Reward版本演进

- **Legacy baseline（口头可简称Reward v1）：** 仓库没有正式命名的 `reward_v1.py`；这里指ShopSimulator原生Reward及早期训练适配口径。原生Loose Additive Reward为 `r_type × (匹配属性数 + 匹配规格数 + 价格是否满足) / (属性总数 + 规格总数 + 1)`；适配层同时计算 `strict = r_type × r_attribute × r_option × r_price`，再组合为 `semantic = full_success + 0.5 × strict + 0.2 × native_reward`，并附加少量步数、过长、未完成与重复动作惩罚。该口径保留了类目、属性、规格、价格四维信号，但Loose可能出现维度补偿，Strict又会在任一维度失败时乘法坍缩，且对唯一Gold、错误购买、循环、超时和主动停止的终局区分不足。
- **v3 → v4：** 将v3中类目与价格等固定 Hard Gate、活跃维度匹配分数，升级为基于用户公开 Query 的可审计 Hard/Soft 约束合同。品类始终为 Hard；“必须、一定、绝对不要、不超过、至少、明确区间”等高置信且可确定性核验的不可妥协要求也进入 Hard；“最好、优先、尽量、大约、左右、预算”等偏好或近似表达进入 Soft；无所谓类表达忽略，复杂歧义语义进入 Needs Review / audit-only，不强行参与评分。任一可评分 Hard 失败即判 `wrong_purchase`；Hard 全通过后，目标商品为 Gold，完全满足 Soft 的替代商品为 Valid，只违反 Soft 的替代商品为 Partial。
- v4 新增第16步起的分段递增步数惩罚；将 `assistant_final` 与连续 Guard 拒绝由无效样本改为 `-0.8` 的有效负样本；并重新校准部分终局分数，其中 Partial 调整为 `0.5 + 0.3 × soft_score`，Loop 调整为 `-0.6`。
- **训练版本：** GRPO100使用Reward v3，GRPO230使用Reward v4。为保证横向可比，本报告中的成功率、Reward值与Reward类型仍统一按当前审计版Reward v4对冻结轨迹离线重放，不反向更新模型参数。

### Reward v3.2历史评分

| 终局 | 基础分 | 判定 |
|---|---|---|
| gold_purchase | 1.00 | 命中目标 ASIN，且类目、价格固定 Hard Gate 通过 |
| valid_alternative_purchase | 0.80 | 替代商品通过类目、价格门，且旧版四个匹配维度全部满足 |
| partial_alternative_purchase | -0.30 + 0.50 × S | 通过类目、价格门后，按品牌、型号、核心功能、关键规格四维匹配率 S 连续给分；不是 Hard 失败后的补分 |
| graceful_stop | -0.15 | 历史版本中充分检索后的合理停止 |
| early_abstain | -0.35 | 过早停止 |
| max_steps | -0.50 | 耗尽最大步数 |
| repeat_loop | -0.65 | 重复或无进展循环 |
| assistant_final | -0.40（训练过滤） | 记录了终局分，但当时按采样无效/优化过滤处理 |
| guard_rejection | 无效 | 连续非法动作终止未作为有效负样本参与优化 |
| wrong_purchase | -0.85 | 类目或价格固定 Hard Gate 失败 |
| reward_unverifiable | 0.00（无效） | 关键证据无法核验，不进入训练更新 |

v3没有累计步数惩罚。`assistant_final`虽记录为-0.40，但在当时训练筛选中被过滤；Guard拒绝同样不作为有效负样本优化。

### Reward v4当前评分

| 终局 | 基础分 | 判定 |
|---|---|---|
| gold_purchase | 1.00 | 所有可评分 Hard 与 Soft 全部通过，且命中目标 ASIN |
| valid_alternative_purchase | 0.80 | 所有可评分 Hard 与 Soft 全部通过，且为替代 ASIN |
| partial_alternative_purchase | 0.50 + 0.30 × soft_score | Hard 全通过，但目标或替代商品至少一个 Soft 失败；Soft 不可核验则为 Reward Unverifiable |
| early_abstain | -0.40 | 仍有探索价值时主动停止 |
| max_steps | 0.00 | 耗尽45步的基础分；随后叠加累计步数惩罚 |
| repeat_loop | -0.60 | 重复或无进展循环 |
| assistant_final | -0.80 | 未调用合法终局工具而直接输出文字 |
| guard_rejection | -0.80 | 连续非法动作达到 Guard 终止条件 |
| wrong_purchase | -1.00 | 任一可评分 Hard 失败 |
| reward_unverifiable | 0.00（无效） | Hard/Soft 证据或冻结合同无法核验，不进入训练更新 |

第16步起累计扣分：16–20步每步-0.01，21–25步每步-0.02，26–30步每步-0.03，31–35步每步-0.04，36–40步每步-0.05，41–45步每步-0.06；45步累计为-1.05。惩罚只改Final Reward，不改变终局类型。

## 总体结果

| 指标 | Base v1 | 普通 SFT v1 | 纠错 SFT v1 | GRPO100 v1 | GRPO230 v2 | Qwen3.8-27B v2 | GRPO230 v3 | Qwen3.8-27B v3 |
|---|---|---|---|---|---|---|---|---|
| 严格 Gold | 0/240 (0.00%) | 119/240 (49.58%) | 137/240 (57.08%) | 150/240 (62.50%) | 160/240 (66.67%) | 145/240 (60.42%) | 169/240 (70.42%) | 161/240 (67.08%) |
| 购买成功 | 0/240 (0.00%) | 131/240 (54.58%) | 155/240 (64.58%) | 166/240 (69.17%) | 174/240 (72.50%) | 164/240 (68.33%) | 185/240 (77.08%) | 175/240 (72.92%) |
| Reward有效 | 240/240 (100.00%) | 240/240 (100.00%) | 240/240 (100.00%) | 240/240 (100.00%) | 240/240 (100.00%) | 240/240 (100.00%) | 240/240 (100.00%) | 240/240 (100.00%) |
| 平均Final Reward | -0.7766 | 0.1742 | 0.3033 | 0.3945 | 0.4906 | 0.4064 | 0.5768 | 0.5312 |
| 平均Weighted Score | 0.0035 | 0.6566 | 0.7323 | 0.7728 | 0.8027 | 0.8534 | 0.9079 | 0.8725 |

## Reward类型

| Reward类型 | Base v1 | 普通 SFT v1 | 纠错 SFT v1 | GRPO100 v1 | GRPO230 v2 | Qwen3.8-27B v2 | GRPO230 v3 | Qwen3.8-27B v3 |
|---|---|---|---|---|---|---|---|---|
| gold_purchase | 0 | 119 | 137 | 150 | 160 | 145 | 169 | 161 |
| valid_alternative_purchase | 0 | 12 | 18 | 16 | 14 | 19 | 16 | 14 |
| partial_alternative_purchase | 0 | 10 | 9 | 5 | 5 | 6 | 13 | 9 |
| wrong_purchase | 1 | 25 | 19 | 22 | 21 | 59 | 33 | 40 |
| assistant_final | 5 | 0 | 3 | 6 | 0 | 0 | 0 | 0 |
| guard_rejection | 188 | 1 | 7 | 3 | 3 | 0 | 4 | 0 |
| max_steps | 1 | 3 | 13 | 12 | 3 | 0 | 4 | 0 |
| repeat_loop | 45 | 70 | 34 | 26 | 34 | 9 | 1 | 8 |
| early_abstain | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 8 |
| reward_unverifiable | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Rubric总体状态

| 状态 | Base v1 | 普通 SFT v1 | 纠错 SFT v1 | GRPO100 v1 | GRPO230 v2 | Qwen3.8-27B v2 | GRPO230 v3 | Qwen3.8-27B v3 |
|---|---|---|---|---|---|---|---|---|
| satisfied | 479 | 1332 | 1454 | 1529 | 1570 | 1454 | 1611 | 1555 |
| violated | 21 | 82 | 56 | 48 | 33 | 77 | 72 | 60 |
| unknown | 1254 | 347 | 252 | 186 | 157 | 230 | 79 | 147 |
| not_applicable | 5 | 8 | 7 | 6 | 9 | 8 | 7 | 7 |

## LLM Judge五维评分

评分规则：0分表示关键行为缺失或明显不合理；1分表示部分做到，但覆盖、证据或效率仍有不足；2分表示该维度完成充分且无明显问题。搜索策略衡量检索覆盖、有效改写与机械重复；候选利用衡量高匹配候选的利用、比较与收敛；证据核验衡量购买前对关键属性、规格和最终价格的检查；决策质量衡量商品、规格及购买/放弃决策；终止效率衡量是否过早购买/放弃、无效探索或耗尽步骤。五维独立评分，不加权、不计算总分。

| 维度 | Base v1 | 普通 SFT v1 | 纠错 SFT v1 | GRPO100 v1 | GRPO230 v2 | Qwen3.8-27B v2 | GRPO230 v3 | Qwen3.8-27B v3 |
|---|---|---|---|---|---|---|---|---|
| 搜索策略 | 0.946 | 1.292 | 1.529 | 1.587 | 1.629 | 1.517 | 1.642 | 1.667 |
| 候选利用 | 0.611 | 1.225 | 1.521 | 1.567 | 1.629 | 1.421 | 1.629 | 1.583 |
| 证据核验 | 0.272 | 0.992 | 1.304 | 1.429 | 1.617 | 1.188 | 1.650 | 1.504 |
| 决策质量 | 0.042 | 1.067 | 1.458 | 1.550 | 1.571 | 1.396 | 1.583 | 1.550 |
| 终止效率 | 0.017 | 1.029 | 1.325 | 1.408 | 1.500 | 1.446 | 1.500 | 1.587 |

## 行为、Token、耗时和上下文

| 指标 | Base v1 | 普通 SFT v1 | 纠错 SFT v1 | GRPO100 v1 | GRPO230 v2 | Qwen3.8-27B v2 | GRPO230 v3 | Qwen3.8-27B v3 |
|---|---|---|---|---|---|---|---|---|
| 平均执行工具调用数 | 5.904 | 16.446 | 16.908 | 15.738 | 10.021 | 5.383 | 10.142 | 5.529 |
| 平均动作尝试数 | 9.942 | 16.517 | 17.250 | 15.908 | 10.125 | 5.550 | 10.467 | 5.575 |
| Guard拒绝次数 | 969 | 17 | 82 | 41 | 25 | 40 | 78 | 11 |
| 重复动作次数 | 664 | 1,455 | 1,518 | 1,238 | 519 | 64 | 535 | 88 |
| 重复搜索次数 | 71 | 69 | 49 | 26 | 45 | 0 | 44 | 3 |
| 上下文使用率p50 | 33.2% | 57.4% | 66.2% | 68.7% | 25.3% | 18.6% | 24.3% | 20.3% |
| 上下文使用率p95 | 99.6% | 100.1% | 110.1% | 111.0% | 95.9% | 41.8% | 99.2% | 49.8% |
| Provider Token p50 | 34,378 | 72,315 | 66,735 | 68,540 | 30,388 | 18,453 | 23,502 | 19,940 |
| Provider Token p95 | 187,416 | 469,629 | 651,025 | 652,094 | 457,577 | 88,313 | 503,026 | 81,906 |
| 端到端耗时p50(s) | 9.5 | 25.8 | 31.4 | 31.8 | 17.6 | 75.0 | 15.9 | 93.2 |
| 端到端耗时p95(s) | 30.7 | 92.1 | 122.1 | 101.9 | 101.2 | 196.3 | 101.2 | 231.0 |
| Observation投影压缩任务 | 240 | 240 | 239 | 240 | 240 | 240 | 1 | 2 |
| 上下文硬溢出任务 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 基础设施无效任务 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## 工具调用次数

逐轨迹汇总实际执行次数；`0` 表示该工具在对应 Tool Schema 中存在但没有被调用，`未使用` 表示当前 8 工具 Schema 已不再暴露该工具。

| 工具 | Base v1 | 普通 SFT v1 | 纠错 SFT v1 | GRPO100 v1 | GRPO230 v2 | Qwen3.8-27B v2 | GRPO230 v3 | Qwen3.8-27B v3 |
|---|---|---|---|---|---|---|---|---|
| `search_products` | 435 | 731 | 713 | 656 | 648 | 287 | 680 | 255 |
| `open_product` | 381 | 668 | 680 | 632 | 581 | 280 | 645 | 300 |
| `select_option` | 27 | 402 | 409 | 421 | 546 | 419 | 401 | 420 |
| `back_to_search` | 325 | 499 | 481 | 421 | 412 | 47 | 445 | 15 |
| `prev_page` | 28 | 732 | 770 | 673 | 1 | 1 | 11 | 62 |
| `next_page` | 6 | 0 | 2 | 11 | 17 | 27 | 21 | 43 |
| `buy_now` | 1 | 166 | 183 | 193 | 200 | 229 | 231 | 224 |
| `finish_without_purchase` | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 8 |
| `view_description` | 143 | 354 | 355 | 312 | 未使用 | 未使用 | 未使用 | 未使用 |
| `view_features` | 67 | 300 | 399 | 375 | 未使用 | 未使用 | 未使用 | 未使用 |
| `view_reviews` | 0 | 95 | 66 | 83 | 未使用 | 未使用 | 未使用 | 未使用 |
| `view_attributes` | 0 | 0 | 0 | 0 | 未使用 | 未使用 | 未使用 | 未使用 |
| **全部工具调用合计** | 1413 | 3947 | 4058 | 3777 | 2405 | 1292 | 2434 | 1327 |

注：Base 历史轨迹另有 4 次已废弃的内部 `think` 调用；它不属于上述 12 个标准购物工具，因此未计入表格。

## 关键阶段迁移

| 阶段 | 成功迁移（Gold + Valid） |
|---|---|
| 纠错 SFT v1 → GRPO100 v1 | 失败→成功 34；成功→失败 23；共同成功 132；共同失败 51 |
| GRPO100 v1 → GRPO230 v2 | 失败→成功 34；成功→失败 26；共同成功 140；共同失败 40 |
| GRPO230 v2 → GRPO230 v3 | 失败→成功 33；成功→失败 22；共同成功 152；共同失败 33 |

## 八类典型轨迹

优先从GRPO230 v3和Qwen3.8-27B v3选择；缺失类型回退到纠错 SFT v1。

| Reward类型 | 模型 | Task | 步骤 | Final Reward | 动作链 |
|---|---|---|---|---|---|
| gold_purchase | GRPO230 v3 | 1456 | 4 | 1.0000 | search_products(圆形树池座凳 玻璃钢 户外) → open_product(913175640650) → select_option(opt_90e077f8fd15f4c0) → buy_now |
| valid_alternative_purchase | GRPO230 v3 | 1249 | 4 | 0.8000 | search_products(炼奶 小包装 180克) → open_product(901038200378) → select_option(opt_acb915b100705681) → buy_now |
| partial_alternative_purchase | GRPO230 v3 | 132 | 5 | 0.5000 | search_products(半透花卉胶带 手帐 10.5m) → open_product(855927020609) → select_option(opt_d5c3e2c391a47625) → select_option(opt_e21e1bea95c29e4a) → buy_now |
| wrong_purchase | GRPO230 v3 | 702 | 18 | -1.0300 | search_products(非洲翠平安扣 宽圆形) → open_product(944966300026) → back_to_search → search_products(非洲翠平安扣 冰种飘花 宽圆) → open_product(899432068512) → back_to_search → … → open_product(944966300026) → select_option(opt_b1ce7ea36af2d2f5) → buy_now |
| assistant_final | 纠错 SFT v1 | 936 | 21 | -0.8700 | search_products(角蛋白护发精华喷雾) → open_product(859958225406) → view_features → prev_page → view_description → prev_page → … → view_description → prev_page → select_option(200ml) |
| guard_rejection | GRPO230 v3 | 87 | 25 | -0.9500 | search_products(冰丝睡衣 薄款 粉色) → next_page → back_to_search → search_products(冰丝睡衣 粉色 薄款) → open_product(942972559344) → back_to_search → … → next_page → open_product(942972559344) → select_option(opt_55d44255d3cc6da2) |
| repeat_loop | GRPO230 v3 | 788 | 45 | -1.6500 | search_products(机械鼠标 静音 定时) → open_product(841526224701) → select_option(opt_7841f5f6b2bc0685) → back_to_search → search_products(机械鼠标点击器 静音 定时) → open_product(836067340537) → … → open_product(937044814994) → back_to_search → search_products(静音按键定时游戏电脑代替手指物理点击器 静音 定时 游戏挂机) |
| early_abstain | Qwen3.8-27B v3 | 843 | 8 | -0.4000 | search_products(婴儿车 高景观 一键坐躺 座椅旋转 折叠拖行 黑色 舞狮图案 出行礼包) → open_product(915650398163) → back_to_search → search_products(婴儿车 黑色 舞狮 传统元素 高景观 座椅旋转 一键坐躺 折叠拖行 出行礼包) → next_page → open_product(822004094120) → select_option(opt_373df7866b13409b) → finish_without_purchase({"reason": "no_suitable_product"}) |

## 文件

- `dashboard.html`：八组聚合前端报告。
- `per-task-comparison.csv/json`：八组模型逐题审计。
- `comparison.json`：八组两两配对及分层比较；页面展示三段关键迁移。
- `judges-*.jsonl`：各组Judge原始结构化结果。
