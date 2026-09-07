# Termination v3.1

Termination v3.1修复v3中的两个循环判定边界问题，终止阈值本身不变。

## 修复一：少量新ASIN也属于运行进展

旧逻辑要求一个结果集至少出现3个新ASIN，才会刷新运行进展。这会把确实发现
1～2个新商品的搜索误计为无进展。

v3.1将两个概念分离：

```text
发现至少1个新ASIN
→ 重置no_progress_steps，保证正常探索继续

发现至少3个新ASIN
→ 才增加effective_result_sets，计入充分探索资格
```

因此少量新商品不会被误杀，也不能被用来刷高`finish_without_purchase`资格。

## 修复二：循环终止增加子原因

顶层终止原因保持兼容：

```text
repeat_loop
max_steps
```

同时新增：

```text
exact_action_repeat  连续第三次执行完全相同动作
no_progress_loop     连续6步没有新增运行证据
max_steps            完成第45个环境动作后终止
```

评测汇总可以分别统计动作复读和一般无进展，不再把两种问题混成一个数字。

## 当前终止与提醒边界

- 精确重复限制仍为2次累计，即第三次相同动作终止；
- 连续6步没有新候选、新商品证据、新规格状态或其他实质进展时终止；
- 最大环境步骤仍为45；
- Evaluation 在连续3步无进展时先显示循环高风险提醒；35步后改用步数收敛提醒，避免两种文案叠加；
- 第35步后显示步数提醒，第40步后进一步强调立即完成规格、购买或合理结束；
- 这些 Prompt/Harness 提醒不改变 Environment 的6步Loop终止和45步上限；
- `repeat_loop`当前基础分为`-0.60`；`max_steps`基础分为`0.00`，但45步会累计
  `-1.06`步数惩罚。

## 源码入口

- 实现：`environments/ShopSimulator/shop_env/web_agent_site/engine/termination.py`
- 环境传递：`environments/ShopSimulator/shop_env/web_agent_site/envs/web_agent_text_env.py`
- 评测指标：`src/shopping_grpo/evaluation/metrics.py`
- 汇总：`src/shopping_grpo/evaluation/summary.py`
- 测试：`environments/ShopSimulator/shop_env/tests/test_termination.py`
