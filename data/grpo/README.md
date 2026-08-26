# GRPO 数据

当前已准备训练probe候选池，但尚未生成正式train/validation Parquet。

- `dev-probe-v1.1/`：冻结开发probe，共100题。
- `training-probe-v1/calibration-200.jsonl`：第一阶段在线probe。
- `training-probe-v1/remaining-600.jsonl`：校准阈值后再决定是否启动的第二阶段。
- `training-probe-v1/data-gate-rejected.jsonl`：未通过离线数据门的隔离记录。

训练probe与SFT、Teacher、dev probe和冻结Final-240按task ID与商品family隔离。
在线门使用4条rollout、最多3次尝试；terminal utility范围不超过`0.025`时重采，
第三次仍无有效差异则停止采集并隔离该任务。
