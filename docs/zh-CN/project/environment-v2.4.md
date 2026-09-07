# Environment v2.4

Environment v2.4冻结当前可共同运行的组件组合：

```text
Environment v2.4
├── Search v2.1
├── Observation v2
├── Tool Schema v2
├── Reward Features v2
├── Query Constraints v1
├── Reward v4
└── Termination v3.1
```

## 变化

- Search升级到v2.1；
- 修正BM25字段权重与FTS5列位置错位；
- 规格索引只保留规格轴、可购买值和价格；
- 环境清单和运行文件哈希重新冻结。

## 未变化

- 工具名称和参数；
- Observation结构；
- Termination阈值；
- Reward奖励数值。

新运行必须通过`data/environment.json`中的版本、商品数据SHA和运行文件SHA校验，
不能只修改版本字符串。历史SFT和
Final-200结果仍按实际运行时版本保留，不回写为v2.4结果。
