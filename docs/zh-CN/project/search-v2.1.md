# Search v2.1

当前标识：`shopsimulator-multifield-bm25-v2.1`。

Search v2.1 使用 SQLite FTS5 多字段 BM25，固定 Top K 为 150、每页 20 个商品。同一次
Query 的结果保存在 session 中，翻页只切片，不重新检索。

相对v2，本版修正了FTS5首列`asin UNINDEXED`造成的字段权重错位，并将规格索引
限制为用户可见的规格轴、可购买规格值和价格，不再索引图片URL、variant ASIN、
可用状态等内部元数据。

## 字段权重

| 字段 | 权重 |
|---|---:|
| title | 3.0 |
| model | 2.5 |
| brand | 2.0 |
| category | 2.0 |
| attributes | 1.5 |
| options | 1.2 |
| bullets | 0.8 |

## 确定性合同

- Query 和商品字段使用同一套 Unicode、大小写、标点和中英文 token 归一化；
- 中文连续文本生成 bigram，字母、数字和型号单独保留；
- 同分时按 ASIN 升序；
- BM25权重显式包含ASIN占位，确保title、model等字段使用文档声明的权重；
- options只索引规格轴、可购买规格值和价格；
- 索引 manifest 冻结商品 SHA、字段权重、Tokenizer、排序规则和运行时版本；
- 索引缺失、损坏或商品哈希不一致时明确失败。

Search v2.1 仍是词法检索，不包含 Dense Retrieval、Cross Encoder 或学习排序。

## 非Final离线验证

使用`tag=eval`且未进入Final-240的1219道任务，以用户完整需求直接查询Gold商品：

| 指标 | Search v2 | Search v2.1 |
|---|---:|---:|
| Gold Recall@1 | 46.35% | 51.93% |
| Gold Recall@5 | 70.06% | 76.37% |
| Gold Recall@20 | 83.84% | 86.30% |
| Gold Recall@150 | 94.50% | 95.32% |

该数据只用于验证搜索实现修复，不使用Final-240结果调字段权重。

## 源码入口

- 实现：`environments/ShopSimulator/shop_env/web_agent_site/engine/search.py`
- 环境配置：`environments/ShopSimulator/shop_env/configs/environment.json`
- 索引由`environments/ShopSimulator/shop_env/scripts/build_index.py`根据冻结商品数据和
  Search合同构建；生成的本地SQLite与manifest不进入Git。
