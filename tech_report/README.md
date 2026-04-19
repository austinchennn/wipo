# Tech Report — WIPO 金融舆情论坛沙盘

技术文档索引。记录架构决策、实现细节、未解决问题和后续规划。

| 文件 | 内容 |
|---|---|
| [01_architecture_decisions.md](01_architecture_decisions.md) | 核心架构决策：Agent 设计、Spider 数据流、规模分层、RAG 访问控制、待确认问题 |
| [02_performance_refactoring.md](02_performance_refactoring.md) | 性能重构：ForumModel 异步化、LLMExtractor 基类、并行 PDF 提取、统一配置与 logging |
| [03_market_exchange.md](03_market_exchange.md) | 交易系统：限价订单簿连续竞价撮合引擎、Agent 分层交易决策、量价逻辑、持仓管理 |
| [04_data_persistence.md](04_data_persistence.md) | 数据持久化：SQLite 存储、时间戳评论回放、全量数据保存与查询接口 |
| [05_frontend_backend.md](05_frontend_backend.md) | 前端↔后端连接：WebSocket 实时推送、REST API、Broadcaster、React Hook |

---

> 新增文档命名规范：`NN_topic_name.md`，NN 为两位数字序号。
