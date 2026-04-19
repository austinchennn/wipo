# 04 — 数据持久化：SQLite 存储与评论回放

> 对应架构决策 5.2，解决模拟数据在进程结束后丢失的问题。

---

## 一、需求分析

| 需求 | 决定 |
|---|---|
| 持久化时机 | 模拟**结束后**一次性写入，过程中不写磁盘 |
| 回放模式 | **离线回放** —— 模拟结束后按 `created_at` 时间戳顺序查询评论流 |
| 存储格式 | **SQLite**（单文件、零配置、支持 SQL 查询、Python 标准库内置） |
| 数据完整性 | 事务保证原子写入；WAL 模式提升并发读性能 |

### 为什么选 SQLite 而不是 JSON？

- **回放查询**：`ORDER BY created_at` 在 SQLite 中是原生索引操作；JSON 需要全量加载 + 内存排序
- **分页/过滤**：`WHERE phase = 2 LIMIT 50` 在 SQL 中一行搞定；JSON 要手动 filter
- **数据规模**：12 轮 × 3 帖子 × 3 Phase × 数百评论 = 数千到数万条记录，SQLite 轻松应对
- **关联查询**：评论 ↔ 帖子 ↔ 情绪快照 ↔ 交易记录，SQL JOIN 天然支持
- **标准库内置**：`sqlite3` 是 Python 标准库，不增加任何依赖

---

## 二、Schema 设计

```
simulations ──┬── posts ──── comments (带 created_at 索引)
              ├── sentiments
              ├── trades
              ├── klines
              └── portfolios
```

### 核心表

| 表 | 主键 | 说明 |
|---|---|---|
| `simulations` | `id` (自增) | 模拟元数据：参数、起止时间 |
| `posts` | `(simulation_id, id)` | 帖子：轮次、主题、内容、创建时间 |
| `comments` | `(simulation_id, id)` | 评论：作者、内容、temp、sentiment、phase、parent_id、**created_at** |
| `sentiments` | `id` (自增) | 情绪快照：round_num/topic/phase + JSON grid |
| `trades` | `id` (自增) | 成交记录：价格、数量、买卖方 |
| `klines` | `id` (自增) | K 线：OHLCV + 事件标注 |
| `portfolios` | `(simulation_id, agent_id)` | 最终持仓：资金、股票 |

### 关键索引

```sql
CREATE INDEX idx_comments_created_at ON comments(simulation_id, created_at);
CREATE INDEX idx_comments_post ON comments(simulation_id, post_id, phase);
```

`idx_comments_created_at` 是回放的核心索引 —— 按时间排序查询不需要全表扫描。

---

## 三、数据流

```
ForumModel.arun()
    │
    │  Phase 1/2/3 → Comment 创建时写入 created_at = datetime.now()
    │  Post 创建时写入 created_at = datetime.now()
    │
    │  ……（12 轮 × 3 帖子 × 3 Phase 全部在内存中完成）……
    │
    ▼
模拟结束
    │
    ▼
SimulationDB.save(model)
    │
    ├─ BEGIN TRANSACTION
    ├─ INSERT simulations (元数据)
    ├─ INSERT posts × 36
    ├─ INSERT comments × N (数百~数千)
    ├─ INSERT sentiments × 108
    ├─ INSERT trades × M
    ├─ INSERT klines × 36
    ├─ INSERT portfolios × agent_count
    ├─ COMMIT
    │
    ▼
output/simulation.db （单文件，可拷贝/分享）
```

### 性能特征

- **零运行时开销**：模拟过程中不做任何 I/O
- **批量写入**：所有 INSERT 在单个事务中完成，WAL 模式下延迟 < 100ms
- **持久化失败不影响模拟**：try/except 包裹，失败只 warning 不崩溃

---

## 四、时间戳设计

### Comment.created_at

每条评论在创建时记录 `datetime.now()`：

```python
Comment(
    id=Comment.make_id(),
    author_id=agent.unique_id,
    ...
    sentiment=result.sentiment,
    created_at=datetime.now(),    # ← 新增
)
```

时间戳反映的是**真实生成时刻**（wallclock time），而不是模拟内时间。这意味着：
- 同一 Phase 内的评论时间差 ≈ LLM 并发响应时间（通常 < 1s）
- 不同 Phase 之间有明显时间间隔
- 不同帖子之间有更大的时间间隔

这恰好形成了天然的"节奏感"，回放时可以据此推断 Phase/Thread 边界。

### Post.created_at

帖子同样记录创建时间，供前端展示发帖时刻。

---

## 五、回放接口

### Python API

```python
from src.persistence import replay_comments

# 全量回放（按时间排序）
for c in replay_comments():
    print(f"[{c['created_at']}] {c['author_name']}: {c['content']}")

# 过滤特定帖子
replay_comments(post_id="R4_financial")

# 只看 Phase 2
replay_comments(phase=2)

# 前 50 条
replay_comments(limit=50)
```

### SimulationDB 完整查询接口

| 方法 | 说明 |
|---|---|
| `replay_comments(sim_id, post_id, phase, limit)` | 按 created_at 排序回放评论 |
| `get_simulation_summary(sim_id)` | 模拟概况（参数、统计） |
| `get_klines(sim_id)` | K 线序列 |
| `get_sentiment_snapshots(sim_id)` | 情绪快照序列（JSON grid 自动解析） |
| `get_portfolios(sim_id)` | 最终持仓 |
| `list_simulations()` | 列出所有模拟记录 |

---

## 六、模型变更

### Comment 数据类新增字段

```python
@dataclass
class Comment:
    ...
    sentiment: Optional[str] = None      # "bull" | "bear" | "neutral"
    ...
    created_at: Optional[datetime] = None # 评论生成时刻
    ...
```

- `sentiment`：原本只在 `CommentResult` 中，现在也存入 `Comment`，持久化后可直接查询
- `created_at`：回放排序依据
- 两个字段均为 `Optional`，向后兼容旧代码

### Post 数据类新增字段

```python
@dataclass
class Post:
    ...
    created_at: Optional[datetime] = None
    ...
```

---

## 七、文件变更清单

| 文件 | 操作 | 说明 |
|---|---|---|
| `src/models.py` | 修改 | Comment 添加 `sentiment` + `created_at`；Post 添加 `created_at`；引入 `datetime` |
| `src/persistence/__init__.py` | 新增 | 导出 `SimulationDB` 和 `replay_comments` |
| `src/persistence/database.py` | 新增 | SQLite 引擎：schema、save()、replay 查询、便捷函数 |
| `src/environment/forum.py` | 修改 | Comment/Post 创建时设置 `sentiment`/`created_at`；`arun()` 结束后调用 `SimulationDB.save()` |
| `tech_report/01_architecture_decisions.md` | 修改 | 5.2 状态更新为"已完成" |

---

## 八、与前端的对接预留

`mockData.js` 中的评论结构已有 `timestamp` 字段（字符串格式 `"14:33:12"`）。
SQLite 中的 `created_at` 是 ISO 8601 格式（`"2026-04-18T14:33:12.123456"`）。

后续实现 API 层（5.3）时：
1. `/api/replay?sim_id=1` 直接查询 SQLite
2. 返回 JSON 流，前端按 `created_at` 顺序逐条渲染
3. 可增加播放速度参数（1x / 2x / 5x）由前端控制间隔
