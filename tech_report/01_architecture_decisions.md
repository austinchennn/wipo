# 架构决策记录 (ADR) — WIPO 金融舆情论坛沙盘

> 记录关键设计决策、讨论过程和未解决问题。  
> 格式：问题 → 决策 → 原因 → 待办/待确认

---

## 一、Agent `comment()` 方法设计

### 决策

**人格属性 → LLM：用自然语言描述，不传原始数值。**

```
risk_tolerance=0.85  →  "你是一个激进的投资者，对高风险高回报的机会极度感兴趣"
expressiveness=0.2   →  "你比较内敛，只有在被强烈触动时才会发言"
```

`build_persona_text()` 函数负责转换，结果缓存在 `agent._persona_text` 避免重复生成。

**原因：** 节省 token，LLM 理解自然语言远比理解抽象 [0,1] 数值准确，且避免 LLM 对数字做过度字面解读。

### temp 的唯一语义：发言意愿

- `temp` **只决定** Agent 是否开口（`temp >= comment_threshold` → 发言）
- `temp` 与情感极性（bull/bear/neutral）**完全无关**
- 情感极性通过独立的 `sentiment` 字段返回（LLM 自行判断）
- 例：一个强烈看空的机构投资者可以有 `temp=0.95`（非常想说）且 `sentiment="bear"`

**Prompt 设计：** 系统 prompt 中明确告知 LLM 该 Agent 的 `comment_threshold`，要求 LLM 的 temp 返回值与人设保持一致（内敛的人不能返回 0.9）。

### Phase 2/3 回复目标选择

- **旧设计（已废弃）：** 基于 `contrarian_tendency` 和 `fomo_susceptibility` 选 temp 最高/最低的评论
- **新设计：** 把候选评论列表（含 ID）传给 LLM，LLM 自己选一条回复（`reply_to_id`），只能选一条
- 如果 LLM 返回无效 ID 或 `null` → 该 Agent 本 Phase **跳过**（不强制随机兜底）

**原因：** 基于内容语义的选择比机械规则更真实，且契合论坛实际行为。

### 返回值结构

```python
@dataclass
class CommentResult:
    comment: str
    temp: float               # 发言意愿 [0,1]
    sentiment: Sentiment      # "bull" | "bear" | "neutral"
    reply_to_id: Optional[str]  # Phase 2/3 回复目标 ID
```

---

## 二、Spider 数据流设计

### 决策

**Spider 抓取的外部数据最终进入 RAG 知识库，不修改 ForumModel 或 Agent 属性。**

```
Spider → cleaner → classifier → DB（含时间戳）
                                     ↓ TTL 检查
                              有效数据（< 1周）→ 直接加载
                              过期数据（> 1周）→ 重新爬取
                                     ↓
                        chunks → RAG 知识库（与 PDF chunks 合并）
```

- Spider 只在**模拟启动前运行一次**
- 缓存 TTL：**1 周**，超期自动重爬
- ForumModel 不感知数据来源（PDF 还是 Spider），通过 RAGSystem 透明访问

### 已废弃的原始设计

- `global_factors.py` — 原本打算把政策文本转为数值因子修改 Agent 属性
- `perception.py` — 原本打算对 Agent 个性化加权
- **现状：** 这两个文件保留为空 stub，不实现。LLM 通过 RAG context 自然处理"感知"差异。

---

## 三、Agent 规模与 Active/Passive 分层

### 决策

**Active/Passive 两层设计，规模由用户在前端配置。**

| 层级 | 行为 | LLM 调用 |
|---|---|---|
| Active Agent | 真实评论，出现在帖子里 | 是（async 并发） |
| Passive Agent | 不产生可见评论 | 否（情绪统计推算） |

**成本参考（gpt-4o-mini）：**

| Active 数量 | 单 Phase 成本 | 完整模拟（12轮）|
|---|---|---|
| 1,000 | $0.24 | ~$26 |
| 10,000 | $2.4 | ~$260 |
| 100,000 | $24 | ~$2,600 |

**推荐：** 自用实验用 500~1,000 Active；企业用户可按需扩展至 100,000+。

### Passive Agent 情绪推算逻辑

Phase 结束后，`passive_inference.py` 根据以下规则推算 Passive Agent 情绪：

1. 同类型 Active Agent 的情绪分布作为先验
2. Passive Agent 自身 `risk_tolerance`（高 → bull 概率↑）、`contrarian_tendency`（高 → 反转主流）、`fomo_susceptibility`（高 → 跟随主流）做个性化偏移
3. 加少量随机噪声（±5%）

### 并发执行

- Active agents 使用 `asyncio.gather()` 并发调用 `acomment()`
- `asyncio.Semaphore(max_concurrent=50)` 限制最大并发数，防止 rate limit
- 默认并发数：50（可通过 `--concurrent` 参数调整）

### SentimentGrid 输出

每个 Phase 结束生成一个 `SentimentGrid`：
```python
{agent_id: "bull" | "bear" | "neutral"}  # 覆盖全部 Active + Passive
```
供前端渲染 100×100 情绪矩阵。

---

## 四、RAG 知识库设计

### 三个知识库对应信息可见性

| 知识库 | 对应 topic | 数据来源 |
|---|---|---|
| product_kb | 产品/业务 | PDF 产品章节 chunks + 行业 Spider chunks |
| financial_kb | 财务 | PDF 财务/MD&A chunks + 宏观 Spider chunks |
| risk_kb | 政策/风险 | PDF 风险因素 chunks + 政策 Spider chunks |

### Agent 访问控制

| Agent 类型 | product_kb | financial_kb | risk_kb |
|---|---|---|---|
| HostAgent | FULL | FULL | FULL |
| NormalAgent | SUMMARY（极简） | HIDDEN | HIDDEN |
| InstTraderAgent | FULL | FULL | FULL |
| RetailTraderAgent | MASKED（数字模糊）| MASKED | MASKED |

---

## 五、待确认问题

### 5.1 股价/市场模拟

**状态：已完成** → 见 [03_market_exchange.md](03_market_exchange.md)

### 5.2 数据持久化

**状态：已完成** → 见 [04_data_persistence.md](04_data_persistence.md)

- SQLite WAL 模式，模拟结束后一次性写入
- 每条评论带 `created_at` 时间戳，支持按时间顺序回放
- 回放为模拟结束后离线查询，过程中不写磁盘

### 5.3 前端 ↔ 后端连接

**状态：已完成** → 见 [05_frontend_backend.md](05_frontend_backend.md)

- FastAPI + WebSocket 实时推送架构
- REST API：模拟启动、状态查询、历史回放、K线/持仓/情绪数据
- WebSocket `/ws/feed`：模拟过程中实时推送所有事件（评论/情绪/K线/系统状态）
- React 前端通过 `useSimulation` hook 管理 WebSocket 生命周期
- Vite dev proxy 转发 `/api` 和 `/ws` 到后端

---

## 六、系统参数速查

```bash
# 基础用法
python run.py

# 完整参数
python run.py \
  --pdf prospectus.pdf \     # 招股书 PDF
  --normal 700 \             # 普通人总数
  --inst 100 \               # 机构总数
  --retail 200 \             # 散户总数
  --active 500 \             # Active Agent 数量（其余为 Passive）
  --model gpt-4o-mini \      # LLM 模型
  --concurrent 50 \          # 并发上限
  --no-rag \                 # 禁用 FAISS（纯静态模式）
  --seed 42                  # 随机种子
```
