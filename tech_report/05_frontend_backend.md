# 05 — 前端 ↔ 后端连接（WebSocket 实时推送）

## 一、架构概览

```
┌────────────┐  WebSocket /ws/feed  ┌────────────────┐  on_event callback  ┌────────────┐
│  React 前端 │◄═══════════════════►│  FastAPI (api/) │◄════════════════════│ ForumModel │
│  (Vite dev) │  REST /api/*        │  + Broadcaster  │                     │ (模拟引擎)  │
└────────────┘                      └────────────────┘                      └────────────┘
```

- **实时通道**：WebSocket `/ws/feed`，服务端主动推送所有模拟事件
- **控制通道**：REST API 负责启动模拟、查询状态、回放历史数据
- **事件钩子**：`ForumModel.on_event` 异步回调，由 API 层注入 `broadcaster.broadcast`

## 二、WebSocket 协议

### 事件类型

| type | event | 触发时机 | 关键字段 |
|------|-------|---------|---------|
| `system` | `sim_start` | 模拟开始 | `n_normal`, `n_inst`, `n_retail`, `total_agents` |
| `system` | `round_start` | 每轮开始 | `round` |
| `system` | `sim_end` | 模拟结束 | `total_posts`, `final_price`, `price_change_pct` |
| `system` | `sim_error` | 异常 | `error` |
| `post` | — | 新帖发布 | `post_id`, `round`, `thread_title`, `host_agent` |
| `phase` | `phase_start` | Phase 开始 | `post_id`, `phase`, `phase_name` |
| `comment` | — | 评论/回复 | `post_id`, `phase`, `agent_name`, `agent_type`, `comment`, `sentiment`, `temp` |
| `sentiment` | — | 情绪网格完成 | `post_id`, `phase`, `bull`, `bear`, `neutral` |
| `trade` | — | 交易回合结束 | `post_id`, `tick`, `open`, `high`, `low`, `close`, `volume` |

### 心跳

客户端发送文本 `"ping"` → 服务端回复 `{"type":"pong"}`。前端每 30 秒发送一次。

## 三、REST API

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/simulate` | 启动模拟（参数：n_normal, n_inst, n_retail, llm_model 等） |
| `GET` | `/api/status` | 当前模拟状态（running, round, total_posts, ws_clients） |
| `GET` | `/api/simulations` | 历史模拟列表（按时间倒序） |
| `GET` | `/api/replay/{id}` | 按时间戳回放评论流 |
| `GET` | `/api/klines/{id}` | K 线数据序列 |
| `GET` | `/api/portfolios/{id}` | 持仓快照 |
| `GET` | `/api/sentiments/{id}` | 情绪快照序列 |

## 四、关键实现

### 4.1 Broadcaster（src/api/broadcaster.py）

全局单例，管理 WebSocket 客户端集合：
- `connect(ws)` / `disconnect(ws)` — 注册/注销客户端
- `broadcast(event: dict)` — JSON 序列化后广播给所有客户端，自动清理断开的连接
- 使用 `asyncio.Lock` 保证并发安全

### 4.2 ForumModel 事件钩子（src/environment/forum.py）

- `on_event: Optional[Callable[[Dict], Awaitable[None]]]` — 可选异步回调
- `_emit(event)` — 辅助方法，调用 `on_event` 并捕获异常（不中断模拟）
- 在 `arun()`、`_run_thread()`、`_phase_1()`、`_reply_phase()` 中插入 8+ 个事件发射点

### 4.3 前端 Hook（frontend/src/hooks/useSimulation.js）

- 自动连接 WebSocket，断开后 2 秒重连
- `handleMessage` 按 `type` 分发到对应 state（comments, posts, klines, sentiments, logs）
- `start(params)` 调用 `POST /api/simulate`

### 4.4 Vite 代理（frontend/vite.config.js）

```js
server: {
  proxy: {
    '/api': { target: 'http://127.0.0.1:8000' },
    '/ws':  { target: 'ws://127.0.0.1:8000', ws: true },
  }
}
```

## 五、文件变更清单

| 文件 | 变更 |
|------|------|
| `src/api/broadcaster.py` | 新建 — 异步事件广播器 |
| `src/api/main.py` | 重写 — FastAPI 完整实现（7 REST + 1 WS） |
| `src/environment/forum.py` | 修改 — 添加 on_event 回调 + _emit + 事件发射 |
| `frontend/src/hooks/useSimulation.js` | 新建 — WebSocket 管理 hook |
| `frontend/src/App.jsx` | 重写 — 模拟仪表盘 UI |
| `frontend/src/index.css` | 重写 — Tailwind 入口 |
| `frontend/src/App.css` | 清空 |
| `frontend/vite.config.js` | 修改 — 添加代理配置 |
| `requirements.txt` | 修改 — 添加 fastapi, uvicorn, websockets |

## 六、验证结果

```
# 启动后端
uvicorn src.api.main:app --port 8000

# 测试 REST
GET /api/status → {"running":false,"round":0,...}
POST /api/simulate → {"ok":true,"message":"模拟已启动"}

# 测试 WebSocket（模拟运行期间）
Event counts: {system: 14, post: 36, phase: 108, comment: 200, sentiment: 99, trade: 36}
Total events: 493
```
