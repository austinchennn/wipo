"""
api.main — FastAPI 入口

REST 接口：
  POST /api/simulate       — 启动模拟（异步，返回 task_id）
  GET  /api/status         — 当前模拟状态
  GET  /api/simulations    — 历史模拟列表
  GET  /api/replay/{id}    — 按时间戳回放评论流
  GET  /api/klines/{id}    — K 线序列
  GET  /api/portfolios/{id} — 持仓快照

WebSocket:
  WS /ws/feed              — 实时事件推送（评论/情绪/K线/系统状态）
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .broadcaster import broadcaster

logger = logging.getLogger(__name__)

app = FastAPI(title="WIPO 金融舆情论坛沙盘 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════
#  全局状态
# ═══════════════════════════════════════════════════

class SimState:
    """模拟运行时状态（单例）。"""
    def __init__(self):
        self.running: bool = False
        self.task: Optional[asyncio.Task] = None
        self.model = None
        self.error: Optional[str] = None

sim_state = SimState()


# ═══════════════════════════════════════════════════
#  请求 / 响应模型
# ═══════════════════════════════════════════════════

class SimulateRequest(BaseModel):
    n_normal: int = Field(20, ge=1, le=100_000)
    n_inst: int = Field(5, ge=1, le=10_000)
    n_retail: int = Field(15, ge=1, le=100_000)
    n_active: Optional[int] = Field(None, ge=1)
    pdf_path: Optional[str] = None
    use_rag: bool = True
    llm_model: str = "gemini-2.5-flash"
    max_concurrent: int = Field(50, ge=1, le=500)
    seed: Optional[int] = 42
    api_key: Optional[str] = None


class StatusResponse(BaseModel):
    running: bool
    round: int = 0
    total_rounds: int = 12
    total_posts: int = 0
    total_comments: int = 0
    ws_clients: int = 0
    error: Optional[str] = None


# ═══════════════════════════════════════════════════
#  WebSocket
# ═══════════════════════════════════════════════════

@app.websocket("/ws/feed")
async def ws_feed(ws: WebSocket):
    """实时事件流 —— 客户端连接后接收所有模拟事件。"""
    await broadcaster.connect(ws)
    try:
        while True:
            # 保持连接；客户端可发送 ping 或命令
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_text('{"type":"pong"}')
    except WebSocketDisconnect:
        pass
    finally:
        await broadcaster.disconnect(ws)


# ═══════════════════════════════════════════════════
#  REST 接口
# ═══════════════════════════════════════════════════

@app.post("/api/simulate")
async def start_simulation(req: SimulateRequest):
    """启动一次模拟（异步执行）。"""
    if sim_state.running:
        return {"ok": False, "error": "模拟正在运行中，请等待结束"}

    # 前端传入的 API Key → 写入环境变量（不持久化）
    if req.api_key:
        os.environ["GOOGLE_API_KEY"] = req.api_key

    sim_state.running = True
    sim_state.error = None

    async def run_sim():
        try:
            from ..environment.forum import ForumModel

            model = ForumModel(
                n_normal=req.n_normal,
                n_inst=req.n_inst,
                n_retail=req.n_retail,
                n_active=req.n_active,
                pdf_path=req.pdf_path,
                use_rag=req.use_rag,
                llm_model=req.llm_model,
                max_concurrent=req.max_concurrent,
                seed=req.seed,
            )
            model.on_event = broadcaster.broadcast
            sim_state.model = model

            await model.arun()
        except Exception as e:
            logger.exception("模拟运行失败")
            sim_state.error = str(e)
            await broadcaster.broadcast({
                "type": "system",
                "event": "sim_error",
                "error": str(e),
            })
        finally:
            sim_state.running = False

    sim_state.task = asyncio.create_task(run_sim())
    return {"ok": True, "message": "模拟已启动"}


@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """获取当前模拟状态。"""
    model = sim_state.model
    if model is None:
        return StatusResponse(
            running=sim_state.running,
            ws_clients=broadcaster.client_count,
            error=sim_state.error,
        )

    total_comments = sum(
        len(p.comments[ph]) for p in model.posts for ph in (1, 2, 3)
    )
    return StatusResponse(
        running=sim_state.running,
        round=model.round_num,
        total_rounds=model.MACRO_ROUNDS,
        total_posts=len(model.posts),
        total_comments=total_comments,
        ws_clients=broadcaster.client_count,
        error=sim_state.error,
    )


@app.get("/api/simulations")
async def list_simulations():
    """查询历史模拟记录。"""
    from ..persistence.database import SimulationDB
    db = SimulationDB()
    try:
        return db.list_simulations()
    finally:
        db.close()


@app.get("/api/replay/{simulation_id}")
async def replay_comments(
    simulation_id: int,
    post_id: Optional[str] = Query(None),
    phase: Optional[int] = Query(None, ge=1, le=3),
    limit: Optional[int] = Query(None, ge=1, le=10_000),
):
    """按时间戳排序回放评论流。"""
    from ..persistence.database import SimulationDB
    db = SimulationDB()
    try:
        return db.replay_comments(
            simulation_id=simulation_id,
            post_id=post_id,
            phase=phase,
            limit=limit,
        )
    finally:
        db.close()


@app.get("/api/klines/{simulation_id}")
async def get_klines(simulation_id: int):
    """获取 K 线序列。"""
    from ..persistence.database import SimulationDB
    db = SimulationDB()
    try:
        return db.get_klines(simulation_id)
    finally:
        db.close()


@app.get("/api/portfolios/{simulation_id}")
async def get_portfolios(simulation_id: int):
    """获取持仓快照。"""
    from ..persistence.database import SimulationDB
    db = SimulationDB()
    try:
        return db.get_portfolios(simulation_id)
    finally:
        db.close()


@app.get("/api/sentiments/{simulation_id}")
async def get_sentiments(simulation_id: int):
    """获取情绪快照序列。"""
    from ..persistence.database import SimulationDB
    db = SimulationDB()
    try:
        return db.get_sentiment_snapshots(simulation_id)
    finally:
        db.close()