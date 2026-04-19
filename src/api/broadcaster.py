"""
api.broadcaster — 异步事件广播器

ForumModel 产生事件 → Broadcaster 分发 → 所有 WebSocket 客户端

事件类型：
  system     — 系统日志（启动、轮次、结束）
  post       — Host 发帖
  comment    — Agent 评论（逐条推送）
  sentiment  — 情绪矩阵快照
  trade      — 交易 Session 撮合结果 + K 线
  phase      — Phase 状态切换
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, Set

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class Broadcaster:
    """管理所有 WebSocket 连接，广播模拟事件。"""

    def __init__(self):
        self._clients: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)
        logger.info("WebSocket 客户端已连接 (总数=%d)", len(self._clients))

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)
        logger.info("WebSocket 客户端已断开 (总数=%d)", len(self._clients))

    async def broadcast(self, event: Dict[str, Any]) -> None:
        """向所有客户端广播一个事件。"""
        if not self._clients:
            return

        event.setdefault("timestamp", datetime.now().isoformat())
        payload = json.dumps(event, ensure_ascii=False, default=str)

        async with self._lock:
            dead: list[WebSocket] = []
            for ws in self._clients:
                try:
                    await ws.send_text(payload)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self._clients.discard(ws)

    @property
    def client_count(self) -> int:
        return len(self._clients)


# 全局单例
broadcaster = Broadcaster()
