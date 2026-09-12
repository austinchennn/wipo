"""
scheduler —— 政策接入调度入口

按设计（见 Spider 数据流设计），政策数据**只在模拟启动前跑一次**，
不在 12 轮中途更新，所以这里不需要 APScheduler / Celery 那类常驻定时器：
「该不该重爬」完全由缓存 TTL 决定，判断本身放在 ingest_graph 的条件边上。

    ensure_fresh_policies()
        → 库里有 1 周内的数据 → 直接复用，不发任何请求
        → 没有 / 过期           → 爬取 → 清洗 → 分类 → 落库
        → 两条路径都产出 Document chunks

用法：
    chunks = ensure_fresh_policies()          # 同步
    chunks = await aensure_fresh_policies()   # 异步
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Iterable, List, Optional

from langchain_core.documents import Document

from ..config import POLICY_CACHE_TTL_DAYS
from ..persistence.policy_store import PolicyStore
from .policy_spider import PolicySource, PolicySpider

logger = logging.getLogger(__name__)


async def aensure_fresh_policies(
    ttl_days: int = POLICY_CACHE_TTL_DAYS,
    sources: Optional[Iterable[PolicySource]] = None,
    db_path: str | Path = "output/policies.db",
) -> List[Document]:
    """跑一遍政策接入图，返回可并入 RAG 的 Document chunks。

    任何一步失败都只记警告并返回空列表——外部数据是增强项，
    不应该让整个模拟起不来。
    """
    from ..graph.ingest_graph import build_ingest_graph, make_initial_state

    store = PolicyStore(db_path)
    try:
        graph = build_ingest_graph(store, PolicySpider(sources=sources))
        state = await graph.ainvoke(make_initial_state(ttl_days))
        logger.debug("[政策接入] 图执行路径: %s", " → ".join(state.get("trace", [])))
        return state.get("chunks", [])
    except Exception as e:
        logger.warning("[政策接入] 失败，本次模拟不接入外部政策: %s", e)
        return []
    finally:
        store.close()


def ensure_fresh_policies(
    ttl_days: int = POLICY_CACHE_TTL_DAYS,
    sources: Optional[Iterable[PolicySource]] = None,
    db_path: str | Path = "output/policies.db",
) -> List[Document]:
    """同步入口（脚本 / REPL 用）。"""
    return asyncio.run(
        aensure_fresh_policies(ttl_days=ttl_days, sources=sources, db_path=db_path)
    )
