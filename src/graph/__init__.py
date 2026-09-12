"""LangGraph 编排层 —— 用状态图描述调度流程。

    thread_graph  一个帖子的生命周期：发帖 → 3 Phase → 撮合结算
    ingest_graph  外部政策接入：TTL 检查 →（命中读缓存 / 过期爬取）→ chunks
"""

from .ingest_graph import (
    IngestState,
    build_ingest_graph,
    policies_to_documents,
    route_on_cache,
)
from .state import ThreadState, make_initial_state
from .thread_graph import (
    PHASE,
    PUBLISH,
    SETTLE,
    build_thread_graph,
    route_after_phase,
    run_thread_graph,
)

__all__ = [
    # thread_graph
    "ThreadState",
    "make_initial_state",
    "build_thread_graph",
    "run_thread_graph",
    "route_after_phase",
    "PUBLISH",
    "SETTLE",
    "PHASE",
    # ingest_graph
    "IngestState",
    "build_ingest_graph",
    "route_on_cache",
    "policies_to_documents",
]
