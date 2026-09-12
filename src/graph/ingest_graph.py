"""
政策接入编排图 —— LangGraph StateGraph

    START → check_cache ─┬─(TTL 命中)─→ load_cache ──────────────┐
                         │                                       ├→ to_chunks → END
                         └─(过期/空库)→ crawl → clean → classify → persist ─┘

和 thread_graph 同样的思路：把「缓存新鲜就别爬」这个决策放到图的条件边上，
而不是埋在调度函数的 if 里。爬取链路四个节点各司其职，任何一环换实现
（换爬虫、换分类器）都不影响其余节点。

产出是一批 Document chunks，由调用方并入 RAG 的 policy 知识库。
ForumModel 与 Agent 都不感知这些数据来自爬虫。
"""

from __future__ import annotations

import logging
import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

from ..config import (
    POLICY_CACHE_TTL_DAYS,
    POLICY_CHUNK_OVERLAP,
    POLICY_CHUNK_SIZE,
)
from ..policy_engine.classifier import aclassify_all
from ..ports.llm import LLMProvider
from ..ports.persistence import PolicyCache
from ..ports.spider import PolicyFeed
from ..spiders.models import CleanedPolicy, RawPolicy
from ..spiders.news_cleaner import clean_all
from ..spiders.policy_spider import PolicySpider

logger = logging.getLogger(__name__)

# 节点名常量
CHECK_CACHE = "check_cache"
LOAD_CACHE = "load_cache"
CRAWL = "crawl"
CLEAN = "clean"
CLASSIFY = "classify"
PERSIST = "persist"
TO_CHUNKS = "to_chunks"


class IngestState(TypedDict, total=False):
    """政策接入图的共享状态。"""

    ttl_days: int
    cache_hit: bool
    raws: List[RawPolicy]
    policies: List[CleanedPolicy]
    chunks: List[Document]
    trace: Annotated[List[str], operator.add]


def make_initial_state(ttl_days: int = POLICY_CACHE_TTL_DAYS) -> IngestState:
    return IngestState(
        ttl_days=ttl_days,
        cache_hit=False,
        raws=[],
        policies=[],
        chunks=[],
        trace=[],
    )


# ═══════════════════════════════════════════════════════
#  路由
# ═══════════════════════════════════════════════════════


def route_on_cache(state: IngestState) -> str:
    """缓存还新鲜就直接读库，否则走完整爬取链路。"""
    return LOAD_CACHE if state.get("cache_hit") else CRAWL


# ═══════════════════════════════════════════════════════
#  chunk 化（纯函数，可独立测试）
# ═══════════════════════════════════════════════════════


def policies_to_documents(
    policies: List[CleanedPolicy],
    chunk_size: int = POLICY_CHUNK_SIZE,
    chunk_overlap: int = POLICY_CHUNK_OVERLAP,
) -> List[Document]:
    """把政策正文切成带 metadata 的 Document，供 FAISS 索引。"""
    if not policies:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    docs: List[Document] = []
    for p in policies:
        for i, piece in enumerate(splitter.split_text(p.to_document_text())):
            docs.append(Document(
                page_content=piece,
                metadata={
                    "topic": "policy",
                    "origin": "spider",       # 区别于招股书 chunks
                    "source": p.source,
                    "url": p.url,
                    "title": p.title,
                    "labels": ",".join(p.labels),
                    "published_at": p.published_at or "",
                    "chunk": i,
                },
            ))
    return docs


# ═══════════════════════════════════════════════════════
#  构图
# ═══════════════════════════════════════════════════════


def build_ingest_graph(
    store: PolicyCache,
    spider: Optional[PolicyFeed] = None,
    llm: Optional[LLMProvider] = None,
):
    """编译一张政策接入图。

    三个依赖都只按 Protocol 约束（PolicyCache / PolicyFeed / LLMProvider），
    所以测试可以塞本地 fixture feed 和内存缓存，不碰网络也不碰 sqlite。
    """
    spider = spider if spider is not None else PolicySpider()

    async def check_cache(state: IngestState) -> Dict[str, Any]:
        ttl = state["ttl_days"]
        hit = store.is_fresh(ttl)
        last = store.latest_fetched_at()
        logger.info(
            "[政策接入] 缓存%s（TTL=%d天，最近抓取=%s，库中%d条）",
            "命中" if hit else "未命中", ttl,
            last.strftime("%Y-%m-%d %H:%M") if last else "无",
            store.count(),
        )
        return {"cache_hit": hit, "trace": [CHECK_CACHE]}

    async def load_cache(state: IngestState) -> Dict[str, Any]:
        policies = store.load_fresh(state["ttl_days"])
        logger.info("[政策接入] 复用缓存 %d 条，跳过爬取", len(policies))
        return {"policies": policies, "trace": [LOAD_CACHE]}

    async def crawl(state: IngestState) -> Dict[str, Any]:
        raws = await spider.afetch_all()
        return {"raws": raws, "trace": [CRAWL]}

    async def clean(state: IngestState) -> Dict[str, Any]:
        return {"policies": clean_all(state["raws"]), "trace": [CLEAN]}

    async def classify(state: IngestState) -> Dict[str, Any]:
        policies = await aclassify_all(state["policies"], llm=llm)
        return {"policies": policies, "trace": [CLASSIFY]}

    async def persist(state: IngestState) -> Dict[str, Any]:
        store.save(state["policies"])
        return {"trace": [PERSIST]}

    async def to_chunks(state: IngestState) -> Dict[str, Any]:
        chunks = policies_to_documents(state["policies"])
        logger.info(
            "[政策接入] %d 条政策 → %d 个 chunks",
            len(state["policies"]), len(chunks),
        )
        return {"chunks": chunks, "trace": [TO_CHUNKS]}

    graph = StateGraph(IngestState)
    for name, fn in (
        (CHECK_CACHE, check_cache), (LOAD_CACHE, load_cache), (CRAWL, crawl),
        (CLEAN, clean), (CLASSIFY, classify), (PERSIST, persist),
        (TO_CHUNKS, to_chunks),
    ):
        graph.add_node(name, fn)

    graph.add_edge(START, CHECK_CACHE)
    graph.add_conditional_edges(
        CHECK_CACHE, route_on_cache, {LOAD_CACHE: LOAD_CACHE, CRAWL: CRAWL}
    )
    graph.add_edge(CRAWL, CLEAN)
    graph.add_edge(CLEAN, CLASSIFY)
    graph.add_edge(CLASSIFY, PERSIST)
    graph.add_edge(PERSIST, TO_CHUNKS)
    graph.add_edge(LOAD_CACHE, TO_CHUNKS)
    graph.add_edge(TO_CHUNKS, END)

    return graph.compile()
