"""
LangGraph 帖子线程图测试

全程注入 NullLLMProvider，即使本机 .env 里有 GOOGLE_API_KEY 也不会发请求。
不需要 monkeypatch 任何模块全局变量。
"""

from __future__ import annotations

import asyncio

import pytest

from src.composition import build_knowledge
from src.environment.forum import ForumModel
from src.llm import NullLLMProvider
from src.graph import PHASE, PUBLISH, SETTLE, route_after_phase, run_thread_graph


# ═══════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════


def make_model(**kwargs) -> ForumModel:
    """全离线装配：NullLLMProvider + 静态知识库 + 不落库。

    注入之前这里需要 monkeypatch 模块全局 _get_structured_llm，
    而且 ForumModel 一定会开 SQLite；现在两者都只是构造参数。
    """
    llm = NullLLMProvider()
    defaults = dict(
        knowledge=build_knowledge(llm),
        llm=llm,
        sink=None,
        n_normal=3, n_inst=1, n_retail=1,
        use_graph=True, seed=42,
    )
    return ForumModel(**{**defaults, **kwargs})


@pytest.fixture
def model() -> ForumModel:
    m = make_model()
    m.round_num = 1
    return m


# ═══════════════════════════════════════════════════════
#  路由（纯函数）
# ═══════════════════════════════════════════════════════


def test_route_continues_when_previous_phase_has_comments():
    route = route_after_phase(1)
    assert route({"phase_counts": {1: 5}}) == PHASE[2]


def test_route_settles_when_previous_phase_is_silent():
    assert route_after_phase(1)({"phase_counts": {1: 0}}) == SETTLE
    assert route_after_phase(2)({"phase_counts": {2: 0}}) == SETTLE


def test_route_settles_when_phase_counts_missing():
    assert route_after_phase(2)({}) == SETTLE


# ═══════════════════════════════════════════════════════
#  图结构
# ═══════════════════════════════════════════════════════


def test_graph_topology(model):
    g = model.thread_graph.get_graph()

    assert {PUBLISH, PHASE[1], PHASE[2], PHASE[3], SETTLE} <= set(g.nodes)

    edges = {(e.source, e.target) for e in g.edges}
    assert (PUBLISH, PHASE[1]) in edges
    assert (PHASE[1], PHASE[2]) in edges    # 有评论 → 继续
    assert (PHASE[1], SETTLE) in edges      # 无评论 → 直接结算
    assert (PHASE[2], PHASE[3]) in edges
    assert (PHASE[2], SETTLE) in edges
    assert (PHASE[3], SETTLE) in edges


def test_graph_is_compiled_once(model):
    assert model.thread_graph is model.thread_graph


# ═══════════════════════════════════════════════════════
#  端到端执行
# ═══════════════════════════════════════════════════════


def test_full_lifecycle(model):
    state = asyncio.run(run_thread_graph(model, "product"))
    trace = state["trace"]

    assert trace[0] == PUBLISH
    assert trace[1] == PHASE[1]
    assert trace[-1] == SETTLE

    # 帖子与情绪快照都落到了 model 上
    assert len(model.posts) == 1
    assert model.posts[0].topic == "product"
    assert model.sentiment_history

    # 条件边的语义：上一 Phase 有评论才会走到下一 Phase
    counts = state["phase_counts"]
    assert (PHASE[2] in trace) == (counts[1] > 0)
    assert (PHASE[3] in trace) == (counts[2] > 0)


def test_silent_thread_skips_reply_phases(model, monkeypatch):
    """所有人都潜水 → Phase 2/3 被条件边跳过，但仍然要结算。"""
    for agent in model.active_participants:
        monkeypatch.setattr(agent, "should_comment", lambda temp: False)

    state = asyncio.run(run_thread_graph(model, "financial"))

    assert state["trace"] == [PUBLISH, PHASE[1], SETTLE]
    assert state["phase_counts"][1] == 0
    assert model.posts[0].comments[2] == []
    assert model.posts[0].comments[3] == []


def test_graph_and_serial_paths_produce_same_shape():
    """图链路与原串行链路跑出来的帖子结构一致（都是 1 帖 3 Phase）。"""
    graph_model = make_model(use_graph=True, seed=7)
    graph_model.round_num = 1
    asyncio.run(graph_model.astep())

    serial_model = make_model(use_graph=False, seed=7)
    serial_model.round_num = 1
    asyncio.run(serial_model.astep())

    assert len(graph_model.posts) == len(serial_model.posts) == 3
    assert [p.topic for p in graph_model.posts] == \
           [p.topic for p in serial_model.posts]
