"""
帖子线程编排图 —— LangGraph StateGraph

把原先写死在 ForumModel._run_thread() 里的串行流程，改写成显式状态图：

    START → publish → phase_1 ─┬─→ phase_2 ─┬─→ phase_3 ─→ settle → END
                               │            │                ↑
                               │            └────────────────┘  Phase2 无评论
                               └─────────────────────────────┘  Phase1 无评论

为什么用图：
  - 「上一 Phase 没人说话就跳过下一 Phase」从函数内的 if 分支变成图上的条件边，
    调度逻辑一眼可见，也可以直接画出来（graph.get_graph().draw_mermaid()）
  - 每个 Phase 是独立节点，后续要插入新节点（监管介入、水军注入、突发新闻）
    只需改图的连线，不用动 ForumModel
  - state["trace"] 天然记录执行路径，测试和回放都能直接断言

节点内部仍然复用 ForumModel 已有的 Phase 实现，本模块只负责编排。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict

from langgraph.graph import END, START, StateGraph

from .state import ThreadState, make_initial_state

if TYPE_CHECKING:  # 避免与 forum.py 形成循环导入
    from ..environment.forum import ForumModel

logger = logging.getLogger(__name__)

# 节点名常量（测试与路由共用，避免裸字符串写错）
PUBLISH = "publish"
SETTLE = "settle"
PHASE = {1: "phase_1", 2: "phase_2", 3: "phase_3"}


# ═══════════════════════════════════════════════════════
#  路由（纯函数，可独立测试）
# ═══════════════════════════════════════════════════════


def route_after_phase(phase: int) -> Callable[[ThreadState], str]:
    """
    生成「Phase N 之后往哪走」的路由函数。

    上一 Phase 一条评论都没有 → 后续 Phase 无回复目标，直接进入撮合结算。
    """

    def route(state: ThreadState) -> str:
        if state.get("phase_counts", {}).get(phase, 0) > 0:
            return PHASE[phase + 1]
        logger.info("  → 跳过 Phase %d（Phase %d 无评论）", phase + 1, phase)
        return SETTLE

    route.__name__ = f"route_after_phase_{phase}"
    return route


# ═══════════════════════════════════════════════════════
#  构图
# ═══════════════════════════════════════════════════════


def build_thread_graph(model: "ForumModel"):
    """为指定 ForumModel 编译一张帖子线程图（节点闭包持有 model）。"""

    def phase_update(
        state: ThreadState, phase: int, snapshot: Any
    ) -> Dict[str, Any]:
        """把一个 Phase 的执行结果折叠进 state。"""
        post = state["post"]
        latest = model.sentiment_history[-1] if model.sentiment_history else None
        return {
            "snapshot": snapshot,
            "phase_counts": {**state["phase_counts"], phase: len(post.comments[phase])},
            "sentiment": latest.summary() if latest else state["sentiment"],
            "trace": [PHASE[phase]],
        }

    async def publish(state: ThreadState) -> Dict[str, Any]:
        """Host 发帖。"""
        post = model._host_publish(state["topic"])
        await model._emit_post(post)
        return {"post": post, "trace": [PUBLISH]}

    async def phase_1(state: ThreadState) -> Dict[str, Any]:
        """Phase 1 —— 直面楼主。"""
        snapshot = await model._phase_1(state["post"])
        return phase_update(state, 1, snapshot)

    def make_reply_node(phase: int):
        """Phase 2（互攻 / 抱团）和 Phase 3（余波）共用回复逻辑。"""

        async def node(state: ThreadState) -> Dict[str, Any]:
            post = state["post"]
            snapshot = await model._reply_phase(
                post, phase, state["snapshot"], post.comments[phase - 1]
            )
            return phase_update(state, phase, snapshot)

        node.__name__ = PHASE[phase]
        return node

    async def settle(state: ThreadState) -> Dict[str, Any]:
        """帖子讨论结束 → 触发交易撮合。"""
        post = state["post"]
        counts = state["phase_counts"]
        logger.info(
            "[结束] %s  P1=%d P2=%d P3=%d",
            post.id, counts.get(1, 0), counts.get(2, 0), counts.get(3, 0),
        )
        await model._settle_trades(post)
        return {"trace": [SETTLE]}

    graph = StateGraph(ThreadState)
    graph.add_node(PUBLISH, publish)
    graph.add_node(PHASE[1], phase_1)
    graph.add_node(PHASE[2], make_reply_node(2))
    graph.add_node(PHASE[3], make_reply_node(3))
    graph.add_node(SETTLE, settle)

    graph.add_edge(START, PUBLISH)
    graph.add_edge(PUBLISH, PHASE[1])
    graph.add_conditional_edges(
        PHASE[1], route_after_phase(1), {PHASE[2]: PHASE[2], SETTLE: SETTLE}
    )
    graph.add_conditional_edges(
        PHASE[2], route_after_phase(2), {PHASE[3]: PHASE[3], SETTLE: SETTLE}
    )
    graph.add_edge(PHASE[3], SETTLE)
    graph.add_edge(SETTLE, END)

    return graph.compile()


async def run_thread_graph(model: "ForumModel", topic: str) -> ThreadState:
    """跑完一个帖子的完整生命周期，返回终态（含 trace）。"""
    return await model.thread_graph.ainvoke(
        make_initial_state(model.round_num, topic)
    )
