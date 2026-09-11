"""
LangGraph 状态定义 —— ThreadState

一个 ThreadState 描述「一个帖子从发布到撮合结算」的完整生命周期，
在 thread_graph 的各节点之间流转。

字段约定：
  - 节点只返回自己修改的字段，其余由 LangGraph 自动保留
  - trace 用 operator.add 归并（append 语义），记录节点执行顺序
"""

from __future__ import annotations

import operator
from typing import Annotated, Dict, List, Optional, TypedDict

from ..models import Post, ThreadSnapshot


class ThreadState(TypedDict, total=False):
    """帖子线程图的共享状态。"""

    round_num: int                      # 所属宏观轮次
    topic: str                          # "product" | "financial" | "policy"
    post: Optional[Post]                # publish 节点产出
    snapshot: Optional[ThreadSnapshot]  # 上一 Phase 结束时的帖子快照
    phase_counts: Dict[int, int]        # {phase: 该 Phase 产生的评论数}
    sentiment: Dict[str, int]           # 最近一次 Phase 的情绪汇总
    trace: Annotated[List[str], operator.add]  # 节点执行轨迹


def make_initial_state(round_num: int, topic: str) -> ThreadState:
    """构造一个帖子线程的初始状态。"""
    return ThreadState(
        round_num=round_num,
        topic=topic,
        post=None,
        snapshot=None,
        phase_counts={1: 0, 2: 0, 3: 0},
        sentiment={"bull": 0, "bear": 0, "neutral": 0},
        trace=[],
    )
