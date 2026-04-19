"""
数据模型 —— Comment / Post / ThreadSnapshot / SentimentGrid

层级结构:  Post (帖子)
    └─ Comment (Phase 1 一级评论)
        └─ Comment (Phase 2 二级评论)
            └─ Comment (Phase 3 三级评论)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Literal, Optional


# ═══════════════════════════════════════════════════════
#  Comment
# ═══════════════════════════════════════════════════════


@dataclass
class Comment:
    """单条评论节点"""

    id: str
    author_id: int
    author_type: str          # 类名，如 "NormalAgent"
    author_name: str          # 显示名，如 "NormalAgent#3"
    content: str
    temp: float               # 发言温度 / 热情值
    phase: int                # 所属阶段 1 / 2 / 3
    sentiment: Optional[str] = None     # "bull" | "bear" | "neutral"
    parent_id: Optional[str] = None
    created_at: Optional[datetime] = None  # 评论生成时刻（用于回放排序）
    children: List[Comment] = field(default_factory=list)

    @staticmethod
    def make_id() -> str:
        return uuid.uuid4().hex[:8]


# ═══════════════════════════════════════════════════════
#  Post
# ═══════════════════════════════════════════════════════


@dataclass
class Post:
    """一个帖子 (Thread) 及其三层评论"""

    id: str
    round_num: int
    topic: str                # "product" | "financial" | "policy"
    content: str
    created_at: Optional[datetime] = None  # 帖子创建时刻
    comments: Dict[int, List[Comment]] = field(
        default_factory=lambda: {1: [], 2: [], 3: []}
    )


# ═══════════════════════════════════════════════════════
#  ThreadSnapshot
# ═══════════════════════════════════════════════════════


@dataclass
class ThreadSnapshot:
    """
    某 Phase 结束后的帖子快照。
    用于在 Phase 之间传递给所有 Agent 阅读。
    """

    post_id: str
    phase: int
    post_content: str
    phase1_comments: List[Comment] = field(default_factory=list)

    def to_text(self, max_comments: int = 30) -> str:
        """将快照序列化为可读文本（供 Agent / LLM 消费）"""

        lines = [
            f"{'=' * 50}",
            f" 帖子 [{self.post_id}]  截至 Phase {self.phase} 快照",
            f"{'=' * 50}",
            f"【主贴】{self.post_content}",
            "",
        ]

        for i, c in enumerate(self.phase1_comments, 1):
            lines.append(
                f"  #{i} [{c.author_name}] "
                f"(temp={c.temp:.2f}): {c.content}"
            )
            for j, sub in enumerate(c.children, 1):
                lines.append(
                    f"    #{i}.{j} └─ [{sub.author_name}] "
                    f"(temp={sub.temp:.2f}): {sub.content}"
                )
                for k, reply in enumerate(sub.children, 1):
                    lines.append(
                        f"      #{i}.{j}.{k} └── [{reply.author_name}] "
                        f"(temp={reply.temp:.2f}): {reply.content}"
                    )

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════
#  SentimentGrid
# ═══════════════════════════════════════════════════════

Sentiment = Literal["bull", "bear", "neutral"]


@dataclass
class SentimentGrid:
    """
    单个 Phase 结束后全体 Agent（active + passive）的情绪快照。

    供前端渲染 10,000 格情绪矩阵（100×100）。
    """

    round_num: int
    topic: str
    phase: int
    # {agent_id: "bull" | "bear" | "neutral"}
    grid: Dict[int, Sentiment] = field(default_factory=dict)

    def summary(self) -> Dict[str, int]:
        """返回各情绪的 Agent 数量统计。"""
        counts: Dict[str, int] = {"bull": 0, "bear": 0, "neutral": 0}
        for s in self.grid.values():
            counts[s] += 1
        return counts

    def to_list(self, agent_ids: List[int]) -> List[Sentiment]:
        """按 agent_ids 顺序返回情绪列表（供前端按序渲染矩阵）。"""
        return [self.grid.get(aid, "neutral") for aid in agent_ids]
