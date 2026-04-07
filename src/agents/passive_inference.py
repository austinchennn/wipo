"""
passive_inference — 为 Passive Agent 推算情绪状态。

Passive Agent 不调 LLM，它们的情绪（bull/bear/neutral）根据以下逻辑推算：
  1. 同类型 Active Agent 本 Phase 的情绪分布作为先验基础
  2. Passive Agent 自身的 risk_tolerance / contrarian_tendency / fomo_susceptibility
     对先验做个性化偏移
  3. 加入少量随机噪声，避免全部 Passive Agent 情绪完全一致

输出：{agent_id: "bull" | "bear" | "neutral"}
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Tuple

from ..models import Sentiment

if TYPE_CHECKING:
    from .base_agent import BaseUserAgent, CommentResult


# ─────────────────────────────────────────────────────────────────
#  类型分布计算
# ─────────────────────────────────────────────────────────────────

AgentTypeDist = Dict[str, Dict[Sentiment, float]]  # {agent_type: {sentiment: prob}}

_FALLBACK_DIST: Dict[Sentiment, float] = {
    "bull": 0.33, "bear": 0.33, "neutral": 0.34
}


def _compute_type_distribution(
    active_results: List[Tuple["BaseUserAgent", "CommentResult"]],
) -> AgentTypeDist:
    """
    从 Active Agent 的本 Phase 结果中计算各类型 Agent 的情绪分布。

    只统计真正发言的 Agent（temp >= threshold）；未发言 Agent 记为 neutral。
    """
    counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"bull": 0, "bear": 0, "neutral": 0}
    )

    for agent, result in active_results:
        agent_type = agent.__class__.__name__
        if isinstance(result, Exception):
            counts[agent_type]["neutral"] += 1
        else:
            sentiment = result.sentiment if agent.should_comment(result.temp) else "neutral"
            counts[agent_type][sentiment] += 1

    # 转成概率
    dist: AgentTypeDist = {}
    for agent_type, c in counts.items():
        total = sum(c.values()) or 1
        dist[agent_type] = {k: v / total for k, v in c.items()}

    return dist


# ─────────────────────────────────────────────────────────────────
#  单 Agent 情绪推算
# ─────────────────────────────────────────────────────────────────

def _infer_one(
    agent: "BaseUserAgent",
    base_dist: Dict[Sentiment, float],
) -> Sentiment:
    """
    用 base_dist 作先验，叠加 Agent 个人属性偏移，采样得出情绪。

    偏移规则：
      - risk_tolerance 高 → bull 概率上升
      - contrarian_tendency 高 → 反转主流（bull↓ bear↑ 或反之）
      - fomo_susceptibility 高 → 跟随主流（放大比例最高的那个）
    """
    bull = base_dist.get("bull", 0.33)
    bear = base_dist.get("bear", 0.33)
    neutral = base_dist.get("neutral", 0.34)

    # 1. risk_tolerance 偏移（[-0.15, +0.15]）
    risk_shift = (agent.risk_tolerance - 0.5) * 0.3
    bull += risk_shift
    bear -= risk_shift * 0.5

    # 2. contrarian_tendency 偏移：逆向人会反转 bull/bear
    if agent.contrarian_tendency > 0.7:
        bull, bear = bear, bull  # 直接翻转

    # 3. fomo_susceptibility 偏移：跟风者放大当前主流
    if agent.fomo_susceptibility > 0.7:
        dominant = max(base_dist, key=base_dist.get)
        if dominant == "bull":
            bull *= 1.3
        elif dominant == "bear":
            bear *= 1.3

    # 4. 加噪声（±5%）
    bull    += random.uniform(-0.05, 0.05)
    bear    += random.uniform(-0.05, 0.05)
    neutral += random.uniform(-0.02, 0.02)

    # 5. 截断并归一
    bull    = max(0.0, bull)
    bear    = max(0.0, bear)
    neutral = max(0.0, neutral)
    total   = bull + bear + neutral or 1.0

    # 6. 按概率采样
    r = random.random() * total
    if r < bull:
        return "bull"
    if r < bull + bear:
        return "bear"
    return "neutral"


# ─────────────────────────────────────────────────────────────────
#  主接口
# ─────────────────────────────────────────────────────────────────

def infer_passive_sentiment(
    passive_agents: List["BaseUserAgent"],
    active_results: List[Tuple["BaseUserAgent", "CommentResult"]],
) -> Dict[int, Sentiment]:
    """
    为所有 Passive Agent 推算本 Phase 的情绪状态。

    参数：
        passive_agents  — Passive Agent 列表
        active_results  — [(agent, CommentResult), ...] 本 Phase Active Agent 结果

    返回：
        {agent_id: "bull" | "bear" | "neutral"}
    """
    type_dist = _compute_type_distribution(active_results)

    grid: Dict[int, Sentiment] = {}
    for agent in passive_agents:
        agent_type = agent.__class__.__name__
        base = type_dist.get(agent_type, _FALLBACK_DIST)
        sentiment = _infer_one(agent, base)
        agent.last_sentiment = sentiment
        grid[agent.unique_id] = sentiment

    return grid
