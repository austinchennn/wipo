"""
trading_agent — Agent 交易决策逻辑

每个 Thread 的 3 Phase 结束后，所有 Agent（Active + Passive）根据最新情绪
和市场状态生成交易指令（限价单）。

决策维度：
  1. 方向（买/卖/观望）  — 主要由 sentiment 决定
  2. 下单量（股数）      — 由 capital_level × risk_tolerance × 可用资金/持仓 决定
  3. 委托价格            — 由 agent 类型差异化：
       - 机构：窄价差挂单（贴近 best bid/ask），有耐心
       - 散户：情绪驱动的价格偏移（急切时追高/杀跌）
       - 普通人：随机性更大，下单概率低

NormalAgent 参与度低：大部分 Phase 不交易（模拟真实市场中"吃瓜群众"行为）。
"""

from __future__ import annotations

import logging
import math
import random
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from ..config import (
    IPO_PRICE,
    TICK_SIZE,
    INST_ORDER_LOTS,
    RETAIL_ORDER_LOTS,
    NORMAL_ORDER_LOTS,
    NORMAL_TRADE_PROBABILITY,
    PASSIVE_TRADE_DISCOUNT,
)
from ..models import Sentiment
from .models import Order, Portfolio, Side

if TYPE_CHECKING:
    from ..agents.base_agent import BaseUserAgent

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
#  价格取整
# ─────────────────────────────────────────────────────────────────

def _round_price(price: float) -> float:
    """委托价对齐到 tick_size（0.01）。"""
    return round(round(price / TICK_SIZE) * TICK_SIZE, 2)


# ─────────────────────────────────────────────────────────────────
#  单 Agent 交易决策
# ─────────────────────────────────────────────────────────────────

def decide_order(
    agent: "BaseUserAgent",
    sentiment: Sentiment,
    last_price: float,
    portfolio: Portfolio,
) -> Optional[Order]:
    """根据 Agent 属性和情绪生成一笔限价单（或 None = 观望）。

    决策链：
        sentiment → 方向
        risk_tolerance × capital_level → 仓位比例
        agent_type → 定价策略
        portfolio → 可用资金/持仓校验
    """
    agent_type = agent.__class__.__name__

    # ── 1. NormalAgent 低参与度 ──
    if agent_type == "NormalAgent":
        if random.random() > NORMAL_TRADE_PROBABILITY:
            return None  # 大多数情况不交易

    # ── 2. Passive Agent 参与度折扣 ──
    if not agent.is_active:
        if random.random() > PASSIVE_TRADE_DISCOUNT:
            return None

    # ── 3. 方向决策 ──
    side = _decide_side(agent, sentiment)
    if side is None:
        return None  # neutral + 低风险偏好 → 观望

    # ── 4. 数量决策 ──
    quantity = _decide_quantity(agent, agent_type, side, last_price, portfolio)
    if quantity <= 0:
        return None

    # ── 5. 价格决策 ──
    price = _decide_price(agent, agent_type, side, last_price, sentiment)
    if price <= 0:
        return None

    return Order(
        agent_id=agent.unique_id,
        agent_type=agent_type,
        side=side,
        price=price,
        quantity=quantity,
    )


# ─────────────────────────────────────────────────────────────────
#  方向决策
# ─────────────────────────────────────────────────────────────────

def _decide_side(
    agent: "BaseUserAgent", sentiment: Sentiment
) -> Optional[Side]:
    """情绪 → 交易方向。

    - bull → 买入（概率根据 risk_tolerance 调整）
    - bear → 卖出（概率根据 risk_tolerance 调整）
    - neutral → 大概率观望，小概率随机方向
    """
    rt = agent.risk_tolerance

    if sentiment == "bull":
        # 高 risk_tolerance → 几乎必买；低 → 仍有概率观望
        if random.random() < 0.5 + rt * 0.5:
            return Side.BUY
        return None

    if sentiment == "bear":
        if random.random() < 0.5 + rt * 0.5:
            return Side.SELL
        return None

    # neutral
    if rt > 0.7 and random.random() < 0.2:
        return random.choice([Side.BUY, Side.SELL])
    return None


# ─────────────────────────────────────────────────────────────────
#  数量决策
# ─────────────────────────────────────────────────────────────────

def _decide_quantity(
    agent: "BaseUserAgent",
    agent_type: str,
    side: Side,
    last_price: float,
    portfolio: Portfolio,
) -> int:
    """决定下单股数。

    核心公式：
        base_lots = 类型基准手数
        risk_factor = risk_tolerance × (0.3 ~ 1.0)
        quantity = base_lots × risk_factor
        + 上限校验（不超过可用资金 / 可用持仓）
    """
    # 基准手数（100 股/手）
    lots_map = {
        "InstTraderAgent":   INST_ORDER_LOTS,
        "RetailTraderAgent": RETAIL_ORDER_LOTS,
        "NormalAgent":       NORMAL_ORDER_LOTS,
    }
    base_lots = lots_map.get(agent_type, RETAIL_ORDER_LOTS)

    # risk_tolerance 映射到 [0.3, 1.0] 的仓位比例
    risk_factor = 0.3 + agent.risk_tolerance * 0.7

    # 加入一点随机性
    noise = random.uniform(0.7, 1.3)
    raw_quantity = int(base_lots * risk_factor * noise)

    # 取整到 100 股（手）
    raw_quantity = max(100, (raw_quantity // 100) * 100)

    # 可用资金/持仓上限
    if side is Side.BUY:
        max_affordable = int(portfolio.available_cash / last_price) if last_price > 0 else 0
        max_affordable = (max_affordable // 100) * 100
        return min(raw_quantity, max_affordable)
    else:
        max_sellable = (portfolio.available_shares // 100) * 100
        return min(raw_quantity, max_sellable)


# ─────────────────────────────────────────────────────────────────
#  价格决策
# ─────────────────────────────────────────────────────────────────

def _decide_price(
    agent: "BaseUserAgent",
    agent_type: str,
    side: Side,
    last_price: float,
    sentiment: Sentiment,
) -> float:
    """决定委托价格。

    定价策略按 Agent 类型分层：

    机构（InstTraderAgent）：
        - 窄价差，贴近市场价 ±0.1%~0.5%
        - 不追涨杀跌，耐心挂单

    散户（RetailTraderAgent）：
        - 宽价差 ±0.5%~3%
        - 情绪激动时追高/杀跌（fomo_susceptibility 放大偏移）
        - emotional_volatility 增加价格随机性

    普通人（NormalAgent）：
        - 随机性最大 ±1%~5%
        - 基本不懂定价，"差不多就行"
    """
    if last_price <= 0:
        last_price = IPO_PRICE

    if agent_type == "InstTraderAgent":
        return _inst_price(agent, side, last_price)
    elif agent_type == "RetailTraderAgent":
        return _retail_price(agent, side, last_price, sentiment)
    else:
        return _normal_price(agent, side, last_price)


def _inst_price(
    agent: "BaseUserAgent", side: Side, last_price: float
) -> float:
    """机构定价：窄价差 ±0.1%~0.5%，理性挂单。"""
    spread_pct = random.uniform(0.001, 0.005)
    if side is Side.BUY:
        # 买入：略低于市价
        price = last_price * (1.0 - spread_pct)
    else:
        # 卖出：略高于市价
        price = last_price * (1.0 + spread_pct)
    return _round_price(price)


def _retail_price(
    agent: "BaseUserAgent",
    side: Side,
    last_price: float,
    sentiment: Sentiment,
) -> float:
    """散户定价：情绪驱动 ±0.5%~3%，FOMO 追涨杀跌。"""
    # 基础偏移
    base_spread = random.uniform(0.005, 0.02)

    # FOMO 放大因子
    fomo = agent.fomo_susceptibility
    emotion = agent.emotional_volatility
    aggression = 1.0 + fomo * 0.5 + emotion * 0.3

    spread = base_spread * aggression

    if side is Side.BUY:
        if sentiment == "bull" and fomo > 0.5:
            # FOMO 追涨：买入价高于市价
            price = last_price * (1.0 + spread * 0.5)
        else:
            price = last_price * (1.0 - spread)
    else:
        if sentiment == "bear" and fomo > 0.5:
            # 恐慌杀跌：卖出价低于市价
            price = last_price * (1.0 - spread * 0.5)
        else:
            price = last_price * (1.0 + spread)

    return _round_price(price)


def _normal_price(
    agent: "BaseUserAgent", side: Side, last_price: float
) -> float:
    """普通人定价：随机噪声大，不精准。"""
    spread = random.uniform(0.01, 0.05)

    if side is Side.BUY:
        price = last_price * (1.0 - spread * random.uniform(0.2, 1.0))
    else:
        price = last_price * (1.0 + spread * random.uniform(0.2, 1.0))

    return _round_price(price)


# ─────────────────────────────────────────────────────────────────
#  批量决策入口
# ─────────────────────────────────────────────────────────────────

class TradingSession:
    """一次交易 Session 的编排器。

    由 ForumModel 在每个 Thread 结束后调用：
        session = TradingSession(exchange, sentiment_grid)
        bar = session.run(all_agents)
    """

    def __init__(
        self,
        exchange: "Exchange",
        sentiment_grid: Dict[int, Sentiment],
    ):
        from .exchange import Exchange
        self._exchange = exchange
        self._grid = sentiment_grid

    def run(
        self,
        all_agents: List["BaseUserAgent"],
        event: Optional[str] = None,
    ) -> "OHLCVBar":
        """为所有 Agent 生成交易决策并提交到 Exchange 撮合。"""
        from .models import OHLCVBar

        orders: List[Order] = []
        last_price = self._exchange.last_price

        for agent in all_agents:
            sentiment = self._grid.get(agent.unique_id, "neutral")
            portfolio = self._exchange.get_portfolio(agent.unique_id)
            if portfolio is None:
                continue

            order = decide_order(agent, sentiment, last_price, portfolio)
            if order is not None:
                orders.append(order)

        logger.info(
            "[交易决策] 总Agent=%d → 生成订单=%d (买=%d 卖=%d)",
            len(all_agents), len(orders),
            sum(1 for o in orders if o.side is Side.BUY),
            sum(1 for o in orders if o.side is Side.SELL),
        )

        return self._exchange.run_session(orders, event=event)
