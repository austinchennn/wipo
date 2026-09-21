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

决策模型（可选）：
  注入 DecisionModel（Jev）后，「买还是卖」和「散户追不追价」由模型对
  Agent 画像 + 市场状态给出的概率决定；下单量仍按资金/持仓规则算（硬约束
  不交给模型）。模型不可用、单次调用失败时，该 Agent 原样走上面的规则。
"""

from __future__ import annotations

import logging
import math
import random
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

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
from ..ports.decision import Assessment, DecisionModel
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
    if not _participates(agent):
        return None
    return build_order(agent, sentiment, last_price, portfolio)


def _participates(agent: "BaseUserAgent") -> bool:
    """参与度闸门：这个 Agent 本轮会不会考虑交易。

    单独拆出来是为了让 TradingSession 能先过闸门、再只为"会下单"的 Agent
    调决策模型——Normal 85% 会被挡掉，不该为它们白付调用费。
    """
    # ── 1. NormalAgent 低参与度 ──
    if agent.__class__.__name__ == "NormalAgent":
        if random.random() > NORMAL_TRADE_PROBABILITY:
            return False  # 大多数情况不交易

    # ── 2. Passive Agent 参与度折扣 ──
    if not agent.is_active:
        if random.random() > PASSIVE_TRADE_DISCOUNT:
            return False

    return True


def build_order(
    agent: "BaseUserAgent",
    sentiment: Sentiment,
    last_price: float,
    portfolio: Portfolio,
    assessment: Optional[Assessment] = None,
) -> Optional[Order]:
    """已通过参与度闸门后，方向 → 数量 → 价格。

    assessment 是决策模型对该 Agent 的回答；为 None（没配模型 / 调用失败）
    或缺字段时，方向和定价走原有规则。
    """
    agent_type = agent.__class__.__name__

    # ── 3. 方向决策 ──
    if _has_answers(assessment, "buy", "sell"):
        side = _side_from_assessment(assessment)
    else:
        side = _decide_side(agent, sentiment)
    if side is None:
        return None  # 观望

    # ── 4. 数量决策 ──
    quantity = _decide_quantity(agent, agent_type, side, last_price, portfolio)
    if quantity <= 0:
        return None

    # ── 5. 价格决策 ──
    urgency = assessment.get("urgent") if assessment else None
    price = _decide_price(agent, agent_type, side, last_price, sentiment, urgency)
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
#  决策模型：问题、状态描述、答案 → 决策
# ─────────────────────────────────────────────────────────────────
# 模型面向的文本集中在这一节，想调措辞不用翻规则代码。

_ROLE = {
    "InstTraderAgent":   "机构投资者",
    "RetailTraderAgent": "散户投资者",
    "NormalAgent":       "很少交易的吃瓜群众",
}

_SENTIMENT_CN = {"bull": "看多", "bear": "看空", "neutral": "中性"}

_Q_BUY = "这位交易者此刻是否想买入股票？"
_Q_SELL = "这位交易者此刻是否想卖出手中的股票？"
_Q_URGENT = "这位交易者是否愿意让价以尽快成交（追涨买入或杀跌卖出）？"


def _questions_for(agent_type: str) -> Dict[str, str]:
    """要问模型的问题。只有散户的定价会用到"追价"，其余不问，省一题。"""
    qs = {"buy": _Q_BUY, "sell": _Q_SELL}
    if agent_type == "RetailTraderAgent":
        qs["urgent"] = _Q_URGENT
    return qs


def _describe(
    agent: "BaseUserAgent",
    sentiment: Sentiment,
    last_price: float,
    portfolio: Portfolio,
    recent_closes: Sequence[float],
) -> str:
    """Agent 画像 + 市场状态 → 给模型看的 state 文本。"""
    agent_type = agent.__class__.__name__
    change = (last_price - IPO_PRICE) / IPO_PRICE * 100 if IPO_PRICE else 0.0
    closes = "、".join(f"{c:.2f}" for c in recent_closes) or "暂无"
    return (
        f"交易者：{_ROLE.get(agent_type, '投资者')}。"
        f"风险承受度 {agent.risk_tolerance:.2f}，"
        f"FOMO 易感度 {agent.fomo_susceptibility:.2f}，"
        f"情绪波动 {agent.emotional_volatility:.2f}，"
        f"逆向倾向 {agent.contrarian_tendency:.2f}（均为 0-1）。\n"
        f"读完最新论坛帖子后的观点：{_SENTIMENT_CN.get(sentiment, sentiment)}。\n"
        f"市场：最新价 {last_price:.2f}（较发行价 {IPO_PRICE:.2f} "
        f"{change:+.1f}%），最近收盘价 {closes}。\n"
        f"持仓：可用资金 {portfolio.available_cash:.0f}，"
        f"可用股数 {portfolio.available_shares}。"
    )


def _has_answers(assessment: Optional[Assessment], *names: str) -> bool:
    return assessment is not None and all(n in assessment for n in names)


def _side_from_assessment(assessment: Assessment) -> Optional[Side]:
    """买/卖概率 → 方向。

    取较大的一边，并以它的概率为"真去下单"的概率：买 0.9 卖 0.1 →
    90% 买；买 0.3 卖 0.2 → 70% 观望。这样模型给的是校准过的概率，
    模拟里保留随机性，而不是一刀切阈值把 Agent 变成同一批人。
    """
    p_buy = min(max(assessment["buy"], 0.0), 1.0)
    p_sell = min(max(assessment["sell"], 0.0), 1.0)
    side, conf = (Side.BUY, p_buy) if p_buy >= p_sell else (Side.SELL, p_sell)
    return side if random.random() < conf else None


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
    urgency: Optional[float] = None,
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
        return _retail_price(agent, side, last_price, sentiment, urgency)
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
    urgency: Optional[float] = None,
) -> float:
    """散户定价：情绪驱动 ±0.5%~3%，FOMO 追涨杀跌。

    urgency 是决策模型给出的"愿意为成交让价"的概率；为 None 时用原规则
    （情绪与方向同向 且 fomo > 0.5）判断追不追。
    """
    # 基础偏移
    base_spread = random.uniform(0.005, 0.02)

    # FOMO 放大因子
    fomo = agent.fomo_susceptibility
    emotion = agent.emotional_volatility
    aggression = 1.0 + fomo * 0.5 + emotion * 0.3

    spread = base_spread * aggression

    if urgency is not None:
        chase = random.random() < urgency
    elif side is Side.BUY:
        chase = sentiment == "bull" and fomo > 0.5
    else:
        chase = sentiment == "bear" and fomo > 0.5

    if side is Side.BUY:
        if chase:
            # FOMO 追涨：买入价高于市价
            price = last_price * (1.0 + spread * 0.5)
        else:
            price = last_price * (1.0 - spread)
    else:
        if chase:
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
        session = TradingSession(exchange, sentiment_grid, decider)
        bar = session.run(all_agents)

    decider 为 None 或不可用 → 全员走规则（与接入决策模型前逐字节一致，
    含随机数消耗顺序，seed 相同结果相同）。
    """

    RECENT_CLOSES = 3   # 给模型看最近几根 K 线的收盘价

    def __init__(
        self,
        exchange: "Exchange",
        sentiment_grid: Dict[int, Sentiment],
        decider: Optional[DecisionModel] = None,
    ):
        from .exchange import Exchange
        self._exchange = exchange
        self._grid = sentiment_grid
        self._decider = decider

    def run(
        self,
        all_agents: List["BaseUserAgent"],
        event: Optional[str] = None,
    ) -> "OHLCVBar":
        """为所有 Agent 生成交易决策并提交到 Exchange 撮合。"""
        if self._decider is not None and self._decider.is_available:
            orders = self._model_orders(all_agents)
        else:
            orders = self._rule_orders(all_agents)

        logger.info(
            "[交易决策] 总Agent=%d → 生成订单=%d (买=%d 卖=%d)",
            len(all_agents), len(orders),
            sum(1 for o in orders if o.side is Side.BUY),
            sum(1 for o in orders if o.side is Side.SELL),
        )

        return self._exchange.run_session(orders, event=event)

    # ── 纯规则路径 ──

    def _rule_orders(self, all_agents: List["BaseUserAgent"]) -> List[Order]:
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
        return orders

    # ── 决策模型路径 ──

    def _model_orders(self, all_agents: List["BaseUserAgent"]) -> List[Order]:
        """先过参与度闸门，再把"会下单"的 Agent 一次性批量交给模型。"""
        last_price = self._exchange.last_price
        closes = [b.close for b in self._exchange.kline_history[-self.RECENT_CLOSES:]]

        candidates = []   # (agent, sentiment, portfolio)
        for agent in all_agents:
            portfolio = self._exchange.get_portfolio(agent.unique_id)
            if portfolio is None or not _participates(agent):
                continue
            sentiment = self._grid.get(agent.unique_id, "neutral")
            candidates.append((agent, sentiment, portfolio))

        requests = [
            (
                _describe(a, s, last_price, pf, closes),
                _questions_for(a.__class__.__name__),
            )
            for a, s, pf in candidates
        ]
        assessments = self._decider.assess_many(requests)

        orders: List[Order] = []
        for (agent, sentiment, portfolio), assessment in zip(candidates, assessments):
            order = build_order(agent, sentiment, last_price, portfolio, assessment)
            if order is not None:
                orders.append(order)

        failed = sum(1 for a in assessments if a is None)
        if failed:
            logger.warning("[交易决策] 决策模型 %d/%d 次调用失败，这些 Agent 已退回规则",
                           failed, len(assessments))
        return orders
