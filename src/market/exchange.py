"""
exchange — 交易所引擎

管理订单簿、持仓账本、K 线生成、交易 Session 生命周期。

时序：
  ForumModel 的每个 Thread（3 Phase 结束后）触发一次交易 Session：
    1. 根据 SentimentGrid → 各 Agent 生成交易决策（Order）
    2. 订单按随机顺序提交到 OrderBook → 连续竞价撮合
    3. 成交 → 更新持仓 → 生成 OHLCV K 线柱
    4. 残余挂单保留到下一个 Session（模拟真实连续交易）
"""

from __future__ import annotations

import logging
import random
from typing import Dict, List, Optional

from ..config import (
    IPO_PRICE,
    INST_INITIAL_CASH,
    RETAIL_INITIAL_CASH,
    NORMAL_INITIAL_CASH,
    INST_INITIAL_SHARES,
    RETAIL_INITIAL_SHARES,
    NORMAL_INITIAL_SHARES,
    TICK_SIZE,
)
from .models import OHLCVBar, Order, OrderBookSnapshot, Portfolio, Side, Trade
from .order_book import OrderBook

logger = logging.getLogger(__name__)


class Exchange:
    """交易所 —— 限价订单簿连续竞价撮合引擎。

    生命周期：
        1. ForumModel.__init__ 时创建 Exchange
        2. 每个 Thread 结束后调用 run_session(orders, event) → 撮合 + K 线
        3. 模拟结束后，exchange.kline_history 包含完整 K 线序列
    """

    def __init__(self, ipo_price: float = IPO_PRICE):
        self._order_book = OrderBook()
        self._portfolios: Dict[int, Portfolio] = {}
        self._tick: int = 0
        self._last_price: float = ipo_price
        self._ipo_price: float = ipo_price
        self.kline_history: List[OHLCVBar] = []
        self.trade_history: List[Trade] = []

    # ── 持仓初始化 ──

    def init_portfolio(self, agent_id: int, agent_type: str) -> None:
        """根据 Agent 类型分配初始资金和股票。"""
        if agent_id in self._portfolios:
            return
        cash_map = {
            "InstTraderAgent":   INST_INITIAL_CASH,
            "RetailTraderAgent": RETAIL_INITIAL_CASH,
            "NormalAgent":       NORMAL_INITIAL_CASH,
        }
        shares_map = {
            "InstTraderAgent":   INST_INITIAL_SHARES,
            "RetailTraderAgent": RETAIL_INITIAL_SHARES,
            "NormalAgent":       NORMAL_INITIAL_SHARES,
        }
        self._portfolios[agent_id] = Portfolio(
            cash=cash_map.get(agent_type, RETAIL_INITIAL_CASH),
            shares=shares_map.get(agent_type, RETAIL_INITIAL_SHARES),
        )

    def get_portfolio(self, agent_id: int) -> Optional[Portfolio]:
        return self._portfolios.get(agent_id)

    @property
    def last_price(self) -> float:
        return self._last_price

    # ── 交易 Session ──

    def run_session(
        self,
        orders: List[Order],
        event: Optional[str] = None,
    ) -> OHLCVBar:
        """执行一次交易 Session（对应一个 Thread 结束后的撮合窗口）。

        流程：
            1. 验证并冻结资金/股票
            2. 随机打乱订单顺序（模拟交易所的非确定性到达）
            3. 逐笔提交到 OrderBook → 连续竞价
            4. 更新持仓
            5. 生成 OHLCV K 线柱

        返回：本次 Session 的 K 线柱。
        """
        self._tick += 1

        # 1. 验证 + 冻结
        valid_orders: List[Order] = []
        for order in orders:
            if self._validate_and_freeze(order):
                valid_orders.append(order)

        # 2. 随机打乱（模拟真实交易所订单到达的随机性）
        random.shuffle(valid_orders)

        # 3. 逐笔撮合
        session_trades: List[Trade] = []
        for order in valid_orders:
            trades = self._order_book.submit(order)
            for trade in trades:
                self._settle(trade)
                session_trades.append(trade)

        # 4. 解冻未成交订单的冻结资金/股票
        self._unfreeze_remaining()

        self.trade_history.extend(session_trades)

        # 5. 更新最新价
        if session_trades:
            self._last_price = session_trades[-1].price

        # 6. 生成 K 线
        bar = self._build_bar(session_trades, event)
        self.kline_history.append(bar)

        logger.info(
            "[撮合] tick=%d | 订单=%d 有效=%d 成交=%d笔 | 价格=%.2f %s",
            self._tick, len(orders), len(valid_orders),
            len(session_trades), self._last_price,
            f"| {event}" if event else "",
        )

        return bar

    # ── 验证 + 冻结 ──

    def _validate_and_freeze(self, order: Order) -> bool:
        """验证订单合法性并冻结相应资金/股票。"""
        pf = self._portfolios.get(order.agent_id)
        if pf is None:
            return False

        if order.side is Side.BUY:
            cost = order.price * order.quantity
            if pf.available_cash < cost:
                return False
            pf.frozen_cash += cost
        else:
            if pf.available_shares < order.quantity:
                return False
            pf.frozen_shares += order.quantity

        return True

    # ── 成交结算 ──

    def _settle(self, trade: Trade) -> None:
        """逐笔成交结算：买方扣钱加股，卖方扣股加钱。"""
        buyer = self._portfolios.get(trade.buyer_id)
        seller = self._portfolios.get(trade.seller_id)

        amount = trade.price * trade.quantity

        if buyer:
            buyer.frozen_cash -= amount   # 释放冻结
            buyer.cash -= amount          # 实际扣款
            buyer.shares += trade.quantity

        if seller:
            seller.frozen_shares -= trade.quantity
            seller.shares -= trade.quantity
            seller.cash += amount

    # ── 解冻残余 ──

    def _unfreeze_remaining(self) -> None:
        """Session 结束后，清空订单簿中的残余挂单并解冻冻结资源。

        注意：真实交易所挂单会持续存在。这里为简化每个 Session
        结束后清理残余。如果需要跨 Session 持续挂单，可移除此逻辑。
        """
        for order in self._order_book._bids:
            pf = self._portfolios.get(order.agent_id)
            if pf:
                pf.frozen_cash -= order.price * order.remaining
        for order in self._order_book._asks:
            pf = self._portfolios.get(order.agent_id)
            if pf:
                pf.frozen_shares -= order.remaining

        self._order_book.clear_all()

    # ── K 线构建 ──

    def _build_bar(
        self, trades: List[Trade], event: Optional[str]
    ) -> OHLCVBar:
        """从本 Session 的成交列表构建 OHLCV K 线柱。"""
        if not trades:
            # 无成交 → 平盘
            return OHLCVBar(
                tick=self._tick,
                open=self._last_price,
                high=self._last_price,
                low=self._last_price,
                close=self._last_price,
                volume=0,
                trade_count=0,
                event=event,
            )

        prices = [t.price for t in trades]
        volumes = [t.quantity for t in trades]

        return OHLCVBar(
            tick=self._tick,
            open=prices[0],
            high=max(prices),
            low=min(prices),
            close=prices[-1],
            volume=sum(volumes),
            trade_count=len(trades),
            event=event,
        )

    # ── 订单簿快照 ──

    def order_book_snapshot(self, depth: int = 15) -> OrderBookSnapshot:
        return self._order_book.snapshot(depth=depth)

    # ── 统计 ──

    def market_summary(self) -> Dict:
        """返回市场汇总信息。"""
        total_volume = sum(t.quantity for t in self.trade_history)
        total_trades = len(self.trade_history)
        pnl_by_type: Dict[str, float] = {}

        return {
            "last_price": self._last_price,
            "ipo_price": self._ipo_price,
            "change_pct": round(
                (self._last_price - self._ipo_price) / self._ipo_price * 100, 2
            ),
            "total_volume": total_volume,
            "total_trades": total_trades,
            "ticks": self._tick,
        }
