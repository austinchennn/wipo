"""
market.models — 交易系统数据模型

Order   — 限价委托单（买/卖）
Trade   — 成交记录
OHLCVBar — K 线柱
OrderBookSnapshot — 订单簿快照（供前端渲染）
PortfolioSnapshot — 持仓快照
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# ═══════════════════════════════════════════════════════
#  订单方向
# ═══════════════════════════════════════════════════════

class Side(str, Enum):
    BUY  = "buy"
    SELL = "sell"


# ═══════════════════════════════════════════════════════
#  限价订单
# ═══════════════════════════════════════════════════════

@dataclass
class Order:
    """一笔限价委托单。

    字段说明：
        agent_id    — 下单 Agent 的 unique_id
        agent_type  — Agent 类名（"InstTraderAgent" / "RetailTraderAgent" / "NormalAgent"）
        side        — 买 / 卖
        price       — 委托价（精确到分）
        quantity    — 委托股数
        remaining   — 剩余未成交股数（初始 = quantity）
        timestamp   — 时序编号（撮合时的单调递增计数器，用于 FIFO）
    """

    agent_id: int
    agent_type: str
    side: Side
    price: float
    quantity: int
    remaining: int = 0
    timestamp: int = 0
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])

    def __post_init__(self):
        if self.remaining == 0:
            self.remaining = self.quantity


# ═══════════════════════════════════════════════════════
#  成交记录
# ═══════════════════════════════════════════════════════

@dataclass
class Trade:
    """一笔成交。由撮合引擎生成。"""

    price: float
    quantity: int
    buy_order_id: str
    sell_order_id: str
    buyer_id: int
    seller_id: int
    timestamp: int = 0


# ═══════════════════════════════════════════════════════
#  K 线
# ═══════════════════════════════════════════════════════

@dataclass
class OHLCVBar:
    """单根 K 线柱（对应一个交易 session / tick）。"""

    tick: int               # 时间序号
    open: float
    high: float
    low: float
    close: float
    volume: int             # 成交量（股数）
    trade_count: int        # 成交笔数
    event: Optional[str] = None  # 触发事件标注


# ═══════════════════════════════════════════════════════
#  订单簿快照
# ═══════════════════════════════════════════════════════

@dataclass
class PriceLevel:
    """一个价位的聚合信息。"""
    price: float
    size: int           # 该价位总挂单量
    order_count: int    # 该价位委托笔数


@dataclass
class OrderBookSnapshot:
    """订单簿快照 —— 供前端渲染 depth chart / order book。"""

    bids: List[PriceLevel] = field(default_factory=list)   # 按价格降序
    asks: List[PriceLevel] = field(default_factory=list)   # 按价格升序
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return round(self.best_ask - self.best_bid, 2)
        return None

    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return round((self.best_ask + self.best_bid) / 2, 2)
        return None


# ═══════════════════════════════════════════════════════
#  持仓  
# ═══════════════════════════════════════════════════════

@dataclass
class Portfolio:
    """单个 Agent 的资金与持仓状态。"""

    cash: float            # 可用资金
    shares: int = 0        # 持股数量
    frozen_cash: float = 0.0   # 挂买单冻结的资金
    frozen_shares: int = 0     # 挂卖单冻结的股票

    @property
    def available_cash(self) -> float:
        return self.cash - self.frozen_cash

    @property
    def available_shares(self) -> int:
        return self.shares - self.frozen_shares

    def market_value(self, price: float) -> float:
        """持仓市值 + 可用资金。"""
        return self.cash + self.shares * price
