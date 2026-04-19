"""
order_book — 连续竞价限价订单簿（价格-时间优先撮合）

实现现实交易所的核心撮合逻辑：
  1. 买单按 价格降序 → 同价按 时间升序（FIFO）
  2. 卖单按 价格升序 → 同价按 时间升序（FIFO）
  3. 新订单到达时立即尝试撮合：
     - 买单价格 ≥ 卖一价 → 成交（按挂单方价格）
     - 卖单价格 ≤ 买一价 → 成交（按挂单方价格）
  4. 未完全成交的部分挂入订单簿等待
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import List, Optional

from sortedcontainers import SortedList

from .models import Order, OrderBookSnapshot, PriceLevel, Side, Trade

logger = logging.getLogger(__name__)


class OrderBook:
    """连续竞价限价订单簿 —— 价格优先、时间优先。"""

    def __init__(self):
        # 买单：按 (-price, timestamp) 排序 → 最高价最早的排前面
        self._bids: SortedList[Order] = SortedList(
            key=lambda o: (-o.price, o.timestamp)
        )
        # 卖单：按 (price, timestamp) 排序 → 最低价最早的排前面
        self._asks: SortedList[Order] = SortedList(
            key=lambda o: (o.price, o.timestamp)
        )
        self._ts_counter: int = 0     # 单调递增时间戳
        self._trades: List[Trade] = []  # 本轮成交记录
        self._last_price: Optional[float] = None

    # ── 属性 ──

    @property
    def best_bid(self) -> Optional[float]:
        return self._bids[0].price if self._bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self._asks[0].price if self._asks else None

    @property
    def last_price(self) -> Optional[float]:
        return self._last_price

    @property
    def trades(self) -> List[Trade]:
        return self._trades

    # ── 核心：提交订单 + 即时撮合 ──

    def submit(self, order: Order) -> List[Trade]:
        """提交一笔限价单，立即尝试撮合。返回本次产生的成交列表。"""
        self._ts_counter += 1
        order.timestamp = self._ts_counter

        new_trades: List[Trade] = []

        if order.side is Side.BUY:
            new_trades = self._match_buy(order)
            if order.remaining > 0:
                self._bids.add(order)
        else:
            new_trades = self._match_sell(order)
            if order.remaining > 0:
                self._asks.add(order)

        self._trades.extend(new_trades)
        return new_trades

    def _match_buy(self, buy: Order) -> List[Trade]:
        """买单 vs 卖方队列撮合。"""
        trades: List[Trade] = []
        while buy.remaining > 0 and self._asks:
            best_sell = self._asks[0]
            if buy.price < best_sell.price:
                break  # 价格不够

            # 成交价 = 挂单方（maker）价格
            fill_price = best_sell.price
            fill_qty = min(buy.remaining, best_sell.remaining)

            trade = Trade(
                price=fill_price,
                quantity=fill_qty,
                buy_order_id=buy.id,
                sell_order_id=best_sell.id,
                buyer_id=buy.agent_id,
                seller_id=best_sell.agent_id,
                timestamp=self._ts_counter,
            )
            trades.append(trade)

            buy.remaining -= fill_qty
            best_sell.remaining -= fill_qty
            self._last_price = fill_price

            if best_sell.remaining == 0:
                self._asks.pop(0)

        return trades

    def _match_sell(self, sell: Order) -> List[Trade]:
        """卖单 vs 买方队列撮合。"""
        trades: List[Trade] = []
        while sell.remaining > 0 and self._bids:
            best_buy = self._bids[0]
            if sell.price > best_buy.price:
                break

            fill_price = best_buy.price
            fill_qty = min(sell.remaining, best_buy.remaining)

            trade = Trade(
                price=fill_price,
                quantity=fill_qty,
                buy_order_id=best_buy.id,
                sell_order_id=sell.id,
                buyer_id=best_buy.agent_id,
                seller_id=sell.agent_id,
                timestamp=self._ts_counter,
            )
            trades.append(trade)

            sell.remaining -= fill_qty
            best_buy.remaining -= fill_qty
            self._last_price = fill_price

            if best_buy.remaining == 0:
                self._bids.pop(0)

        return trades

    # ── 撤单 ──

    def cancel(self, order_id: str) -> Optional[Order]:
        """撤销尚未完全成交的挂单。返回被撤的 Order 或 None。"""
        for book in (self._bids, self._asks):
            for i, o in enumerate(book):
                if o.id == order_id:
                    book.pop(i)
                    return o
        return None

    # ── 清空队列中的残余挂单（Round 结束时） ──

    def clear_all(self) -> None:
        """清空全部挂单（新一天开盘前调用）。"""
        self._bids.clear()
        self._asks.clear()

    def drain_trades(self) -> List[Trade]:
        """取出并清空本轮积累的成交记录。"""
        t = list(self._trades)
        self._trades.clear()
        return t

    # ── 快照 ──

    def snapshot(self, depth: int = 15) -> OrderBookSnapshot:
        """生成 N 档深度的订单簿快照。"""
        bids: List[PriceLevel] = []
        asks: List[PriceLevel] = []

        # 聚合买方
        bid_levels: dict[float, list[int]] = defaultdict(list)
        for o in self._bids:
            bid_levels[o.price].append(o.remaining)
        for price in sorted(bid_levels, reverse=True)[:depth]:
            sizes = bid_levels[price]
            bids.append(PriceLevel(price=price, size=sum(sizes), order_count=len(sizes)))

        # 聚合卖方
        ask_levels: dict[float, list[int]] = defaultdict(list)
        for o in self._asks:
            ask_levels[o.price].append(o.remaining)
        for price in sorted(ask_levels)[:depth]:
            sizes = ask_levels[price]
            asks.append(PriceLevel(price=price, size=sum(sizes), order_count=len(sizes)))

        return OrderBookSnapshot(
            bids=bids,
            asks=asks,
            best_bid=self.best_bid,
            best_ask=self.best_ask,
        )
