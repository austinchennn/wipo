# 03 — 交易系统：限价订单簿连续竞价撮合引擎

> 完整实现 IPO 后的二级市场交易模拟，还原真实交易所的量价撮合逻辑。

---

## 背景

论坛讨论产出的 SentimentGrid（bull/bear/neutral）此前仅用于前端可视化，未对股价产生任何影响。前端 `mockData.js` 已预留 K 线、订单簿、成交日志等数据结构，后端缺失撮合引擎。

**需求：**
- 机构和散户各自独立决定是否交易、方向、数量、价格
- 机构初始资金 = 散户 × 100
- 按交易所机制自动撮合，驱动股价变化

---

## 架构总览

```
ForumModel._run_thread(post)
    │
    ├── Phase 1 → Phase 2 → Phase 3（舆论讨论）
    │                                   │
    │                          SentimentGrid（最新情绪）
    │                                   │
    │                                   ▼
    │                      TradingSession.run(all_agents)
    │                           │
    │                ┌──────────┴──────────┐
    │                ▼                     ▼
    │        decide_order()         decide_order()
    │        （每个 Agent）          （每个 Agent）
    │                │                     │
    │                └──────────┬──────────┘
    │                           ▼
    │                   Exchange.run_session(orders)
    │                           │
    │                ┌──────────┴──────────┐
    │                ▼                     ▼
    │         验证 + 冻结资金       OrderBook.submit()
    │                              （连续竞价撮合）
    │                                   │
    │                                   ▼
    │                         成交结算 → 更新持仓
    │                                   │
    │                                   ▼
    │                          生成 OHLCV K 线柱
    │
    ▼
  下一个 Thread...
```

每个宏观轮次 × 3 主题 = 3 次交易 Session → 12 轮 = **36 根 K 线柱**。

---

## 模块设计

### 文件结构

```
src/market/
    __init__.py           # 导出 Exchange, TradingSession
    models.py             # Order, Trade, OHLCVBar, Portfolio, OrderBookSnapshot
    order_book.py         # 限价订单簿（价格-时间优先撮合）
    exchange.py           # 交易所引擎（持仓管理 + K 线生成）
    trading_agent.py      # Agent 交易决策逻辑
```

### 模块职责

| 模块 | 职责 | 依赖 |
|---|---|---|
| `models.py` | 纯数据结构，无逻辑 | 无 |
| `order_book.py` | 撮合核心：价格优先-时间优先连续竞价 | `sortedcontainers.SortedList` |
| `exchange.py` | 持仓账本、资金冻结/结算、K 线生成 | `order_book` |
| `trading_agent.py` | Agent → Order 决策（方向/数量/价格） | `config`, Agent 属性 |

---

## 一、限价订单簿（OrderBook）

### 撮合规则

完全还原交易所的 **价格优先-时间优先** 连续竞价机制：

```
买方队列：按价格降序排列，同价按时间升序（先到先得）
卖方队列：按价格升序排列，同价按时间升序（先到先得）

新订单到达 → 立即尝试撮合：
  买单价格 ≥ 卖一价 → 成交（按挂单方价格 = maker price）
  卖单价格 ≤ 买一价 → 成交（按挂单方价格 = maker price）
  未完全成交部分 → 挂入订单簿等待
```

### 实现细节

使用 `sortedcontainers.SortedList` 维护两个有序队列：

```python
# 买方：(-price, timestamp) → 最高价最早排前
self._bids = SortedList(key=lambda o: (-o.price, o.timestamp))

# 卖方：(price, timestamp) → 最低价最早排前
self._asks = SortedList(key=lambda o: (o.price, o.timestamp))
```

**为什么用 SortedList 而非 heapq？**

| 特性 | SortedList | heapq |
|---|---|---|
| 插入 | O(log n) | O(log n) |
| 取最优 | O(1) | O(1) |
| 任意位置删除（撤单） | O(log n) | O(n) |
| 遍历导出快照 | O(n) 有序 | 需排序 O(n log n) |

撤单和快照导出是订单簿的高频操作，SortedList 更合适。

### 成交价规则

**Maker 价格原则**：成交价 = 已挂在订单簿上的那一方（maker）的委托价。

```
场景：卖一价 130.00，新买单 130.50
成交价 = 130.00（对买方更有利，符合真实交易所规则）

场景：买一价 131.00，新卖单 130.50
成交价 = 131.00（对卖方更有利）
```

---

## 二、交易所引擎（Exchange）

### 持仓与资金管理

每个 Agent 维护一个 `Portfolio`：

```python
@dataclass
class Portfolio:
    cash: float            # 可用资金
    shares: int            # 持股数量
    frozen_cash: float     # 买单冻结资金
    frozen_shares: int     # 卖单冻结股票
```

#### 初始配置

| Agent 类型 | 初始资金 | 初始持股 | 设计意图 |
|---|---|---|---|
| InstTraderAgent | $10,000,000 | 10,000 股 | 机构资金雄厚，IPO 配售 |
| RetailTraderAgent | $100,000 | 100 股 | 散户资金少，少量配售 |
| NormalAgent | $50,000 | 0 股 | 普通人无配售，需买入才能持有 |

**机构 : 散户 = 100 : 1**（资金倍数），符合真实市场的资金分布。

### 交易 Session 生命周期

```
run_session(orders, event):
    1. validate_and_freeze()    ← 检查资金/持仓，冻结对应数额
    2. random.shuffle(orders)   ← 模拟随机到达顺序
    3. for order in orders:
         order_book.submit()    ← 连续竞价撮合
         settle(trade)          ← 逐笔成交结算
    4. unfreeze_remaining()     ← 解冻未成交挂单
    5. build_bar()              ← 生成 OHLCV K 线柱
```

#### 资金冻结机制

```
买单提交前：frozen_cash += price × quantity
  → 确保不会超额买入（可用资金 = cash - frozen_cash）

卖单提交前：frozen_shares += quantity
  → 确保不会超额卖出（可用股票 = shares - frozen_shares）

成交时：
  买方：frozen_cash -= amount, cash -= amount, shares += qty
  卖方：frozen_shares -= qty, shares -= qty, cash += amount
```

### K 线生成

每次 Session 结束后从成交记录构建 OHLCV：

```python
@dataclass
class OHLCVBar:
    tick: int          # 时间序号
    open: float        # 第一笔成交价
    high: float        # 最高成交价
    low: float         # 最低成交价
    close: float       # 最后一笔成交价
    volume: int        # 成交量（股数）
    trade_count: int   # 成交笔数
    event: str | None  # 触发事件标注
```

无成交时平盘（O=H=L=C=上一个收盘价，V=0）。

---

## 三、Agent 交易决策

### 决策链

```
Agent 属性 + SentimentGrid + 市场状态
    │
    ├── 1. 参与度过滤
    │      NormalAgent: 85% 概率不交易
    │      Passive Agent: 60% 概率不交易
    │
    ├── 2. 方向决策（买/卖/观望）
    │      bull + 高 risk_tolerance → 买入概率 ↑
    │      bear + 高 risk_tolerance → 卖出概率 ↑
    │      neutral → 大概率观望
    │
    ├── 3. 数量决策
    │      base_lots × risk_factor × noise
    │      上限 = min(计算量, 可用资金/持仓)
    │
    └── 4. 价格决策（按 Agent 类型分层）
           机构：窄价差 ±0.1%~0.5%
           散户：宽价差 ±0.5%~3%，FOMO 追涨杀跌
           普通人：随机噪声 ±1%~5%
```

### 方向决策

```python
sentiment == "bull":
    P(buy)  = 0.5 + risk_tolerance × 0.5    # 高风险偏好者几乎必买
    P(skip) = 1 - P(buy)

sentiment == "bear":
    P(sell) = 0.5 + risk_tolerance × 0.5    # 高风险偏好者几乎必卖
    P(skip) = 1 - P(sell)

sentiment == "neutral":
    if risk_tolerance > 0.7: P(random_trade) = 0.2
    else: P(skip) = 1.0
```

### 数量决策

| Agent 类型 | 基准手数 | 典型单笔 | 占资金比例 |
|---|---|---|---|
| InstTraderAgent | 5,000 股（50 手） | 1,500~6,500 股 | ~10% |
| RetailTraderAgent | 500 股（5 手） | 200~650 股 | ~50% |
| NormalAgent | 200 股（2 手） | 100~260 股 | ~50% |

实际数量 = `base_lots × risk_factor(0.3~1.0) × noise(0.7~1.3)`，然后取整到 100 股（手），最后受可用资金/持仓上限约束。

### 价格决策 —— 三层定价策略

这是区分机构和散户行为最关键的设计：

#### 机构定价（InstTraderAgent）

```
特征：理性、耐心、窄价差
买入价 = last_price × (1.0 - uniform(0.001, 0.005))    # 低于市价 0.1%~0.5%
卖出价 = last_price × (1.0 + uniform(0.001, 0.005))    # 高于市价 0.1%~0.5%
```

机构是 **流动性提供者（maker）**：挂单贴近市价，等待对手方成交。价差窄意味着它们总能以接近公允价格成交。

#### 散户定价（RetailTraderAgent）

```
特征：情绪驱动、FOMO、追涨杀跌
base_spread = uniform(0.005, 0.02)                      # 基础 0.5%~2%
aggression  = 1.0 + fomo × 0.5 + emotional_volatility × 0.3

bull + FOMO > 0.5 → 买入价 = last_price × (1.0 + spread × 0.5)   # 追高！
bear + FOMO > 0.5 → 卖出价 = last_price × (1.0 - spread × 0.5)   # 杀跌！
```

散户是 **流动性消耗者（taker）**：情绪激动时愿意以更差的价格"抢着成交"。FOMO 高的散户会主动追涨杀跌，成为机构的对手方。

#### 普通人定价（NormalAgent）

```
特征：随机、不精准、"差不多就行"
spread = uniform(0.01, 0.05)                            # 1%~5% 偏移
价格 = last_price × (1.0 ± spread × uniform(0.2, 1.0))
```

普通人下单量最小、频率最低（仅 15% 参与率），对市场影响有限。

### 定价策略与撮合的交互效果

这种分层定价设计自然产生以下市场微观结构：

```
场景：bull sentiment 主导
  ┌─────────────────────────────────────┐
  │  散户（FOMO）: 买入价 133.00        │  ← taker，主动追高
  │  机构 A: 卖出价 130.65              │  ← maker，挂在卖方
  │  机构 B: 卖出价 130.50              │  ← maker
  │  成交：散户@130.50, 散户@130.65     │  ← 成交价=maker价格
  │  → 机构赚取 spread，散户追高买入    │
  └─────────────────────────────────────┘

场景：bear sentiment 主导
  ┌─────────────────────────────────────┐
  │  散户（恐慌）: 卖出价 126.00        │  ← taker，恐慌杀跌
  │  机构 A: 买入价 129.35              │  ← maker，低位挂买
  │  成交：散户@129.35                  │  ← 机构低吸，散户割肉
  └─────────────────────────────────────┘
```

**宏观效果：** 机构的窄价差挂单提供了流动性和价格锚定；散户的情绪化追涨杀跌驱动了价格波动。两者共同作用还原了真实市场的核心动力学。

---

## 四、与 ForumModel 的集成

### 时序

```
ForumModel.__init__() → Exchange() 创建 + 全员初始化 Portfolio

每个 Thread:
  Phase 1 → Phase 2 → Phase 3
       ↓
  最后一个 SentimentGrid（Phase 3）
       ↓
  TradingSession.run(all_agents, event="XX讨论结束")
       ↓
  Exchange.run_session(orders)
       ↓
  K 线更新，持仓变化
```

### 关键代码

```python
# forum.py — _run_thread() 末尾新增
async def _run_thread(self, post: Post):
    snap1 = await self._phase_1(post)
    snap2 = await self._phase_2(post, snap1)
    await self._phase_3(post, snap2)

    # ── 帖子讨论结束 → 触发交易撮合 ──
    latest_sg = self.sentiment_history[-1]
    all_agents = self.active_participants + self.passive_participants
    event = f"{self.TOPIC_CN[post.topic]}讨论结束"
    session = TradingSession(self.exchange, latest_sg.grid)
    session.run(all_agents, event=event)
```

### 运行结果示例

90 Agent（50 普通 + 10 机构 + 30 散户），1 轮 3 主题：

```
[撮合] tick=1 | 订单=14 有效=14 成交=3笔 | 价格=130.49 | 产品信息讨论结束
[撮合] tick=2 | 订单=10 有效=10 成交=2笔 | 价格=131.42 | 财务信息讨论结束
[撮合] tick=3 | 订单=10 有效=10 成交=1笔 | 价格=132.07 | 政策信息讨论结束

市场汇总 | 最新价=132.07 IPO价=130.00 涨跌=+1.59% 总成交=900股 K线=3根
```

---

## 五、配置参数

所有交易相关参数集中在 `src/config.py`：

```python
# ── 交易系统 ──
IPO_PRICE            = 130.0           # 发行价
TICK_SIZE            = 0.01            # 最小价格变动单位

# 初始资金（机构 : 散户 = 100 : 1）
INST_INITIAL_CASH    = 10_000_000.0    # $10M
RETAIL_INITIAL_CASH  = 100_000.0       # $100K
NORMAL_INITIAL_CASH  = 50_000.0        # $50K

# 初始持股（IPO 配售）
INST_INITIAL_SHARES  = 10_000          # 机构配售
RETAIL_INITIAL_SHARES = 100            # 散户配售
NORMAL_INITIAL_SHARES = 0             # 普通人无配售

# 下单基准手数
INST_ORDER_LOTS      = 5_000           # 50 手
RETAIL_ORDER_LOTS    = 500             # 5 手
NORMAL_ORDER_LOTS    = 200             # 2 手

# 参与率
NORMAL_TRADE_PROBABILITY = 0.15        # 普通人每次交易概率
PASSIVE_TRADE_DISCOUNT   = 0.40        # Passive Agent 参与率
```

---

## 六、为什么不用"情绪 → 股价"简单映射

最初的备选方案是用公式直接把情绪比例映射成价格：

```python
# 直接映射（已否决）
bull_ratio = n_bull / n_total
price_change = (bull_ratio - 0.5) * volatility_factor
new_price = last_price * (1 + price_change)
```

**否决原因：**

1. **丧失微观结构**：真实股价由订单流驱动，不是由投票产生。同样 60% 看涨但区别在于：3 个机构同时挂大单 vs 100 个散户各挂小单，市场影响完全不同。

2. **机构权重问题无法优雅解决**：简单映射需要人为设定"机构权重 = 散户 × N"，但实际上机构的影响力体现在下单量和价格策略，不是投票权。限价订单簿自然解决了这个问题——机构单笔 5,000 股 vs 散户 500 股，对订单簿的冲击天然不同。

3. **无法产生真实的价格序列**：直接映射产生的价格缺乏波动的微观结构（bid-ask spread、成交量集中在特定价位、大单扫单导致的跳价等），K 线看起来像噪声而非市场。

4. **失去散户追涨杀跌的涌现行为**：限价订单簿 + 分层定价策略自然产生"散户追高 → 机构出货"或"散户恐慌抛售 → 机构低吸"的真实市场动力学。

---

## 七、文件变更清单

```
src/market/__init__.py        ← 新增：模块入口
src/market/models.py          ← 新增：Order, Trade, OHLCVBar, Portfolio, OrderBookSnapshot
src/market/order_book.py      ← 新增：限价订单簿（SortedList, 价格-时间优先）
src/market/exchange.py        ← 新增：交易所引擎（持仓管理 + K 线生成）
src/market/trading_agent.py   ← 新增：Agent 交易决策（方向/数量/价格）
src/config.py                 ← 新增 15 个交易参数
src/environment/forum.py      ← 集成：Thread 结束后触发交易 Session
requirements.txt              ← 新增 sortedcontainers>=2.4
```
