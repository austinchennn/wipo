"""
全局配置常量 —— 代替散落在各模块的魔术数字。

修改任何参数时只需改这一处，所有模块自动生效。
"""

# ── LLM 模型 ──
DEFAULT_LLM_MODEL: str = "gemini-2.5-flash"
DEFAULT_EMBEDDING_MODEL: str = "models/embedding-001"

# ── LLM Temperature ──
AGENT_LLM_TEMPERATURE: float = 0.8    # 评论生成：多样性优先
EXTRACTOR_LLM_TEMPERATURE: float = 0.0  # 结构化提取：确定性优先

# ── 论坛调度 ──
MAX_CONCURRENT_LLM_CALLS: int = 50    # asyncio Semaphore 上限，防 rate limit

# ── PDF 分块 ──
CHUNK_SIZE: int = 1500
CHUNK_OVERLAP: int = 200
MAX_CONTEXT_CHARS: int = 12_000        # chunks_to_context 拼合上限

# ── RAG 检索 ──
RAG_QUERY_K: int = 4     # KnowledgeBase.query / query_text 默认 top-k
RAG_RETRIEVE_K: int = 3  # RAGSystem.retrieve_for_agent 默认 top-k

# ── 交易系统 ──
IPO_PRICE: float = 130.0          # IPO 发行价（模拟起始价）
TICK_SIZE: float = 0.01           # 最小价格变动单位（1 分）

# 初始资金：机构 = 散户 × 100
INST_INITIAL_CASH: float = 10_000_000.0    # 机构初始资金 $10M
RETAIL_INITIAL_CASH: float = 100_000.0     # 散户初始资金 $100K
NORMAL_INITIAL_CASH: float = 50_000.0      # 普通人初始资金 $50K

# 初始持仓（股）：IPO 配售
INST_INITIAL_SHARES: int = 10_000          # 机构配售 10,000 股
RETAIL_INITIAL_SHARES: int = 100           # 散户配售 100 股
NORMAL_INITIAL_SHARES: int = 0             # 普通人无配售

# 每次下单基准手数（100 股/手）
INST_ORDER_LOTS: int = 5000      # 机构基准 50 手 = 5000 股
RETAIL_ORDER_LOTS: int = 500     # 散户基准 5 手 = 500 股
NORMAL_ORDER_LOTS: int = 200     # 普通人基准 2 手 = 200 股

# 交易参与率
NORMAL_TRADE_PROBABILITY: float = 0.15    # 普通人下单概率 15%
PASSIVE_TRADE_DISCOUNT: float = 0.40      # Passive Agent 下单概率 40%
