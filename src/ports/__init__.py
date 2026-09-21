"""
ports —— 抽象层（依赖反转的边界）

这个包只有 Protocol 定义，没有任何实现，也不 import 任何第三方库
（LangChain / httpx / sqlite3 一个都没有）。所有依赖箭头都指向这里。

    上层                     ports                 具体实现
    ─────────────────────────────────────────────────────────
    BaseUserAgent      →  LLMProvider        ←  GeminiProvider / NullLLMProvider
    TradingSession     →  DecisionModel      ←  TypeSafeDecisionModel / NullDecisionModel
    classifier         →  LLMProvider        ←      同上
    ForumModel         →  SimulationSink     ←  SimulationDB
    ForumModel         →  KnowledgeProvider  ←  RAGSystem
    ingest_graph       →  PolicyCache        ←  PolicyStore
    ingest_graph       →  PolicyFeed         ←  PolicySpider / FakeFeed

装配发生在唯一的 composition root（src/composition.py），
只有 run.py 和 api/main.py 会调它。
"""

from .decision import Assessment, DecisionModel
from .knowledge import KnowledgeProvider
from .llm import EmbeddingModel, LLMProvider, StructuredModel
from .persistence import PolicyCache, SimulationSink
from .spider import PolicyFeed

__all__ = [
    "LLMProvider",
    "StructuredModel",
    "EmbeddingModel",
    "DecisionModel",
    "Assessment",
    "SimulationSink",
    "PolicyCache",
    "PolicyFeed",
    "KnowledgeProvider",
]
