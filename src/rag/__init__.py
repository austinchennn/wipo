"""rag — 三个 LangChain FAISS 知识库 + Agent 访问控制。

三个知识库（KnowledgeBase）：
    product_kb   → 产品/业务章节 chunks
    financial_kb → 财务/MD&A 章节 chunks
    risk_kb      → 风险/政策章节 chunks

RAGSystem：
    - 持有三个 KB
    - retrieve_for_agent(agent, topic, query) — 按可见性规则返回检索结果
    - build_from_extraction(result)           — 从 ExtractionResult 批量构建

快速使用：
    from src.rag import RAGSystem
    from src.extractors import make_mock_extraction

    result = make_mock_extraction()
    rag = RAGSystem.build_from_extraction(result)

    # Agent 查询（access control 自动执行）
    context = rag.retrieve_for_agent(agent, topic="product", query="竞争对手")
"""

from .knowledge_base import KnowledgeBase, RAGSystem
from .access_control import VISIBILITY_MATRIX, AccessLevel, get_access_level

__all__ = [
    "KnowledgeBase",
    "RAGSystem",
    "VISIBILITY_MATRIX",
    "AccessLevel",
    "get_access_level",
]
