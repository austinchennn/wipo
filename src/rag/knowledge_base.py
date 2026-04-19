"""
knowledge_base — 三个 LangChain FAISS 知识库 + RAGSystem 门面。

KnowledgeBase：
    - 单个 FAISS 向量存储（对应一个 topic）
    - build(chunks, embeddings) → 构建索引
    - query(question, k) → 返回最相关的 Document 列表

RAGSystem：
    - 持有 product_kb / financial_kb / risk_kb 三个 KB
    - retrieve_for_agent(agent, topic, query) — 按 AccessLevel 控制返回内容
    - build_from_extraction(result) — 从 ExtractionResult 一键构建
    - get_static_section(topic, agent_type) — 直接返回 Extractor 生成的静态文本
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from ..config import (
    DEFAULT_EMBEDDING_MODEL,
    RAG_QUERY_K,
    RAG_RETRIEVE_K,
)
from .access_control import (
    AccessLevel,
    apply_access_control,
    get_access_level,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..extractors.pipeline import ExtractionResult
    from ..agents.base_agent import BaseUserAgent


# ─────────────────────────────────────────────────────────────────
#  KnowledgeBase — 单 topic FAISS 知识库
# ─────────────────────────────────────────────────────────────────

class KnowledgeBase:
    """单个话题的 FAISS 向量知识库。"""

    def __init__(self, topic: str):
        self.topic = topic
        self._vectorstore: Optional[FAISS] = None

    # ── 构建 ──

    @classmethod
    def build(
        cls,
        topic: str,
        chunks: List[Document],
        embeddings: GoogleGenerativeAIEmbeddings,
    ) -> "KnowledgeBase":
        """从 Document 列表构建 FAISS 索引。"""
        kb = cls(topic=topic)
        if not chunks:
            return kb
        kb._vectorstore = FAISS.from_documents(chunks, embeddings)
        return kb

    @classmethod
    def build_empty(cls, topic: str) -> "KnowledgeBase":
        """创建空 KB（无 PDF chunks 时使用，RAG 查询返回空列表）。"""
        return cls(topic=topic)

    # ── 检索 ──

    def query(self, question: str, k: int = RAG_QUERY_K) -> List[Document]:
        """语义检索，返回最相关的 k 个 Document。"""
        if self._vectorstore is None:
            return []
        return self._vectorstore.similarity_search(question, k=k)

    def query_text(self, question: str, k: int = RAG_QUERY_K) -> str:
        """检索后拼合为纯文本字符串。"""
        docs = self.query(question, k=k)
        if not docs:
            return ""
        parts = []
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get("page", "?")
            parts.append(f"[检索段落{i}·第{page}页]\n{doc.page_content.strip()}")
        return "\n\n".join(parts)

    @property
    def is_ready(self) -> bool:
        return self._vectorstore is not None


# ─────────────────────────────────────────────────────────────────
#  RAGSystem — 三库统一门面 + 访问控制
# ─────────────────────────────────────────────────────────────────

class RAGSystem:
    """
    管理三个 KnowledgeBase，并根据 Agent 类型强制执行访问控制。

    内部持有的静态摘要（extraction_result）用于快速返回预先提取的
    Level A/B/C 文本，避免每次都做向量检索。

    两种查询模式：
      1. get_static_section(topic, agent_type)
         → 直接返回 Extractor 生成的静态分层文本（主帖发布用）
      2. retrieve_for_agent(agent, topic, query)
         → 做 FAISS 检索后，根据访问级别处理返回文本（Agent 评论时用）
    """

    def __init__(
        self,
        product_kb: KnowledgeBase,
        financial_kb: KnowledgeBase,
        risk_kb: KnowledgeBase,
        extraction_result: Optional["ExtractionResult"] = None,
    ):
        self._kbs: Dict[str, KnowledgeBase] = {
            "product":   product_kb,
            "financial": financial_kb,
            "policy":    risk_kb,
        }
        self._extraction = extraction_result

    # ── 构建 ──

    @classmethod
    def build_from_extraction(
        cls,
        result: "ExtractionResult",
        model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> "RAGSystem":
        """从 ExtractionResult 一键构建三个 KB。

        需要 GOOGLE_API_KEY 环境变量。
        若 chunks 为空（如 make_mock_extraction），返回空 KB（静态模式仍可用）。
        """
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "RAGSystem 需要 GOOGLE_API_KEY 以构建向量索引。"
            )

        embeddings = GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)

        def _build(topic: str, chunks: List[Document]) -> KnowledgeBase:
            if chunks:
                logger.info("[RAG] 构建 %s KB (%d chunks)...", topic, len(chunks))
                return KnowledgeBase.build(topic, chunks, embeddings)
            logger.info("[RAG] %s KB 无 chunks，使用空索引（静态模式）", topic)
            return KnowledgeBase.build_empty(topic)

        return cls(
            product_kb=_build("product", result.product_chunks),
            financial_kb=_build("financial", result.financial_chunks),
            risk_kb=_build("policy", result.risk_chunks),
            extraction_result=result,
        )

    @classmethod
    def build_static_only(
        cls, result: "ExtractionResult"
    ) -> "RAGSystem":
        """构建纯静态 RAGSystem（不构建 FAISS 索引，无需 API Key）。

        适用于：无 PDF chunks（make_mock_extraction）或只需要静态文本的场景。
        retrieve_for_agent 在此模式下退化为 get_static_section。
        """
        return cls(
            product_kb=KnowledgeBase.build_empty("product"),
            financial_kb=KnowledgeBase.build_empty("financial"),
            risk_kb=KnowledgeBase.build_empty("policy"),
            extraction_result=result,
        )

    # ── 模式 1：静态分层文本（主帖发布） ──

    def get_static_section(self, topic: str, agent_type: str) -> str:
        """
        返回 Extractor 预先生成的分层静态文本。

        用于 ForumModel 替换原来的 raw_sections 查询，
        也用于 Agent 在没有具体 query 时获取话题背景。
        """
        if self._extraction is not None:
            return self._extraction.get_section_for_agent(topic, agent_type)

        # 无 extraction_result 时降级到 HIDDEN
        return "[此信息对你不可见]"

    # ── 模式 2：RAG 检索（Agent 评论时） ──

    def retrieve_for_agent(
        self,
        agent: "BaseUserAgent",
        topic: str,
        query: str,
        k: int = RAG_RETRIEVE_K,
    ) -> str:
        """
        根据 agent 类型和话题执行 FAISS 检索，并强制应用访问控制。

        返回值直接可追加到 Agent 的 comment context 中。
        """
        agent_type = agent.__class__.__name__
        level: AccessLevel = get_access_level(agent_type, topic)

        # HIDDEN → 直接拒绝，不进行检索
        if level == AccessLevel.HIDDEN:
            return "[此信息对你不可见]"

        kb = self._kbs.get(topic)
        if kb is None or not kb.is_ready:
            # KB 未就绪时降级为静态文本
            return self.get_static_section(topic, agent_type)

        # 执行向量检索
        raw_text = kb.query_text(query, k=k)
        if not raw_text:
            return self.get_static_section(topic, agent_type)

        # 应用访问控制变换
        return apply_access_control(raw_text, level)

    # ── 工具方法 ──

    def get_raw_sections_for_host(self) -> Dict[str, str]:
        """返回主持人（HostAgent）视角的完整三段文本，用于生成主贴。"""
        if self._extraction is not None:
            return self._extraction.raw_sections
        return {
            "product":   "[产品信息未加载]",
            "financial": "[财务信息未加载]",
            "policy":    "[政策信息未加载]",
        }

    def status(self) -> str:
        """返回三个 KB 的就绪状态字符串。"""
        lines = ["RAGSystem 状态:"]
        for topic, kb in self._kbs.items():
            state = "就绪 ✓" if kb.is_ready else "空索引（静态模式）"
            lines.append(f"  {topic:12s} → {state}")
        return "\n".join(lines)
