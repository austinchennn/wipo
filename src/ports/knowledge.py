"""
ports.knowledge —— ForumModel 对知识库的最小需求面

原来 ForumModel.get_agent_context() 直接摸 rag_system._kbs[topic].is_ready
这个私有属性，跨模块访问私有状态。这里把它提升成契约的一部分（is_ready），
ForumModel 就不用再知道 RAGSystem 内部长什么样。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Protocol, runtime_checkable

if TYPE_CHECKING:
    from langchain_core.documents import Document


@runtime_checkable
class KnowledgeProvider(Protocol):
    """分层知识库 —— 按 Agent 类型返回其有权看到的内容。"""

    def is_ready(self, topic: str) -> bool:
        """该 topic 的向量索引是否可用（否则调用方走静态文本）。"""
        ...

    def get_static_section(self, topic: str, agent_type: str) -> str: ...

    def retrieve_for_agent(
        self, agent: Any, topic: str, query: str, k: int = 3
    ) -> str: ...

    def add_policy_chunks(self, chunks: List["Document"]) -> int:
        """并入外部政策 chunks，返回接收条数。"""
        ...

    def get_raw_sections_for_host(self) -> dict: ...
