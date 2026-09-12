"""
ports.llm —— LLM 能力的抽象

上层（Agent、Extractor、Classifier、RAG）只依赖这里的 Protocol，
不知道背后是 Gemini、OpenAI 还是测试替身。

为什么用 Protocol 而不是 ABC：
  - LangChain 的 Runnable / Embeddings 是第三方类型，没法让它们继承我们的 ABC，
    但它们天然满足这里的结构化契约
  - 测试替身不用继承、不用注册，写个有对应方法的类就行

「没有可用后端」不是异常，而是一种实现（NullLLMProvider）：
structured() / embeddings() 返回 None，调用方据此走 mock 或规则兜底。
"""

from __future__ import annotations

from typing import Any, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class StructuredModel(Protocol):
    """已绑定输出 schema 的对话模型（对应 llm.with_structured_output(X)）。

    同步异步都要有：Extractor 在线程池里走 invoke()，
    Agent 和 Classifier 在事件循环里走 ainvoke()。
    """

    def invoke(self, messages: Any) -> Any:
        """返回 schema 对应的 pydantic 实例。"""
        ...

    async def ainvoke(self, messages: Any) -> Any:
        """invoke 的异步版。"""
        ...


@runtime_checkable
class EmbeddingModel(Protocol):
    """向量化模型 —— FAISS 只需要这两个方法。"""

    def embed_documents(self, texts: List[str]) -> List[List[float]]: ...

    def embed_query(self, text: str) -> List[float]: ...


@runtime_checkable
class LLMProvider(Protocol):
    """LLM 能力的来源。

    实现方负责持有 API Key、模型名、连接复用等一切具体细节；
    调用方只问「给我一个绑定了这个 schema 的模型」。
    """

    @property
    def is_available(self) -> bool:
        """后端是否可用（无 Key / 未配置时为 False）。"""
        ...

    def structured(
        self, schema: type, *, temperature: Optional[float] = None
    ) -> Optional[StructuredModel]:
        """返回绑定 schema 的结构化模型；不可用时返回 None。"""
        ...

    def embeddings(self) -> Optional[EmbeddingModel]:
        """返回向量化模型；不可用时返回 None。"""
        ...
