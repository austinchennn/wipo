"""
llm.gemini —— LLMProvider 的 Gemini 实现

原来 base_agent 和 classifier 各自维护一个 `global _llm_instance` 懒加载
单例。复用客户端本身是对的（省掉重复建连），错的是把它放在模块全局：
一个进程里只能有一套配置，测试只能靠 monkeypatch 模块变量来打桩。

这里把同样的缓存挪进实例：一个 Provider 对象缓存自己的模型，
两个 Provider（不同 Key / 不同模型）可以在同一进程里并存。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from ..ports.llm import EmbeddingModel, StructuredModel
from ..settings import Settings

logger = logging.getLogger(__name__)


class GeminiProvider:
    """基于 langchain-google-genai 的 LLMProvider 实现。

    langchain 的 import 推迟到真正要建模型时，这样没配 Key 的环境
    （比如跑测试）根本不会加载这些重依赖。
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        # 键含 temperature：Agent 用 0.8、Extractor 和 Classifier 用 0.0，
        # 是两个不同的模型实例，不能共用一个缓存位。
        self._structured_cache: Dict[Tuple[Any, Optional[float]], Any] = {}
        self._embeddings: Optional[Any] = None

    @property
    def is_available(self) -> bool:
        return bool(self._settings.google_api_key)

    def structured(
        self, schema: type, *, temperature: Optional[float] = None
    ) -> Optional[StructuredModel]:
        if not self.is_available:
            return None

        key = (schema, temperature)
        if key not in self._structured_cache:
            from langchain_google_genai import ChatGoogleGenerativeAI

            temp = (
                temperature if temperature is not None
                else self._settings.agent_temperature
            )
            llm = ChatGoogleGenerativeAI(
                model=self._settings.chat_model,
                temperature=temp,
                google_api_key=self._settings.google_api_key,
            )
            self._structured_cache[key] = llm.with_structured_output(schema)
            logger.debug(
                "[LLM] 新建结构化模型 schema=%s temperature=%s",
                getattr(schema, "__name__", schema), temp,
            )

        return self._structured_cache[key]

    def embeddings(self) -> Optional[EmbeddingModel]:
        if not self.is_available:
            return None

        if self._embeddings is None:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            self._embeddings = GoogleGenerativeAIEmbeddings(
                model=self._settings.embedding_model,
                google_api_key=self._settings.google_api_key,
            )
        return self._embeddings

    def __repr__(self) -> str:
        return f"GeminiProvider(model={self._settings.chat_model!r})"
