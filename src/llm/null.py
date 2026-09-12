"""
llm.null —— 「没有 LLM」的显式实现

原来"没配 Key"这件事表现为四处各自写 `if not api_key: return None`，
是一种散落的隐式状态。这里把它变成一个实现类：调用方永远拿到一个
LLMProvider，只是这一个什么都给不了。

好处是测试不用再 monkeypatch 模块全局变量，传这个对象就行。
"""

from __future__ import annotations

from typing import Any, List, Optional

from ..ports.llm import EmbeddingModel, StructuredModel


class NullLLMProvider:
    """满足 LLMProvider 契约，但永远返回 None。"""

    @property
    def is_available(self) -> bool:
        return False

    def structured(
        self, schema: type, *, temperature: Optional[float] = None
    ) -> Optional[StructuredModel]:
        return None

    def embeddings(self) -> Optional[EmbeddingModel]:
        return None

    def __repr__(self) -> str:
        return "NullLLMProvider()"
