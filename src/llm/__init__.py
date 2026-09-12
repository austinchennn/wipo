"""
llm —— LLMProvider 的具体实现

    GeminiProvider    langchain-google-genai
    NullLLMProvider   没有可用后端时的显式实现（Agent 走 mock，分类走规则）

选哪个由 composition root 决定，上层模块只见 ports.LLMProvider。
"""

from .gemini import GeminiProvider
from .null import NullLLMProvider

__all__ = ["GeminiProvider", "NullLLMProvider"]
