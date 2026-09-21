"""
settings —— 运行时配置的唯一来源

原来 os.environ.get("GOOGLE_API_KEY") 散在四个文件里
（base_agent / classifier / knowledge_base / base_extractor），
每处各自判断"有没有 Key"，api/main.py 为了换 Key 只能写
os.environ[...] = key 去改全进程状态。

现在读环境变量这件事只发生在 Settings.from_env() 一处，
其余模块拿到的是一个不可变对象。

config.py 放的是不随环境变化的领域常量（TTL、chunk 大小、IPO 价格），
settings.py 放的是随部署环境变化的东西（Key、模型名）。两者不重叠。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Optional

from .config import (
    AGENT_LLM_TEMPERATURE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
)


@dataclass(frozen=True)
class Settings:
    """一次运行的全部环境相关配置。不可变——要改用 with_api_key()。"""

    google_api_key: Optional[str] = None
    # Jev（langchain-typesafe）的 Key。SDK 自己从 TYPESAFE_API_KEY 读，
    # 这里只用来判断"配没配"，决定 composition 要不要装配 Jev
    typesafe_api_key: Optional[str] = None
    chat_model: str = DEFAULT_LLM_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    agent_temperature: float = AGENT_LLM_TEMPERATURE
    simulation_db_path: str = "output/simulation.db"
    policy_db_path: str = "output/policies.db"

    @classmethod
    def from_env(cls, load_dotenv: bool = True) -> "Settings":
        """从环境变量装配。整个项目里唯一读 os.environ 的地方。"""
        if load_dotenv:
            try:
                from dotenv import load_dotenv as _load
                _load()
            except ImportError:
                pass

        return cls(
            google_api_key=os.environ.get("GOOGLE_API_KEY") or None,
            typesafe_api_key=os.environ.get("TYPESAFE_API_KEY") or None,
            chat_model=os.environ.get("AGENT_LLM_MODEL", DEFAULT_LLM_MODEL),
            embedding_model=os.environ.get(
                "EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
            ),
        )

    def with_api_key(self, api_key: Optional[str]) -> "Settings":
        """返回换了 Key 的新实例。

        给 API 层用：前端传进来的 Key 只影响这一次模拟，
        不再需要写 os.environ 污染整个进程。
        """
        if not api_key:
            return self
        return replace(self, google_api_key=api_key)

    def with_model(self, chat_model: Optional[str]) -> "Settings":
        """返回换了模型名的新实例（CLI --model 用）。"""
        if not chat_model:
            return self
        return replace(self, chat_model=chat_model)

    @property
    def has_llm(self) -> bool:
        return bool(self.google_api_key)

    @property
    def has_decision_model(self) -> bool:
        return bool(self.typesafe_api_key)
