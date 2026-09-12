"""
ports.spider —— 政策条目来源的抽象

实现可以是真爬虫（PolicySpider）、本地 fixture、或测试替身。
ingest_graph 只认这个契约，不认具体类。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..spiders.models import RawPolicy


@runtime_checkable
class PolicyFeed(Protocol):
    """一次拉取，返回未清洗的政策条目。"""

    async def afetch_all(self) -> List["RawPolicy"]: ...
