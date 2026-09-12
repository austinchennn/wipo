"""
ports.persistence —— 落库目标的抽象

修正了原来反向的依赖方向：persistence.database 曾经 import ForumModel
做 isinstance 检查，等于底层依赖上层。现在两边都只依赖这里的 Protocol。
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, List, Optional, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:  # 只用于类型标注，运行时无依赖
    from ..spiders.models import CleanedPolicy


@runtime_checkable
class SimulationSink(Protocol):
    """模拟结束后的落库目标（SQLite、内存、/dev/null）。"""

    def save(self, model: Any) -> int:
        """保存整场模拟，返回 simulation_id。"""
        ...

    def close(self) -> None: ...


@runtime_checkable
class PolicyCache(Protocol):
    """外部政策缓存 —— TTL 判断和读写。"""

    def is_fresh(self, ttl_days: int) -> bool: ...

    def load_fresh(self, ttl_days: int) -> List["CleanedPolicy"]: ...

    def save(self, policies: Sequence["CleanedPolicy"]) -> int: ...

    def latest_fetched_at(self) -> Optional[datetime]: ...

    def count(self) -> int: ...

    def close(self) -> None: ...
