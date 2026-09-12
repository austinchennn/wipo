"""
persistence — 模拟数据持久化（SQLite）

支持：
  - 模拟结束后一次性写入全量数据
  - 按时间戳顺序回放评论流
  - 查询任意快照（情绪矩阵、K 线、持仓）
  - 外部政策缓存（PolicyStore，跨模拟复用，1 周 TTL）
"""

from .database import SimulationDB, replay_comments
from .policy_store import PolicyStore

__all__ = ["SimulationDB", "replay_comments", "PolicyStore"]
