"""
persistence — 模拟数据持久化（SQLite）

支持：
  - 模拟结束后一次性写入全量数据
  - 按时间戳顺序回放评论流
  - 查询任意快照（情绪矩阵、K 线、持仓）
"""

from .database import SimulationDB, replay_comments

__all__ = ["SimulationDB", "replay_comments"]
