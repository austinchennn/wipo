"""
persistence.database — SQLite 持久化引擎

设计原则：
  - 模拟过程中 **不写磁盘**，所有数据保持内存态
  - 模拟结束后调用 save() **一次性批量写入**（WAL 模式 + 事务）
  - 回放时只读查询，按 created_at 排序返回评论流

表结构：
  simulations   — 模拟元数据（参数、时间）
  posts         — 帖子
  comments      — 评论（含时间戳，支持按 created_at 回放）
  sentiments    — 情绪快照（JSON grid）
  trades        — 成交记录
  klines        — K 线
  portfolios    — 最终持仓
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

# ── 建表 SQL ──

_SCHEMA = """
CREATE TABLE IF NOT EXISTS simulations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    seed INTEGER,
    n_normal INTEGER NOT NULL,
    n_inst INTEGER NOT NULL,
    n_retail INTEGER NOT NULL,
    n_active INTEGER,
    llm_model TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT
);

CREATE TABLE IF NOT EXISTS posts (
    id TEXT NOT NULL,
    simulation_id INTEGER NOT NULL,
    round_num INTEGER NOT NULL,
    topic TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT,
    PRIMARY KEY (simulation_id, id),
    FOREIGN KEY (simulation_id) REFERENCES simulations(id)
);

CREATE TABLE IF NOT EXISTS comments (
    id TEXT NOT NULL,
    simulation_id INTEGER NOT NULL,
    post_id TEXT NOT NULL,
    author_id INTEGER NOT NULL,
    author_type TEXT NOT NULL,
    author_name TEXT NOT NULL,
    content TEXT NOT NULL,
    temp REAL NOT NULL,
    sentiment TEXT,
    phase INTEGER NOT NULL,
    parent_id TEXT,
    created_at TEXT,
    PRIMARY KEY (simulation_id, id),
    FOREIGN KEY (simulation_id) REFERENCES simulations(id),
    FOREIGN KEY (simulation_id, post_id) REFERENCES posts(simulation_id, id)
);

CREATE INDEX IF NOT EXISTS idx_comments_created_at
    ON comments(simulation_id, created_at);

CREATE INDEX IF NOT EXISTS idx_comments_post
    ON comments(simulation_id, post_id, phase);

CREATE TABLE IF NOT EXISTS sentiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    simulation_id INTEGER NOT NULL,
    round_num INTEGER NOT NULL,
    topic TEXT NOT NULL,
    phase INTEGER NOT NULL,
    grid_json TEXT NOT NULL,
    FOREIGN KEY (simulation_id) REFERENCES simulations(id)
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    simulation_id INTEGER NOT NULL,
    tick INTEGER NOT NULL,
    price REAL NOT NULL,
    quantity INTEGER NOT NULL,
    buy_order_id TEXT,
    sell_order_id TEXT,
    buyer_id INTEGER NOT NULL,
    seller_id INTEGER NOT NULL,
    match_timestamp INTEGER NOT NULL,
    FOREIGN KEY (simulation_id) REFERENCES simulations(id)
);

CREATE TABLE IF NOT EXISTS klines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    simulation_id INTEGER NOT NULL,
    tick INTEGER NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume INTEGER NOT NULL,
    trade_count INTEGER NOT NULL,
    event TEXT,
    FOREIGN KEY (simulation_id) REFERENCES simulations(id)
);

CREATE TABLE IF NOT EXISTS portfolios (
    simulation_id INTEGER NOT NULL,
    agent_id INTEGER NOT NULL,
    agent_type TEXT NOT NULL,
    cash REAL NOT NULL,
    shares INTEGER NOT NULL,
    frozen_cash REAL NOT NULL DEFAULT 0,
    frozen_shares INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (simulation_id, agent_id),
    FOREIGN KEY (simulation_id) REFERENCES simulations(id)
);
"""


class SimulationDB:
    """SQLite 持久化引擎 —— 写入 + 查询。

    用法（写入）：
        db = SimulationDB("output/sim.db")
        db.save(model)          # ForumModel 模拟结束后调用
        db.close()

    用法（回放）：
        db = SimulationDB("output/sim.db")
        for c in db.replay_comments(sim_id=1):
            print(c["created_at"], c["author_name"], c["content"])
        db.close()
    """

    def __init__(self, db_path: str | Path = "output/simulation.db"):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        logger.info("持久化数据库已就绪: %s", self._path)

    def close(self) -> None:
        self._conn.close()

    # ═══════════════════════════════════════════════════
    #  一次性保存完整模拟
    # ═══════════════════════════════════════════════════

    def save(self, model) -> int:
        """保存 ForumModel 的全量数据到 SQLite。

        在一个事务中批量写入，保证原子性。
        返回 simulation_id。
        """
        # 延迟导入避免循环引用；测试时允许 duck-typing
        try:
            from ..environment.forum import ForumModel
            if not isinstance(model, ForumModel):
                logger.debug("save() 接收到非 ForumModel 实例，以 duck-typing 模式运行")
        except ImportError:
            pass

        now = datetime.now().isoformat()

        with self._transaction() as cur:
            # 1. simulations
            n_active = len(model.active_participants)
            n_total = n_active + len(model.passive_participants)
            cur.execute(
                """INSERT INTO simulations
                   (seed, n_normal, n_inst, n_retail, n_active, llm_model,
                    started_at, ended_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    getattr(model, '_seed', None),
                    model.n_normal, model.n_inst, model.n_retail,
                    n_active, getattr(model, 'llm_model', 'gemini-2.5-flash'),
                    getattr(model, '_started_at', now),
                    now,
                ),
            )
            sim_id = cur.lastrowid

            # 2. posts + comments
            for post in model.posts:
                cur.execute(
                    """INSERT INTO posts
                       (id, simulation_id, round_num, topic, content, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        post.id, sim_id, post.round_num, post.topic,
                        post.content,
                        post.created_at.isoformat() if post.created_at else None,
                    ),
                )
                # 三个 phase 的评论
                for phase in (1, 2, 3):
                    for c in post.comments[phase]:
                        self._insert_comment(cur, sim_id, post.id, c)

            # 3. sentiments
            for sg in model.sentiment_history:
                grid_json = json.dumps(
                    {str(k): v for k, v in sg.grid.items()},
                    ensure_ascii=False,
                )
                cur.execute(
                    """INSERT INTO sentiments
                       (simulation_id, round_num, topic, phase, grid_json)
                       VALUES (?, ?, ?, ?, ?)""",
                    (sim_id, sg.round_num, sg.topic, sg.phase, grid_json),
                )

            # 4. trades
            if hasattr(model, 'exchange'):
                tick_for_trade = {}
                for bar in model.exchange.kline_history:
                    tick_for_trade[bar.tick] = bar.tick

                for t in model.exchange.trade_history:
                    cur.execute(
                        """INSERT INTO trades
                           (simulation_id, tick, price, quantity,
                            buy_order_id, sell_order_id,
                            buyer_id, seller_id, match_timestamp)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            sim_id, 0, t.price, t.quantity,
                            t.buy_order_id, t.sell_order_id,
                            t.buyer_id, t.seller_id, t.timestamp,
                        ),
                    )

                # 5. klines
                for bar in model.exchange.kline_history:
                    cur.execute(
                        """INSERT INTO klines
                           (simulation_id, tick, open, high, low, close,
                            volume, trade_count, event)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            sim_id, bar.tick, bar.open, bar.high,
                            bar.low, bar.close, bar.volume,
                            bar.trade_count, bar.event,
                        ),
                    )

                # 6. portfolios
                for agent_id, pf in model.exchange._portfolios.items():
                    agent_type = self._find_agent_type(model, agent_id)
                    cur.execute(
                        """INSERT INTO portfolios
                           (simulation_id, agent_id, agent_type,
                            cash, shares, frozen_cash, frozen_shares)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            sim_id, agent_id, agent_type,
                            pf.cash, pf.shares,
                            pf.frozen_cash, pf.frozen_shares,
                        ),
                    )

        total_comments = sum(
            len(p.comments[ph]) for p in model.posts for ph in (1, 2, 3)
        )
        logger.info(
            "模拟数据已保存 | sim_id=%d 帖子=%d 评论=%d 情绪快照=%d "
            "成交=%d K线=%d 持仓=%d",
            sim_id, len(model.posts), total_comments,
            len(model.sentiment_history),
            len(model.exchange.trade_history) if hasattr(model, 'exchange') else 0,
            len(model.exchange.kline_history) if hasattr(model, 'exchange') else 0,
            len(model.exchange._portfolios) if hasattr(model, 'exchange') else 0,
        )

        return sim_id

    def _insert_comment(
        self, cur: sqlite3.Cursor, sim_id: int, post_id: str, comment
    ) -> None:
        """插入单条评论（不递归子评论——子评论已在 post.comments[phase] 中）。"""
        cur.execute(
            """INSERT INTO comments
               (id, simulation_id, post_id, author_id, author_type,
                author_name, content, temp, sentiment, phase,
                parent_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                comment.id, sim_id, post_id, comment.author_id,
                comment.author_type, comment.author_name,
                comment.content, comment.temp,
                comment.sentiment, comment.phase,
                comment.parent_id,
                comment.created_at.isoformat() if comment.created_at else None,
            ),
        )

    @staticmethod
    def _find_agent_type(model, agent_id: int) -> str:
        """从 model 中查找 agent_id 对应的类型名。"""
        all_agents = model.active_participants + model.passive_participants
        for a in all_agents:
            if a.unique_id == agent_id:
                return a.__class__.__name__
        return "Unknown"

    # ═══════════════════════════════════════════════════
    #  回放查询
    # ═══════════════════════════════════════════════════

    def replay_comments(
        self,
        simulation_id: int = 1,
        post_id: Optional[str] = None,
        phase: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """按 created_at 时间顺序返回评论流（用于回放）。

        参数：
            simulation_id — 模拟 ID（默认最近一次）
            post_id       — 可选，只返回特定帖子的评论
            phase         — 可选，只返回特定 Phase 的评论
            limit         — 可选，限制返回数量

        返回：
            评论字典列表，按 created_at 升序排列。
        """
        sql = """
            SELECT c.*, p.topic, p.round_num
            FROM comments c
            JOIN posts p ON c.simulation_id = p.simulation_id AND c.post_id = p.id
            WHERE c.simulation_id = ?
        """
        params: List[Any] = [simulation_id]

        if post_id is not None:
            sql += " AND c.post_id = ?"
            params.append(post_id)
        if phase is not None:
            sql += " AND c.phase = ?"
            params.append(phase)

        sql += " ORDER BY c.created_at ASC"

        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)

        cur = self._conn.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]

    def get_simulation_summary(
        self, simulation_id: int = 1
    ) -> Optional[Dict[str, Any]]:
        """获取模拟概况。"""
        cur = self._conn.execute(
            "SELECT * FROM simulations WHERE id = ?", (simulation_id,)
        )
        row = cur.fetchone()
        if row is None:
            return None

        summary = dict(row)

        # 统计
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM posts WHERE simulation_id = ?",
            (simulation_id,),
        )
        summary["total_posts"] = cur.fetchone()[0]

        cur = self._conn.execute(
            "SELECT COUNT(*) FROM comments WHERE simulation_id = ?",
            (simulation_id,),
        )
        summary["total_comments"] = cur.fetchone()[0]

        cur = self._conn.execute(
            "SELECT COUNT(*) FROM trades WHERE simulation_id = ?",
            (simulation_id,),
        )
        summary["total_trades"] = cur.fetchone()[0]

        return summary

    def get_klines(self, simulation_id: int = 1) -> List[Dict[str, Any]]:
        """获取 K 线序列。"""
        cur = self._conn.execute(
            "SELECT * FROM klines WHERE simulation_id = ? ORDER BY tick",
            (simulation_id,),
        )
        return [dict(row) for row in cur.fetchall()]

    def get_sentiment_snapshots(
        self, simulation_id: int = 1
    ) -> List[Dict[str, Any]]:
        """获取情绪快照序列。"""
        cur = self._conn.execute(
            """SELECT * FROM sentiments
               WHERE simulation_id = ?
               ORDER BY round_num, topic, phase""",
            (simulation_id,),
        )
        rows = []
        for row in cur.fetchall():
            d = dict(row)
            d["grid"] = json.loads(d.pop("grid_json"))
            rows.append(d)
        return rows

    def get_portfolios(
        self, simulation_id: int = 1
    ) -> List[Dict[str, Any]]:
        """获取最终持仓快照。"""
        cur = self._conn.execute(
            "SELECT * FROM portfolios WHERE simulation_id = ? ORDER BY agent_id",
            (simulation_id,),
        )
        return [dict(row) for row in cur.fetchall()]

    def list_simulations(self) -> List[Dict[str, Any]]:
        """列出所有模拟记录。"""
        cur = self._conn.execute(
            "SELECT * FROM simulations ORDER BY id DESC"
        )
        return [dict(row) for row in cur.fetchall()]

    # ── 内部工具 ──

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Cursor, None, None]:
        """事务上下文管理器。"""
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()


# ═══════════════════════════════════════════════════════
#  便捷函数
# ═══════════════════════════════════════════════════════

def replay_comments(
    db_path: str | Path = "output/simulation.db",
    simulation_id: int = 1,
    **kwargs,
) -> List[Dict[str, Any]]:
    """一行代码回放评论流。

    用法：
        from src.persistence import replay_comments
        for c in replay_comments():
            print(f"[{c['created_at']}] {c['author_name']}: {c['content']}")
    """
    db = SimulationDB(db_path)
    try:
        return db.replay_comments(simulation_id=simulation_id, **kwargs)
    finally:
        db.close()
