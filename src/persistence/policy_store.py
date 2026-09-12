"""
persistence.policy_store —— 外部政策缓存（SQLite）

与 SimulationDB 分库：模拟数据一次一库，政策缓存跨模拟复用。

缓存策略（见 Spider 数据流设计）：
  - 每条政策按 hash 去重，upsert 时刷新 fetched_at
  - load_fresh(ttl) 只返回 fetched_at 在 TTL 内的记录
  - is_fresh() 为 True → 调度器直接跳过爬取，复用已有 chunks
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from ..config import POLICY_CACHE_TTL_DAYS
from ..spiders.models import CleanedPolicy

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS policies (
    hash         TEXT PRIMARY KEY,
    source       TEXT NOT NULL,
    url          TEXT NOT NULL,
    title        TEXT NOT NULL,
    text         TEXT NOT NULL,
    lang         TEXT NOT NULL,
    labels_json  TEXT NOT NULL,
    published_at TEXT,
    fetched_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_policies_fetched_at ON policies(fetched_at);
"""


class PolicyStore:
    """政策缓存读写。

    用法：
        store = PolicyStore()
        if not store.is_fresh():
            store.save(crawl_and_clean())
        policies = store.load_fresh()
        store.close()
    """

    def __init__(self, db_path: str | Path = "output/policies.db"):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "PolicyStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ═══════════════════════════════════════════════════
    #  写入
    # ═══════════════════════════════════════════════════

    def save(self, policies: List[CleanedPolicy]) -> int:
        """按 hash upsert，返回写入条数。"""
        if not policies:
            return 0

        rows = [
            (
                p.hash, p.source, p.url, p.title, p.text, p.lang,
                json.dumps(p.labels, ensure_ascii=False),
                p.published_at,
                p.fetched_at.isoformat(),
            )
            for p in policies
        ]
        with self._conn:
            self._conn.executemany(
                """
                INSERT INTO policies
                    (hash, source, url, title, text, lang,
                     labels_json, published_at, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(hash) DO UPDATE SET
                    text        = excluded.text,
                    labels_json = excluded.labels_json,
                    fetched_at  = excluded.fetched_at
                """,
                rows,
            )
        logger.info("[政策缓存] 写入 %d 条", len(rows))
        return len(rows)

    # ═══════════════════════════════════════════════════
    #  读取 / TTL
    # ═══════════════════════════════════════════════════

    def load_fresh(
        self, ttl_days: int = POLICY_CACHE_TTL_DAYS
    ) -> List[CleanedPolicy]:
        """返回 TTL 内的政策，按发布时间倒序。"""
        cutoff = (datetime.now() - timedelta(days=ttl_days)).isoformat()
        cur = self._conn.execute(
            "SELECT * FROM policies WHERE fetched_at >= ? "
            "ORDER BY COALESCE(published_at, fetched_at) DESC",
            (cutoff,),
        )
        return [self._row_to_policy(r) for r in cur.fetchall()]

    def is_fresh(self, ttl_days: int = POLICY_CACHE_TTL_DAYS) -> bool:
        """库里是否还有 TTL 内的数据（有则不必重爬）。"""
        cutoff = (datetime.now() - timedelta(days=ttl_days)).isoformat()
        cur = self._conn.execute(
            "SELECT 1 FROM policies WHERE fetched_at >= ? LIMIT 1", (cutoff,)
        )
        return cur.fetchone() is not None

    def latest_fetched_at(self) -> Optional[datetime]:
        """最近一次成功爬取的时间，空库返回 None。"""
        cur = self._conn.execute("SELECT MAX(fetched_at) AS t FROM policies")
        row = cur.fetchone()
        if row is None or row["t"] is None:
            return None
        return datetime.fromisoformat(row["t"])

    def count(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) AS c FROM policies"
        ).fetchone()["c"]

    def purge_expired(self, ttl_days: int = POLICY_CACHE_TTL_DAYS) -> int:
        """删除超过 TTL 的记录，返回删除条数。"""
        cutoff = (datetime.now() - timedelta(days=ttl_days)).isoformat()
        with self._conn:
            cur = self._conn.execute(
                "DELETE FROM policies WHERE fetched_at < ?", (cutoff,)
            )
        return cur.rowcount

    # ─────────────────────────────────────────────

    @staticmethod
    def _row_to_policy(row: sqlite3.Row) -> CleanedPolicy:
        return CleanedPolicy(
            hash=row["hash"],
            source=row["source"],
            url=row["url"],
            title=row["title"],
            text=row["text"],
            lang=row["lang"],
            labels=json.loads(row["labels_json"]),
            published_at=row["published_at"],
            fetched_at=datetime.fromisoformat(row["fetched_at"]),
        )
