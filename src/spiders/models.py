"""
spiders.models —— 爬虫链路的数据 schema

    PolicySpider  →  RawPolicy      （原始 HTML / 摘要，未清洗）
    news_cleaner  →  CleanedPolicy  （纯文本正文 + 语言标记）
    classifier    →  CleanedPolicy.labels 填充（多标签）

两个 dataclass 都带 fetched_at 时间戳，供 PolicyStore 做 1 周 TTL 判断。
content_hash 是去重主键：同一条政策被不同源转载时只保留一份。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

_HASH_SEP = "|#|"


def content_hash(*parts: str) -> str:
    """对标题 / URL 做稳定哈希，作为去重主键。"""
    joined = _HASH_SEP.join(p.strip() for p in parts if p)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]


# ═══════════════════════════════════════════════════════
#  RawPolicy —— 爬虫直接产出
# ═══════════════════════════════════════════════════════


@dataclass
class RawPolicy:
    """一条未清洗的政策条目（RSS entry 或列表页条目）。"""

    source: str                          # 源标识，如 "sec" / "csrc"
    url: str
    title: str
    raw_body: str = ""                   # 原始 HTML / summary 片段
    published_at: Optional[str] = None   # 源站给的发布时间
    fetched_at: datetime = field(default_factory=datetime.now)

    @property
    def hash(self) -> str:
        return content_hash(self.title, self.url)


# ═══════════════════════════════════════════════════════
#  CleanedPolicy —— 清洗 + 分类后的结构化政策
# ═══════════════════════════════════════════════════════


@dataclass
class CleanedPolicy:
    """清洗后的政策正文，labels 由 policy_engine.classifier 填充。"""

    hash: str
    source: str
    url: str
    title: str
    text: str                        # 去标签后的纯正文
    lang: str = "en"                 # "zh" | "en"
    labels: List[str] = field(default_factory=list)
    published_at: Optional[str] = None
    fetched_at: datetime = field(default_factory=datetime.now)

    def to_document_text(self) -> str:
        """拼成进 RAG 知识库的一段文本。"""
        label_line = f"【标签】{', '.join(self.labels)}\n" if self.labels else ""
        return (
            f"【外部政策 · {self.source}】{self.title}\n"
            f"{label_line}"
            f"{self.text}"
        )
