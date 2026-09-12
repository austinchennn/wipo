"""
policy_spider —— 抓取公开政策源（RSS / Atom）

只走各监管机构面向机器消费的公开 feed，不做整站爬取、不绕反爬。
解析用标准库 xml.etree，不额外引入 feedparser。

用法：
    spider = PolicySpider()
    raws = await spider.afetch_all()        # List[RawPolicy]

    # 自定义源
    spider = PolicySpider(sources=[PolicySource("csrc", "https://.../rss.xml")])
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional
from xml.etree import ElementTree

import httpx

from ..config import POLICY_HTTP_TIMEOUT, POLICY_MAX_ITEMS_PER_SOURCE
from .models import RawPolicy

logger = logging.getLogger(__name__)

# SEC 等机构的合理访问policy要求 UA 里带可联系的标识
USER_AGENT = "wipo-research-bot/0.1 (+https://github.com/austinchennn/wipo)"

# Atom 命名空间（RSS 2.0 无命名空间）
_ATOM = "{http://www.w3.org/2005/Atom}"


@dataclass(frozen=True)
class PolicySource:
    """一个政策源。"""

    name: str      # 源标识，写进 RawPolicy.source
    url: str       # RSS / Atom feed 地址


# 已实测可用（2026-09）。sec.gov 的 /rss/* 旧路径现在一律 403，不要再用。
DEFAULT_SOURCES: tuple[PolicySource, ...] = (
    PolicySource("sec-press", "https://www.sec.gov/news/pressreleases.rss"),
    PolicySource("sec-speech", "https://www.sec.gov/news/speeches-statements.rss"),
    PolicySource("fed-press", "https://www.federalreserve.gov/feeds/press_all.xml"),
)


class PolicySpider:
    """把若干 RSS / Atom feed 拉成 RawPolicy 列表。"""

    def __init__(
        self,
        sources: Optional[Iterable[PolicySource]] = None,
        max_items: int = POLICY_MAX_ITEMS_PER_SOURCE,
        timeout: float = POLICY_HTTP_TIMEOUT,
    ):
        self.sources: List[PolicySource] = list(sources or DEFAULT_SOURCES)
        self.max_items = max_items
        self.timeout = timeout

    # ═══════════════════════════════════════════════════
    #  抓取
    # ═══════════════════════════════════════════════════

    async def afetch_all(self) -> List[RawPolicy]:
        """并发拉取所有源；单个源失败只记日志，不影响其余源。"""
        async with httpx.AsyncClient(
            timeout=self.timeout,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
        ) as client:
            results = await asyncio.gather(
                *(self._afetch_one(client, s) for s in self.sources),
                return_exceptions=True,
            )

        raws: List[RawPolicy] = []
        for source, res in zip(self.sources, results):
            if isinstance(res, Exception):
                logger.warning("[爬虫] 源 %s 抓取失败: %s", source.name, res)
                continue
            logger.info("[爬虫] 源 %s → %d 条", source.name, len(res))
            raws.extend(res)
        return raws

    def fetch_all(self) -> List[RawPolicy]:
        """同步入口（脚本 / REPL 用）。"""
        return asyncio.run(self.afetch_all())

    async def _afetch_one(
        self, client: httpx.AsyncClient, source: PolicySource
    ) -> List[RawPolicy]:
        resp = await client.get(source.url)
        resp.raise_for_status()
        return parse_feed(resp.text, source.name)[: self.max_items]


# ═══════════════════════════════════════════════════════
#  Feed 解析（纯函数，可脱网测试）
# ═══════════════════════════════════════════════════════


def parse_feed(xml_text: str, source: str) -> List[RawPolicy]:
    """解析 RSS 2.0 或 Atom，返回 RawPolicy 列表。格式不认识时返回空列表。"""
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError as e:
        logger.warning("[爬虫] 源 %s XML 解析失败: %s", source, e)
        return []

    items = root.findall(".//item")
    if items:
        return [_from_rss_item(it, source) for it in items]

    entries = root.findall(f".//{_ATOM}entry")
    return [_from_atom_entry(e, source) for e in entries]


def _text(node: Optional[ElementTree.Element]) -> str:
    return (node.text or "").strip() if node is not None else ""


def _from_rss_item(item: ElementTree.Element, source: str) -> RawPolicy:
    return RawPolicy(
        source=source,
        url=_text(item.find("link")),
        title=_text(item.find("title")),
        raw_body=_text(item.find("description")),
        published_at=_text(item.find("pubDate")) or None,
    )


def _from_atom_entry(entry: ElementTree.Element, source: str) -> RawPolicy:
    link = entry.find(f"{_ATOM}link")
    url = link.get("href", "") if link is not None else ""
    body = _text(entry.find(f"{_ATOM}content")) or _text(entry.find(f"{_ATOM}summary"))
    return RawPolicy(
        source=source,
        url=url,
        title=_text(entry.find(f"{_ATOM}title")),
        raw_body=body,
        published_at=_text(entry.find(f"{_ATOM}updated")) or None,
    )
