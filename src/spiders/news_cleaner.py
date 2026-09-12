"""
news_cleaner —— 清洗爬虫噪音，提取关键政策文本

输入 RawPolicy → 输出 CleanedPolicy：
  1. 去 HTML 标签（丢弃 script / style 内容，实体转义还原）
  2. 折叠空白、剥掉常见尾部噪音（"Read more"、"分享到"…）
  3. 语言检测（CJK 字符占比）
  4. 正文过短视为噪音直接丢弃（中英文阈值不同，中文信息密度更高）
  5. 按 content_hash 去重（同一条政策被多源转载只留一份）

解析用标准库 html.parser，不引入 bs4。
"""

from __future__ import annotations

import logging
import re
from html.parser import HTMLParser
from typing import Iterable, List, Optional

from ..config import POLICY_MIN_BODY_CHARS
from .models import CleanedPolicy, RawPolicy

logger = logging.getLogger(__name__)

_SKIP_TAGS = {"script", "style", "noscript"}
_WHITESPACE = re.compile(r"\s+")
_CJK = re.compile(r"[一-鿿]")

# 尾部噪音：源站模板里的固定尾巴
_TAIL_NOISE = re.compile(
    r"(read more|continue reading|查看更多|阅读全文|分享到|责任编辑)[：:\s].*$",
    re.IGNORECASE | re.DOTALL,
)


# ═══════════════════════════════════════════════════════
#  去标签
# ═══════════════════════════════════════════════════════


class _TextExtractor(HTMLParser):
    """把 HTML 抽成纯文本，跳过 script / style。"""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._parts: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0:
            self._parts.append(data)

    @property
    def text(self) -> str:
        return "".join(self._parts)


def strip_html(raw: str) -> str:
    """去标签 + 折叠空白 + 去尾部噪音。"""
    if not raw:
        return ""
    parser = _TextExtractor()
    try:
        parser.feed(raw)
        parser.close()
        text = parser.text
    except Exception:            # 畸形 HTML 兜底：退回正则去标签
        text = re.sub(r"<[^>]+>", " ", raw)

    text = _WHITESPACE.sub(" ", text).strip()
    return _TAIL_NOISE.sub("", text).strip()


def detect_lang(text: str) -> str:
    """按 CJK 字符占比判定 zh / en。"""
    if not text:
        return "en"
    cjk = len(_CJK.findall(text))
    return "zh" if cjk / len(text) > 0.1 else "en"


# ═══════════════════════════════════════════════════════
#  清洗入口
# ═══════════════════════════════════════════════════════


def min_chars_for(lang: str, min_chars: int = POLICY_MIN_BODY_CHARS) -> int:
    """中文信息密度远高于英文：同样字数的中文正文要长得多，阈值减半。"""
    return min_chars if lang == "en" else max(20, min_chars // 2)


def clean(
    raw: RawPolicy, min_chars: int = POLICY_MIN_BODY_CHARS
) -> Optional[CleanedPolicy]:
    """清洗单条；正文过短返回 None（调用方丢弃）。"""
    title = strip_html(raw.title)
    body = strip_html(raw.raw_body)
    lang = detect_lang(f"{title}{body}")

    threshold = min_chars_for(lang, min_chars)
    if len(body) < threshold:
        logger.debug(
            "[清洗] 丢弃过短条目（%d < %d 字, lang=%s）: %s",
            len(body), threshold, lang, title[:40],
        )
        return None

    return CleanedPolicy(
        hash=raw.hash,
        source=raw.source,
        url=raw.url,
        title=title,
        text=body,
        lang=lang,
        published_at=raw.published_at,
        fetched_at=raw.fetched_at,
    )


def clean_all(
    raws: Iterable[RawPolicy], min_chars: int = POLICY_MIN_BODY_CHARS
) -> List[CleanedPolicy]:
    """批量清洗 + 按 hash 去重，保持原顺序。"""
    seen: set[str] = set()
    cleaned: List[CleanedPolicy] = []
    dropped = 0

    for raw in raws:
        item = clean(raw, min_chars=min_chars)
        if item is None:
            dropped += 1
            continue
        if item.hash in seen:
            dropped += 1
            continue
        seen.add(item.hash)
        cleaned.append(item)

    logger.info("[清洗] 保留 %d 条，丢弃 %d 条（过短 / 重复）", len(cleaned), dropped)
    return cleaned
