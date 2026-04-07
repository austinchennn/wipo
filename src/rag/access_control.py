"""
access_control — Agent 类型 × 话题 → 访问级别映射。

完整可见性矩阵：
                    product      financial    policy(risk)
  HostAgent         FULL         FULL         FULL
  NormalAgent       SUMMARY      HIDDEN       HIDDEN
  InstTraderAgent   FULL         FULL         FULL
  RetailTraderAgent MASKED       MASKED       MASKED

AccessLevel 含义：
  FULL    → 返回原始 RAG 检索结果（完整 chunk 文本）
  SUMMARY → 返回 RAG 结果的极简摘要（Level C 风格）
  MASKED  → 返回 RAG 结果，但数字/专利号用 ████ 替换
  HIDDEN  → 拒绝访问，直接返回固定屏蔽文本
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Dict


class AccessLevel(Enum):
    FULL    = "full"
    SUMMARY = "summary"
    MASKED  = "masked"
    HIDDEN  = "hidden"


# ─────────────────────────────────────────────────────────────────
#  可见性矩阵
#  键：(agent_class_name, topic)   值：AccessLevel
# ─────────────────────────────────────────────────────────────────

VISIBILITY_MATRIX: Dict[tuple[str, str], AccessLevel] = {
    # HostAgent — 全知全觉
    ("HostAgent", "product"):   AccessLevel.FULL,
    ("HostAgent", "financial"): AccessLevel.FULL,
    ("HostAgent", "policy"):    AccessLevel.FULL,

    # NormalAgent — 只能看通俗产品摘要
    ("NormalAgent", "product"):   AccessLevel.SUMMARY,
    ("NormalAgent", "financial"): AccessLevel.HIDDEN,
    ("NormalAgent", "policy"):    AccessLevel.HIDDEN,

    # InstTraderAgent — 机构，全信息无遮挡
    ("InstTraderAgent", "product"):   AccessLevel.FULL,
    ("InstTraderAgent", "financial"): AccessLevel.FULL,
    ("InstTraderAgent", "policy"):    AccessLevel.FULL,

    # RetailTraderAgent — 散户，什么都能看但都被模糊
    ("RetailTraderAgent", "product"):   AccessLevel.MASKED,
    ("RetailTraderAgent", "financial"): AccessLevel.MASKED,
    ("RetailTraderAgent", "policy"):    AccessLevel.MASKED,
}


def get_access_level(agent_type: str, topic: str) -> AccessLevel:
    """返回 agent_type 对 topic 的访问级别，未知组合默认 HIDDEN。"""
    return VISIBILITY_MATRIX.get((agent_type, topic), AccessLevel.HIDDEN)


# ─────────────────────────────────────────────────────────────────
#  文本变换函数
# ─────────────────────────────────────────────────────────────────

_NUMBER_PATTERN = re.compile(
    r"""
    (?:
        \$[\d,]+(?:\.\d+)?(?:[MBK]|\s*(?:million|billion|thousand))? # 金额
        | \d{1,3}(?:,\d{3})*(?:\.\d+)?%                               # 百分比
        | US\d[\d,./\-]+                                               # 专利号 US...
        | \d+\.\d+(?:[MBK])?                                           # 小数
        | \d{4,}                                                       # 4位以上整数
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

_PATENT_PATTERN = re.compile(
    r"(?:patent|专利|US|EP|CN)\s*(?:No\.?\s*)?\d[\d,./\-]+",
    re.IGNORECASE,
)

_LEGAL_PATTERN = re.compile(
    r"(?:Section|§|Rule|Regulation|CFR|USC|法规|条款)\s+[\w\d.()]+",
    re.IGNORECASE,
)


def apply_mask(text: str) -> str:
    """对文本执行掩码处理（MASKED 级别）：
    - 具体金额、百分比、4位以上数字 → ████
    - 专利号 → [专利编号]
    保留文本结构和定性描述。
    """
    text = _PATENT_PATTERN.sub("[专利编号]", text)
    text = _NUMBER_PATTERN.sub("████", text)
    if len(text) > 300:
        text = text[:300] + "…… [内容部分模糊]"
    return f"[掩码信息] {text}"


def apply_summary(text: str, max_chars: int = 150) -> str:
    """对文本执行极简摘要处理（SUMMARY 级别）：
    截断并去除所有数字描述，保留最核心的一句话。
    """
    # 移除数字密集的句子
    sentences = re.split(r"[。.！!？?]", text)
    clean = [
        s.strip() for s in sentences
        if s.strip() and len(re.findall(r"\d", s)) < 3
    ]
    result = "。".join(clean[:2])
    if len(result) > max_chars:
        result = result[:max_chars] + "……"
    return f"[摘要信息] {result}" if result else "[摘要信息] 信息已精简处理"


def apply_access_control(text: str, level: AccessLevel) -> str:
    """根据访问级别对文本做对应处理。"""
    if level == AccessLevel.FULL:
        return text
    if level == AccessLevel.MASKED:
        return apply_mask(text)
    if level == AccessLevel.SUMMARY:
        return apply_summary(text)
    # HIDDEN
    return "[此信息对你不可见]"
