"""
classifier —— 给清洗后的政策打多标签

两条路径：
  1. 注入的 LLMProvider 可用 → 结构化输出（temperature=0，分类要确定性）
  2. provider 为 None / 不可用 / 调用失败 → 关键词规则兜底（中英双语词表）

LLM 由调用方注入，本模块不碰环境变量，也不持有任何全局状态。

标签只用于让 Agent 在 RAG 检索时更容易命中相关政策，
不参与任何数值计算——按设计，外部数据不修改 Agent 属性。
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from ..config import EXTRACTOR_LLM_TEMPERATURE, MAX_CONCURRENT_LLM_CALLS
from ..ports.llm import LLMProvider, StructuredModel
from ..spiders.models import CleanedPolicy

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
#  标签体系
# ═══════════════════════════════════════════════════════

LABELS: Tuple[str, ...] = (
    "ipo",               # 发行 / 上市 / 招股
    "disclosure",        # 信息披露
    "enforcement",       # 执法 / 处罚 / 诉讼
    "monetary",          # 货币政策 / 利率
    "market-structure",  # 交易规则 / 市场结构
    "esg",               # 可持续 / 气候 / 治理
)

# 关键词兜底词表（中英混排，全部小写匹配）
_RULES: Dict[str, Tuple[str, ...]] = {
    "ipo": (
        "initial public offering", "ipo", "going public", "s-1", "prospectus",
        "underwrit", "首次公开发行", "上市", "招股", "发行人", "承销",
    ),
    "disclosure": (
        "disclosure", "reporting requirement", "filing", "transparency",
        "material information", "信息披露", "披露", "定期报告", "财务报告",
    ),
    "enforcement": (
        "charge", "fraud", "enforcement", "penalt", "sanction", "settle",
        "violat", "处罚", "执法", "违规", "欺诈", "立案", "调查",
    ),
    "monetary": (
        "interest rate", "monetary policy", "federal funds", "inflation",
        "fomc", "利率", "货币政策", "降准", "降息", "通胀",
    ),
    "market-structure": (
        "trading rule", "market structure", "clearing", "settlement cycle",
        "exchange rule", "liquidity", "交易规则", "市场结构", "清算", "做市",
    ),
    "esg": (
        "climate", "sustainab", "esg", "greenhouse", "diversity",
        "气候", "可持续", "环保", "碳中和", "绿色",
    ),
}


def classify_by_rules(text: str) -> List[str]:
    """关键词兜底分类；一条都没命中时返回空列表。"""
    lowered = text.lower()
    hit = [
        label for label, words in _RULES.items()
        if any(w in lowered for w in words)
    ]
    return hit


# ═══════════════════════════════════════════════════════
#  LLM 分类
# ═══════════════════════════════════════════════════════


class PolicyLabels(BaseModel):
    """LLM 结构化输出 schema。"""

    labels: List[str] = Field(
        description="从给定标签集中选出的标签，可多选，无匹配则返回空列表",
    )


def _structured(llm: Optional[LLMProvider]) -> Optional[StructuredModel]:
    """从注入的 provider 取一个绑定 PolicyLabels 的模型。

    llm 为 None 或后端不可用 → 返回 None，调用方走关键词规则。
    """
    if llm is None:
        return None
    return llm.structured(PolicyLabels, temperature=EXTRACTOR_LLM_TEMPERATURE)


def _prompt(policy: CleanedPolicy) -> str:
    return (
        f"给下面这条监管政策 / 新闻打标签，只能从这个集合里选：\n"
        f"{', '.join(LABELS)}\n\n"
        f"可以多选，也可以一个都不选（返回空列表）。不要发明新标签。\n\n"
        f"【标题】{policy.title}\n"
        f"【正文】{policy.text[:1500]}"
    )


async def aclassify(
    policy: CleanedPolicy, llm: Optional[LLMProvider] = None
) -> List[str]:
    """给单条政策打标签；LLM 不可用或失败时退回规则。"""
    model = _structured(llm)
    fallback = classify_by_rules(f"{policy.title} {policy.text}")

    if model is None:
        return fallback

    try:
        out: PolicyLabels = await model.ainvoke(_prompt(policy))
    except Exception as e:
        logger.debug("[分类] LLM 调用失败，退回规则: %s", e)
        return fallback

    # LLM 可能返回集合外的标签，过滤掉
    valid = [l for l in out.labels if l in LABELS]
    return valid or fallback


async def aclassify_all(
    policies: Sequence[CleanedPolicy],
    llm: Optional[LLMProvider] = None,
    max_concurrent: int = MAX_CONCURRENT_LLM_CALLS,
) -> List[CleanedPolicy]:
    """并发给一批政策打标签，就地写回 labels 字段并返回同一批对象。"""
    if not policies:
        return list(policies)

    if _structured(llm) is None:
        logger.info("[分类] LLM 不可用，使用关键词规则兜底")

    semaphore = asyncio.Semaphore(max_concurrent)

    async def one(p: CleanedPolicy) -> List[str]:
        async with semaphore:
            return await aclassify(p, llm)

    results = await asyncio.gather(
        *(one(p) for p in policies), return_exceptions=True
    )

    for policy, labels in zip(policies, results):
        if isinstance(labels, Exception):
            policy.labels = classify_by_rules(f"{policy.title} {policy.text}")
        else:
            policy.labels = labels

    tagged = sum(1 for p in policies if p.labels)
    logger.info("[分类] %d 条中 %d 条命中标签", len(policies), tagged)
    return list(policies)
