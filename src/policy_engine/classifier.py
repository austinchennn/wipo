"""
classifier —— 给清洗后的政策打多标签

两条路径：
  1. 有 GOOGLE_API_KEY → LLM 结构化输出（temperature=0，分类要确定性）
  2. 无 Key / 调用失败 → 关键词规则兜底（中英双语词表）

标签只用于让 Agent 在 RAG 检索时更容易命中相关政策，
不参与任何数值计算——按设计，外部数据不修改 Agent 属性。
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from ..config import EXTRACTOR_LLM_TEMPERATURE, MAX_CONCURRENT_LLM_CALLS
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


_llm_instance = None
_llm_initialized = False


def _get_classifier_llm():
    """懒加载分类 LLM；无 API Key 返回 None（调用方走规则兜底）。"""
    global _llm_instance, _llm_initialized
    if _llm_initialized:
        return _llm_instance

    _llm_initialized = True
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.info("[分类] 无 GOOGLE_API_KEY，使用关键词规则兜底")
        return None

    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model=os.environ.get("AGENT_LLM_MODEL", "gemini-2.5-flash"),
        temperature=EXTRACTOR_LLM_TEMPERATURE,
        google_api_key=api_key,
    )
    _llm_instance = llm.with_structured_output(PolicyLabels)
    return _llm_instance


def _prompt(policy: CleanedPolicy) -> str:
    return (
        f"给下面这条监管政策 / 新闻打标签，只能从这个集合里选：\n"
        f"{', '.join(LABELS)}\n\n"
        f"可以多选，也可以一个都不选（返回空列表）。不要发明新标签。\n\n"
        f"【标题】{policy.title}\n"
        f"【正文】{policy.text[:1500]}"
    )


async def aclassify(policy: CleanedPolicy) -> List[str]:
    """给单条政策打标签；LLM 不可用或失败时退回规则。"""
    llm = _get_classifier_llm()
    fallback = classify_by_rules(f"{policy.title} {policy.text}")

    if llm is None:
        return fallback

    try:
        out: PolicyLabels = await llm.ainvoke(_prompt(policy))
    except Exception as e:
        logger.debug("[分类] LLM 调用失败，退回规则: %s", e)
        return fallback

    # LLM 可能返回集合外的标签，过滤掉
    valid = [l for l in out.labels if l in LABELS]
    return valid or fallback


async def aclassify_all(
    policies: Sequence[CleanedPolicy],
    max_concurrent: int = MAX_CONCURRENT_LLM_CALLS,
) -> List[CleanedPolicy]:
    """并发给一批政策打标签，就地写回 labels 字段并返回同一批对象。"""
    if not policies:
        return list(policies)

    semaphore = asyncio.Semaphore(max_concurrent)

    async def one(p: CleanedPolicy) -> List[str]:
        async with semaphore:
            return await aclassify(p)

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
