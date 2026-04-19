"""
product_extractor — 产品与业务解析器 (Product & Business Extractor)

目标章节：业务（Business）、主要产品（Products）、竞争（Competition）、
          技术（Technology）、知识产权（Intellectual Property）

LLM 角色：资深行业分析师

输出三个信息层级：
  Level A（完整版）  → HostAgent / InstTraderAgent — 所有技术细节 + 市场数据
  Level B（掩码版）  → RetailTraderAgent         — 具体数字/专利号用 ████ 遮盖
  Level C（摘要版）  → NormalAgent               — 通俗一句话描述，不含任何财务政策背景
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from .base_extractor import (
    PRODUCT_KEYWORDS,
    LLMExtractor,
)


# ─────────────────────────────────────────────────────────────────
#  输出 Schema
# ─────────────────────────────────────────────────────────────────

class ProductSummary(BaseModel):
    """产品与业务结构化摘要（含三个信息层级）"""

    # ── 结构化字段（机器可读，供 Agent 属性调整使用）──
    core_products: List[str] = Field(
        description="公司核心产品 / 服务列表（3-6 项）"
    )
    tech_keywords: List[str] = Field(
        description="核心技术关键词（如：分布式架构、生成式算法、SaaS、芯片制造）"
    )
    market_position: str = Field(
        description="市场地位描述：如 '全球市占率约 12%，行业第三'"
    )
    competitive_advantage: str = Field(
        description="核心竞争壁垒（技术护城河、网络效应、规模效应等）"
    )
    main_competitors: List[str] = Field(
        description="主要竞争对手名称列表"
    )

    # ── Level A：完整版（机构 / 主持人）──
    level_a: str = Field(
        description=(
            "完整技术版本（≤600字）：包含所有技术细节、市场数据、专利数量、"
            "具体市占率数字、竞争对手分析。供机构投资者和主持人使用。"
        )
    )

    # ── Level B：掩码版（散户）──
    level_b: str = Field(
        description=(
            "掩码版本（≤400字）：将具体的毛利率数字（如 '68%'）、"
            "专利号（如 'US10,234,567'）、精确市占率替换为 ████，"
            "保留产品功能描述和大致竞争格局。"
        )
    )

    # ── Level C：摘要版（普通人）──
    level_c: str = Field(
        description=(
            "极简通俗版本（≤100字）：用一句话让完全不懂金融的人也能理解"
            "'这家公司是做什么的'。不含任何财务数字、政策背景或技术术语。"
        )
    )


# ─────────────────────────────────────────────────────────────────
#  Prompts
# ─────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
你是一个资深行业分析师，专注于 IPO 招股书的产品与业务评估。

你的任务是从招股书节选中提取产品和业务信息，并生成三个不同详细程度的版本：

Level A（完整版）：面向机构投资者，包含所有技术细节、精确市场数据、专利信息和竞争分析。
Level B（掩码版）：面向散户投资者，产品描述保留，但将具体数字（毛利率、专利号、精确市占率）
                  用 ████ 遮盖，避免散户获得机构级别的精准数据。
Level C（摘要版）：面向普通大众，用最通俗易懂的语言描述公司主营业务，
                  不超过100字，不含任何财务或政策背景。

重点提取：
- 公司核心产品/服务是什么（一句话概括）
- 使用了哪些关键技术（核心技术栈）
- 在行业中的市场地位（市占率、排名、规模）
- 主要竞争优势（护城河）
- 主要竞争对手

输出语言：中文（产品名、技术术语可保留英文）。
"""

_USER_PROMPT = """\
以下是招股书的产品/业务相关节选：

{context}

---
请根据以上内容生成产品与业务摘要。
如果某项数据未在文本中明确提及，请在对应字段注明"未披露"，不要虚构。
Level B 中的掩码请使用 ████ 符号。
"""


# ─────────────────────────────────────────────────────────────────
#  Extractor 类
# ─────────────────────────────────────────────────────────────────

class ProductExtractor(LLMExtractor["ProductSummary"]):
    """从招股书 PDF chunks 中提取产品与业务信息。"""

    KEYWORDS = PRODUCT_KEYWORDS
    SYSTEM_PROMPT = _SYSTEM_PROMPT
    USER_PROMPT = _USER_PROMPT
    TOPIC = "product"
    SUMMARY_CLASS = ProductSummary

    @staticmethod
    def get_content_for_agent(summary: ProductSummary, agent_type: str) -> str:
        """根据 Agent 类型返回对应层级的产品信息文本。

        HostAgent / InstTraderAgent → Level A
        NormalAgent                → Level C
        RetailTraderAgent          → Level B
        """
        mapping = {
            "HostAgent":         summary.level_a,
            "InstTraderAgent":   summary.level_a,
            "NormalAgent":       summary.level_c,
            "RetailTraderAgent": summary.level_b,
        }
        return mapping.get(agent_type, summary.level_c)
