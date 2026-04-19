"""
risk_extractor — 风险与政策关联解析器 (Risk & Policy Extractor)

目标章节：Risk Factors（风险因素）、Legal Proceedings（法律诉讼）、
          Regulatory Environment（监管环境）、Government Approvals

LLM 角色：风控专家

输出两个信息层级：
  高敏感版（high_sensitivity）→ HostAgent / InstTraderAgent — 完整风险条目，所有细节
  普通版（normal）            → RetailTraderAgent          — 仅前5条，法律条文掩码
  [不可见]                   → NormalAgent               — 整个模块对普通人隐藏
"""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field

from .base_extractor import (
    RISK_KEYWORDS,
    LLMExtractor,
)


# ─────────────────────────────────────────────────────────────────
#  输出 Schema
# ─────────────────────────────────────────────────────────────────

class RiskItem(BaseModel):
    """单条风险条目"""
    category: Literal[
        "行业监管风险", "法律诉讼风险", "财务风险", "市场竞争风险",
        "运营风险", "宏观政策风险", "其他"
    ] = Field(description="风险分类")
    priority: int = Field(
        ge=1, le=10,
        description=(
            "优先级（1-10）：10=最致命。判断依据：招股书中该风险的"
            "篇幅、排列位置（越靠前越重要）、管理层强调程度。"
        )
    )
    title: str = Field(description="风险标题（≤30字，精炼概括）")
    description_full: str = Field(
        description="完整风险描述（≤200字），含具体法规条文、数字和案例"
    )
    description_masked: str = Field(
        description=(
            "掩码版描述（≤150字）：将具体法律条文编号（如 'Section 15(d)'、"
            "'17 CFR §240'）替换为 [法规条文]，保留风险实质内容。"
        )
    )
    policy_keywords: List[str] = Field(
        description=(
            "可能与宏观政策产生联动的关键词（如 '出口限制'、'利率敏感'、"
            "'反垄断调查'、'数据本地化'），用于外部政策对撞分析。"
        )
    )
    is_enacted_risk: bool = Field(
        description="True=已有具体监管动作/诉讼在进行；False=潜在/假设性风险"
    )


class RiskSummary(BaseModel):
    """风险因素结构化摘要（含高敏感版和普通版）"""

    # ── 结构化字段 ──
    risks: List[RiskItem] = Field(
        description="按优先级降序排列的完整风险列表（最多15条）"
    )
    top_risk_category: str = Field(
        description="最主要的风险类别（如 '行业监管风险'）"
    )
    overall_risk_score: float = Field(
        ge=0.0, le=1.0,
        description="综合风险评分：1.0=极高风险，0.0=极低风险"
    )
    policy_correlation_score: float = Field(
        ge=0.0, le=1.0,
        description=(
            "宏观政策关联度：该公司风险有多少比例受宏观政策驱动。"
            "1.0=完全政策驱动（如芯片出口管制），0.0=纯商业竞争风险"
        )
    )

    # ── 高敏感版：所有风险，完整细节（机构/主持人）──
    high_sensitivity_version: str = Field(
        description=(
            "高敏感完整版（≤800字）：列出所有重大风险条目，"
            "包含具体法规编号、历史诉讼金额、监管机构名称，"
            "并给出政策关联分析。"
        )
    )

    # ── 普通版：前5条 + 法律条文掩码（散户）──
    normal_version: str = Field(
        description=(
            "普通版（≤400字）：仅保留优先级最高的5条风险，"
            "具体法律条文引用替换为 [相关法规]，"
            "具体罚款金额替换为 '相当规模的罚款'，"
            "保留风险的实质内容和影响判断。"
        )
    )


# ─────────────────────────────────────────────────────────────────
#  Prompts
# ─────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
你是一个风控专家，专注于 IPO 招股书的风险因素深度解析。

你的任务是：
1. 从招股书的「风险因素（Risk Factors）」章节提取所有重大风险。
2. 将风险归类为：行业监管风险、法律诉讼风险、财务风险、市场竞争风险、运营风险、宏观政策风险、其他。
3. 按优先级排序：判断依据是该风险在招股书中的排列位置（越靠前越致命）、篇幅大小、管理层强调程度。
4. 提取每条风险中可能与宏观政策产生联动的关键词，用于后续政策冲击分析。
5. 生成两个版本：
   - 高敏感版：所有细节，含具体法规条文编号、案例、数字。
   - 普通版：仅前5条，法律条文掩码，避免信息过载。

重要原则：
- 只提取正式的风险因素，不包括公司自己的正面宣传
- 优先关注那些公司在最前面、用最大篇幅讨论的风险（通常是最致命的）
- policy_keywords 要具体（如"出口管制"而非"政策风险"）

输出语言：中文（法规名称、机构名称可保留英文）。
"""

_USER_PROMPT = """\
以下是招股书的风险/合规相关节选：

{context}

---
请提取完整的风险因素摘要。

注意：
- risks 列表按 priority 降序排列（priority=10 最先）。
- description_masked 中的掩码请使用 [法规条文] 和 [相关机构] 标记。
- policy_keywords 只填写会与外部宏观政策发生联动的词汇，不是泛泛的风险描述。
- overall_risk_score 和 policy_correlation_score 必须给出 [0.0, 1.0] 的数值。
"""


# ─────────────────────────────────────────────────────────────────
#  Extractor 类
# ─────────────────────────────────────────────────────────────────

class RiskExtractor(LLMExtractor["RiskSummary"]):
    """从招股书 PDF chunks 中提取风险与政策关联信息。"""

    KEYWORDS = RISK_KEYWORDS
    SYSTEM_PROMPT = _SYSTEM_PROMPT
    USER_PROMPT = _USER_PROMPT
    TOPIC = "policy"
    SUMMARY_CLASS = RiskSummary

    @staticmethod
    def get_content_for_agent(summary: RiskSummary, agent_type: str) -> str:
        """根据 Agent 类型返回对应层级的风险信息文本。

        HostAgent / InstTraderAgent → 高敏感版（完整细节）
        RetailTraderAgent           → 普通版（前5条 + 掩码）
        NormalAgent                 → "[此信息对你不可见]"
        """
        if agent_type in ("HostAgent", "InstTraderAgent"):
            return summary.high_sensitivity_version
        if agent_type == "RetailTraderAgent":
            return summary.normal_version
        return "[此信息对你不可见]"

    @staticmethod
    def get_policy_keywords(summary: RiskSummary) -> List[str]:
        """聚合所有风险条目的宏观政策关键词（去重），用于外部政策对撞分析。"""
        seen: set[str] = set()
        result: List[str] = []
        for item in summary.risks:
            for kw in item.policy_keywords:
                if kw not in seen:
                    seen.add(kw)
                    result.append(kw)
        return result
