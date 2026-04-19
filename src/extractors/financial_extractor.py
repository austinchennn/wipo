"""
financial_extractor — 财务数据深度解析器 (Financial Deep-Dive Extractor)

目标章节：Selected Financial Data、MD&A（管理层讨论与分析）、
          Financial Statements、Use of Proceeds

LLM 角色：高级审计师

输出两个信息层级：
  机构版（institution）→ HostAgent / InstTraderAgent — 结构化 JSON 完整财务报表
  散户版（retail）     → RetailTraderAgent          — 具体利润数字模糊化处理
  [不可见]            → NormalAgent               — 整个模块对普通人隐藏
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from .base_extractor import (
    FINANCIAL_KEYWORDS,
    LLMExtractor,
)


# ─────────────────────────────────────────────────────────────────
#  输出 Schema
# ─────────────────────────────────────────────────────────────────

class YearlyMetrics(BaseModel):
    """单一财年核心财务指标"""
    year: str = Field(description="财年标识，如 '2024' 或 'FY2024'")
    revenue: Optional[str] = Field(None, description="营收（含单位，如 '$1.2B'）")
    net_income: Optional[str] = Field(None, description="净利润（亏损用负数表示）")
    gross_margin: Optional[str] = Field(None, description="毛利率，如 '68.5%'")
    operating_cash_flow: Optional[str] = Field(None, description="经营活动现金流")
    yoy_revenue_growth: Optional[str] = Field(None, description="营收同比增速，如 '+45%'")
    yoy_profit_growth: Optional[str] = Field(None, description="净利润同比增速")


class MDAAnalysis(BaseModel):
    """管理层讨论与分析（MD&A）情绪分析"""
    tone: Literal["自信", "防御", "中性", "推诿", "谨慎乐观"] = Field(
        description="管理层整体措辞温度"
    )
    tone_evidence: str = Field(
        description="支撑该判断的原文关键短语（1-3 句，引号标注）"
    )
    key_explanations: str = Field(
        description="管理层对异常增长或下滑给出的主要解释（≤200字）"
    )


class FinancialSummary(BaseModel):
    """财务数据结构化摘要（含机构版和散户版）"""

    # ── 结构化字段（机器可读）──
    metrics_by_year: List[YearlyMetrics] = Field(
        description="按年度排列的财务指标（最多3个财年，从新到旧）"
    )
    mda_analysis: MDAAnalysis = Field(
        description="MD&A 章节情绪与内容分析"
    )
    use_of_proceeds: str = Field(
        description="募资用途简述（研发/并购/偿债/运营资本/创始人套现等）"
    )
    revenue_quality_score: float = Field(
        ge=0.0, le=1.0,
        description=(
            "收入质量评分：1.0=高质量（经常性收入为主、客户分散）；"
            "0.0=低质量（一次性收入、客户高度集中）"
        )
    )
    financial_health_score: float = Field(
        ge=0.0, le=1.0,
        description="财务健康度：1.0=现金充裕无债务，0.0=濒临资金链断裂"
    )

    # ── 机构版：完整结构化报表叙述 ──
    institution_version: str = Field(
        description=(
            "机构级完整版（≤600字）：包含所有精确数字（营收 $X.XXB、"
            "净利率 XX.X%、现金 $XXM）、三年完整对比、增长率计算结果、"
            "MD&A 情绪判断、募资用途合理性评价。"
        )
    )

    # ── 散户版：关键数字模糊化 ──
    retail_version: str = Field(
        description=(
            "散户版（≤400字）：将具体金额（如 '$45,230,122'）替换为"
            "近似描述（如 '约 4500 万美元'），将精确百分比（如 '68.3%'）"
            "替换为区间描述（如 '约 65-70%'），保留趋势判断和定性描述。"
            "示例：'利润同比增长约 20%' 而非 '$45,230,122 增长至 $54,123,456'。"
        )
    )


# ─────────────────────────────────────────────────────────────────
#  Prompts
# ─────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
你是一个高级审计师，专注于 IPO 招股书的财务数据深度分析。

你的任务是：
1. 从招股书节选中精确提取过去三年（或可用年份）的核心财务指标。
2. 自动计算同比（YoY）增长率（若原文未直接给出）。
3. 分析 MD&A 章节中管理层的措辞温度：他们在解释业绩时是自信、防御还是推诿？
4. 生成两个版本：
   - 机构版：精确数字，完整对比，量化所有指标
   - 散户版：将具体金额替换为近似描述，精确百分比替换为区间，但保留趋势和定性判断

重点关注：
- 收入质量（经常性 vs 一次性，客户集中度）
- 现金跑道（账面现金 ÷ 月均烧钱速率）
- 募资用途的合理性（研发 vs 套现）
- 管理层对下滑或异常的解释是否可信

输出语言：中文（财务术语、金额可保留英文数字格式）。
"""

_USER_PROMPT = """\
以下是招股书的财务/MD&A 相关节选：

{context}

---
请提取完整的财务数据摘要。

注意：
- 散户版中的金额近似原则：千万级别精度到"约 X 千万"，亿级精度到"约 X 亿"。
- 精确百分比用区间代替：实际值 ±5% 的范围。
- 若数据未在文本中出现，字段填 null，不要推算或虚构。
- revenue_quality_score 和 financial_health_score 需给出 [0.0, 1.0] 的精确数值。
"""


# ─────────────────────────────────────────────────────────────────
#  Extractor 类
# ─────────────────────────────────────────────────────────────────

class FinancialExtractor(LLMExtractor["FinancialSummary"]):
    """从招股书 PDF chunks 中提取财务数据。"""

    KEYWORDS = FINANCIAL_KEYWORDS
    SYSTEM_PROMPT = _SYSTEM_PROMPT
    USER_PROMPT = _USER_PROMPT
    TOPIC = "financial"
    SUMMARY_CLASS = FinancialSummary

    @staticmethod
    def get_content_for_agent(
        summary: FinancialSummary, agent_type: str
    ) -> str:
        """根据 Agent 类型返回对应层级的财务信息文本。

        HostAgent / InstTraderAgent → 机构版（完整数字）
        RetailTraderAgent           → 散户版（模糊数字）
        NormalAgent                 → "[此信息对你不可见]"（整个财务模块隐藏）
        """
        if agent_type in ("HostAgent", "InstTraderAgent"):
            return summary.institution_version
        if agent_type == "RetailTraderAgent":
            return summary.retail_version
        return "[此信息对你不可见]"
