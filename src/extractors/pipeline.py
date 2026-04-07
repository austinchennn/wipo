"""
pipeline — 三个 Extractor 的编排器。

从一份 PDF 一次性跑完 ProductExtractor / FinancialExtractor / RiskExtractor，
返回 ExtractionResult 供 ForumModel 直接消费。

用法：
    from src.extractors import extract_all_from_pdf

    result = extract_all_from_pdf("prospectus.pdf")

    # 主持人发帖用的三段原始文本（Level A / 机构版 / 高敏感版）
    raw_sections = result.raw_sections

    # RAG 知识库索引用的 Document chunks（按 topic 分类）
    product_chunks  = result.product_chunks
    financial_chunks = result.financial_chunks
    risk_chunks     = result.risk_chunks
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.documents import Document

from .base_extractor import load_pdf_chunks
from .financial_extractor import FinancialExtractor, FinancialSummary
from .product_extractor import ProductExtractor, ProductSummary
from .risk_extractor import RiskExtractor, RiskSummary


# ─────────────────────────────────────────────────────────────────
#  结果容器
# ─────────────────────────────────────────────────────────────────

@dataclass
class ExtractionResult:
    """三个 Extractor 运行结果的统一容器。"""

    # ── 结构化摘要 ──
    product_summary: ProductSummary
    financial_summary: FinancialSummary
    risk_summary: RiskSummary

    # ── RAG 索引用 Document 列表（已打好 topic 标签）──
    product_chunks: List[Document] = field(default_factory=list)
    financial_chunks: List[Document] = field(default_factory=list)
    risk_chunks: List[Document] = field(default_factory=list)

    @property
    def raw_sections(self) -> Dict[str, str]:
        """
        生成 ForumModel 所需的 raw_sections 字典。

        键名对应 ForumModel.TOPICS：
            "product"   → ProductSummary.level_a（主持人/机构全文）
            "financial" → FinancialSummary.institution_version
            "policy"    → RiskSummary.high_sensitivity_version
        """
        return {
            "product":   self.product_summary.level_a,
            "financial": self.financial_summary.institution_version,
            "policy":    self.risk_summary.high_sensitivity_version,
        }

    def get_section_for_agent(
        self, topic: str, agent_type: str
    ) -> str:
        """
        根据 Agent 类型和话题返回对应层级的文本。

        对应可见性规则：
            product:   Host/Inst → Level A | Normal → Level C | Retail → Level B
            financial: Host/Inst → 机构版 | Retail → 散户版   | Normal → 不可见
            policy:    Host/Inst → 高敏感 | Retail → 普通版   | Normal → 不可见
        """
        if topic == "product":
            return ProductExtractor.get_content_for_agent(
                self.product_summary, agent_type
            )
        if topic == "financial":
            return FinancialExtractor.get_content_for_agent(
                self.financial_summary, agent_type
            )
        if topic == "policy":
            return RiskExtractor.get_content_for_agent(
                self.risk_summary, agent_type
            )
        return "[未知话题]"

    def all_chunks(self) -> List[Document]:
        """所有 chunks 的合并列表（调试 / 整体 RAG 用）。"""
        return self.product_chunks + self.financial_chunks + self.risk_chunks


# ─────────────────────────────────────────────────────────────────
#  主入口函数
# ─────────────────────────────────────────────────────────────────

def extract_all_from_pdf(
    pdf_path: str | Path,
    model: str = "gpt-4o-mini",
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
    verbose: bool = True,
) -> ExtractionResult:
    """从 PDF 招股书中一次性提取三类信息。

    参数：
        pdf_path     — 用户上传的 PDF 路径
        model        — OpenAI 模型名（gpt-4o-mini / gpt-4o）
        chunk_size   — 分块大小（字符数）
        chunk_overlap — 分块重叠（字符数）
        verbose      — 是否打印进度

    返回：
        ExtractionResult（含三份摘要 + 三份 RAG chunks）
    """
    pdf_path = Path(pdf_path)

    # ── 加载环境变量（如果有 .env 文件）──
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    _log = print if verbose else lambda *a, **k: None

    _log(f"\n{'─' * 55}")
    _log(f"  招股书解析启动: {pdf_path.name}")
    _log(f"{'─' * 55}")

    # Step 1: PDF → chunks
    _log("  [1/4] PDF 分块...")
    all_chunks = load_pdf_chunks(
        pdf_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    _log(f"        共 {len(all_chunks)} 个 chunk（{chunk_size}字/块）")

    # Step 2: 产品与业务提取
    _log("  [2/4] 提取产品与业务信息 (ProductExtractor)...")
    product_extractor = ProductExtractor(model=model)
    product_summary, product_chunks = product_extractor.extract(all_chunks)
    _log(f"        命中 {len(product_chunks)} 个产品相关 chunk")

    # Step 3: 财务数据提取
    _log("  [3/4] 提取财务数据 (FinancialExtractor)...")
    financial_extractor = FinancialExtractor(model=model)
    financial_summary, financial_chunks = financial_extractor.extract(all_chunks)
    _log(f"        命中 {len(financial_chunks)} 个财务相关 chunk")

    # Step 4: 风险与政策提取
    _log("  [4/4] 提取风险与政策信息 (RiskExtractor)...")
    risk_extractor = RiskExtractor(model=model)
    risk_summary, risk_chunks = risk_extractor.extract(all_chunks)
    _log(f"        命中 {len(risk_chunks)} 个风险相关 chunk")
    _log(
        f"        综合风险评分: {risk_summary.overall_risk_score:.2f} | "
        f"政策关联度: {risk_summary.policy_correlation_score:.2f}"
    )

    _log(f"{'─' * 55}")
    _log("  解析完成 ✓")
    _log(f"{'─' * 55}\n")

    return ExtractionResult(
        product_summary=product_summary,
        financial_summary=financial_summary,
        risk_summary=risk_summary,
        product_chunks=product_chunks,
        financial_chunks=financial_chunks,
        risk_chunks=risk_chunks,
    )


# ─────────────────────────────────────────────────────────────────
#  Mock 结果（测试用，不需要 PDF 和 API Key）
# ─────────────────────────────────────────────────────────────────

def make_mock_extraction() -> ExtractionResult:
    """生成硬编码的 Mock ExtractionResult，用于无 PDF / 无 API Key 时的本地测试。"""
    from .product_extractor import ProductSummary
    from .financial_extractor import FinancialSummary, MDAAnalysis, YearlyMetrics
    from .risk_extractor import RiskSummary, RiskItem

    product = ProductSummary(
        core_products=["AI 芯片", "数据中心加速卡", "边缘推理模块"],
        tech_keywords=["分布式架构", "生成式AI加速", "TSMC 3nm工艺", "PCIe 5.0"],
        market_position="全球 AI 训练芯片市占率约 8%，行业第三",
        competitive_advantage="自研架构的每瓦性能领先竞品 40%，已形成软件生态护城河",
        main_competitors=["NVIDIA", "AMD", "Intel Gaudi"],
        level_a=(
            "【产品信息·完整版】该公司主营 AI 芯片设计，核心产品覆盖数据中心训练卡"
            "与边缘推理模块。采用 TSMC 3nm 制程、自研分布式计算架构，每瓦性能较"
            "竞品高出约 40%。全球 AI 训练芯片市占率约 8%（行业第三），已获 23 项核心"
            "专利。主要竞争对手：NVIDIA H200、AMD MI300X、Intel Gaudi 3。"
        ),
        level_b=(
            "【产品信息·掩码版】该公司主营 AI 芯片，产品覆盖数据中心与边缘计算。"
            "采用先进制程工艺，性能处于行业前列，全球市占率约 ████%（行业第三），"
            "持有 ████ 项专利。"
        ),
        level_c=(
            "这家公司做 AI 芯片，芯片用来让计算机更快地运行人工智能程序，"
            "是这个领域全球排名前三的公司之一。"
        ),
    )

    financial = FinancialSummary(
        metrics_by_year=[
            YearlyMetrics(
                year="2024", revenue="$12亿", net_income="$2.16亿",
                gross_margin="68%", operating_cash_flow="$1.8亿",
                yoy_revenue_growth="+45%", yoy_profit_growth="+38%",
            ),
            YearlyMetrics(
                year="2023", revenue="$8.3亿", net_income="$1.57亿",
                gross_margin="65%", operating_cash_flow="$1.2亿",
                yoy_revenue_growth="+62%", yoy_profit_growth="+55%",
            ),
        ],
        mda_analysis=MDAAnalysis(
            tone="谨慎乐观",
            tone_evidence='"我们预期增长势头将延续，但承认地缘政治不确定性是主要变量"',
            key_explanations="管理层将 2024Q3 增速放缓归因于客户库存消化周期，预计 2025H1 恢复加速。",
        ),
        use_of_proceeds="40% 研发（下一代架构），35% 销售网络拓展，25% 运营资本",
        revenue_quality_score=0.72,
        financial_health_score=0.80,
        institution_version=(
            "【财务信息·机构版】2024 年营收 $12 亿（同比+45%），净利率 18%，毛利率 68%。"
            "三年复合增速 53%，现金储备 $3.2 亿，无长期债务。管理层措辞谨慎乐观，"
            "将短期放缓归因于客户库存消化，属于行业共性问题。募资用途以研发为主，"
            "无明显创始人套现迹象。收入质量评分 0.72（前五大客户占比 38%）。"
        ),
        retail_version=(
            "【财务信息·散户版】公司营收约 10 多亿美元，同比增长约 40-50%，"
            "利润率约 65-70%，公司账上有几亿现金且没有明显债务压力。"
            "管理层整体表现乐观，增速阶段性放缓是行业普遍情况。"
            "募资主要用于研发和业务扩展。"
        ),
    )

    risk = RiskSummary(
        risks=[
            RiskItem(
                category="行业监管风险",
                priority=10,
                title="美国出口管制实体清单风险",
                description_full="美国商务部 BIS 已于 2025Q4 扩大对华先进芯片出口限制，"
                                 "包含 A100/H100 等型号，公司产品性能参数接近管控阈值，"
                                 "存在被纳入限制的法律风险。",
                description_masked="[相关机构] 已扩大对华先进芯片出口限制，"
                                   "公司产品性能参数接近 [管控阈值]，存在被限制的风险。",
                policy_keywords=["出口管制", "实体清单", "对华限制", "BIS"],
                is_enacted_risk=True,
            ),
            RiskItem(
                category="市场竞争风险",
                priority=7,
                title="NVIDIA 市场地位压制",
                description_full="NVIDIA CUDA 生态系统具有强大的软件锁定效应，"
                                 "客户迁移成本高，公司需投入大量资源建立替代软件栈。",
                description_masked="行业龙头具有强大的软件锁定效应，客户迁移成本高，"
                                   "公司需持续投入资源建立差异化竞争力。",
                policy_keywords=["反垄断", "技术标准"],
                is_enacted_risk=False,
            ),
        ],
        top_risk_category="行业监管风险",
        overall_risk_score=0.72,
        policy_correlation_score=0.85,
        high_sensitivity_version=(
            "【政策风险·完整版】综合风险评分 0.72，政策关联度高（0.85）。\n"
            "最高优先级风险：美国 BIS 出口管制扩大，公司产品参数接近管控阈值，"
            "若被列入实体清单将直接影响约 38% 来自中国区的营收。\n"
            "其他重大风险：NVIDIA 生态护城河导致客户迁移成本高企。\n"
            "政策关联词：出口管制、实体清单、对华限制、反垄断。"
        ),
        normal_version=(
            "【政策风险·普通版】主要风险（前2条）：\n"
            "1. [相关机构] 扩大先进芯片出口限制，公司产品存在被限制的法律风险（高危）\n"
            "2. 行业竞争壁垒高，软件生态锁定效应强，竞争压力持续存在（中危）"
        ),
    )

    return ExtractionResult(
        product_summary=product,
        financial_summary=financial,
        risk_summary=risk,
        product_chunks=[],
        financial_chunks=[],
        risk_chunks=[],
    )
