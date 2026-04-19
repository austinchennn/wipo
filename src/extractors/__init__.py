"""extractors — 从用户上传的招股书 PDF 中用 LLM 提取三类结构化信息。

三个 Extractor：
    ProductExtractor   — 产品与业务（Level A/B/C 三层级）
    FinancialExtractor — 财务数据（机构版 / 散户版）
    RiskExtractor      — 风险与政策（高敏感版 / 普通版）

快速使用：
    from src.extractors import extract_all_from_pdf, make_mock_extraction

    # 真实 PDF（需要 GOOGLE_API_KEY）
    result = extract_all_from_pdf("prospectus.pdf")

    # 本地测试（无需 PDF 和 API Key）
    result = make_mock_extraction()

    # result.raw_sections       → ForumModel 主帖用的三段全文
    # result.product_chunks     → Product RAG KB 的 Document 列表
    # result.financial_chunks   → Financial RAG KB 的 Document 列表
    # result.risk_chunks        → Risk RAG KB 的 Document 列表
    # result.get_section_for_agent(topic, agent_type) → 按 Agent 类型返回对应层级
"""

from .base_extractor import load_pdf_chunks, get_llm
from .product_extractor import ProductExtractor, ProductSummary
from .financial_extractor import FinancialExtractor, FinancialSummary
from .risk_extractor import RiskExtractor, RiskSummary
from .pipeline import extract_all_from_pdf, make_mock_extraction, ExtractionResult

__all__ = [
    "load_pdf_chunks",
    "get_llm",
    "ProductExtractor",
    "ProductSummary",
    "FinancialExtractor",
    "FinancialSummary",
    "RiskExtractor",
    "RiskSummary",
    "extract_all_from_pdf",
    "make_mock_extraction",
    "ExtractionResult",
]
