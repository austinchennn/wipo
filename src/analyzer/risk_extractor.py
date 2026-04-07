"""risk_extractor — 已迁移至 src/extractors/risk_extractor.py

保留此文件仅为向后兼容。请直接使用：

    from src.extractors import RiskExtractor, extract_all_from_pdf
"""

from ..extractors.risk_extractor import RiskExtractor, RiskSummary, RiskItem

__all__ = ["RiskExtractor", "RiskSummary", "RiskItem"]
