"""
base_extractor — PDF 加载、分块、LLM 工厂。

所有 Extractor 的公共基础设施：
  - load_pdf_chunks()       : pdfplumber 读取 PDF → Document 列表
  - filter_chunks_by_keywords(): 关键词过滤定位目标章节
  - get_llm()               : 构建 ChatOpenAI 实例
  - chunks_to_context()     : 拼合文本供 LLM 消费
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import pdfplumber
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ─────────────────────────────────────────────────────────────────
#  章节定位关键词
# ─────────────────────────────────────────────────────────────────

PRODUCT_KEYWORDS = [
    # 中文
    "产品", "业务", "主营", "技术", "研发", "市场", "竞争", "客户",
    "供应链", "专利", "知识产权", "品牌", "运营", "行业", "解决方案",
    # 英文
    "business", "product", "technology", "market", "competition",
    "customer", "r&d", "research", "intellectual property", "patent",
    "supply chain", "industry", "solution", "service",
]

FINANCIAL_KEYWORDS = [
    # 中文
    "营收", "收入", "利润", "净利", "毛利", "现金流", "资产", "负债",
    "募资", "财务", "审计", "管理层讨论", "增长", "股东权益",
    # 英文
    "revenue", "income", "profit", "ebitda", "cash flow", "balance sheet",
    "assets", "liabilities", "equity", "gross margin", "net income",
    "selected financial data", "md&a", "management discussion",
    "use of proceeds", "dilution", "quarterly", "fiscal year",
]

RISK_KEYWORDS = [
    # 中文
    "风险", "监管", "合规", "诉讼", "法规", "政策", "反垄断",
    "数据安全", "出口管制", "资本管制", "许可", "审批", "政府",
    # 英文
    "risk", "risk factors", "regulatory", "regulation", "compliance",
    "litigation", "legal proceedings", "government", "antitrust",
    "data security", "export control", "license", "approval", "sec",
]


# ─────────────────────────────────────────────────────────────────
#  PDF 加载 + 分块
# ─────────────────────────────────────────────────────────────────

def load_pdf_chunks(
    pdf_path: str | Path,
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
) -> List[Document]:
    """用 pdfplumber 读取 PDF，RecursiveCharacterTextSplitter 分块。

    返回 LangChain Document 列表，每条含 page / source 元数据。
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

    raw_pages: List[Document] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text = (page.extract_text() or "").strip()
            if text:
                raw_pages.append(
                    Document(
                        page_content=text,
                        metadata={"source": str(pdf_path), "page": i},
                    )
                )

    if not raw_pages:
        raise ValueError(f"PDF 解析结果为空，请检查文件: {pdf_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ". ", " ", ""],
    )
    return splitter.split_documents(raw_pages)


def filter_chunks_by_keywords(
    chunks: List[Document],
    keywords: List[str],
    min_hits: int = 1,
) -> List[Document]:
    """保留至少命中 min_hits 个关键词的 chunk（不区分大小写）。"""
    result = []
    for doc in chunks:
        text_lower = doc.page_content.lower()
        hits = sum(1 for kw in keywords if kw.lower() in text_lower)
        if hits >= min_hits:
            result.append(doc)
    return result or chunks  # 无命中时回退全量


# ─────────────────────────────────────────────────────────────────
#  LLM 工厂
# ─────────────────────────────────────────────────────────────────

def get_llm(
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> ChatOpenAI:
    """构建 ChatOpenAI 实例（从环境变量读取 OPENAI_API_KEY）。"""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "缺少 OPENAI_API_KEY，请在 .env 或 Shell 中配置。"
        )
    return ChatOpenAI(model=model, temperature=temperature, api_key=api_key)


# ─────────────────────────────────────────────────────────────────
#  Chunks → 单段上下文文本
# ─────────────────────────────────────────────────────────────────

def chunks_to_context(
    chunks: List[Document],
    max_chars: int = 12_000,
) -> str:
    """将多个 chunk 拼成一段文本，超出 max_chars 时截断。"""
    parts: List[str] = []
    total = 0
    for i, doc in enumerate(chunks, 1):
        header = f"[段落{i} · 第{doc.metadata.get('page', '?')}页]\n"
        segment = header + doc.page_content.strip() + "\n\n"
        if total + len(segment) > max_chars:
            parts.append(
                f"[... 剩余内容截断，共 {len(chunks)} 段落 ...]"
            )
            break
        parts.append(segment)
        total += len(segment)
    return "".join(parts)
