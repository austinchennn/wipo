"""
base_extractor — PDF 加载、分块、LLM 工厂。

所有 Extractor 的公共基础设施：
  - load_pdf_chunks()       : pdfplumber 读取 PDF → Document 列表
  - filter_chunks_by_keywords(): 关键词过滤定位目标章节
  - get_llm()               : 构建 ChatGoogleGenerativeAI 实例
  - chunks_to_context()     : 拼合文本供 LLM 消费
"""

from __future__ import annotations

import os
from copy import copy
from pathlib import Path
from typing import Generic, List, Tuple, TypeVar

import pdfplumber
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import CHUNK_OVERLAP, CHUNK_SIZE, EXTRACTOR_LLM_TEMPERATURE, MAX_CONTEXT_CHARS


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
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
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
    model: str = "gemini-2.5-flash",
    temperature: float = EXTRACTOR_LLM_TEMPERATURE,
) -> ChatGoogleGenerativeAI:
    """构建 ChatGoogleGenerativeAI 实例（从环境变量读取 GOOGLE_API_KEY）。"""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "缺少 GOOGLE_API_KEY，请在 .env 或 Shell 中配置。"
        )
    return ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=api_key)


# ─────────────────────────────────────────────────────────────────
#  Chunks → 单段上下文文本
# ─────────────────────────────────────────────────────────────────

def chunks_to_context(
    chunks: List[Document],
    max_chars: int = MAX_CONTEXT_CHARS,
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


# ─────────────────────────────────────────────────────────────────
#  通用 LLM Extractor 基类
# ─────────────────────────────────────────────────────────────────

T = TypeVar("T")


class LLMExtractor(Generic[T]):
    """三个 Extractor 的通用基类，消除重复的 __init__ 和 extract 逻辑。

    子类只需定义以下类变量：
        KEYWORDS      — 关键词过滤列表
        SYSTEM_PROMPT — 系统提示词
        USER_PROMPT   — 用户提示词模板（含 {context} 占位符）
        TOPIC         — 话题标签（写入 Document.metadata["topic"]）
        SUMMARY_CLASS — Pydantic 输出 Schema 类
    """

    KEYWORDS: List[str] = []
    SYSTEM_PROMPT: str = ""
    USER_PROMPT: str = ""
    TOPIC: str = ""
    SUMMARY_CLASS: type = None  # type: ignore[assignment]

    def __init__(self, model: str = "gemini-2.5-flash") -> None:
        llm = get_llm(model=model)
        self._structured_llm = llm.with_structured_output(self.SUMMARY_CLASS)

    def extract(
        self,
        all_chunks: List[Document],
        min_keyword_hits: int = 1,
    ) -> Tuple[T, List[Document]]:
        """关键词过滤 → 上下文拼合 → LLM 结构化提取 → 打 topic 标签。

        返回带有 topic 元数据标签的新 Document 列表（不修改 all_chunks 原始对象）。
        """
        filtered = filter_chunks_by_keywords(
            all_chunks, self.KEYWORDS, min_hits=min_keyword_hits
        )
        context = chunks_to_context(filtered, max_chars=12_000)
        messages = [
            ("system", self.SYSTEM_PROMPT),
            ("human", self.USER_PROMPT.format(context=context)),
        ]
        summary: T = self._structured_llm.invoke(messages)

        # 拷贝 Document 再打标签，避免并行提取时修改共享对象
        tagged: List[Document] = []
        for doc in filtered:
            new_doc = copy(doc)
            new_doc.metadata = {**doc.metadata, "topic": self.TOPIC}
            tagged.append(new_doc)

        return summary, tagged
