"""
composition —— 唯一的装配点（composition root）

整个项目里只有这里知道「抽象」对应哪个「实现」：

    LLMProvider       → GeminiProvider / NullLLMProvider
    KnowledgeProvider → RAGSystem（FAISS 或静态模式）
    SimulationSink    → SimulationDB / None
    DecisionModel     → TypeSafeDecisionModel（Jev） / NullDecisionModel

其余模块只依赖 src/ports 里的 Protocol。调用这里的只有两个入口：
run.py（CLI）和 api/main.py（HTTP）。

哪些东西不注入：Exchange、TradingSession、各种 Agent 都是纯内存的领域
对象，没有外部依赖也没有 I/O，给它们做依赖反转只是加仪式感，不解耦
任何东西。这里只反转真正带 I/O 的三类：LLM、知识库、落库。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from .config import MAX_CONCURRENT_LLM_CALLS
from .decision import NullDecisionModel, TypeSafeDecisionModel
from .llm import GeminiProvider, NullLLMProvider
from .ports.decision import DecisionModel
from .ports.knowledge import KnowledgeProvider
from .ports.llm import LLMProvider
from .ports.persistence import SimulationSink
from .settings import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunOptions:
    """一次模拟的运行参数。

    原来这十来个参数在 run.py 和 api/main.py 各拼一遍，加一个要改两处
    （上个 PR 加 use_spider 就改了三处）。收敛到一个对象后只改这里。
    """

    n_normal: int = 20
    n_inst: int = 5
    n_retail: int = 15
    n_active: Optional[int] = None
    pdf_path: Optional[str] = None
    use_rag: bool = True
    use_graph: bool = False
    use_spider: bool = False
    persist: bool = True
    use_decision_model: bool = True   # 配了 TYPESAFE_API_KEY 时交易走 Jev
    max_concurrent: int = MAX_CONCURRENT_LLM_CALLS
    seed: Optional[int] = 42


# ═══════════════════════════════════════════════════════
#  各依赖的装配
# ═══════════════════════════════════════════════════════


def build_llm_provider(settings: Settings) -> LLMProvider:
    """有 Key 用 Gemini，没有就明确返回 NullLLMProvider。

    返回 Null 而不是"不可用的 Gemini"，是为了让日志和 repr 直接说清楚
    这次运行没有 LLM，而不是等到第一次调用返回 None 才发现。
    """
    provider = GeminiProvider(settings)
    if provider.is_available:
        return provider

    logger.warning("未检测到 GOOGLE_API_KEY：Agent 走 mock 评论，政策分类走关键词规则")
    return NullLLMProvider()


def build_decision_model(settings: Settings) -> DecisionModel:
    """配了 TYPESAFE_API_KEY 且 SDK 可用就用 Jev，否则明确返回 Null。

    Null 时交易决策整体退回规则，行为与接入 Jev 之前完全一致。
    """
    if settings.has_decision_model:
        model = TypeSafeDecisionModel()
        if model.is_available:
            return model
    logger.info("未启用 Jev（无 TYPESAFE_API_KEY 或未安装 langchain-typesafe）：交易走规则")
    return NullDecisionModel()


def build_knowledge(
    llm: LLMProvider,
    pdf_path: Optional[str] = None,
    use_rag: bool = True,
) -> KnowledgeProvider:
    """装配分层知识库。

    有 PDF 就跑 Extractor，没有就用内置 mock；能拿到 embedding 模型
    就建 FAISS 索引，拿不到降级成静态文本模式。
    这段判断原来在 ForumModel 里，属于装配逻辑而不是领域逻辑。
    """
    from .extractors.pipeline import extract_all_from_pdf, make_mock_extraction
    from .rag.knowledge_base import RAGSystem

    if pdf_path:
        result = extract_all_from_pdf(pdf_path, llm)
    else:
        result = make_mock_extraction()

    embeddings = llm.embeddings() if use_rag else None
    if embeddings is None:
        if use_rag and pdf_path:
            logger.warning("无可用 embedding 模型，RAG 降级为静态模式")
        return RAGSystem.build_static_only(result)

    rag = RAGSystem.build_from_extraction(result, embeddings)
    logger.info(rag.status())
    return rag


def build_sink(
    settings: Settings, enabled: bool = True
) -> Optional[SimulationSink]:
    """落库目标；enabled=False 返回 None（测试和一次性跑分用）。"""
    if not enabled:
        return None
    from .persistence.database import SimulationDB

    return SimulationDB(settings.simulation_db_path)


# ═══════════════════════════════════════════════════════
#  整场模拟
# ═══════════════════════════════════════════════════════


def build_simulation(settings: Settings, opts: RunOptions):
    """装配一个可直接 arun() 的 ForumModel。"""
    from .environment.forum import ForumModel

    llm = build_llm_provider(settings)

    return ForumModel(
        knowledge=build_knowledge(llm, opts.pdf_path, opts.use_rag),
        llm=llm,
        sink=build_sink(settings, opts.persist),
        decider=(build_decision_model(settings)
                 if opts.use_decision_model else None),
        policy_db_path=settings.policy_db_path,
        n_normal=opts.n_normal,
        n_inst=opts.n_inst,
        n_retail=opts.n_retail,
        n_active=opts.n_active,
        max_concurrent=opts.max_concurrent,
        use_graph=opts.use_graph,
        use_spider=opts.use_spider,
        seed=opts.seed,
    )
