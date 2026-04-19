"""
论坛沙盘主模型 —— ForumModel

调度逻辑:
  12 宏观轮次 (Macro-Round)
    × 3 帖子 (产品 → 财务 → 政策)
      × 3 Phase (直面楼主 → 互攻/抱团 → 余波)

Agent 分层：
  Active Agents  (n_active)  — 实际调 LLM，产生可见评论，并发执行
  Passive Agents (n_total - n_active) — 不调 LLM，Phase 结束后统计推算情绪

每个 Phase 结束后生成 SentimentGrid，覆盖全部 n_total 个 Agent 的情绪状态。
"""

from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from mesa import Model

from ..agents.base_agent import (
    BaseUserAgent,
    CommentResult,
    HostAgent,
    InstTraderAgent,
    NormalAgent,
    RetailTraderAgent,
)
from ..agents.passive_inference import infer_passive_sentiment
from ..config import MAX_CONCURRENT_LLM_CALLS
from ..market.exchange import Exchange
from ..market.trading_agent import TradingSession
from ..models import Comment, Post, Sentiment, SentimentGrid, ThreadSnapshot

logger = logging.getLogger(__name__)


class ForumModel(Model):
    """金融舆情论坛沙盘 —— Mesa Model"""

    MACRO_ROUNDS = 12
    TOPICS = ("product", "financial", "policy")
    TOPIC_CN: Dict[str, str] = {
        "product":   "产品信息",
        "financial": "财务信息",
        "policy":    "政策信息",
    }

    def __init__(
        self,
        n_normal: int = 20,
        n_inst: int = 5,
        n_retail: int = 15,
        n_active: Optional[int] = None,
        raw_sections: Optional[Dict[str, str]] = None,
        pdf_path: Optional[str] = None,
        llm_model: str = "gemini-2.5-flash",
        use_rag: bool = True,
        max_concurrent: int = MAX_CONCURRENT_LLM_CALLS,
        seed: Optional[int] = None,
    ):
        """
        参数：
            n_normal      — 普通 Agent 总数
            n_inst        — 机构 Agent 总数
            n_retail      — 散户 Agent 总数
            n_active      — 实际调 LLM 的 Agent 数量上限
                            None = 全部 Active（适合小规模测试）
                            整数 = 按比例从各类型中抽取 Active，其余为 Passive
            raw_sections  — 手动传入三段文本（向后兼容旧接口）
            pdf_path      — 用户上传的招股书 PDF 路径
            llm_model     — Extractor 使用的 LLM 模型
            use_rag       — 是否构建 FAISS RAG 知识库
            max_concurrent — 并发 LLM 调用上限（防止 rate limit）
            seed          — 随机种子
        """
        super().__init__(seed=seed)

        self.n_normal = n_normal
        self.n_inst = n_inst
        self.n_retail = n_retail
        self.max_concurrent = max_concurrent
        self.rag_system = None

        # ── 事件回调（WebSocket 实时推送用）──
        # 签名: async def on_event(event: Dict[str, Any]) -> None
        self.on_event: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None

        # ── 信息来源 ──
        if pdf_path is not None:
            self.raw_sections = self._load_from_pdf(
                pdf_path, llm_model=llm_model, use_rag=use_rag
            )
        elif raw_sections is not None:
            self.raw_sections = raw_sections
        else:
            self.raw_sections = self._load_mock()

        # ── 创建全量 Agents（先全部 is_active=True）──
        self.host = HostAgent(self)

        all_normal  = [NormalAgent(self)      for _ in range(n_normal)]
        all_inst    = [InstTraderAgent(self)   for _ in range(n_inst)]
        all_retail  = [RetailTraderAgent(self) for _ in range(n_retail)]
        all_agents  = all_normal + all_inst + all_retail

        # ── 按 n_active 分层 ──
        n_total = len(all_agents)
        if n_active is None or n_active >= n_total:
            # 全部 Active
            self.active_participants:  List[BaseUserAgent] = all_agents
            self.passive_participants: List[BaseUserAgent] = []
        else:
            # 按类型比例抽取 Active
            active_normal  = self._sample(all_normal,  n_active, n_normal,  n_total)
            active_inst    = self._sample(all_inst,    n_active, n_inst,    n_total)
            active_retail  = self._sample(all_retail,  n_active, n_retail,  n_total)

            active_set = set(id(a) for a in active_normal + active_inst + active_retail)

            self.active_participants  = [a for a in all_agents if id(a) in active_set]
            self.passive_participants = [a for a in all_agents if id(a) not in active_set]

            for a in self.passive_participants:
                a.is_active = False

        # ── 全 Agent ID 列表（情绪矩阵排序用）──
        self.all_agent_ids: List[int] = [a.unique_id for a in all_agents]

        # ── 交易所 ──
        self.exchange = Exchange()
        for a in all_agents:
            self.exchange.init_portfolio(a.unique_id, a.__class__.__name__)

        # ── 存档 ──
        self.posts: List[Post] = []
        self.sentiment_history: List[SentimentGrid] = []
        self.round_num: int = 0

    # ─────────────────── 工具：按比例抽样 ───────────────────

    async def _emit(self, event: Dict[str, Any]) -> None:
        """触发事件回调（如果已设置）。"""
        if self.on_event is not None:
            try:
                await self.on_event(event)
            except Exception as e:
                logger.debug("事件推送失败: %s", e)

    @staticmethod
    def _sample(
        pool: List[BaseUserAgent],
        n_active: int,
        type_count: int,
        total: int,
    ) -> List[BaseUserAgent]:
        """按类型占比从 pool 中随机抽取 Active Agent。"""
        k = max(1, round(n_active * type_count / total))
        k = min(k, len(pool))
        return random.sample(pool, k)

    # ═══════════════════════════════════════════════════
    #  信息来源初始化
    # ═══════════════════════════════════════════════════

    def _load_from_pdf(
        self, pdf_path: str, llm_model: str, use_rag: bool
    ) -> Dict[str, str]:
        from ..extractors.pipeline import extract_all_from_pdf
        from ..rag.knowledge_base import RAGSystem

        result = extract_all_from_pdf(pdf_path, model=llm_model)

        if use_rag:
            try:
                self.rag_system = RAGSystem.build_from_extraction(result)
                logger.info(self.rag_system.status())
            except EnvironmentError as e:
                logger.warning("RAG 构建失败（%s），降级为静态模式", e)
                self.rag_system = RAGSystem.build_static_only(result)
        else:
            self.rag_system = RAGSystem.build_static_only(result)

        return result.raw_sections

    def _load_mock(self) -> Dict[str, str]:
        from ..extractors.pipeline import make_mock_extraction
        from ..rag.knowledge_base import RAGSystem

        result = make_mock_extraction()
        self.rag_system = RAGSystem.build_static_only(result)
        return result.raw_sections

    # ═══════════════════════════════════════════════════
    #  Agent 可见上下文
    # ═══════════════════════════════════════════════════

    def get_agent_context(
        self,
        agent: BaseUserAgent,
        topic: str,
        query: str = "",
    ) -> str:
        if self.rag_system is None:
            return agent.get_visible_content(self.raw_sections, topic)

        agent_type = agent.__class__.__name__

        if query and self.rag_system._kbs.get(topic) and \
                self.rag_system._kbs[topic].is_ready:
            return self.rag_system.retrieve_for_agent(agent, topic, query)

        return self.rag_system.get_static_section(topic, agent_type)

    # ═══════════════════════════════════════════════════
    #  异步并发 Phase 执行器
    # ═══════════════════════════════════════════════════

    async def _run_agents_async(
        self,
        agents: List[BaseUserAgent],
        context_fn,                          # agent → str
        candidates: Optional[List[Comment]] = None,
    ) -> List[Tuple[BaseUserAgent, CommentResult]]:
        """
        并发调用所有 Active Agent 的 acomment()。
        semaphore 限制最大并发数，防止 rate limit。
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def call_one(agent: BaseUserAgent):
            async with semaphore:
                context = context_fn(agent)
                return agent, await agent.acomment(context, candidates)

        tasks = [call_one(a) for a in agents]
        raw = await asyncio.gather(*tasks, return_exceptions=True)

        # 把 Exception 包装成 mock result，保证后续处理不崩
        results: List[Tuple[BaseUserAgent, CommentResult]] = []
        for agent, res in zip(agents, raw):
            if isinstance(res, Exception):
                results.append((agent, agent._mock_comment(candidates)))
            else:
                results.append(res)
        return results

    # ═══════════════════════════════════════════════════
    #  SentimentGrid 构建
    # ═══════════════════════════════════════════════════

    def _build_sentiment_grid(
        self,
        round_num: int,
        topic: str,
        phase: int,
        active_results: List[Tuple[BaseUserAgent, CommentResult]],
    ) -> SentimentGrid:
        """
        组装 SentimentGrid：
          - Active agents 的情绪从 CommentResult 读取（未发言的记为 neutral）
          - Passive agents 的情绪由 passive_inference 推算
        """
        grid: Dict[int, Sentiment] = {}

        # Active
        for agent, result in active_results:
            sentiment = (
                result.sentiment if agent.should_comment(result.temp)
                else "neutral"
            )
            grid[agent.unique_id] = sentiment

        # Passive
        if self.passive_participants:
            passive_grid = infer_passive_sentiment(
                self.passive_participants, active_results
            )
            grid.update(passive_grid)

        return SentimentGrid(
            round_num=round_num,
            topic=topic,
            phase=phase,
            grid=grid,
        )

    # ═══════════════════════════════════════════════════
    #  主循环
    # ═══════════════════════════════════════════════════

    def run(self):
        """同步入口：整个模拟在一个 asyncio event loop 中完成（仅调用一次）。"""
        asyncio.run(self.arun())

    async def arun(self):
        """异步主循环 —— 消除了每个 Phase 独立创建 event loop 的开销。"""
        self._started_at = datetime.now().isoformat()

        n_active  = len(self.active_participants)
        n_passive = len(self.passive_participants)
        n_total   = n_active + n_passive + 1  # +1 for host

        logger.info(
            "金融舆情论坛沙盘启动 | 总Agent=%d (Active=%d Passive=%d Host=1) "
            "普通=%d 机构=%d 散户=%d",
            n_total, n_active, n_passive,
            sum(1 for a in self.active_participants if isinstance(a, NormalAgent)),
            sum(1 for a in self.active_participants if isinstance(a, InstTraderAgent)),
            sum(1 for a in self.active_participants if isinstance(a, RetailTraderAgent)),
        )

        await self._emit({
            "type": "system",
            "event": "sim_start",
            "n_total": n_total,
            "n_active": n_active,
            "n_passive": n_passive,
            "rounds": self.MACRO_ROUNDS,
        })

        for r in range(self.MACRO_ROUNDS):
            self.round_num = r + 1
            logger.info("═══ 宏观轮次 %d / %d ═══", self.round_num, self.MACRO_ROUNDS)
            await self._emit({
                "type": "system",
                "event": "round_start",
                "round": self.round_num,
                "total_rounds": self.MACRO_ROUNDS,
            })
            await self.astep()

        total_comments = sum(
            len(p.comments[ph]) for p in self.posts for ph in (1, 2, 3)
        )
        logger.info(
            "模拟结束 | 帖子=%d 评论=%d SentimentGrid快照=%d",
            len(self.posts), total_comments, len(self.sentiment_history),
        )

        # 交易所汇总
        ms = self.exchange.market_summary()
        logger.info(
            "市场汇总 | 最新价=%.2f IPO价=%.2f 涨跌=%.2f%% "
            "总成交=%d股 成交笔数=%d K线=%d根",
            ms["last_price"], ms["ipo_price"], ms["change_pct"],
            ms["total_volume"], ms["total_trades"], ms["ticks"],
        )

        # ── 持久化：模拟结束后一次性写入 SQLite ──
        try:
            from ..persistence.database import SimulationDB
            db = SimulationDB()
            sim_id = db.save(self)
            db.close()
            logger.info("模拟数据已持久化 | simulation_id=%d", sim_id)
        except Exception as e:
            logger.warning("持久化失败（不影响模拟结果）: %s", e)

        await self._emit({
            "type": "system",
            "event": "sim_end",
            "total_posts": len(self.posts),
            "total_comments": total_comments,
            "market": ms,
        })

    def step(self):
        """Mesa API 兼容接口（单轮同步执行）。完整模拟请使用 run()。"""
        asyncio.run(self.astep())

    async def astep(self):
        for topic in self.TOPICS:
            post = self._host_publish(topic)
            await self._run_thread(post)

    # ═══════════════════════════════════════════════════
    #  发帖
    # ═══════════════════════════════════════════════════

    def _host_publish(self, topic: str) -> Post:
        content = self.raw_sections.get(topic, "[内容未加载]")
        post = Post(
            id=f"R{self.round_num}_{topic}",
            round_num=self.round_num,
            topic=topic,
            content=content,
            created_at=datetime.now(),
        )
        self.posts.append(post)
        logger.info("[发帖] %s  主题: %s", post.id, self.TOPIC_CN[topic])
        return post

    # ═══════════════════════════════════════════════════
    #  帖子生命周期: 3 Phase
    # ═══════════════════════════════════════════════════

    async def _run_thread(self, post: Post):
        await self._emit({
            "type": "post",
            "post_id": post.id,
            "round": post.round_num,
            "topic": post.topic,
            "topic_cn": self.TOPIC_CN[post.topic],
            "content": post.content[:500],
            "created_at": post.created_at.isoformat() if post.created_at else None,
        })

        snap1 = await self._phase_1(post)
        snap2 = await self._phase_2(post, snap1)
        await self._phase_3(post, snap2)

        p1, p2, p3 = (len(post.comments[i]) for i in (1, 2, 3))
        logger.info("[结束] %s  P1=%d P2=%d P3=%d", post.id, p1, p2, p3)

        # ── 帖子讨论结束 → 触发交易撮合 ──
        latest_sg = self.sentiment_history[-1] if self.sentiment_history else None
        if latest_sg is not None:
            all_agents = self.active_participants + self.passive_participants
            event = f"{self.TOPIC_CN[post.topic]}讨论结束"
            session = TradingSession(self.exchange, latest_sg.grid)
            bar = session.run(all_agents, event=event)
            await self._emit({
                "type": "trade",
                "post_id": post.id,
                "event": event,
                "last_price": self.exchange.last_price,
                "bar": {
                    "tick": bar.tick,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "trade_count": bar.trade_count,
                } if bar else None,
            })

    # ─────────────────────────────────────────────
    #  Phase 1 — 直面楼主
    # ─────────────────────────────────────────────

    async def _phase_1(self, post: Post) -> ThreadSnapshot:
        logger.info("── Phase 1: 直面楼主 ──")
        await self._emit({
            "type": "phase", "event": "phase_start",
            "post_id": post.id, "phase": 1, "label": "直面楼主",
        })

        agents = list(self.active_participants)
        random.shuffle(agents)

        def ctx(agent: BaseUserAgent) -> str:
            visible = self.get_agent_context(
                agent, post.topic, query=post.content[:200]
            )
            return (
                f"【主贴 · {self.TOPIC_CN[post.topic]}】\n{post.content}\n\n"
                f"【你可见的相关信息】\n{visible}"
            )

        active_results = await self._run_agents_async(agents, ctx, candidates=None)

        for agent, result in active_results:
            if not agent.should_comment(result.temp):
                continue
            comment = Comment(
                id=Comment.make_id(),
                author_id=agent.unique_id,
                author_type=agent.__class__.__name__,
                author_name=f"{agent.__class__.__name__}#{agent.unique_id}",
                content=result.comment,
                temp=result.temp,
                phase=1,
                sentiment=result.sentiment,
                created_at=datetime.now(),
            )
            post.comments[1].append(comment)
            await self._emit({
                "type": "comment",
                "post_id": post.id,
                "comment": {
                    "id": comment.id,
                    "agentId": comment.author_id,
                    "agentType": comment.author_type,
                    "content": comment.content,
                    "temp": comment.temp,
                    "sentiment": comment.sentiment,
                    "phase": 1,
                    "parentId": None,
                    "timestamp": comment.created_at.isoformat() if comment.created_at else None,
                },
            })

        sg = self._build_sentiment_grid(
            self.round_num, post.topic, 1, active_results
        )
        self.sentiment_history.append(sg)
        s = sg.summary()
        logger.info(
            "  → 评论: %d 条 | 情绪: bull=%d bear=%d neutral=%d",
            len(post.comments[1]), s["bull"], s["bear"], s["neutral"],
        )
        await self._emit({
            "type": "sentiment",
            "post_id": post.id, "phase": 1,
            "summary": s,
            "total_comments": len(post.comments[1]),
        })

        return self._build_snapshot(post, phase=1)

    # ─────────────────────────────────────────────
    #  Phase 2 / 3 — 共用回复逻辑
    # ─────────────────────────────────────────────

    async def _reply_phase(
        self,
        post: Post,
        phase: int,
        prev_snapshot: ThreadSnapshot,
        prev_comments: List[Comment],
    ) -> ThreadSnapshot:
        """Phase 2（互攻/抱团）和 Phase 3（余波）的公共实现。"""
        label = "互攻 / 抱团" if phase == 2 else "余波"
        logger.info("── Phase %d: %s ──", phase, label)
        await self._emit({
            "type": "phase", "event": "phase_start",
            "post_id": post.id, "phase": phase, "label": label,
        })

        if not prev_comments:
            logger.info("  → 跳过（Phase %d 无评论）", phase - 1)
            return self._build_snapshot(post, phase=phase)

        agents = list(self.active_participants)
        random.shuffle(agents)
        ctx_prefix = prev_snapshot.to_text()

        def ctx(agent: BaseUserAgent) -> str:
            rag = self.get_agent_context(
                agent, post.topic, query=ctx_prefix[:300]
            )
            return f"{ctx_prefix}\n\n【你可见的相关知识库信息】\n{rag}"

        active_results = await self._run_agents_async(
            agents, ctx, candidates=prev_comments
        )

        for agent, result in active_results:
            if not agent.should_comment(result.temp):
                continue
            target = next(
                (c for c in prev_comments
                 if c.id == result.reply_to_id and c.author_id != agent.unique_id),
                None,
            )
            if target is None:
                continue
            reply = Comment(
                id=Comment.make_id(),
                author_id=agent.unique_id,
                author_type=agent.__class__.__name__,
                author_name=f"{agent.__class__.__name__}#{agent.unique_id}",
                content=result.comment,
                temp=result.temp,
                phase=phase,
                sentiment=result.sentiment,
                parent_id=target.id,
                created_at=datetime.now(),
            )
            target.children.append(reply)
            post.comments[phase].append(reply)
            await self._emit({
                "type": "comment",
                "post_id": post.id,
                "comment": {
                    "id": reply.id,
                    "agentId": reply.author_id,
                    "agentType": reply.author_type,
                    "content": reply.content,
                    "temp": reply.temp,
                    "sentiment": reply.sentiment,
                    "phase": phase,
                    "parentId": reply.parent_id,
                    "timestamp": reply.created_at.isoformat() if reply.created_at else None,
                },
            })

        sg = self._build_sentiment_grid(
            self.round_num, post.topic, phase, active_results
        )
        self.sentiment_history.append(sg)
        s = sg.summary()
        logger.info(
            "  → 评论: %d 条 | 情绪: bull=%d bear=%d neutral=%d",
            len(post.comments[phase]), s["bull"], s["bear"], s["neutral"],
        )
        await self._emit({
            "type": "sentiment",
            "post_id": post.id, "phase": phase,
            "summary": s,
            "total_comments": len(post.comments[phase]),
        })

        return self._build_snapshot(post, phase=phase)

    async def _phase_2(
        self, post: Post, prev_snapshot: ThreadSnapshot
    ) -> ThreadSnapshot:
        return await self._reply_phase(post, 2, prev_snapshot, post.comments[1])

    async def _phase_3(
        self, post: Post, prev_snapshot: ThreadSnapshot
    ) -> ThreadSnapshot:
        return await self._reply_phase(post, 3, prev_snapshot, post.comments[2])

    # ═══════════════════════════════════════════════════
    #  快照构建
    # ═══════════════════════════════════════════════════

    def _build_snapshot(self, post: Post, phase: int) -> ThreadSnapshot:
        return ThreadSnapshot(
            post_id=post.id,
            phase=phase,
            post_content=post.content,
            phase1_comments=list(post.comments[1]),
        )
