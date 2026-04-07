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
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from ..models import Comment, Post, Sentiment, SentimentGrid, ThreadSnapshot

# 并发 LLM 调用时的最大同时请求数（避免触发 API rate limit）
_MAX_CONCURRENT = 50


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
        llm_model: str = "gpt-4o-mini",
        use_rag: bool = True,
        max_concurrent: int = _MAX_CONCURRENT,
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

        # ── 存档 ──
        self.posts: List[Post] = []
        self.sentiment_history: List[SentimentGrid] = []
        self.round_num: int = 0

    # ─────────────────── 工具：按比例抽样 ───────────────────

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
                print(self.rag_system.status())
            except EnvironmentError as e:
                print(f"  [警告] RAG 构建失败（{e}），降级为静态模式")
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

    def _run_phase(
        self,
        agents: List[BaseUserAgent],
        context_fn,
        candidates: Optional[List[Comment]] = None,
    ) -> List[Tuple[BaseUserAgent, CommentResult]]:
        """同步包装：在 Mesa 同步上下文中执行异步并发调用。"""
        return asyncio.run(
            self._run_agents_async(agents, context_fn, candidates)
        )

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
        n_active  = len(self.active_participants)
        n_passive = len(self.passive_participants)
        n_total   = n_active + n_passive + 1  # +1 for host

        print(
            f"\n{'#' * 60}\n"
            f"  金融舆情论坛沙盘启动\n"
            f"  总 Agent: {n_total}  "
            f"(Active={n_active}  Passive={n_passive}  Host=1)\n"
            f"  Active 构成: "
            f"普通={sum(1 for a in self.active_participants if isinstance(a, NormalAgent))}  "
            f"机构={sum(1 for a in self.active_participants if isinstance(a, InstTraderAgent))}  "
            f"散户={sum(1 for a in self.active_participants if isinstance(a, RetailTraderAgent))}\n"
            f"{'#' * 60}"
        )

        for r in range(self.MACRO_ROUNDS):
            self.round_num = r + 1
            print(f"\n{'=' * 55}")
            print(f"  宏观轮次 {self.round_num} / {self.MACRO_ROUNDS}")
            print(f"{'=' * 55}")
            self.step()

        total_comments = sum(
            len(p.comments[ph]) for p in self.posts for ph in (1, 2, 3)
        )
        print(
            f"\n{'#' * 60}\n"
            f"  模拟结束 | 帖子={len(self.posts)}  评论={total_comments}\n"
            f"  SentimentGrid 快照数: {len(self.sentiment_history)}\n"
            f"{'#' * 60}"
        )

    def step(self):
        for topic in self.TOPICS:
            post = self._host_publish(topic)
            self._run_thread(post)

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
        )
        self.posts.append(post)
        print(f"\n  📌 [发帖] {post.id}  主题: {self.TOPIC_CN[topic]}")
        return post

    # ═══════════════════════════════════════════════════
    #  帖子生命周期: 3 Phase
    # ═══════════════════════════════════════════════════

    def _run_thread(self, post: Post):
        snap1 = self._phase_1(post)
        snap2 = self._phase_2(post, snap1)
        self._phase_3(post, snap2)

        p1, p2, p3 = (len(post.comments[i]) for i in (1, 2, 3))
        print(f"  🔒 [结束] {post.id}  P1={p1}  P2={p2}  P3={p3}")

    # ─────────────────────────────────────────────
    #  Phase 1 — 直面楼主
    # ─────────────────────────────────────────────

    def _phase_1(self, post: Post) -> ThreadSnapshot:
        print(f"    ── Phase 1: 直面楼主 ──")

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

        active_results = self._run_phase(agents, ctx, candidates=None)

        for agent, result in active_results:
            if not agent.should_comment(result.temp):
                continue
            post.comments[1].append(Comment(
                id=Comment.make_id(),
                author_id=agent.unique_id,
                author_type=agent.__class__.__name__,
                author_name=f"{agent.__class__.__name__}#{agent.unique_id}",
                content=result.comment,
                temp=result.temp,
                phase=1,
            ))

        # SentimentGrid（Active + Passive 推算）
        sg = self._build_sentiment_grid(
            self.round_num, post.topic, 1, active_results
        )
        self.sentiment_history.append(sg)
        s = sg.summary()
        print(
            f"      → 评论: {len(post.comments[1])} 条 | "
            f"情绪: bull={s['bull']} bear={s['bear']} neutral={s['neutral']}"
        )

        return self._build_snapshot(post, phase=1)

    # ─────────────────────────────────────────────
    #  Phase 2 — 互攻 / 抱团
    # ─────────────────────────────────────────────

    def _phase_2(
        self, post: Post, prev_snapshot: ThreadSnapshot
    ) -> ThreadSnapshot:
        print(f"    ── Phase 2: 互攻 / 抱团 ──")

        if not post.comments[1]:
            print("      → 跳过（Phase 1 无评论）")
            return self._build_snapshot(post, phase=2)

        agents = list(self.active_participants)
        random.shuffle(agents)
        ctx_prefix = prev_snapshot.to_text()
        candidates = post.comments[1]

        def ctx(agent: BaseUserAgent) -> str:
            rag = self.get_agent_context(
                agent, post.topic, query=ctx_prefix[:300]
            )
            return f"{ctx_prefix}\n\n【你可见的相关知识库信息】\n{rag}"

        active_results = self._run_phase(agents, ctx, candidates=candidates)

        for agent, result in active_results:
            if not agent.should_comment(result.temp):
                continue
            target = next(
                (c for c in candidates
                 if c.id == result.reply_to_id and c.author_id != agent.unique_id),
                None,
            )
            if target is None:
                continue
            sub = Comment(
                id=Comment.make_id(),
                author_id=agent.unique_id,
                author_type=agent.__class__.__name__,
                author_name=f"{agent.__class__.__name__}#{agent.unique_id}",
                content=result.comment,
                temp=result.temp,
                phase=2,
                parent_id=target.id,
            )
            target.children.append(sub)
            post.comments[2].append(sub)

        sg = self._build_sentiment_grid(
            self.round_num, post.topic, 2, active_results
        )
        self.sentiment_history.append(sg)
        s = sg.summary()
        print(
            f"      → 评论: {len(post.comments[2])} 条 | "
            f"情绪: bull={s['bull']} bear={s['bear']} neutral={s['neutral']}"
        )

        return self._build_snapshot(post, phase=2)

    # ─────────────────────────────────────────────
    #  Phase 3 — 余波
    # ─────────────────────────────────────────────

    def _phase_3(
        self, post: Post, prev_snapshot: ThreadSnapshot
    ) -> ThreadSnapshot:
        print(f"    ── Phase 3: 余波 ──")

        if not post.comments[2]:
            print("      → 跳过（Phase 2 无评论）")
            return self._build_snapshot(post, phase=3)

        agents = list(self.active_participants)
        random.shuffle(agents)
        ctx_prefix = prev_snapshot.to_text()
        candidates = post.comments[2]

        def ctx(agent: BaseUserAgent) -> str:
            rag = self.get_agent_context(
                agent, post.topic, query=ctx_prefix[:300]
            )
            return f"{ctx_prefix}\n\n【你可见的相关知识库信息】\n{rag}"

        active_results = self._run_phase(agents, ctx, candidates=candidates)

        for agent, result in active_results:
            if not agent.should_comment(result.temp):
                continue
            target = next(
                (c for c in candidates
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
                phase=3,
                parent_id=target.id,
            )
            target.children.append(reply)
            post.comments[3].append(reply)

        sg = self._build_sentiment_grid(
            self.round_num, post.topic, 3, active_results
        )
        self.sentiment_history.append(sg)
        s = sg.summary()
        print(
            f"      → 评论: {len(post.comments[3])} 条 | "
            f"情绪: bull={s['bull']} bear={s['bear']} neutral={s['neutral']}"
        )

        return self._build_snapshot(post, phase=3)

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
