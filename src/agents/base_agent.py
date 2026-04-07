"""智能体定义 —— 基类 (BaseUserAgent) + 四种角色

属性生成规则:
  • gender / age  — 离散均匀分布
  • 其余 13 维    — N(0.5, 0.15) 映射 [0, 1]，硬编码 5% 概率为极端值 (0 或 1)

信息可见性壁垒:
  HostAgent          product=FULL     financial=FULL     policy=FULL
  NormalAgent        product=FULL     financial=HIDDEN   policy=HIDDEN
  InstTraderAgent    product=FULL     financial=FULL     policy=FULL
  RetailTraderAgent  product=MASKED   financial=MASKED   policy=MASKED

comment() 设计：
  - 15 维人格属性 → 自然语言 persona 描述（不传原始数值给 LLM）
  - LLM 返回 {"comment": str, "temp": float, "reply_to_id": str | null}
  - temp 仅表示"发言意愿"（0=不想说，1=强烈想说），和情感极性无关
  - LLM 自己决定在 Phase 2/3 回复哪条评论（only one）
"""

from __future__ import annotations

import asyncio
import os
import random
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from mesa import Agent
from pydantic import BaseModel, Field

from ..models import Comment, Sentiment


# ═══════════════════════════════════════════════════════
#  信息可见性（保留旧接口，ForumModel 仍可调用）
# ═══════════════════════════════════════════════════════

class InfoVisibility(Enum):
    FULL   = "full"
    MASKED = "masked"
    HIDDEN = "hidden"


_VISIBILITY: Dict[str, Dict[str, InfoVisibility]] = {
    "HostAgent": {
        "product":   InfoVisibility.FULL,
        "financial": InfoVisibility.FULL,
        "policy":    InfoVisibility.FULL,
    },
    "NormalAgent": {
        "product":   InfoVisibility.FULL,
        "financial": InfoVisibility.HIDDEN,
        "policy":    InfoVisibility.HIDDEN,
    },
    "InstTraderAgent": {
        "product":   InfoVisibility.FULL,
        "financial": InfoVisibility.FULL,
        "policy":    InfoVisibility.FULL,
    },
    "RetailTraderAgent": {
        "product":   InfoVisibility.MASKED,
        "financial": InfoVisibility.MASKED,
        "policy":    InfoVisibility.MASKED,
    },
}


# ═══════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════

def _gen_normal_attr(
    mu: float = 0.5,
    sigma: float = 0.15,
    outlier_prob: float = 0.05,
) -> float:
    if random.random() < outlier_prob:
        return float(random.choice([0.0, 1.0]))
    return float(np.clip(np.random.normal(mu, sigma), 0.0, 1.0))


def mask_content(text: str) -> str:
    masked = re.sub(r"\d+\.?\d*", "***", text)
    if len(masked) > 100:
        masked = masked[:100] + "……[信息不完整]"
    return f"[MASKED] {masked}"


# ═══════════════════════════════════════════════════════
#  LLM 结构化输出 Schema
# ═══════════════════════════════════════════════════════

class CommentOutput(BaseModel):
    """LLM 返回的评论结构"""
    comment: str = Field(description="你的评论正文内容")
    temp: float = Field(
        ge=0.0, le=1.0,
        description=(
            "你对这个话题的发言意愿强度（0.0=完全不想说，1.0=非常想说）。"
            "请根据你的人设和话题对你的触动程度来决定，不要与人设矛盾。"
        ),
    )
    sentiment: Literal["bull", "bear", "neutral"] = Field(
        description=(
            "你对这个 IPO 标的的当前情绪倾向："
            "bull=看多/乐观，bear=看空/悲观，neutral=中立/不确定。"
            "这与 temp 完全无关——高 temp 只代表你很想说话，不代表你看多。"
        ),
    )
    reply_to_id: Optional[str] = Field(
        None,
        description=(
            "Phase 2/3 专用：你选择回复的那条评论的 ID（从候选列表中选一个）。"
            "Phase 1 直接返回 null。"
        ),
    )


@dataclass
class CommentResult:
    """comment() / acomment() 的统一返回结构（替代裸 tuple）。"""
    comment: str
    temp: float
    sentiment: Sentiment
    reply_to_id: Optional[str] = None


# ═══════════════════════════════════════════════════════
#  模块级 LLM 懒初始化（所有 Agent 共享同一个实例）
# ═══════════════════════════════════════════════════════

_llm_instance = None

def _get_structured_llm():
    """懒加载：首次调用时初始化 LLM，之后复用。"""
    global _llm_instance
    if _llm_instance is None:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        from langchain_openai import ChatOpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None  # 无 API Key → 退回 mock
        llm = ChatOpenAI(
            model=os.environ.get("AGENT_LLM_MODEL", "gpt-4o-mini"),
            temperature=0.8,  # 评论需要多样性
            api_key=api_key,
        )
        _llm_instance = llm.with_structured_output(CommentOutput)
    return _llm_instance


# ═══════════════════════════════════════════════════════
#  Persona 自然语言生成
# ═══════════════════════════════════════════════════════

def _attr_desc(value: float, high: str, low: str, mid: str = "") -> str:
    """把 [0,1] 属性值转成自然语言片段。"""
    if value >= 0.7:
        return high
    if value <= 0.3:
        return low
    return mid


def build_persona_text(agent: "BaseUserAgent") -> str:
    """将 15 维人格属性转化为 LLM 可理解的自然语言人设描述。

    不向 LLM 传递原始数字，避免 LLM 被数字误导。
    """
    gender = "男性" if agent.gender == 1 else "女性"
    parts = [f"你是一个 {gender}，年龄 {agent.age} 岁。"]

    # 风险偏好
    parts.append(_attr_desc(
        agent.risk_tolerance,
        high="你是一个激进的投资者，对高风险高回报的机会极度感兴趣，敢于重仓押注。",
        low="你是一个极度保守的投资者，对任何风险都非常厌恶，倾向于观望或持有现金。",
        mid="你对投资风险持中性态度，会根据具体情况判断。",
    ))

    # 财务认知
    parts.append(_attr_desc(
        agent.financial_literacy,
        high="你拥有深厚的财务与金融专业知识，能够独立解读财报和估值模型。",
        low="你对金融知识了解有限，主要依赖他人观点和直觉做判断。",
        mid="你有一定的金融常识，能理解基本概念但不擅长深度分析。",
    ))

    # 逆向思维
    parts.append(_attr_desc(
        agent.contrarian_tendency,
        high="你有强烈的逆向思维，倾向于在大众看涨时质疑泡沫，在大众恐慌时发现机会。",
        low="你倾向于认同和跟随主流观点，相信共识的力量。",
        mid="",
    ))

    # FOMO / 羊群效应
    parts.append(_attr_desc(
        agent.fomo_susceptibility,
        high="你很容易受到市场情绪感染，害怕错过热点，容易跟风追涨杀跌。",
        low="你不容易被市场情绪左右，对热点保持冷静距离。",
        mid="",
    ))

    # 表达欲 / 活跃度
    parts.append(_attr_desc(
        agent.expressiveness,
        high="你非常喜欢在论坛上发表观点，是个活跃的讨论者。",
        low="你在论坛上相当内敛，只有在被强烈触动时才会发言。",
        mid="",
    ))

    # 情绪波动
    parts.append(_attr_desc(
        agent.emotional_volatility,
        high="你情绪波动明显，讨论时容易表现出激动、愤怒或兴奋的情绪。",
        low="你情绪稳定，即便在激烈讨论中也能保持克制的语气。",
        mid="",
    ))

    # 政治倾向（影响政策话题）
    parts.append(_attr_desc(
        agent.political_leaning,
        high="你在政治上偏右翼，支持市场自由化、减少监管和政府干预。",
        low="你在政治上偏左翼，更关注监管公平和防范资本垄断。",
        mid="",
    ))

    # 机构信任度
    parts.append(_attr_desc(
        agent.trust_in_institutions,
        high="你相当信任监管机构和主流媒体的信息。",
        low="你对官方机构和主流媒体持怀疑态度，更信任草根信息。",
        mid="",
    ))

    # 资本量级（影响言论的"底气"）
    parts.append(_attr_desc(
        agent.capital_level,
        high="你资金量较大，在讨论中会更多考虑大资金的操作逻辑。",
        low="你是小资金玩家，对资金量大的操作可能感到遥远。",
        mid="",
    ))

    # 过滤掉空字符串
    return " ".join(p for p in parts if p.strip())


# ═══════════════════════════════════════════════════════
#  BaseUserAgent（15 维属性）
# ═══════════════════════════════════════════════════════

class BaseUserAgent(Agent):
    """智能体基类 —— 15 维属性 + LLM 驱动的评论生成"""

    def __init__(self, model, is_active: bool = True):
        super().__init__(model)

        # ── 是否为 Active Agent（调 LLM）或 Passive Agent（仅做情绪推算）──
        self.is_active: bool = is_active

        # ── 基础属性 ──
        self.gender: int = random.choice([1, 2])
        self.age: int = random.randint(18, 80)

        # ── 正态分布属性 (0‒1) ──
        self.tech_preference: float      = _gen_normal_attr()
        self.environmental_love: float   = _gen_normal_attr()
        self.political_leaning: float    = _gen_normal_attr()
        self.expressiveness: float       = _gen_normal_attr()
        self.risk_tolerance: float       = _gen_normal_attr()
        self.financial_literacy: float   = _gen_normal_attr()
        self.fomo_susceptibility: float  = _gen_normal_attr()
        self.trust_in_institutions: float = _gen_normal_attr()
        self.attention_span: float       = _gen_normal_attr()
        self.capital_level: float        = _gen_normal_attr()
        self.contrarian_tendency: float  = _gen_normal_attr()
        self.emotional_volatility: float = _gen_normal_attr()
        self.policy_sensitivity: float   = _gen_normal_attr()

        # 缓存 persona 文本（避免每次 comment() 重新生成）
        self._persona_text: Optional[str] = None

        # 最近一次评论的情绪（passive inference 读取）
        self.last_sentiment: Sentiment = "neutral"

    # ─────────── 可见性（旧接口，保留兼容）───────────

    @property
    def _visibility(self) -> Dict[str, InfoVisibility]:
        return _VISIBILITY[self.__class__.__name__]

    def get_visible_content(
        self,
        raw_sections: Dict[str, str],
        topic: str,
    ) -> str:
        vis = self._visibility[topic]
        raw = raw_sections[topic]
        if vis is InfoVisibility.FULL:
            return raw
        if vis is InfoVisibility.MASKED:
            return mask_content(raw)
        return "[此信息对你不可见]"

    # ─────────── 发言阈值 ───────────

    @property
    def comment_threshold(self) -> float:
        """expressiveness 越高 → 阈值越低 → 越爱发言"""
        return 1.0 - self.expressiveness

    def should_comment(self, temp: float) -> bool:
        """temp ≥ 阈值 → 开口发言；否则潜水"""
        return temp >= self.comment_threshold

    # ─────────── Persona 文本（懒缓存）───────────

    @property
    def persona_text(self) -> str:
        if self._persona_text is None:
            self._persona_text = build_persona_text(self)
        return self._persona_text

    # ─────────── 核心：LLM 评论生成 ───────────

    def comment(
        self,
        context: str,
        candidates: Optional[List[Comment]] = None,
    ) -> CommentResult:
        """
        同步版评论生成（供单次调试或小规模测试使用）。
        大规模并发请使用 acomment()。

        参数：
            context    — 帖子内容 + 该 Agent 可见的 RAG 信息
            candidates — Phase 2/3 专用：候选评论列表（LLM 从中选一条回复）

        返回：CommentResult（comment, temp, sentiment, reply_to_id）
        """
        llm = _get_structured_llm()
        if llm is None:
            return self._mock_comment(candidates)

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(context, candidates)

        try:
            out: CommentOutput = llm.invoke([
                ("system", system_prompt),
                ("human", user_prompt),
            ])
            result = CommentResult(
                comment=out.comment,
                temp=out.temp,
                sentiment=out.sentiment,
                reply_to_id=out.reply_to_id,
            )
            self.last_sentiment = result.sentiment
            return result
        except Exception:
            return self._mock_comment(candidates)

    async def acomment(
        self,
        context: str,
        candidates: Optional[List[Comment]] = None,
    ) -> CommentResult:
        """
        异步版评论生成（供大规模并发调用使用）。
        ForumModel._run_phase_async() 会用 asyncio.gather() 并发调用此方法。
        """
        llm = _get_structured_llm()
        if llm is None:
            return self._mock_comment(candidates)

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(context, candidates)

        try:
            out: CommentOutput = await llm.ainvoke([
                ("system", system_prompt),
                ("human", user_prompt),
            ])
            result = CommentResult(
                comment=out.comment,
                temp=out.temp,
                sentiment=out.sentiment,
                reply_to_id=out.reply_to_id,
            )
            self.last_sentiment = result.sentiment
            return result
        except Exception:
            return self._mock_comment(candidates)

    def _build_system_prompt(self) -> str:
        threshold = self.comment_threshold
        role_label = {
            "HostAgent":        "论坛主持人（信息最完整）",
            "NormalAgent":      "普通投资者（财务和政策信息不可见）",
            "InstTraderAgent":  "机构分析师（拥有完整信息）",
            "RetailTraderAgent": "散户投资者（信息经过模糊处理）",
        }.get(self.__class__.__name__, "论坛用户")

        return f"""\
你正在扮演一个在 IPO 论坛讨论中的真实用户。

【你的角色】{role_label}
【你的人设】{self.persona_text}

【发言规则】
- 你的回复必须符合你的人设，不能前后矛盾（例如：如果你是内敛的人，不应该返回 0.9 的发言意愿）。
- temp 字段代表你对这个话题的"发言意愿"（0.0=完全不想说，1.0=非常强烈想说）。
- 注意：你的发言意愿阈值约为 {threshold:.2f}，如果 temp 低于该值，代表你选择沉默。
  请根据这个话题对你的触动程度来决定 temp，不要机械地返回固定值。
- 评论内容要符合你的知识水平和情绪风格，机构投资者用专业术语，普通人用通俗语言。
- 字数控制在 30-150 字，像真实论坛评论一样自然，不要写成分析报告。
- 可以表达看多、看空、或中立，也可以质疑或提问。"""

    def _build_user_prompt(
        self,
        context: str,
        candidates: Optional[List[Comment]],
    ) -> str:
        parts = [
            "【当前讨论内容】",
            context,
        ]

        if candidates:
            others = [c for c in candidates if c.author_id != self.unique_id]
            if others:
                parts.append("\n【你可以选择回复以下评论中的一条（只能选一条）】")
                for c in others[:10]:  # 最多展示 10 条候选，避免 token 过多
                    parts.append(
                        f"  ID={c.id} | [{c.author_name}]: {c.content[:100]}"
                    )
                parts.append(
                    "\n请在 reply_to_id 字段填写你选择回复的评论 ID，"
                    "如果你不想回复任何人则填 null。"
                )
            else:
                parts.append("\n（该 Phase 中暂无其他人的评论可回复）")

        parts.append("\n请以 JSON 格式返回你的 comment、temp 和 reply_to_id。")
        return "\n".join(parts)

    def _mock_comment(
        self, candidates: Optional[List[Comment]] = None
    ) -> CommentResult:
        """无 API Key 时的 Mock 实现（返回 CommentResult）。"""
        mock_text = (
            f"[{self.__class__.__name__}#{self.unique_id}] "
            f"我对此的看法是…… "
            f"（mock · age={self.age} · "
            f"risk={self.risk_tolerance:.2f} · "
            f"fin_lit={self.financial_literacy:.2f}）"
        )
        mock_temp = random.random()
        mock_sentiment: Sentiment = random.choice(["bull", "bear", "neutral"])

        reply_id = None
        if candidates:
            others = [c for c in candidates if c.author_id != self.unique_id]
            if others:
                reply_id = random.choice(others).id

        self.last_sentiment = mock_sentiment
        return CommentResult(
            comment=mock_text,
            temp=mock_temp,
            sentiment=mock_sentiment,
            reply_to_id=reply_id,
        )

    # ─────────── Mesa 标准接口 ───────────

    def step(self):
        """由 ForumModel 显式调度各 Phase，此处留空"""


# ═══════════════════════════════════════════════════════
#  角色子类
# ═══════════════════════════════════════════════════════

class HostAgent(BaseUserAgent):
    """主持人 —— 拥有全部信息，负责发帖。始终是 Active Agent。"""

    def __init__(self, model):
        super().__init__(model, is_active=True)

    def create_post_content(self, raw_sections: Dict[str, str], topic: str) -> str:
        return raw_sections[topic]


class NormalAgent(BaseUserAgent):
    """普通人 —— 仅可见无掩码的产品信息"""

    def __init__(self, model, is_active: bool = True):
        super().__init__(model, is_active=is_active)


class InstTraderAgent(BaseUserAgent):
    """机构交易者 —— 全信息无掩码"""

    def __init__(self, model, is_active: bool = True):
        super().__init__(model, is_active=is_active)


class RetailTraderAgent(BaseUserAgent):
    """散户交易者 —— 可见全部三模块但均被掩码处理"""

    def __init__(self, model, is_active: bool = True):
        super().__init__(model, is_active=is_active)
