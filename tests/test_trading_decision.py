"""
交易决策接入 DecisionModel（Jev）的行为测试

用假的 DecisionModel 驱动，不依赖 langchain-typesafe、不联网。

覆盖：
  1. 模型的买/卖概率真的决定方向（并且能覆盖规则）
  2. 模型不可用 / 单次失败 → 原规则兜底
  3. 参与度闸门在调模型之前（被挡掉的 Agent 不产生调用）
  4. 散户追价由模型的 urgent 决定；机构不问 urgent
  5. TypeSafe 适配器：解析响应、失败返回 None、批量保序
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List, Mapping, Optional, Sequence

import pytest

from src.config import IPO_PRICE
from src.decision import NullDecisionModel, TypeSafeDecisionModel
from src.market.exchange import Exchange
from src.market.models import Portfolio, Side
from src.market.trading_agent import (
    TradingSession,
    build_order,
    decide_order,
)
from src.ports.decision import AssessRequest, Assessment


# ═══════════════════════════════════════════════════════
#  测试替身
# ═══════════════════════════════════════════════════════


# 交易逻辑按 agent.__class__.__name__ 分派，所以替身类要同名
class RetailTraderAgent:
    def __init__(self, uid=1, risk=1.0, fomo=0.0, is_active=True):
        self.unique_id = uid
        self.risk_tolerance = risk
        self.fomo_susceptibility = fomo
        self.emotional_volatility = 0.5
        self.contrarian_tendency = 0.5
        self.is_active = is_active


class InstTraderAgent(RetailTraderAgent):
    pass


class NormalAgent(RetailTraderAgent):
    pass


class FakeDecisionModel:
    """按名字返回固定概率；记录每次收到的请求。"""

    def __init__(self, answers: Optional[Assessment] = None, fail: bool = False):
        self.answers = answers or {}
        self.fail = fail
        self.requests: List[AssessRequest] = []

    @property
    def is_available(self) -> bool:
        return True

    def assess(self, state: str, questions: Mapping[str, str]) -> Optional[Assessment]:
        self.requests.append((state, questions))
        if self.fail:
            return None
        return {q: self.answers.get(q, 0.0) for q in questions}

    def assess_many(self, requests: Sequence[AssessRequest]):
        return [self.assess(*r) for r in requests]


def portfolio() -> Portfolio:
    return Portfolio(cash=1_000_000.0, shares=10_000)


# ═══════════════════════════════════════════════════════
#  1. 方向由模型决定
# ═══════════════════════════════════════════════════════


def test_model_buy_probability_drives_side():
    order = build_order(
        RetailTraderAgent(), "neutral", IPO_PRICE, portfolio(),
        {"buy": 1.0, "sell": 0.0, "urgent": 0.0},
    )
    assert order is not None and order.side is Side.BUY


def test_model_sell_probability_drives_side():
    order = build_order(
        RetailTraderAgent(), "neutral", IPO_PRICE, portfolio(),
        {"buy": 0.0, "sell": 1.0, "urgent": 0.0},
    )
    assert order is not None and order.side is Side.SELL


def test_model_hold_overrides_bullish_rules():
    """规则下 bull + risk=1.0 必买；模型说都不想 → 观望，且不回退到规则。"""
    order = build_order(
        RetailTraderAgent(risk=1.0), "bull", IPO_PRICE, portfolio(),
        {"buy": 0.0, "sell": 0.0, "urgent": 0.0},
    )
    assert order is None


# ═══════════════════════════════════════════════════════
#  2. 兜底
# ═══════════════════════════════════════════════════════


@pytest.mark.parametrize("assessment", [None, {}, {"buy": 0.9}])
def test_missing_or_partial_assessment_falls_back_to_rules(assessment):
    """None（调用失败）或缺字段 → 走规则：bull + risk=1.0 概率 1.0 买入。"""
    order = build_order(
        RetailTraderAgent(risk=1.0), "bull", IPO_PRICE, portfolio(), assessment
    )
    assert order is not None and order.side is Side.BUY


def test_session_without_decider_matches_rules_path():
    """decider 为 None / Null：与原来的 decide_order 循环产生同样的订单（同 seed）。"""
    import random

    def run(decider):
        random.seed(7)
        ex = Exchange()
        agents = [RetailTraderAgent(uid=i, risk=0.8) for i in range(10)]
        for a in agents:
            ex.init_portfolio(a.unique_id, "RetailTraderAgent")
        grid = {a.unique_id: "bull" for a in agents}
        captured = []
        real = ex.run_session
        ex.run_session = lambda orders, event=None: (captured.extend(orders), real(orders, event))[1]
        TradingSession(ex, grid, decider).run(agents)
        return [(o.agent_id, o.side, o.price, o.quantity) for o in captured]

    assert run(None) == run(NullDecisionModel())


def test_failed_model_calls_fall_back_per_agent():
    ex = Exchange()
    agent = RetailTraderAgent(uid=1, risk=1.0)
    ex.init_portfolio(1, "RetailTraderAgent")
    model = FakeDecisionModel(fail=True)

    captured = []
    real = ex.run_session
    ex.run_session = lambda orders, event=None: (captured.extend(orders), real(orders, event))[1]
    TradingSession(ex, {1: "bull"}, model).run([agent])

    assert len(model.requests) == 1                 # 确实问过
    assert [o.side for o in captured] == [Side.BUY]  # 失败后走规则：bull + risk 1.0 → 买


# ═══════════════════════════════════════════════════════
#  3. 参与度闸门在模型之前
# ═══════════════════════════════════════════════════════


def test_gated_out_agents_never_reach_the_model(monkeypatch):
    import src.market.trading_agent as ta

    monkeypatch.setattr(ta.random, "random", lambda: 0.99)   # Normal 概率 0.15 → 全挡掉
    ex = Exchange()
    agents = [NormalAgent(uid=i) for i in range(5)]
    for a in agents:
        ex.init_portfolio(a.unique_id, "NormalAgent")
    model = FakeDecisionModel({"buy": 1.0})

    TradingSession(ex, {}, model).run(agents)

    assert model.requests == []


# ═══════════════════════════════════════════════════════
#  4. 追价
# ═══════════════════════════════════════════════════════


def test_retail_urgency_moves_price_across_market():
    """urgent=1 买入追高（价 > 市价）；urgent=0 买入挂低（价 < 市价）。"""
    ask = {"buy": 1.0, "sell": 0.0}
    chase = build_order(RetailTraderAgent(), "neutral", IPO_PRICE, portfolio(), {**ask, "urgent": 1.0})
    patient = build_order(RetailTraderAgent(), "neutral", IPO_PRICE, portfolio(), {**ask, "urgent": 0.0})

    assert chase.price > IPO_PRICE > patient.price


def test_only_retail_is_asked_about_urgency():
    ex = Exchange()
    inst, retail = InstTraderAgent(uid=1), RetailTraderAgent(uid=2)
    ex.init_portfolio(1, "InstTraderAgent")
    ex.init_portfolio(2, "RetailTraderAgent")
    model = FakeDecisionModel({"buy": 1.0})

    TradingSession(ex, {}, model).run([inst, retail])

    asked = {tuple(sorted(q)) for _, q in model.requests}
    assert asked == {("buy", "sell"), ("buy", "sell", "urgent")}


def test_state_text_carries_profile_and_market():
    ex = Exchange()
    ex.init_portfolio(1, "RetailTraderAgent")
    model = FakeDecisionModel({"buy": 1.0})

    TradingSession(ex, {1: "bear"}, model).run([RetailTraderAgent(uid=1, risk=0.25)])

    state = model.requests[0][0]
    assert "散户投资者" in state
    assert "风险承受度 0.25" in state
    assert "看空" in state
    assert f"{IPO_PRICE:.2f}" in state


# ═══════════════════════════════════════════════════════
#  5. TypeSafe 适配器
# ═══════════════════════════════════════════════════════


class FakeClassifier:
    def __init__(self, probs=None, boom=False):
        self.probs = probs or {}
        self.boom = boom
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload)
        if self.boom:
            raise RuntimeError("network down")
        nouls = {k: SimpleNamespace(noul=self.probs[k]) for k in payload["questions"]}
        return SimpleNamespace(nouls=nouls)


def fake_noul(instructions):
    return SimpleNamespace(instructions=instructions)


def test_adapter_parses_noul_probabilities():
    clf = FakeClassifier({"buy": 0.93, "sell": 0.04})
    model = TypeSafeDecisionModel(classifier=clf, noul=fake_noul)

    out = model.assess("state text", {"buy": "b?", "sell": "s?"})

    assert out == {"buy": 0.93, "sell": 0.04}
    payload = clf.calls[0]
    assert payload["state"] == "state text"
    assert payload["questions"]["buy"].instructions == "b?"


def test_adapter_returns_none_on_failure():
    model = TypeSafeDecisionModel(classifier=FakeClassifier(boom=True), noul=fake_noul)
    assert model.assess("s", {"buy": "b?"}) is None


def test_adapter_returns_none_when_response_lacks_a_question():
    class Partial:
        def invoke(self, payload):
            return SimpleNamespace(nouls={})   # 缺 "buy"

    model = TypeSafeDecisionModel(classifier=Partial(), noul=fake_noul)
    assert model.assess("s", {"buy": "b?"}) is None


def test_adapter_assess_many_keeps_order_and_isolates_failures():
    class Flaky:
        def invoke(self, payload):
            state = payload["state"]
            if state == "bad":
                raise RuntimeError("boom")
            return SimpleNamespace(nouls={"q": SimpleNamespace(noul=float(state))})

    model = TypeSafeDecisionModel(classifier=Flaky(), noul=fake_noul)
    out = model.assess_many([("0.1", {"q": ""}), ("bad", {"q": ""}), ("0.9", {"q": ""})])

    assert out == [{"q": 0.1}, None, {"q": 0.9}]


def test_adapter_without_sdk_is_unavailable():
    """没装 langchain-typesafe（或没 Key）→ 不抛异常，is_available=False。"""
    model = TypeSafeDecisionModel()
    if not model.is_available:
        assert model.assess("s", {"q": ""}) is None
        assert model.assess_many([("s", {"q": ""})]) == [None]
