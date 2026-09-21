"""
decision.null —— 「没有决策模型」的显式实现

和 NullLLMProvider 同一个思路：调用方永远拿到一个 DecisionModel，
只是这一个什么都回答不了，决策点自然退回原有的规则。
"""

from __future__ import annotations

from typing import List, Mapping, Optional, Sequence

from ..ports.decision import AssessRequest, Assessment


class NullDecisionModel:
    """满足 DecisionModel 契约，但永远返回 None。"""

    @property
    def is_available(self) -> bool:
        return False

    def assess(
        self, state: str, questions: Mapping[str, str]
    ) -> Optional[Assessment]:
        return None

    def assess_many(
        self, requests: Sequence[AssessRequest]
    ) -> List[Optional[Assessment]]:
        return [None] * len(requests)

    def __repr__(self) -> str:
        return "NullDecisionModel()"
