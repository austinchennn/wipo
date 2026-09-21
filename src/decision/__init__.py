"""
decision —— DecisionModel 的具体实现

    TypeSafeDecisionModel   Jev（langchain-typesafe），可选依赖
    NullDecisionModel       没有可用后端时的显式实现（决策点走原有规则）

选哪个由 composition root 决定，上层模块只见 ports.DecisionModel。
"""

from .null import NullDecisionModel
from .typesafe import TypeSafeDecisionModel

__all__ = ["NullDecisionModel", "TypeSafeDecisionModel"]
