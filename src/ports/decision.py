"""
ports.decision —— 「快速结构化决策」能力的抽象

和 LLMProvider 的区别：LLM 生成文本（论坛评论、抽取），DecisionModel 只回答
是非题、给出概率。适合交易方向、追价力度这类"输入一段状态，输出一个可以直接
拿去采样 / 比较的数"的决策点——不需要也不该走一次完整的 LLM 生成。

契约刻意收得很窄：

  - 问题是 {名字: 判断标准} 的是非题，返回 {名字: 该命题为真的概率}
  - 不可用 / 调用失败一律返回 None，调用方据此走原有规则兜底，不抛异常
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

# {问题名: 命题为真的概率 ∈ [0, 1]}
Assessment = Dict[str, float]

# (状态描述, {问题名: 判断标准})
AssessRequest = Tuple[str, Mapping[str, str]]


@runtime_checkable
class DecisionModel(Protocol):
    """对一段状态回答一组是非题。"""

    @property
    def is_available(self) -> bool:
        """后端是否可用（无 Key / 没装 SDK 时为 False）。"""
        ...

    def assess(
        self, state: str, questions: Mapping[str, str]
    ) -> Optional[Assessment]:
        """返回每个问题为真的概率；失败返回 None。"""
        ...

    def assess_many(
        self, requests: Sequence[AssessRequest]
    ) -> List[Optional[Assessment]]:
        """批量版：结果与 requests 一一对应，单条失败只影响那一条。"""
        ...
