"""
decision.typesafe —— 用 TypeSafe AI 的 Jev 实现 DecisionModel

Jev 是 System One 分类模型：不生成文本，给一个状态和几个问题，直接返回
带概率的类型化答案。参考 https://www.langchain.com/blog/building-a-harness-with-jev

这里只用其中的 Noul（是非题，返回命题为真的概率）：

    classifier.invoke({"state": ..., "questions": {"buy": Noul(instructions=...)}})
    response.nouls["buy"].noul   # 0.0 ~ 1.0

整个项目只有本文件 import langchain_typesafe（见 tests/test_architecture.py）。
SDK 自己从环境变量 TYPESAFE_API_KEY 取密钥；Settings 负责判断有没有配，
这里不碰 os.environ，也不猜测构造函数有哪些参数。
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, List, Mapping, Optional, Sequence

from ..ports.decision import AssessRequest, Assessment

logger = logging.getLogger(__name__)


class TypeSafeDecisionModel:
    """Jev 适配器。

    classifier / noul 可注入，测试不必装 SDK、也不必联网；
    生产环境两个都留空，从 langchain_typesafe 懒加载。
    """

    def __init__(
        self,
        classifier: Any = None,
        noul: Optional[Callable[..., Any]] = None,
        max_workers: int = 8,
    ):
        self._max_workers = max_workers
        self._classifier = classifier
        self._noul = noul

        if self._classifier is None or self._noul is None:
            try:
                from langchain_typesafe import Noul, TypeSafeClassifier

                self._noul = noul or Noul
                self._classifier = classifier or TypeSafeClassifier()
            except Exception as e:  # ImportError，或缺 TYPESAFE_API_KEY 时 SDK 抛的错
                logger.warning("Jev 不可用（%s），决策点退回规则", e)
                self._classifier = None

    @property
    def is_available(self) -> bool:
        return self._classifier is not None and self._noul is not None

    def assess(
        self, state: str, questions: Mapping[str, str]
    ) -> Optional[Assessment]:
        if not self.is_available:
            return None
        try:
            response = self._classifier.invoke({
                "state": state,
                "questions": {
                    name: self._noul(instructions=criteria)
                    for name, criteria in questions.items()
                },
            })
            return {name: float(response.nouls[name].noul) for name in questions}
        except Exception as e:
            # 网络 / 限流 / 响应缺字段都算失败，交给调用方兜底
            logger.debug("[Jev] 调用失败: %s", e)
            return None

    def assess_many(
        self, requests: Sequence[AssessRequest]
    ) -> List[Optional[Assessment]]:
        """线程池并发；pool.map 保序，所以结果和 requests 一一对应。"""
        if not requests:
            return []
        if not self.is_available:
            return [None] * len(requests)
        workers = min(self._max_workers, len(requests))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(lambda r: self.assess(*r), requests))

    def __repr__(self) -> str:
        return f"TypeSafeDecisionModel(available={self.is_available})"
