"""
policy_engine —— 政策文本的语义加工

    classifier      给 CleanedPolicy 打多标签（LLM，无 Key 时规则兜底）
    global_factors  已废弃：原打算把政策转成数值因子改写 Agent 属性，
                    现在改为让 LLM 通过 RAG 检索自然感知，保留空 stub。
"""

from .classifier import LABELS, aclassify, aclassify_all, classify_by_rules

__all__ = ["LABELS", "aclassify", "aclassify_all", "classify_by_rules"]
