"""
spiders —— 外部政策数据接入

    policy_spider  抓取（RSS / Atom 公开源）
    news_cleaner   清洗（去标签、抽正文、去重、语言检测）
    scheduler      调度入口（1 周 TTL 缓存，命中则不爬）

产出的 CleanedPolicy 最终转成 Document chunks 并入 RAG 的 policy 知识库，
ForumModel 与 Agent 都不感知数据来源。
"""

from .models import CleanedPolicy, RawPolicy, content_hash

__all__ = ["RawPolicy", "CleanedPolicy", "content_hash"]
