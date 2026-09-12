"""
外部政策接入链路测试

全程离线：爬虫用假 spider 注入，分类器强制走规则兜底，
不发任何 HTTP 请求、不调用任何 LLM。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pytest

from src.graph.ingest_graph import (
    CHECK_CACHE,
    CLASSIFY,
    CLEAN,
    CRAWL,
    LOAD_CACHE,
    PERSIST,
    TO_CHUNKS,
    build_ingest_graph,
    make_initial_state,
    policies_to_documents,
    route_on_cache,
)
from src.persistence import PolicyStore
from src.policy_engine import classifier
from src.spiders.models import CleanedPolicy, RawPolicy
from src.spiders.news_cleaner import (
    clean,
    clean_all,
    detect_lang,
    min_chars_for,
    strip_html,
)
from src.spiders.policy_spider import parse_feed

# ═══════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════

RSS_XML = """<?xml version="1.0"?><rss version="2.0"><channel>
<item>
  <title>SEC Adopts Final Rule</title>
  <link>https://sec.gov/a</link>
  <description>&lt;p&gt;The Commission today adopted amendments to enhance prospectus
  disclosure requirements for initial public offerings.&lt;/p&gt;&lt;script&gt;evil()&lt;/script&gt;
  Read more: click here</description>
  <pubDate>Tue, 09 Sep 2026 10:00:00 GMT</pubDate>
</item>
<item><title>Too short</title><link>https://sec.gov/b</link>
  <description>nope</description></item>
</channel></rss>"""

ATOM_XML = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry>
  <title>证监会发布信息披露新规</title>
  <link href="https://csrc.gov.cn/x"/>
  <summary>为进一步规范首次公开发行股票的信息披露行为，保护投资者合法权益，
  证监会今日发布相关规定，自发布之日起施行。</summary>
  <updated>2026-09-10T00:00:00Z</updated>
</entry></feed>"""


class FakeSpider:
    """替身爬虫：返回固定条目，记录被调用次数。"""

    def __init__(self):
        self.calls = 0

    async def afetch_all(self):
        self.calls += 1
        return parse_feed(RSS_XML, "sec") + parse_feed(ATOM_XML, "csrc")


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    """分类器强制走关键词规则，测试不触网。"""
    monkeypatch.setattr(classifier, "_get_classifier_llm", lambda: None)


@pytest.fixture
def store(tmp_path) -> PolicyStore:
    s = PolicyStore(tmp_path / "policies.db")
    yield s
    s.close()


# ═══════════════════════════════════════════════════════
#  Feed 解析
# ═══════════════════════════════════════════════════════


def test_parse_rss():
    items = parse_feed(RSS_XML, "sec")
    assert [i.title for i in items] == ["SEC Adopts Final Rule", "Too short"]
    assert items[0].url == "https://sec.gov/a"
    assert items[0].published_at.startswith("Tue, 09 Sep 2026")


def test_parse_atom():
    items = parse_feed(ATOM_XML, "csrc")
    assert len(items) == 1
    assert items[0].url == "https://csrc.gov.cn/x"
    assert "信息披露" in items[0].title


def test_parse_malformed_returns_empty():
    assert parse_feed("this is not xml", "x") == []


# ═══════════════════════════════════════════════════════
#  清洗
# ═══════════════════════════════════════════════════════


def test_strip_html_drops_script_and_tail_noise():
    text = strip_html("<p>Real body.</p><script>evil()</script>Read more: junk")
    assert "Real body." in text
    assert "evil" not in text
    assert "junk" not in text


def test_detect_lang():
    assert detect_lang("The Commission adopted amendments") == "en"
    assert detect_lang("证监会今日发布相关规定") == "zh"


def test_chinese_threshold_is_lower_than_english():
    """中文信息密度更高，同样字数信息量更大，阈值必须更低。"""
    assert min_chars_for("zh") < min_chars_for("en")


def test_clean_drops_short_body():
    raw = RawPolicy(source="s", url="u", title="t", raw_body="nope")
    assert clean(raw) is None


def test_clean_all_dedupes():
    raws = parse_feed(RSS_XML, "sec")
    once = clean_all(raws)
    twice = clean_all(raws + raws)
    assert len(once) == len(twice) == 1      # 短的那条被丢，重复的被去重


def test_clean_keeps_chinese_entry():
    cleaned = clean_all(parse_feed(ATOM_XML, "csrc"))
    assert len(cleaned) == 1
    assert cleaned[0].lang == "zh"


# ═══════════════════════════════════════════════════════
#  分类
# ═══════════════════════════════════════════════════════


@pytest.mark.parametrize("text,expected", [
    ("SEC adopted prospectus disclosure rules for an initial public offering",
     {"ipo", "disclosure"}),
    ("The SEC charged the founder with fraud and sought penalties", {"enforcement"}),
    ("The FOMC held the federal funds rate steady amid inflation", {"monetary"}),
    ("证监会规范首次公开发行的信息披露", {"ipo", "disclosure"}),
    ("A local museum opened a new wing", set()),
])
def test_classify_by_rules(text, expected):
    assert set(classifier.classify_by_rules(text)) == expected


def test_classify_filters_unknown_labels(monkeypatch):
    """LLM 返回集合外的标签要被过滤掉，不能污染标签体系。"""

    class FakeLLM:
        async def ainvoke(self, _prompt):
            return classifier.PolicyLabels(labels=["ipo", "made-up-label"])

    monkeypatch.setattr(classifier, "_get_classifier_llm", lambda: FakeLLM())
    p = CleanedPolicy(hash="h", source="s", url="u", title="t", text="body")
    assert asyncio.run(classifier.aclassify(p)) == ["ipo"]


# ═══════════════════════════════════════════════════════
#  缓存
# ═══════════════════════════════════════════════════════


def test_store_ttl_filters_stale(store):
    fresh = CleanedPolicy(hash="a", source="s", url="u", title="t", text="body")
    stale = CleanedPolicy(hash="b", source="s", url="u", title="t", text="body",
                          fetched_at=datetime.now() - timedelta(days=30))
    store.save([fresh, stale])

    assert store.count() == 2
    assert [p.hash for p in store.load_fresh()] == ["a"]
    assert store.is_fresh() is True
    assert store.purge_expired() == 1


def test_store_upsert_refreshes_labels(store):
    p = CleanedPolicy(hash="a", source="s", url="u", title="t", text="body")
    store.save([p])
    p.labels = ["ipo"]
    store.save([p])
    assert store.count() == 1
    assert store.load_fresh()[0].labels == ["ipo"]


def test_empty_store_is_not_fresh(store):
    assert store.is_fresh() is False
    assert store.latest_fetched_at() is None


# ═══════════════════════════════════════════════════════
#  图
# ═══════════════════════════════════════════════════════


def test_route_on_cache():
    assert route_on_cache({"cache_hit": True}) == LOAD_CACHE
    assert route_on_cache({"cache_hit": False}) == CRAWL
    assert route_on_cache({}) == CRAWL


def test_graph_topology(store):
    g = build_ingest_graph(store, FakeSpider()).get_graph()
    edges = {(e.source, e.target) for e in g.edges}
    assert (CHECK_CACHE, CRAWL) in edges
    assert (CHECK_CACHE, LOAD_CACHE) in edges
    assert (CRAWL, CLEAN) in edges
    assert (CLEAN, CLASSIFY) in edges
    assert (CLASSIFY, PERSIST) in edges
    assert (PERSIST, TO_CHUNKS) in edges
    assert (LOAD_CACHE, TO_CHUNKS) in edges


def test_cold_start_crawls_and_persists(store):
    spider = FakeSpider()
    graph = build_ingest_graph(store, spider)
    state = asyncio.run(graph.ainvoke(make_initial_state()))

    assert state["trace"] == [CHECK_CACHE, CRAWL, CLEAN, CLASSIFY, PERSIST, TO_CHUNKS]
    assert spider.calls == 1
    assert store.count() == 2          # 英文 1 条 + 中文 1 条，短的被丢
    assert state["chunks"]


def test_warm_cache_skips_crawl(store):
    spider = FakeSpider()
    graph = build_ingest_graph(store, spider)

    asyncio.run(graph.ainvoke(make_initial_state()))
    state = asyncio.run(graph.ainvoke(make_initial_state()))

    assert state["trace"] == [CHECK_CACHE, LOAD_CACHE, TO_CHUNKS]
    assert spider.calls == 1           # 第二次一个请求都没发
    assert state["chunks"]


def test_expired_cache_recrawls(store):
    spider = FakeSpider()
    graph = build_ingest_graph(store, spider)
    asyncio.run(graph.ainvoke(make_initial_state()))

    # TTL 收紧到 0 天 → 库里的数据立刻算过期
    state = asyncio.run(graph.ainvoke(make_initial_state(ttl_days=0)))
    assert CRAWL in state["trace"]
    assert spider.calls == 2


# ═══════════════════════════════════════════════════════
#  chunk 化 + 访问控制
# ═══════════════════════════════════════════════════════


def test_policies_to_documents_tags_metadata():
    p = CleanedPolicy(hash="h", source="sec", url="https://x", title="T",
                      text="body " * 50, labels=["ipo"])
    docs = policies_to_documents([p])
    assert docs
    meta = docs[0].metadata
    assert meta["topic"] == "policy"
    assert meta["origin"] == "spider"     # 与招股书 chunks 区分
    assert meta["labels"] == "ipo"


def test_empty_policies_produce_no_documents():
    assert policies_to_documents([]) == []


def test_external_policy_respects_visibility_matrix():
    """外部政策必须走可见性矩阵，不能绕过信息壁垒。"""
    from src.extractors.pipeline import make_mock_extraction
    from src.rag import RAGSystem

    rag = RAGSystem.build_static_only(make_mock_extraction())
    p = CleanedPolicy(
        hash="h", source="sec", url="u", title="SEC Rule",
        text="Amendments affect 1,250 issuers and a $25 million threshold.",
    )
    assert rag.add_policy_chunks(policies_to_documents([p])) == 1

    host = rag.get_static_section("policy", "HostAgent")
    retail = rag.get_static_section("policy", "RetailTraderAgent")
    normal = rag.get_static_section("policy", "NormalAgent")

    assert "$25 million" in host                  # FULL 看原文
    assert "$25 million" not in retail            # MASKED 数字被遮
    assert "SEC Rule" in retail
    assert "SEC Rule" not in normal               # HIDDEN 完全看不到
