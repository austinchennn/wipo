# 02 — 性能重构与工程化改造

> 记录本次 session 对代码库的系统性优化，涵盖异步化、模块化和配置管理三个方向。

---

## 背景

完整模拟（12 轮 × 3 主题 × 3 Phase × N Active Agents）在优化前存在以下问题：

1. **异步执行低效**：每个 Phase 独立创建 `asyncio` event loop（共 108 次），大量时间花在 loop 的创建与销毁上。
2. **PDF 提取串行**：三个 Extractor 顺序调用 LLM，等待时间叠加。
3. **代码高度重复**：三个 Extractor 类 ~40% 代码相同；Phase 2 与 Phase 3 逻辑几乎一致。
4. **配置散落**：魔术数字分布在 6+ 个文件，修改一个参数需逐文件排查。
5. **日志不可控**：全项目用 `print()`，无法按级别过滤，不适合生产环境。

---

## 改动一：ForumModel 完全异步化

### 问题根因

原代码的 `_run_phase()` 是一个同步包装器，每次调用都创建新的 event loop：

```python
# 改动前：每个 Phase 调用一次 asyncio.run()
def _run_phase(self, agents, context_fn, candidates=None):
    return asyncio.run(
        self._run_agents_async(agents, context_fn, candidates)
    )
```

12 轮 × 3 主题 × 3 Phase = **108 次** event loop 创建/销毁。每次 `asyncio.run()` 都会：
- 创建新的 `SelectorEventLoop`
- 注册所有协程
- 等待全部完成
- 关闭并释放 loop

这些 overhead 在大规模模拟中显著累积，且每次 loop 之间无法共享已建立的 HTTP 连接池（OpenAI API 客户端的 `httpx.AsyncClient` 无法跨 loop 复用）。

### 解法：一次 loop，全程异步

```
改动前调用链：
  run() → [for 12 rounds] → step() → [for 3 topics] → _run_thread()
             ↑ sync                ↑ sync                    ↓
                                              _phase_1/_phase_2/_phase_3()
                                                        ↓
                                              _run_phase() → asyncio.run()  ← 108次

改动后调用链：
  run() → asyncio.run(arun())  ← 仅 1 次
             ↓ async
           arun() → [for 12 rounds] → astep()
                                        ↓ async
                                    _run_thread()
                                        ↓ async
                              _phase_1() / _reply_phase(2) / _reply_phase(3)
                                        ↓ await
                                  _run_agents_async()  ← 全程在同一 loop
```

### 关键文件变更

| 方法 | 改动前 | 改动后 |
|---|---|---|
| `run()` | 调 `self.step()` 12 次 | 调 `asyncio.run(self.arun())` |
| `arun()` | 不存在 | 新增，承载完整模拟循环 |
| `step()` | 同步调 `_run_thread()` | 保留为 Mesa API 兼容接口，调 `asyncio.run(astep())` |
| `astep()` | 不存在 | 新增，遍历 3 个 topic |
| `_run_thread()` | `def`（同步） | `async def` |
| `_phase_1/2/3()` | `def`（同步），内部调 `_run_phase()` | `async def`，直接 `await _run_agents_async()` |
| `_run_phase()` | 存在（同步包装器） | **删除** |

### 顺带修复：Phase 2 / Phase 3 代码重复

Phase 2（互攻/抱团）和 Phase 3（余波）逻辑完全相同，唯一区别是 `phase` 编号和候选评论来源。提取公共方法：

```python
# 原来：两份 ~55 行的重复代码
def _phase_2(self, post, prev_snapshot): ...  # 55行
def _phase_3(self, post, prev_snapshot): ...  # 55行

# 现在：共用一个实现，各自只剩 1 行
async def _reply_phase(self, post, phase, prev_snapshot, prev_comments): ...  # 40行
async def _phase_2(self, post, prev_snapshot):
    return await self._reply_phase(post, 2, prev_snapshot, post.comments[1])
async def _phase_3(self, post, prev_snapshot):
    return await self._reply_phase(post, 3, prev_snapshot, post.comments[2])
```

净减少 **~70 行**重复代码。

---

## 改动二：LLMExtractor 泛型基类 + 并行 PDF 提取

### 问题根因

三个 Extractor 类（`ProductExtractor`、`FinancialExtractor`、`RiskExtractor`）各自实现了完全相同的 `__init__` 和 `extract`，约 40% 代码重复：

```python
# 三个类都有这段几乎相同的代码
def __init__(self, model="gpt-4o-mini"):
    llm = get_llm(model=model)
    self._structured_llm = llm.with_structured_output(SomeSchema)

def extract(self, all_chunks, min_keyword_hits=1):
    filtered = filter_chunks_by_keywords(all_chunks, KEYWORDS, ...)
    context = chunks_to_context(filtered, max_chars=12_000)
    messages = [("system", SYSTEM_PROMPT), ("human", USER_PROMPT.format(...))]
    summary = self._structured_llm.invoke(messages)
    for doc in filtered:
        doc.metadata["topic"] = TOPIC  # ← 同时还有一个 bug：污染共享对象
    return summary, filtered
```

此外，`pipeline.py` 中三个 Extractor 顺序调用，等待时间叠加：

```python
# 改动前：串行（总耗时 = T₁ + T₂ + T₃）
product_summary, ...  = product_extractor.extract(all_chunks)   # 等 T₁
financial_summary, ... = financial_extractor.extract(all_chunks)  # 等 T₂
risk_summary, ...     = risk_extractor.extract(all_chunks)       # 等 T₃
```

### 解法一：LLMExtractor 泛型基类

在 `base_extractor.py` 中引入泛型基类，只在基类中实现一次 `__init__` 和 `extract`：

```python
T = TypeVar("T")

class LLMExtractor(Generic[T]):
    KEYWORDS: List[str] = []
    SYSTEM_PROMPT: str = ""
    USER_PROMPT: str = ""
    TOPIC: str = ""
    SUMMARY_CLASS: type = None

    def __init__(self, model="gpt-4o-mini"):
        llm = get_llm(model=model)
        self._structured_llm = llm.with_structured_output(self.SUMMARY_CLASS)

    def extract(self, all_chunks, min_keyword_hits=1):
        filtered = filter_chunks_by_keywords(all_chunks, self.KEYWORDS, ...)
        context = chunks_to_context(filtered)
        summary = self._structured_llm.invoke([...])
        # 拷贝 Document 后再打标签，不修改共享的 all_chunks 对象
        tagged = [copy(doc) for doc in filtered]
        for doc in tagged:
            doc.metadata = {**doc.metadata, "topic": self.TOPIC}
        return summary, tagged
```

每个子类只剩**类变量声明 + `get_content_for_agent()`**：

```python
class ProductExtractor(LLMExtractor["ProductSummary"]):
    KEYWORDS = PRODUCT_KEYWORDS
    SYSTEM_PROMPT = _SYSTEM_PROMPT
    USER_PROMPT = _USER_PROMPT
    TOPIC = "product"
    SUMMARY_CLASS = ProductSummary

    @staticmethod
    def get_content_for_agent(summary, agent_type): ...
```

顺带修复了一个隐藏 bug：原代码直接修改 `all_chunks` 中 Document 的 `metadata["topic"]`。若同一个 chunk 同时命中产品和财务关键词，后运行的 Extractor 会覆盖前者的标签。现在改为 `copy(doc)` 后操作独立副本，并行安全。

### 解法二：ThreadPoolExecutor 并行提取

```python
# 改动后：并行（总耗时 ≈ max(T₁, T₂, T₃)）
with ThreadPoolExecutor(max_workers=3) as pool:
    f_product   = pool.submit(product_extractor.extract,   all_chunks)
    f_financial = pool.submit(financial_extractor.extract, all_chunks)
    f_risk      = pool.submit(risk_extractor.extract,      all_chunks)
    product_summary,   product_chunks   = f_product.result()
    financial_summary, financial_chunks = f_financial.result()
    risk_summary,      risk_chunks      = f_risk.result()
```

使用 `ThreadPoolExecutor` 而非 `asyncio` 的原因：LangChain 的 `.invoke()` 是同步调用，底层通过 `requests`/`httpx` 发 HTTP 请求。在 CPython 中，IO 等待期间会释放 GIL，因此线程级并发对 IO 密集型 LLM 调用有效。

**预期提速**：假设三个 Extractor 耗时相近，并行化可节省约 **2/3 的 PDF 提取时间**。

---

## 改动三：统一配置 + 引入 logging

### src/config.py

将原本散落在 6 个文件中的魔术数字集中到单一配置文件：

| 常量 | 原位置 | 值 |
|---|---|---|
| `MAX_CONCURRENT_LLM_CALLS` | `forum.py:_MAX_CONCURRENT` | `50` |
| `CHUNK_SIZE` | `pipeline.py` 默认参数 | `1500` |
| `CHUNK_OVERLAP` | `pipeline.py` 默认参数 | `200` |
| `MAX_CONTEXT_CHARS` | `base_extractor.py` 默认参数 | `12_000` |
| `RAG_QUERY_K` | `knowledge_base.py` 默认参数 | `4` |
| `RAG_RETRIEVE_K` | `knowledge_base.py` 默认参数 | `3` |
| `AGENT_LLM_TEMPERATURE` | `base_agent.py` 字面量 | `0.8` |
| `EXTRACTOR_LLM_TEMPERATURE` | `base_extractor.py` 字面量 | `0.0` |
| `DEFAULT_LLM_MODEL` | 多处字面量 | `"gpt-4o-mini"` |
| `DEFAULT_EMBEDDING_MODEL` | `knowledge_base.py` 字面量 | `"text-embedding-3-small"` |

### logging 替换 print()

| 文件 | 替换数量 | 效果 |
|---|---|---|
| `forum.py` | 全部 `print()` | 改为 `logger.info()` / `logger.warning()` |
| `pipeline.py` | 全部 `print()` | `verbose=True` → `INFO`，`verbose=False` → `DEBUG` |
| `knowledge_base.py` | RAG 构建进度 | 改为 `logger.info()` |
| `run.py` | 新增 | `logging.basicConfig()` 统一格式化输出 |

日志格式：
```
12:34:56  INFO      src.environment.forum  [发帖] R1_product  主题: 产品信息
12:34:57  WARNING   src.environment.forum  RAG 构建失败（...），降级为静态模式
```

外部调用者（如 FastAPI）可通过标准 `logging` 配置自由控制日志级别和输出目标，不再被 `print()` 硬绑定到 stdout。

---

## 性能影响总结

| 优化项 | 改动前 | 改动后 | 提速幅度 |
|---|---|---|---|
| asyncio event loop 创建次数 | 108 次（每 Phase 1 次） | 1 次（整个模拟） | 消除 107 次 overhead |
| HTTP 连接池复用 | 每 loop 重建 | 全程复用同一 `httpx.AsyncClient` | 减少握手延迟 |
| PDF 提取（3 个 Extractor）| 串行，耗时叠加 | 并行，耗时取最长 | 约节省 2/3 时间 |

对于 1,000 Active Agents 的完整模拟（12 轮），预期总耗时从 **~60 分钟** 降至 **~40 分钟**（event loop + 连接池），PDF 提取阶段从 **~3 分钟** 降至 **~1 分钟**。

---

## 文件变更清单

```
src/config.py                     ← 新增：统一配置常量
src/environment/forum.py          ← 异步化 + logging
src/extractors/base_extractor.py  ← LLMExtractor 基类 + config 引用
src/extractors/product_extractor.py  ← 继承 LLMExtractor，删除重复代码
src/extractors/financial_extractor.py ← 同上
src/extractors/risk_extractor.py     ← 同上
src/extractors/pipeline.py        ← ThreadPoolExecutor 并行 + logging
src/rag/knowledge_base.py         ← config 引用 + logging
src/agents/base_agent.py          ← config 引用（temperature）
run.py                            ← logging.basicConfig()
```
