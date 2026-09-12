# 06 — 依赖反转与 composition root

> 把散落在各模块里的依赖装配收敛到一个装配点，用 Protocol 区分抽象与具体实现。

---

## 背景

重构前这个项目**没有 composition root —— 每个模块都是自己的 composition root**。

症状不是抽象的"代码味道"，是四个可以直接指出来的事实：

### 1. 同一件事在四个文件里各做一遍

`os.environ.get("GOOGLE_API_KEY")` 出现在 `base_agent.py`、`classifier.py`、
`knowledge_base.py`、`base_extractor.py`，每处各自判断"有没有 Key"、各自决定
降级策略。memory 里记的那次"迁移到 Gemini"，要同时改这四个文件。

### 2. 测试只能靠打桩模块全局变量

```python
# 重构前
monkeypatch.setattr(base_agent, "_get_structured_llm", lambda: None)
monkeypatch.setattr(classifier, "_get_classifier_llm", lambda: None)
```

需要 monkeypatch 一个模块级私有函数才能测，就是"这里没有接缝"的确凿证据。

### 3. 换 API Key 只能改全进程状态

```python
# 重构前 api/main.py
if req.api_key:
    os.environ["GOOGLE_API_KEY"] = req.api_key
```

因为 LLM 是 `global _llm_instance` 懒加载单例，一个进程只能有一套配置。
两个并发模拟用不同 Key 会互相覆盖。

### 4. ForumModel 既是领域模型又是装配器

自己跑 Extractor、自己建 FAISS、自己 `new Exchange()`、在 `arun()` 里硬编码
`SimulationDB()`。派生出 12 处函数内 import——之所以不能提到模块顶层，
一半是为了断循环依赖，一半是为了"延迟到真要用时"，两者都是耦合过紧的表现。

---

## 改动一：ports 包 —— 抽象的落点

新增 `src/ports/`，只有 Protocol 定义，**不 import 任何第三方库**
（langchain / httpx / sqlite3 / mesa 一个都没有）。所有依赖箭头指向它。

| Protocol | 上层使用者 | 具体实现 |
|---|---|---|
| `LLMProvider` | BaseUserAgent、classifier、LLMExtractor | `GeminiProvider` / `NullLLMProvider` |
| `KnowledgeProvider` | ForumModel | `RAGSystem` |
| `SimulationSink` | ForumModel | `SimulationDB` |
| `PolicyCache` | ingest_graph | `PolicyStore` |
| `PolicyFeed` | ingest_graph | `PolicySpider` |

### 为什么用 Protocol 而不是 ABC

1. LangChain 的 `Runnable`、`Embeddings` 是第三方类型，没法让它们继承我们的
   ABC，但它们天然满足结构化契约
2. 测试替身不用继承、不用注册——已有的 `FakeSpider` 一行没改就满足了 `PolicyFeed`
3. `ForumModel` 的继承位已经被 Mesa 的 `Model` 占了

代价是结构化子类型不会在定义处报错，所以补了 `isinstance` 断言把契约钉住
（`runtime_checkable` 只检查方法名，不检查签名，这一点要清楚）。

---

## 改动二：Settings —— 配置的唯一入口

```python
@dataclass(frozen=True)
class Settings:
    google_api_key: Optional[str] = None
    chat_model: str = DEFAULT_LLM_MODEL
    ...
    @classmethod
    def from_env(cls) -> "Settings": ...   # 全项目唯一读 os.environ 的地方

    def with_api_key(self, key) -> "Settings": ...  # 返回新实例，不改全局
```

和 `config.py` 的分工：`config.py` 放不随环境变化的领域常量（TTL、chunk 大小、
IPO 价格），`settings.py` 放随部署环境变化的东西（Key、模型名）。两者不重叠。

API 层因此不用再写 `os.environ[...] = key`：

```python
settings = Settings.from_env().with_api_key(req.api_key).with_model(req.llm_model)
```

前端传的 Key 只作用于这一次模拟。

---

## 改动三：NullLLMProvider —— 把隐式状态变成显式实现

"没配 Key"原来表现为四处各自写 `if not api_key: return None`，是一种散落的
隐式状态。现在它是一个实现类：调用方永远拿到一个 `LLMProvider`，只是这一个
什么都给不了。

`GeminiProvider` 把原来两处 `global _llm_instance` 的缓存挪进实例。复用客户端
本身是对的（省掉重复建连），错的是放在模块全局。缓存键带 `temperature`，因为
Agent 用 0.8、Extractor 和 Classifier 用 0.0，是两个不同的模型实例。

---

## 改动四：composition root

新增 `src/composition.py`，整个项目里**只有这里知道抽象对应哪个实现**。

```python
def build_simulation(settings: Settings, opts: RunOptions) -> ForumModel:
    llm = build_llm_provider(settings)          # Gemini 还是 Null
    return ForumModel(
        knowledge=build_knowledge(llm, opts.pdf_path, opts.use_rag),
        llm=llm,
        sink=build_sink(settings, opts.persist),
        ...
    )
```

`RunOptions` 收敛了原本在 `run.py` 和 `api/main.py` 各拼一遍的十来个参数。
三个入口（`run.py` / `api/main.py` / `engine.py`）现在都只调 `build_simulation()`。

`ForumModel._load_from_pdf()` / `_load_mock()` 整个删掉搬进 `build_knowledge()`
——"有 PDF 就跑 Extractor、拿不到 embedding 就降级静态模式"是装配决策，
不是领域逻辑。`pdf_path` / `llm_model` / `use_rag` 三个参数也随之移出 ForumModel。

### 哪些东西**不**注入

`Exchange`、`TradingSession`、各种 Agent 都是纯内存领域对象，没有外部依赖也
没有 I/O，给它们做依赖反转只是加仪式感，不解耦任何东西。这次只反转真正带
I/O 的三类：**LLM、知识库、落库**。

---

## 改动五：修正一处反向依赖

```python
# 重构前 persistence/database.py
from ..environment.forum import ForumModel
if not isinstance(model, ForumModel): ...
```

底层持久化 import 上层领域模型，依赖方向反了。现在 `save()` 只按
`SimulationSink` 契约工作。

---

## 改动六：架构测试

依赖反转做完不写测试，下一个人随手加一行 import 就退回去了。
`tests/test_architecture.py` 用 AST 静态检查源码（不 import 被检查的模块），
17 条跑 0.2 秒：

- ports 不依赖任何第三方库和任何具体实现
- 只有 `settings.py` 允许读 `os.environ`
- 不允许模块级 LLM 单例（`global _llm_*`）
- 只有 `llm/gemini.py` 允许 import `langchain_google_genai`
- `persistence` 不许反向依赖 `environment`
- 每个具体实现都要通过对应 Protocol 的 `isinstance` 检查

**这些测试写完立刻抓到一处真实泄漏**：`forum.py` 还 import 着
`NullLLMProvider`，因为 `llm` 参数当时是 `Optional`、默认降级。修法是把 `llm`
改成必填——没有 LLM 要显式传 `NullLLMProvider()`，而不是省略参数悄悄降级。

---

## 结果

| 指标 | 重构前 | 重构后 |
|---|---|---|
| 读 `os.environ` 的文件 | 5 | 1 |
| 模块级 LLM 单例 | 2 | 0 |
| 测试里 monkeypatch 依赖 | 4 处 | 0 |
| 换 LLM 供应商要改的文件 | 4 | 1 |
| 测试总数 | 34 | 54 |

行为没有变化：`run.py` 两条链路、`--graph`、`--spider` 实测输出一致。

---

## 遗留

- **`ForumModel` 仍然是个大类。** 它现在只剩调度职责（不再装配），但"领域模型"
  和"编排器"仍然混在一起。彻底拆开要动 Mesa 的 `Model` 继承关系和前端事件流，
  这次没做。
- **`runtime_checkable` 只检查方法名，不检查签名。** `isinstance(x, PolicyFeed)`
  不能保证 `afetch_all` 的参数对得上。要真正卡住签名得上 mypy。
- **Agent 通过 `self.model.llm` 取 provider**，不是构造注入。Mesa 的 Agent 本来
  就持有 model，加构造参数要改四个子类，收益不大；代价是这条依赖在类型上是隐式的。
