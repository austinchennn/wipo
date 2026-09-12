"""
架构约束测试

依赖反转做完不写测试的话，下一个人随手加一行 import 就退回去了。
这里用 AST 静态检查源码，不 import 被检查的模块，跑起来很快。

约束：
  1. ports 包不依赖任何第三方库和任何具体实现
  2. 只有 settings.py 允许读 os.environ
  3. 不允许模块级 LLM 单例
  4. 领域层不允许直接 import 具体 LLM 供应商
  5. 底层 persistence 不允许反向依赖上层 environment
  6. 具体实现确实满足对应的 Protocol
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterator, List, Set

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"


# ═══════════════════════════════════════════════════════
#  AST 工具
# ═══════════════════════════════════════════════════════


def _strip_type_checking(tree: ast.AST) -> ast.AST:
    """删掉 `if TYPE_CHECKING:` 块 —— 那里的 import 运行时不存在。"""

    class Stripper(ast.NodeTransformer):
        def visit_If(self, node: ast.If):
            test = node.test
            if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
                return node.orelse or None
            return self.generic_visit(node)

    return Stripper().visit(tree)


def _parse(path: Path, *, runtime_only: bool = True) -> ast.AST:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return _strip_type_checking(tree) if runtime_only else tree


def imported_modules(path: Path, *, runtime_only: bool = True) -> Set[str]:
    """文件里 import 的模块名（相对 import 还原成 src.x.y 形式）。"""
    tree = _parse(path, runtime_only=runtime_only)
    found: Set[str] = set()

    # 把 src/a/b.py 变成 ["src", "a"]，用于解析相对 import
    rel_parts = path.relative_to(ROOT).with_suffix("").parts
    pkg_parts = list(rel_parts[:-1])

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                found.add(node.module or "")
            else:
                base = pkg_parts[: len(pkg_parts) - node.level + 1]
                found.add(".".join(base + ([node.module] if node.module else [])))
    return found


def python_files(*dirs: Path) -> Iterator[Path]:
    for d in dirs:
        if d.is_file():
            yield d
            continue
        for p in sorted(d.rglob("*.py")):
            if "__pycache__" not in p.parts:
                yield p


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


# ═══════════════════════════════════════════════════════
#  1. ports 是纯抽象
# ═══════════════════════════════════════════════════════

THIRD_PARTY = (
    "langchain", "langchain_core", "langchain_google_genai",
    "langchain_community", "langchain_text_splitters", "langgraph",
    "httpx", "mesa", "faiss", "sqlite3", "pdfplumber", "fastapi", "numpy",
)


@pytest.mark.parametrize("path", list(python_files(SRC / "ports")), ids=rel)
def test_ports_import_no_third_party(path: Path):
    """ports 只能依赖标准库 typing —— 它是所有依赖箭头的终点。"""
    offenders = {
        m for m in imported_modules(path)
        if m.split(".")[0] in THIRD_PARTY
    }
    assert not offenders, f"{rel(path)} 拉进了第三方依赖: {sorted(offenders)}"


@pytest.mark.parametrize("path", list(python_files(SRC / "ports")), ids=rel)
def test_ports_import_no_implementation(path: Path):
    """ports 不能在运行时依赖任何具体实现模块。"""
    banned = ("src.llm", "src.rag", "src.persistence", "src.environment",
              "src.agents", "src.graph", "src.extractors")
    offenders = {
        m for m in imported_modules(path)
        if any(m.startswith(b) for b in banned)
    }
    assert not offenders, f"{rel(path)} 依赖了具体实现: {sorted(offenders)}"


# ═══════════════════════════════════════════════════════
#  2-3. 配置与全局状态
# ═══════════════════════════════════════════════════════


def _reads_environ(path: Path) -> bool:
    for node in ast.walk(_parse(path, runtime_only=False)):
        if (isinstance(node, ast.Attribute)
                and node.attr == "environ"
                and isinstance(node.value, ast.Name)
                and node.value.id == "os"):
            return True
    return False


def test_only_settings_reads_environ():
    """读环境变量只允许发生在 Settings 里。

    重构前这件事散在 base_agent / classifier / knowledge_base /
    base_extractor 四个文件，api 还要写 os.environ[...] 改全进程状态。
    """
    offenders = [
        rel(p) for p in python_files(SRC, ROOT / "run.py")
        if p.name != "settings.py" and _reads_environ(p)
    ]
    assert not offenders, f"这些文件绕过 Settings 直接读环境变量: {offenders}"


def test_no_module_level_llm_singleton():
    """不允许 `global _llm_instance` 这类模块级 LLM 缓存。

    复用客户端要做，但要放在 Provider 实例里，否则一个进程只能有一套配置。
    """
    offenders: List[str] = []
    for p in python_files(SRC):
        for node in ast.walk(_parse(p, runtime_only=False)):
            if isinstance(node, ast.Global) and any(
                "llm" in name.lower() for name in node.names
            ):
                offenders.append(f"{rel(p)}:{node.lineno}")
    assert not offenders, f"出现模块级 LLM 单例: {offenders}"


# ═══════════════════════════════════════════════════════
#  4-5. 依赖方向
# ═══════════════════════════════════════════════════════

# 只有装配层允许 import 具体的 LLM 供应商
ALLOWED_GENAI_IMPORTERS = {"src/llm/gemini.py"}


def test_only_adapter_imports_gemini():
    """领域层不许直接 import langchain_google_genai。"""
    offenders = [
        rel(p) for p in python_files(SRC)
        if rel(p) not in ALLOWED_GENAI_IMPORTERS
        and any(m.startswith("langchain_google_genai")
                for m in imported_modules(p))
    ]
    assert not offenders, f"这些文件绕过 LLMProvider 直接用 Gemini: {offenders}"


def test_persistence_does_not_depend_on_environment():
    """底层持久化不许反向依赖上层领域模型。"""
    offenders = [
        rel(p) for p in python_files(SRC / "persistence")
        if any(m.startswith("src.environment") for m in imported_modules(p))
    ]
    assert not offenders, f"persistence 反向依赖了 environment: {offenders}"


def test_only_composition_root_builds_concrete_deps():
    """具体实现的装配只应发生在 composition root 和入口。"""
    allowed = {
        "src/composition.py", "src/llm/__init__.py", "src/llm/gemini.py",
        "src/llm/null.py", "src/environment/engine.py",
    }
    offenders = [
        rel(p) for p in python_files(SRC)
        if rel(p) not in allowed
        and any(m in ("src.llm", "src.llm.gemini") for m in imported_modules(p))
    ]
    assert not offenders, f"这些文件自己装配了具体 LLM 实现: {offenders}"


# ═══════════════════════════════════════════════════════
#  6. 实现确实满足契约
# ═══════════════════════════════════════════════════════


def test_concrete_classes_satisfy_their_ports():
    """每个具体实现都要通过对应 Protocol 的 isinstance 检查。"""
    from src.extractors.pipeline import make_mock_extraction
    from src.llm import GeminiProvider, NullLLMProvider
    from src.persistence.policy_store import PolicyStore
    from src.ports import (
        KnowledgeProvider, LLMProvider, PolicyCache, PolicyFeed, SimulationSink,
    )
    from src.rag import RAGSystem
    from src.settings import Settings
    from src.spiders.policy_spider import PolicySpider

    assert isinstance(NullLLMProvider(), LLMProvider)
    assert isinstance(GeminiProvider(Settings()), LLMProvider)
    assert isinstance(PolicySpider(), PolicyFeed)
    assert isinstance(RAGSystem.build_static_only(make_mock_extraction()),
                      KnowledgeProvider)


def test_sqlite_implementations_satisfy_their_ports(tmp_path):
    from src.persistence.database import SimulationDB
    from src.persistence.policy_store import PolicyStore
    from src.ports import PolicyCache, SimulationSink

    db = SimulationDB(tmp_path / "sim.db")
    store = PolicyStore(tmp_path / "pol.db")
    try:
        assert isinstance(db, SimulationSink)
        assert isinstance(store, PolicyCache)
    finally:
        db.close()
        store.close()
