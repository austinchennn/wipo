"""入口模块 —— 广播宏观政策给所有智能体并运行模拟

用法:
    python -m src.environment.engine
    或在项目根目录: python run.py
"""

from __future__ import annotations

from ..composition import RunOptions, build_simulation
from ..settings import Settings
from .forum import ForumModel


def run_simulation(
    n_normal: int = 20,
    n_inst: int = 5,
    n_retail: int = 15,
    use_graph: bool = False,
    use_spider: bool = False,
    seed: int | None = 42,
) -> ForumModel:
    """从环境装配并执行 12 轮完整模拟（便捷入口，装配细节见 composition）。"""
    model = build_simulation(
        Settings.from_env(),
        RunOptions(
            n_normal=n_normal,
            n_inst=n_inst,
            n_retail=n_retail,
            use_graph=use_graph,
            use_spider=use_spider,
            seed=seed,
        ),
    )
    model.run()
    return model


if __name__ == "__main__":
    run_simulation()
