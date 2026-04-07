"""入口模块 —— 广播宏观政策给所有智能体并运行模拟

用法:
    python -m src.environment.engine
    或在项目根目录: python run.py
"""

from __future__ import annotations

from .forum import ForumModel


def run_simulation(
    n_normal: int = 20,
    n_inst: int = 5,
    n_retail: int = 15,
    seed: int | None = 42,
) -> ForumModel:
    """创建 ForumModel 并执行 12 轮完整模拟"""
    model = ForumModel(
        n_normal=n_normal,
        n_inst=n_inst,
        n_retail=n_retail,
        seed=seed,
    )
    model.run()
    return model


if __name__ == "__main__":
    run_simulation()
