"""
启动论坛沙盘模拟

用法:
    # 使用内置 Mock 数据（无需 PDF / API Key）
    python run.py

    # 传入招股书 PDF
    python run.py --pdf path/to/prospectus.pdf

    # 设置 Active Agent 数量（其余为 Passive，情绪统计推算）
    python run.py --normal 5000 --inst 500 --retail 4500 --active 1000

    # 完整参数示例
    python run.py --pdf prospectus.pdf \\
                  --normal 700 --inst 100 --retail 200 \\
                  --active 1000 --concurrent 50 \\
                  --model gpt-4o-mini --no-rag --seed 42

成本参考（gpt-4o-mini）：
    1,000 active agents → ~$26 / 完整模拟（12 轮）
    10,000 active agents → ~$260 / 完整模拟
"""

from __future__ import annotations

import argparse

from src.environment.forum import ForumModel


def main():
    parser = argparse.ArgumentParser(description="金融舆情论坛沙盘")

    # ── Agent 规模 ──
    parser.add_argument("--normal",     type=int,  default=20,
                        help="普通人总数（默认 20）")
    parser.add_argument("--inst",       type=int,  default=5,
                        help="机构交易者总数（默认 5）")
    parser.add_argument("--retail",     type=int,  default=15,
                        help="散户交易者总数（默认 15）")
    parser.add_argument("--active",     type=int,  default=None,
                        help="实际调 LLM 的 Agent 数量上限（默认=全部 Active）")

    # ── 数据来源 ──
    parser.add_argument("--pdf",        type=str,  default=None,
                        help="招股书 PDF 路径（不传则使用内置 Mock 数据）")
    parser.add_argument("--no-rag",     action="store_true",
                        help="禁用 FAISS RAG 知识库（仅使用静态提取文本）")

    # ── LLM 配置 ──
    parser.add_argument("--model",      type=str,  default="gpt-4o-mini",
                        help="LLM 模型名（默认 gpt-4o-mini）")
    parser.add_argument("--concurrent", type=int,  default=50,
                        help="并发 LLM 调用上限（默认 50，防 rate limit）")

    parser.add_argument("--seed",       type=int,  default=42,
                        help="随机种子（默认 42）")

    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    model = ForumModel(
        n_normal=args.normal,
        n_inst=args.inst,
        n_retail=args.retail,
        n_active=args.active,
        pdf_path=args.pdf,
        llm_model=args.model,
        use_rag=not args.no_rag,
        max_concurrent=args.concurrent,
        seed=args.seed,
    )
    model.run()


if __name__ == "__main__":
    main()
