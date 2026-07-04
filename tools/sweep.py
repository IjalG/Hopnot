#!/usr/bin/env python3
"""Hopnot 参数扫描器 —— 批量测试参数组合对检索质量的影响。

用法:
  python tools/sweep.py                          # 默认扫描
  python tools/sweep.py --recall 0.1 0.3 0.5     # 自定义阈值
  python tools/sweep.py --output ./results.json  # 保存结果
"""

from __future__ import annotations

import itertools
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

# ── 添加项目根目录 ────────────────────────────────────────────────
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from hopnot import HippocampusMemorySystem, get_default_config
from hopnot.embedding import DummyEmbedding


# ── 测试数据集 ────────────────────────────────────────────────────

SAMPLE_KNOWLEDGE = """\
(人工智能, 包含, 机器学习)
(机器学习, 包含, 深度学习)
(深度学习, 使用, 神经网络)
(神经网络, 由, 神经元)
(机器学习, 分类, 监督学习)
(机器学习, 分类, 无监督学习)
(监督学习, 需要, 标注数据)
(无监督学习, 不需要, 标注数据)
(Python, 是, 编程语言)
(Python, 用于, 数据分析)
(Python, 用于, 人工智能)
(数据分析, 使用, pandas)
(数据分析, 使用, numpy)
"""

TEST_QUERIES = [
    # (query, expected_nodes)  expected_nodes 为空时只检查是否冷启动
    ("人工智能", ["人工智能", "机器学习", "深度学习"]),
    ("Python", ["Python"]),
    ("监督学习", ["监督学习", "机器学习"]),
    ("pandas", ["pandas", "数据分析"]),
    ("一个完全不存在的全新概念XKCD", []),  # 期望冷启动
    ("机器学习有哪些分类", ["机器学习", "监督学习", "无监督学习"]),
]


# ── 评估器 ────────────────────────────────────────────────────────

class Evaluator:
    """在测试数据集上评估参数组合。"""

    def __init__(self, config_override: dict[str, Any] | None = None) -> None:
        cfg = get_default_config()
        if config_override:
            for k, v in config_override.items():
                setattr(cfg, k, v)

        self.system = HippocampusMemorySystem(
            embedding=DummyEmbedding(dim=64, seed=42),
            config=cfg,
        )
        self._inject_knowledge()

    def _inject_knowledge(self) -> None:
        self.system.consolidate(SAMPLE_KNOWLEDGE, query="inject")

    def evaluate(self) -> dict[str, Any]:
        """运行所有测试查询，返回评估指标。"""
        total = len(TEST_QUERIES)
        cold_start_count = 0
        recall_scores = []
        activation_stats = []

        for query, expected in TEST_QUERIES:
            result = self.system.retrieve(query)

            if not expected:
                # 期望冷启动
                if result.cold_start:
                    cold_start_count += 1
                continue

            if result.cold_start:
                continue  # 不该冷启动却没命中

            activated_names = set()
            for nid, act in result.activated_nodes:
                node = self.system.graph.get_node(nid)
                if node:
                    activated_names.add(node.name)

            # 召回率：期望节点中有多少被激活
            if expected:
                hit = sum(1 for e in expected if e in activated_names)
                recall = hit / len(expected)
                recall_scores.append(recall)

            # 激活值分布
            if result.activated_nodes:
                acts = [a for _, a in result.activated_nodes]
                activation_stats.append({
                    "max": max(acts),
                    "mean": sum(acts) / len(acts),
                    "count": len(acts),
                })

        # 汇总
        cold_start_after = self.system.graph.node_count() - 11  # 11 = 初始节点数

        return {
            "node_count": self.system.graph.node_count(),
            "avg_recall": sum(recall_scores) / len(recall_scores) if recall_scores else 0,
            "cold_start_rate": cold_start_count / sum(1 for _, e in TEST_QUERIES if not e),
            "new_nodes_created": cold_start_after,
            "avg_activation_max": (
                sum(s["max"] for s in activation_stats) / len(activation_stats)
                if activation_stats else 0
            ),
            "avg_activation_count": (
                sum(s["count"] for s in activation_stats) / len(activation_stats)
                if activation_stats else 0
            ),
        }


# ── 参数扫描 ──────────────────────────────────────────────────────

DEFAULT_PARAM_GRID = {
    "recall_threshold": [0.05, 0.10, 0.20, 0.30],
    "restart_prob": [0.3, 0.4, 0.5],
    "decision_threshold": [0.35, 0.45, 0.55],
    "num_seeds": [2, 3, 5],
}

SWEEP_DISPLAY = {
    "recall_threshold": "τ_recall",
    "restart_prob": "ρ",
    "decision_threshold": "Θ",
    "num_seeds": "N",
}


def run_sweep(
    param_grid: dict[str, list[Any]] | None = None,
    output: str | None = None,
) -> list[dict[str, Any]]:
    """运行参数扫描。"""
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID

    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    results: list[dict[str, Any]] = []

    total = len(combos)
    print(f"参数扫描: {total} 个组合\n")
    print(f"{'#':>3}  {'  '.join(f'{SWEEP_DISPLAY.get(k, k):>10}' for k in keys)}  "
          f"{'召回率':>8}  {'冷启动率':>8}  {'节点数':>6}  {'耗时':>6}")
    print("-" * 80)

    for i, combo in enumerate(combos):
        config = dict(zip(keys, combo))
        t0 = time.time()

        try:
            evalator = Evaluator(config)
            metrics = evalator.evaluate()
            elapsed = time.time() - t0

            row = {**config, **metrics, "time": round(elapsed, 2)}
            results.append(row)

            vals = "  ".join(f"{v:>10}" for v in combo)
            print(f"{i+1:>3}  {vals}  "
                  f"{metrics['avg_recall']:>8.3f}  "
                  f"{metrics['cold_start_rate']:>8.2f}  "
                  f"{metrics['node_count']:>6}  "
                  f"{elapsed:>5.1f}s")

        except Exception as e:
            print(f"{i+1:>3}  {'  '.join(str(v) for v in combo)}  ERROR: {e}")

    # 排序：按召回率降序
    results.sort(key=lambda r: r.get("avg_recall", 0), reverse=True)

    print(f"\n--- 最佳参数组合 (按召回率) ---")
    for rank, r in enumerate(results[:5], 1):
        params = ", ".join(f"{k}={r[k]}" for k in keys)
        print(f"  #{rank}  {params}  → 召回率 {r['avg_recall']:.3f}")

    if output:
        Path(output).write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n结果已保存: {output}")

    return results


# ── 单次评估 ──────────────────────────────────────────────────────

def quick_eval(**params: Any) -> None:
    """使用指定参数单次评估。"""
    print(f"参数: {params}")
    e = Evaluator(params)
    m = e.evaluate()
    print(f"  节点数: {m['node_count']}")
    print(f"  平均召回率: {m['avg_recall']:.3f}")
    print(f"  冷启动率: {m['cold_start_rate']:.2f}")
    print(f"  平均激活值(最大): {m['avg_activation_max']:.4f}")
    print(f"  平均激活节点数: {m['avg_activation_count']:.1f}")


# ── 命令行 ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hopnot 参数扫描器")
    parser.add_argument("--recall", nargs="+", type=float, help="τ_recall 取值列表")
    parser.add_argument("--rho", nargs="+", type=float, help="ρ 取值列表")
    parser.add_argument("--theta", nargs="+", type=float, help="Θ 取值列表")
    parser.add_argument("--seeds", nargs="+", type=int, help="N 取值列表")
    parser.add_argument("--output", "-o", type=str, help="保存结果到 JSON 文件")
    parser.add_argument("--quick", nargs="*", help="快速单次评估：key=value ...")

    args = parser.parse_args()

    if args.quick:
        params = {}
        for kv in args.quick:
            k, _, v = kv.partition("=")
            try:
                params[k] = float(v) if "." in v else int(v)
            except ValueError:
                params[k] = v
        quick_eval(**params)
    else:
        grid = {}
        if args.recall:
            grid["recall_threshold"] = args.recall
        if args.rho:
            grid["restart_prob"] = args.rho
        if args.theta:
            grid["decision_threshold"] = args.theta
        if args.seeds:
            grid["num_seeds"] = args.seeds

        run_sweep(grid if grid else None, output=args.output)
