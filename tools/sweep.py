#!/usr/bin/env python3
"""Hopnot 参数扫描器 —— 批量测试参数组合对检索质量的影响。

用法:
  # 快速单次评估（使用 Qwen3 嵌入）
  python tools/sweep.py --scan data01 --quick recall_threshold=0.10

  # 全量扫描（使用 Qwen3 嵌入，默认）
  python tools/sweep.py --scan data01

  # 全量扫描（使用 DummyEmbedding，快速但不准）
  python tools/sweep.py --scan data01 --embedding dummy
"""

from __future__ import annotations

import itertools
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── 添加项目根目录 ────────────────────────────────────────────────
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from hopnot import HippocampusMemorySystem, get_default_config
from hopnot.embedding import DummyEmbedding


# ── 数据集 ────────────────────────────────────────────────────────

@dataclass
class Dataset:
    name: str
    knowledge: str
    queries: list[tuple[str, list[str]]]


def load_knowledge(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_queries(path: str | Path) -> list[tuple[str, list[str]]]:
    queries: list[tuple[str, list[str]]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "|" not in line:
            continue
        query, _, expected = line.partition("|")
        query = query.strip()
        expected_list = [e.strip() for e in expected.split(",") if e.strip()]
        queries.append((query, expected_list))
    return queries


# ── 内置默认数据集 ────────────────────────────────────────────────

DEFAULT_KNOWLEDGE = """\
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

DEFAULT_QUERIES = [
    ("人工智能", ["人工智能", "机器学习", "深度学习"]),
    ("Python", ["Python"]),
    ("监督学习", ["监督学习", "机器学习"]),
    ("pandas", ["pandas", "数据分析"]),
    ("完全不存在的全新概念XKCD", []),
    ("机器学习有哪些分类", ["机器学习", "监督学习", "无监督学习"]),
]

DEFAULT_DATASET = Dataset(name="默认", knowledge=DEFAULT_KNOWLEDGE, queries=DEFAULT_QUERIES)


# ── 评估器 ────────────────────────────────────────────────────────

class Evaluator:
    """在指定数据集上评估参数组合。
    
    Args:
        dataset: 测试数据集
        config_override: 要覆盖的参数字典
        embedder: 嵌入模型实例（复用外部创建，避免重复加载）
    """

    def __init__(
        self,
        dataset: Dataset,
        config_override: dict[str, Any] | None = None,
        embedder=None,
    ) -> None:
        cfg = get_default_config()
        # 评估时不过滤低激活节点，保证召回率统计完整
        cfg.output_threshold = 0.0
        if config_override:
            for k, v in config_override.items():
                setattr(cfg, k, v)

        if embedder is None:
            embedder = DummyEmbedding(dim=64, seed=42)

        self.system = HippocampusMemorySystem(embedding=embedder, config=cfg)
        self.dataset = dataset
        self._inject_knowledge()

    def _inject_knowledge(self) -> None:
        self.system.consolidate(self.dataset.knowledge, query="inject")

    def evaluate(self) -> dict[str, Any]:
        cold_start_ok = 0
        cold_start_total = 0
        recall_scores = []
        activation_stats = []

        for query, expected in self.dataset.queries:
            result = self.system.retrieve(query)

            if not expected:
                cold_start_total += 1
                if result.cold_start:
                    cold_start_ok += 1
                continue

            if result.cold_start:
                recall_scores.append(0.0)
                continue

            activated_names = set()
            for nid, act in result.activated_nodes:
                node = self.system.graph.get_node(nid)
                if node:
                    activated_names.add(node.name)

            hit = sum(1 for e in expected if e in activated_names)
            recall_scores.append(hit / len(expected))

            if result.activated_nodes:
                acts = [a for _, a in result.activated_nodes]
                activation_stats.append({
                    "max": max(acts), "mean": sum(acts) / len(acts), "count": len(acts),
                })

        return {
            "node_count": self.system.graph.node_count(),
            "avg_recall": sum(recall_scores) / len(recall_scores) if recall_scores else 0,
            "query_count": len(recall_scores),
            "cold_start_accuracy": cold_start_ok / cold_start_total if cold_start_total > 0 else 1.0,
            "avg_activation_max": sum(s["max"] for s in activation_stats) / len(activation_stats) if activation_stats else 0,
            "avg_activation_count": sum(s["count"] for s in activation_stats) / len(activation_stats) if activation_stats else 0,
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
    datasets: list[Dataset],
    embedder,
    param_grid: dict[str, list[Any]] | None = None,
    output: str | None = None,
) -> list[dict[str, Any]]:
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID

    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    results: list[dict[str, Any]] = []
    total_combos = len(combos) * len(datasets)

    header = f"{'#':>3}  {'  '.join(f'{SWEEP_DISPLAY.get(k, k):>10}' for k in keys)}"
    for ds in datasets:
        header += f"  {f'{ds.name[:4]}召回':>8}  {f'{ds.name[:4]}冷启':>8}"
    header += f"  {'平均召回':>8}  {'节点':>6}"
    print(f"数据集: {' | '.join(f'{d.name}({len(d.queries)}查询)' for d in datasets)}")
    print(f"参数扫描: {total_combos} 个组合（{type(embedder).__name__}）\n")
    print(header)
    print("-" * len(header))

    for i, combo in enumerate(combos):
        config = dict(zip(keys, combo))
        t0 = time.time()

        try:
            row: dict[str, Any] = {**config}
            per_dataset_metrics = []

            for ds in datasets:
                evalator = Evaluator(ds, config, embedder=embedder)
                metrics = evalator.evaluate()
                per_dataset_metrics.append(metrics)

            avg_recall = sum(m["avg_recall"] for m in per_dataset_metrics) / len(per_dataset_metrics)
            avg_cold = sum(m["cold_start_accuracy"] for m in per_dataset_metrics) / len(per_dataset_metrics)
            total_nodes = max(m["node_count"] for m in per_dataset_metrics)

            row["per_dataset"] = per_dataset_metrics
            row["avg_recall"] = avg_recall
            row["avg_cold_start"] = avg_cold
            row["total_nodes"] = total_nodes
            row["time"] = round(time.time() - t0, 2)
            results.append(row)

            vals = "  ".join(f"{v:>10}" for v in combo)
            recall_strs = "".join(f"  {m['avg_recall']:>8.3f}  {m['cold_start_accuracy']:>8.2f}" for m in per_dataset_metrics)
            print(f"{i+1:>3}  {vals}{recall_strs}  {avg_recall:>8.3f}  {total_nodes:>6}")

        except Exception as e:
            print(f"{i+1:>3}  {'  '.join(str(v) for v in combo)}  ERROR: {e}")

    results.sort(key=lambda r: r.get("avg_recall", 0), reverse=True)

    print(f"\n--- 最佳参数组合 (按平均召回率) ---")
    for rank, r in enumerate(results[:5], 1):
        params = ", ".join(f"{k}={r[k]}" for k in keys)
        recalls = " | ".join(f"{d.name}={m['avg_recall']:.3f}" for d, m in zip(datasets, r.get("per_dataset", [])))
        print(f"  #{rank}  {params}  →  {recalls}  |  avg={r['avg_recall']:.3f}")

    if output:
        Path(output).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已保存: {output}")

    return results


# ── 快速单次评估 ──────────────────────────────────────────────────

def quick_eval(datasets: list[Dataset], embedder, **params: Any) -> None:
    print(f"参数: {params}")
    for ds in datasets:
        e = Evaluator(ds, params, embedder=embedder)
        m = e.evaluate()
        print(f"  [{ds.name}]")
        print(f"    节点数: {m['node_count']}")
        print(f"    召回率: {m['avg_recall']:.3f} ({m['query_count']} 个查询)")
        print(f"    冷启动准确率: {m['cold_start_accuracy']:.0%}")
        print(f"    平均最大激活值: {m['avg_activation_max']:.4f}")


# ── 命令行 ────────────────────────────────────────────────────────

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Hopnot 参数扫描器")
    parser.add_argument("--recall", nargs="+", type=float, help="τ_recall 取值列表")
    parser.add_argument("--rho", nargs="+", type=float, help="ρ 取值列表")
    parser.add_argument("--theta", nargs="+", type=float, help="Θ 取值列表")
    parser.add_argument("--seeds", nargs="+", type=int, help="N 取值列表")
    parser.add_argument("--output", "-o", type=str, help="保存结果到 JSON 文件")
    parser.add_argument(
        "--embedding", choices=["qwen3", "dummy"], default="qwen3",
        help="嵌入模型：qwen3（真实语义，默认）| dummy（快速测试）",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="运行设备：cpu（默认）| cuda（需要 GPU）",
    )
    parser.add_argument("--data", nargs="+", action="append", metavar="PATH",
                        help="外部数据集：knowledge.txt queries.txt [场景名]")
    parser.add_argument("--datasets", type=str,
                        help="数据集索引文件，每行: knowledge.txt queries.txt 场景名")
    parser.add_argument("--scan", type=str, default="",
                        help="扫描文件夹，自动配对 knowledge_xxx.txt + queries_xxx.txt")
    parser.add_argument("--quick", nargs="*", help="快速单次评估：key=value ...")
    return parser.parse_args()


def _scan_folder(folder: str) -> list[Dataset]:
    fdir = Path(folder)
    if not fdir.is_dir():
        print(f"[警告] 目录不存在: {folder}")
        return []
    k_map: dict[str, Path] = {}
    q_map: dict[str, Path] = {}
    for f in fdir.glob("*.txt"):
        stem = f.stem
        if stem.startswith("knowledge_") or stem.startswith("queries_"):
            prefix, _, suffix = stem.partition("_")
            if suffix:
                (k_map if prefix == "knowledge" else q_map)[suffix] = f
    datasets = []
    for suffix in sorted(set(k_map.keys()) | set(q_map.keys())):
        kp, qp = k_map.get(suffix), q_map.get(suffix)
        if kp and qp:
            datasets.append(Dataset(name=suffix, knowledge=load_knowledge(kp), queries=load_queries(qp)))
        elif kp and not qp:
            print(f"[警告] {kp.name} 缺少对应的 queries_{suffix}.txt")
        elif qp and not kp:
            print(f"[警告] {qp.name} 缺少对应的 knowledge_{suffix}.txt")
    return datasets


def load_datasets_from_args(args) -> list[Dataset]:
    datasets = []
    if args.data:
        for spec in args.data:
            if len(spec) < 2:
                continue
            name = spec[2] if len(spec) > 2 else Path(spec[0]).stem
            datasets.append(Dataset(name=name, knowledge=load_knowledge(spec[0]), queries=load_queries(spec[1])))
    if args.datasets:
        for line in Path(args.datasets).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                kp, qp = Path(parts[0]), Path(parts[1])
                if kp.exists() and qp.exists():
                    name = parts[2] if len(parts) > 2 else kp.stem
                    datasets.append(Dataset(name=name, knowledge=load_knowledge(kp), queries=load_queries(qp)))
    if args.scan:
        datasets.extend(_scan_folder(args.scan))
    return datasets


def main():
    args = parse_args()

    datasets = load_datasets_from_args(args)
    if not datasets:
        datasets = [DEFAULT_DATASET]

    # 创建嵌入模型（复用，避免重复加载）
    if args.embedding == "qwen3":
        print("正在加载 Qwen3-Embedding-0.6B...", end=" ", flush=True)
        t0 = time.time()
        try:
            from hopnot.embedding import Qwen3Embedding
            embedder = Qwen3Embedding(device=args.device)
            print(f"完成 ({time.time()-t0:.1f}s, device={args.device})")
        except Exception as e:
            print(f"\n[失败] Qwen3 加载错误: {e}")
            print("[回退] 使用 DummyEmbedding（随机向量，结果不可靠）")
            embedder = DummyEmbedding(dim=64, seed=42)
            print(f"完成 ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"失败: {e}")
            print("回退到 DummyEmbedding")
            embedder = DummyEmbedding(dim=64, seed=42)
    else:
        embedder = DummyEmbedding(dim=64, seed=42)

    grid = {}
    if args.recall:
        grid["recall_threshold"] = args.recall
    if args.rho:
        grid["restart_prob"] = args.rho
    if args.theta:
        grid["decision_threshold"] = args.theta
    if args.seeds:
        grid["num_seeds"] = args.seeds

    if args.quick is not None:
        params = {}
        for kv in args.quick:
            k, _, v = kv.partition("=")
            try:
                params[k] = float(v) if "." in v else int(v)
            except ValueError:
                params[k] = v
        quick_eval(datasets, embedder, **params)
    else:
        run_sweep(datasets, embedder, grid if grid else None, output=args.output)


if __name__ == "__main__":
    main()
