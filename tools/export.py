#!/usr/bin/env python3
"""Hopnot 记忆图导出工具 —— 导出为 Graphviz DOT / JSON / 文本格式。

用法:
  python tools/export.py                         # 使用默认记忆打印到终端
  python tools/export.py --format dot > graph.dot # 导出 DOT 格式
  python tools/export.py --format json           # 导出 JSON
  python tools/export.py --input snapshot.json   # 从快照导入
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from hopnot import HippocampusMemorySystem, get_default_config
from hopnot.embedding import DummyEmbedding


def build_system(snapshot_path: str | None = None) -> HippocampusMemorySystem:
    """构建系统，可选从快照恢复。"""
    system = HippocampusMemorySystem(
        embedding=DummyEmbedding(dim=64, seed=42),
        config=get_default_config(),
    )

    if snapshot_path:
        data = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
        # 重建节点
        for nid, ndata in data.get("nodes", {}).items():
            from hopnot.types import Node
            node = Node(
                id=ndata["id"],
                name=ndata["name"],
                b=ndata["b"],
                p=ndata.get("p", [0.0] * len(ndata["b"])),
                l3=ndata.get("l3", 0.3),
                freq=ndata.get("freq", 0),
            )
            system.graph.add_node(node)
        # 重建边
        for key, edata in data.get("edges", {}).items():
            from hopnot.types import Edge, EdgeType, EdgeOrigin
            edge = Edge(
                source=edata["source"],
                target=edata["target"],
                l2=edata.get("l2", 0.45),
                confidence=edata.get("confidence", 0.5),
                edge_type=EdgeType(edata.get("edge_type", "ASSOC")),
                origin=EdgeOrigin(edata.get("origin", "explicit_fact")),
                refines=edata.get("refines"),
            )
            system.graph.add_edge(edge)
    else:
        # 注入示例数据
        system.consolidate(
            "(Hopnot, 是, 纯图结构记忆模拟)\n"
            "(检索阶段, 包含, 种子选取)\n"
            "(检索阶段, 包含, 随机游走扩散)\n"
            "(检索阶段, 包含, 输出截断)\n"
            "(整理阶段, 包含, 节点定位)\n"
            "(整理阶段, 包含, 边处理决策树)\n"
            "(整理阶段, 包含, 偏置漂移)\n"
            "(整理阶段, 包含, 三角闭合)",
            query="init",
        )

    return system


def export_text(system: HippocampusMemorySystem) -> str:
    """文本格式输出。"""
    lines = [
        f"Hopnot 记忆图",
        f"节点: {system.graph.node_count()}  边: {system.graph.edge_count()}",
        "",
    ]
    nodes = system.graph.get_all_nodes()
    for node in sorted(nodes, key=lambda n: n.l3, reverse=True):
        edges = system.graph.get_out_edges(node.id)
        edge_str = " | ".join(
            f"→ {system.graph.get_node(t).name if system.graph.get_node(t) else t[:8]}"
            f" (L2={e.l2:.2f})"
            for t, e in edges[:5]
        ) if edges else "(无出边)"
        lines.append(f"  {node.name}  L3={node.l3:.2f}  freq={node.freq}")
        lines.append(f"    {edge_str}")
    return "\n".join(lines)


def export_dot(system: HippocampusMemorySystem) -> str:
    """Graphviz DOT 格式输出。"""
    lines = [
        'digraph Hopnot {',
        '  rankdir=LR;',
        '  node [shape=box, style=filled, fillcolor="#E8F0FE", fontname="sans"];',
        '  edge [fontname="sans", fontsize=10];',
    ]

    nodes = system.graph.get_all_nodes()
    for node in nodes:
        l3_color = int(200 - node.l3 * 100)
        fillcolor = f"#E8F0FE" if node.l3 > 0.4 else f"#F5F5F5"
        lines.append(
            f'  "{node.id}" [label="{node.name}\\nL3={node.l3:.2f}", fillcolor="{fillcolor}"];'
        )

    seen: set[tuple[str, str]] = set()
    for node in nodes:
        for target, edge in system.graph.get_out_edges(node.id):
            key = (node.id, target)
            if key in seen:
                continue
            seen.add(key)
            color = "#FF6B6B" if edge.edge_type.name == "NOT" else "#4A90D9"
            style = "dashed" if edge.origin.name == "INFERRED" else "solid"
            label = f"L2={edge.l2:.2f}"
            if edge.refines:
                label += f"\\nrefines={edge.refines}"
            lines.append(
                f'  "{node.id}" -> "{target}" [label="{label}", '
                f'color="{color}", style="{style}"];'
            )

    lines.append("}")
    return "\n".join(lines)


def export_json(system: HippocampusMemorySystem, pretty: bool = True) -> str:
    """JSON 格式输出。"""
    return json.dumps(
        system.graph.to_snapshot(),
        ensure_ascii=False,
        indent=2 if pretty else None,
    )


# ── 命令行 ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Hopnot 记忆图导出工具")
    parser.add_argument("--format", choices=["text", "dot", "json"], default="text")
    parser.add_argument("--input", "-i", type=str, help="从快照 JSON 导入")
    parser.add_argument("--output", "-o", type=str, help="写入文件（默认输出到终端）")
    parser.add_argument(
        "--scan", type=str, default="",
        help="扫描文件夹，自动导入所有 knowledge_xxx.txt + queries_xxx.txt 并导出",
    )
    parser.add_argument(
        "--outdir", type=str, default="",
        help="--scan 时输出目录（默认同 --scan 目录）",
    )
    args = parser.parse_args()

    if args.scan:
        scan_dir = Path(args.scan)
        out_dir = Path(args.outdir or args.scan)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 用与 sweep 同样的配对逻辑
        k_map: dict[str, Path] = {}
        q_map: dict[str, Path] = {}
        for f in scan_dir.glob("*.txt"):
            stem = f.stem
            if stem.startswith("knowledge_") or stem.startswith("queries_"):
                prefix, _, suffix = stem.partition("_")
                if suffix:
                    (k_map if prefix == "knowledge" else q_map)[suffix] = f

        all_suffixes = sorted(set(k_map.keys()) | set(q_map.keys()))
        exported = 0
        for suffix in all_suffixes:
            kpath = k_map.get(suffix)
            qpath = q_map.get(suffix)
            if not kpath or not qpath:
                continue
            # 注入知识构建系统
            system = HippocampusMemorySystem(
                embedding=DummyEmbedding(dim=64, seed=42),
                config=get_default_config(),
            )
            system.consolidate(kpath.read_text(encoding="utf-8"), query="import")

            # 根据格式导出
            if args.format == "text":
                output = export_text(system)
            elif args.format == "dot":
                output = export_dot(system)
            else:
                output = export_json(system)

            fname = out_dir / f"{suffix}.{args.format}"
            fname.write_text(output, encoding="utf-8")
            print(f"已导出: {fname}")
            exported += 1

        if exported == 0:
            print(f"[提示] {scan_dir} 中未找到配对的 knowledge_*.txt + queries_*.txt")
        else:
            print(f"共导出 {exported} 个数据集")
    else:
        system = build_system(args.input)
        if args.format == "text":
            output = export_text(system)
        elif args.format == "dot":
            output = export_dot(system)
        else:
            output = export_json(system)

        if args.output:
            Path(args.output).write_text(output, encoding="utf-8")
            print(f"已保存: {args.output}")
        else:
            print(output)
