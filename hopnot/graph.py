"""核心图数据结构 —— 存储所有节点和边，提供图操作接口。"""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from typing import Optional

import numpy as np

from .config import HippocampusConfig, get_default_config
from .types import Edge, EdgeOrigin, EdgeType, Node, RedlistEntry


def _edge_key(source: str, target: str, tag: str = "") -> tuple[str, str, str]:
    return (source, target, tag)


class MemoryGraph:
    """有向加权记忆图。支持同一节点对之间存在多条边。"""

    def __init__(self, config: Optional[HippocampusConfig] = None) -> None:
        self.config = config or get_default_config()
        self._nodes: dict[str, Node] = {}
        self._edges: dict[tuple[str, str, str], Edge] = {}
        self._out_edges: dict[str, list[tuple[str, str, Edge]]] = defaultdict(list)
        self._in_edges: dict[str, list[tuple[str, str, Edge]]] = defaultdict(list)
        self._redlist: dict[tuple[str, str], RedlistEntry] = {}
        self._alias_index: dict[str, str] = {}
        self.operation_log: list[dict] = []
        self.conflict_log: list[dict] = []
        self.merged_log: list[dict] = []
        self._out_degree_cache: Optional[list[int]] = None
        self._freq_cache: Optional[list[int]] = None
        self._stats_updated_at: float = 0.0

        # ── NumPy 矩阵缓存 ────────────────────────────────────────
        self._emb_matrix: Optional[np.ndarray] = None   # (n_nodes, dim)
        self._emb_node_order: list[str] = []             # 行索引 → node_id

    # ── 节点操作 ──────────────────────────────────────────────────

    def add_node(self, node: Node) -> str:
        self._nodes[node.id] = node
        for alias in node.aliases:
            self._alias_index[alias] = node.id
        self._invalidate_all()
        return node.id

    def get_node(self, node_id: str) -> Optional[Node]:
        return self._nodes.get(node_id)

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def get_all_nodes(self) -> list[Node]:
        return list(self._nodes.values())

    def node_count(self) -> int:
        return len(self._nodes)

    def remove_node(self, node_id: str) -> bool:
        if node_id not in self._nodes:
            return False
        for _, tag, _ in list(self._out_edges.get(node_id, [])):
            self._remove_edge(node_id, tag)
        for src, tag, _ in list(self._in_edges.get(node_id, [])):
            self._remove_edge(src, tag, target_fixed=node_id)
        node = self._nodes.pop(node_id)
        for alias in node.aliases:
            self._alias_index.pop(alias, None)
        self._invalidate_all()
        return True

    # ── 别名操作 ──────────────────────────────────────────────────

    def add_alias(self, node_id: str, alias: str) -> None:
        if node_id in self._nodes:
            self._nodes[node_id].aliases.add(alias)
            self._alias_index[alias] = node_id

    def resolve_alias(self, text: str) -> Optional[str]:
        return self._alias_index.get(text)

    def find_node_by_name(self, name: str) -> Optional[Node]:
        for node in self._nodes.values():
            if node.name == name:
                return node
        return None

    # ── 边操作 ────────────────────────────────────────────────────

    def _make_tag(self, edge: Edge) -> str:
        if edge.refines is not None:
            return f"refines:{edge.refines}"
        return ""

    def add_edge(self, edge: Edge) -> None:
        tag = self._make_tag(edge)
        key = _edge_key(edge.source, edge.target, tag)
        self._edges[key] = edge

        out_list = self._out_edges[edge.source]
        idx = next((i for i, (t, tg, _) in enumerate(out_list) if t == edge.target and tg == tag), None)
        if idx is not None:
            out_list[idx] = (edge.target, tag, edge)
        else:
            out_list.append((edge.target, tag, edge))

        in_list = self._in_edges[edge.target]
        idx = next((i for i, (s, tg, _) in enumerate(in_list) if s == edge.source and tg == tag), None)
        if idx is not None:
            in_list[idx] = (edge.source, tag, edge)
        else:
            in_list.append((edge.source, tag, edge))

        self._invalidate_all()

    def get_edge(self, source: str, target: str) -> Optional[Edge]:
        return self._edges.get(_edge_key(source, target, ""))

    def get_edge_by_tag(self, source: str, target: str, tag: str) -> Optional[Edge]:
        return self._edges.get(_edge_key(source, target, tag))

    def get_all_edges_between(self, source: str, target: str) -> list[Edge]:
        return [e for (s, t, _), e in self._edges.items() if s == source and t == target]

    def has_edge(self, source: str, target: str) -> bool:
        return self.get_edge(source, target) is not None

    def has_refinement_edge(self, source: str, target: str) -> bool:
        return any(
            tag.startswith("refines:")
            for (s, t, tag), _ in self._edges.items()
            if s == source and t == target
        )

    def _remove_edge(self, source: str, tag: str = "", target_fixed: Optional[str] = None) -> bool:
        out_list = self._out_edges.get(source, [])
        to_remove = [(t, tg) for t, tg, _ in out_list if tg == tag]
        if not to_remove:
            return False
        t, tg = to_remove[0]
        key = _edge_key(source, target_fixed if target_fixed is not None else t, tg)
        self._edges.pop(key, None)
        self._out_edges[source] = [(t2, tg2, e) for t2, tg2, e in out_list if not (t2 == t and tg2 == tag)]
        target_key = t if target_fixed is None else target_fixed
        self._in_edges[target_key] = [
            (s, tg2, e) for s, tg2, e in self._in_edges.get(target_key, [])
            if not (s == source and tg2 == tag)
        ]
        self._invalidate_all()
        return True

    def remove_edge(self, source: str, target: str) -> bool:
        key = _edge_key(source, target, "")
        if key not in self._edges:
            return False
        del self._edges[key]
        self._out_edges[source] = [(t, tg, e) for t, tg, e in self._out_edges.get(source, []) if not (t == target and tg == "")]
        self._in_edges[target] = [(s, tg, e) for s, tg, e in self._in_edges.get(target, []) if not (s == source and tg == "")]
        self._invalidate_all()
        return True

    def get_out_edges(self, node_id: str) -> list[tuple[str, Edge]]:
        return [(t, e) for t, _, e in self._out_edges.get(node_id, [])]

    def get_in_edges(self, node_id: str) -> list[tuple[str, Edge]]:
        return [(s, e) for s, _, e in self._in_edges.get(node_id, [])]

    def get_out_neighbors(self, node_id: str) -> list[str]:
        seen: set[str] = set()
        return [t for t, _, _ in self._out_edges.get(node_id, []) if not (t in seen or seen.add(t))]

    def get_in_neighbors(self, node_id: str) -> list[str]:
        seen: set[str] = set()
        return [s for s, _, _ in self._in_edges.get(node_id, []) if not (s in seen or seen.add(s))]

    def out_degree(self, node_id: str) -> int:
        return len(self.get_out_neighbors(node_id))

    def in_degree(self, node_id: str) -> int:
        return len(self.get_in_neighbors(node_id))

    def edge_count(self) -> int:
        return len(self._edges)

    # ── NumPy 矩阵缓存 ────────────────────────────────────────────

    def _invalidate_all(self) -> None:
        """节点/边变化时使所有缓存失效。"""
        self._out_degree_cache = None
        self._freq_cache = None
        self._stats_updated_at = 0.0
        self._emb_matrix = None
        self._emb_node_order.clear()

    def _sync_emb_matrix(self) -> np.ndarray:
        """确保嵌入矩阵有效，返回 (n_nodes, dim) 矩阵。"""
        if self._emb_matrix is not None:
            return self._emb_matrix
        nodes = list(self._nodes.values())
        if not nodes:
            self._emb_matrix = np.empty((0, 0), dtype=np.float32)
            return self._emb_matrix
        dim = len(nodes[0].b)
        self._emb_node_order = [n.id for n in nodes]
        # 逐行构建，避免大列表转换
        m = np.empty((len(nodes), dim), dtype=np.float32)
        for i, node in enumerate(nodes):
            ev = self.compute_effective_vector(node)
            m[i] = ev
        self._emb_matrix = m
        return self._emb_matrix

    def batch_cosine_similarity(self, query_vec: list[float]) -> list[tuple[str, float]]:
        """一次矩阵乘法算全库余弦相似度，返回 [(node_id, sim)]。"""
        q = np.array(query_vec, dtype=np.float32)
        mat = self._sync_emb_matrix()
        if mat.size == 0:
            return []
        sims = q @ mat.T  # (d,) @ (d, n) → (n,)
        return list(zip(self._emb_node_order, sims.tolist()))

    def get_embeddings_subset(self, node_ids: list[str]) -> np.ndarray:
        """获取部分节点的嵌入矩阵 (m, dim)。"""
        mat = self._sync_emb_matrix()
        if mat.size == 0:
            return np.empty((0, 0), dtype=np.float32)
        indices = [self._emb_node_order.index(nid) for nid in node_ids if nid in self._emb_node_order]
        return mat[indices] if indices else np.empty((0, mat.shape[1]), dtype=np.float32)

    # ── 红名单操作 ────────────────────────────────────────────────

    def add_redlist(self, entry: RedlistEntry) -> None:
        key = (entry.source, entry.target)
        self._redlist[key] = entry
        self.log_operation("redlist_add", {"source": entry.source, "target": entry.target, "new_l2": entry.new_l2, "reason": entry.reason})

    def get_redlist(self, source: str, target: str) -> Optional[RedlistEntry]:
        return self._redlist.get((source, target))

    def remove_expired_redlist(self, now: Optional[float] = None) -> None:
        if now is None:
            now = time.time()
        for k in [k for k, v in self._redlist.items() if v.expire_at is not None and v.expire_at <= now]:
            del self._redlist[k]

    def is_redlist_active(self, source: str, target: str, now: Optional[float] = None) -> bool:
        entry = self._redlist.get((source, target))
        if entry is None:
            return False
        if entry.expire_at is not None:
            if now is None:
                now = time.time()
            if entry.expire_at <= now:
                return False
        return True

    # ── 向量操作工具 ──────────────────────────────────────────────

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        if not a or not b:
            return 0.0
        ab = dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    @staticmethod
    def normalize(vec: list[float]) -> list[float]:
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec] if norm else vec

    @staticmethod
    def vector_add(a: list[float], b: list[float]) -> list[float]:
        return [x + y for x, y in zip(a, b)]

    @staticmethod
    def vector_scale(vec: list[float], s: float) -> list[float]:
        return [x * s for x in vec]

    @staticmethod
    def vector_norm(vec: list[float]) -> float:
        return math.sqrt(sum(x * x for x in vec))

    def clamp_vector_norm(self, vec: list[float], max_norm: float) -> list[float]:
        n = self.vector_norm(vec)
        return [x * (max_norm / n) for x in vec] if n > max_norm else vec

    def compute_effective_vector(self, node: Node) -> list[float]:
        alpha = self.config.alpha_bias
        biased = self.vector_add(node.b, self.vector_scale(node.p, alpha))
        return self.normalize(biased)

    # ── 全库统计 ──────────────────────────────────────────────────

    def _ensure_stats(self) -> None:
        if self._out_degree_cache is not None:
            return
        self._out_degree_cache = [self.out_degree(nid) for nid in self._nodes]
        self._freq_cache = [n.freq for n in self._nodes.values()]

    def percentile_95(self, values: list[int]) -> int:
        if not values:
            return 0
        sorted_vals = sorted(values)
        idx = int(math.ceil(0.95 * len(sorted_vals))) - 1
        return sorted_vals[max(0, idx)]

    def get_deg_p95(self) -> int:
        self._ensure_stats()
        return self.percentile_95(self._out_degree_cache or [])

    def get_freq_p95(self) -> int:
        self._ensure_stats()
        return self.percentile_95(self._freq_cache or [])

    def get_avg_out_degree(self) -> float:
        self._ensure_stats()
        vals = self._out_degree_cache or []
        return sum(vals) / len(vals) if vals else 0.0

    # ── 日志 ──────────────────────────────────────────────────────

    def log_operation(self, op_type: str, details: dict) -> None:
        self.operation_log.append({"type": op_type, "time": time.time(), **details})

    def log_conflict(self, entry: dict) -> None:
        self.conflict_log.append({"time": time.time(), **entry})

    def log_merge(self, unk_id: str, target_id: str) -> None:
        self.merged_log.append({"unk_node_id": unk_id, "target_node_id": target_id, "time": time.time()})

    # ── 闲聊门控 ──────────────────────────────────────────────────

    def is_chitchat(self, query: str) -> bool:
        import re
        return any(re.match(p, query.strip()) for p in self.config.chitchat_patterns)

    # ── 序列化快照 ────────────────────────────────────────────────

    def to_snapshot(self) -> dict:
        return {
            "nodes": {nid: {"id": n.id, "name": n.name, "b": n.b, "p": n.p, "l3": n.l3, "freq": n.freq,
                            "created_at": n.created_at, "last_visited": n.last_visited, "aliases": list(n.aliases)}
                      for nid, n in self._nodes.items()},
            "edges": {f"{e.source}->{e.target}" + (f"({e.refines})" if e.refines else ""): e.__dict__
                      for key, e in self._edges.items()},
        }
