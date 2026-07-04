"""检索阶段 —— 记忆读取。

实现规范（v1.7）：
1. 有效边权重计算（含细化边压制与红名单覆盖）
2. 种子节点选取（三阶段：粗筛→邻居投票→能量归一化）
3. 记忆扩散（带前缀均值语义评估的 RWR）
4. 输出格式（Token 截断池拆分 + 冷启动豁免）
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import HippocampusConfig, get_default_config
from .embedding import BaseEmbedding
from .graph import MemoryGraph
from .types import Edge, EdgeOrigin, EdgeType, Node

logger = logging.getLogger(__name__)

# ── 可选加速库检测 ──────────────────────────────────────────────
try:
    import scipy.sparse as sp
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

try:
    import faiss
    HAVE_FAISS = True
except ImportError:
    HAVE_FAISS = False

FAISS_MIN_NODES = 5000  # FAISS 生效的最小节点数


# ── 检索结果数据结构 ──────────────────────────────────────────

@dataclass
class RetrievalResult:
    activated_nodes: list[tuple[str, float]] = field(default_factory=list)
    tokens_used: int = 0
    cold_start: bool = False
    cold_start_node_id: Optional[str] = None
    query: str = ""
    seed_nodes: list[str] = field(default_factory=list)
    num_steps: int = 0


# ── 检索实现 ──────────────────────────────────────────────────

class MemoryRetrieval:
    def __init__(
        self,
        graph: MemoryGraph,
        embedding: BaseEmbedding,
        config: Optional[HippocampusConfig] = None,
    ) -> None:
        self.graph = graph
        self.embed = embedding
        self.config = config or get_default_config()
        # FAISS 索引（惰性构建）
        self._faiss_index = None
        self._faiss_node_order: list[str] = []

    # ═══════════════════════════════════════════════════════════
    #  §2.1  有效边权重计算
    # ═══════════════════════════════════════════════════════════

    def compute_effective_weight(
        self,
        edge: Edge,
        l3_eff: float,
        has_refinement: bool = False,
        redlist_override: Optional[float] = None,
        now: Optional[float] = None,
    ) -> float:
        if edge.edge_type == EdgeType.NOT:
            return 0.0
        if redlist_override is not None:
            return max(-0.2, min(1.0, redlist_override))
        refine_penalty = self.config.decay_general if has_refinement and edge.refines is None else 1.0
        base = 1.0 * edge.l2 * (0.6 + 0.4 * l3_eff) * refine_penalty
        w = base + 0.0 * 0.3  # L1 预留
        w += random.gauss(0.0, 0.05)
        return max(-0.2, min(1.0, w))

    # ═══════════════════════════════════════════════════════════
    #  §2.2  种子节点选取
    # ═══════════════════════════════════════════════════════════

    def _exact_match_candidates(self, query: str) -> list[str]:
        candidates: list[str] = []
        nid = self.graph.resolve_alias(query)
        if nid is not None:
            candidates.append(nid)
        node = self.graph.find_node_by_name(query)
        if node is not None and node.id not in candidates:
            candidates.append(node.id)
        for alias, nid in self.graph._alias_index.items():
            if query.lower() in alias.lower() and nid not in candidates:
                candidates.append(nid)
        return candidates

    def _build_faiss_index(self) -> None:
        """惰性构建 FAISS 索引（节点数 > 5000 时启用）。"""
        if not HAVE_FAISS:
            return
        nodes = self.graph.get_all_nodes()
        if len(nodes) < FAISS_MIN_NODES:
            return
        mat = self.graph._sync_emb_matrix()
        if mat.size == 0:
            return
        dim = mat.shape[1]
        self._faiss_index = faiss.IndexFlatIP(dim)
        self._faiss_index.add(mat.astype(np.float32))
        self._faiss_node_order = list(self.graph._emb_node_order)

    def recall_candidates(self, query: str) -> tuple[list[tuple[str, float]], bool, Optional[str]]:
        """§2.2.1 候选召回（粗筛）+ 冷启动门控。"""
        query_embed = self.embed.embed(query)

        if self.graph.is_chitchat(query):
            logger.info("Chitchat gate triggered for query: %s", query)
            return [], False, None

        # ── 批量余弦相似度 ────────────────────────────────────────
        n_nodes = self.graph.node_count()
        candidates: list[tuple[str, float]] = []

        if HAVE_FAISS and n_nodes >= FAISS_MIN_NODES:
            # FAISS 近似搜索
            if self._faiss_index is None:
                self._build_faiss_index()
            if self._faiss_index is not None:
                q = np.array([query_embed], dtype=np.float32)
                scores, indices = self._faiss_index.search(q, min(100, n_nodes))
                for score, idx in zip(scores[0], indices[0]):
                    if idx >= 0 and idx < len(self._faiss_node_order) and score >= self.config.recall_threshold:
                        candidates.append((self._faiss_node_order[idx], float(score)))
        else:
            # NumPy 矩阵批量余弦
            batch = self.graph.batch_cosine_similarity(query_embed)
            for nid, sim in batch:
                if sim >= self.config.recall_threshold:
                    candidates.append((nid, sim))

        # 精确实例匹配（确保精确命中仍在候选池中）
        exact_ids = self._exact_match_candidates(query)
        if exact_ids:
            batch = dict(self.graph.batch_cosine_similarity(query_embed)) if candidates else {}
            for nid in exact_ids:
                if not any(c[0] == nid for c in candidates):
                    sim = batch.get(nid, 0.5) if batch else 0.5
                    candidates.append((nid, max(sim, 0.5)))

        # 冷启动门控
        if not candidates:
            logger.info("Cold start triggered for query: %s", query)
            cold_node = Node(
                id=f"cold_{int(time.time() * 1000)}_{hash(query) % 10000}",
                name=query, b=query_embed, l3=self.config.l3_cold_start,
                created_at=time.time(), last_visited=time.time(),
            )
            self.graph.add_node(cold_node)
            return [(cold_node.id, 1.0)], True, cold_node.id

        return candidates, False, None

    def context_disambiguation(self, candidates: list[tuple[str, float]], query: str) -> list[tuple[str, float]]:
        """§2.2.2 上下文消歧 —— 邻居投票。"""
        query_embed = self.embed.embed(query)
        cfg = self.config
        scored: list[tuple[str, float]] = []

        for node_id, sim in candidates:
            node = self.graph.get_node(node_id)
            if node is None:
                continue

            out_edges = self.graph.get_out_edges(node_id)
            neighbor_count = len(out_edges)

            if neighbor_count > cfg.neighbor_cutoff:
                out_edges.sort(key=lambda x: x[1].l2, reverse=True)
                out_edges = out_edges[:cfg.neighbor_cutoff]

            if neighbor_count > 0:
                # 批量算邻居相似度
                neighbor_ids = [t for t, _ in out_edges if self.graph.get_node(t) is not None]
                if neighbor_ids:
                    sub_mat = self.graph.get_embeddings_subset(neighbor_ids)
                    if sub_mat.size > 0:
                        q = np.array(query_embed, dtype=np.float32)
                        sims = q @ sub_mat.T
                        avg_neighbor = float(sims.mean())
                    else:
                        avg_neighbor = 0.0
                else:
                    avg_neighbor = 0.0
            else:
                avg_neighbor = 0.0

            recency_bonus = cfg.recency_bias if (time.time() - node.last_visited) < 30.0 else 0.0
            score = sim + cfg.neighbor_vote_weight * avg_neighbor + recency_bonus
            scored.append((node_id, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:cfg.num_seeds]

    def seed_energy_normalization(self, seeds: list[tuple[str, float]]) -> dict[str, float]:
        total = sum(score for _, score in seeds) or 1.0
        return {nid: score / total for nid, score in seeds}

    # ═══════════════════════════════════════════════════════════
    #  §2.3  记忆扩散（RWR）
    # ═══════════════════════════════════════════════════════════

    def build_transition_matrix(
        self,
        active_nodes: set[str],
    ) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
        """构建局部转移矩阵。"""
        W: dict[str, dict[str, float]] = {}
        raw: dict[str, dict[str, float]] = {}

        for node_id in active_nodes:
            out_edges = self.graph.get_out_edges(node_id)
            weights: dict[str, float] = {}
            for target_id, edge in out_edges:
                if edge.edge_type == EdgeType.NOT:
                    continue
                target_node = self.graph.get_node(target_id)
                if target_node is None:
                    continue
                redlist_entry = self.graph.get_redlist(node_id, target_id)
                redlist_override = None
                l3_eff = target_node.l3
                if redlist_entry is not None and self.graph.is_redlist_active(node_id, target_id):
                    redlist_override = redlist_entry.new_l2
                    l3_eff = target_node.l3 * redlist_entry.penalty_factor
                has_refinement = self.graph.has_refinement_edge(node_id, target_id)
                w = self.compute_effective_weight(edge, l3_eff, has_refinement=has_refinement, redlist_override=redlist_override)
                if w > 0:
                    weights[target_id] = w

            if weights:
                total_w = sum(weights.values())
                W[node_id] = {t: w / total_w for t, w in weights.items()}
                raw[node_id] = weights
            else:
                W[node_id] = {}
                raw[node_id] = {}

        return W, raw

    def _build_sparse_W(self, active_nodes: list[str], W_dict: dict[str, dict[str, float]]) -> Optional[sp.csr_matrix]:
        """构建稀疏转移矩阵。"""
        if not HAVE_SCIPY or not active_nodes:
            return None
        n = len(active_nodes)
        idx_map = {nid: i for i, nid in enumerate(active_nodes)}
        rows, cols, data = [], [], []
        for nid in active_nodes:
            i = idx_map[nid]
            out = W_dict.get(nid, {})
            for target, prob in out.items():
                j = idx_map.get(target)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
                    data.append(prob)
        return sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32) if data else None

    def compute_path_coherence(self, path: list[str]) -> float:
        k = len(path) - 1
        if k <= 0:
            return 1.0
        cumulative = 1.0
        total = 0.0
        for i in range(1, len(path)):
            prev_node = self.graph.get_node(path[i - 1])
            curr_node = self.graph.get_node(path[i])
            if prev_node is None or curr_node is None:
                cumulative *= 1.0
            else:
                e_prev = self.graph.compute_effective_vector(prev_node)
                e_curr = self.graph.compute_effective_vector(curr_node)
                x = 1.0 - self.graph.cosine_similarity(e_prev, e_curr)
                cumulative *= x
            total += cumulative
        return total / k

    def compute_gamma(self) -> float:
        avg_deg = self.graph.get_avg_out_degree()
        return max(self.config.gamma_dynamic_base, avg_deg * self.config.gamma_dynamic_mult)

    def expand_decision(self, path: list[str], candidate_id: str, w_k_k1: float) -> tuple[bool, float]:
        cfg = self.config
        current_id = path[-1]
        seed_id = path[0]
        current_node = self.graph.get_node(current_id)
        candidate_node = self.graph.get_node(candidate_id)
        seed_node = self.graph.get_node(seed_id)
        if current_node is None or candidate_node is None or seed_node is None:
            return False, 0.0
        if w_k_k1 < cfg.min_edge_weight:
            return False, 0.0
        e_curr = self.graph.compute_effective_vector(current_node)
        e_cand = self.graph.compute_effective_vector(candidate_node)
        if 1.0 - self.graph.cosine_similarity(e_curr, e_cand) > cfg.max_semantic_drift:
            return False, 0.0
        if len(path) > cfg.max_path_depth:
            return False, 0.0
        A = max(0.0, w_k_k1)
        S = self.compute_path_coherence(path + [candidate_id])
        e_seed = self.graph.compute_effective_vector(seed_node)
        N_raw = 1.0 - self.graph.cosine_similarity(e_seed, e_cand)
        alpha, beta = cfg.novelty_lower, cfg.novelty_upper
        N_star = 0.0 if N_raw < alpha else (0.0 if N_raw > beta else (N_raw - alpha) / (beta - alpha))
        out_deg = self.graph.out_degree(candidate_id)
        out_edges = self.graph.get_out_edges(candidate_id)
        avg_w = sum(e.l2 for _, e in out_edges) / len(out_edges) if out_edges else 0.0
        D_star = min(1.0, (out_deg * avg_w) / self.compute_gamma())
        psi = cfg.weight_activation * A + cfg.weight_coherence * S + cfg.weight_novelty * N_star + cfg.weight_diffusion * D_star
        return psi >= cfg.decision_threshold, psi

    def random_walk_with_restart(
        self,
        seed_energy: dict[str, float],
    ) -> tuple[dict[str, float], int]:
        """§2.3 RWR —— 可选稀疏矩阵加速。"""
        cfg = self.config
        s = dict(seed_energy)
        s0 = dict(seed_energy)
        active = set(s.keys())
        paths: dict[str, list[str]] = {}

        for step in range(cfg.max_steps):
            W_dict, raw_weights = self.build_transition_matrix(active)
            if not W_dict:
                break
            total_energy = sum(s.values())
            if total_energy < cfg.energy_threshold:
                break

            # 扩展决策
            new_candidates: dict[str, float] = {}
            for node_id, energy in s.items():
                if energy <= 0 or node_id not in raw_weights:
                    continue
                for target_id, w in raw_weights[node_id].items():
                    cp = paths.get(node_id, [node_id])
                    ok, psi = self.expand_decision(cp, target_id, w)
                    if ok:
                        prob = W_dict[node_id].get(target_id, 0.0)
                        ce = energy * prob * psi
                        new_candidates[target_id] = max(new_candidates.get(target_id, 0.0), ce)
                        if target_id not in paths:
                            paths[target_id] = cp + [target_id]

            for nid, e in new_candidates.items():
                s[nid] = max(s.get(nid, 0.0), e)

            # ── RWR 步 ────────────────────────────────────────────
            active_list = list(active)
            if HAVE_SCIPY and len(active_list) >= 10:
                # 稀疏矩阵路径
                W_sp = self._build_sparse_W(active_list, W_dict)
                if W_sp is not None:
                    idx_map = {nid: i for i, nid in enumerate(active_list)}
                    n = len(active_list)
                    s_vec = np.array([s.get(nid, 0.0) for nid in active_list], dtype=np.float32)
                    transfer = s_vec @ W_sp
                    s_vec_next = (1 - cfg.restart_prob) * transfer + cfg.restart_prob * s_vec
                    s_next = {}
                    for i, nid in enumerate(active_list):
                        v = float(s_vec_next[i])
                        if v > 0:
                            s_next[nid] = v
                    # 吸收节点的处理：能量按 s0 分布回到种子
                    for nid in active_list:
                        if nid not in W_dict or not W_dict[nid]:
                            e = float(s_vec[idx_map[nid]])
                            if e > 0:
                                for seed_id, seed_e in s0.items():
                                    s_next[seed_id] = s_next.get(seed_id, 0.0) + e * cfg.restart_prob * seed_e
                else:
                    s_next = {}
                    for nid in active_list:
                        e = s.get(nid, 0.0)
                        if e <= 0:
                            continue
                        out_dist = W_dict.get(nid, {})
                        if not out_dist:
                            for seed_id, seed_e in s0.items():
                                s_next[seed_id] = s_next.get(seed_id, 0.0) + e * seed_e
                        else:
                            for target, prob in out_dist.items():
                                s_next[target] = s_next.get(target, 0.0) + e * prob
                    transfer_total = sum(s_next.values())
                    if transfer_total > 0:
                        for seed_id, seed_e in s0.items():
                            s_next[seed_id] = s_next.get(seed_id, 0.0) + cfg.restart_prob * seed_e * total_energy
                        scale = (1 - cfg.restart_prob) * total_energy / transfer_total
                        for k in s_next:
                            s_next[k] *= scale
            else:
                # dict 路径（原实现）
                s_next: dict[str, float] = {}
                for node_id in active:
                    if node_id not in W_dict or not W_dict[node_id]:
                        for seed_id, seed_e in s0.items():
                            s_next[seed_id] = s_next.get(seed_id, 0.0) + s[node_id] * seed_e
                        continue
                    for target_id, prob in W_dict[node_id].items():
                        s_next[target_id] = s_next.get(target_id, 0.0) + s[node_id] * prob
                transfer_total = sum(s_next.values())
                if transfer_total > 0:
                    for seed_id, seed_e in s0.items():
                        s_next[seed_id] = s_next.get(seed_id, 0.0) + cfg.restart_prob * seed_e * total_energy
                    scale = (1 - cfg.restart_prob) * total_energy / transfer_total
                    for k in s_next:
                        s_next[k] *= scale

            new_active = set(s_next.keys())
            for nid in new_candidates:
                new_active.add(nid)
            all_keys = set(s.keys()) | set(s_next.keys())
            diff = sum(abs(s.get(k, 0.0) - s_next.get(k, 0.0)) for k in all_keys)
            s = s_next
            active = new_active
            if diff < cfg.convergence_threshold:
                break

        return s, step + 1

    # ═══════════════════════════════════════════════════════════
    #  §2.4  输出格式
    # ═══════════════════════════════════════════════════════════

    def format_output(
        self,
        activation: dict[str, float],
        query: str,
        seed_nodes: list[str],
        num_steps: int,
        is_cold_start: bool = False,
        cold_start_node_id: Optional[str] = None,
    ) -> RetrievalResult:
        cfg = self.config
        items = [(nid, act) for nid, act in activation.items() if act > cfg.output_threshold]
        items.sort(key=lambda x: x[1], reverse=True)

        if is_cold_start and cold_start_node_id is not None:
            return RetrievalResult(
                activated_nodes=[(cold_start_node_id, 1.0)], tokens_used=0,
                cold_start=True, cold_start_node_id=cold_start_node_id,
                query=query, seed_nodes=seed_nodes, num_steps=num_steps,
            )

        tokens_used = 0
        truncated: list[tuple[str, float]] = []
        for nid, act in items:
            estimated_tokens = len(self.graph.get_node(nid).name if self.graph.get_node(nid) else nid) + 5
            if tokens_used + estimated_tokens > cfg.memory_context_tokens:
                logger.warning("Memory context token limit exceeded: %d/%d, truncating after %s",
                               tokens_used + estimated_tokens, cfg.memory_context_tokens, nid)
                break
            truncated.append((nid, act))
            tokens_used += estimated_tokens

        return RetrievalResult(
            activated_nodes=truncated, tokens_used=tokens_used,
            cold_start=False, query=query, seed_nodes=seed_nodes, num_steps=num_steps,
        )

    # ═══════════════════════════════════════════════════════════
    #  主入口
    # ═══════════════════════════════════════════════════════════

    def retrieve(self, query: str) -> RetrievalResult:
        logger.info("Retrieval query: %s", query)
        candidates, is_cold_start, cold_node_id = self.recall_candidates(query)

        if is_cold_start and cold_node_id is not None:
            return RetrievalResult(
                activated_nodes=[(cold_node_id, 1.0)], tokens_used=0,
                cold_start=True, cold_start_node_id=cold_node_id,
                query=query, seed_nodes=[cold_node_id], num_steps=0,
            )
        if not candidates:
            return RetrievalResult(query=query, seed_nodes=[], num_steps=0)

        seeds = self.context_disambiguation(candidates, query)
        seed_ids = [sid for sid, _ in seeds]
        if not seeds:
            return RetrievalResult(query=query, seed_nodes=[], num_steps=0)

        logger.info("Seeds after disambiguation: %s", seed_ids)
        seed_energy = self.seed_energy_normalization(seeds)
        activation, steps = self.random_walk_with_restart(seed_energy)
        result = self.format_output(activation, query, seed_ids, steps, is_cold_start=False)
        return result
