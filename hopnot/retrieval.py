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

from .config import HippocampusConfig, get_default_config
from .embedding import BaseEmbedding
from .graph import MemoryGraph
from .types import Edge, EdgeOrigin, EdgeType, Node

logger = logging.getLogger(__name__)


# ── 检索结果数据结构 ──────────────────────────────────────────

@dataclass
class RetrievalResult:
    """一次检索的输出结果。

    Attributes:
        activated_nodes: 按激活值降序排列的 (节点ID, 激活值) 列表。
        tokens_used: 用于记忆注入的 Token 数。
        cold_start: 是否为冷启动。
        cold_start_node_id: 冷启动节点 ID（如有）。
        query: 原始查询。
        seed_nodes: 初始种子节点 ID 列表。
        num_steps: 随机游走迭代步数。
    """
    activated_nodes: list[tuple[str, float]] = field(default_factory=list)
    tokens_used: int = 0
    cold_start: bool = False
    cold_start_node_id: Optional[str] = None
    query: str = ""
    seed_nodes: list[str] = field(default_factory=list)
    num_steps: int = 0


# ── 检索实现 ──────────────────────────────────────────────────

class MemoryRetrieval:
    """记忆检索器。

    负责种子选取、随机游走扩散和结果组装。

    Usage:
        retrieval = MemoryRetrieval(graph, embedding, config)
        result = retrieval.retrieve("用户查询")
    """

    def __init__(
        self,
        graph: MemoryGraph,
        embedding: BaseEmbedding,
        config: Optional[HippocampusConfig] = None,
    ) -> None:
        self.graph = graph
        self.embed = embedding
        self.config = config or get_default_config()

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
        """计算有效边权重 w_ij（§2.1）。

        计算顺序编码规范：
        1. 若 type = NOT，直接返回 0.0
        2. 检查红名单覆盖（最高优先级）
        3. 检查细化边压制标记
        4. 计算基础权重（L2 × 热度项 × 压制系数）
        5. 叠加上下文偏置 L1
        6. 叠加探索噪声
        7. 截断至有效区间 [-0.2, 1.0]
        """
        # Step 1: NOT 短路
        if edge.edge_type == EdgeType.NOT:
            return 0.0

        # Step 2: 红名单覆盖
        if redlist_override is not None:
            return max(-0.2, min(1.0, redlist_override))

        # Step 3: 细化边压制标记
        refine_penalty = self.config.decay_general if has_refinement and edge.refines is None else 1.0

        # Step 4: 基础权重
        # TypeWeight: 1.0 for ASSOC (NOT already handled)
        type_weight = 1.0
        base = type_weight * edge.l2 * (0.6 + 0.4 * l3_eff) * refine_penalty

        # Step 5: 上下⽂偏置 L1（当前简化为 0，因为 L1 不存储，每次检索时计算）
        # 这里 L1 为会话上下文临时偏置，暂未实现完整 L1 计算
        l1_bias = 0.0  # 可为未来扩展预留

        w = base + l1_bias * 0.3

        # Step 6: 探索噪声 ε_ij ~ N(0, 0.05)
        noise = random.gauss(0.0, 0.05)
        w += noise

        # Step 7: 截断
        return max(-0.2, min(1.0, w))

    # ═══════════════════════════════════════════════════════════
    #  §2.2  种子节点选取
    # ═══════════════════════════════════════════════════════════

    def _exact_match_candidates(self, query: str) -> list[str]:
        """精确匹配候选：通过别名或 name 精确命中。"""
        candidates: list[str] = []
        # 通过别名解析
        node_id = self.graph.resolve_alias(query)
        if node_id is not None:
            candidates.append(node_id)

        # 通过 name 精确匹配
        node = self.graph.find_node_by_name(query)
        if node is not None and node.id not in candidates:
            candidates.append(node.id)

        # alias_of 扩展匹配：对于别名索引中每个条目，检查查询是否是其别名
        for alias, nid in self.graph._alias_index.items():
            if query.lower() in alias.lower() and nid not in candidates:
                candidates.append(nid)

        return candidates

    def recall_candidates(self, query: str) -> tuple[list[tuple[str, float]], bool, Optional[str]]:
        """§2.2.1 候选召回（粗筛）+ 冷启动门控。

        Returns:
            (candidates_with_scores, is_cold_start, cold_start_node_id)
            其中 candidates_with_scores: list[(node_id, similarity)]
        """
        query_embed = self.embed.embed(query)

        # 闲聊门控
        if self.graph.is_chitchat(query):
            logger.info("Chitchat gate triggered for query: %s", query)
            return [], False, None

        # 全库相似度计算
        candidates: list[tuple[str, float]] = []
        for node in self.graph.get_all_nodes():
            eff_vec = self.graph.compute_effective_vector(node)
            sim = self.graph.cosine_similarity(query_embed, eff_vec)
            if sim >= self.config.recall_threshold:
                candidates.append((node.id, sim))

        # 精确实例匹配
        exact_ids = self._exact_match_candidates(query)
        for nid in exact_ids:
            if not any(c[0] == nid for c in candidates):
                sim = self.graph.cosine_similarity(
                    query_embed,
                    self.graph.compute_effective_vector(self.graph.get_node(nid)),
                )
                candidates.append((nid, max(sim, 0.5)))  # 给精确匹配一个基础分

        # 冷启动门控：若候选池为空
        if not candidates:
            logger.info("Cold start triggered for query: %s", query)
            # 创建冷启动节点
            cold_node = Node(
                id=f"cold_{int(time.time() * 1000)}_{hash(query) % 10000}",
                name=query,
                b=query_embed,
                l3=self.config.l3_cold_start,
                created_at=time.time(),
                last_visited=time.time(),
            )
            self.graph.add_node(cold_node)
            # 冷启动节点激活值 = 1.0
            candidates = [(cold_node.id, 1.0)]
            return candidates, True, cold_node.id

        return candidates, False, None

    def context_disambiguation(self, candidates: list[tuple[str, float]], query: str) -> list[tuple[str, float]]:
        """§2.2.2 上下文消歧（精筛）—— 邻居投票。

        对每个候选 v_i:
          ContextScore = s_i + λ * (1/|N^+(v_i)|) * Σ <e_q, e_{v_j}>
                        + δ * 1_{recent}(v_i)
        """
        query_embed = self.embed.embed(query)
        cfg = self.config
        scored: list[tuple[str, float]] = []

        for node_id, sim in candidates:
            node = self.graph.get_node(node_id)
            if node is None:
                continue

            # 邻居平均匹配度
            out_edges = self.graph.get_out_edges(node_id)
            neighbor_count = len(out_edges)

            # 编码约束：出度 > 200 时按 L2 截断
            if neighbor_count > cfg.neighbor_cutoff:
                out_edges.sort(key=lambda x: x[1].l2, reverse=True)
                out_edges = out_edges[:cfg.neighbor_cutoff]

            if neighbor_count > 0:
                neighbor_sim_sum = 0.0
                for target_id, edge in out_edges:
                    target_node = self.graph.get_node(target_id)
                    if target_node is not None:
                        eff_t = self.graph.compute_effective_vector(target_node)
                        neighbor_sim_sum += self.graph.cosine_similarity(query_embed, eff_t)
                avg_neighbor = neighbor_sim_sum / len(out_edges)
            else:
                avg_neighbor = 0.0

            # 近期偏置（前 3 轮被激活 —— 简化实现：最近30秒内被访问的作为近期）
            recency_bonus = cfg.recency_bias if (time.time() - node.last_visited) < 30.0 else 0.0

            score = sim + cfg.neighbor_vote_weight * avg_neighbor + recency_bonus
            scored.append((node_id, score))

        # 按得分降序取 Top-N
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:cfg.num_seeds]

    def seed_energy_normalization(self, seeds: list[tuple[str, float]]) -> dict[str, float]:
        """§2.2.3 种子能量归一化。

        对每个种子 v_i:
          α_i = ContextScore(v_i) / Σ ContextScore(v_j)
        """
        total = sum(score for _, score in seeds) or 1.0
        return {nid: score / total for nid, score in seeds}

    # ═══════════════════════════════════════════════════════════
    #  §2.3  记忆扩散（RWR）
    # ═══════════════════════════════════════════════════════════

    def build_transition_matrix(
        self,
        active_nodes: set[str],
    ) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
        """构建局部转移矩阵（行归一化，自动排除 NOT 边）。

        Returns:
            (W: 转移概率矩阵, raw_weights: 原始权重矩阵)
        """
        W: dict[str, dict[str, float]] = {}
        raw: dict[str, dict[str, float]] = {}

        for node_id in active_nodes:
            out_edges = self.graph.get_out_edges(node_id)
            weights: dict[str, float] = {}
            for target_id, edge in out_edges:
                if edge.edge_type == EdgeType.NOT:
                    continue

                # 计算有效权重
                target_node = self.graph.get_node(target_id)
                if target_node is None:
                    continue

                # 检查红名单
                redlist_entry = self.graph.get_redlist(node_id, target_id)
                redlist_override = None
                l3_eff = target_node.l3
                if redlist_entry is not None and self.graph.is_redlist_active(node_id, target_id):
                    redlist_override = redlist_entry.new_l2
                    l3_eff = target_node.l3 * redlist_entry.penalty_factor

                # 检查细化边压制：若该节点对之间存在至少一条细化关系边
                # （带 refines 标记），则泛化关系边的检索权重乘以衰减系数。
                has_refinement = self.graph.has_refinement_edge(node_id, target_id)

                w = self.compute_effective_weight(
                    edge, l3_eff,
                    has_refinement=has_refinement,
                    redlist_override=redlist_override,
                )
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

    def compute_path_coherence(self, path: list[str]) -> float:
        """§2.3.3 路径语义连贯性 S(P_k)。

        S(P_k) = (1/k) * Σ_{i=1}^{k} Π_{j=1}^{i} x_j
        其中 x_j = 1 - <e_{v_{j-1}}, e_{v_j}>
        """
        k = len(path) - 1  # 边数
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
                sim = self.graph.cosine_similarity(e_prev, e_curr)
                x = 1.0 - sim
                cumulative *= x
            total += cumulative

        return total / k

    def compute_gamma(self) -> float:
        """§2.3.4 动态 γ 计算。"""
        avg_deg = self.graph.get_avg_out_degree()
        return max(self.config.gamma_dynamic_base, avg_deg * self.config.gamma_dynamic_mult)

    def expand_decision(self, path: list[str], candidate_id: str, w_k_k1: float) -> tuple[bool, float]:
        """§2.3.4 节点扩展决策（综合值得度 Ψ）。

        Returns:
            (是否扩展, Ψ 值)
        """
        cfg = self.config
        current_id = path[-1]
        seed_id = path[0]

        current_node = self.graph.get_node(current_id)
        candidate_node = self.graph.get_node(candidate_id)
        seed_node = self.graph.get_node(seed_id)

        if current_node is None or candidate_node is None or seed_node is None:
            return False, 0.0

        # 硬性拒绝检查
        # 1. 边太弱
        if w_k_k1 < cfg.min_edge_weight:
            return False, 0.0

        # 2. 单步语义跑偏
        e_curr = self.graph.compute_effective_vector(current_node)
        e_cand = self.graph.compute_effective_vector(candidate_node)
        step_drift = 1.0 - self.graph.cosine_similarity(e_curr, e_cand)
        if step_drift > cfg.max_semantic_drift:
            return False, 0.0

        # 3. 深度超标
        if len(path) > cfg.max_path_depth:
            return False, 0.0

        # A. 激活潜力
        A = max(0.0, w_k_k1)

        # B. 语义连贯性
        test_path = path + [candidate_id]
        S = self.compute_path_coherence(test_path)

        # C. 新颖性（带饱和门控）
        e_seed = self.graph.compute_effective_vector(seed_node)
        N_raw = 1.0 - self.graph.cosine_similarity(e_seed, e_cand)
        alpha = cfg.novelty_lower
        beta = cfg.novelty_upper
        if N_raw < alpha:
            N_star = 0.0
        elif N_raw > beta:
            N_star = 0.0
        else:
            N_star = (N_raw - alpha) / (beta - alpha)

        # D. 扩散潜力
        out_deg = self.graph.out_degree(candidate_id)
        out_edges = self.graph.get_out_edges(candidate_id)
        avg_w = sum(e.l2 for _, e in out_edges) / len(out_edges) if out_edges else 0.0
        gamma = self.compute_gamma()
        D_star = min(1.0, (out_deg * avg_w) / gamma)

        # 综合得分
        psi = (
            cfg.weight_activation * A
            + cfg.weight_coherence * S
            + cfg.weight_novelty * N_star
            + cfg.weight_diffusion * D_star
        )

        return psi >= cfg.decision_threshold, psi

    def random_walk_with_restart(
        self,
        seed_energy: dict[str, float],
    ) -> tuple[dict[str, float], int]:
        """§2.3 记忆扩散（Random Walk with Restart）。

        Args:
            seed_energy: 种子能量分布 s_0 (node_id -> α_i)

        Returns:
            (s_T: 最终激活分布, steps: 迭代步数)
        """
        cfg = self.config

        # 初始状态向量
        s = dict(seed_energy)
        s0 = dict(seed_energy)

        # 活跃节点集合（所有有能量的节点）
        active = set(s.keys())

        # 路径跟踪（为语义连贯性保留）
        # 对每个活跃节点，记录从其种子出发的最优路径
        paths: dict[str, list[str]] = {}

        for step in range(cfg.max_steps):
            # 构建转移矩阵（基于当前活跃集）
            W, raw_weights = self.build_transition_matrix(active)
            if not W:
                break

            # 检查能量耗尽
            total_energy = sum(s.values())
            if total_energy < cfg.energy_threshold:
                break

            # 扩展决策：对所有当前节点的出边候选进行评估
            new_candidates: dict[str, float] = {}
            for node_id, energy in s.items():
                if energy <= 0:
                    continue
                if node_id not in raw_weights:
                    continue
                for target_id, w in raw_weights[node_id].items():
                    current_path = paths.get(node_id, [node_id])
                    should_expand, psi = self.expand_decision(
                        current_path, target_id, w
                    )
                    if should_expand:
                        # 候选能量 = 源能量 × 转移概率 × Ψ
                        trans_prob = W.get(node_id, {}).get(target_id, 0.0)
                        cand_energy = energy * trans_prob * psi
                        if target_id in new_candidates:
                            new_candidates[target_id] = max(new_candidates[target_id], cand_energy)
                        else:
                            new_candidates[target_id] = cand_energy
                        # 记录最优路径
                        if target_id not in paths:
                            paths[target_id] = current_path + [target_id]

            # 合并新候选到 s
            for nid, e in new_candidates.items():
                if nid in s:
                    s[nid] = max(s[nid], e)
                else:
                    s[nid] = e

            # ── RWR 迭代 ────────────────────────────────────────
            # 标准公式: s_{t+1} = (1-ρ) · s_t × W + ρ · s_0
            # 1. 先计算转移部分: s_t × W (所有能量通过行归一化转移矩阵传播)
            # 2. 再以 ρ 概率回到种子分布
            s_next: dict[str, float] = {}
            total_energy = sum(s.values())

            for node_id in active:
                if node_id not in W or not W[node_id]:
                    # 吸收节点：所有能量回到种子（无出边可转移）
                    for seed_id, seed_e in s0.items():
                        s_next[seed_id] = s_next.get(seed_id, 0.0) + s[node_id] * seed_e
                    continue

                out_dist = W[node_id]
                for target_id, prob in out_dist.items():
                    s_next[target_id] = s_next.get(target_id, 0.0) + s[node_id] * prob

            # 应用重启: s_{t+1} = (1-ρ) · S_transfer + ρ · s_0
            transfer_total = sum(s_next.values())
            if transfer_total > 0:
                for seed_id, seed_e in s0.items():
                    restart_amount = cfg.restart_prob * seed_e * total_energy
                    s_next[seed_id] = s_next.get(seed_id, 0.0) + restart_amount
                # 缩放转移部分使总能量守恒
                scale = (1 - cfg.restart_prob) * total_energy / transfer_total if transfer_total > 0 else 0
                for k in s_next:
                    s_next[k] = s_next[k] * scale if scale > 0 else s_next[k]

            # 更新活跃集
            new_active = set(s_next.keys())
            for nid in new_candidates:
                new_active.add(nid)

            # 检查收敛
            all_keys = set(s.keys()) | set(s_next.keys())
            diff = sum(abs(s.get(k, 0.0) - s_next.get(k, 0.0)) for k in all_keys)
            s = s_next
            active = new_active

            if diff < cfg.convergence_threshold:
                break

        return s, step + 1

    # ═══════════════════════════════════════════════════════════
    #  §2.4  输出格式（Token 截断 + 冷启动豁免）
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
        """§2.4 组装输出结果。

        按激活值降序排列，应用 Token 截断池拆分。
        冷启动节点豁免排序与截断。
        """
        cfg = self.config

        # 过滤低激活值
        items = [(nid, act) for nid, act in activation.items() if act > cfg.output_threshold]
        items.sort(key=lambda x: x[1], reverse=True)

        # 冷启动豁免
        if is_cold_start and cold_start_node_id is not None:
            return RetrievalResult(
                activated_nodes=[(cold_start_node_id, 1.0)],
                tokens_used=0,  # 豁免不计
                cold_start=True,
                cold_start_node_id=cold_start_node_id,
                query=query,
                seed_nodes=seed_nodes,
                num_steps=num_steps,
            )

        # Token 截断池拆分
        tokens_used = 0
        truncated: list[tuple[str, float]] = []
        for nid, act in items:
            # 模拟 Token 数估算：每节点约 10-20 tokens（节点名 + 激活值 + 格式）
            estimated_tokens = len(self.graph.get_node(nid).name if self.graph.get_node(nid) else nid) + 5
            if tokens_used + estimated_tokens > cfg.memory_context_tokens:
                logger.warning(
                    "Memory context token limit exceeded: %d/%d, truncating after %s",
                    tokens_used + estimated_tokens,
                    cfg.memory_context_tokens,
                    nid,
                )
                break
            truncated.append((nid, act))
            tokens_used += estimated_tokens

        return RetrievalResult(
            activated_nodes=truncated,
            tokens_used=tokens_used,
            cold_start=False,
            query=query,
            seed_nodes=seed_nodes,
            num_steps=num_steps,
        )

    # ═══════════════════════════════════════════════════════════
    #  主入口
    # ═══════════════════════════════════════════════════════════

    def retrieve(self, query: str) -> RetrievalResult:
        """执行一次完整的记忆检索。

        Args:
            query: 用户查询文本。

        Returns:
            RetrievalResult 包含激活节点列表和元信息。
        """
        logger.info("Retrieval query: %s", query)

        # §2.2.1 候选召回 + 冷启动门控
        candidates, is_cold_start, cold_node_id = self.recall_candidates(query)

        # 冷启动：跳过随机游走，直接返回
        if is_cold_start and cold_node_id is not None:
            return RetrievalResult(
                activated_nodes=[(cold_node_id, 1.0)],
                tokens_used=0,
                cold_start=True,
                cold_start_node_id=cold_node_id,
                query=query,
                seed_nodes=[cold_node_id],
                num_steps=0,
            )

        if not candidates:
            return RetrievalResult(
                query=query,
                seed_nodes=[],
                num_steps=0,
            )

        # §2.2.2 上下文消歧
        seeds = self.context_disambiguation(candidates, query)
        seed_ids = [sid for sid, _ in seeds]

        if not seeds:
            return RetrievalResult(query=query, seed_nodes=[], num_steps=0)

        logger.info("Seeds after disambiguation: %s", seed_ids)

        # §2.2.3 种子能量归一化
        seed_energy = self.seed_energy_normalization(seeds)

        # §2.3 记忆扩散
        activation, steps = self.random_walk_with_restart(seed_energy)

        # §2.4 输出格式化
        result = self.format_output(
            activation, query, seed_ids, steps,
            is_cold_start=False,
        )

        return result
