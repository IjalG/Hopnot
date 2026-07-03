"""整理阶段 —— 写入/巩固记忆。

实现规范（v1.7）：
1. 原子化拆解（LLM 输出 → 三元组集合）
2. 节点定位/创建（短文本软关联 + UNK阻尼合并）
3. 边处理（核心决策树：4 分支）
4. 时间衰减（混合衰减）
5. 节点偏置漂移（漂移触发门控 + 方向分治）
6. 增量式三角闭合（防爆炸）
7. L3热度更新（分位数归一化 + 否定边惩罚）
8. 日志归档
9. 人工权威干预接口（惰性回滚）
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from .config import HippocampusConfig, get_default_config
from .embedding import BaseEmbedding
from .graph import MemoryGraph
from .types import (
    Edge,
    EdgeOrigin,
    EdgeType,
    MergedLogEntry,
    Node,
    RedlistEntry,
)

logger = logging.getLogger(__name__)


class MemoryConsolidation:
    """记忆整理器。

    负责将 LLM 输出拆解为三元组，写入图库并更新权重。
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
    #  §3.1  原子化拆解
    # ═══════════════════════════════════════════════════════════

    def decompose_triples(self, text: str) -> list[tuple[str, str, str]]:
        """将 LLM 输出文本拆解为 (subject, relation, object) 三元组。

        注意：在生产环境中，这里应调用冻结的千问 0.5B 进行三元组提取。
        当前实现使用一种基于规则的简单提取器作为占位。
        """
        triples: list[tuple[str, str, str]] = []
        lines = text.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 尝试解析 `(主体, 关系, 客体)` 格式
            if line.startswith("(") and line.endswith(")"):
                inner = line[1:-1]
                parts = [p.strip() for p in inner.split(",")]
                if len(parts) >= 3:
                    triples.append((parts[0], parts[1], parts[2]))
        return triples

    def set_triple_decomposer(self, decomposer) -> None:
        """设置自定义三元组拆解器。"""
        self.decompose_triples = decomposer

    # ═══════════════════════════════════════════════════════════
    #  §3.2  节点定位/创建
    # ═══════════════════════════════════════════════════════════

    def _is_short_text(self, text: str) -> bool:
        """判断是否为短文本（≤2 汉字或 ≤3 英文字符）。"""
        # 统计中文字符
        cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        if cjk_count >= 2:
            return len(text) <= 2
        return len(text.strip()) <= 3

    def locate_or_create_node(self, text: str) -> Node:
        """§3.2 节点定位/创建（短文本软关联 + UNK阻尼合并）。

        Returns:
            匹配到的或新建的 Node。
        """
        cfg = self.config
        text = text.strip()
        if not text:
            raise ValueError("Empty text for node localization")

        # Step 1: 前置精确匹配
        exact_id = self.graph.resolve_alias(text)
        if exact_id is not None:
            node = self.graph.get_node(exact_id)
            if node is not None:
                return node

        node_by_name = self.graph.find_node_by_name(text)
        if node_by_name is not None:
            return node_by_name

        text_embed = self.embed.embed(text)
        all_nodes = self.graph.get_all_nodes()

        # Step 2: 短文本软关联
        if self._is_short_text(text):
            if not all_nodes:
                # 无现有节点，允许新建 UNK_
                return self._create_unk_node(text, text_embed)

            # 计算与所有节点的相似度，取 Top 1
            best_sim = -1.0
            best_node = None
            for node in all_nodes:
                eff_v = self.graph.compute_effective_vector(node)
                sim = self.graph.cosine_similarity(text_embed, eff_v)
                if sim > best_sim:
                    best_sim = sim
                    best_node = node

            if best_sim > cfg.short_alias_threshold and best_node is not None:
                # 映射到最相似节点，记录 alias_of
                self.graph.add_alias(best_node.id, text)
                logger.info("Short text soft-association: '%s' -> %s (sim=%.3f)",
                            text, best_node.name, best_sim)
                return best_node
            else:
                # 允许新建 UNK_
                return self._create_unk_node(text, text_embed)

        # Step 3: 向量匹配（长度达标时）
        best_sim = -1.0
        best_node = None
        for node in all_nodes:
            sim = self.graph.cosine_similarity(text_embed, node.b)  # 用基座向量匹配
            if sim > best_sim:
                best_sim = sim
                best_node = node

        if best_sim > cfg.merge_threshold and best_node is not None:
            logger.info("Vector match: '%s' -> %s (sim=%.3f)", text, best_node.name, best_sim)
            return best_node

        # 不匹配：新建节点
        node = Node(
            id=f"n_{int(time.time() * 1000)}_{hash(text) % 100000}",
            name=text,
            b=text_embed,
            l3=cfg.l3_initial,
            freq=0,
            created_at=time.time(),
            last_visited=time.time(),
        )
        self.graph.add_node(node)
        logger.info("New node created: %s (name='%s')", node.id, text)
        return node

    def _create_unk_node(self, text: str, embed_vec: list[float]) -> Node:
        """创建 UNK_ 前缀节点。"""
        cfg = self.config
        node = Node(
            id=f"unk_{int(time.time() * 1000)}_{hash(text) % 100000}",
            name=f"UNK_{text}",
            b=embed_vec,
            l3=cfg.l3_initial,
            freq=0,
            created_at=time.time(),
            last_visited=time.time(),
        )
        self.graph.add_node(node)
        logger.info("UNK node created: %s (name='%s')", node.id, node.name)
        return node

    def merge_unk_node(self, unk_node_id: str, target_node_id: str) -> None:
        """UNK_ 节点合并规则（§3.2 底部）。

        1. 偏置向量阻尼合并
        2. 重定向所有边
        3. 删除 UNK_ 节点
        4. 合并 alias_of 元数据
        """
        unk_node = self.graph.get_node(unk_node_id)
        target_node = self.graph.get_node(target_node_id)
        if unk_node is None or target_node is None:
            logger.warning("Cannot merge UNK node: one or both missing (%s, %s)",
                           unk_node_id, target_node_id)
            return

        if not unk_node.name.startswith("UNK_"):
            logger.warning("Node %s is not an UNK_ node", unk_node_id)
            return

        cfg = self.config

        # 1. 偏置向量阻尼合并
        freq_v = target_node.freq
        freq_unk = unk_node.freq
        unk_weight = min(1.0, freq_unk / 10.0 + 0.2)  # 饱和增长函数

        denominator = freq_v + unk_weight
        if denominator > 0:
            new_p = [
                (pv * freq_v + pu * unk_weight) / denominator
                for pv, pu in zip(target_node.p, unk_node.p)
            ]
            target_node.p = self.graph.clamp_vector_norm(new_p, cfg.p_norm_max)

        # 2. 重定向所有边 (入边和出边)
        # 出边
        for target_id, edge in self.graph.get_out_edges(unk_node_id):
            new_edge = Edge(
                source=target_node_id,
                target=edge.target,
                l2=edge.l2,
                confidence=edge.confidence,
                edge_type=edge.edge_type,
                origin=edge.origin,
                refines=edge.refines,
                recovery_count=edge.recovery_count,
                version=1,
                created_at=edge.created_at,
                last_visited=edge.last_visited,
                access_count=edge.access_count,
            )
            # 若目标节点间已有边，则取权重更高者
            existing = self.graph.get_edge(target_node_id, edge.target)
            if existing is None or existing.l2 < edge.l2:
                self.graph.add_edge(new_edge)

        # 入边
        for source_id, edge in self.graph.get_in_edges(unk_node_id):
            new_edge = Edge(
                source=edge.source,
                target=target_node_id,
                l2=edge.l2,
                confidence=edge.confidence,
                edge_type=edge.edge_type,
                origin=edge.origin,
                refines=edge.refines,
                recovery_count=edge.recovery_count,
                version=1,
                created_at=edge.created_at,
                last_visited=edge.last_visited,
                access_count=edge.access_count,
            )
            existing = self.graph.get_edge(edge.source, target_node_id)
            if existing is None or existing.l2 < edge.l2:
                self.graph.add_edge(new_edge)

        # 3. 删除 UNK_ 节点
        self.graph.remove_node(unk_node_id)

        # 4. 合并 alias_of 元数据
        for alias in unk_node.aliases:
            self.graph.add_alias(target_node_id, alias)
        self.graph.add_alias(target_node_id, unk_node.name.replace("UNK_", ""))

        self.graph.log_merge(unk_node_id, target_node_id)
        logger.info("Merged UNK node %s into %s", unk_node_id, target_node_id)

    # ═══════════════════════════════════════════════════════════
    #  §3.3  边处理（核心决策树）
    # ═══════════════════════════════════════════════════════════

    def process_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str = "关联",
    ) -> Edge:
        """§3.3 边处理核心决策树。

        Args:
            source_id: 主体节点 ID。
            target_id: 客体节点 ID。
            relation: 关系名称（用于细化边标记）。

        Returns:
            处理后的 Edge 对象。
        """
        cfg = self.config
        now = time.time()
        existing_edge = self.graph.get_edge(source_id, target_id)

        if existing_edge is None:
            # 分支 1：边不存在（新建）
            edge = Edge(
                source=source_id,
                target=target_id,
                l2=cfg.l2_initial + cfg.l2_bonus,  # 0.30 + 0.15
                confidence=cfg.c_initial,
                edge_type=EdgeType.ASSOC,
                origin=EdgeOrigin.EXPLICIT_FACT,
                created_at=now,
                last_visited=now,
                access_count=1,
                version=1,
            )
            self.graph.add_edge(edge)
            logger.debug("New edge: %s -> %s (l2=%.3f)", source_id, target_id, edge.l2)
            self.graph.log_operation("edge_create", {
                "source": source_id, "target": target_id, "l2": edge.l2,
            })
            return edge

        # 检查关系是否一致
        is_same_type = (existing_edge.edge_type == EdgeType.ASSOC)
        is_refinement = (relation != "关联" and existing_edge.refines is None)

        if is_same_type and not is_refinement:
            # 分支 2：边存在且关系一致
            if existing_edge.confidence >= cfg.c_conflict_threshold:
                # 正常状态：普通强化
                existing_edge.confidence = min(1.0, existing_edge.confidence + 0.05)
                existing_edge.l2 = min(1.0, existing_edge.l2 + cfg.l2_hebb_increment)
            else:
                # 存疑/濒死状态：恢复红利
                existing_edge.confidence = min(
                    cfg.confidence_max_recovery,
                    existing_edge.confidence + cfg.confidence_recovery,
                )
                existing_edge.l2 = max(
                    cfg.l2_prune_threshold + 0.02,
                    existing_edge.l2 + cfg.l2_recovery_bonus,
                )
                existing_edge.recovery_count += 1
                if existing_edge.recovery_count >= cfg.recovery_alert_threshold:
                    logger.warning(
                        "Edge recovery alert: %s -> %s (count=%d)",
                        source_id, target_id, existing_edge.recovery_count,
                    )

            existing_edge.last_visited = now
            existing_edge.access_count += 1
            existing_edge.version += 1
            logger.debug("Strengthened edge: %s -> %s (l2=%.3f, c=%.3f)",
                         source_id, target_id, existing_edge.l2, existing_edge.confidence)
            return existing_edge

        elif not is_same_type:
            # 分支 3：边存在且关系冲突
            existing_edge.confidence = max(0.0, existing_edge.confidence - 0.03)
            existing_edge.version += 1

            if existing_edge.confidence < cfg.c_conflict_threshold:
                logger.info("Edge confidence dropped below threshold: %s -> %s (c=%.3f)",
                            source_id, target_id, existing_edge.confidence)

            # 新建否定边
            not_edge = Edge(
                source=source_id,
                target=target_id,
                l2=cfg.l2_initial,
                confidence=cfg.c_initial,
                edge_type=EdgeType.NOT,
                origin=EdgeOrigin.EXPLICIT_FACT,
                created_at=now,
                last_visited=now,
                access_count=1,
                version=1,
            )
            self.graph.add_edge(not_edge)
            self.graph.log_conflict({
                "source": source_id,
                "target": target_id,
                "existing_type": existing_edge.edge_type.value,
                "new_type": EdgeType.NOT.value,
                "confidence_before": existing_edge.confidence + 0.03,
                "confidence_after": existing_edge.confidence,
            })
            logger.info("Conflict: created NOT edge %s -> %s", source_id, target_id)
            return not_edge

        else:
            # 分支 4：边存在且关系更细化（共存 + 细化标记）
            # 保留旧边的"关联"关系，新建细化边
            refine_edge = Edge(
                source=source_id,
                target=target_id,
                l2=cfg.l2_initial + cfg.l2_bonus,  # 走分支 1 流程
                confidence=cfg.c_initial,
                edge_type=EdgeType.ASSOC,
                origin=EdgeOrigin.EXPLICIT_FACT,
                refines="关联",  # 指明它细化了哪条泛化边
                created_at=now,
                last_visited=now,
                access_count=1,
                version=1,
            )
            self.graph.add_edge(refine_edge)
            logger.info("Refinement edge: %s -> %s (refines='关联')", source_id, target_id)
            return refine_edge

    # ═══════════════════════════════════════════════════════════
    #  §3.4  时间衰减（混合衰减）
    # ═══════════════════════════════════════════════════════════

    def apply_time_decay(self, now: Optional[float] = None) -> int:
        """§3.4 对所有边应用混合时间衰减。

        Returns:
            更新的边数。
        """
        if now is None:
            now = time.time()
        cfg = self.config
        updated = 0

        for edge in list(self.graph._edges.values()):
            delta_days = (now - edge.last_visited) / 86400.0

            # 轮次衰减(每次访问)
            decay_amount = cfg.decay_per_access * max(1, edge.access_count - 1)

            # 长尾冷惩罚（超过 30 天未访问）
            if delta_days > cfg.long_tail_days:
                decay_amount += cfg.decay_long_tail

            new_l2 = max(cfg.l2_min, edge.l2 - decay_amount)
            if new_l2 != edge.l2:
                edge.l2 = new_l2
                edge.version += 1
                updated += 1

        return updated

    # ═══════════════════════════════════════════════════════════
    #  §3.5  节点偏置漂移
    # ═══════════════════════════════════════════════════════════

    def apply_bias_drift(self, edge: Edge) -> None:
        """§3.5 节点偏置漂移（漂移触发门控 + 方向分治）。

        仅在 origin = explicit_fact 时触发。inferred 边绝不触发。
        """
        if edge.origin != EdgeOrigin.EXPLICIT_FACT:
            return

        source_node = self.graph.get_node(edge.source)
        target_node = self.graph.get_node(edge.target)
        if source_node is None or target_node is None:
            return

        cfg = self.config

        # 核心节点保护：L3 > 0.9 时偏置更新率减半
        source_half = cfg.core_bias_half and source_node.l3 > cfg.l3_core_threshold
        target_half = cfg.core_bias_half and target_node.l3 > cfg.l3_core_threshold

        if edge.edge_type == EdgeType.ASSOC:
            # 正向关联：靠拢
            lambda_factor = cfg.lambda_new if edge.access_count <= 1 else cfg.lambda_confirm * edge.confidence

            if source_half:
                lambda_factor *= 0.5

            # p_A += λ · (b_B - b_A)
            diff_ab = [tb - sa for sa, tb in zip(source_node.b, target_node.b)]
            source_node.p = self.graph.vector_add(
                source_node.p, self.graph.vector_scale(diff_ab, lambda_factor)
            )

            # p_B += λ · (b_A - b_B)
            if not target_half:
                diff_ba = [sa - tb for sa, tb in zip(source_node.b, target_node.b)]
                target_node.p = self.graph.vector_add(
                    target_node.p, self.graph.vector_scale(diff_ba, lambda_factor)
                )

        elif edge.edge_type == EdgeType.NOT:
            # 否定关联：推远
            lambda_not = cfg.lambda_not

            # p_A -= λ_not · (b_B - b_A)
            diff_ab = [tb - sa for sa, tb in zip(source_node.b, target_node.b)]
            source_node.p = self.graph.vector_add(
                source_node.p, self.graph.vector_scale(diff_ab, -lambda_not)
            )

            # p_B -= λ_not · (b_A - b_B)
            diff_ba = [sa - tb for sa, tb in zip(source_node.b, target_node.b)]
            target_node.p = self.graph.vector_add(
                target_node.p, self.graph.vector_scale(diff_ba, -lambda_not)
            )

        # 保护机制：裁剪 ||p|| ≤ 0.5
        source_node.p = self.graph.clamp_vector_norm(source_node.p, cfg.p_norm_max)
        target_node.p = self.graph.clamp_vector_norm(target_node.p, cfg.p_norm_max)

    # ═══════════════════════════════════════════════════════════
    #  §3.6  增量式三角闭合（防爆炸）
    # ═══════════════════════════════════════════════════════════

    def triangular_closure(self, source_id: str, target_id: str) -> list[Edge]:
        """§3.6 增量式三角闭合。

        仅在写入新边 A → B 时局部触发：
        1. 检查 B 的出边邻居 Top 50（按 L2 降序）
        2. 若存在 C ∈ N^+(B) 且 L2_{BC} > 0.4：
           - 若 A → C 不存在或 L2_{AC} < 0.1，创建联想边

        Returns:
            新创建的联想边列表。
        """
        cfg = self.config
        new_edges: list[Edge] = []

        # 检查 B 的出边邻居
        out_edges = self.graph.get_out_edges(target_id)
        # 按 L2 降序取 Top 50
        out_edges.sort(key=lambda x: x[1].l2, reverse=True)
        out_edges = out_edges[:cfg.triangle_topk]

        for neighbor_id, b_to_c_edge in out_edges:
            if b_to_c_edge.l2 < cfg.triangle_min_l2:
                continue
            if b_to_c_edge.edge_type == EdgeType.NOT:
                continue

            # 检查 A → C
            existing = self.graph.get_edge(source_id, neighbor_id)
            if existing is not None and existing.l2 >= 0.1:
                continue

            # 获取 A→B 的 L2 值
            ab_edge = self.graph.get_edge(source_id, target_id)
            l2_ab = ab_edge.l2 if ab_edge is not None else 0.3
            # 创建联想边: L2_AC = κ · (L2_AB + L2_BC) / 2
            new_l2 = cfg.triangle_discount * (l2_ab + b_to_c_edge.l2) / 2.0
            inferred_edge = Edge(
                source=source_id,
                target=neighbor_id,
                l2=new_l2,
                confidence=0.3,
                edge_type=EdgeType.ASSOC,
                origin=EdgeOrigin.INFERRED,
                created_at=time.time(),
                last_visited=time.time(),
                access_count=0,
                version=1,
            )
            self.graph.add_edge(inferred_edge)
            new_edges.append(inferred_edge)
            logger.debug("Triangular closure: %s -> %s (l2=%.3f, inferred from %s -> %s)",
                         source_id, neighbor_id, new_l2, target_id, neighbor_id)

        return new_edges

    # ═══════════════════════════════════════════════════════════
    #  §3.7  L3 热度更新
    # ═══════════════════════════════════════════════════════════

    def update_l3(self, node_ids: list[str]) -> None:
        """§3.7 L3 热度更新（分位数归一化 + 否定边惩罚）。

        L3 = 0.7 × L3 + 0.3 × (deg_ratio + freq_ratio) × 0.5
        后处理否定边惩罚。
        """
        cfg = self.config
        deg_p95 = self.graph.get_deg_p95()
        freq_p95 = self.graph.get_freq_p95()

        for nid in node_ids:
            node = self.graph.get_node(nid)
            if node is None:
                continue

            # 增加访问频率
            node.freq += 1

            # 出度比率
            deg = self.graph.out_degree(nid)
            if deg <= deg_p95:
                deg_ratio = deg / deg_p95 if deg_p95 > 0 else 0.0
            else:
                deg_ratio = 1.0

            # 频率比率
            freq = node.freq
            if freq <= freq_p95:
                freq_ratio = freq / freq_p95 if freq_p95 > 0 else 0.0
            else:
                freq_ratio = 1.0

            # 新 L3 计算
            new_l3 = cfg.l3_smoothing * node.l3 + cfg.l3_update_rate * (deg_ratio + freq_ratio) * 0.5

            # 否定边惩罚后处理
            count_not = 0
            for _, edge in self.graph.get_out_edges(nid):
                if edge.edge_type == EdgeType.NOT:
                    count_not += 1
            new_l3 *= (1.0 - cfg.l3_not_penalty * count_not)

            # 强制下限
            node.l3 = max(cfg.l3_min, new_l3)

    # ═══════════════════════════════════════════════════════════
    #  §3.8  日志与归档
    # ═══════════════════════════════════════════════════════════

    def archive_low_weight_edges(self) -> list[Edge]:
        """低权修剪（离线批次）：L2 < 0.08 的边移入冷归档。

        Returns:
            被归档的边列表。
        """
        cfg = self.config
        archived: list[Edge] = []
        edges_to_remove: list[tuple[str, str, str]] = []

        for key, edge in list(self.graph._edges.items()):
            if edge.l2 < cfg.l2_prune_threshold:
                archived.append(edge)
                edges_to_remove.append(key)

        for source, target, tag in edges_to_remove:
            if tag:
                self.graph._remove_edge(source, tag, target_fixed=target)
            else:
                self.graph.remove_edge(source, target)

        if archived:
            logger.info("Archived %d low-weight edges (L2 < %.3f)",
                        len(archived), cfg.l2_prune_threshold)
            self.graph.log_operation("archive", {"count": len(archived)})

        return archived

    # ═══════════════════════════════════════════════════════════
    #  §3.9  人工权威干预接口（联动修复 + 惰性回滚）
    # ═══════════════════════════════════════════════════════════

    def apply_redlist_override(
        self,
        source_id: str,
        target_id: str,
        new_l2: float,
        penalty_factor: float = 0.5,
        expire_at: Optional[float] = None,
        restore_l3_on_expire: bool = True,
        reason: str = "",
    ) -> RedlistEntry:
        """§3.9 添加人工权威干预红名单条目。

        1. 检索时覆盖 L2（§2.1 中由 compute_effective_weight 执行）
        2. 物理 L3 存储不变（由红名单中的 penalty_factor 控制）
        3. 惰性回滚：检索时检查过期
        4. 记录审计日志
        """
        entry = RedlistEntry(
            source=source_id,
            target=target_id,
            new_l2=new_l2,
            penalty_factor=penalty_factor,
            expire_at=expire_at,
            restore_l3_on_expire=restore_l3_on_expire,
            reason=reason,
            created_at=time.time(),
        )
        self.graph.add_redlist(entry)
        logger.info("Redlist override: %s -> %s (l2=%.3f, penalty=%.2f, reason=%s)",
                    source_id, target_id, new_l2, penalty_factor, reason)
        return entry

    def redlist_lazy_rollback(self, node_id: str, now: Optional[float] = None) -> bool:
        """惰性回滚：检索时检查红名单是否过期，若是且 restore_l3_on_expire=True 则重算 L3。

        Returns:
            是否执行了回滚。
        """
        if now is None:
            now = time.time()

        node = self.graph.get_node(node_id)
        if node is None:
            return False

        rolled_back = False
        for (source, target), entry in list(self.graph._redlist.items()):
            if source != node_id and target != node_id:
                continue
            if entry.expire_at is not None and entry.expire_at <= now:
                if entry.restore_l3_on_expire:
                    # 触发 L3 物理重算
                    self.update_l3([node_id])
                    rolled_back = True
                    logger.info("Lazy rollback: L3 recomputed for %s due to expired redlist", node_id)
                # 移除过期条目
                del self.graph._redlist[(source, target)]

        return rolled_back

    # ═══════════════════════════════════════════════════════════
    #  单次整理流水线
    # ═══════════════════════════════════════════════════════════

    def consolidate(self, llm_output: str, query: str) -> dict[str, Any]:
        """执行一次完整的记忆整理流水线。

        Args:
            llm_output: LLM 返回的文本输出。
            query: 原始用户查询（用于节点定位中的冷启动判断）。

        Returns:
            包含所有操作统计的字典。
        """
        stats: dict[str, Any] = {
            "triples": 0,
            "nodes_created": 0,
            "edges_created": 0,
            "triangular_edges": 0,
            "nodes_updated": [],
        }
        now = time.time()

        # §3.1 原子化拆解
        triples = self.decompose_triples(llm_output)
        stats["triples"] = len(triples)

        if not triples:
            logger.info("No triples extracted from LLM output")
            return stats

        for subject, relation, obj in triples:
            # §3.2 节点定位/创建
            subj_node = self.locate_or_create_node(subject)
            obj_node = self.locate_or_create_node(obj)

            if not subj_node.name.startswith("UNK_") and subj_node.id not in stats["nodes_updated"]:
                stats["nodes_updated"].append(subj_node.id)
            if not obj_node.name.startswith("UNK_") and obj_node.id not in stats["nodes_updated"]:
                stats["nodes_updated"].append(obj_node.id)

            # §3.3 边处理
            edge = self.process_edge(subj_node.id, obj_node.id, relation)
            stats["edges_created"] += 1

            # §3.5 偏置漂移（仅 explicit_fact 触发）
            self.apply_bias_drift(edge)

            # §3.6 增量式三角闭合
            inferred_edges = self.triangular_closure(subj_node.id, obj_node.id)
            stats["triangular_edges"] += len(inferred_edges)

        # §3.7 L3 热度更新
        self.update_l3(stats["nodes_updated"])

        # §3.4 时间衰减
        decayed = self.apply_time_decay(now)
        stats["edges_decayed"] = decayed

        logger.info(
            "Consolidation complete: %d triples, %d edges created, %d triangular, %d decayed",
            stats["triples"], stats["edges_created"],
            stats["triangular_edges"], decayed,
        )

        return stats
