"""主协调器 —— hopnot完整闭环。

实现规范（v1.7）§4.1 完整闭环流程图：

用户 Query
    ↓
┌─────────────────────────────────────────────────────────────────┐
│  检 索 阶 段 （读）                                            │
│ ① 种子选取（闲聊门控 + 冷启动 → 激活值=1.0，Prompt指令）      │
│ ② 随机游走（NOT剔除 + 细化边压制 + 红名单覆盖）               │
│ ③ Token池拆分截断（冷启动节点豁免）                           │
└─────────────────────────┬───────────────────────────────────────┘
                          ↓
                ┌─────────────────┐
                │  注入 LLM        │
                └────────┬─────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│  整 理 阶 段 （写）                                            │
│ ① 原子化拆解（千问0.5B冻结）                                  │
│ ② 节点定位（UNK阻尼合并）                                     │
│ ③ 边处理（细化边打标，乐观锁并发控制）                        │
│ ④ 混合时间衰减                                               │
│ ⑤ 偏置漂移（仅 explicit_fact）                               │
│ ⑥ 增量三角闭合（origin=inferred，不触发漂移）                │
│ ⑦ L3热度更新（分位数归一化 + 否定惩罚）                      │
│ ⑧ 日志归档 + 红名单惰性回滚                                  │
└─────────────────────────────────────────────────────────────────┘
                          ↓
                  记忆库已更新，等待下次检索
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .config import HippocampusConfig, get_default_config
from .consolidation import MemoryConsolidation
from .embedding import BaseEmbedding, DummyEmbedding
from .graph import MemoryGraph
from .retrieval import MemoryRetrieval, RetrievalResult

logger = logging.getLogger(__name__)


class HippocampusMemorySystem:
    """hopnot主协调器。

    封装检索与整理阶段的完整闭环。典型用法：

        system = HippocampusMemorySystem()
        # 检索
        result = system.retrieve("用户的查询")
        # 获取检索结果的 Prompt 注入文本
        prompt = system.build_prompt(result, "用户查询")
        # 整理（将 LLM 输出写回记忆）
        stats = system.consolidate(llm_output, query)
    """

    def __init__(
        self,
        graph: Optional[MemoryGraph] = None,
        embedding: Optional[BaseEmbedding] = None,
        config: Optional[HippocampusConfig] = None,
    ) -> None:
        """初始化主协调器。

        Args:
            graph: 记忆图实例。若为 None 则新建。
            embedding: 嵌入模型实例。若为 None 则使用 DummyEmbedding。
            config: 配置。若为 None 则使用默认配置。
        """
        self.config = config or get_default_config()
        self.graph = graph or MemoryGraph(self.config)
        self.embedding = embedding or DummyEmbedding()
        self.retrieval = MemoryRetrieval(self.graph, self.embedding, self.config)
        self.consolidation = MemoryConsolidation(self.graph, self.embedding, self.config)

        # 会话上下文跟踪（用于近期偏置计算等）
        self._recent_queries: list[str] = []
        self._last_retrieval_result: Optional[RetrievalResult] = None

    # ═══════════════════════════════════════════════════════════
    #  检索阶段
    # ═══════════════════════════════════════════════════════════

    def retrieve(self, query: str) -> RetrievalResult:
        """执行检索阶段。

        Args:
            query: 用户查询文本。

        Returns:
            RetrievalResult 包含激活节点列表。
        """
        self._recent_queries.append(query)
        # 保持最近 10 条查询
        if len(self._recent_queries) > 10:
            self._recent_queries.pop(0)

        result = self.retrieval.retrieve(query)
        self._last_retrieval_result = result
        return result

    def build_prompt(self, result: RetrievalResult, query: str) -> str:
        """构建用于注入 LLM 的 Prompt。

        根据检索结果组装上下文注入文本。
        冷启动时注入特殊指令。
        """
        if result.cold_start and result.cold_start_node_id is not None:
            cold_node = self.graph.get_node(result.cold_start_node_id)
            node_name = cold_node.name if cold_node else result.cold_start_node_id
            return (
                f"【记忆系统提示】当前查询触及全新知识点（节点: {node_name}，首次创建），暂无关联上下文。\n"
                f"请基于自身知识回答用户问题，但回答内容需尽量丰富、结构化，"
                f"以便系统将您的回答拆解为三元组进行记忆存储。\n"
                f"用户查询: {query}"
            )

        if not result.activated_nodes:
            return query

        # 组装记忆上下文
        lines = ["【记忆上下文 — 按激活值降序排列】"]
        for rank, (node_id, activation) in enumerate(result.activated_nodes, 1):
            node = self.graph.get_node(node_id)
            node_name = node.name if node else node_id
            lines.append(f"  {rank}. [{activation:.4f}] {node_name}")

        lines.append("")
        lines.append(f"记忆注入 Token 数: {result.tokens_used}")
        lines.append(f"用户查询: {query}")

        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════
    #  整理阶段
    # ═══════════════════════════════════════════════════════════

    def consolidate(self, llm_output: str, query: str) -> dict[str, Any]:
        """执行整理阶段，将 LLM 输出写回记忆。

        Args:
            llm_output: LLM 的输出文本。
            query: 原始用户查询。

        Returns:
            整理统计字典。
        """
        stats = self.consolidation.consolidate(llm_output, query)

        # 红名单惰性回滚（检索时检查）
        if self._last_retrieval_result is not None:
            for node_id, _ in self._last_retrieval_result.activated_nodes:
                self.consolidation.redlist_lazy_rollback(node_id)

        return stats

    # ═══════════════════════════════════════════════════════════
    #  完整闭环
    # ═══════════════════════════════════════════════════════════

    def process_query(
        self,
        query: str,
        llm_response: Optional[str] = None,
    ) -> dict[str, Any]:
        """执行一次完整的检索→(LLM)→整理闭环。

        Args:
            query: 用户查询。
            llm_response: LLM 的响应文本。若为 None 则不执行整理阶段。

        Returns:
            {"retrieval": RetrievalResult, "prompt": str, "consolidation": dict|None}
        """
        # 检索
        result = self.retrieve(query)
        prompt = self.build_prompt(result, query)

        output = {
            "retrieval": result,
            "prompt": prompt,
            "consolidation": None,
        }

        # 整理
        if llm_response is not None:
            stats = self.consolidate(llm_response, query)
            output["consolidation"] = stats

        return output

    # ═══════════════════════════════════════════════════════════
    #  运维接口
    # ═══════════════════════════════════════════════════════════

    def apply_human_correction(
        self,
        source: str,
        target: str,
        new_l2: float,
        penalty_factor: float = 0.5,
        expire_at: Optional[float] = None,
        restore_l3_on_expire: bool = True,
        reason: str = "",
    ) -> None:
        """人工权威干预接口（/memory_correction）。

        联动修复 + 惰性回滚。对应 §3.9。
        """
        self.consolidation.apply_redlist_override(
            source_id=source,
            target_id=target,
            new_l2=new_l2,
            penalty_factor=penalty_factor,
            expire_at=expire_at,
            restore_l3_on_expire=restore_l3_on_expire,
            reason=reason,
        )

    def archive_low_weight_edges(self) -> list:
        """执行低权边归档（§3.8）。"""
        return self.consolidation.archive_low_weight_edges()

    def merge_all_unk_nodes(self) -> int:
        """对所有 UNK_ 节点执行合并检查。返回合并的数量。"""
        merged = 0
        all_nodes = self.graph.get_all_nodes()

        # 收集所有 UNK_ 节点
        unk_nodes = [n for n in all_nodes if n.name.startswith("UNK_")]
        normal_nodes = [n for n in all_nodes if not n.name.startswith("UNK_")]

        for unk_node in unk_nodes:
            if not self.graph.has_node(unk_node.id):
                continue  # 可能已被合并

            # 尝试找最佳匹配的正常节点
            best_sim = 0.0
            best_node = None
            for norm_node in normal_nodes:
                sim = self.graph.cosine_similarity(unk_node.b, norm_node.b)
                if sim > best_sim:
                    best_sim = sim
                    best_node = norm_node

            if best_sim > self.config.merge_threshold and best_node is not None:
                self.consolidation.merge_unk_node(unk_node.id, best_node.id)
                merged += 1

        return merged

    def get_stats(self) -> dict[str, Any]:
        """获取系统统计信息。"""
        return {
            "node_count": self.graph.node_count(),
            "edge_count": self.graph.edge_count(),
            "redlist_count": len(self.graph._redlist),
            "operation_log_count": len(self.graph.operation_log),
            "conflict_log_count": len(self.graph.conflict_log),
            "merged_log_count": len(self.graph.merged_log),
            "avg_out_degree": self.graph.get_avg_out_degree(),
        }
