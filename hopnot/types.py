"""hopnot — 核心数据类型。"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class EdgeType(Enum):
    """边关系类型。"""
    ASSOC = "ASSOC"          # 正向关联
    NOT = "NOT"              # 否定关联


class EdgeOrigin(Enum):
    """边来源标识。"""
    EXPLICIT_FACT = "explicit_fact"   # 显式事实：来自 LLM 输出拆解
    INFERRED = "inferred"             # 联想推断：由三角闭合生成


@dataclass
class Node:
    """记忆节点（记忆原子）。

    Attributes:
        id: 全局唯一标识符。
        name: 节点名称（文本标签）。
        b: 基座语义向量 (frozen) —— 永不修改，仅由 embedding 初始化。
        p: 个人语义偏置向量（漂移层）—— 初始为零向量，可在整理阶段微调。
        l3: 节点基础势能 [0, 1]。
        freq: 节点被访问的总次数。
        created_at: 创建时间戳（秒）。
        last_visited: 最近一次访问时间戳（秒）。
        aliases: 别名集合（含 alias_of 扩展映射）。
    """
    id: str
    name: str
    b: list[float] = field(repr=False)          # 基座语义向量
    p: list[float] = field(default_factory=list, repr=False)  # 偏置向量
    l3: float = 0.3
    freq: int = 0
    created_at: float = 0.0
    last_visited: float = 0.0
    aliases: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not self.p:
            self.p = [0.0] * len(self.b) if self.b else []

    @property
    def dim(self) -> int:
        return len(self.b)


@dataclass
class Edge:
    """有向边（关联关系）。

    Attributes:
        source: 源节点 ID。
        target: 目标节点 ID。
        l2: 长期关联权重 [0, 1]。
        confidence: 边置信度 [0, 1]。
        edge_type: 边关系类型。
        origin: 边来源标识。
        refines: 细化边标注 —— 指明该边细化了哪条泛化边（如 "关联"）。
        recovery_count: 边恢复次数。
        version: 乐观锁版本号（用于并发写安全）。
        created_at: 创建时间戳（秒）。
        last_visited: 最近一次被访问时间戳（秒）。
        access_count: 被访问的总次数。
    """
    source: str
    target: str
    l2: float = 0.45
    confidence: float = 0.5
    edge_type: EdgeType = EdgeType.ASSOC
    origin: EdgeOrigin = EdgeOrigin.EXPLICIT_FACT
    refines: Optional[str] = None
    recovery_count: int = 0
    version: int = 1
    created_at: float = 0.0
    last_visited: float = 0.0
    access_count: int = 1


@dataclass
class RedlistEntry:
    """人工权威干预红名单条目。

    Attributes:
        source: 源节点 ID。
        target: 目标节点 ID。
        new_l2: 检索期覆盖的 L2 值。
        penalty_factor: 热度惩罚系数（L3 乘数）。
        expire_at: 过期时间戳（秒），None 表示永不过期。
        restore_l3_on_expire: 过期时是否回滚 L3 物理值。
        invalidated_not_edges: 被无效化的 NOT 边列表。
        reason: 干预原因说明。
        created_at: 创建时间戳。
    """
    source: str
    target: str
    new_l2: float
    penalty_factor: float = 0.5
    expire_at: Optional[float] = None
    restore_l3_on_expire: bool = True
    invalidated_not_edges: list[tuple[str, str]] = field(default_factory=list)
    reason: str = ""
    created_at: float = 0.0


@dataclass
class MergedLogEntry:
    """UNK_ 节点合并日志条目。"""
    unk_node_id: str
    target_node_id: str
    merged_at: float = 0.0


@dataclass
class ConflictLogEntry:
    """边冲突日志条目。"""
    source: str
    target: str
    existing_type: EdgeType
    new_type: EdgeType
    confidence_before: float
    confidence_after: float
    created_at: float = 0.0
