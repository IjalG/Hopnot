"""hopnot —— Hippocampus Memory System.

非神经网络、纯图结构的动态记忆模拟系统（检索 + 整理闭环）。
版本: v1.7
"""

from .config import HippocampusConfig, get_default_config
from .types import (
    EdgeType,
    EdgeOrigin,
    Node,
    Edge,
    RedlistEntry,
    MergedLogEntry,
    ConflictLogEntry,
)
from .graph import MemoryGraph
from .embedding import BaseEmbedding, DummyEmbedding, Qwen3Embedding
from .retrieval import MemoryRetrieval
from .consolidation import MemoryConsolidation
from .memory_system import HippocampusMemorySystem

__version__ = "1.7.0"
__all__ = [
    "HippocampusConfig",
    "get_default_config",
    "EdgeType",
    "EdgeOrigin",
    "Node",
    "Edge",
    "RedlistEntry",
    "MergedLogEntry",
    "ConflictLogEntry",
    "MemoryGraph",
    "BaseEmbedding",
    "DummyEmbedding",
    "Qwen3Embedding",
    "MemoryRetrieval",
    "MemoryConsolidation",
    "HippocampusMemorySystem",
]
