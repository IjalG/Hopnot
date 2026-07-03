# API 参考

## 核心类

### `HippocampusMemorySystem`

主协调器，封装检索与整理阶段的完整闭环。

```python
system = HippocampusMemorySystem(
    graph=None,       # 可传入已有 MemoryGraph
    embedding=None,   # 嵌入模型，默认 DummyEmbedding
    config=None,      # 配置，默认 HippocampusConfig()
)
```

**方法：**

| 方法 | 说明 |
|:---|:---|
| `retrieve(query)` → `RetrievalResult` | 执行检索 |
| `build_prompt(result, query)` → `str` | 构建注入 LLM 的 Prompt |
| `consolidate(llm_output, query)` → `dict` | 执行整理 |
| `process_query(query, llm_response)` → `dict` | 完整闭环 |
| `apply_human_correction(...)` | 人工干预 |
| `archive_low_weight_edges()` | 低权归档 |
| `merge_all_unk_nodes()` → `int` | 合并 UNK 节点 |
| `get_stats()` → `dict` | 系统统计 |

---

### `MemoryGraph`

有向加权记忆图，管理节点和边。

```python
graph = MemoryGraph(config=None)
```

**节点操作：**

| 方法 | 说明 |
|:---|:---|
| `add_node(node)` | 添加节点 |
| `get_node(node_id)` → `Node` | 获取节点 |
| `remove_node(node_id)` | 删除节点 |
| `get_all_nodes()` → `list[Node]` | 所有节点 |
| `node_count()` → `int` | 节点数 |

**边操作：**

| 方法 | 说明 |
|:---|:---|
| `add_edge(edge)` | 添加边 |
| `get_edge(source, target)` → `Edge` | 获取泛化边 |
| `has_edge(source, target)` → `bool` | 是否存在 |
| `has_refinement_edge(s, t)` → `bool` | 是否存在细化边 |
| `get_out_edges(node_id)` → `list[tuple[str, Edge]]` | 出边 |
| `get_in_edges(node_id)` → `list[tuple[str, Edge]]` | 入边 |
| `remove_edge(source, target)` | 删除边 |

**向量工具：**

| 方法 | 说明 |
|:---|:---|
| `cosine_similarity(a, b)` → `float` | 余弦相似度 |
| `normalize(vec)` → `list[float]` | L2 归一化 |
| `compute_effective_vector(node)` → `list[float]` | 运行时有效向量 |
| `clamp_vector_norm(vec, max_norm)` → `list[float]` | 模长裁剪 |

**红名单：**

| 方法 | 说明 |
|:---|:---|
| `add_redlist(entry)` | 添加红名单条目 |
| `is_redlist_active(source, target)` → `bool` | 是否生效 |
| `remove_expired_redlist()` | 移除过期条目 |

---

### `MemoryRetrieval`

检索阶段实现。

```python
retrieval = MemoryRetrieval(graph, embedding, config=None)
result = retrieval.retrieve(query)
```

### `MemoryConsolidation`

整理阶段实现。

```python
consolidation = MemoryConsolidation(graph, embedding, config=None)
stats = consolidation.consolidate(llm_output, query)
```

## 数据类

### `Node`

```python
Node(
    id=str,           # 唯一标识
    name=str,         # 节点名称
    b=list[float],    # 基座语义向量（冻结）
    p=list[float],    # 个人偏置向量（可漂移）
    l3=0.3,           # 基础势能
    freq=0,           # 访问频率
    created_at=0.0,   # 创建时间戳
    last_visited=0.0, # 最近访问
    aliases=set(),    # 别名集合
)
```

### `Edge`

```python
Edge(
    source=str,          # 源节点 ID
    target=str,          # 目标节点 ID
    l2=0.45,             # 长期关联权重
    confidence=0.5,      # 置信度
    edge_type=EdgeType.ASSOC,  # ASSOC / NOT
    origin=EdgeOrigin.EXPLICIT_FACT,  # explicit_fact / inferred
    refines=None,        # 细化标注（如 "关联"）
    recovery_count=0,    # 恢复次数
    version=1,           # 乐观锁版本号
)
```

### `RetrievalResult`

```python
RetrievalResult(
    activated_nodes=[(node_id, activation)],  # 按激活值降序
    tokens_used=0,        # Token 数
    cold_start=False,     # 是否冷启动
    cold_start_node_id=None,  # 冷启动节点 ID
    query="",             # 原始查询
    seed_nodes=[],        # 种子节点
    num_steps=0,          # 迭代步数
)
```

## 枚举

### `EdgeType`

| 值 | 说明 |
|:---|:---|
| `ASSOC` | 正向关联 |
| `NOT` | 否定关联 |

### `EdgeOrigin`

| 值 | 说明 |
|:---|:---|
| `EXPLICIT_FACT` | 显式事实（来自 LLM） |
| `INFERRED` | 联想推断（三角闭合生成） |

## 嵌入接口

### `BaseEmbedding`

```python
class BaseEmbedding(ABC):
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
    @property
    def dim(self) -> int: ...
```

### `Qwen3Embedding`

```python
embedder = Qwen3Embedding(
    model_path="",   # 默认从 .model_cache/Qwen/Qwen3-Embedding-0___6B 加载
    device="cpu",    # 或 "cuda"
)
```

### `DummyEmbedding`

```python
embedder = DummyEmbedding(
    dim=64,   # 向量维度
    seed=42,  # 随机种子（保证可重复）
)
```

## 配置

### `HippocampusConfig`

所有超参数见 [参数调优](tuning.md)。

```python
config = get_default_config()
config.recall_threshold = 0.55
config.merge_threshold = 0.92
```
