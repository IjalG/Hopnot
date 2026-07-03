# API Reference

## Core Classes

### `HippocampusMemorySystem`

Main orchestrator wrapping retrieval and consolidation.

```python
system = HippocampusMemorySystem(
    graph=None,       # existing MemoryGraph (optional)
    embedding=None,   # embedding model, default DummyEmbedding
    config=None,      # config, default HippocampusConfig()
)
```

**Methods:**

| Method | Description |
|:---|:---|
| `retrieve(query)` → `RetrievalResult` | Execute retrieval |
| `build_prompt(result, query)` → `str` | Build prompt for LLM injection |
| `consolidate(llm_output, query)` → `dict` | Execute consolidation |
| `process_query(query, llm_response)` → `dict` | Full pipeline |
| `apply_human_correction(...)` | Human intervention |
| `archive_low_weight_edges()` | Archive low-weight edges |
| `merge_all_unk_nodes()` → `int` | Merge UNK nodes |
| `get_stats()` → `dict` | System statistics |

---

### `MemoryGraph`

Directed weighted memory graph.

```python
graph = MemoryGraph(config=None)
```

**Node operations:**

| Method | Description |
|:---|:---|
| `add_node(node)` | Add a node |
| `get_node(node_id)` → `Node` | Get node by ID |
| `remove_node(node_id)` | Remove a node |
| `get_all_nodes()` → `list[Node]` | All nodes |
| `node_count()` → `int` | Node count |

**Edge operations:**

| Method | Description |
|:---|:---|
| `add_edge(edge)` | Add an edge |
| `get_edge(source, target)` → `Edge` | Get general edge |
| `has_edge(source, target)` → `bool` | Check existence |
| `has_refinement_edge(s, t)` → `bool` | Check refinement edge |
| `get_out_edges(node_id)` → `list[tuple[str, Edge]]` | Outgoing edges |
| `get_in_edges(node_id)` → `list[tuple[str, Edge]]` | Incoming edges |
| `remove_edge(source, target)` | Remove an edge |

**Vector utilities:**

| Method | Description |
|:---|:---|
| `cosine_similarity(a, b)` → `float` | Cosine similarity |
| `normalize(vec)` → `list[float]` | L2 normalization |
| `compute_effective_vector(node)` → `list[float]` | Runtime effective vector |
| `clamp_vector_norm(vec, max_norm)` → `list[float]` | Clamp vector norm |

**Redlist:**

| Method | Description |
|:---|:---|
| `add_redlist(entry)` | Add redlist entry |
| `is_redlist_active(source, target)` → `bool` | Check active |
| `remove_expired_redlist()` | Remove expired entries |

---

### `MemoryRetrieval`

Retrieval phase implementation.

```python
retrieval = MemoryRetrieval(graph, embedding, config=None)
result = retrieval.retrieve(query)
```

### `MemoryConsolidation`

Consolidation phase implementation.

```python
consolidation = MemoryConsolidation(graph, embedding, config=None)
stats = consolidation.consolidate(llm_output, query)
```

## Data Classes

### `Node`

```python
Node(
    id=str,           # unique ID
    name=str,         # node name
    b=list[float],    # base semantic vector (frozen)
    p=list[float],    # personal bias vector (driftable)
    l3=0.3,           # base potential
    freq=0,           # access frequency
    created_at=0.0,   # creation timestamp
    last_visited=0.0, # last access timestamp
    aliases=set(),    # aliases
)
```

### `Edge`

```python
Edge(
    source=str,              # source node ID
    target=str,              # target node ID
    l2=0.45,                 # long-term weight
    confidence=0.5,          # confidence
    edge_type=EdgeType.ASSOC,  # ASSOC / NOT
    origin=EdgeOrigin.EXPLICIT_FACT,  # explicit_fact / inferred
    refines=None,            # refinement annotation (e.g. "关联")
    recovery_count=0,        # recovery count
    version=1,               # optimistic lock version
)
```

### `RetrievalResult`

```python
RetrievalResult(
    activated_nodes=[(node_id, activation)],
    tokens_used=0,
    cold_start=False,
    cold_start_node_id=None,
    query="",
    seed_nodes=[],
    num_steps=0,
)
```

## Enums

### `EdgeType`

| Value | Description |
|:---|:---|
| `ASSOC` | Positive association |
| `NOT` | Negative association |

### `EdgeOrigin`

| Value | Description |
|:---|:---|
| `EXPLICIT_FACT` | Explicit fact (from LLM) |
| `INFERRED` | Inferred (triangular closure) |

## Embedding Interface

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
    model_path="",   # defaults to .model_cache/Qwen/Qwen3-Embedding-0___6B
    device="cpu",    # or "cuda"
)
```

### `DummyEmbedding`

```python
embedder = DummyEmbedding(
    dim=64,   # vector dimension
    seed=42,  # random seed (deterministic)
)
```

## Configuration

### `HippocampusConfig`

See [Tuning Guide](tuning.md) for all hyperparameters.

```python
config = get_default_config()
config.recall_threshold = 0.55
config.merge_threshold = 0.92
```
