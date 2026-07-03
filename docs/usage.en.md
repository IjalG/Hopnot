# Usage Guide

## Installation

```bash
pip install torch transformers sentence-transformers modelscope
git clone git@github.com:IjalG/hopnot.git
cd hopnot
```

## Basic Usage

### Initialize

```python
from hopnot import HippocampusMemorySystem

# Uses DummyEmbedding (random vectors, for testing)
system = HippocampusMemorySystem()
```

### Use Qwen3-Embedding Model

```python
from hopnot import Qwen3Embedding, HippocampusMemorySystem, get_default_config

config = get_default_config()
config.recall_threshold = 0.55
config.merge_threshold = 0.92

embedder = Qwen3Embedding(device="cpu")  # auto-downloads ~1.1GB on first run
system = HippocampusMemorySystem(embedding=embedder, config=config)
```

### Write Memory

```python
# Decompose LLM output into triples and write to graph
system.consolidate(
    "(Python, is, programming language)\n"
    "(Python, used for, AI development)",
    query="system init",
)
```

### Read Memory

```python
result = system.retrieve("Python")
for node_id, activation in result.activated_nodes:
    node = system.graph.get_node(node_id)
    print(f"[{activation:.4f}] {node.name}")
```

### Build LLM Prompt

```python
prompt = system.build_prompt(result, "Python")
print(prompt)
# [Memory context — sorted by activation descending]
#   1. [0.0124] Deep Learning
#   2. [0.0112] Machine Learning
#   ...
```

### Full Pipeline

```python
output = system.process_query(
    query="What is deep learning?",
    llm_response="(deep learning, uses, neural networks)",
)
```

## Embedding Models

### Qwen3-Embedding-0.6B (recommended)

Auto-downloads from ModelScope: https://modelscope.cn/models/Qwen/Qwen3-Embedding-0.6B

```python
from hopnot import Qwen3Embedding
e = Qwen3Embedding(device="cpu")    # or "cuda"
e.embed("artificial intelligence")   # → 1024-dim vector
e.embed_batch(["A", "B"])            # batch embedding
```

### DummyEmbedding (testing)

```python
from hopnot import DummyEmbedding
e = DummyEmbedding(dim=64, seed=42)
```

### Custom Embedding

Inherit `BaseEmbedding`:

```python
from hopnot import BaseEmbedding

class MyEmbedding(BaseEmbedding):
    @property
    def dim(self) -> int:
        return 768

    def embed(self, text: str) -> list[float]:
        # Call your model
        return normalized_vector
```

## Admin

### Human Correction

```python
system.apply_human_correction(
    source="node_A",
    target="node_B",
    new_l2=0.95,
    reason="human confirmed strong association",
)
```

### System Statistics

```python
stats = system.get_stats()
# {'node_count': 5, 'edge_count': 7, ...}
```

### Merge UNK Nodes

```python
system.merge_all_unk_nodes()
```

### Archive Low-Weight Edges

```python
system.archive_low_weight_edges()
```
