# 使用指南

## 安装

```bash
pip install hopnot  # 或直接从源码安装
pip install torch transformers sentence-transformers modelscope
```

## 基础用法

### 初始化

```python
from hopnot import HippocampusMemorySystem

# 默认使用 DummyEmbedding（随机向量，用于测试）
system = HippocampusMemorySystem()
```

### 使用 Qwen3-Embedding 模型

```python
from hopnot import Qwen3Embedding, HippocampusMemorySystem, get_default_config

config = get_default_config()
config.recall_threshold = 0.55
config.merge_threshold = 0.92

embedder = Qwen3Embedding(device="cpu")  # 首次自动下载 ~1.1GB
system = HippocampusMemorySystem(embedding=embedder, config=config)
```

### 写入记忆

```python
# 将 LLM 输出拆解为三元组写入
system.consolidate(
    "(人工智能, 包含, 机器学习)\n"
    "(机器学习, 包含, 深度学习)",
    query="系统初始化",
)
```

### 读取记忆

```python
result = system.retrieve("人工智能")
for node_id, activation in result.activated_nodes:
    node = system.graph.get_node(node_id)
    print(f"[{activation:.4f}] {node.name}")
```

### 构建 LLM Prompt

```python
prompt = system.build_prompt(result, "人工智能")
print(prompt)
# 【记忆上下文 — 按激活值降序排列】
#   1. [0.0124] 深度学习
#   2. [0.0112] 机器学习
#   ...
```

### 完整闭环

```python
output = system.process_query(
    query="深度学习是什么",
    llm_response="(深度学习, 使用, 神经网络)",
)
```

## 嵌入模型

### Qwen3-Embedding-0.6B（推荐）

从 ModelScope 自动下载：https://modelscope.cn/models/Qwen/Qwen3-Embedding-0.6B

```python
from hopnot import Qwen3Embedding
e = Qwen3Embedding(device="cpu")   # 或 "cuda"
e.embed("人工智能")                 # → 1024 维向量
e.embed_batch(["A", "B"])          # 批量
```

### DummyEmbedding（测试用）

```python
from hopnot import DummyEmbedding
e = DummyEmbedding(dim=64, seed=42)
```

### 自定义嵌入模型

继承 `BaseEmbedding` 即可：

```python
from hopnot import BaseEmbedding

class MyEmbedding(BaseEmbedding):
    @property
    def dim(self) -> int:
        return 768

    def embed(self, text: str) -> list[float]:
        # 调用你的模型
        return normalized_vector
```

## 运维接口

### 人工纠偏

```python
system.apply_human_correction(
    source="node_A",
    target="node_B",
    new_l2=0.95,
    reason="人工确认强关联",
)
```

### 查看系统统计

```python
stats = system.get_stats()
# {'node_count': 5, 'edge_count': 7, ...}
```

### 合并 UNK 节点

```python
system.merge_all_unk_nodes()
```

### 低权边归档

```python
system.archive_low_weight_edges()
```
