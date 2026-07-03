# 参数调优

## 全部超参数

| 符号 | 含义 | 默认值 | 阶段 |
|:---|:---|:---:|:---|
| $\tau_{\text{recall}}$ | 粗筛召回阈值 | 0.10 | 检索 |
| $N$ | 种子数量 | 3 | 检索 |
| $\lambda$ | 邻居投票权重 | 0.50 | 检索 |
| $\delta$ | 近期偏置加成 | 0.20 | 检索 |
| $\rho$ | RWR 重启概率 | 0.40 | 检索 |
| $\Theta$ | 扩展决策阈值 | 0.45 | 检索 |
| $\omega_A,\omega_S,\omega_N,\omega_D$ | Ψ 权重 | 0.25, 0.35, 0.20, 0.20 | 检索 |
| $\tau_{\text{output}}$ | 输出激活阈值 | 0.005 | 检索 |
| $\tau_{\text{merge}}$ | 节点复用相似度 | 0.78 | 整理 |
| $L2_{\text{initial}}$ | 新边初始权重 | 0.30 | 整理 |
| $\lambda_{\text{new}}$ | 新建边漂移率 | 0.005 | 整理 |
| $\lambda_{\text{confirm}}$ | 确认边漂移系数 | 0.008 | 整理 |
| $\kappa$ | 三角闭合折扣 | 0.50 | 整理 |
| $L3_{\text{min}}$ | L3 下限 | 0.05 | 整理 |

完整参数表见 `knot/config.py`。

## Qwen3-Embedding 适配

Qwen3-Embedding-0.6B 是 decoder-only 专用的嵌入模型，其向量空间较密集，
建议调整以下参数：

```python
config.recall_threshold = 0.55    # 默认 0.10，调高防止误匹配
config.merge_threshold = 0.92     # 默认 0.78，调高防止语义相近节点误合并
config.short_alias_threshold = 0.70  # 默认 0.50
```

## 自定义嵌入模型

继承 `BaseEmbedding`，实现 `embed()` 和 `dim` 属性即可。

不同嵌入模型可能需要不同的阈值。经验法则：

| 嵌入模型类型 | τ_recall | τ_merge |
|:---|:---:|:---:|
| 专用编码器（BGE, GTE） | 0.10-0.20 | 0.78-0.85 |
| decoder-only 通用模型 | 0.30-0.55 | 0.85-0.92 |
| 随机/测试向量 | 0.05-0.15 | 0.50-0.70 |

## 其他

- **闲聊门控**：`config.chitchat_patterns` 可自定义正则列表
- **时间衰减**：`config.decay_per_access` 控制遗忘速度
- **核心节点保护**：`L3 > 0.9` 时偏置更新率减半
