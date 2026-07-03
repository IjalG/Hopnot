# Tuning Guide

## All Hyperparameters

| Symbol | Meaning | Default | Phase |
|:---|:---|:---:|:---|
| $\tau_{\text{recall}}$ | Coarse recall threshold | 0.10 | Retrieval |
| $N$ | Seed count | 3 | Retrieval |
| $\lambda$ | Neighbor vote weight | 0.50 | Retrieval |
| $\delta$ | Recency bias | 0.20 | Retrieval |
| $\rho$ | RWR restart probability | 0.40 | Retrieval |
| $\Theta$ | Expansion decision threshold | 0.45 | Retrieval |
| $\omega_A,\omega_S,\omega_N,\omega_D$ | Ψ weights | 0.25, 0.35, 0.20, 0.20 | Retrieval |
| $\tau_{\text{output}}$ | Output activation threshold | 0.005 | Retrieval |
| $\tau_{\text{merge}}$ | Node reuse similarity | 0.78 | Consolidation |
| $L2_{\text{initial}}$ | New edge initial weight | 0.30 | Consolidation |
| $\lambda_{\text{new}}$ | New edge drift rate | 0.005 | Consolidation |
| $\lambda_{\text{confirm}}$ | Confirm edge drift factor | 0.008 | Consolidation |
| $\kappa$ | Triangular closure discount | 0.50 | Consolidation |
| $L3_{\text{min}}$ | L3 floor | 0.05 | Consolidation |

Full parameter table in `hopnot/config.py`.

## Qwen3-Embedding Adaptation

Qwen3-Embedding-0.6B produces denser vectors. Recommended adjustments:

```python
config.recall_threshold = 0.55    # default 0.10, raised to avoid false matches
config.merge_threshold = 0.92     # default 0.78, raised to avoid merging similar concepts
config.short_alias_threshold = 0.70  # default 0.50
```

## Custom Embedding Models

Implement `BaseEmbedding` with `embed()` and `dim` property.

Rule of thumb for thresholds:

| Embedding Type | τ_recall | τ_merge |
|:---|:---:|:---:|
| Dedicated encoder (BGE, GTE) | 0.10-0.20 | 0.78-0.85 |
| Decoder-only general model | 0.30-0.55 | 0.85-0.92 |
| Random/test vectors | 0.05-0.15 | 0.50-0.70 |

## Other Tips

- **Chitchat gate**: customize `config.chitchat_patterns` regex list
- **Time decay**: `config.decay_per_access` controls forgetting speed
- **Core node protection**: bias drift rate halves when `L3 > 0.9`
