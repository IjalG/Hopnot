# 技术规格

> 对应规范文档 v1.7 的 §1 和 §4.1。

## 核心数据模型

所有记忆存储在一个有向加权图 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ 中。

| 符号 | 含义 | 范围 |
|:---|:---|:---|
| $\mathcal{V}$ | 节点集合（记忆原子） | 有限集 |
| $\mathcal{E}$ | 有向边集合（关联关系） | $\subseteq \mathcal{V} \times \mathcal{V}$ |
| $\mathbf{b}_{v_i}$ | 基座语义向量（**冻结**） | 归一化，永不修改 |
| $\mathbf{p}_{v_i}$ | 个人语义偏置（漂移层） | 初始为零，$\|\mathbf{p}\| \le 0.5$ |
| $\mathbf{e}_{v_i}$ | 运行时有效向量 | 检索时临时计算 |
| $L2_{ij}$ | 边长期关联权重 | $[0, 1]$，下限 $0.001$ |
| $L3_i$ | 节点基础势能 | $[0, 1]$ |
| $c_{ij}$ | 边置信度 | $[0, 1]$ |
| $\text{type}_{ij}$ | 边关系类型 | `ASSOC` / `NOT` |

**运行时有效向量：**

$$\mathbf{e}_{v_i} = \text{norm}(\mathbf{b}_{v_i} + 0.1 \cdot \mathbf{p}_{v_i})$$

## 双向量架构

每个节点维护两个向量：

- **基座向量 b** — 由嵌入模型生成，永久冻结。作为语义的锚点。
- **偏置向量 p** — 初始为零，在整理阶段通过漂移微调。实现个性化记忆。

## 闭环流程

```
用户 Query
    ↓
┌──────────────────────────────────────┐
│  检 索 阶 段 （读）                  │
│ ① 种子选取（闲聊门控 + 冷启动）      │
│ ② 随机游走扩散（RWR）               │
│ ③ Token 截断输出                    │
└──────────┬───────────────────────────┘
           ↓
    ┌──────────────┐
    │  注入 LLM    │
    └──────┬───────┘
           ↓
┌──────────────────────────────────────┐
│  整 理 阶 段 （写）                  │
│ ① 原子化拆解 → 三元组                │
│ ② 节点定位（UNK 阻尼合并）           │
│ ③ 边处理（4 分支决策树）             │
│ ④ 混合时间衰减                      │
│ ⑤ 偏置漂移                          │
│ ⑥ 增量三角闭合                      │
│ ⑦ L3 热度更新                       │
│ ⑧ 日志归档 + 红名单                 │
└──────────────────────────────────────┘
           ↓
    记忆库已更新，等待下次检索
```

## 检索阶段

### 有效边权重

$$w_{ij} = \text{TypeWeight} \times L2_{ij} \times (0.6 + 0.4 \times L3_j^{\text{eff}}) \times \text{RefinePenalty} + L1_{ij} \times 0.3 + \varepsilon_{ij}$$

- $\varepsilon_{ij} \sim \mathcal{N}(0, 0.05)$
- $\text{TypeWeight} = -0.2$ 若 `type=NOT`，否则 $1.0$
- $\text{RefinePenalty} = 0.3$ 若存在细化边，否则 $1.0$
- 截断：$\max(-0.2, \min(1, w_{ij}))$

### 种子选取

1. **粗筛**：$\mathcal{C} = \{v_i \mid \langle \mathbf{e}_q, \mathbf{e}_{v_i} \rangle \ge 0.1\} \cup \mathcal{C}_{\text{exact}}$
2. **闲聊门控**：匹配 `^(你好|谢谢|再见|...` → 返回空集
3. **冷启动**：无候选时创建新节点，$L3=0.2$，激活值 $=1.0$，跳过游走
4. **邻居投票**：$\text{ContextScore} = s_i + 0.5 \times \text{avg\_neighbor} + 0.2 \times \text{recency}$
5. **能量归一化**：$\alpha_i = \text{score}_i / \sum \text{score}_j$

### 随机游走 (RWR)

$$\mathbf{s}_{t+1} = (1 - \rho) \cdot \mathbf{s}_t \times \mathbf{W} + \rho \cdot \mathbf{s}_0, \quad \rho = 0.4$$

- 转移矩阵 $\mathbf{W}$ 行归一化，排除 NOT 边
- 路径连贯性：$S(P_k) = \frac{1}{k}\sum_{i=1}^k \prod_{j=1}^i (1 - \langle \mathbf{e}_{v_{j-1}}, \mathbf{e}_{v_j} \rangle)$
- 扩展决策 $\Psi = 0.25A + 0.35S + 0.20N^* + 0.20D^*$，阈值 $0.45$

### 输出截断

- `MaxTotalTokens` = 4096
- `ReservedTokens` = 512
- `MemoryContextTokens` = 3584

## 整理阶段

### 边处理决策树

| 分支 | 条件 | 操作 |
|:---|:---|:---|
| 1 | 边不存在 | 新建：$L2=0.45,\ c=0.5$ |
| 2a | $c \ge 0.3$（一致） | 强化：$L2\text{+=}0.05,\ c\text{+=}0.05$ |
| 2b | $c < 0.3$（濒死） | 恢复：$L2\text{+=}0.08,\ c\text{+=}0.10$ |
| 3 | 关系冲突 | 降权 $c\text{-=}0.03$，新建 NOT 边 |
| 4 | 关系更细化 | 保留旧边，新建 `refines="关联"`边 |

### 偏置漂移

- **ASSOC（靠拢）**：$\mathbf{p}_A \leftarrow \mathbf{p}_A + \lambda \cdot (\mathbf{b}_B - \mathbf{b}_A)$
- **NOT（推远）**：$\mathbf{p}_A \leftarrow \mathbf{p}_A - \lambda_{\text{not}} \cdot (\mathbf{b}_B - \mathbf{b}_A)$
- 仅 `origin="explicit_fact"` 触发

### 三角闭合

仅局部触发：$A \to B$ 新建时，检查 $B$ 的出边邻居 Top 50：

$$L2_{AC} = 0.5 \cdot \frac{L2_{AB} + L2_{BC}}{2}$$

### L3 热度更新

$$L3 = 0.7 \times L3 + 0.3 \times (\text{deg\_ratio} + \text{freq\_ratio}) \times 0.5$$

- 否定边惩罚：$L3 \times (1 - 0.005 \times \text{count\_NOT})$，下限 $0.05$
