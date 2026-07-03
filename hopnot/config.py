"""系统配置 —— 所有超参数汇总表（v1.7 最终封板版）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HippocampusConfig:
    """hopnot 超参数配置。"""

    # ── 检索阶段 ──────────────────────────────────────────────────
    # 种子选取
    recall_threshold: float = 0.10          # τ_recall: 粗筛召回阈值
    num_seeds: int = 3                      # N: 种子数量
    neighbor_vote_weight: float = 0.50      # λ: 邻居投票权重
    recency_bias: float = 0.20              # δ: 近期偏置加成
    neighbor_cutoff: int = 200              # K_neighbor: 邻居投票截断数

    # 边权重计算
    decay_general: float = 0.30             # 泛化边衰减系数（细化边压制）

    # 随机游走
    restart_prob: float = 0.40              # ρ: 重启概率
    max_steps: int = 20                     # T_max: 最大迭代步数
    energy_threshold: float = 0.001         # 能量耗尽阈值
    convergence_threshold: float = 1e-5     # 收敛 L1 差阈值

    # 路径决策
    novelty_lower: float = 0.15             # α: 新颖性下界
    novelty_upper: float = 0.75             # β: 新颖性上界
    gamma_dynamic_base: float = 3.0         # γ 动态基数
    gamma_dynamic_mult: float = 1.5         # γ 动态乘数
    decision_threshold: float = 0.45        # Θ: 综合决策阈值
    weight_activation: float = 0.25         # ω_A
    weight_coherence: float = 0.35          # ω_S
    weight_novelty: float = 0.20            # ω_N
    weight_diffusion: float = 0.20          # ω_D

    # 硬性拒绝
    min_edge_weight: float = 0.10           # 边太弱拒绝
    max_semantic_drift: float = 0.85        # 单步跑偏拒绝
    max_path_depth: int = 5                 # 深度超标拒绝

    # 输出截断
    output_threshold: float = 0.005         # τ_output: 输出激活阈值
    max_total_tokens: int = 4096            # LLM 总上下文窗口
    reserved_tokens: int = 512              # System+User 预留
    memory_context_tokens: int = 3584       # 记忆注入池

    # ── 整理阶段 ──────────────────────────────────────────────────
    # 节点定位
    merge_threshold: float = 0.78           # τ_merge: 节点复用相似度
    short_alias_threshold: float = 0.50     # τ_short_alias: 短文本软关联阈值

    # 边新建
    l2_initial: float = 0.30                # 新边初始权重
    l2_bonus: float = 0.15                  # 新边创建红利

    # 边更新
    l2_hebb_increment: float = 0.05         # Hebbian 强化增量
    l2_recovery_bonus: float = 0.08         # 恢复红利增量
    confidence_recovery: float = 0.10       # 恢复置信度增量
    confidence_max_recovery: float = 0.50   # 恢复置信度上限

    # 时间衰减
    decay_per_access: float = 0.0003        # 每次访问衰减
    decay_long_tail: float = 0.01           # 长尾惩罚（超30天）
    l2_min: float = 0.001                   # 权重下限
    l2_prune_threshold: float = 0.08        # 低权修剪阈值
    long_tail_days: float = 30.0            # 长尾天数阈值

    # 偏置漂移
    lambda_new: float = 0.005               # 新建边漂移率（ASSOC）
    lambda_confirm: float = 0.008           # 确认边漂移系数
    lambda_not: float = 0.002               # NOT 边推远漂移率
    alpha_bias: float = 0.10                # α: 偏置向量融合系数
    p_norm_max: float = 0.50                # 偏置向量最大模长
    l3_core_threshold: float = 0.90         # 核心节点 L3 阈值
    core_bias_half: bool = True             # 核心节点偏置更新率减半

    # 增量三角闭合
    triangle_discount: float = 0.50         # κ: 三角闭合折扣系数
    triangle_min_l2: float = 0.40           # τ_tri: 三角闭合最低边权
    triangle_topk: int = 50                 # topk_tri: 局部出度检查上限

    # 置信度
    c_initial: float = 0.50                 # 新边初始置信度
    c_conflict_threshold: float = 0.30      # 冲突警告阈值

    # L3 更新
    l3_initial: float = 0.30                # 新节点初始 L3
    l3_cold_start: float = 0.20             # 冷启动节点 L3
    l3_not_penalty: float = 0.005           # 否定边出度惩罚系数
    l3_min: float = 0.05                    # L3 下限
    l3_smoothing: float = 0.70              # L3 滑动平滑系数
    l3_update_rate: float = 0.30            # L3 新值贡献率

    # ── 闲聊门控 ──────────────────────────────────────────────────
    chitchat_patterns: tuple = (
        r"^(你好|谢谢|再见|今天天气|现在几点|你是谁|帮我|请)",
    )

    # ── 运维 ──────────────────────────────────────────────────────
    recovery_alert_threshold: int = 3       # 3个月内恢复次数告警
    recovery_alert_days: float = 90.0       # 告警窗口天数
    alias_of_threshold: float = 0.50        # alias_of 自动扩展阈值

    # ── 并发控制 ─────────────────────────────────────────────────
    use_optimistic_locking: bool = True     # 使用乐观锁


def get_default_config() -> HippocampusConfig:
    """获取默认配置（v1.7 封板版）。"""
    return HippocampusConfig()
