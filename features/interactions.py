"""Interaction features: VRP regime × catalyst cross signals (43 features).
交互特征：VRP 状态 × 催化事件交叉信号（43 个特征）。

This is Layer 3 of the feature pipeline. It captures NON-LINEAR effects
that tree-based models can learn but benefit from being pre-computed:
这是特征管线的第 3 层。它捕捉树模型可以学习但预计算后效果更好的非线性效应：

Key insight / 核心洞察:
  High VRP alone doesn't guarantee a big move. FOMC alone doesn't either.
  But high VRP + FOMC imminent = explosive vol is very likely.
  单独的高 VRP 不保证大波动。单独的 FOMC 也不保证。
  但高 VRP + FOMC 临近 = 极端波动率的可能性非常高。

Feature structure (43 total) / 特征结构（共 43 个）:

  A. Regime flags (6):           VRP/vol bucketed into discrete states.
     状态标记 (6)：               VRP/波动率分桶为离散状态。
     vrp_high (>75th pct), vrp_extreme (>90th), vrp_positive (>0),
     vrp_negative (<-5%), high_vol_regime (RV>75th), low_vol_regime (RV<25th).

  B. Catalyst proximity flags (7): How close is the next event.
     催化事件接近度标记 (7)：      距下一个事件有多近。
     fomc/nfp_imminent (≤1 day), fomc/nfp_this_week (≤5 days),
     any_catalyst_imminent, vix_spiked_3d, big_move_recent_3d.

  C. Cross features (30):        Binary product of regime × catalyst.
     交叉特征 (30)：              状态 × 催化事件的二元乘积。
     Six sub-groups / 六个子组：
       - VRP × Event (9):        高 VRP 遇上事件催化 = 波动率释放
       - High vol × Event (4):   高已实现波动率 + 事件 = 动量延续
       - Gamma trap (4):         低 VRP/低波动率 + 事件 = 意外飙升
       - Momentum × Event (5):   VIX 飙升/大波动 + 事件 = 波动率集群
       - Rate × Event (1):       利率冲击 + FOMC = 双重不确定性
       - Continuous (4):         VRP × 事件紧迫度连续值
       - Yield curve × Event (2): 曲线倒挂 + 事件 = 衰退恐慌
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def build_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build cross features: VRP regime state × catalyst proximity.
    构建交叉特征：VRP 状态 × 催化事件接近度。

    All cross features are binary products (A × B) or continuous products.
    They fire only when BOTH conditions are true simultaneously.
    所有交叉特征为二元乘积 (A × B) 或连续乘积。
    仅当两个条件同时为真时才触发。

    Args:
        df: DataFrame with base (Layer 1) + external (Layer 2) features.
            包含基础（第 1 层）+ 外部（第 2 层）特征的 DataFrame。

    Returns:
        DataFrame with 43 additional interaction features.
        添加了 43 个交互特征的 DataFrame。
    """
    out = df.copy()

    # ── A. VRP regime buckets (4 flags) ──
    # VRP 状态分桶（4 个标记）
    # Classify VRP into discrete regimes for cross features.
    # 将 VRP 分类为离散状态用于交叉特征。
    vrp = out["vrp_20d"]
    out["vrp_high"] = (vrp > vrp.quantile(0.75)).astype(int)       # Top 25% / 前 25%
    out["vrp_extreme"] = (vrp > vrp.quantile(0.90)).astype(int)    # Top 10% / 前 10%
    out["vrp_positive"] = (vrp > 0).astype(int)                    # IV > RV (fear premium) / IV > RV（恐慌溢价）
    out["vrp_negative"] = (vrp < -0.05).astype(int)                # IV << RV (complacent) / IV << RV（自满）

    # ── A. Vol regime (2 flags) ──
    # 波动率状态（2 个标记）
    rv20 = out["realized_vol_20d"]
    out["high_vol_regime"] = (rv20 > rv20.quantile(0.75)).astype(int)  # High RV / 高已实现波动率
    out["low_vol_regime"] = (rv20 < rv20.quantile(0.25)).astype(int)   # Low RV / 低已实现波动率

    # ── B. Catalyst proximity flags (7 flags) ──
    # 催化事件接近度标记（7 个）
    out["fomc_imminent"] = (out["days_to_fomc"] <= 1).astype(int)      # FOMC today or tomorrow / FOMC 今天或明天
    out["fomc_this_week"] = (out["days_to_fomc"] <= 5).astype(int)     # FOMC within 5 days / FOMC 5 天内
    out["nfp_imminent"] = (out["days_to_nfp"] <= 1).astype(int)        # NFP today or tomorrow / NFP 今天或明天
    out["nfp_this_week"] = (out["days_to_nfp"] <= 5).astype(int)       # NFP within 5 days / NFP 5 天内
    out["any_catalyst_imminent"] = (                                    # Any event imminent / 任何事件临近
        (out["fomc_imminent"] == 1)
        | (out["nfp_imminent"] == 1)
        | (out["is_earnings_season"] == 1)
    ).astype(int)

    # Recent vol spike flags / 近期波动率飙升标记
    out["vix_spiked_3d"] = out["vix_spike"].rolling(3).max().fillna(0).astype(int)  # VIX spiked in last 3 days / 近 3 天 VIX 飙升
    out["big_move_recent_3d"] = (                                                    # >2% move in last 3 days / 近 3 天出现 >2% 波动
        out["abs_close_to_close_gt_2pct"].rolling(3).max().shift(1).fillna(0).astype(int)
    )

    # ══ C. CROSS FEATURES: regime × catalyst (30 features) ══
    # 交叉特征：状态 × 催化事件（30 个）

    # ── VRP × Event (9): High fear premium + event = vol release ──
    # 高恐慌溢价 + 事件 = 波动率释放
    out["vrp_high_X_fomc_imminent"] = out["vrp_high"] * out["fomc_imminent"]
    out["vrp_high_X_nfp_imminent"] = out["vrp_high"] * out["nfp_imminent"]
    out["vrp_high_X_earnings"] = out["vrp_high"] * out["is_earnings_season"]
    out["vrp_high_X_any_catalyst"] = out["vrp_high"] * out["any_catalyst_imminent"]
    out["vrp_extreme_X_fomc_imminent"] = out["vrp_extreme"] * out["fomc_imminent"]
    out["vrp_extreme_X_any_catalyst"] = out["vrp_extreme"] * out["any_catalyst_imminent"]
    out["vrp_pos_X_fomc"] = out["vrp_positive"] * out["fomc_imminent"]
    out["vrp_pos_X_nfp"] = out["vrp_positive"] * out["nfp_imminent"]
    out["vrp_pos_X_earnings"] = out["vrp_positive"] * out["is_earnings_season"]

    # ── High vol × Event (4): Vol momentum + event = continuation ──
    # 高波动率动量 + 事件 = 趋势延续
    out["highvol_X_fomc"] = out["high_vol_regime"] * out["fomc_imminent"]
    out["highvol_X_nfp"] = out["high_vol_regime"] * out["nfp_imminent"]
    out["highvol_X_earnings"] = out["high_vol_regime"] * out["is_earnings_season"]
    out["highvol_X_any_catalyst"] = out["high_vol_regime"] * out["any_catalyst_imminent"]

    # ── Gamma trap (4): Complacency + event = surprise spike ──
    # Gamma 陷阱（4）：自满 + 事件 = 意外飙升
    # When VRP is negative or vol is low, the market is unprepared for shocks.
    # Events into complacency often produce outsized moves.
    # 当 VRP 为负或波动率低时，市场对冲击毫无准备。
    # 自满状态下的事件通常产生超预期波动。
    out["complacent_X_fomc"] = out["vrp_negative"] * out["fomc_imminent"]
    out["complacent_X_nfp"] = out["vrp_negative"] * out["nfp_imminent"]
    out["lowvol_X_fomc"] = out["low_vol_regime"] * out["fomc_imminent"]
    out["lowvol_X_any_catalyst"] = out["low_vol_regime"] * out["any_catalyst_imminent"]

    # ── Momentum × Event (5): Recent vol clustering + event = amplification ──
    # 动量 × 事件（5）：近期波动率集群 + 事件 = 放大效应
    out["vix_spike3d_X_fomc"] = out["vix_spiked_3d"] * out["fomc_imminent"]
    out["vix_spike3d_X_any_catalyst"] = out["vix_spiked_3d"] * out["any_catalyst_imminent"]
    out["big_move_3d_X_fomc"] = out["big_move_recent_3d"] * out["fomc_imminent"]
    out["big_move_3d_X_any_catalyst"] = out["big_move_recent_3d"] * out["any_catalyst_imminent"]
    out["big_move_3d_X_earnings"] = out["big_move_recent_3d"] * out["is_earnings_season"]

    # ── Rate × Event (1): Bond shock + FOMC = double uncertainty ──
    # 利率 × 事件（1）：债券冲击 + FOMC = 双重不确定性
    out["rate_shock_X_fomc"] = out["rate_shock"] * out["fomc_imminent"]

    # ── Continuous interactions (4): VRP × event urgency (not just binary) ──
    # 连续交互（4）：VRP × 事件紧迫度（不仅仅是二值）
    # urgency = 1 / (days_to_event + 1), decays as event approaches.
    # 紧迫度 = 1 / (距事件天数 + 1)，随事件临近而衰减。
    out["vrp_X_fomc_urgency"] = out["vrp_20d"] * (1 / (out["days_to_fomc"] + 1))
    out["vrp_X_nfp_urgency"] = out["vrp_20d"] * (1 / (out["days_to_nfp"] + 1))
    out["vrp_zscore_X_any_catalyst"] = out["vrp_20d_zscore"] * out["any_catalyst_imminent"]
    out["vrp_zscore_X_fomc"] = out["vrp_20d_zscore"] * out["fomc_imminent"]

    # ── Yield curve × Event (2): Recession fear + FOMC = policy panic ──
    # 收益率曲线 × 事件（2）：衰退恐慌 + FOMC = 政策恐慌
    out["curve_inverted_X_fomc"] = out["yield_curve_inverted"] * out["fomc_imminent"]
    out["curve_inverted_X_any_catalyst"] = out["yield_curve_inverted"] * out["any_catalyst_imminent"]

    return out
