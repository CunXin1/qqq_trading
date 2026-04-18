"""Canonical feature lists — single source of truth.
标准特征名称列表——唯一事实来源。

Every script that needs a feature list imports from here. This ensures
consistency between training (cli/pipeline.py) and evaluation (eval/).
所有需要特征列表的脚本都从这里导入。这确保了训练（cli/pipeline.py）
和评估（eval/）之间的一致性。

Feature counts / 特征数量:
  get_base_features():              53  (QQQ price/volume only)
                                         （仅 QQQ 价格/成交量）
  get_refined_external_features():  26  (VIX/VVIX/rates/events)
                                         （VIX/VVIX/利率/事件）
  get_interaction_features():       43  (regime × catalyst crosses)
                                         （状态 × 催化事件交叉）
  get_path_features():              30  (smoothness/Hurst, optional)
                                         （平滑度/Hurst，可选）
  get_0dte_premarket_features():     3  (same-day premarket, optional)
                                         （当天盘前，可选）

  get_full_features():             122  (base + external + interactions, default)
                                         （基础 + 外部 + 交互，默认组合）

Model → Feature mapping / 模型 → 特征映射:
  All three production models (range_0dte, otc_0dte, c2c_1dte) use
  the same 122-feature set from get_full_features(include_interactions=True).
  三个生产模型都使用 get_full_features(include_interactions=True) 的 122 特征集。
  The feature list is saved alongside each model as a sidecar CSV file
  (e.g. range_0dte_2pct_model.csv) for inference-time validation.
  特征列表作为伴随 CSV 文件与每个模型一起保存（如 range_0dte_2pct_model.csv），
  用于推理时验证。
"""
from __future__ import annotations


def get_base_features(include_premarket: bool = False) -> list[str]:
    """53 base features from QQQ price/volume only.
    仅基于 QQQ 价格/成交量的 53 个基础特征。

    These features are available without any external data sources.
    这些特征无需任何外部数据源即可获得。

    Args:
        include_premarket: If True, add 3 same-day premarket features (56 total).
                           若为 True，添加 3 个当天盘前特征（共 56 个）。
    """
    features = []

    # Lagged returns (15): ret/abs_ret/range at lag 1-5
    # 滞后收益率 (15)：收益率/绝对收益率/振幅 滞后 1-5 天
    for lag in range(1, 6):
        features += [f"ret_lag{lag}", f"abs_ret_lag{lag}", f"range_lag{lag}"]

    # Rolling stats (12): mean abs return, std, realized vol at 5/10/20/60d
    # 滚动统计 (12)：5/10/20/60 天窗口的均值绝对收益率、标准差、已实现波动率
    for w in [5, 10, 20, 60]:
        features += [f"mean_abs_ret_{w}d", f"std_ret_{w}d", f"realized_vol_{w}d"]

    # Vol ratios (2): short-term / long-term vol regime indicator
    # 波动率比率 (2)：短期/长期波动率状态指标
    features += ["vol_ratio_5_60", "vol_ratio_10_60"]

    # Drawdown/runup lags (6): max intraday DD/RU at lag 1-3
    # 回撤/反弹滞后 (6)：滞后 1-3 天的日内最大回撤/反弹
    for lag in range(1, 4):
        features += [f"max_dd_lag{lag}", f"max_ru_lag{lag}"]

    # Volume (2): relative volume and volume trend
    # 成交量 (2)：相对成交量和成交量趋势
    features += ["vol_ratio_20d", "vol_trend_5_20"]

    # Calendar (6): day-of-week, month, month edges, OPEX, days since big move
    # 日历 (6)：星期几、月份、月初/月末、OPEX 周、距大波动天数
    features += [
        "dow", "month", "is_month_start", "is_month_end",
        "is_opex_week", "days_since_2pct_move",
    ]

    # Technical (8): MA distances, RSI, high/low proximity
    # 技术指标 (8)：MA 距离、RSI、高低点接近度
    for w in [20, 50, 200]:
        features += [f"dist_from_ma{w}"]
    features += ["rsi_14"]
    for w in [20, 50]:
        features += [f"proximity_{w}d_high", f"proximity_{w}d_low"]

    # Gap (2): yesterday's overnight gap
    # 跳空 (2)：昨日隔夜跳空
    features += ["gap_ret_lag1", "abs_gap_lag1"]

    if include_premarket:
        features += ["premarket_ret_today", "premarket_range_today", "premarket_vol_ratio"]

    return features


def get_refined_external_features() -> list[str]:
    """26 refined external features — VRP, VIX dynamics, rates, events.
    26 个精炼外部特征——VRP、VIX 动态、利率、事件。

    Organized by data source / 按数据源组织:
      VRP (5):            Volatility risk premium at multiple windows + z-score.
                          多窗口波动率风险溢价 + z-score。
      VIX dynamics (4):   VIX changes and spike detection (not raw level).
                          VIX 变化和飙升检测（非原始水平）。
      VVIX (2):           Vol-of-vol features.
                          波动率的波动率特征。
      Interest rates (5): Yield curve and rate shock features.
                          收益率曲线和利率冲击特征。
      Event calendar (10): FOMC, NFP, earnings flags + proximity countdowns.
                           FOMC、NFP、财报季标记 + 接近度倒计时。
    """
    return [
        # VRP (5) / 波动率风险溢价
        "vrp_20d", "vrp_10d", "vrp_5d", "vrp_20d_zscore", "vrp_20d_change_5d",
        # VIX dynamics (4) / VIX 动态
        "vix_pct_change_1d", "vix_pct_change_5d", "vix_range_1d", "vix_spike",
        # VVIX (2) / VIX 的波动率
        "vvix_vix_ratio", "vvix_change_1d",
        # Interest rates (5) / 利率
        "yield_curve_slope", "yield_curve_inverted",
        "yield_10y_change_1d", "yield_10y_vol_20d", "rate_shock",
        # Event calendar (10) / 事件日历
        "is_fomc_day", "is_fomc_eve", "days_to_fomc", "fomc_week",
        "is_nfp_day", "is_nfp_eve", "days_to_nfp",
        "is_macro_event_day", "is_macro_event_eve",
        "is_earnings_season",
    ]


def get_interaction_features() -> list[str]:
    """43 interaction features — regime flags + cross signals.
    43 个交互特征——状态标记 + 交叉信号。

    Organized by interaction type / 按交互类型组织:
      Regime flags (6):       VRP/vol discrete state buckets.
                              VRP/波动率离散状态分桶。
      Catalyst flags (7):     Event proximity + recent vol momentum.
                              事件接近度 + 近期波动率动量。
      VRP × Event (9):        High fear + event = vol release.
                              高恐慌 + 事件 = 波动率释放。
      High vol × Event (4):   Vol momentum + event = continuation.
                              波动率动量 + 事件 = 延续。
      Gamma trap (4):         Complacency + event = surprise spike.
                              自满 + 事件 = 意外飙升。
      Momentum × Event (5):   Recent spikes + event = vol clustering.
                              近期飙升 + 事件 = 波动率集群。
      Rate × Event (1):       Bond shock + FOMC.
                              债券冲击 + FOMC。
      Continuous (4):         VRP × urgency (smooth, not binary).
                              VRP × 紧迫度（连续值，非二值）。
      Yield curve × Event (2): Inversion + event = policy panic.
                               倒挂 + 事件 = 政策恐慌。
    """
    return [
        # Regime flags (6) / 状态标记
        "vrp_high", "vrp_extreme", "vrp_positive", "vrp_negative",
        "high_vol_regime", "low_vol_regime",
        # Catalyst flags (7) / 催化事件标记
        "fomc_imminent", "fomc_this_week", "nfp_imminent", "nfp_this_week",
        "any_catalyst_imminent", "vix_spiked_3d", "big_move_recent_3d",
        # VRP × Event crosses (9) / VRP × 事件交叉
        "vrp_high_X_fomc_imminent", "vrp_high_X_nfp_imminent",
        "vrp_high_X_earnings", "vrp_high_X_any_catalyst",
        "vrp_extreme_X_fomc_imminent", "vrp_extreme_X_any_catalyst",
        "vrp_pos_X_fomc", "vrp_pos_X_nfp", "vrp_pos_X_earnings",
        # High vol × Event (4) / 高波动率 × 事件
        "highvol_X_fomc", "highvol_X_nfp", "highvol_X_earnings", "highvol_X_any_catalyst",
        # Gamma trap (4) / Gamma 陷阱
        "complacent_X_fomc", "complacent_X_nfp",
        "lowvol_X_fomc", "lowvol_X_any_catalyst",
        # Momentum × Event (5) / 动量 × 事件
        "vix_spike3d_X_fomc", "vix_spike3d_X_any_catalyst",
        "big_move_3d_X_fomc", "big_move_3d_X_any_catalyst", "big_move_3d_X_earnings",
        # Rate × Event (1) / 利率 × 事件
        "rate_shock_X_fomc",
        # Continuous interactions (4) / 连续交互
        "vrp_X_fomc_urgency", "vrp_X_nfp_urgency",
        "vrp_zscore_X_any_catalyst", "vrp_zscore_X_fomc",
        # Yield curve × Event (2) / 收益率曲线 × 事件
        "curve_inverted_X_fomc", "curve_inverted_X_any_catalyst",
    ]


def get_path_features() -> list[str]:
    """30 path dependency / smoothness features (optional, not used in production).
    30 个路径依赖 / 平滑度特征（可选，生产环境默认不使用）。

    Computed at two windows (63d ≈ 1 quarter, 126d ≈ 6 months).
    在两个窗口（63 天 ≈ 1 季度、126 天 ≈ 6 个月）上计算。
    Each window produces 7 core features + 8 cross features = 15 × 2 = 30.
    每个窗口产出 7 个核心特征 + 8 个交叉特征 = 15 × 2 = 30。
    """
    features = []
    for w in ["63d", "126d"]:
        features += [
            # Core path features (7) / 核心路径特征
            f"trend_r2_{w}", f"fractal_eff_{w}", f"choppiness_{w}",
            f"hurst_{w}", f"trend_strength_{w}", f"up_day_ratio_{w}",
            f"max_dd_window_{w}",
            # Smoothness regime flags (3) / 平滑度状态标记
            f"smooth_trend_{w}", f"very_smooth_{w}", f"choppy_{w}",
            # Smoothness × catalyst/VRP/vol crosses (5) / 平滑度 × 催化事件/VRP/波动率交叉
            f"smooth_{w}_X_catalyst", f"smooth_{w}_X_fomc",
            f"smooth_{w}_X_vrp_neg", f"smooth_{w}_X_vrp_neg_X_catalyst",
            f"smooth_{w}_X_lowvol",
        ]
    return features


def get_0dte_premarket_features() -> list[str]:
    """3 additional premarket features for 0DTE models.
    0DTE 模型的 3 个额外盘前特征。

    These are same-day features available before 9:30 open.
    Not shifted because they represent today's pre-market conditions.
    这些是 9:30 开盘前可用的当天特征。
    不做 shift 因为它们代表今天的盘前状态。
    """
    return ["premarket_ret_today", "premarket_range_today", "premarket_vol_ratio"]


def get_full_features(
    include_interactions: bool = True,
    include_path: bool = False,
    include_premarket: bool = False,
) -> list[str]:
    """Composite feature list used by training and evaluation.
    训练和评估使用的组合特征列表。

    Default configuration (122 features) / 默认配置（122 个特征）:
      53 base + 26 external + 43 interactions = 122

    Optional extensions / 可选扩展:
      + 30 path features  (include_path=True)      → 152
      + 3 premarket       (include_premarket=True)  → 125 or 155

    Args:
        include_interactions: Include Layer 3 interaction features (default True).
                              包含第 3 层交互特征（默认 True）。
        include_path:         Include Layer 4 path features (default False).
                              包含第 4 层路径特征（默认 False）。
        include_premarket:    Include 0DTE premarket features (default False).
                              包含 0DTE 盘前特征（默认 False）。

    Returns:
        Ordered list of feature column names.
        有序的特征列名列表。
    """
    feats = get_base_features()
    feats += get_refined_external_features()
    if include_interactions:
        feats += get_interaction_features()
    if include_path:
        feats += get_path_features()
    if include_premarket:
        feats += get_0dte_premarket_features()
    return feats
