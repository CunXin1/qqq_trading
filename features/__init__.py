"""Feature engineering pipeline.
特征工程管线。

This package transforms raw daily metrics + external market data into
model-ready features. The pipeline is organized in four layers:
本包将原始日频指标 + 外部市场数据转换为模型可用的特征。
管线分为四层：

Layer 1 — base.py:         53 features from QQQ price/volume only.
第 1 层 — base.py:          仅基于 QQQ 价格/成交量的 53 个特征。
    Lagged returns, rolling vol, drawdown/runup, volume ratios,
    calendar, technical indicators, gap features.
    滞后收益率、滚动波动率、回撤/反弹、成交量比率、
    日历特征、技术指标、跳空特征。

Layer 2 — external.py:     26 features from VIX/VVIX/rates/events.
第 2 层 — external.py:      基于 VIX/VVIX/利率/事件的 26 个特征。
    VRP (Volatility Risk Premium), VIX dynamics, VVIX ratio,
    yield curve, rate shocks, FOMC/NFP/earnings season flags.
    VRP（波动率风险溢价）、VIX 动态、VVIX 比率、
    收益率曲线、利率冲击、FOMC/NFP/财报季标记。

Layer 3 — interactions.py: 43 cross-interaction features.
第 3 层 — interactions.py:   43 个交叉交互特征。
    Regime flags × catalyst proximity: captures non-linear effects
    like "high VRP + FOMC imminent = extreme vol likely".
    状态标记 × 催化事件接近度：捕捉非线性效应，
    如"高 VRP + FOMC 临近 = 极端波动率可能性大"。

Layer 4 — path.py:         30 path-dependency features (optional).
第 4 层 — path.py:           30 个路径依赖特征（可选）。
    Trend smoothness (R²), fractal efficiency, Hurst exponent,
    choppiness index — not used in production models by default.
    趋势平滑度（R²）、分形效率、Hurst 指数、
    震荡指数——默认不在生产模型中使用。

Registry — registry.py:    Canonical feature name lists.
注册表 — registry.py:       标准特征名称列表。
    Single source of truth for which features go into each model.
    每个模型使用哪些特征的唯一事实来源。

Total features in production: 53 + 26 + 43 = 122 (base + external + interactions).
生产环境总特征数：53 + 26 + 43 = 122（基础 + 外部 + 交互）。
"""
from features.base import engineer_base_features
from features.external import engineer_all_external
from features.interactions import build_interaction_features
from features.path import build_path_features
from features.registry import get_full_features, get_base_features
