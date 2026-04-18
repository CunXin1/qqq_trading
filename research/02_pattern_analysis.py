"""
Phase 2 (research): Pattern Discovery & Visualization.
第二阶段（研究）：模式发现与可视化。

Performs 8 pattern analyses on QQQ daily metrics data, each generating a
chart saved to CHART_DIR. Finds statistical patterns that precede large moves.
对 QQQ 每日指标数据进行 8 项模式分析，每项生成图表保存至 CHART_DIR。
寻找大幅波动前的统计模式。

Analyses / 分析项:
  1. Yearly frequency — large-move day frequency by year.
     年度频率 — 按年统计大幅波动日频率。
  2. Calendar effects — day-of-week, month, OPEX week, month-start/end.
     日历效应 — 星期、月份、期权到期周、月初/月末。
  3. Volatility clustering — ACF of absolute returns, conditional probabilities.
     波动率聚集 — 绝对收益率自相关函数、条件概率。
  4. Pre-market signal — correlation between pre-market range/volume and
     regular-session large moves.
     盘前信号 — 盘前振幅/成交量与常规交易时段大幅波动的相关性。
  5. Gap analysis — overnight gap size/direction vs intraday move.
     跳空分析 — 隔夜跳空幅度/方向与日内波动的关系。
  6. Volume signal — volume ratio quintiles vs same-day/next-day large moves.
     成交量信号 — 成交量比率五分位与当日/次日大幅波动的关系。
  7. Regime analysis — volatility regime classification and transition matrix.
     市场状态分析 — 波动率状态分类及转移矩阵。
  8. Consecutive patterns — streak length and mean-reversion effects.
     连续模式 — 连涨/连跌天数及均值回归效应。
"""
import sys
from pathlib import Path

# Ensure the package is importable when running standalone.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from utils.paths import OUTPUT_DIR, CHART_DIR
from utils.plotting import setup_matplotlib

setup_matplotlib()


def load_daily():
    df = pd.read_parquet(OUTPUT_DIR / "daily_metrics.parquet")
    df.index = pd.to_datetime(df.index)
    return df


# -- 1. Frequency over time -------------------------------------------------

def plot_yearly_frequency(daily):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("QQQ Large Move Days - Yearly Frequency", fontsize=14)

    for ax, thresh in zip(axes.flat, [1, 2, 3, 5]):
        col = f"abs_close_to_close_gt_{thresh}pct"
        yearly = daily.groupby(daily.index.year)[col].agg(["sum", "count"])
        yearly["pct"] = yearly["sum"] / yearly["count"] * 100
        ax.bar(yearly.index, yearly["pct"], color="steelblue", alpha=0.8)
        ax.set_title(f"|Close-to-Close| > {thresh}%")
        ax.set_ylabel("% of Trading Days")
        ax.set_xlabel("Year")
        ax.axhline(yearly["pct"].mean(), color="red", ls="--", alpha=0.6,
                    label=f"Avg: {yearly['pct'].mean():.1f}%")
        ax.legend()

    plt.tight_layout()
    plt.savefig(CHART_DIR / "01_yearly_frequency.png")
    plt.close()
    print("  Saved 01_yearly_frequency.png")


# -- 2. Calendar effects ----------------------------------------------------

def analyze_calendar_effects(daily):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Calendar Effects on Large Moves (|C2C| > 2%)", fontsize=14)

    target = daily["abs_close_to_close_gt_2pct"]

    # Day of week
    ax = axes[0, 0]
    dow = daily.groupby(daily.index.dayofweek)[target.name].mean() * 100
    dow.index = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    ax.bar(dow.index, dow.values, color="steelblue")
    ax.set_title("Day of Week")
    ax.set_ylabel("% of Days with |C2C| > 2%")

    # Chi-squared test
    observed = daily.groupby(daily.index.dayofweek)[target.name].sum()
    expected_freq = daily.groupby(daily.index.dayofweek)[target.name].count() * target.mean()
    chi2, pval = stats.chisquare(observed, expected_freq)
    ax.set_xlabel(f"Chi2 p-value: {pval:.4f}")

    # Month
    ax = axes[0, 1]
    monthly = daily.groupby(daily.index.month)[target.name].mean() * 100
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.bar(month_labels, monthly.values, color="coral")
    ax.set_title("Month of Year")
    ax.set_ylabel("% of Days with |C2C| > 2%")
    ax.tick_params(axis="x", rotation=45)

    # Options expiration (3rd Friday)
    ax = axes[1, 0]
    daily_copy = daily.copy()
    daily_copy["is_opex"] = False
    for idx in daily_copy.index:
        if idx.weekday() == 4:  # Friday
            day = idx.day
            if 15 <= day <= 21:
                daily_copy.loc[idx, "is_opex"] = True

    # OpEx week (Mon-Fri of opex week)
    daily_copy["is_opex_week"] = False
    opex_dates = daily_copy[daily_copy["is_opex"]].index
    for opex_date in opex_dates:
        week_start = opex_date - pd.Timedelta(days=opex_date.weekday())
        week_end = week_start + pd.Timedelta(days=4)
        mask = (daily_copy.index >= week_start) & (daily_copy.index <= week_end)
        daily_copy.loc[mask, "is_opex_week"] = True

    opex_rate = daily_copy.groupby("is_opex_week")[target.name].mean() * 100
    ax.bar(["Non-OpEx Week", "OpEx Week"], opex_rate.values, color=["gray", "orange"])
    ax.set_title("Options Expiration Week Effect")
    ax.set_ylabel("% of Days with |C2C| > 2%")

    # First/last day of month
    ax = axes[1, 1]
    daily_copy["is_month_start"] = daily_copy.index.is_month_start | (
        daily_copy.index == daily_copy.groupby(daily_copy.index.to_period("M")).transform("first").index
    )
    daily_copy["month_period"] = daily_copy.index.to_period("M")
    first_days = daily_copy.groupby("month_period").head(1).index
    last_days = daily_copy.groupby("month_period").tail(1).index
    daily_copy["position"] = "Mid-Month"
    daily_copy.loc[daily_copy.index.isin(first_days), "position"] = "First Day"
    daily_copy.loc[daily_copy.index.isin(last_days), "position"] = "Last Day"

    pos_rate = daily_copy.groupby("position")[target.name].mean() * 100
    order = ["First Day", "Mid-Month", "Last Day"]
    ax.bar(order, [pos_rate[k] for k in order], color=["green", "gray", "red"])
    ax.set_title("Month Position Effect")
    ax.set_ylabel("% of Days with |C2C| > 2%")

    plt.tight_layout()
    plt.savefig(CHART_DIR / "02_calendar_effects.png")
    plt.close()
    print("  Saved 02_calendar_effects.png")

    return daily_copy


# -- 3. Volatility clustering -----------------------------------------------

def analyze_vol_clustering(daily):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Volatility Clustering Analysis", fontsize=14)

    abs_ret = daily["abs_close_to_close"].dropna()

    # ACF of absolute returns
    ax = axes[0, 0]
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(abs_ret, lags=40, ax=ax, alpha=0.05)
    ax.set_title("ACF of |Close-to-Close Returns|")

    # Conditional probability
    ax = axes[0, 1]
    thresholds = [1, 2, 3, 5]
    for thresh in thresholds:
        col = f"abs_close_to_close_gt_{thresh}pct"
        daily[f"prev_{col}"] = daily[col].shift(1)

    cond_probs = []
    uncond_probs = []
    for thresh in thresholds:
        col = f"abs_close_to_close_gt_{thresh}pct"
        prev_col = f"prev_{col}"
        valid = daily.dropna(subset=[prev_col])
        uncond = valid[col].mean()
        cond = valid.loc[valid[prev_col] == True, col].mean()
        cond_probs.append(cond * 100)
        uncond_probs.append(uncond * 100)

    x = np.arange(len(thresholds))
    width = 0.35
    ax.bar(x - width / 2, uncond_probs, width, label="Unconditional", color="steelblue")
    ax.bar(x + width / 2, cond_probs, width, label="After Large Move", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels([f">{t}%" for t in thresholds])
    ax.set_ylabel("Probability (%)")
    ax.set_title("P(Large Move Today | Large Move Yesterday)")
    ax.legend()

    # P(large move | large move in past 5 days)
    ax = axes[1, 0]
    cond5_probs = []
    for thresh in thresholds:
        col = f"abs_close_to_close_gt_{thresh}pct"
        rolling5 = daily[col].rolling(5).sum().shift(1)
        valid = daily[rolling5.notna()].copy()
        valid["had_recent"] = rolling5[rolling5.notna()] > 0
        cond5 = valid.loc[valid["had_recent"], col].mean() * 100
        cond5_probs.append(cond5)

    ax.bar(x - width / 2, uncond_probs, width, label="Unconditional", color="steelblue")
    ax.bar(x + width / 2, cond5_probs, width, label="After Move in Past 5d", color="orange")
    ax.set_xticks(x)
    ax.set_xticklabels([f">{t}%" for t in thresholds])
    ax.set_ylabel("Probability (%)")
    ax.set_title("P(Large Move Today | Large Move in Past 5 Days)")
    ax.legend()

    # Rolling realized vol
    ax = axes[1, 1]
    rv20 = daily["close_to_close_ret"].rolling(20).std() * np.sqrt(252) * 100
    ax.plot(daily.index, rv20, color="steelblue", alpha=0.7, linewidth=0.8)
    ax.set_title("20-Day Rolling Annualized Volatility (%)")
    ax.set_ylabel("Volatility (%)")
    ax.axhline(rv20.mean(), color="red", ls="--", alpha=0.5, label=f"Mean: {rv20.mean():.1f}%")
    ax.legend()

    plt.tight_layout()
    plt.savefig(CHART_DIR / "03_vol_clustering.png")
    plt.close()
    print("  Saved 03_vol_clustering.png")


# -- 4. Pre-market signal ---------------------------------------------------

def analyze_premarket_signal(daily):
    has_pre = daily.dropna(subset=["premarket_ret", "premarket_range"]).copy()
    if len(has_pre) < 100:
        print("  Insufficient pre-market data, skipping.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Pre-Market Signal Analysis", fontsize=14)

    # Pre-market range vs intraday range
    ax = axes[0, 0]
    ax.scatter(has_pre["premarket_range"] * 100, has_pre["intraday_range"] * 100,
               alpha=0.15, s=5, color="steelblue")
    r, p = stats.pearsonr(has_pre["premarket_range"], has_pre["intraday_range"])
    ax.set_xlabel("Pre-Market Range (%)")
    ax.set_ylabel("Regular Session Range (%)")
    ax.set_title(f"Pre-Market Range vs Intraday Range (r={r:.3f}, p={p:.2e})")

    # Pre-market |return| vs |close-to-close|
    ax = axes[0, 1]
    ax.scatter(has_pre["premarket_ret"].abs() * 100, has_pre["abs_close_to_close"] * 100,
               alpha=0.15, s=5, color="coral")
    r2, p2 = stats.pearsonr(has_pre["premarket_ret"].abs(), has_pre["abs_close_to_close"])
    ax.set_xlabel("|Pre-Market Return| (%)")
    ax.set_ylabel("|Close-to-Close Return| (%)")
    ax.set_title(f"|Pre-Market Ret| vs |C2C| (r={r2:.3f}, p={p2:.2e})")

    # Pre-market range quintiles vs large move probability
    ax = axes[1, 0]
    has_pre["pre_range_q"] = pd.qcut(has_pre["premarket_range"], 5, labels=False, duplicates="drop")
    q_prob = has_pre.groupby("pre_range_q")["abs_close_to_close_gt_2pct"].mean() * 100
    ax.bar(range(len(q_prob)), q_prob.values, color="steelblue")
    ax.set_xticks(range(len(q_prob)))
    ax.set_xticklabels([f"Q{i+1}" for i in range(len(q_prob))])
    ax.set_xlabel("Pre-Market Range Quintile (Q1=smallest)")
    ax.set_ylabel("% Days with |C2C| > 2%")
    ax.set_title("Pre-Market Range Quintile -> Large Move Probability")

    # Pre-market volume quintile
    ax = axes[1, 1]
    has_pre["pre_vol_q"] = pd.qcut(has_pre["volume_premarket"], 5, labels=False, duplicates="drop")
    v_prob = has_pre.groupby("pre_vol_q")["abs_close_to_close_gt_2pct"].mean() * 100
    ax.bar(range(len(v_prob)), v_prob.values, color="orange")
    ax.set_xticks(range(len(v_prob)))
    ax.set_xticklabels([f"Q{i+1}" for i in range(len(v_prob))])
    ax.set_xlabel("Pre-Market Volume Quintile (Q1=lowest)")
    ax.set_ylabel("% Days with |C2C| > 2%")
    ax.set_title("Pre-Market Volume Quintile -> Large Move Probability")

    plt.tight_layout()
    plt.savefig(CHART_DIR / "04_premarket_signal.png")
    plt.close()
    print("  Saved 04_premarket_signal.png")


# -- 5. Gap analysis --------------------------------------------------------

def analyze_gaps(daily):
    valid = daily.dropna(subset=["gap_return"]).copy()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Gap Analysis", fontsize=14)

    # Gap distribution
    ax = axes[0, 0]
    ax.hist(valid["gap_return"] * 100, bins=100, color="steelblue", alpha=0.7, edgecolor="none")
    ax.set_xlabel("Gap Return (%)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Overnight Gaps")
    ax.axvline(0, color="black", ls="--", alpha=0.5)

    # |Gap| vs intraday range
    ax = axes[0, 1]
    ax.scatter(valid["gap_return"].abs() * 100, valid["intraday_range"] * 100,
               alpha=0.15, s=5, color="coral")
    r, p = stats.pearsonr(valid["gap_return"].abs(), valid["intraday_range"])
    ax.set_xlabel("|Gap Return| (%)")
    ax.set_ylabel("Intraday Range (%)")
    ax.set_title(f"|Gap| vs Intraday Range (r={r:.3f})")

    # Gap quintile vs large move prob
    ax = axes[1, 0]
    valid["abs_gap_q"] = pd.qcut(valid["gap_return"].abs(), 5, labels=False, duplicates="drop")
    gq_prob = valid.groupby("abs_gap_q")["abs_close_to_close_gt_2pct"].mean() * 100
    ax.bar(range(len(gq_prob)), gq_prob.values, color="steelblue")
    ax.set_xticks(range(len(gq_prob)))
    ax.set_xticklabels([f"Q{i+1}" for i in range(len(gq_prob))])
    ax.set_xlabel("|Gap| Quintile (Q1=smallest)")
    ax.set_ylabel("% Days with |C2C| > 2%")
    ax.set_title("|Gap| Quintile -> Large Move Probability")

    # Gap direction effect
    ax = axes[1, 1]
    valid["gap_dir"] = np.where(valid["gap_return"] > 0.005, "Gap Up >0.5%",
                        np.where(valid["gap_return"] < -0.005, "Gap Down <-0.5%", "Small Gap"))
    dir_prob = valid.groupby("gap_dir")["abs_close_to_close_gt_2pct"].mean() * 100
    order = ["Gap Down <-0.5%", "Small Gap", "Gap Up >0.5%"]
    colors = ["red", "gray", "green"]
    ax.bar(order, [dir_prob.get(k, 0) for k in order], color=colors)
    ax.set_ylabel("% Days with |C2C| > 2%")
    ax.set_title("Gap Direction -> Large Move Probability")

    plt.tight_layout()
    plt.savefig(CHART_DIR / "05_gap_analysis.png")
    plt.close()
    print("  Saved 05_gap_analysis.png")


# -- 6. Volume signal -------------------------------------------------------

def analyze_volume_signal(daily):
    valid = daily.dropna(subset=["volume_regular"]).copy()
    valid["vol_ratio"] = valid["volume_regular"] / valid["volume_regular"].rolling(20).mean()
    valid = valid.dropna(subset=["vol_ratio"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Volume Signal Analysis", fontsize=14)

    # Prior day volume ratio vs next day |C2C|
    ax = axes[0]
    valid["next_large"] = valid["abs_close_to_close_gt_2pct"].shift(-1)
    v = valid.dropna(subset=["next_large"])
    v["vol_ratio_q"] = pd.qcut(v["vol_ratio"], 5, labels=False, duplicates="drop")
    vq_prob = v.groupby("vol_ratio_q")["next_large"].mean() * 100
    ax.bar(range(len(vq_prob)), vq_prob.values, color="steelblue")
    ax.set_xticks(range(len(vq_prob)))
    ax.set_xticklabels([f"Q{i+1}" for i in range(len(vq_prob))])
    ax.set_xlabel("Today's Volume Ratio Quintile (Q1=lowest)")
    ax.set_ylabel("% Next Day with |C2C| > 2%")
    ax.set_title("Today's Volume Ratio -> Tomorrow Large Move Prob")

    # High volume days tend to be large move days themselves
    ax = axes[1]
    same_day = valid.copy()
    same_day["vol_ratio_q"] = pd.qcut(same_day["vol_ratio"], 5, labels=False, duplicates="drop")
    sd_prob = same_day.groupby("vol_ratio_q")["abs_close_to_close_gt_2pct"].mean() * 100
    ax.bar(range(len(sd_prob)), sd_prob.values, color="coral")
    ax.set_xticks(range(len(sd_prob)))
    ax.set_xticklabels([f"Q{i+1}" for i in range(len(sd_prob))])
    ax.set_xlabel("Volume Ratio Quintile (Q1=lowest)")
    ax.set_ylabel("% Days with |C2C| > 2%")
    ax.set_title("Volume Ratio -> Same-Day Large Move")

    plt.tight_layout()
    plt.savefig(CHART_DIR / "06_volume_signal.png")
    plt.close()
    print("  Saved 06_volume_signal.png")


# -- 7. Regime analysis -----------------------------------------------------

def analyze_regimes(daily):
    valid = daily.dropna(subset=["close_to_close_ret"]).copy()
    valid["rv20"] = valid["close_to_close_ret"].rolling(20).std() * np.sqrt(252)
    valid = valid.dropna(subset=["rv20"])

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Volatility Regime Analysis", fontsize=14)

    # Regime definition
    vol_median = valid["rv20"].median()
    vol_q75 = valid["rv20"].quantile(0.75)
    valid["regime"] = np.where(valid["rv20"] > vol_q75, "High Vol",
                      np.where(valid["rv20"] > vol_median, "Med Vol", "Low Vol"))

    # Large move frequency by regime
    ax = axes[0, 0]
    for thresh in [1, 2, 3, 5]:
        col = f"abs_close_to_close_gt_{thresh}pct"
        regime_prob = valid.groupby("regime")[col].mean() * 100
        order = ["Low Vol", "Med Vol", "High Vol"]
        ax.plot(order, [regime_prob.get(k, 0) for k in order], "o-", label=f">{thresh}%")
    ax.set_ylabel("% of Days")
    ax.set_title("Large Move Frequency by Vol Regime")
    ax.legend()

    # Rolling vol with regime coloring
    ax = axes[0, 1]
    colors = {"Low Vol": "green", "Med Vol": "orange", "High Vol": "red"}
    for regime, color in colors.items():
        mask = valid["regime"] == regime
        ax.scatter(valid.index[mask], valid["rv20"][mask] * 100,
                   s=1, alpha=0.4, color=color, label=regime)
    ax.set_title("20-Day Realized Vol with Regime Labels")
    ax.set_ylabel("Annualized Vol (%)")
    ax.legend(markerscale=5)

    # Transition matrix
    ax = axes[1, 0]
    valid["next_regime"] = valid["regime"].shift(-1)
    trans = pd.crosstab(valid["regime"], valid["next_regime"], normalize="index")
    order = ["Low Vol", "Med Vol", "High Vol"]
    trans = trans.reindex(index=order, columns=order, fill_value=0)
    sns.heatmap(trans, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax)
    ax.set_title("Regime Transition Matrix (row -> column)")

    # Days in each regime
    ax = axes[1, 1]
    regime_counts = valid["regime"].value_counts()
    ax.pie([regime_counts.get(k, 0) for k in order], labels=order,
           colors=["green", "orange", "red"], autopct="%1.1f%%")
    ax.set_title("Time Spent in Each Regime")

    plt.tight_layout()
    plt.savefig(CHART_DIR / "07_regime_analysis.png")
    plt.close()
    print("  Saved 07_regime_analysis.png")


# -- 8. Consecutive patterns ------------------------------------------------

def analyze_consecutive_patterns(daily):
    valid = daily.dropna(subset=["close_to_close_ret"]).copy()
    valid["direction"] = np.sign(valid["close_to_close_ret"])

    # Count consecutive same-direction days
    valid["streak"] = 0
    streak = 0
    prev_dir = 0
    streaks = []
    for i, d in enumerate(valid["direction"]):
        if d == prev_dir and d != 0:
            streak += 1
        else:
            streak = 1
        streaks.append(streak * d)
        prev_dir = d
    valid["streak"] = streaks

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Consecutive Day Patterns", fontsize=14)

    # After N consecutive up/down days, probability of large move next day
    ax = axes[0]
    valid["next_large_2pct"] = valid["abs_close_to_close_gt_2pct"].shift(-1)
    streak_vals = range(-5, 6)
    probs = []
    for s in streak_vals:
        if s == 0:
            probs.append(np.nan)
            continue
        mask = valid["streak"] == s
        if mask.sum() > 10:
            probs.append(valid.loc[mask, "next_large_2pct"].mean() * 100)
        else:
            probs.append(np.nan)

    colors_bar = ["red" if s < 0 else "green" if s > 0 else "gray" for s in streak_vals]
    ax.bar(streak_vals, probs, color=colors_bar, alpha=0.7)
    ax.set_xlabel("Consecutive Day Streak (negative=down, positive=up)")
    ax.set_ylabel("% Next Day |C2C| > 2%")
    ax.set_title("Streak Length -> Next Day Large Move Probability")
    ax.axhline(valid["abs_close_to_close_gt_2pct"].mean() * 100, color="black",
               ls="--", alpha=0.5, label="Unconditional")
    ax.legend()

    # Mean reversion: after large move, what happens next day?
    ax = axes[1]
    valid["prev_ret"] = valid["close_to_close_ret"].shift(1)
    valid["prev_ret_q"] = pd.qcut(valid["prev_ret"].dropna(), 10, labels=False, duplicates="drop")

    valid_q = valid.dropna(subset=["prev_ret_q"])
    mean_next = valid_q.groupby("prev_ret_q")["close_to_close_ret"].mean() * 100
    ax.bar(range(len(mean_next)), mean_next.values,
           color=["red" if v < 0 else "green" for v in mean_next.values], alpha=0.7)
    ax.set_xticks(range(len(mean_next)))
    ax.set_xticklabels([f"D{i+1}" for i in range(len(mean_next))])
    ax.set_xlabel("Previous Day Return Decile (D1=worst, D10=best)")
    ax.set_ylabel("Mean Next Day Return (%)")
    ax.set_title("Mean Reversion: Prev Day Return Decile -> Next Day Return")
    ax.axhline(0, color="black", ls="-", alpha=0.3)

    plt.tight_layout()
    plt.savefig(CHART_DIR / "08_consecutive_patterns.png")
    plt.close()
    print("  Saved 08_consecutive_patterns.png")


# -- Print key findings -----------------------------------------------------

def print_key_findings(daily):
    print("\n" + "=" * 70)
    print("KEY PATTERN FINDINGS")
    print("=" * 70)

    target = "abs_close_to_close_gt_2pct"
    uncond = daily[target].mean()
    print(f"\nUnconditional P(|C2C| > 2%) = {uncond:.1%}")

    # Volatility clustering
    prev = daily[target].shift(1)
    valid = daily[prev.notna()]
    cond = valid.loc[prev.dropna() == True, target].mean()
    print(f"P(|C2C| > 2% | yesterday was >2%) = {cond:.1%}  (vs {uncond:.1%} unconditional)")
    print(f"  -> {cond/uncond:.1f}x more likely after a large move day")

    # Pre-market
    has_pre = daily.dropna(subset=["premarket_range"]).copy()
    if len(has_pre) > 100:
        pre_q5 = has_pre["premarket_range"].quantile(0.8)
        high_pre = has_pre[has_pre["premarket_range"] > pre_q5]
        low_pre = has_pre[has_pre["premarket_range"] <= has_pre["premarket_range"].quantile(0.2)]
        print(f"\nHigh pre-market range (top 20%): P(|C2C|>2%) = {high_pre[target].mean():.1%}")
        print(f"Low pre-market range (bottom 20%): P(|C2C|>2%) = {low_pre[target].mean():.1%}")

    # Gap
    valid_gap = daily.dropna(subset=["gap_return"]).copy()
    big_gap = valid_gap[valid_gap["gap_return"].abs() > 0.01]
    small_gap = valid_gap[valid_gap["gap_return"].abs() < 0.002]
    print(f"\nBig gap (|gap|>1%): P(|C2C|>2%) = {big_gap[target].mean():.1%}")
    print(f"Small gap (|gap|<0.2%): P(|C2C|>2%) = {small_gap[target].mean():.1%}")

    # Day of week
    dow_rates = daily.groupby(daily.index.dayofweek)[target].mean()
    best_dow = dow_rates.idxmax()
    worst_dow = dow_rates.idxmin()
    dow_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
    print(f"\nHighest large-move day: {dow_names[best_dow]} ({dow_rates[best_dow]:.1%})")
    print(f"Lowest large-move day: {dow_names[worst_dow]} ({dow_rates[worst_dow]:.1%})")


def main():
    print("Loading daily metrics...")
    daily = load_daily()

    print("\n1. Plotting yearly frequency...")
    plot_yearly_frequency(daily)

    print("2. Analyzing calendar effects...")
    analyze_calendar_effects(daily)

    print("3. Analyzing volatility clustering...")
    analyze_vol_clustering(daily)

    print("4. Analyzing pre-market signal...")
    analyze_premarket_signal(daily)

    print("5. Analyzing gaps...")
    analyze_gaps(daily)

    print("6. Analyzing volume signal...")
    analyze_volume_signal(daily)

    print("7. Analyzing regimes...")
    analyze_regimes(daily)

    print("8. Analyzing consecutive patterns...")
    analyze_consecutive_patterns(daily)

    print_key_findings(daily)

    print(f"\nAll charts saved to {CHART_DIR}")


if __name__ == "__main__":
    main()
