"""
Phase 1 (research): Build daily metrics from 1-min QQQ data.
第一阶段（研究）：从 QQQ 1 分钟数据构建每日指标。

Reads the full 1-minute OHLCV parquet file and computes daily aggregated
metrics, then prints comprehensive summary statistics covering:
读取完整的 1 分钟 OHLCV parquet 文件，计算每日聚合指标，并输出全面的汇总统计：

  - Return statistics (close-to-close, open-to-close, intraday range, gap).
    收益率统计（收盘到收盘、开盘到收盘、日内振幅、跳空缺口）。
  - Large-move day counts & percentages at 1%/2%/3%/5% thresholds.
    大幅波动日计数和百分比（1%/2%/3%/5% 阈值）。
  - Max drawdown / runup statistics.
    最大回撤/最大涨幅统计。
  - Pre-market session metrics (return and range, where available).
    盘前交易时段指标（收益率和振幅，如有数据）。

Output: daily_metrics.parquet saved to OUTPUT_DIR.
输出：daily_metrics.parquet 保存至 OUTPUT_DIR。
"""
import sys
from pathlib import Path

# Ensure the package is importable when running standalone.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.paths import OUTPUT_DIR, DATA_DIR, PROJECT_ROOT
from data.daily_metrics import (
    load_1min_data,
    build_daily_metrics,
)


def print_summary(daily):
    print("\n" + "=" * 70)
    print("QQQ DAILY METRICS SUMMARY")
    print("=" * 70)
    print(f"Total trading days: {len(daily)}")
    print(f"Date range: {daily.index[0].date()} to {daily.index[-1].date()}")

    print("\n--- Return Statistics ---")
    for col in ["close_to_close_ret", "open_to_close_ret", "intraday_range", "gap_return"]:
        s = daily[col].dropna()
        print(f"\n{col}:")
        print(f"  Mean:   {s.mean():.4%}")
        print(f"  Std:    {s.std():.4%}")
        print(f"  Min:    {s.min():.4%}")
        print(f"  Max:    {s.max():.4%}")
        print(f"  Skew:   {s.skew():.4f}")
        print(f"  Kurt:   {s.kurtosis():.4f}")

    print("\n--- Large Move Day Counts ---")
    header = f"{'Metric':<30} {'> 1%':>8} {'> 2%':>8} {'> 3%':>8} {'> 5%':>8}"
    print(header)
    print("-" * len(header))
    for metric in ["abs_close_to_close", "abs_open_to_close", "intraday_range"]:
        counts = []
        for thresh in [1, 2, 3, 5]:
            col = f"{metric}_gt_{thresh}pct"
            counts.append(daily[col].sum())
        print(f"{metric:<30} {counts[0]:>8.0f} {counts[1]:>8.0f} {counts[2]:>8.0f} {counts[3]:>8.0f}")

    # Percentage
    print("\n--- Large Move Day Percentages ---")
    print(header)
    print("-" * len(header))
    for metric in ["abs_close_to_close", "abs_open_to_close", "intraday_range"]:
        pcts = []
        for thresh in [1, 2, 3, 5]:
            col = f"{metric}_gt_{thresh}pct"
            pcts.append(daily[col].mean() * 100)
        print(f"{metric:<30} {pcts[0]:>7.1f}% {pcts[1]:>7.1f}% {pcts[2]:>7.1f}% {pcts[3]:>7.1f}%")

    print("\n--- Max Drawdown / Runup ---")
    for col in ["max_drawdown", "max_runup"]:
        s = daily[col].dropna()
        print(f"\n{col}:")
        print(f"  Mean:  {s.mean():.4%}")
        print(f"  Worst: {s.min() if 'drawdown' in col else s.max():.4%}")

    print("\n--- Pre-market (where available) ---")
    pre_avail = daily["premarket_ret"].dropna()
    print(f"  Days with pre-market data: {len(pre_avail)}")
    if len(pre_avail) > 0:
        print(f"  Mean pre-market return: {pre_avail.mean():.4%}")
        print(f"  Mean pre-market range:  {daily['premarket_range'].dropna().mean():.4%}")


def main():
    parquet_path = DATA_DIR / "QQQ_1min_adjusted.parquet"
    print("Loading adjusted 1-min data...")
    df = load_1min_data(parquet_path)
    print(f"  Loaded {len(df):,} bars, {df.index.date[0]} to {df.index.date[-1]}")

    print("Computing daily metrics...")
    daily = build_daily_metrics(df)

    # Save
    out_path = OUTPUT_DIR / "daily_metrics.parquet"
    daily.to_parquet(out_path)
    print(f"\nSaved daily metrics to {out_path}")
    print(f"Shape: {daily.shape}")

    print_summary(daily)

    # Also save top 20 largest move days for quick reference
    print("\n--- Top 20 Largest |Close-to-Close| Move Days ---")
    top = daily.nlargest(20, "abs_close_to_close")[
        ["close_to_close_ret", "open_to_close_ret", "intraday_range", "gap_return", "volume_regular"]
    ]
    print(top.to_string())

    return daily


if __name__ == "__main__":
    daily = main()
