"""
Check QQQ 1-min parquet coverage against NYSE calendar.
检查 QQQ 1 分钟 parquet 数据相对 NYSE 日历的覆盖率。

Compares actual data against the official NYSE trading calendar so that:
将实际数据与 NYSE 官方交易日历对比，确保：
  - Real holidays (Good Friday, MLK, Juneteenth, etc.) are NOT flagged as gaps.
    真实节假日（耶稳受难日、马丁路德金日、六月节等）不被误报为缺口。
  - Half-days (9:30-13:00) are reported with their reduced expected bar count.
    半天交易日（9:30-13:00）按缩短后的预期K线数报告。
  - Only genuinely missing trading sessions are flagged.
    仅标记真正缺失的交易日。

This is a data integrity tool — run it after updating parquets (update_parquet.py)
to verify no trading days were lost or corrupted.
这是一个数据完整性工具——在更新 parquet 文件（update_parquet.py）后运行，
以验证没有交易日丢失或损坏。

Report sections / 报告包含以下部分:
  1. Coverage summary: NYSE days vs data days, half-days, pre/post market presence.
     覆盖率概览：NYSE 交易日 vs 数据天数、半天交易日、盘前盘后覆盖情况。
  2. Missing sessions: Trading days with zero data (data gap).
     缺失交易日：数据为零的交易日（数据缺口）。
  3. Missing pre/post: Full days lacking pre-market or after-hours bars.
     缺失盘前/盘后：缺少盘前或盘后 K 线的全天交易日。
  4. RTH deficit: Days where regular-session bar count is suspiciously low.
     常规时段不足：常规交易时段 K 线数量异常偏少的日子。

Usage / 用法:
    python3 datasets/check_coverage.py                  # 2010 - present / 2010至今
    python3 datasets/check_coverage.py --from 2015-01-01
    python3 datasets/check_coverage.py --file unadjusted
"""

import argparse
from datetime import date
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal

DATA_DIR = Path(__file__).resolve().parent

# Trading session time boundaries (US/Eastern)
# 交易时段时间边界（美东时间）
PREMARKET_START = (4, 0)      # Pre-market opens / 盘前开始
RTH_START = (9, 30)           # Regular Trading Hours start / 常规交易时段开始
RTH_END = (16, 0)             # Regular Trading Hours end / 常规交易时段结束
AFTERHOURS_END = (20, 0)      # After-hours end / 盘后结束


def classify_session(ts: pd.Timestamp) -> str:
    """Classify a 1-min bar timestamp into its trading session.
    将一根 1 分钟 K 线的时间戳分类到对应的交易时段。

    Returns / 返回:
        "pre"   — pre-market (4:00–9:29)  / 盘前
        "rth"   — regular session (9:30–15:59) / 常规交易时段
        "post"  — after-hours (16:00–19:59) / 盘后
        "other" — outside all sessions / 所有时段之外
    """
    hm = (ts.hour, ts.minute)
    if PREMARKET_START <= hm < RTH_START:
        return "pre"
    if RTH_START <= hm < RTH_END:
        return "rth"
    if RTH_END <= hm < AFTERHOURS_END:
        return "post"
    return "other"


def analyze(df: pd.DataFrame) -> pd.DataFrame:
    """Count 1-min bars per session per day.
    统计每天每个交易时段的 1 分钟 K 线数量。

    Returns a DataFrame indexed by date with columns: pre, rth, post, other, total.
    返回以日期为索引的 DataFrame，列包含：pre、rth、post、other、total。
    """
    sessions = df.index.to_series().map(classify_session)
    per = pd.DataFrame({"date": df.index.date, "session": sessions.values})
    counts = per.groupby(["date", "session"]).size().unstack(fill_value=0)
    for c in ("pre", "rth", "post", "other"):
        if c not in counts.columns:
            counts[c] = 0
    counts["total"] = counts[["pre", "rth", "post", "other"]].sum(axis=1)
    return counts.sort_index()


def nyse_schedule(start: date, end: date) -> pd.DataFrame:
    """Return NYSE schedule with expected RTH minutes for each trading day (handles half-days).
    返回 NYSE 交易日历，包含每个交易日预期的常规时段分钟数（包含半天交易日处理）。

    Uses the pandas_market_calendars library which knows about all NYSE holidays
    and early-close days (day before Thanksgiving, Christmas Eve, etc.).
    使用 pandas_market_calendars 库，该库知晓所有 NYSE 节假日
    和提前收盘日（感恩节前一天、平安夜等）。

    Returns / 返回:
        DataFrame with columns: expected_rth (int minutes), early_close (bool).
        DataFrame 包含列：expected_rth（预期分钟数）、early_close（是否提前收盘）。
    """
    nyse = mcal.get_calendar("NYSE")
    sched = nyse.schedule(start_date=start.isoformat(), end_date=end.isoformat())
    # market_open / market_close are in UTC; convert to ET for minute math
    # market_open / market_close 为 UTC 时间；转换为美东时间以计算分钟数
    et_open = sched["market_open"].dt.tz_convert("US/Eastern")
    et_close = sched["market_close"].dt.tz_convert("US/Eastern")
    # Expected RTH bars: close - open in minutes (open bar is the first minute)
    # 预期常规时段 K 线数：收盘 - 开盘的分钟数（开盘那根是第一根）
    expected = ((et_close - et_open).dt.total_seconds() / 60).astype(int)
    out = pd.DataFrame({
        "expected_rth": expected.values,
        "early_close": (expected.values < 390),   # Full day = 390 min (6.5h) / 全天 = 390分钟
    }, index=sched.index.date)
    return out


def report(name: str, path: Path, start: date, end: date):
    """Generate a coverage report for one parquet file.
    为一个 parquet 文件生成覆盖率报告。

    Compares actual bar counts against NYSE expected counts, and reports:
    将实际 K 线数量与 NYSE 预期数量对比，报告以下内容：
    - Fully missing trading days (data gaps) / 完全缺失的交易日（数据缺口）
    - Days missing pre-market or after-hours / 缺少盘前或盘后数据的日子
    - Days with significantly fewer RTH bars than expected / 常规时段 K 线数显著不足的日子

    Args:
        name:  Label for this file ("adjusted" or "unadjusted").
               文件标签（"adjusted" 或 "unadjusted"）。
        path:  Path to the parquet file.
               parquet 文件路径。
        start: Report start date. / 报告起始日期。
        end:   Report end date. / 报告结束日期。
    """
    if not path.exists():
        print(f"\n[{name}] {path} NOT FOUND")
        return

    df = pd.read_parquet(path)
    df = df[(df.index.date >= start) & (df.index.date <= end)]
    if df.empty:
        print(f"\n[{name}] no bars in range {start}..{end}")
        return

    counts = analyze(df)
    sched = nyse_schedule(start, end)

    # Merge: every NYSE trading day + actual counts (fill 0 if missing)
    # 合并：NYSE 每个交易日 + 实际 K 线数（缺失补 0）
    merged = sched.join(counts, how="left").fillna(0)
    for c in ("pre", "rth", "post", "total"):
        merged[c] = merged[c].astype(int)

    nyse_days = len(sched)
    data_days = int((merged["total"] > 0).sum())
    missing_sessions = merged[merged["total"] == 0]

    # Pre/post presence — only meaningful for full trading days (not half-days)
    # 盘前/盘后覆盖率——仅对全天交易日有意义（半天交易日除外）
    full_days = merged[~merged["early_close"]]
    days_with_pre = int((full_days["pre"] > 0).sum())
    days_with_post = int((full_days["post"] > 0).sum())

    # RTH gap: actual RTH bars < expected - 5 (tolerate a couple missing)
    # 常规时段缺口：实际 K 线数 < 预期 - 5（容忍少量缺失）
    rth_deficit = merged[(merged["rth"] > 0) & (merged["rth"] < merged["expected_rth"] - 5)]

    # Fully-missing full days (no data at all on a day NYSE said was open)
    # 完全缺失的交易日（NYSE 标记为开盘但无任何数据）
    truly_missing = merged[merged["total"] == 0]

    print(f"\n=== {name}  ({start} → {end}) ===")
    print(f"  file:                       {path.name}")
    print(f"  NYSE trading days in range: {nyse_days}")
    print(f"  days with any data:         {data_days}")
    print(f"  days with NO data (gap):    {len(truly_missing)}")
    print(f"  NYSE half-days in range:    {int(merged['early_close'].sum())}")
    print(f"  full days w/ pre-market:    {days_with_pre}/{len(full_days)} "
          f"({days_with_pre/len(full_days)*100:.2f}%)")
    print(f"  full days w/ after-hours:   {days_with_post}/{len(full_days)} "
          f"({days_with_post/len(full_days)*100:.2f}%)")
    print(f"  days with RTH deficit (>5): {len(rth_deficit)}")

    if len(truly_missing) > 0:
        print(f"\n  --- trading days with NO data ({len(truly_missing)}) ---")
        for dt, row in truly_missing.iterrows():
            print(f"    {dt}  ({pd.Timestamp(dt).day_name()})  "
                  f"expected_rth={row['expected_rth']}")

    # Full-session days missing pre-market / 缺少盘前数据的全天交易日
    missing_pre = full_days[full_days["pre"] == 0]
    if len(missing_pre) > 0:
        print(f"\n  --- full days MISSING pre-market ({len(missing_pre)}; first 30) ---")
        for dt, row in missing_pre.head(30).iterrows():
            print(f"    {dt}  ({pd.Timestamp(dt).day_name()})  total_bars={row['total']}")
        if len(missing_pre) > 30:
            print(f"    ... and {len(missing_pre)-30} more")

    missing_post = full_days[full_days["post"] == 0]
    if len(missing_post) > 0:
        print(f"\n  --- full days MISSING after-hours ({len(missing_post)}; first 30) ---")
        for dt, row in missing_post.head(30).iterrows():
            print(f"    {dt}  ({pd.Timestamp(dt).day_name()})  total_bars={row['total']}")
        if len(missing_post) > 30:
            print(f"    ... and {len(missing_post)-30} more")

    if len(rth_deficit) > 0:
        print(f"\n  --- days with RTH deficit ({len(rth_deficit)}; first 20) ---")
        for dt, row in rth_deficit.head(20).iterrows():
            print(f"    {dt}  rth={row['rth']}/{row['expected_rth']}  "
                  f"short={row['expected_rth']-row['rth']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="start", default="2010-01-01", help="Start date (YYYY-MM-DD)")
    p.add_argument("--to", dest="end", default=None, help="End date (default: today)")
    p.add_argument("--file", choices=["adjusted", "unadjusted", "both"], default="both")
    args = p.parse_args()

    start = pd.Timestamp(args.start).date()
    end = pd.Timestamp(args.end).date() if args.end else date.today()

    targets = []
    if args.file in ("adjusted", "both"):
        targets.append(("adjusted", DATA_DIR / "QQQ_1min_adjusted.parquet"))
    if args.file in ("unadjusted", "both"):
        targets.append(("unadjusted", DATA_DIR / "QQQ_1min_unadjusted.parquet"))

    for name, path in targets:
        report(name, path, start, end)


if __name__ == "__main__":
    main()
