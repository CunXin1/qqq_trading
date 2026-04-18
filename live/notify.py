"""
Daily pre-market prediction + Bark push notification.
每日盘前预测 + Bark 推送通知。

Runs at 9:29 AM ET (6:29 AM PT) — right before market open,
when pre-market data is most complete.
在美东 9:29（太平洋 6:29）运行——开盘前，盘前数据最完整时。

Flow / 流程:
  1. Fetch latest data (IBKR -> yfinance fallback).
     获取最新数据（IBKR → yfinance 备用）。
  2. Merge into historical parquets (bridges training ↔ live).
     合并到历史 parquet（衔接训练 ↔ 实时）。
  3. Build full 122-feature set + run 0DTE Range model prediction.
     构建完整 122 特征集 + 运行 0DTE Range 模型预测。
  4. Format signal + context → push to iPhone via Bark.
     格式化信号 + 上下文 → 通过 Bark 推送到 iPhone。

Notification content / 通知内容:
  - Signal level: HIGH (≥70%), ELEVATED (≥50%), MODERATE (≥30%), LOW (<30%).
    信号级别：高 (≥70%)、升高 (≥50%)、中等 (≥30%)、低 (<30%)。
  - Previous day's close, range, C2C, RV20, VRP.
    前一天的收盘价、振幅、C2C、RV20、VRP。
  - Today's pre-market range/return (if available).
    今天的盘前振幅/收益率（如可用）。
  - Upcoming events: FOMC, NFP, earnings season.
    即将到来的事件：FOMC、NFP、财报季。
  - Action suggestion: "BUY STRADDLE AT OPEN" for HIGH signals.
    操作建议：HIGH 信号时"开盘买入跨式"。

Usage / 用法:
    python -m live.notify                     # run once / 运行一次
    python -m live.notify --bark-key YOUR_KEY # override key / 覆盖密钥
    python -m live.notify --dry-run            # preview without pushing / 预览不推送

Config / 配置:
    Set BARK_KEY in config/bark.txt (one line, just the key).
    在 config/bark.txt 中设置 BARK_KEY（一行，仅密钥）。
    Or pass via --bark-key argument, or set BARK_KEY env variable.
    或通过 --bark-key 参数传入，或设置 BARK_KEY 环境变量。

Cron (Mac Mini, Pacific Time — 6:29 AM PT = 9:29 AM ET):
    29 6 * * 1-5 cd /path/to/qqq_trading && python -m live.notify >> logs/notify.log 2>&1
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import requests


def get_bark_key(args_key=None):
    """Get Bark key from args > env > config file."""
    if args_key:
        return args_key

    env_key = os.environ.get("BARK_KEY")
    if env_key:
        return env_key

    config_path = Path(__file__).resolve().parents[1] / "config" / "bark.txt"
    if config_path.exists():
        return config_path.read_text().strip()

    return None


def update_data():
    """Fetch latest data and merge into historical parquets."""
    from live.fetch_data import IBKRSource, YFinanceSource, fetch_from_source
    from live.fetch_data import fetch_yields, get_events
    from live.fetch_data import add_qqq_derived, add_vix_derived
    from live.fetch_data import merge_with_historical

    days = 5  # enough for rolling features to bridge

    # Try IBKR first
    result = None
    source = None

    try:
        ibkr = IBKRSource(port=7496)
        asyncio.run(ibkr.connect())
        result = asyncio.run(fetch_from_source(ibkr, days))
        asyncio.run(ibkr.disconnect())
        if result["qqq"].empty:
            result = None
        else:
            source = "IBKR"
    except Exception:
        pass

    # Fallback to yfinance
    if result is None:
        try:
            yf_src = YFinanceSource()
            asyncio.run(yf_src.connect())
            result = asyncio.run(fetch_from_source(yf_src, days))
            source = "yfinance"
        except Exception as e:
            print(f"  Data fetch failed: {e}")
            return None

    yields = fetch_yields(days)

    qqq = result["qqq"]
    premarket = result["premarket"]
    vix = result["vix"]

    if not qqq.empty:
        qqq = add_qqq_derived(qqq)
    if not vix.empty:
        vix.index.name = "date"
        vix = add_vix_derived(vix)

    merge_with_historical(qqq, premarket, vix, yields, source)
    return source


def run_prediction():
    """Build features and run model prediction.

    Uses yesterday's close data + today's pre-market data (available at 9:29 AM ET).
    """
    from utils.paths import OUTPUT_DIR, MODEL_DIR
    from models.training import load_model
    from features.base import engineer_base_features
    from features.external import engineer_all_external
    from features.interactions import build_interaction_features

    daily = pd.read_parquet(OUTPUT_DIR / "daily_metrics.parquet")
    daily.index = pd.to_datetime(daily.index)
    ext = pd.read_parquet(OUTPUT_DIR / "external_data.parquet")
    ext.index = pd.to_datetime(ext.index)

    df = engineer_base_features(daily)
    df = engineer_all_external(df, ext)
    df = build_interaction_features(df)

    model, feature_cols = load_model(MODEL_DIR / "interaction_model.joblib")
    available = [f for f in feature_cols if f in df.columns]

    # Latest row = last trading day's features -> predicts today
    latest = df.iloc[-1]
    data_date = str(latest.name.date())
    X = latest[available].values.reshape(1, -1)
    prob = model.predict_proba(X)[:, 1][0]

    # Try to get today's live pre-market data for extra context
    live_dir = OUTPUT_DIR / "live"
    premarket_ctx = {}
    if (live_dir / "live_premarket.csv").exists():
        pre = pd.read_csv(live_dir / "live_premarket.csv", index_col=0, parse_dates=True)
        if not pre.empty:
            latest_pre = pre.iloc[-1]
            premarket_ctx["pre_range"] = latest_pre.get("premarket_range", 0) * 100
            premarket_ctx["pre_ret"] = latest_pre.get("premarket_ret", 0) * 100
            premarket_ctx["pre_date"] = str(pre.index[-1].date())

    # Gather context
    context = {
        "data_date": data_date,
        "prob": prob,
        "range_pct": latest.get("intraday_range", 0) * 100,
        "c2c_pct": latest.get("close_to_close_ret", 0) * 100,
        "close": latest.get("reg_close", 0),
        "rv20": latest.get("realized_vol_20d", 0) * 100,
    }
    context.update(premarket_ctx)

    # Event flags (for today, shift forward from data)
    events = []
    days_to_fomc = latest.get("days_to_fomc", 99)
    days_to_nfp = latest.get("days_to_nfp", 99)
    # Data is yesterday, so today = days_to - 1 (approx)
    if days_to_fomc <= 2:
        events.append(f"FOMC in {max(0, days_to_fomc-1)}d")
    if days_to_nfp <= 2:
        events.append(f"NFP in {max(0, days_to_nfp-1)}d")
    if latest.get("is_earnings_season", 0) == 1:
        events.append("EARN")
    context["events"] = ", ".join(events) if events else "None"

    # VRP
    vrp = latest.get("vrp_20d", 0)
    context["vrp"] = vrp
    if vrp < -0.05:
        context["vrp_label"] = "Complacent"
    elif vrp > 0.05:
        context["vrp_label"] = "Fearful"
    else:
        context["vrp_label"] = "Neutral"

    return prob, context


def format_message(prob, ctx):
    """Format prediction into Bark notification."""
    # Signal level
    if prob >= 0.7:
        signal = "HIGH"
    elif prob >= 0.5:
        signal = "ELEVATED"
    elif prob >= 0.3:
        signal = "MODERATE"
    else:
        signal = "LOW"

    title = f"QQQ {signal} ({prob:.0%})"

    body_lines = [
        f"0DTE Range>2%: {prob:.1%}",
        f"",
        f"Prev close: ${ctx['close']:.2f}",
        f"Prev day: Range {ctx['range_pct']:.1f}%, C2C {ctx['c2c_pct']:+.1f}%",
        f"RV20: {ctx['rv20']:.1f}%  VRP: {ctx['vrp']:+.3f} ({ctx['vrp_label']})",
    ]

    # Pre-market context if available
    if "pre_range" in ctx:
        body_lines.append(f"Pre-market: Range {ctx['pre_range']:.2f}%, Ret {ctx['pre_ret']:+.2f}%")

    body_lines.append(f"Events: {ctx['events']}")

    if prob >= 0.7:
        body_lines.extend(["", ">>> BUY STRADDLE AT OPEN <<<"])
    elif prob >= 0.5:
        body_lines.extend(["", "Consider straddle at open"])

    body = "\n".join(body_lines)
    return title, body


def send_bark(key, title, body, dry_run=False):
    """Send push notification via Bark."""
    if dry_run:
        print(f"\n[DRY RUN] Would send:")
        print(f"  Title: {title}")
        print(f"  Body:\n{body}")
        return True

    url = f"https://api.day.app/{key}"
    data = {
        "title": title,
        "body": body,
        "group": "QQQ",
        "sound": "minuet",
    }

    try:
        resp = requests.post(url, json=data, timeout=10)
        if resp.status_code == 200:
            result = resp.json()
            if result.get("code") == 200:
                print(f"  Bark push sent successfully")
                return True
            else:
                print(f"  Bark error: {result}")
                return False
        else:
            print(f"  Bark HTTP error: {resp.status_code}")
            return False
    except Exception as e:
        print(f"  Bark failed: {e}")
        return False


def parse_args():
    parser = argparse.ArgumentParser(description="Daily QQQ prediction + Bark notification")
    parser.add_argument("--bark-key", type=str, default=None, help="Bark push key")
    parser.add_argument("--dry-run", action="store_true", help="Preview without sending")
    parser.add_argument("--skip-update", action="store_true", help="Skip data fetch, use existing parquets")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Only push if prob >= threshold (default: 0.3, always push)")
    return parser.parse_args()


def main():
    args = parse_args()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'=' * 50}")
    print(f"QQQ Daily Signal — {now}")
    print(f"{'=' * 50}")

    # Step 1: Update data
    if not args.skip_update:
        print("\n[1] Updating data...")
        source = update_data()
        if source:
            print(f"  Data source: {source}")
        else:
            print("  WARNING: Data update failed, using existing parquets")
    else:
        print("\n[1] Skipping data update (--skip-update)")

    # Step 2: Predict
    print("\n[2] Running prediction...")
    prob, ctx = run_prediction()
    title, body = format_message(prob, ctx)

    print(f"\n  {title}")
    print(f"  {body}")

    # Step 3: Push
    if prob < args.threshold:
        print(f"\n[3] Prob {prob:.1%} < threshold {args.threshold:.0%}, skipping push")
        return

    bark_key = get_bark_key(args.bark_key)
    if not bark_key:
        print(f"\n[3] No Bark key configured. Set in config/bark.txt or --bark-key")
        print(f"    Skipping push notification.")
        return

    print(f"\n[3] Sending Bark notification...")
    send_bark(bark_key, title, body, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
