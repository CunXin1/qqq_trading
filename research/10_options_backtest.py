"""
Phase 10 (research): Options-oriented backtest for 0DTE and 1DTE straddles.

0DTE: Buy straddle at open (9:30), expires at close (16:00)
  -> Predict TODAY's |open-to-close| and intraday_range
  -> Features: pre-market + prior day data (available before 9:30)

1DTE: Buy straddle at today's close, expires tomorrow close
  -> Predict TOMORROW's |close-to-close|
  -> Features: today's full data (available after 16:00)

Refactored to import from qqq_trading package instead of duplicating code.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from qqq_trading.utils.paths import OUTPUT_DIR, CHART_DIR
from qqq_trading.utils.plotting import setup_matplotlib
from qqq_trading.features.registry import (
    get_base_features, get_refined_external_features, get_interaction_features,
    get_full_features, get_0dte_premarket_features,
)
from qqq_trading.models.training import create_model, compute_pos_weight
from qqq_trading.config import ModelConfig

setup_matplotlib()
import matplotlib.pyplot as plt


# ===================================================================
# Feature sets for 0DTE / 1DTE
# ===================================================================

def get_1dte_features():
    """1DTE: all features available at TODAY's close. Predicts TOMORROW."""
    return get_full_features(include_interactions=True)


def get_0dte_features():
    """0DTE: features available BEFORE today's open (9:30).
    Same as 1DTE + pre-market data for TODAY.
    Key difference: pre-market features are for SAME day (not shifted).
    """
    features = get_1dte_features()
    # Add same-day pre-market features
    features += [
        "premarket_ret_today", "premarket_range_today", "premarket_vol_ratio",
    ]
    # Add same-day gap (today's open vs yesterday's close)
    features += ["gap_return_today"]
    return features


# ===================================================================
# Build 0DTE-specific features
# ===================================================================

def build_0dte_features(df):
    """Add same-day pre-market features for 0DTE prediction."""
    out = df.copy()

    # Same-day pre-market features (NOT shifted -- this is today's pre-market)
    out["premarket_ret_today"] = out["premarket_ret"]
    out["premarket_range_today"] = out["premarket_range"]
    out["premarket_vol_ratio"] = (
        out["volume_premarket"] /
        out["volume_premarket"].rolling(20).mean().shift(1)
    )

    # Same-day gap (today's open vs yesterday's close)
    out["gap_return_today"] = out["gap_return"]

    return out


# ===================================================================
# Define targets
# ===================================================================

def build_targets(df):
    """Build all target variables."""
    out = df.copy()

    # 1DTE targets: TOMORROW's move (shift -1)
    for thresh in [1, 2, 3, 5]:
        out[f"target_1dte_c2c_{thresh}pct"] = (
            out["abs_close_to_close"].shift(-1) > thresh / 100
        ).astype(float)
        out.loc[out.index[-1], f"target_1dte_c2c_{thresh}pct"] = np.nan

    # 0DTE targets: TODAY's move (no shift -- same day)
    for thresh in [1, 2, 3, 5]:
        out[f"target_0dte_o2c_{thresh}pct"] = (
            out["abs_open_to_close"] > thresh / 100
        ).astype(float)
        out[f"target_0dte_range_{thresh}pct"] = (
            out["intraday_range"] > thresh / 100
        ).astype(float)

    return out


# ===================================================================
# Training
# ===================================================================

def train_and_backtest(df, feat_cols, target_col, label):
    """Train XGBoost + LightGBM and run backtest."""
    available = [f for f in feat_cols if f in df.columns]

    # Drop rows where core features are NaN
    core_cols = ["realized_vol_20d", "ret_lag1"]
    core_available = [c for c in core_cols if c in df.columns]
    mask = df[core_available + [target_col]].notna().all(axis=1)
    valid = df.loc[mask]

    train = valid.loc[:"2022-12-31"]
    test = valid.loc["2023-01-01":]

    if len(train) < 200 or len(test) < 50:
        print(f"  {label}: insufficient data (train={len(train)}, test={len(test)})")
        return None

    X_tr, y_tr = train[available].values, train[target_col].values.astype(int)
    X_te, y_te = test[available].values, test[target_col].values.astype(int)

    pos_w = compute_pos_weight(y_tr)

    results = {}
    for mname, model_type in [("XGBoost", "xgboost"), ("LightGBM", "lightgbm")]:
        model = create_model(model_type, pos_weight=pos_w)
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, proba)
        ap = average_precision_score(y_te, proba)
        brier = brier_score_loss(y_te, proba)

        results[mname] = {
            "model": model, "proba": proba,
            "auc": auc, "ap": ap, "brier": brier,
        }

    return {
        "results": results,
        "y_test": y_te,
        "test_idx": test.index,
        "train_n": len(train), "test_n": len(test),
        "train_pos_rate": y_tr.mean(), "test_pos_rate": y_te.mean(),
        "test_pos_n": y_te.sum(),
        "label": label,
        "feat_cols": available,
    }


def print_results(res, show_alerts=True):
    if res is None:
        return
    label = res["label"]
    print(f"\n  {label}")
    print(f"  Train: {res['train_n']} days ({res['train_pos_rate']:.1%} pos)")
    print(f"  Test:  {res['test_n']} days ({res['test_pos_rate']:.1%} pos, "
          f"{res['test_pos_n']:.0f} events)")

    y_te = res["y_test"]
    for mname in ["XGBoost", "LightGBM"]:
        m = res["results"][mname]
        print(f"    {mname}: AUC={m['auc']:.4f}  AP={m['ap']:.4f}  Brier={m['brier']:.4f}")

    if show_alerts:
        # Use best model
        best = max(res["results"], key=lambda k: res["results"][k]["auc"])
        proba = res["results"][best]["proba"]
        print(f"    Backtest ({best}):")
        print(f"    {'Thresh':>8} {'Alerts':>7} {'Hits':>5} {'HR':>7} {'Cov':>7} {'Lift':>6}")
        print(f"    {'-'*42}")
        for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            mask = proba >= t
            n_a = mask.sum()
            if n_a == 0:
                continue
            hits = y_te[mask].sum()
            hr = y_te[mask].mean()
            cov = hits / y_te.sum() if y_te.sum() > 0 else 0
            lift = hr / y_te.mean() if y_te.mean() > 0 else 0
            print(f"    {t:>8.1f} {n_a:>7} {hits:>5.0f} {hr:>6.0%} {cov:>6.0%} {lift:>5.1f}x")


# ===================================================================
# Comprehensive visualization
# ===================================================================

def plot_options_backtest(all_0dte, all_1dte):
    fig, axes = plt.subplots(3, 2, figsize=(18, 18))
    fig.suptitle("Options Straddle Backtest: 0DTE vs 1DTE (Test: 2023.1 - 2026.2)", fontsize=15)

    thresholds = [1, 2, 3, 5]
    colors = {1: "steelblue", 2: "coral", 3: "orange", 5: "red"}
    conf_levels = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    # -- Row 1: AUC comparison --
    ax = axes[0, 0]
    x = np.arange(len(thresholds))
    w = 0.2

    aucs_0dte_o2c = []
    aucs_0dte_range = []
    aucs_1dte = []
    for t in thresholds:
        key_o2c = f"0DTE |O2C|>{t}%"
        key_range = f"0DTE Range>{t}%"
        key_1dte = f"1DTE |C2C|>{t}%"

        aucs_0dte_o2c.append(max(all_0dte[key_o2c]["results"].values(),
                                  key=lambda v: v["auc"])["auc"] if key_o2c in all_0dte and all_0dte[key_o2c] else 0)
        aucs_0dte_range.append(max(all_0dte[key_range]["results"].values(),
                                    key=lambda v: v["auc"])["auc"] if key_range in all_0dte and all_0dte[key_range] else 0)
        aucs_1dte.append(max(all_1dte[key_1dte]["results"].values(),
                              key=lambda v: v["auc"])["auc"] if key_1dte in all_1dte and all_1dte[key_1dte] else 0)

    ax.bar(x - w, aucs_0dte_o2c, w, label="0DTE |O2C|", color="steelblue")
    ax.bar(x, aucs_0dte_range, w, label="0DTE Range", color="coral")
    ax.bar(x + w, aucs_1dte, w, label="1DTE |C2C|", color="green")
    ax.set_xticks(x)
    ax.set_xticklabels([f">{t}%" for t in thresholds])
    ax.set_ylabel("AUC-ROC")
    ax.set_title("AUC-ROC: 0DTE vs 1DTE")
    ax.legend(fontsize=9)
    for i, (a, b, c) in enumerate(zip(aucs_0dte_o2c, aucs_0dte_range, aucs_1dte)):
        ax.text(i - w, a + 0.005, f"{a:.3f}", ha="center", fontsize=7)
        ax.text(i, b + 0.005, f"{b:.3f}", ha="center", fontsize=7)
        ax.text(i + w, c + 0.005, f"{c:.3f}", ha="center", fontsize=7)

    # -- Row 1 right: Base rates --
    ax = axes[0, 1]
    rates_0dte_o2c = []
    rates_0dte_range = []
    rates_1dte = []
    for t in thresholds:
        key_o2c = f"0DTE |O2C|>{t}%"
        key_range = f"0DTE Range>{t}%"
        key_1dte = f"1DTE |C2C|>{t}%"
        rates_0dte_o2c.append(all_0dte[key_o2c]["test_pos_rate"] * 100 if key_o2c in all_0dte and all_0dte[key_o2c] else 0)
        rates_0dte_range.append(all_0dte[key_range]["test_pos_rate"] * 100 if key_range in all_0dte and all_0dte[key_range] else 0)
        rates_1dte.append(all_1dte[key_1dte]["test_pos_rate"] * 100 if key_1dte in all_1dte and all_1dte[key_1dte] else 0)

    ax.bar(x - w, rates_0dte_o2c, w, label="0DTE |O2C|", color="steelblue")
    ax.bar(x, rates_0dte_range, w, label="0DTE Range", color="coral")
    ax.bar(x + w, rates_1dte, w, label="1DTE |C2C|", color="green")
    ax.set_xticks(x)
    ax.set_xticklabels([f">{t}%" for t in thresholds])
    ax.set_ylabel("Base Rate (%)")
    ax.set_title("Event Frequency in Test Period")
    ax.legend(fontsize=9)

    # -- Row 2: Hit rate curves for 2% threshold --
    ax = axes[1, 0]
    for key, label, color, marker in [
        ("0DTE |O2C|>2%", "0DTE |O2C|>2%", "steelblue", "o"),
        ("0DTE Range>2%", "0DTE Range>2%", "coral", "s"),
        ("1DTE |C2C|>2%", "1DTE |C2C|>2%", "green", "^"),
    ]:
        source = all_0dte if "0DTE" in key else all_1dte
        if key not in source or source[key] is None:
            continue
        res = source[key]
        best = max(res["results"], key=lambda k: res["results"][k]["auc"])
        proba = res["results"][best]["proba"]
        y_te = res["y_test"]
        hrs = [y_te[proba >= c].mean() * 100 if (proba >= c).sum() > 0 else 0 for c in conf_levels]
        ax.plot(conf_levels, hrs, f"{marker}-", label=label, color=color, linewidth=2)
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Hit Rate (%)")
    ax.set_title("Hit Rate @ >2% Threshold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # -- Row 2 right: Hit rate curves for 1% threshold --
    ax = axes[1, 1]
    for key, label, color, marker in [
        ("0DTE |O2C|>1%", "0DTE |O2C|>1%", "steelblue", "o"),
        ("0DTE Range>1%", "0DTE Range>1%", "coral", "s"),
        ("1DTE |C2C|>1%", "1DTE |C2C|>1%", "green", "^"),
    ]:
        source = all_0dte if "0DTE" in key else all_1dte
        if key not in source or source[key] is None:
            continue
        res = source[key]
        best = max(res["results"], key=lambda k: res["results"][k]["auc"])
        proba = res["results"][best]["proba"]
        y_te = res["y_test"]
        hrs = [y_te[proba >= c].mean() * 100 if (proba >= c).sum() > 0 else 0 for c in conf_levels]
        ax.plot(conf_levels, hrs, f"{marker}-", label=label, color=color, linewidth=2)
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Hit Rate (%)")
    ax.set_title("Hit Rate @ >1% Threshold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # -- Row 3: Alert count + coverage for 2% --
    ax = axes[2, 0]
    for key, label, color in [
        ("0DTE |O2C|>2%", "0DTE |O2C|", "steelblue"),
        ("0DTE Range>2%", "0DTE Range", "coral"),
        ("1DTE |C2C|>2%", "1DTE |C2C|", "green"),
    ]:
        source = all_0dte if "0DTE" in key else all_1dte
        if key not in source or source[key] is None:
            continue
        res = source[key]
        best = max(res["results"], key=lambda k: res["results"][k]["auc"])
        proba = res["results"][best]["proba"]
        counts = [(proba >= c).sum() for c in conf_levels]
        ax.plot(conf_levels, counts, "o--", label=label, color=color)
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Number of Alerts")
    ax.set_title("Alert Count @ >2% (lower = more selective)")
    ax.legend(fontsize=9)

    # -- Row 3 right: Summary table as text --
    ax = axes[2, 1]
    ax.axis("off")
    summary_text = "SUMMARY TABLE (Best Model, @0.5 threshold)\n"
    summary_text += "-" * 60 + "\n"
    summary_text += f"{'Scenario':<22} {'Base%':>6} {'AUC':>6} {'Alerts':>7} {'HR':>6} {'Lift':>5}\n"
    summary_text += "-" * 60 + "\n"

    for scenario_name, source, thresholds_to_show in [
        ("0DTE", all_0dte, [("0DTE |O2C|>1%", ">1% O2C"), ("0DTE |O2C|>2%", ">2% O2C"),
                            ("0DTE Range>2%", ">2% Range"), ("0DTE Range>3%", ">3% Range")]),
        ("1DTE", all_1dte, [("1DTE |C2C|>1%", ">1% C2C"), ("1DTE |C2C|>2%", ">2% C2C"),
                            ("1DTE |C2C|>3%", ">3% C2C")]),
    ]:
        summary_text += f"\n{scenario_name}:\n"
        for key, short in thresholds_to_show:
            if key not in source or source[key] is None:
                continue
            res = source[key]
            best = max(res["results"], key=lambda k: res["results"][k]["auc"])
            proba = res["results"][best]["proba"]
            y_te = res["y_test"]
            auc = res["results"][best]["auc"]
            base = y_te.mean()
            mask = proba >= 0.5
            alerts = mask.sum()
            hr = y_te[mask].mean() if alerts > 0 else 0
            lift = hr / base if base > 0 else 0
            summary_text += f"  {short:<20} {base:>5.1%} {auc:>6.3f} {alerts:>7} {hr:>5.0%} {lift:>4.1f}x\n"

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    plt.savefig(CHART_DIR / "17_options_backtest.png")
    plt.close()
    print(f"\nSaved 17_options_backtest.png")


# ===================================================================
# Main
# ===================================================================

def main():
    print("Loading data...")
    df = pd.read_parquet(OUTPUT_DIR / "interaction_features.parquet")
    df.index = pd.to_datetime(df.index)

    print("Building 0DTE features...")
    df = build_0dte_features(df)
    df = build_targets(df)

    # -------------------------------------------
    # 1DTE backtest
    # -------------------------------------------
    print("\n" + "=" * 80)
    print("  1DTE STRADDLE: Buy at today's close, expires tomorrow close")
    print("  Target: tomorrow |close_to_close| > X%")
    print("  Features: everything up to today's close")
    print("=" * 80)

    feat_1dte = get_1dte_features()
    all_1dte = {}

    for thresh in [1, 2, 3, 5]:
        target = f"target_1dte_c2c_{thresh}pct"
        label = f"1DTE |C2C|>{thresh}%"
        res = train_and_backtest(df, feat_1dte, target, label)
        all_1dte[label] = res
        print_results(res)

    # -------------------------------------------
    # 0DTE backtest
    # -------------------------------------------
    print("\n" + "=" * 80)
    print("  0DTE STRADDLE: Buy at open (9:30), expires at close (16:00)")
    print("  Target A: today |open_to_close| > X%")
    print("  Target B: today intraday_range > X%")
    print("  Features: pre-market + prior day data (before 9:30)")
    print("=" * 80)

    feat_0dte = get_0dte_features()
    all_0dte = {}

    for thresh in [1, 2, 3, 5]:
        # Target A: |open_to_close|
        target_a = f"target_0dte_o2c_{thresh}pct"
        label_a = f"0DTE |O2C|>{thresh}%"
        res_a = train_and_backtest(df, feat_0dte, target_a, label_a)
        all_0dte[label_a] = res_a
        print_results(res_a)

        # Target B: intraday range
        target_b = f"target_0dte_range_{thresh}pct"
        label_b = f"0DTE Range>{thresh}%"
        res_b = train_and_backtest(df, feat_0dte, target_b, label_b)
        all_0dte[label_b] = res_b
        print_results(res_b)

    # -------------------------------------------
    # Summary
    # -------------------------------------------
    print("\n" + "=" * 80)
    print("  COMPREHENSIVE SUMMARY")
    print("=" * 80)

    print(f"\n  {'Scenario':<25} {'Base':>6} {'AUC':>6} "
          f"| {'@0.3':>10} {'@0.5':>10} {'@0.6':>10} {'@0.7':>10}")
    print(f"  {'':25} {'Rate':>6} {'':>6} "
          f"| {'A/HR':>10} {'A/HR':>10} {'A/HR':>10} {'A/HR':>10}")
    print(f"  {'-'*90}")

    print(f"\n  --- 1DTE (buy close, expire tomorrow close) ---")
    for thresh in [1, 2, 3, 5]:
        key = f"1DTE |C2C|>{thresh}%"
        if key not in all_1dte or all_1dte[key] is None:
            continue
        res = all_1dte[key]
        best = max(res["results"], key=lambda k: res["results"][k]["auc"])
        proba = res["results"][best]["proba"]
        y_te = res["y_test"]
        auc = res["results"][best]["auc"]

        cells = []
        for c in [0.3, 0.5, 0.6, 0.7]:
            mask = proba >= c
            n_a = mask.sum()
            hr = y_te[mask].mean() if n_a > 0 else 0
            cells.append(f"{n_a:>3}/{hr:>4.0%}")

        print(f"  {key:<25} {y_te.mean():>5.1%} {auc:>6.3f} "
              f"| {cells[0]:>10} {cells[1]:>10} {cells[2]:>10} {cells[3]:>10}")

    print(f"\n  --- 0DTE |Open-to-Close| (buy open, expire close) ---")
    for thresh in [1, 2, 3, 5]:
        key = f"0DTE |O2C|>{thresh}%"
        if key not in all_0dte or all_0dte[key] is None:
            continue
        res = all_0dte[key]
        best = max(res["results"], key=lambda k: res["results"][k]["auc"])
        proba = res["results"][best]["proba"]
        y_te = res["y_test"]
        auc = res["results"][best]["auc"]

        cells = []
        for c in [0.3, 0.5, 0.6, 0.7]:
            mask = proba >= c
            n_a = mask.sum()
            hr = y_te[mask].mean() if n_a > 0 else 0
            cells.append(f"{n_a:>3}/{hr:>4.0%}")

        print(f"  {key:<25} {y_te.mean():>5.1%} {auc:>6.3f} "
              f"| {cells[0]:>10} {cells[1]:>10} {cells[2]:>10} {cells[3]:>10}")

    print(f"\n  --- 0DTE Intraday Range (buy open, range > X%) ---")
    for thresh in [1, 2, 3, 5]:
        key = f"0DTE Range>{thresh}%"
        if key not in all_0dte or all_0dte[key] is None:
            continue
        res = all_0dte[key]
        best = max(res["results"], key=lambda k: res["results"][k]["auc"])
        proba = res["results"][best]["proba"]
        y_te = res["y_test"]
        auc = res["results"][best]["auc"]

        cells = []
        for c in [0.3, 0.5, 0.6, 0.7]:
            mask = proba >= c
            n_a = mask.sum()
            hr = y_te[mask].mean() if n_a > 0 else 0
            cells.append(f"{n_a:>3}/{hr:>4.0%}")

        print(f"  {key:<25} {y_te.mean():>5.1%} {auc:>6.3f} "
              f"| {cells[0]:>10} {cells[1]:>10} {cells[2]:>10} {cells[3]:>10}")

    # Plot
    plot_options_backtest(all_0dte, all_1dte)


if __name__ == "__main__":
    main()
