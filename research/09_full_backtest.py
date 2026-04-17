"""
Phase 9 (research): Full backtest across all thresholds (1%, 2%, 3%, 5%).
Uses the interaction model (XGBoost) with walk-forward on test period 2023-2026.

Refactored to import from qqq_trading package instead of duplicating code.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, ConfusionMatrixDisplay,
)

from utils.paths import OUTPUT_DIR, CHART_DIR
from utils.plotting import setup_matplotlib
from features.registry import get_full_features
from models.training import create_model, compute_pos_weight
from config import ModelConfig

setup_matplotlib()
import matplotlib.pyplot as plt


def run_backtest_for_threshold(df, feat_cols, thresh_pct):
    """Train model for a specific threshold and run full backtest."""
    target_src = f"abs_close_to_close_gt_{thresh_pct}pct"
    target_col = f"target_next_{thresh_pct}pct"
    df[target_col] = df[target_src].shift(-1).astype(float)

    # Drop NaN on base features + target
    base_feats_core = ["realized_vol_20d", "ret_lag1", "close_to_close_ret"]
    base_mask = df[base_feats_core + [target_col]].notna().all(axis=1)
    valid = df.loc[base_mask]

    train = valid.loc[:"2022-12-31"]
    test = valid.loc["2023-01-01":]

    X_tr = train[feat_cols].values
    y_tr = train[target_col].values.astype(int)
    X_te = test[feat_cols].values
    y_te = test[target_col].values.astype(int)

    pos_w = compute_pos_weight(y_tr)

    # Train XGBoost
    model_xgb = create_model("xgboost", pos_weight=pos_w)
    model_xgb.fit(X_tr, y_tr)
    proba_xgb = model_xgb.predict_proba(X_te)[:, 1]

    # Train LightGBM
    model_lgb = create_model("lightgbm", pos_weight=pos_w)
    model_lgb.fit(X_tr, y_tr)
    proba_lgb = model_lgb.predict_proba(X_te)[:, 1]

    return {
        "thresh_pct": thresh_pct,
        "train_n": len(train),
        "test_n": len(test),
        "train_pos_rate": y_tr.mean(),
        "test_pos_rate": y_te.mean(),
        "test_pos_n": y_te.sum(),
        "y_test": y_te,
        "test_idx": test.index,
        "test_ret": test["close_to_close_ret"].shift(-1).values,  # actual next day return
        "xgb": {
            "model": model_xgb,
            "proba": proba_xgb,
            "auc": roc_auc_score(y_te, proba_xgb),
            "ap": average_precision_score(y_te, proba_xgb),
            "brier": brier_score_loss(y_te, proba_xgb),
        },
        "lgb": {
            "model": model_lgb,
            "proba": proba_lgb,
            "auc": roc_auc_score(y_te, proba_lgb),
            "ap": average_precision_score(y_te, proba_lgb),
            "brier": brier_score_loss(y_te, proba_lgb),
        },
    }


def print_backtest_results(result):
    t = result["thresh_pct"]
    print(f"\n{'='*80}")
    print(f"  TARGET: Next day |Close-to-Close| > {t}%")
    print(f"{'='*80}")
    print(f"  Test period: 2023-01-03 to 2026-02-20 ({result['test_n']} trading days)")
    print(f"  Train period: 2000-2019 + 2020-2022 ({result['train_n']} days)")
    print(f"  Train positive rate: {result['train_pos_rate']:.1%}")
    print(f"  Test positive rate:  {result['test_pos_rate']:.1%} "
          f"({result['test_pos_n']:.0f} days out of {result['test_n']})")

    for mname, mkey in [("XGBoost", "xgb"), ("LightGBM", "lgb")]:
        m = result[mkey]
        print(f"\n  --- {mname} ---")
        print(f"  AUC-ROC: {m['auc']:.4f}    Avg Precision: {m['ap']:.4f}    Brier: {m['brier']:.4f}")

        y_te = result["y_test"]
        proba = m["proba"]

        print(f"\n  {'Threshold':>10} {'Alerts':>8} {'Hits':>6} {'Misses':>8} "
              f"{'HitRate':>8} {'FalseAlarm':>10} {'Coverage':>10} {'Lift':>6}")
        print(f"  {'-'*70}")

        for conf in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            mask = proba >= conf
            n_alerts = mask.sum()
            if n_alerts == 0:
                continue
            hits = y_te[mask].sum()
            misses = n_alerts - hits
            hr = y_te[mask].mean()
            false_alarm = misses / n_alerts
            coverage = hits / y_te.sum() if y_te.sum() > 0 else 0
            lift = hr / y_te.mean() if y_te.mean() > 0 else 0
            print(f"  {conf:>10.1f} {n_alerts:>8d} {hits:>6.0f} {misses:>8.0f} "
                  f"{hr:>8.1%} {false_alarm:>10.1%} {coverage:>10.1%} {lift:>5.1f}x")

    # Day-by-day log for very high confidence predictions
    print(f"\n  --- High confidence alerts (XGBoost >= 0.6) ---")
    proba = result["xgb"]["proba"]
    y_te = result["y_test"]
    test_idx = result["test_idx"]
    high_conf = proba >= 0.6
    if high_conf.sum() > 0:
        print(f"  {'Date':>12} {'Proba':>8} {'Actual':>10} {'Next Ret':>10} {'Result':>8}")
        print(f"  {'-'*52}")
        for i in np.where(high_conf)[0]:
            date = test_idx[i]
            p = proba[i]
            actual = y_te[i]
            next_ret = result["test_ret"][i] if i < len(result["test_ret"]) else np.nan
            hit = "HIT" if actual == 1 else "miss"
            ret_str = f"{next_ret:.2%}" if not np.isnan(next_ret) else "N/A"
            actual_str = "YES" if actual == 1 else "no"
            print(f"  {date.strftime('%Y-%m-%d'):>12} {p:>8.1%} "
                  f"{actual_str:>10} {ret_str:>10} {hit:>8}")
    else:
        print(f"  (no alerts at this threshold)")


def plot_all_thresholds(all_results):
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Full Backtest: XGBoost Interaction Model (Test: 2023.1 - 2026.2)", fontsize=14)

    thresholds = [1, 2, 3, 5]
    colors = {1: "steelblue", 2: "coral", 3: "orange", 5: "red"}

    # -- Plot 1: AUC by threshold --
    ax = axes[0, 0]
    x = range(len(thresholds))
    xgb_aucs = [all_results[t]["xgb"]["auc"] for t in thresholds]
    lgb_aucs = [all_results[t]["lgb"]["auc"] for t in thresholds]
    w = 0.35
    ax.bar([i - w/2 for i in x], xgb_aucs, w, label="XGBoost", color="steelblue")
    ax.bar([i + w/2 for i in x], lgb_aucs, w, label="LightGBM", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels([f">{t}%" for t in thresholds])
    ax.set_ylabel("AUC-ROC")
    ax.set_title("AUC-ROC by Threshold")
    ax.legend()
    for i, (xa, la) in enumerate(zip(xgb_aucs, lgb_aucs)):
        ax.text(i - w/2, xa + 0.005, f"{xa:.3f}", ha="center", fontsize=8)
        ax.text(i + w/2, la + 0.005, f"{la:.3f}", ha="center", fontsize=8)

    # -- Plot 2: Hit rate curves (XGBoost) --
    ax = axes[0, 1]
    conf_thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    for t in thresholds:
        y_te = all_results[t]["y_test"]
        proba = all_results[t]["xgb"]["proba"]
        hrs = []
        for c in conf_thresholds:
            mask = proba >= c
            hr = y_te[mask].mean() * 100 if mask.sum() > 0 else 0
            hrs.append(hr)
        ax.plot(conf_thresholds, hrs, "o-", label=f">{t}%", color=colors[t], linewidth=2)

    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Hit Rate (%)")
    ax.set_title("XGBoost Hit Rate by Confidence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # -- Plot 3: Alerts count curves --
    ax = axes[0, 2]
    for t in thresholds:
        proba = all_results[t]["xgb"]["proba"]
        counts = [(proba >= c).sum() for c in conf_thresholds]
        ax.plot(conf_thresholds, counts, "s--", label=f">{t}%", color=colors[t])
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Number of Alerts")
    ax.set_title("Alert Count by Confidence")
    ax.legend()

    # -- Plot 4-7: Confusion matrices at optimal threshold for each target --
    from models.evaluation import find_optimal_threshold

    for idx, t in enumerate(thresholds):
        if idx >= 2:
            ax = axes[1, idx - 2 + 1]
        else:
            ax = axes[1, idx]

        y_te = all_results[t]["y_test"]
        proba = all_results[t]["xgb"]["proba"]

        best_th = find_optimal_threshold(y_te, proba)
        # Compute F1 at best threshold
        pred = (proba >= best_th).astype(int)
        tp = ((pred == 1) & (y_te == 1)).sum()
        fp = ((pred == 1) & (y_te == 0)).sum()
        fn = ((pred == 0) & (y_te == 1)).sum()
        pr = tp / (tp + fp) if (tp + fp) > 0 else 0
        re = tp / (tp + fn) if (tp + fn) > 0 else 0
        best_f1 = 2 * pr * re / (pr + re) if (pr + re) > 0 else 0

        cm = confusion_matrix(y_te, pred)
        ConfusionMatrixDisplay(cm, display_labels=["Normal", f">{t}%"]).plot(ax=ax)
        ax.set_title(f">{t}% (thresh={best_th:.2f}, F1={best_f1:.2f})")

    # Adjust last subplot if only 4 thresholds
    if len(thresholds) < 4:
        axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(CHART_DIR / "16_full_backtest.png")
    plt.close()
    print(f"\nSaved 16_full_backtest.png")


def print_summary_table(all_results):
    """Print a clean summary across all thresholds."""
    print("\n" + "=" * 90)
    print("  SUMMARY: XGBoost Interaction Model -- All Thresholds")
    print("  Test Period: 2023-01-03 to 2026-02-20")
    print("=" * 90)

    print(f"\n  {'Target':<12} {'Base':>6} {'AUC':>6} {'AP':>6} "
          f"| {'@0.3':>12} {'@0.4':>12} {'@0.5':>12} {'@0.6':>12}")
    print(f"  {'':12} {'Rate':>6} {'':>6} {'':>6} "
          f"| {'A / HR':>12} {'A / HR':>12} {'A / HR':>12} {'A / HR':>12}")
    print(f"  {'-'*85}")

    for t in [1, 2, 3, 5]:
        r = all_results[t]
        y_te = r["y_test"]
        proba = r["xgb"]["proba"]
        base_rate = y_te.mean()
        auc = r["xgb"]["auc"]
        ap = r["xgb"]["ap"]

        cells = []
        for c in [0.3, 0.4, 0.5, 0.6]:
            mask = proba >= c
            n_a = mask.sum()
            hr = y_te[mask].mean() if n_a > 0 else 0
            cells.append(f"{n_a:>4}/{hr:>5.0%}")

        print(f"  >{t}% move    {base_rate:>5.1%} {auc:>6.3f} {ap:>6.3f} "
              f"| {cells[0]:>12} {cells[1]:>12} {cells[2]:>12} {cells[3]:>12}")

    # Same for LightGBM
    print(f"\n  --- LightGBM ---")
    print(f"  {'Target':<12} {'Base':>6} {'AUC':>6} {'AP':>6} "
          f"| {'@0.3':>12} {'@0.4':>12} {'@0.5':>12} {'@0.6':>12}")
    print(f"  {'-'*85}")

    for t in [1, 2, 3, 5]:
        r = all_results[t]
        y_te = r["y_test"]
        proba = r["lgb"]["proba"]
        base_rate = y_te.mean()
        auc = r["lgb"]["auc"]
        ap = r["lgb"]["ap"]

        cells = []
        for c in [0.3, 0.4, 0.5, 0.6]:
            mask = proba >= c
            n_a = mask.sum()
            hr = y_te[mask].mean() if n_a > 0 else 0
            cells.append(f"{n_a:>4}/{hr:>5.0%}")

        print(f"  >{t}% move    {base_rate:>5.1%} {auc:>6.3f} {ap:>6.3f} "
              f"| {cells[0]:>12} {cells[1]:>12} {cells[2]:>12} {cells[3]:>12}")


def main():
    print("Loading interaction features...")
    df = pd.read_parquet(OUTPUT_DIR / "interaction_features.parquet")
    df.index = pd.to_datetime(df.index)

    all_feats = get_full_features(include_interactions=True)
    available = [f for f in all_feats if f in df.columns]
    print(f"Features: {len(available)}")

    # Run backtest for each threshold
    all_results = {}
    for thresh in [1, 2, 3, 5]:
        result = run_backtest_for_threshold(df, available, thresh)
        all_results[thresh] = result
        print_backtest_results(result)

    # Summary
    print_summary_table(all_results)

    # Plot
    plot_all_thresholds(all_results)


if __name__ == "__main__":
    main()
