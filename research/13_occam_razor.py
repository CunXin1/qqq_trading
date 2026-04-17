"""
Phase 13 (research): Occam's Razor / Top-K Ablation.
Strip model down to Top 3/5/10/20 features and see how much AUC drops.
If core alpha is concentrated in a few features -> robust.
If AUC collapses -> model was overfitting to noise.

Refactored to import from qqq_trading package instead of duplicating code.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score, average_precision_score

from utils.paths import OUTPUT_DIR, CHART_DIR
from utils.plotting import setup_matplotlib
from features.registry import get_full_features
from models.training import train_model
from utils.splits import walk_forward_splits

setup_matplotlib()
import matplotlib.pyplot as plt


def _train_xgb(X_tr, y_tr):
    """Train XGBoost via the package's train_model helper."""
    return train_model(X_tr, y_tr, model_type="xgboost")


def evaluate(model, X_te, y_te):
    proba = model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, proba)
    ap = average_precision_score(y_te, proba)

    results = {"auc": auc, "ap": ap, "proba": proba}
    for t in [0.5, 0.6, 0.7]:
        mask = proba >= t
        results[f"alerts_{int(t*10)}"] = mask.sum()
        results[f"hr_{int(t*10)}"] = y_te[mask].mean() if mask.sum() > 0 else 0
    return results


# ===================================================================
# Top-K ablation for a single target
# ===================================================================

def run_topk_ablation(df, feat_cols, target_col, label, premarket_feats=None):
    """Train with full features, get importance, then retrain with Top-K."""
    print(f"\n{'='*80}")
    print(f"  OCCAM'S RAZOR: {label}")
    print(f"{'='*80}")

    core_mask = df[["realized_vol_20d", "ret_lag1", target_col]].notna().all(axis=1)
    valid = df.loc[core_mask]
    train = valid.loc[:"2022-12-31"]
    test = valid.loc["2023-01-01":]

    X_tr_full = train[feat_cols].values
    y_tr = train[target_col].values.astype(int)
    X_te_full = test[feat_cols].values
    y_te = test[target_col].values.astype(int)

    # Step 1: Train full model to get feature importance ranking
    model_full = _train_xgb(X_tr_full, y_tr)
    importance = model_full.feature_importances_
    rank_idx = np.argsort(importance)[::-1]

    # Print top 20
    print(f"\n  Feature importance ranking (full model, {len(feat_cols)} features):")
    for i in range(min(20, len(rank_idx))):
        fi = rank_idx[i]
        print(f"    {i+1:>3}. {feat_cols[fi]:<35} {importance[fi]:.4f}")

    # Step 2: Evaluate full model
    full_results = evaluate(model_full, X_te_full, y_te)
    print(f"\n  Full model ({len(feat_cols)} features): "
          f"AUC={full_results['auc']:.4f}  AP={full_results['ap']:.4f}")

    # Step 3: Top-K ablation
    k_values = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, len(feat_cols)]

    print(f"\n  {'K':>5} {'Features':>10} {'AUC':>7} {'dAUC':>7} {'AP':>7} "
          f"{'A@.5':>5} {'HR@.5':>6} {'A@.6':>5} {'HR@.6':>6} {'A@.7':>5} {'HR@.7':>6}")
    print(f"  {'-'*85}")

    ablation_results = []

    for k in k_values:
        if k > len(feat_cols):
            continue

        top_k_idx = rank_idx[:k]
        top_k_names = [feat_cols[i] for i in top_k_idx]

        X_tr_k = train[top_k_names].values
        X_te_k = test[top_k_names].values

        model_k = _train_xgb(X_tr_k, y_tr)
        res_k = evaluate(model_k, X_te_k, y_te)
        delta = res_k["auc"] - full_results["auc"]

        print(f"  {k:>5} {len(top_k_names):>10} {res_k['auc']:>7.4f} {delta:>+7.4f} {res_k['ap']:>7.4f} "
              f"{res_k['alerts_5']:>5} {res_k['hr_5']:>5.0%} "
              f"{res_k['alerts_6']:>5} {res_k['hr_6']:>5.0%} "
              f"{res_k['alerts_7']:>5} {res_k['hr_7']:>5.0%}")

        ablation_results.append({
            "k": k, "auc": res_k["auc"], "ap": res_k["ap"], "delta": delta,
            "features": top_k_names[:5],  # store first 5 for reference
            "alerts_5": res_k["alerts_5"], "hr_5": res_k["hr_5"],
            "alerts_6": res_k["alerts_6"], "hr_6": res_k["hr_6"],
            "alerts_7": res_k["alerts_7"], "hr_7": res_k["hr_7"],
        })

    # Step 4: Walk-forward with Top-5 to double-check
    print(f"\n  --- Walk-Forward validation with Top-5 vs Full ---")
    top5_names = [feat_cols[i] for i in rank_idx[:5]]
    print(f"  Top-5 features: {top5_names}")

    wf_full_aucs = []
    wf_top5_aucs = []

    wf_splits = walk_forward_splits(
        valid, test_years=list(range(2015, 2027)),
        train_window_years=5, purge_days=5,
    )

    for split in wf_splits:
        train_wf = split["train"]
        test_wf = split["test"]
        test_year = split["year"]

        t_mask = train_wf[target_col].notna()
        te_mask = test_wf[target_col].notna()
        if t_mask.sum() < 200 or te_mask.sum() < 50:
            continue

        y_tr_wf = train_wf.loc[t_mask, target_col].values.astype(int)
        y_te_wf = test_wf.loc[te_mask, target_col].values.astype(int)
        if y_te_wf.sum() == 0 or y_te_wf.sum() == len(y_te_wf):
            continue

        # Full
        m_full = _train_xgb(train_wf.loc[t_mask, feat_cols].values, y_tr_wf)
        p_full = m_full.predict_proba(test_wf.loc[te_mask, feat_cols].values)[:, 1]
        auc_full = roc_auc_score(y_te_wf, p_full)

        # Top-5
        m_top5 = _train_xgb(train_wf.loc[t_mask, top5_names].values, y_tr_wf)
        p_top5 = m_top5.predict_proba(test_wf.loc[te_mask, top5_names].values)[:, 1]
        auc_top5 = roc_auc_score(y_te_wf, p_top5)

        wf_full_aucs.append({"year": test_year, "auc": auc_full})
        wf_top5_aucs.append({"year": test_year, "auc": auc_top5})

    if wf_full_aucs:
        df_wf_full = pd.DataFrame(wf_full_aucs)
        df_wf_top5 = pd.DataFrame(wf_top5_aucs)
        print(f"\n  {'Year':>6} {'Full AUC':>10} {'Top5 AUC':>10} {'Gap':>8}")
        print(f"  {'-'*36}")
        for (_, rf), (_, r5) in zip(df_wf_full.iterrows(), df_wf_top5.iterrows()):
            gap = r5["auc"] - rf["auc"]
            print(f"  {int(rf['year']):>6} {rf['auc']:>10.3f} {r5['auc']:>10.3f} {gap:>+8.3f}")
        print(f"\n  Mean Full: {df_wf_full['auc'].mean():.3f}  Mean Top5: {df_wf_top5['auc'].mean():.3f}  "
              f"Gap: {df_wf_top5['auc'].mean() - df_wf_full['auc'].mean():+.3f}")

    # Diagnosis
    top3_auc = [r for r in ablation_results if r["k"] == 3][0]["auc"] if any(r["k"] == 3 for r in ablation_results) else 0
    top5_auc = [r for r in ablation_results if r["k"] == 5][0]["auc"] if any(r["k"] == 5 for r in ablation_results) else 0
    full_auc = full_results["auc"]

    retention_3 = top3_auc / full_auc * 100
    retention_5 = top5_auc / full_auc * 100

    print(f"\n  === DIAGNOSIS ===")
    print(f"  Full AUC:  {full_auc:.4f}")
    print(f"  Top-3 AUC: {top3_auc:.4f} ({retention_3:.1f}% retained)")
    print(f"  Top-5 AUC: {top5_auc:.4f} ({retention_5:.1f}% retained)")

    if retention_5 > 95:
        verdict = "EXCELLENT - Core alpha in top 5, rest is noise. Very robust."
    elif retention_5 > 90:
        verdict = "GOOD - Most alpha in top features. Minor benefit from others."
    elif retention_5 > 80:
        verdict = "MODERATE - Meaningful contribution from many features. Some overfit risk."
    else:
        verdict = "WARNING - Alpha is dispersed across many features. High overfit risk."
    print(f"  Verdict: {verdict}")

    return {
        "label": label,
        "full_auc": full_auc,
        "ablation": ablation_results,
        "importance_rank": rank_idx,
        "feat_cols": feat_cols,
        "wf_full": wf_full_aucs,
        "wf_top5": wf_top5_aucs,
    }


# ===================================================================
# Visualization
# ===================================================================

def plot_occam(all_results):
    n = len(all_results)
    fig, axes = plt.subplots(2, n, figsize=(7 * n, 12))
    if n == 1:
        axes = axes.reshape(2, 1)
    fig.suptitle("Occam's Razor: Top-K Feature Ablation", fontsize=14)

    for col, (target_name, res) in enumerate(all_results.items()):
        ablation = pd.DataFrame(res["ablation"])

        # Top row: AUC vs K
        ax = axes[0, col]
        ax.plot(ablation["k"], ablation["auc"], "o-", color="steelblue", linewidth=2, markersize=6)
        ax.axhline(res["full_auc"], color="red", ls="--", alpha=0.5, label=f"Full ({res['full_auc']:.3f})")
        ax.axhline(0.5, color="gray", ls=":", alpha=0.3, label="Random")

        # Shade the "sweet spot"
        ax.axvspan(3, 10, alpha=0.1, color="green", label="Sweet spot")

        ax.set_xlabel("Number of Features (K)")
        ax.set_ylabel("Test AUC-ROC")
        ax.set_title(f"{target_name}")
        ax.legend(fontsize=8)
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 3, 5, 10, 20, 50, 100])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.grid(True, alpha=0.3)

        # Annotate key points
        for _, row in ablation.iterrows():
            if row["k"] in [1, 3, 5, 10]:
                ax.annotate(f"K={int(row['k'])}\n{row['auc']:.3f}",
                           (row["k"], row["auc"]),
                           textcoords="offset points", xytext=(10, 5), fontsize=8)

        # Bottom row: Walk-forward comparison
        ax = axes[1, col]
        if res["wf_full"] and res["wf_top5"]:
            df_f = pd.DataFrame(res["wf_full"])
            df_5 = pd.DataFrame(res["wf_top5"])
            ax.plot(df_f["year"], df_f["auc"], "o-", label=f"Full ({df_f['auc'].mean():.3f})",
                    color="steelblue", linewidth=2)
            ax.plot(df_5["year"], df_5["auc"], "s--", label=f"Top-5 ({df_5['auc'].mean():.3f})",
                    color="coral", linewidth=2)
            ax.axhline(0.5, color="gray", ls=":", alpha=0.3)
            ax.set_xlabel("Test Year")
            ax.set_ylabel("WF AUC")
            ax.set_title(f"Walk-Forward: Full vs Top-5")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(CHART_DIR / "20_occam_razor.png")
    plt.close()
    print(f"\nSaved 20_occam_razor.png")


# ===================================================================
# Main
# ===================================================================

def main():
    print("Loading data...")
    df = pd.read_parquet(OUTPUT_DIR / "interaction_features.parquet")
    df.index = pd.to_datetime(df.index)

    full_feats = get_full_features(include_interactions=True)
    available = [f for f in full_feats if f in df.columns]

    # Build targets
    for thresh in [1, 2, 3]:
        c2c = f"target_1dte_c2c_{thresh}pct"
        if c2c not in df.columns:
            df[c2c] = (df["abs_close_to_close"].shift(-1) > thresh / 100).astype(float)
            df.loc[df.index[-1], c2c] = np.nan

        rng = f"target_0dte_range_{thresh}pct"
        if rng not in df.columns:
            df[rng] = (df["intraday_range"] > thresh / 100).astype(float)

    # Add premarket for 0DTE
    pre_feats = [f for f in ["premarket_ret", "premarket_range", "gap_return"]
                 if f in df.columns and f not in available]
    feat_0dte = available + pre_feats

    all_results = {}

    # Test each target
    all_results["1DTE |C2C|>2%"] = run_topk_ablation(
        df, available, "target_1dte_c2c_2pct", "1DTE |C2C|>2%"
    )
    all_results["0DTE Range>2%"] = run_topk_ablation(
        df, feat_0dte, "target_0dte_range_2pct", "0DTE Range>2%"
    )
    all_results["0DTE Range>3%"] = run_topk_ablation(
        df, feat_0dte, "target_0dte_range_3pct", "0DTE Range>3%"
    )

    plot_occam(all_results)


if __name__ == "__main__":
    main()
