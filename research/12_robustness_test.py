"""
Phase 12 (research): Robustness validation.
1. Monotonicity test: does hit rate increase smoothly with confidence?
2. Purged walk-forward CV: does AUC hold across time?
3. Feature decay: do features lose power over time?

Refactored to import from qqq_trading package instead of duplicating code.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score, average_precision_score

from qqq_trading.utils.paths import OUTPUT_DIR, CHART_DIR
from qqq_trading.utils.plotting import setup_matplotlib
from qqq_trading.features.registry import get_full_features
from qqq_trading.models.training import train_model
from qqq_trading.utils.splits import walk_forward_splits

setup_matplotlib()
import matplotlib.pyplot as plt


def _train_xgb(X_tr, y_tr):
    """Train XGBoost via the package's train_model helper."""
    return train_model(X_tr, y_tr, model_type="xgboost")


# ===================================================================
# TEST 1: Monotonicity -- does hit rate increase smoothly?
# ===================================================================

def test_monotonicity(df, feat_cols, targets):
    """Fine-grained threshold analysis for each target."""
    print("\n" + "=" * 80)
    print("  TEST 1: MONOTONICITY (hit rate vs confidence threshold)")
    print("=" * 80)

    core_mask = df[["realized_vol_20d", "ret_lag1"]].notna().all(axis=1)
    valid = df.loc[core_mask]
    train = valid.loc[:"2022-12-31"]
    test = valid.loc["2023-01-01":]

    thresholds = np.arange(0.10, 0.96, 0.05)
    all_results = {}

    for target_name, target_col in targets.items():
        if target_col not in df.columns:
            continue
        t_mask = train[target_col].notna()
        te_mask = test[target_col].notna()
        X_tr = train.loc[t_mask, feat_cols].values
        y_tr = train.loc[t_mask, target_col].values.astype(int)
        X_te = test.loc[te_mask, feat_cols].values
        y_te = test.loc[te_mask, target_col].values.astype(int)

        model = _train_xgb(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]

        print(f"\n  {target_name} (base rate={y_te.mean():.1%}, N_test={len(y_te)})")
        print(f"  {'Thresh':>8} {'Alerts':>7} {'Hits':>5} {'HR':>7} {'Monotone':>9}")
        print(f"  {'-'*40}")

        prev_hr = 0
        monotone_violations = 0
        hrs = []
        alerts_list = []

        for t in thresholds:
            mask = proba >= t
            n_a = mask.sum()
            if n_a < 3:
                hrs.append(np.nan)
                alerts_list.append(n_a)
                continue
            hr = y_te[mask].mean()
            hrs.append(hr)
            alerts_list.append(n_a)

            is_mono = "ok" if hr >= prev_hr - 0.02 else "VIOLATION"
            if hr < prev_hr - 0.02 and n_a >= 5:
                monotone_violations += 1
            print(f"  {t:>8.2f} {n_a:>7} {y_te[mask].sum():>5.0f} {hr:>6.1%} {is_mono:>9}")
            prev_hr = hr

        verdict = "PASS" if monotone_violations <= 2 else "FAIL"
        print(f"  Monotonicity violations (N>=5): {monotone_violations} -> {verdict}")

        all_results[target_name] = {
            "thresholds": thresholds,
            "hrs": hrs,
            "alerts": alerts_list,
            "base_rate": y_te.mean(),
        }

    return all_results


# ===================================================================
# TEST 2: Purged Walk-Forward Cross-Validation
# ===================================================================

def test_walk_forward(df, feat_cols, targets, train_years=5, purge_days=5):
    """Walk-forward with purge gap between train and test."""
    print("\n" + "=" * 80)
    print(f"  TEST 2: PURGED WALK-FORWARD CV (train={train_years}yr, purge={purge_days}d)")
    print("=" * 80)

    core_mask = df[["realized_vol_20d", "ret_lag1"]].notna().all(axis=1)
    valid = df.loc[core_mask]

    test_years = list(range(2010, 2027))

    all_results = {}

    for target_name, target_col in targets.items():
        if target_col not in df.columns:
            continue

        splits = walk_forward_splits(
            valid, test_years=test_years,
            train_window_years=train_years, purge_days=purge_days,
        )
        year_results = []

        for split in splits:
            train_data = split["train"]
            test_data = split["test"]
            test_year = split["year"]

            t_mask = train_data[target_col].notna()
            te_mask = test_data[target_col].notna()

            if t_mask.sum() < 200 or te_mask.sum() < 50:
                continue

            X_tr = train_data.loc[t_mask, feat_cols].values
            y_tr = train_data.loc[t_mask, target_col].values.astype(int)
            X_te = test_data.loc[te_mask, feat_cols].values
            y_te = test_data.loc[te_mask, target_col].values.astype(int)

            if y_te.sum() == 0 or y_te.sum() == len(y_te):
                continue

            model = _train_xgb(X_tr, y_tr)
            proba = model.predict_proba(X_te)[:, 1]

            auc = roc_auc_score(y_te, proba)
            ap = average_precision_score(y_te, proba)

            # HR at 0.5
            mask05 = proba >= 0.5
            a5 = mask05.sum()
            h5 = y_te[mask05].mean() if a5 > 0 else 0

            train_start_year = test_year - train_years
            train_end_year = test_year - 1

            year_results.append({
                "test_year": test_year,
                "train_period": f"{train_start_year}-{train_end_year}",
                "n_train": t_mask.sum(),
                "n_test": te_mask.sum(),
                "pos_rate": y_te.mean(),
                "auc": auc,
                "ap": ap,
                "alerts_05": a5,
                "hr_05": h5,
            })

        if not year_results:
            continue

        yr_df = pd.DataFrame(year_results)
        mean_auc = yr_df["auc"].mean()
        std_auc = yr_df["auc"].std()
        min_auc = yr_df["auc"].min()
        max_auc = yr_df["auc"].max()

        print(f"\n  {target_name}")
        print(f"  {'Year':>6} {'Train':>14} {'N_tr':>6} {'N_te':>5} {'Pos%':>6} "
              f"{'AUC':>7} {'AP':>7} {'A@.5':>5} {'HR@.5':>6}")
        print(f"  {'-'*65}")
        for _, r in yr_df.iterrows():
            print(f"  {int(r['test_year']):>6} {r['train_period']:>14} "
                  f"{int(r['n_train']):>6} {int(r['n_test']):>5} {r['pos_rate']:>5.1%} "
                  f"{r['auc']:>7.3f} {r['ap']:>7.3f} {int(r['alerts_05']):>5} {r['hr_05']:>5.0%}")

        print(f"\n  Summary: AUC mean={mean_auc:.3f} std={std_auc:.3f} "
              f"min={min_auc:.3f} max={max_auc:.3f}")

        # Compare to static split
        static_train = valid.loc[:"2022-12-31"]
        static_test = valid.loc["2023-01-01":]
        st_mask = static_train[target_col].notna()
        ste_mask = static_test[target_col].notna()
        if st_mask.sum() > 200 and ste_mask.sum() > 50:
            model_s = _train_xgb(
                static_train.loc[st_mask, feat_cols].values,
                static_train.loc[st_mask, target_col].values.astype(int),
            )
            proba_s = model_s.predict_proba(static_test.loc[ste_mask, feat_cols].values)[:, 1]
            y_te_s = static_test.loc[ste_mask, target_col].values.astype(int)
            static_auc = roc_auc_score(y_te_s, proba_s)
            gap = static_auc - mean_auc
            print(f"  Static AUC (train ~2022, test 2023+): {static_auc:.3f}")
            print(f"  Gap (static - WF mean): {gap:+.3f}")
            verdict = "CONSISTENT" if abs(gap) < 0.05 else ("OVERFIT WARNING" if gap > 0.05 else "WF BETTER")
            print(f"  Verdict: {verdict}")

        all_results[target_name] = yr_df

    return all_results


# ===================================================================
# TEST 3: Feature Decay -- do top features lose power over time?
# ===================================================================

def test_feature_decay(df, feat_cols, target_col):
    """Train on different eras, check if same features remain important."""
    print("\n" + "=" * 80)
    print("  TEST 3: FEATURE DECAY (do top features change over time?)")
    print("=" * 80)

    core_mask = df[["realized_vol_20d", "ret_lag1", target_col]].notna().all(axis=1)
    valid = df.loc[core_mask]

    eras = [
        ("2000-2007", "2000-01-01", "2007-12-31"),
        ("2008-2015", "2008-01-01", "2015-12-31"),
        ("2016-2022", "2016-01-01", "2022-12-31"),
    ]

    era_importances = {}

    for era_name, start, end in eras:
        era_data = valid.loc[start:end]
        if len(era_data) < 200:
            continue

        X = era_data[feat_cols].values
        y = era_data[target_col].values.astype(int)

        model = _train_xgb(X, y)
        imp = model.feature_importances_
        era_importances[era_name] = imp

        top10_idx = np.argsort(imp)[-10:][::-1]
        print(f"\n  {era_name} (N={len(era_data)}, pos={y.mean():.1%})")
        print(f"  Top 10 features:")
        for rank, i in enumerate(top10_idx):
            print(f"    {rank+1:>2}. {feat_cols[i]:<35} {imp[i]:.4f}")

    # Stability: rank correlation between eras
    if len(era_importances) >= 2:
        from scipy.stats import spearmanr
        print(f"\n  Feature Importance Rank Correlation (Spearman):")
        era_names = list(era_importances.keys())
        for i in range(len(era_names)):
            for j in range(i + 1, len(era_names)):
                rho, p = spearmanr(era_importances[era_names[i]],
                                   era_importances[era_names[j]])
                stability = "STABLE" if rho > 0.7 else ("MODERATE" if rho > 0.5 else "UNSTABLE")
                print(f"    {era_names[i]} vs {era_names[j]}: rho={rho:.3f} (p={p:.2e}) -> {stability}")

    return era_importances


# ===================================================================
# Visualization
# ===================================================================

def plot_robustness(mono_results, wf_results, era_importances, feat_cols):
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    # -- Plot 1: Monotonicity --
    ax = fig.add_subplot(gs[0, 0])
    for target_name, res in mono_results.items():
        hrs = [h * 100 if h is not None and not np.isnan(h) else np.nan for h in res["hrs"]]
        ax.plot(res["thresholds"], hrs, "o-", label=target_name, markersize=4)
        ax.axhline(res["base_rate"] * 100, ls="--", alpha=0.2)
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Hit Rate (%)")
    ax.set_title("Test 1: Monotonicity Check\n(HR should increase smoothly with threshold)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # -- Plot 2: Alert count monotonicity --
    ax = fig.add_subplot(gs[0, 1])
    for target_name, res in mono_results.items():
        ax.plot(res["thresholds"], res["alerts"], "s--", label=target_name, markersize=4)
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Number of Alerts")
    ax.set_title("Alert Count (should decrease smoothly)")
    ax.legend(fontsize=8)

    # -- Plot 3: Walk-forward AUC over time --
    ax = fig.add_subplot(gs[1, 0])
    for target_name, yr_df in wf_results.items():
        ax.plot(yr_df["test_year"], yr_df["auc"], "o-", label=target_name, markersize=5)
        ax.axhline(yr_df["auc"].mean(), ls="--", alpha=0.3)
    ax.axhline(0.5, color="black", ls=":", alpha=0.3, label="Random (0.5)")
    ax.set_xlabel("Test Year")
    ax.set_ylabel("OOS AUC-ROC")
    ax.set_title("Test 2: Walk-Forward AUC by Year\n(5yr rolling train, 5d purge)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # -- Plot 4: Walk-forward pos rate vs AUC --
    ax = fig.add_subplot(gs[1, 1])
    for target_name, yr_df in wf_results.items():
        ax.scatter(yr_df["pos_rate"] * 100, yr_df["auc"], label=target_name, s=40, alpha=0.7)
    ax.set_xlabel("Positive Rate in Test Year (%)")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("AUC vs Base Rate (does model work in all regimes?)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # -- Plot 5: Feature stability heatmap --
    ax = fig.add_subplot(gs[2, 0])
    if len(era_importances) >= 2:
        era_names = list(era_importances.keys())
        avg_imp = np.mean([era_importances[e] for e in era_names], axis=0)
        top20_idx = np.argsort(avg_imp)[-20:][::-1]
        data = np.array([[era_importances[e][i] for e in era_names] for i in top20_idx])
        data_norm = data / data.sum(axis=0, keepdims=True)

        import seaborn as sns
        sns.heatmap(data_norm, annot=True, fmt=".3f", cmap="YlOrRd",
                    xticklabels=era_names,
                    yticklabels=[feat_cols[i] for i in top20_idx],
                    ax=ax)
        ax.set_title("Test 3: Feature Importance Stability Across Eras\n(normalized)")
        ax.tick_params(axis="y", labelsize=7)

    # -- Plot 6: Summary verdict --
    ax = fig.add_subplot(gs[2, 1])
    ax.axis("off")

    verdicts = []
    verdicts.append("ROBUSTNESS TEST SUMMARY")
    verdicts.append("=" * 40)

    # Monotonicity verdict
    for target_name, res in mono_results.items():
        hrs_clean = [h for h in res["hrs"] if h is not None and not np.isnan(h)]
        violations = sum(1 for i in range(1, len(hrs_clean)) if hrs_clean[i] < hrs_clean[i-1] - 0.03)
        v = "PASS" if violations <= 2 else "WARN"
        verdicts.append(f"\nMonotonicity {target_name}: {v} ({violations} violations)")

    # Walk-forward verdict
    for target_name, yr_df in wf_results.items():
        mean_auc = yr_df["auc"].mean()
        std_auc = yr_df["auc"].std()
        min_auc = yr_df["auc"].min()
        v = "PASS" if mean_auc > 0.65 and min_auc > 0.5 else "WARN"
        verdicts.append(f"\nWalk-Forward {target_name}:")
        verdicts.append(f"  Mean AUC={mean_auc:.3f} +/- {std_auc:.3f}")
        verdicts.append(f"  Min={min_auc:.3f} -> {v}")

    # Feature stability
    if len(era_importances) >= 2:
        from scipy.stats import spearmanr
        era_names = list(era_importances.keys())
        rhos = []
        for i in range(len(era_names)):
            for j in range(i+1, len(era_names)):
                rho, _ = spearmanr(era_importances[era_names[i]], era_importances[era_names[j]])
                rhos.append(rho)
        avg_rho = np.mean(rhos)
        v = "STABLE" if avg_rho > 0.7 else ("MODERATE" if avg_rho > 0.5 else "UNSTABLE")
        verdicts.append(f"\nFeature Stability: avg rho={avg_rho:.3f} -> {v}")

    ax.text(0.05, 0.95, "\n".join(verdicts), transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.savefig(CHART_DIR / "19_robustness.png", bbox_inches="tight")
    plt.close()
    print(f"\nSaved 19_robustness.png")


# ===================================================================
# Main
# ===================================================================

def main():
    print("Loading data...")
    df = pd.read_parquet(OUTPUT_DIR / "interaction_features.parquet")
    df.index = pd.to_datetime(df.index)

    all_feats = get_full_features(include_interactions=True)
    available = [f for f in all_feats if f in df.columns]
    print(f"Features: {len(available)}")

    # Build targets
    for thresh in [1, 2, 3]:
        c2c = f"target_1dte_c2c_{thresh}pct"
        if c2c not in df.columns:
            df[c2c] = (df["abs_close_to_close"].shift(-1) > thresh / 100).astype(float)
            df.loc[df.index[-1], c2c] = np.nan

        o2c = f"target_0dte_o2c_{thresh}pct"
        if o2c not in df.columns:
            df[o2c] = (df["abs_open_to_close"] > thresh / 100).astype(float)

        rng = f"target_0dte_range_{thresh}pct"
        if rng not in df.columns:
            df[rng] = (df["intraday_range"] > thresh / 100).astype(float)

    # Define targets to test
    targets_1dte = {
        "1DTE |C2C|>1%": "target_1dte_c2c_1pct",
        "1DTE |C2C|>2%": "target_1dte_c2c_2pct",
        "1DTE |C2C|>3%": "target_1dte_c2c_3pct",
    }
    targets_0dte = {
        "0DTE |O2C|>2%": "target_0dte_o2c_2pct",
        "0DTE Range>2%": "target_0dte_range_2pct",
        "0DTE Range>3%": "target_0dte_range_3pct",
    }
    all_targets = {**targets_1dte, **targets_0dte}

    # Add premarket for 0DTE
    feat_0dte = available + [f for f in ["premarket_ret", "premarket_range", "gap_return"]
                             if f in df.columns and f not in available]

    # TEST 1: Monotonicity
    mono_1dte = test_monotonicity(df, available, targets_1dte)
    mono_0dte = test_monotonicity(df, feat_0dte, targets_0dte)
    mono_all = {**mono_1dte, **mono_0dte}

    # TEST 2: Walk-forward
    wf_1dte = test_walk_forward(df, available, targets_1dte, train_years=5, purge_days=5)
    wf_0dte = test_walk_forward(df, feat_0dte, targets_0dte, train_years=5, purge_days=5)
    wf_all = {**wf_1dte, **wf_0dte}

    # TEST 3: Feature decay
    era_imp = test_feature_decay(df, available, "target_1dte_c2c_2pct")

    # Plot
    plot_robustness(mono_all, wf_all, era_imp, available)


if __name__ == "__main__":
    main()
