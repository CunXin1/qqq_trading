"""
Phase 5: Analyze optimal training window length for prediction.

Tests different historical lookback windows using walk-forward evaluation.
Uses shared feature registry and model creation from the qqq_trading package;
all window-analysis logic is unique to this research script.
"""
import sys
from pathlib import Path

# Ensure the package is importable when running standalone.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from utils.paths import OUTPUT_DIR, CHART_DIR
from utils.plotting import setup_matplotlib
from features.registry import get_base_features
from models.training import create_model, compute_pos_weight
from config import ModelConfig

setup_matplotlib()


def load_features():
    df = pd.read_parquet(OUTPUT_DIR / "engineered_features.parquet")
    df.index = pd.to_datetime(df.index)
    return df


# ====================================================================
# Test 1: Fixed test period, vary training start year
# ====================================================================

def test_fixed_window_starts(df, feature_cols, target_col):
    """Keep test period fixed (2023-2026), vary how far back training goes."""
    print("\n" + "=" * 70)
    print("TEST 1: Fixed test period (2023-2026), vary training start year")
    print("=" * 70)

    test = df.loc["2023-01-01":].dropna(subset=feature_cols + [target_col])
    X_test = test[feature_cols].values
    y_test = test[target_col].values.astype(int)

    start_years = list(range(2000, 2020))
    results = []

    # Config matching the original script's hyperparameters
    config = ModelConfig(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
    )

    for start_year in start_years:
        train = df.loc[f"{start_year}-01-01":"2022-12-31"].dropna(subset=feature_cols + [target_col])
        if len(train) < 200:
            continue

        X_train = train[feature_cols].values
        y_train = train[target_col].values.astype(int)
        n_train = len(train)
        pos_rate_train = y_train.mean()

        pos_weight = compute_pos_weight(y_train)
        model = create_model("xgboost", config=config, pos_weight=pos_weight, random_state=42)
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        brier = brier_score_loss(y_test, y_proba)

        # Hit rate at threshold 0.5
        alerts = (y_proba >= 0.5).sum()
        hit_rate = y_test[y_proba >= 0.5].mean() if alerts > 0 else 0

        results.append({
            "start_year": start_year,
            "n_train": n_train,
            "train_years": 2022 - start_year + 1,
            "pos_rate_train": pos_rate_train,
            "auc": auc,
            "ap": ap,
            "brier": brier,
            "alerts": alerts,
            "hit_rate": hit_rate,
        })

        print(f"  Train {start_year}-2022 ({n_train:>5d} days, {2022-start_year+1:>2d}yr): "
              f"AUC={auc:.4f}  AP={ap:.4f}  Brier={brier:.4f}  "
              f"Alerts@0.5={alerts:>3d}  HitRate={hit_rate:.1%}")

    return pd.DataFrame(results)


# ====================================================================
# Test 2: Walk-forward with rolling window
# ====================================================================

def test_rolling_window(df, feature_cols, target_col):
    """Walk-forward: retrain every year with a fixed-length rolling window."""
    print("\n" + "=" * 70)
    print("TEST 2: Walk-forward (retrain yearly) with different window sizes")
    print("=" * 70)

    window_years = [3, 5, 7, 10, 15, 99]  # 99 = expanding (use all history)
    test_years = list(range(2015, 2026))

    all_results = {w: [] for w in window_years}

    config = ModelConfig(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
    )

    for test_year in test_years:
        test_data = df.loc[f"{test_year}-01-01":f"{test_year}-12-31"].dropna(
            subset=feature_cols + [target_col]
        )
        if len(test_data) < 50:
            continue

        X_test = test_data[feature_cols].values
        y_test = test_data[target_col].values.astype(int)

        for w in window_years:
            if w == 99:
                train_start = "2000-01-01"
            else:
                train_start = f"{test_year - w}-01-01"
            train_end = f"{test_year - 1}-12-31"

            train_data = df.loc[train_start:train_end].dropna(
                subset=feature_cols + [target_col]
            )
            if len(train_data) < 200:
                continue

            X_train = train_data[feature_cols].values
            y_train = train_data[target_col].values.astype(int)

            pos_weight = compute_pos_weight(y_train)
            model = create_model("xgboost", config=config, pos_weight=pos_weight, random_state=42)
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_test)[:, 1]

            try:
                auc = roc_auc_score(y_test, y_proba)
            except ValueError:
                auc = np.nan

            all_results[w].append({
                "test_year": test_year,
                "window": f"{w}yr" if w < 99 else "all",
                "n_train": len(train_data),
                "auc": auc,
            })

    # Summarize
    print(f"\n{'Window':<10} {'Mean AUC':>10} {'Std AUC':>10} {'Min AUC':>10} {'Max AUC':>10}")
    print("-" * 52)
    summary = {}
    for w in window_years:
        res = pd.DataFrame(all_results[w])
        if res.empty:
            continue
        label = f"{w}yr" if w < 99 else "all"
        mean_auc = res["auc"].mean()
        std_auc = res["auc"].std()
        min_auc = res["auc"].min()
        max_auc = res["auc"].max()
        print(f"{label:<10} {mean_auc:>10.4f} {std_auc:>10.4f} {min_auc:>10.4f} {max_auc:>10.4f}")
        summary[label] = res

    return all_results, summary


# ====================================================================
# Test 3: Regime-aware analysis
# ====================================================================

def test_regime_performance(df, feature_cols, target_col):
    """Check if model performance differs in high-vol vs low-vol regimes."""
    print("\n" + "=" * 70)
    print("TEST 3: Performance by volatility regime (test period 2015-2026)")
    print("=" * 70)

    # Train on 2000-2014, test on 2015-2026
    train = df.loc[:"2014-12-31"].dropna(subset=feature_cols + [target_col])
    test = df.loc["2015-01-01":].dropna(subset=feature_cols + [target_col])

    X_train = train[feature_cols].values
    y_train = train[target_col].values.astype(int)

    config = ModelConfig(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
    )
    pos_weight = compute_pos_weight(y_train)
    model = create_model("xgboost", config=config, pos_weight=pos_weight, random_state=42)
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(test[feature_cols].values)[:, 1]

    # Compute rolling vol for regime classification
    rv20 = df["close_to_close_ret"].rolling(20).std() * np.sqrt(252)
    test_rv = rv20.loc[test.index].dropna()
    common_idx = test.index.intersection(test_rv.index)

    test_sub = test.loc[common_idx].copy()
    test_sub["proba"] = model.predict_proba(test_sub[feature_cols].values)[:, 1]
    test_sub["rv20"] = test_rv.loc[common_idx]

    # Split into regimes
    vol_median = test_sub["rv20"].median()
    vol_q75 = test_sub["rv20"].quantile(0.75)

    regimes = {
        "Low Vol (bottom 50%)": test_sub["rv20"] <= vol_median,
        "Med Vol (50-75%)": (test_sub["rv20"] > vol_median) & (test_sub["rv20"] <= vol_q75),
        "High Vol (top 25%)": test_sub["rv20"] > vol_q75,
    }

    print(f"\n{'Regime':<25} {'N':>6} {'Base Rate':>10} {'AUC':>8} {'AP':>8} "
          f"{'Alerts@0.5':>10} {'Hit Rate':>10}")
    print("-" * 80)

    for regime_name, mask in regimes.items():
        sub = test_sub[mask]
        y_true = sub[target_col].values.astype(int)
        y_prob = sub["proba"].values

        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = np.nan

        ap = average_precision_score(y_true, y_prob) if y_true.sum() > 0 else np.nan
        alerts = (y_prob >= 0.5).sum()
        hit_rate = y_true[y_prob >= 0.5].mean() if alerts > 0 else 0

        print(f"{regime_name:<25} {len(sub):>6} {y_true.mean():>10.1%} {auc:>8.4f} {ap:>8.4f} "
              f"{alerts:>10} {hit_rate:>10.1%}")


# ====================================================================
# Test 4: Train sample decay weighting
# ====================================================================

def test_sample_weighting(df, feature_cols, target_col):
    """Use all data but exponentially decay older samples."""
    print("\n" + "=" * 70)
    print("TEST 4: Exponential decay weighting (use all data, downweight old)")
    print("=" * 70)

    train = df.loc[:"2022-12-31"].dropna(subset=feature_cols + [target_col])
    test = df.loc["2023-01-01":].dropna(subset=feature_cols + [target_col])

    X_train = train[feature_cols].values
    y_train = train[target_col].values.astype(int)
    X_test = test[feature_cols].values
    y_test = test[target_col].values.astype(int)

    # Days from end of training
    days_from_end = (train.index[-1] - train.index).days

    half_lives = [365, 730, 1460, 2920, 5840, None]  # 1yr, 2yr, 4yr, 8yr, 16yr, None=uniform
    labels = ["1yr", "2yr", "4yr", "8yr", "16yr", "uniform"]

    print(f"\n{'Half-Life':<12} {'AUC':>8} {'AP':>8} {'Brier':>8} {'Alerts@0.5':>10} {'Hit Rate':>10}")
    print("-" * 60)

    config = ModelConfig(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
    )

    results = []
    for hl, label in zip(half_lives, labels):
        if hl is not None:
            weights = np.exp(-np.log(2) * days_from_end / hl)
        else:
            weights = np.ones(len(train))

        pos_weight = compute_pos_weight(y_train)
        model = create_model("xgboost", config=config, pos_weight=pos_weight, random_state=42)
        model.fit(X_train, y_train, sample_weight=weights)
        y_proba = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        brier = brier_score_loss(y_test, y_proba)
        alerts = (y_proba >= 0.5).sum()
        hit_rate = y_test[y_proba >= 0.5].mean() if alerts > 0 else 0

        print(f"{label:<12} {auc:>8.4f} {ap:>8.4f} {brier:>8.4f} {alerts:>10} {hit_rate:>10.1%}")
        results.append({"half_life": label, "auc": auc, "ap": ap, "brier": brier,
                        "alerts": alerts, "hit_rate": hit_rate})

    return pd.DataFrame(results)


# ====================================================================
# Plotting
# ====================================================================

def plot_results(fixed_results, rolling_results, decay_results):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Optimal Training Window Analysis", fontsize=14)

    # Plot 1: Fixed start year
    ax = axes[0, 0]
    ax.plot(fixed_results["start_year"], fixed_results["auc"], "o-", color="steelblue", label="AUC-ROC")
    ax.plot(fixed_results["start_year"], fixed_results["ap"], "s--", color="coral", label="Avg Precision")
    best_idx = fixed_results["auc"].idxmax()
    best_year = fixed_results.loc[best_idx, "start_year"]
    best_auc = fixed_results.loc[best_idx, "auc"]
    ax.axvline(best_year, color="green", ls=":", alpha=0.7, label=f"Best: {best_year} (AUC={best_auc:.3f})")
    ax.set_xlabel("Training Start Year")
    ax.set_ylabel("Score")
    ax.set_title("Test 1: AUC vs Training Start Year\n(test: 2023-2026)")
    ax.legend(fontsize=9)

    # Plot 2: Fixed start - training size vs positive rate
    ax = axes[0, 1]
    ax2 = ax.twinx()
    ax.bar(fixed_results["start_year"], fixed_results["n_train"], color="lightblue", alpha=0.7, label="Training days")
    ax2.plot(fixed_results["start_year"], fixed_results["pos_rate_train"] * 100, "ro-", markersize=4, label="Positive rate (%)")
    ax.set_xlabel("Training Start Year")
    ax.set_ylabel("Training Days", color="steelblue")
    ax2.set_ylabel("Positive Rate (%)", color="red")
    ax.set_title("Training Set Size & Positive Rate by Start Year")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

    # Plot 3: Walk-forward rolling window
    ax = axes[1, 0]
    window_labels = ["3yr", "5yr", "7yr", "10yr", "15yr", "all"]
    for label in window_labels:
        if label in rolling_results:
            res = rolling_results[label]
            ax.plot(res["test_year"], res["auc"], "o-", label=label, markersize=4)
    ax.set_xlabel("Test Year")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Test 2: Walk-Forward AUC by Window Size")
    ax.legend(fontsize=9)

    # Plot 4: Decay weighting
    ax = axes[1, 1]
    x = range(len(decay_results))
    ax.bar(x, decay_results["auc"], color="steelblue", alpha=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(decay_results["half_life"])
    ax.set_xlabel("Decay Half-Life")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Test 4: Exponential Decay Weighting\n(all data, downweight older)")
    for i, (auc, hr) in enumerate(zip(decay_results["auc"], decay_results["hit_rate"])):
        ax.text(i, auc + 0.003, f"{auc:.3f}\nHR:{hr:.0%}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(CHART_DIR / "12_window_analysis.png")
    plt.close()
    print(f"\nSaved 12_window_analysis.png")


def main():
    print("Loading engineered features...")
    df = load_features()
    feature_cols = get_base_features()
    target_col = "target_next_day_2pct"

    # Ensure target exists
    if target_col not in df.columns:
        df[target_col] = df["abs_close_to_close_gt_2pct"].shift(-1).astype(float)

    # Run all tests
    fixed_results = test_fixed_window_starts(df, feature_cols, target_col)
    rolling_raw, rolling_summary = test_rolling_window(df, feature_cols, target_col)
    test_regime_performance(df, feature_cols, target_col)
    decay_results = test_sample_weighting(df, feature_cols, target_col)

    # Convert rolling for plotting
    rolling_for_plot = {}
    for w, label in [(3, "3yr"), (5, "5yr"), (7, "7yr"), (10, "10yr"), (15, "15yr"), (99, "all")]:
        if w in rolling_raw and rolling_raw[w]:
            rolling_for_plot[label] = pd.DataFrame(rolling_raw[w])

    plot_results(fixed_results, rolling_for_plot, decay_results)

    # Final recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    best_fixed = fixed_results.loc[fixed_results["auc"].idxmax()]
    print(f"\nBest training start year (Test 1): {int(best_fixed['start_year'])} "
          f"(AUC={best_fixed['auc']:.4f}, {int(best_fixed['train_years'])} years of data)")

    best_decay = decay_results.loc[decay_results["auc"].idxmax()]
    print(f"Best decay half-life (Test 4): {best_decay['half_life']} "
          f"(AUC={best_decay['auc']:.4f})")


if __name__ == "__main__":
    main()
