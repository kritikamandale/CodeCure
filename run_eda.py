"""
Feature Selection & Exploratory Data Analysis
==============================================
Selects optimal features and generates EDA visualizations for epidemic forecasting.

Outputs:
  - feature_config.json (selected features + metadata)
  - EDA plots: ACF, PACF, correlation heatmap, trend plots
"""

import io
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import config

# Ensure plot output directory
PLOT_DIR = config.OUTPUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "cases_7day_avg"

# Columns to exclude from feature selection (identifiers, targets, derived targets, flags)
EXCLUDE_COLS = [
    "country", "date",
    # Raw targets / close derivatives
    "cases_7day_avg", "cases_7day_avg_smoothed",
    "daily_new_cases", "daily_new_cases_smoothed",
    "daily_new_deaths", "daily_new_deaths_smoothed",
    "confirmed", "deaths", "recovered",
    # Anomaly metadata
    "anomaly_daily_new_cases", "anomaly_daily_new_deaths",
    "anomaly_cases_7day_avg", "is_anomaly",
    "zscore_daily_new_cases", "zscore_daily_new_deaths",
    "zscore_cases_7day_avg",
]


def setup_logging():
    fmt = "%(asctime)s | %(levelname)-7s | %(message)s"
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(fmt, "%H:%M:%S"))
    fh = logging.FileHandler(config.LOGS_DIR / "eda_selection.log", "w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt, "%H:%M:%S"))
    logging.basicConfig(level=logging.DEBUG, handlers=[console, fh])


# ═══════════════════════════════════════════════════════════════════════
# 1. FEATURE SELECTION
# ═══════════════════════════════════════════════════════════════════════

def compute_correlations(df: pd.DataFrame, logger: logging.Logger):
    """Compute Pearson correlation of each feature with the target."""
    logger.info("== [1/4] Computing Correlations with Target ==")

    numeric = df.select_dtypes(include=[np.number])
    candidate_cols = [c for c in numeric.columns if c not in EXCLUDE_COLS]

    correlations = numeric[candidate_cols].corrwith(df[TARGET_COL]).dropna()
    correlations = correlations.abs().sort_values(ascending=False)

    logger.info(f"  Candidate features: {len(candidate_cols)}")
    logger.info(f"  Top-10 correlations:")
    for feat, corr in correlations.head(10).items():
        logger.info(f"    {feat:40s} |r| = {corr:.4f}")

    return correlations


def filter_weak_features(correlations: pd.Series, threshold: float = 0.15,
                         logger: logging.Logger = None):
    """Remove features with |correlation| < threshold."""
    logger.info(f"== [2/4] Filtering Weak Features (|r| < {threshold}) ==")

    strong = correlations[correlations >= threshold]
    weak = correlations[correlations < threshold]

    logger.info(f"  Strong features (kept): {len(strong)}")
    logger.info(f"  Weak features (dropped): {len(weak)}")
    if len(weak) > 0:
        logger.info(f"  Dropped: {list(weak.index)}")

    return strong.index.tolist()


def compute_vif(df: pd.DataFrame, features: list, logger: logging.Logger):
    """
    Compute Variance Inflation Factor for multicollinearity detection.
    Uses correlation matrix method (faster than OLS for many features).
    VIF_j = 1 / (1 - R_j^2) where R_j^2 is from regressing feature j on all others.
    """
    logger.info("== [3/4] Checking Multicollinearity (VIF) ==")

    X = df[features].dropna()

    # Standardize
    X_std = (X - X.mean()) / X.std()
    X_std = X_std.dropna(axis=1)

    valid_features = X_std.columns.tolist()

    # Correlation matrix method for VIF
    corr_matrix = X_std.corr().values
    try:
        inv_corr = np.linalg.inv(corr_matrix)
        vif_values = np.diag(inv_corr)
    except np.linalg.LinAlgError:
        logger.warning("  Correlation matrix is singular, using pseudo-inverse")
        inv_corr = np.linalg.pinv(corr_matrix)
        vif_values = np.diag(inv_corr)

    vif_df = pd.DataFrame({
        "feature": valid_features,
        "vif": vif_values
    }).sort_values("vif", ascending=False)

    logger.info(f"  VIF computed for {len(valid_features)} features")
    logger.info(f"  Top-10 VIF:")
    for _, row in vif_df.head(10).iterrows():
        flag = " *** HIGH" if row["vif"] > 10 else ""
        logger.info(f"    {row['feature']:40s} VIF = {row['vif']:8.2f}{flag}")

    return vif_df


def drop_high_vif(vif_df: pd.DataFrame, features: list, threshold: float = 10.0,
                  logger: logging.Logger = None):
    """Iteratively drop highest-VIF feature until all VIF < threshold."""
    logger.info(f"== [4/4] Dropping High VIF Features (> {threshold}) ==")

    high_vif = vif_df[vif_df["vif"] > threshold]["feature"].tolist()

    if not high_vif:
        logger.info("  No high-VIF features found -- all features retained")
        return features

    # Keep features below threshold
    keep = vif_df[vif_df["vif"] <= threshold]["feature"].tolist()
    dropped = [f for f in features if f not in keep and f in high_vif]

    logger.info(f"  Dropped {len(dropped)} high-VIF features: {dropped}")
    logger.info(f"  Retained: {len(keep)} features")

    return keep


def run_feature_selection(df: pd.DataFrame):
    logger = logging.getLogger("selection")
    logger.info("=" * 62)
    logger.info("  FEATURE SELECTION")
    logger.info("=" * 62)

    # Step 1: Correlations
    correlations = compute_correlations(df, logger)

    # Step 2: Filter weak
    strong_features = filter_weak_features(correlations, threshold=0.15, logger=logger)

    # Step 3: VIF
    vif_df = compute_vif(df, strong_features, logger)

    # Step 4: Drop high VIF
    final_features = drop_high_vif(vif_df, strong_features, threshold=10.0, logger=logger)

    # Build config
    feature_metadata = {}
    for f in final_features:
        corr_val = float(correlations.get(f, 0))
        vif_row = vif_df[vif_df["feature"] == f]
        vif_val = float(vif_row["vif"].values[0]) if len(vif_row) > 0 else None
        feature_metadata[f] = {"correlation": round(corr_val, 4), "vif": round(vif_val, 2) if vif_val else None}

    logger.info(f"\n  FINAL: {len(final_features)} features selected")
    return final_features, feature_metadata, correlations, vif_df


# ═══════════════════════════════════════════════════════════════════════
# 2. EDA PLOTS
# ═══════════════════════════════════════════════════════════════════════

def plot_acf_pacf(df: pd.DataFrame, logger: logging.Logger):
    """Generate ACF and PACF plots for the target variable (United States)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    logger.info("-- Plotting ACF/PACF --")

    country = "United States"
    ts = df[df["country"] == country].sort_values("date")[TARGET_COL].dropna().values

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(f"ACF & PACF -- {TARGET_COL} ({country})", fontsize=15, fontweight="bold", y=0.98)

    plot_acf(ts, lags=40, ax=axes[0], color="#2196F3", alpha=0.05)
    axes[0].set_title("Autocorrelation Function (ACF)", fontsize=12)
    axes[0].set_xlabel("Lag (days)")
    axes[0].set_ylabel("Correlation")
    axes[0].axhline(y=0, color="k", linewidth=0.5)

    plot_pacf(ts, lags=40, ax=axes[1], color="#FF5722", alpha=0.05, method="ywm")
    axes[1].set_title("Partial Autocorrelation Function (PACF)", fontsize=12)
    axes[1].set_xlabel("Lag (days)")
    axes[1].set_ylabel("Correlation")
    axes[1].axhline(y=0, color="k", linewidth=0.5)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = PLOT_DIR / "acf_pacf.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"    Saved: {path.name}")
    return path


def plot_correlation_heatmap(df: pd.DataFrame, features: list, logger: logging.Logger):
    """Generate a correlation heatmap of selected features + target."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logger.info("-- Plotting Correlation Heatmap --")

    cols = [TARGET_COL] + [f for f in features if f in df.columns][:25]  # limit to 25 for readability
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)

    # Add correlation values
    for i in range(len(corr)):
        for j in range(len(corr)):
            val = corr.values[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson Correlation", fontsize=10)

    ax.set_title("Feature Correlation Heatmap", fontsize=15, fontweight="bold", pad=15)
    plt.tight_layout()

    path = PLOT_DIR / "correlation_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"    Saved: {path.name}")
    return path


def plot_feature_trends(df: pd.DataFrame, features: list, logger: logging.Logger):
    """Plot trends of key features alongside the target for one country."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logger.info("-- Plotting Feature Trends --")

    country = "United States"
    cdf = df[df["country"] == country].sort_values("date").copy()

    # Select interesting features to plot
    trend_features = []
    priority = ["rt_estimate", "stringency_index", "growth_rate_7d", "mobility_composite_lag7",
                "rolling_mean_7", "doubling_time", "positive_rate", "ewm_mean_7"]
    for f in priority:
        if f in features and f in cdf.columns:
            trend_features.append(f)
    if len(trend_features) < 6:
        for f in features:
            if f not in trend_features and f in cdf.columns:
                trend_features.append(f)
            if len(trend_features) >= 6:
                break

    n_plots = min(len(trend_features), 6)
    fig, axes = plt.subplots(n_plots + 1, 1, figsize=(16, 3.5 * (n_plots + 1)), sharex=True)
    fig.suptitle(f"Feature Trends -- {country}", fontsize=16, fontweight="bold", y=1.0)

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#00BCD4", "#E91E63"]

    # Target plot
    axes[0].plot(cdf["date"], cdf[TARGET_COL], color="#1565C0", linewidth=1.5, label=TARGET_COL)
    axes[0].fill_between(cdf["date"], cdf[TARGET_COL], alpha=0.15, color="#1565C0")
    axes[0].set_ylabel(TARGET_COL, fontsize=9)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title("Target Variable", fontsize=11, fontweight="bold")

    # Feature plots
    for i, feat in enumerate(trend_features[:n_plots]):
        ax = axes[i + 1]
        ax.plot(cdf["date"], cdf[feat], color=colors[i % len(colors)], linewidth=1.2, label=feat)
        ax.fill_between(cdf["date"], cdf[feat], alpha=0.1, color=colors[i % len(colors)])
        ax.set_ylabel(feat, fontsize=8)
        ax.legend(loc="upper right", fontsize=8)

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Date", fontsize=11)
    plt.tight_layout()

    path = PLOT_DIR / "feature_trends.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"    Saved: {path.name}")
    return path


def plot_target_by_country(df: pd.DataFrame, logger: logging.Logger):
    """Plot target variable across all countries."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logger.info("-- Plotting Target by Country --")

    fig, ax = plt.subplots(figsize=(16, 6))
    colors = ["#1565C0", "#E65100", "#2E7D32", "#6A1B9A", "#C62828"]

    for i, country in enumerate(config.TARGET_COUNTRIES):
        cdf = df[df["country"] == country].sort_values("date")
        ax.plot(cdf["date"], cdf[TARGET_COL], color=colors[i % len(colors)],
                linewidth=1.3, label=country, alpha=0.85)

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("7-Day Avg New Cases", fontsize=11)
    ax.set_title("COVID-19 Cases: 7-Day Rolling Average by Country",
                 fontsize=14, fontweight="bold", pad=10)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = PLOT_DIR / "target_by_country.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"    Saved: {path.name}")
    return path


def run_stationarity_test(df: pd.DataFrame, logger: logging.Logger):
    """Run Augmented Dickey-Fuller test for stationarity on the target."""
    from statsmodels.tsa.stattools import adfuller

    logger.info("-- Stationarity Test (ADF) --")
    results = {}

    for country in config.TARGET_COUNTRIES:
        ts = df[df["country"] == country].sort_values("date")[TARGET_COL].dropna().values
        adf_stat, p_value, used_lag, nobs, crit_values, icbest = adfuller(ts, maxlag=30)

        stationary = p_value < 0.05
        results[country] = {
            "adf_statistic": round(float(adf_stat), 4),
            "p_value": round(float(p_value), 6),
            "used_lag": int(used_lag),
            "stationary": stationary,
            "critical_values": {k: round(float(v), 4) for k, v in crit_values.items()},
        }
        status = "STATIONARY" if stationary else "NON-STATIONARY"
        logger.info(f"    {country:20s} ADF={adf_stat:.4f}, p={p_value:.6f} -> {status}")

    return results


def plot_stationarity_summary(adf_results: dict, logger: logging.Logger):
    """Plot stationarity test results as a summary chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logger.info("-- Plotting Stationarity Summary --")

    countries = list(adf_results.keys())
    p_values = [adf_results[c]["p_value"] for c in countries]
    colors_bar = ["#4CAF50" if p < 0.05 else "#F44336" for p in p_values]
    labels = ["Stationary" if p < 0.05 else "Non-stationary" for p in p_values]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.barh(countries, p_values, color=colors_bar, edgecolor="white", height=0.6)
    ax.axvline(x=0.05, color="#333", linewidth=2, linestyle="--", label="Significance (p=0.05)")

    for bar, lbl, pv in zip(bars, labels, p_values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f" {lbl} (p={pv:.4f})", va="center", fontsize=9)

    ax.set_xlabel("p-value (ADF Test)", fontsize=11)
    ax.set_title("Augmented Dickey-Fuller Stationarity Test", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    plt.tight_layout()
    path = PLOT_DIR / "stationarity_test.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"    Saved: {path.name}")
    return path


def plot_feature_importance_bar(feature_metadata: dict, logger: logging.Logger):
    """Bar chart of feature correlations with target."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logger.info("-- Plotting Feature Importance --")

    sorted_feats = sorted(feature_metadata.items(), key=lambda x: x[1]["correlation"], reverse=True)
    names = [f[0] for f in sorted_feats[:25]]
    corrs = [f[1]["correlation"] for f in sorted_feats[:25]]

    fig, ax = plt.subplots(figsize=(12, max(6, len(names) * 0.35)))
    colors_bar = ["#1565C0" if c > 0.3 else "#42A5F5" if c > 0.2 else "#90CAF9" for c in corrs]

    bars = ax.barh(range(len(names)), corrs, color=colors_bar, edgecolor="white", height=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()

    for bar, corr in zip(bars, corrs):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{corr:.3f}", va="center", fontsize=8)

    ax.set_xlabel("|Correlation| with Target", fontsize=11)
    ax.set_title("Selected Features: Correlation with Target Variable",
                 fontsize=14, fontweight="bold", pad=10)
    ax.grid(True, alpha=0.3, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = PLOT_DIR / "feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"    Saved: {path.name}")
    return path


# ═══════════════════════════════════════════════════════════════════════
# 3. SAVE CONFIG
# ═══════════════════════════════════════════════════════════════════════

def save_feature_config(features, metadata, adf_results):
    """Save final feature selection config as JSON."""
    config_data = {
        "target": TARGET_COL,
        "n_features": len(features),
        "selected_features": features,
        "feature_details": metadata,
        "stationarity_tests": adf_results,
        "selection_criteria": {
            "min_correlation": 0.15,
            "max_vif": 10.0,
            "method": "correlation_filter + VIF_multicollinearity_check",
        },
    }

    path = config.OUTPUT_DIR / "feature_config.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2, default=str)

    return path


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    t_start = time.perf_counter()

    # Load feature matrix
    src = config.OUTPUT_DIR / "feature_matrix.parquet"
    if not src.exists():
        logger.error(f"Feature matrix not found: {src}. Run run_features.py first.")
        return 1

    logger.info(f"Loading feature matrix from {src.name}")
    df = pd.read_parquet(src)
    logger.info(f"  Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")

    # ── Feature Selection ──
    final_features, feature_metadata, correlations, vif_df = run_feature_selection(df)

    # ── EDA Plots ──
    logger.info("")
    logger.info("=" * 62)
    logger.info("  EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 62)

    plot_acf_pacf(df, logger)
    plot_correlation_heatmap(df, final_features, logger)
    plot_feature_trends(df, final_features, logger)
    plot_target_by_country(df, logger)

    adf_results = run_stationarity_test(df, logger)
    plot_stationarity_summary(adf_results, logger)
    plot_feature_importance_bar(feature_metadata, logger)

    # ── Save Config ──
    cfg_path = save_feature_config(final_features, feature_metadata, adf_results)
    logger.info(f"\n  Feature config saved: {cfg_path}")

    elapsed = time.perf_counter() - t_start

    # ── Summary ──
    print("")
    print("=" * 62)
    print("  FEATURE SELECTION & EDA COMPLETE")
    print("=" * 62)
    print(f"  Time:              {elapsed:.2f}s")
    print(f"  Features selected: {len(final_features)}")
    print(f"  Plots generated:   {len(list(PLOT_DIR.glob('*.png')))}")
    print(f"  Config:            {cfg_path}")
    print(f"  Plots dir:         {PLOT_DIR}")
    print("=" * 62)

    return 0


if __name__ == "__main__":
    sys.exit(main())
