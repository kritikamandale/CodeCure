"""
Feature Engineering Module
==========================
Generates high-quality engineered features for epidemic forecasting.
All operations are vectorized and maintain strict temporal order (no leakage).

Feature groups:
  1. Temporal (cyclical sin/cos encoding)
  2. Lag features (1, 3, 7, 14 day lags of target)
  3. Rolling statistics (mean, std, max over 7-day window)
  4. Epidemiological (growth rate, doubling time, CFR, Rt)
  5. Mobility (7-day lagged mobility variables)
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

TARGET_COL = "cases_7day_avg"


# ═══════════════════════════════════════════════════════════════════════
# 1. TEMPORAL FEATURES
# ═══════════════════════════════════════════════════════════════════════

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical temporal encodings (no leakage — derived from date only).

      - day_of_week_sin, day_of_week_cos  (period = 7)
      - month_sin, month_cos              (period = 12)
      - day_of_year_sin, day_of_year_cos  (period = 365.25)
      - days_since_outbreak               (integer, per country)
      - is_weekend                        (boolean flag)
    """
    logger.info("── [1/5] Temporal Features ──")

    dt = df["date"].dt

    # Day of week: sin/cos encoding (period = 7)
    dow = dt.dayofweek.astype(np.float32)
    df["day_of_week_sin"] = np.sin(2 * np.pi * dow / 7).astype(np.float32)
    df["day_of_week_cos"] = np.cos(2 * np.pi * dow / 7).astype(np.float32)

    # Month: sin/cos encoding (period = 12)
    month = dt.month.astype(np.float32)
    df["month_sin"] = np.sin(2 * np.pi * month / 12).astype(np.float32)
    df["month_cos"] = np.cos(2 * np.pi * month / 12).astype(np.float32)

    # Day of year: sin/cos encoding (period = 365.25, captures seasonality)
    doy = dt.dayofyear.astype(np.float32)
    df["day_of_year_sin"] = np.sin(2 * np.pi * doy / 365.25).astype(np.float32)
    df["day_of_year_cos"] = np.cos(2 * np.pi * doy / 365.25).astype(np.float32)

    # Days since outbreak: per-country integer count from first date
    df["days_since_outbreak"] = df.groupby("country")["date"].transform(
        lambda x: (x - x.min()).dt.days
    ).astype(np.int32)

    # Weekend flag
    df["is_weekend"] = (dow >= 5).astype(np.int8)

    n_features = 8
    logger.info(f"    Added {n_features} temporal features")
    return df


# ═══════════════════════════════════════════════════════════════════════
# 2. LAG FEATURES
# ═══════════════════════════════════════════════════════════════════════

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lagged values of the target variable per country.
    Lags are strictly backward-looking (no future leakage).

      - lag_1, lag_3, lag_7, lag_14   (target variable shifted)
      - lag_1_deaths, lag_7_deaths    (deaths shifted)
    """
    logger.info("── [2/5] Lag Features ──")

    lag_configs = [
        (TARGET_COL, [1, 3, 7, 14]),
        ("daily_new_deaths", [1, 7]),
    ]

    n_features = 0
    for col, lags in lag_configs:
        if col not in df.columns:
            logger.warning(f"    Column '{col}' not found, skipping lags")
            continue

        for lag in lags:
            col_name = f"lag_{lag}" if col == TARGET_COL else f"lag_{lag}_{col.replace('daily_new_', '')}"
            df[col_name] = df.groupby("country")[col].shift(lag).astype(np.float32)
            n_features += 1

    logger.info(f"    Added {n_features} lag features")
    return df


# ═══════════════════════════════════════════════════════════════════════
# 3. ROLLING FEATURES
# ═══════════════════════════════════════════════════════════════════════

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling window statistics of the target variable.
    Uses ONLY past data (no center=True) to prevent leakage.

      - rolling_mean_7, rolling_std_7, rolling_max_7
      - rolling_mean_14, rolling_std_14
      - rolling_mean_21
      - rolling_change_7  (pct change of rolling mean)
      - ewm_mean_7        (exponentially weighted mean, span=7)
    """
    logger.info("── [3/5] Rolling Features ──")

    n_features = 0

    # 7-day rolling stats
    for stat, method in [("mean", "mean"), ("std", "std"), ("max", "max")]:
        col_name = f"rolling_{stat}_7"
        df[col_name] = (
            df.groupby("country")[TARGET_COL]
            .transform(lambda x: getattr(x.rolling(7, min_periods=1), method)())
            .astype(np.float32)
        )
        n_features += 1

    # 14-day rolling mean and std
    for stat, method in [("mean", "mean"), ("std", "std")]:
        col_name = f"rolling_{stat}_14"
        df[col_name] = (
            df.groupby("country")[TARGET_COL]
            .transform(lambda x: getattr(x.rolling(14, min_periods=1), method)())
            .astype(np.float32)
        )
        n_features += 1

    # 21-day rolling mean (longer trend)
    df["rolling_mean_21"] = (
        df.groupby("country")[TARGET_COL]
        .transform(lambda x: x.rolling(21, min_periods=1).mean())
        .astype(np.float32)
    )
    n_features += 1

    # Percentage change of 7-day rolling mean (momentum indicator)
    df["rolling_change_7"] = (
        df.groupby("country")["rolling_mean_7"]
        .transform(lambda x: x.pct_change(periods=7))
        .astype(np.float32)
    )
    n_features += 1

    # Exponentially weighted mean (span=7, more weight on recent data)
    df["ewm_mean_7"] = (
        df.groupby("country")[TARGET_COL]
        .transform(lambda x: x.ewm(span=7, min_periods=1).mean())
        .astype(np.float32)
    )
    n_features += 1

    logger.info(f"    Added {n_features} rolling features")
    return df


# ═══════════════════════════════════════════════════════════════════════
# 4. EPIDEMIOLOGICAL FEATURES
# ═══════════════════════════════════════════════════════════════════════

def add_epi_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add epidemiological indicator features:

      - growth_rate:       (new_cases_today / new_cases_yesterday) - 1
      - growth_rate_7d:    7-day smoothed growth rate
      - doubling_time:     ln(2) / ln(1 + growth_rate_7d)
      - case_fatality_rate: cumulative deaths / cumulative confirmed
      - recovery_rate:     cumulative recovered / cumulative confirmed
      - Rt_estimate:       ratio of new cases (7d sum now / 7d sum prior)
      - active_ratio:      active cases / confirmed cases
      - cases_per_million: daily new cases per million population
    """
    logger.info("── [4/5] Epidemiological Features ──")

    n_features = 0

    # ── Growth rate (daily) ──
    prev_cases = df.groupby("country")[TARGET_COL].shift(1)
    df["growth_rate"] = np.where(
        prev_cases > 0,
        (df[TARGET_COL] - prev_cases) / prev_cases,
        0.0,
    ).astype(np.float32)
    n_features += 1

    # ── Smoothed growth rate (7-day average) ──
    df["growth_rate_7d"] = (
        df.groupby("country")["growth_rate"]
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
        .astype(np.float32)
    )
    n_features += 1

    # ── Doubling time: ln(2) / ln(1 + growth_rate_7d) ──
    gr = df["growth_rate_7d"].clip(lower=0.001)  # avoid log(1+0)=0 division
    df["doubling_time"] = (np.log(2) / np.log(1 + gr)).astype(np.float32)
    # Cap at reasonable bounds (0.5 to 365 days)
    df["doubling_time"] = df["doubling_time"].clip(0.5, 365.0)
    n_features += 1

    # ── Case fatality rate (cumulative) ──
    if all(c in df.columns for c in ["deaths", "confirmed"]):
        df["cfr"] = np.where(
            df["confirmed"] > 0,
            df["deaths"] / df["confirmed"],
            0.0,
        ).astype(np.float32)
        n_features += 1

    # ── Recovery rate ──
    if all(c in df.columns for c in ["recovered", "confirmed"]):
        df["recovery_rate"] = np.where(
            df["confirmed"] > 0,
            df["recovered"] / df["confirmed"],
            0.0,
        ).astype(np.float32)
        n_features += 1

    # ── Rt estimate (7-day ratio method) ──
    # Rt ≈ (sum of new cases in last 7 days) / (sum of new cases in prior 7 days)
    if "daily_new_cases" in df.columns:
        sum_current_7 = (
            df.groupby("country")["daily_new_cases"]
            .transform(lambda x: x.rolling(7, min_periods=1).sum())
        )
        sum_prior_7 = (
            df.groupby("country")["daily_new_cases"]
            .transform(lambda x: x.shift(7).rolling(7, min_periods=1).sum())
        )
        df["rt_estimate"] = np.where(
            sum_prior_7 > 0,
            sum_current_7 / sum_prior_7,
            np.nan,
        ).astype(np.float32)
        n_features += 1

    # ── Active case ratio ──
    if all(c in df.columns for c in ["active_cases", "confirmed"]):
        df["active_ratio"] = np.where(
            df["confirmed"] > 0,
            df["active_cases"] / df["confirmed"],
            0.0,
        ).astype(np.float32)
        n_features += 1

    # ── Cases per million ──
    if "population" in df.columns:
        df["cases_per_million"] = (
            df[TARGET_COL] / (df["population"] / 1e6)
        ).astype(np.float32)
        n_features += 1

    logger.info(f"    Added {n_features} epidemiological features")
    return df


# ═══════════════════════════════════════════════════════════════════════
# 5. MOBILITY FEATURES (7-day lagged)
# ═══════════════════════════════════════════════════════════════════════

def add_mobility_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 7-day lagged mobility features.
    Mobility changes take roughly a week to impact case counts,
    so we lag them to align with epidemiological effect.

      - mobility_retail_lag7, mobility_transit_lag7, ...
      - mobility_composite_lag7 (average of key mobility metrics)
    """
    logger.info("── [5/5] Mobility Features (7-day lag) ──")

    mobility_cols = [c for c in df.columns if c.startswith("mobility_")]
    n_features = 0

    for col in mobility_cols:
        lagged_col = f"{col}_lag7"
        df[lagged_col] = (
            df.groupby("country")[col]
            .shift(7)
            .astype(np.float32)
        )
        n_features += 1

    # Composite mobility index (average of retail, transit, workplaces)
    composite_cols = ["mobility_retail_lag7", "mobility_transit_lag7", "mobility_workplaces_lag7"]
    existing = [c for c in composite_cols if c in df.columns]
    if existing:
        df["mobility_composite_lag7"] = df[existing].mean(axis=1).astype(np.float32)
        n_features += 1

    logger.info(f"    Added {n_features} lagged mobility features")
    return df


# ═══════════════════════════════════════════════════════════════════════
# POST-PROCESSING
# ═══════════════════════════════════════════════════════════════════════

def handle_feature_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle NaNs introduced by lag/rolling operations.
    Strategy:
      - Forward-fill within each country group
      - Fill remaining NaNs with 0
      - Replace inf values with NaN then 0
    """
    logger.info("── Post-processing: handling NaNs ──")

    # Replace infinities
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Count before
    nulls_before = df.isnull().sum().sum()

    # Forward fill per country
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df.groupby("country")[numeric_cols].transform(
        lambda x: x.ffill()
    )

    # Fill remaining with 0 (leading NaNs from lags)
    df[numeric_cols] = df[numeric_cols].fillna(0)

    nulls_after = df.isnull().sum().sum()
    logger.info(f"    NaNs: {nulls_before:,} -> {nulls_after:,}")

    return df


def optimize_feature_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast all numeric columns to float32 for memory efficiency."""
    logger.info("── Post-processing: optimizing dtypes ──")

    mem_before = df.memory_usage(deep=True).sum() / 1e6

    float64_cols = df.select_dtypes(include=["float64"]).columns
    df[float64_cols] = df[float64_cols].astype(np.float32)

    mem_after = df.memory_usage(deep=True).sum() / 1e6
    logger.info(f"    Memory: {mem_before:.1f} MB -> {mem_after:.1f} MB")

    return df


# ═══════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the complete feature engineering pipeline.
    Maintains strict temporal order throughout.
    """
    logger.info("=" * 60)
    logger.info("  FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 60)

    t_start = time.perf_counter()

    # Ensure temporal order
    df = df.sort_values(["country", "date"]).reset_index(drop=True)

    # Run all feature groups
    df = add_temporal_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_epi_features(df)
    df = add_mobility_lagged_features(df)

    # Post-processing
    df = handle_feature_missing_values(df)
    df = optimize_feature_dtypes(df)

    elapsed = time.perf_counter() - t_start

    # Summary
    feature_groups = {
        "temporal": [c for c in df.columns if c in [
            "day_of_week_sin", "day_of_week_cos", "month_sin", "month_cos",
            "day_of_year_sin", "day_of_year_cos", "days_since_outbreak", "is_weekend"
        ]],
        "lag": [c for c in df.columns if c.startswith("lag_")],
        "rolling": [c for c in df.columns if c.startswith("rolling_") or c.startswith("ewm_")],
        "epidemiological": [c for c in df.columns if c in [
            "growth_rate", "growth_rate_7d", "doubling_time", "cfr",
            "recovery_rate", "rt_estimate", "active_ratio", "cases_per_million"
        ]],
        "mobility_lagged": [c for c in df.columns if c.endswith("_lag7")],
    }

    total_new = sum(len(v) for v in feature_groups.values())

    logger.info("")
    logger.info("=" * 60)
    logger.info("  FEATURE ENGINEERING COMPLETE")
    logger.info(f"  Time: {elapsed:.2f}s")
    logger.info(f"  New features: {total_new}")
    logger.info(f"  Total columns: {len(df.columns)}")
    logger.info(f"  Shape: {df.shape[0]:,} x {df.shape[1]}")
    logger.info("-" * 60)
    for group, cols in feature_groups.items():
        logger.info(f"  {group}: {len(cols)} features")
        for c in cols:
            logger.info(f"      {c}")
    logger.info("=" * 60)

    return df
