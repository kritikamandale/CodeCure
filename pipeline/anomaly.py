"""
Anomaly Detection Module
========================
Detects and handles anomalous data points using z-score method.
  - Flag anomalies (z-score > 3) but DON'T remove
  - Smooth isolated spikes using local median replacement
"""

import logging

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect anomalies in key epidemiological columns using z-scores.
    
    Method:
      - Compute per-country rolling z-scores (window=21 days)
      - Flag rows where |z-score| > threshold (default: 3.0)
      - Create boolean flag columns: anomaly_<column>
    
    Anomalies are flagged but NOT removed (downstream models should
    be aware of them).
    """
    logger.info("═══ Detecting Anomalies ═══")
    
    target_cols = [
        "daily_new_cases",
        "daily_new_deaths",
        "cases_7day_avg",
    ]
    
    # Filter to columns that exist
    target_cols = [c for c in target_cols if c in df.columns]
    
    df = df.sort_values(["country", "date"]).copy()
    
    anomaly_summary = {}
    
    for col in target_cols:
        flag_col = f"anomaly_{col}"
        zscore_col = f"zscore_{col}"
        
        # Compute rolling mean and std per country (21-day window for stability)
        rolling_mean = df.groupby("country")[col].transform(
            lambda x: x.rolling(21, min_periods=7, center=True).mean()
        )
        rolling_std = df.groupby("country")[col].transform(
            lambda x: x.rolling(21, min_periods=7, center=True).std()
        )
        
        # Z-score
        df[zscore_col] = np.where(
            rolling_std > 0,
            (df[col] - rolling_mean) / rolling_std,
            0.0,
        )
        
        # Flag anomalies
        df[flag_col] = np.abs(df[zscore_col]) > config.ZSCORE_THRESHOLD
        
        n_anomalies = df[flag_col].sum()
        pct = n_anomalies / len(df) * 100
        anomaly_summary[col] = n_anomalies
        
        logger.info(f"  {col}: {n_anomalies:,} anomalies ({pct:.2f}%)")
    
    # Overall anomaly flag (any column)
    anomaly_flag_cols = [f"anomaly_{c}" for c in target_cols]
    df["is_anomaly"] = df[anomaly_flag_cols].any(axis=1)
    total_anomalous = df["is_anomaly"].sum()
    logger.info(f"  Total anomalous rows: {total_anomalous:,} ({total_anomalous / len(df) * 100:.2f}%)")
    
    return df


def smooth_spikes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Smooth isolated spikes using local median replacement.
    
    For each flagged anomaly, replace the value with the median of 
    a 5-day window centered on that point. This preserves trends
    while removing implausible single-day spikes.
    
    Creates smoothed columns alongside originals so both are available.
    """
    logger.info("═══ Smoothing Isolated Spikes ═══")
    
    target_cols = ["daily_new_cases", "daily_new_deaths"]
    target_cols = [c for c in target_cols if c in df.columns]
    
    df = df.sort_values(["country", "date"]).copy()
    
    smoothed_count = 0
    
    for col in target_cols:
        flag_col = f"anomaly_{col}"
        smoothed_col = f"{col}_smoothed"
        
        if flag_col not in df.columns:
            continue
        
        # Compute rolling median (5-day window)
        rolling_median = df.groupby("country")[col].transform(
            lambda x: x.rolling(5, min_periods=1, center=True).median()
        )
        
        # Replace anomalous values with median; keep non-anomalous as-is
        df[smoothed_col] = np.where(
            df[flag_col],
            rolling_median,
            df[col],
        )
        
        n_smoothed = df[flag_col].sum()
        smoothed_count += n_smoothed
        logger.info(f"  {col}: smoothed {n_smoothed:,} values")
    
    # Recompute 7-day rolling average on smoothed data
    if "daily_new_cases_smoothed" in df.columns:
        df["cases_7day_avg_smoothed"] = (
            df.groupby("country")["daily_new_cases_smoothed"]
            .transform(lambda x: x.rolling(config.ROLLING_WINDOW, min_periods=1).mean())
        )
        logger.info("  Recomputed cases_7day_avg_smoothed on cleaned data")
    
    logger.info(f"  Total smoothed values: {smoothed_count:,}")
    
    return df
