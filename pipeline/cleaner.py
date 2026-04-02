"""
Cleaner Module
==============
Handles missing value imputation and memory optimization.
  - Forward fill per country
  - Linear interpolation (max 3-day gaps)
  - Downcast floats to float32
"""

import logging

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values per country:
      1. Forward fill (ffill) – carries last known value forward
      2. Linear interpolation (limit=3) – fills short gaps smoothly
      3. Backward fill remaining leading NaNs
    
    Preserves temporal order to prevent data leakage.
    """
    logger.info("═══ Handling Missing Values ═══")
    
    # Report pre-imputation stats
    null_before = df.isnull().sum()
    total_nulls_before = null_before.sum()
    logger.info(f"  Total NaNs before: {total_nulls_before:,}")
    
    # Show top-10 null columns
    top_nulls = null_before[null_before > 0].sort_values(ascending=False).head(10)
    for col, count in top_nulls.items():
        pct = count / len(df) * 100
        logger.info(f"    {col}: {count:,} ({pct:.1f}%)")
    
    # Ensure sorted
    df = df.sort_values(["country", "date"]).copy()
    
    # Get numeric columns (don't interpolate non-numeric)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # ── Step 1: Forward fill per country ──
    df[numeric_cols] = df.groupby("country")[numeric_cols].transform(
        lambda x: x.ffill()
    )
    
    # ── Step 2: Linear interpolation (max 3-day gaps) ──
    df[numeric_cols] = df.groupby("country")[numeric_cols].transform(
        lambda x: x.interpolate(method="linear", limit=config.INTERPOLATION_LIMIT, limit_direction="forward")
    )
    
    # ── Step 3: Backward fill remaining leading NaNs ──
    df[numeric_cols] = df.groupby("country")[numeric_cols].transform(
        lambda x: x.bfill()
    )
    
    # ── Step 4: Fill any remaining NaNs with 0 (edge cases) ──
    remaining = df[numeric_cols].isnull().sum().sum()
    if remaining > 0:
        logger.info(f"  Filling {remaining:,} remaining NaNs with 0")
        df[numeric_cols] = df[numeric_cols].fillna(0)
    
    total_nulls_after = df.isnull().sum().sum()
    logger.info(f"  Total NaNs after: {total_nulls_after:,}")
    logger.info(f"  Imputed: {total_nulls_before - total_nulls_after:,} values")
    
    return df


def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast numeric columns to float32 to reduce memory footprint.
    Reports memory usage before/after.
    """
    logger.info("═══ Optimizing Memory ═══")
    
    mem_before = df.memory_usage(deep=True).sum() / 1e6
    
    # Downcast float columns
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype(config.FLOAT_DTYPE)
    
    # Downcast integer columns where safe
    int_cols = df.select_dtypes(include=["int64"]).columns
    for col in int_cols:
        col_min, col_max = df[col].min(), df[col].max()
        if col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
            df[col] = df[col].astype(np.int32)
    
    mem_after = df.memory_usage(deep=True).sum() / 1e6
    reduction = (1 - mem_after / mem_before) * 100
    
    logger.info(f"  Memory: {mem_before:.1f} MB → {mem_after:.1f} MB ({reduction:.1f}% reduction)")
    
    return df
