"""
Transformer Module
==================
Converts JHU wide → long format, aggregates by country/date,
computes daily new cases and 7-day rolling averages.
All operations are vectorized with pandas for performance.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


def transform_jhu_wide_to_long(jhu_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Convert JHU wide-format time series to long format.
    
    Input:  Province/State | Country/Region | Lat | Long | 1/22/20 | 1/23/20 | ...
    Output: country | date | confirmed | deaths | recovered
    """
    logger.info("═══ Transforming JHU: Wide → Long ═══")
    
    long_frames = {}
    date_cols_cache = None
    
    for metric, df in jhu_dfs.items():
        # Identify date columns (everything after Province/State, Country/Region, Lat, Long)
        id_cols = ["Province/State", "Country/Region", "Lat", "Long"]
        id_cols_present = [c for c in id_cols if c in df.columns]
        date_cols = [c for c in df.columns if c not in id_cols_present]
        
        if date_cols_cache is None:
            date_cols_cache = date_cols
        
        # Melt: wide → long (vectorized)
        melted = df.melt(
            id_vars=id_cols_present,
            value_vars=date_cols,
            var_name="date_str",
            value_name=metric,
        )
        
        # Parse dates
        melted["date"] = pd.to_datetime(melted["date_str"], format="mixed", dayfirst=False)
        melted.drop(columns=["date_str"], inplace=True)
        
        # Harmonize country names
        if "Country/Region" in melted.columns:
            melted["country"] = melted["Country/Region"].replace(config.COUNTRY_NAME_MAP)
        
        long_frames[metric] = melted
        logger.info(f"  {metric}: {melted.shape[0]:,} rows after melt")
    
    return long_frames


def aggregate_by_country_date(long_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Aggregate JHU data by country and date.
    Sums across provinces/states within a country.
    Then merges confirmed, deaths, recovered into one DataFrame.
    """
    logger.info("═══ Aggregating by Country × Date ═══")
    
    merged = None
    
    for metric, df in long_frames.items():
        # Group by country + date, sum the metric (handles province-level data)
        agg = (
            df.groupby(["country", "date"], as_index=False)[metric]
            .sum()
        )
        
        if merged is None:
            merged = agg
        else:
            merged = merged.merge(agg, on=["country", "date"], how="outer")
        
        logger.info(f"  {metric}: {agg['country'].nunique():,} countries")
    
    # Sort for temporal consistency (critical for no-leakage)
    merged.sort_values(["country", "date"], inplace=True)
    merged.reset_index(drop=True, inplace=True)
    
    logger.info(f"  Merged: {merged.shape[0]:,} rows × {merged.shape[1]:,} cols")
    logger.info(f"  Date range: {merged['date'].min()} → {merged['date'].max()}")
    logger.info(f"  Countries: {merged['country'].nunique():,}")
    
    return merged


def compute_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived features per country:
      - daily_new_cases: diff of confirmed (clipped to 0)
      - daily_new_deaths: diff of deaths (clipped to 0)
      - daily_new_recovered: diff of recovered (clipped to 0)
      - cases_7day_avg: 7-day rolling mean of daily_new_cases (TARGET VARIABLE)
      - deaths_7day_avg: 7-day rolling mean of daily_new_deaths
      - active_cases: confirmed - deaths - recovered
      - case_fatality_rate: deaths / confirmed
    
    All operations are vectorized per-group using groupby + transform.
    """
    logger.info("═══ Computing Daily Features ═══")
    
    # Ensure sorted by country, date
    df = df.sort_values(["country", "date"]).copy()
    
    # Daily diffs (vectorized per group)
    for cum_col, daily_col in [
        ("confirmed", "daily_new_cases"),
        ("deaths", "daily_new_deaths"),
        ("recovered", "daily_new_recovered"),
    ]:
        if cum_col in df.columns:
            df[daily_col] = df.groupby("country")[cum_col].diff()
            # Clip negative corrections to 0 (data corrections cause negatives)
            df[daily_col] = df[daily_col].clip(lower=0)
    
    # Rolling averages (target variable)
    window = config.ROLLING_WINDOW
    for daily_col, avg_col in [
        ("daily_new_cases", "cases_7day_avg"),
        ("daily_new_deaths", "deaths_7day_avg"),
    ]:
        if daily_col in df.columns:
            df[avg_col] = (
                df.groupby("country")[daily_col]
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )
    
    # Active cases
    if all(c in df.columns for c in ["confirmed", "deaths", "recovered"]):
        df["active_cases"] = df["confirmed"] - df["deaths"] - df["recovered"]
        df["active_cases"] = df["active_cases"].clip(lower=0)
    
    # Case fatality rate
    if all(c in df.columns for c in ["confirmed", "deaths"]):
        df["case_fatality_rate"] = np.where(
            df["confirmed"] > 0,
            df["deaths"] / df["confirmed"],
            0.0,
        )
    
    logger.info(f"  Added columns: {[c for c in df.columns if c not in ['country', 'date', 'confirmed', 'deaths', 'recovered']]}")
    logger.info(f"  Target variable 'cases_7day_avg' range: "
                f"{df['cases_7day_avg'].min():.1f} – {df['cases_7day_avg'].max():.1f}")
    
    return df
