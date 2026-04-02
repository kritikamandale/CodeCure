"""
Merger Module
=============
Merges JHU, OWID, and Google Mobility datasets on [country, date].
Handles country name harmonization and feature selection.
"""

import logging

import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


def prepare_owid(owid_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare OWID data for merging:
      - Rename 'location' → 'country'
      - Parse dates
      - Apply country name mapping
      - Drop non-country aggregates (World, continents, etc.)
    """
    logger.info("── Preparing OWID for merge ──")
    
    df = owid_df.copy()
    
    # Rename
    df.rename(columns={"location": "country"}, inplace=True)
    
    # Parse dates
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    
    # Harmonize country names
    df["country"] = df["country"].replace(config.COUNTRY_NAME_MAP)
    
    # Drop aggregates (OWID includes World, continent-level rows)
    aggregates = [
        "World", "Africa", "Asia", "Europe", "North America",
        "South America", "Oceania", "European Union",
        "High income", "Low income", "Lower middle income",
        "Upper middle income", "International",
    ]
    df = df[~df["country"].isin(aggregates)].copy()
    
    logger.info(f"  OWID ready: {df.shape[0]:,} rows, {df['country'].nunique():,} countries")
    return df


def prepare_mobility(mobility_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare Google Mobility data for merging:
      - Rename columns to short names
      - Rename 'country_region' → 'country'
      - Parse dates
      - Aggregate sub-national data to country level (mean)
    """
    logger.info("── Preparing Mobility for merge ──")
    
    df = mobility_df.copy()
    
    # Rename mobility columns
    df.rename(columns=config.MOBILITY_RENAME, inplace=True)
    df.rename(columns={"country_region": "country"}, inplace=True)
    
    # Parse dates
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    
    # Harmonize country names
    df["country"] = df["country"].replace(config.COUNTRY_NAME_MAP)
    
    # Aggregate sub-national to country level
    mobility_cols = list(config.MOBILITY_RENAME.values())
    existing_mob_cols = [c for c in mobility_cols if c in df.columns]
    
    df = (
        df.groupby(["country", "date"], as_index=False)[existing_mob_cols]
        .mean()
    )
    
    logger.info(f"  Mobility ready: {df.shape[0]:,} rows, {df['country'].nunique():,} countries")
    return df


def merge_all_datasets(
    jhu_df: pd.DataFrame,
    owid_df: pd.DataFrame,
    mobility_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge all three datasets on [country, date] using left joins.
    JHU is the base (leftmost) since it has the most complete country/date coverage.
    
    Features included:
      - JHU: confirmed, deaths, recovered, daily_new_cases, cases_7day_avg, ...
      - OWID: vaccinations, tests_per_thousand, stringency_index, demographics
      - Mobility: retail, transit, workplaces, ...
    """
    logger.info("═══ Merging Datasets ═══")
    
    # Prepare auxiliary datasets
    owid_ready = prepare_owid(owid_df)
    mobility_ready = prepare_mobility(mobility_df)
    
    # ── Merge JHU + OWID ──
    # Drop JHU columns that OWID also provides (avoid _x/_y suffixes)
    owid_merge_cols = [c for c in owid_ready.columns if c not in jhu_df.columns or c in ["country", "date"]]
    
    merged = jhu_df.merge(
        owid_ready[owid_merge_cols],
        on=["country", "date"],
        how="left",
    )
    logger.info(f"  After JHU + OWID: {merged.shape[0]:,} rows × {merged.shape[1]:,} cols")
    
    # ── Merge + Mobility ──
    merged = merged.merge(
        mobility_ready,
        on=["country", "date"],
        how="left",
    )
    logger.info(f"  After + Mobility: {merged.shape[0]:,} rows × {merged.shape[1]:,} cols")
    
    # ── Ensure temporal order (no leakage) ──
    merged.sort_values(["country", "date"], inplace=True)
    merged.reset_index(drop=True, inplace=True)
    
    # ── Report coverage ──
    total_cells = merged.shape[0] * merged.shape[1]
    null_cells = merged.isnull().sum().sum()
    logger.info(f"  Coverage: {(1 - null_cells / total_cells) * 100:.1f}% non-null")
    logger.info(f"  Countries: {merged['country'].nunique():,}")
    logger.info(f"  Date range: {merged['date'].min()} → {merged['date'].max()}")
    
    return merged
