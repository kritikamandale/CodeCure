"""
Exporter Module
===============
Exports the final dataset as:
  - master_dataset.csv / .parquet
  - Per-country datasets for target countries
  - Data quality report
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


def validate_temporal_order(df: pd.DataFrame) -> bool:
    """
    Validate no temporal leakage:
      - Dates are strictly ascending per country
      - No duplicate (country, date) pairs
    """
    logger.info("── Validating temporal order ──")
    
    # Check duplicates
    dupes = df.duplicated(subset=["country", "date"], keep=False)
    if dupes.any():
        n_dupes = dupes.sum()
        logger.warning(f"  ⚠ Found {n_dupes:,} duplicate (country, date) rows — deduplicating")
        df = df.drop_duplicates(subset=["country", "date"], keep="last")
    
    # Check ascending order per country
    is_sorted = df.groupby("country")["date"].apply(
        lambda x: x.is_monotonic_increasing
    )
    unsorted_countries = is_sorted[~is_sorted].index.tolist()
    if unsorted_countries:
        logger.warning(f"  ⚠ Non-monotonic dates in: {unsorted_countries}")
        return False
    
    logger.info("  ✓ Temporal order validated — no leakage")
    return True


def generate_quality_report(df: pd.DataFrame, output_path: Path) -> None:
    """Generate a data quality report summarizing the dataset."""
    logger.info("── Generating Data Quality Report ──")
    
    report_lines = [
        "╔══════════════════════════════════════════════════════════════╗",
        "║          CODECURE DATA PIPELINE — QUALITY REPORT            ║",
        "╚══════════════════════════════════════════════════════════════╝",
        "",
        f"Generated: {pd.Timestamp.now().isoformat()}",
        f"Total rows: {df.shape[0]:,}",
        f"Total columns: {df.shape[1]:,}",
        f"Countries: {df['country'].nunique():,}",
        f"Date range: {df['date'].min()} → {df['date'].max()}",
        f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB",
        "",
        "── Column Summary ──",
    ]
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        nulls = df[col].isnull().sum()
        null_pct = nulls / len(df) * 100
        
        if df[col].dtype in [np.float32, np.float64, np.int32, np.int64]:
            stats = f"  min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}"
        else:
            stats = f"  unique={df[col].nunique()}"
        
        report_lines.append(f"  {col} ({dtype}): {nulls:,} nulls ({null_pct:.1f}%){stats}")
    
    # Anomaly summary
    anomaly_cols = [c for c in df.columns if c.startswith("anomaly_")]
    if anomaly_cols:
        report_lines.append("")
        report_lines.append("── Anomaly Summary ──")
        for col in anomaly_cols:
            n = df[col].sum()
            report_lines.append(f"  {col}: {n:,} flagged ({n / len(df) * 100:.2f}%)")
        
        if "is_anomaly" in df.columns:
            total = df["is_anomaly"].sum()
            report_lines.append(f"  Total anomalous rows: {total:,} ({total / len(df) * 100:.2f}%)")
    
    # Country breakdown
    report_lines.append("")
    report_lines.append("── Per-Country Row Counts ──")
    country_counts = df["country"].value_counts().head(20)
    for country, count in country_counts.items():
        report_lines.append(f"  {country}: {count:,} rows")
    
    report_text = "\n".join(report_lines)
    
    output_path.write_text(report_text, encoding="utf-8")
    logger.info(f"  ✓ Quality report saved: {output_path.name}")
    
    # Also print to console
    print("\n" + report_text + "\n")


def export_master_dataset(df: pd.DataFrame) -> None:
    """
    Export the complete master dataset as CSV and Parquet.
    Validates temporal order before export.
    """
    logger.info("═══ Exporting Master Dataset ═══")
    
    # Validate
    validate_temporal_order(df)
    
    # ── CSV ──
    csv_path = config.OUTPUT_DIR / "master_dataset.csv"
    df.to_csv(csv_path, index=False)
    csv_size = csv_path.stat().st_size / 1e6
    logger.info(f"  ✓ CSV: {csv_path.name} ({csv_size:.1f} MB)")
    
    # ── Parquet (much faster for downstream ML) ──
    parquet_path = config.OUTPUT_DIR / "master_dataset.parquet"
    df.to_parquet(parquet_path, index=False, engine="pyarrow", compression="snappy")
    parquet_size = parquet_path.stat().st_size / 1e6
    logger.info(f"  ✓ Parquet: {parquet_path.name} ({parquet_size:.1f} MB)")
    logger.info(f"  Size ratio: Parquet is {csv_size / parquet_size:.1f}x smaller than CSV")
    
    # ── Quality report ──
    report_path = config.OUTPUT_DIR / "data_quality_report.txt"
    generate_quality_report(df, report_path)


def export_country_datasets(df: pd.DataFrame) -> None:
    """
    Export per-country datasets for the target countries.
    Each as CSV + Parquet.
    """
    logger.info("═══ Exporting Per-Country Datasets ═══")
    
    for country in config.TARGET_COUNTRIES:
        country_df = df[df["country"] == country].copy()
        
        if country_df.empty:
            logger.warning(f"  ⚠ No data for '{country}' — skipping")
            continue
        
        # Sanitize filename
        safe_name = country.lower().replace(" ", "_")
        
        # CSV
        csv_path = config.COUNTRY_OUTPUT_DIR / f"{safe_name}.csv"
        country_df.to_csv(csv_path, index=False)
        
        # Parquet
        parquet_path = config.COUNTRY_OUTPUT_DIR / f"{safe_name}.parquet"
        country_df.to_parquet(parquet_path, index=False, engine="pyarrow", compression="snappy")
        
        logger.info(
            f"  ✓ {country}: {country_df.shape[0]:,} rows × {country_df.shape[1]:,} cols "
            f"({csv_path.stat().st_size / 1e6:.1f} MB csv, "
            f"{parquet_path.stat().st_size / 1e6:.1f} MB parquet)"
        )
    
    logger.info(f"  Country datasets saved to: {config.COUNTRY_OUTPUT_DIR}")
