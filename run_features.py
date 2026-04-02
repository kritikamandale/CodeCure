"""
Feature Engineering Runner
==========================
Loads the master dataset, runs feature engineering, and exports feature_matrix.parquet.

Usage:
    python run_features.py
"""

import logging
import sys
import time
import io
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import config
from pipeline.feature_engineer import engineer_features


def setup_logging():
    log_format = "%(asctime)s | %(levelname)-7s | %(message)s"
    date_format = "%H:%M:%S"

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format, date_format))

    file_handler = logging.FileHandler(
        config.LOGS_DIR / "feature_engineering.log", mode="w", encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))

    logging.basicConfig(level=logging.DEBUG, handlers=[console, file_handler])


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    t_start = time.perf_counter()

    # ── Load master dataset ──
    src = config.OUTPUT_DIR / "master_dataset.parquet"
    if not src.exists():
        logger.error(f"Master dataset not found at {src}. Run run_pipeline.py first.")
        return 1

    logger.info(f"Loading master dataset from {src.name}")
    df = pd.read_parquet(src)
    logger.info(f"  Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")

    # ── Run feature engineering ──
    df = engineer_features(df)

    # ── Export ──
    out_parquet = config.OUTPUT_DIR / "feature_matrix.parquet"
    out_csv = config.OUTPUT_DIR / "feature_matrix.csv"

    df.to_parquet(out_parquet, index=False, engine="pyarrow", compression="snappy")
    pq_size = out_parquet.stat().st_size / 1e6
    logger.info(f"  Saved: {out_parquet.name} ({pq_size:.1f} MB)")

    df.to_csv(out_csv, index=False)
    csv_size = out_csv.stat().st_size / 1e6
    logger.info(f"  Saved: {out_csv.name} ({csv_size:.1f} MB)")

    # ── Per-country feature matrices ──
    logger.info("Exporting per-country feature matrices...")
    for country in config.TARGET_COUNTRIES:
        cdf = df[df["country"] == country]
        if cdf.empty:
            continue
        safe = country.lower().replace(" ", "_")
        cpath = config.COUNTRY_OUTPUT_DIR / f"{safe}_features.parquet"
        cdf.to_parquet(cpath, index=False, engine="pyarrow", compression="snappy")
        logger.info(f"    {country}: {cdf.shape[0]:,} rows x {cdf.shape[1]} cols")

    elapsed = time.perf_counter() - t_start

    # ── Final summary ──
    print("")
    print("=" * 62)
    print("  FEATURE MATRIX READY")
    print("=" * 62)
    print(f"  Time:          {elapsed:.2f}s")
    print(f"  Shape:         {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"  Parquet:       {pq_size:.1f} MB")
    print(f"  CSV:           {csv_size:.1f} MB")
    print(f"  Nulls:         {df.isnull().sum().sum()}")
    print(f"  Countries:     {df['country'].nunique()}")
    print(f"  Date range:    {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"  Output:        {out_parquet}")
    print("=" * 62)

    return 0


if __name__ == "__main__":
    sys.exit(main())
