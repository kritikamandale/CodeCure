"""
CodeCure — Epidemic Spread Prediction Data Pipeline
=====================================================
Main orchestrator that runs the complete ETL pipeline:
  1. Load → 2. Transform → 3. Merge → 4. Clean → 5. Detect Anomalies → 6. Export

Usage:
    python run_pipeline.py                   # Uses synthetic data (for dev/demo)
    python run_pipeline.py --download        # Downloads real data from sources
    python run_pipeline.py --force-download  # Force re-download (ignore cache)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

import config
from pipeline.loader import load_jhu_data, load_owid_data, load_mobility_data, generate_synthetic_data
from pipeline.transformer import transform_jhu_wide_to_long, aggregate_by_country_date, compute_daily_features
from pipeline.merger import merge_all_datasets
from pipeline.cleaner import handle_missing_values, optimize_memory
from pipeline.anomaly import detect_anomalies, smooth_spikes
from pipeline.exporter import export_master_dataset, export_country_datasets


def setup_logging() -> None:
    """Configure structured logging to console and file."""
    log_format = "%(asctime)s │ %(levelname)-7s │ %(message)s"
    date_format = "%H:%M:%S"

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format, date_format))

    # File handler
    file_handler = logging.FileHandler(
        config.LOGS_DIR / "pipeline.log", mode="w", encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))

    logging.basicConfig(level=logging.DEBUG, handlers=[console, file_handler])


def print_banner() -> None:
    """Print the pipeline startup banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║     ██████╗ ██████╗ ██████╗ ███████╗ ██████╗██╗   ██╗██████╗ ███████╗║
║    ██╔════╝██╔═══██╗██╔══██╗██╔════╝██╔════╝██║   ██║██╔══██╗██╔════╝║
║    ██║     ██║   ██║██║  ██║█████╗  ██║     ██║   ██║██████╔╝█████╗  ║
║    ██║     ██║   ██║██║  ██║██╔══╝  ██║     ██║   ██║██╔══██╗██╔══╝  ║
║    ╚██████╗╚██████╔╝██████╔╝███████╗╚██████╗╚██████╔╝██║  ██║███████╗║
║     ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝║
║                                                                      ║
║     Epidemic Spread Prediction — Data Pipeline v1.0                  ║
║     High-Performance ETL for Epidemiological Modeling                ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def run_pipeline(use_synthetic: bool = True, force_download: bool = False) -> pd.DataFrame:
    """
    Execute the complete data pipeline.
    
    Args:
        use_synthetic: If True, generate synthetic data (no internet needed)
        force_download: If True, re-download all data from sources
    
    Returns:
        Final merged and cleaned DataFrame
    """
    logger = logging.getLogger(__name__)
    t_start = time.perf_counter()

    # ═══════════════════════════════════════════════════════════════
    # STEP 1: LOAD DATA
    # ═══════════════════════════════════════════════════════════════
    logger.info("┌─────────────────────────────────────┐")
    logger.info("│ STEP 1/6 — LOADING DATA             │")
    logger.info("└─────────────────────────────────────┘")
    t1 = time.perf_counter()

    if use_synthetic:
        data = generate_synthetic_data()
        jhu_dfs = data["jhu"]
        owid_df = data["owid"]
        mobility_df = data["mobility"]
    else:
        jhu_dfs = load_jhu_data(force_download=force_download)
        owid_df = load_owid_data(force_download=force_download)
        mobility_df = load_mobility_data(force_download=force_download)

    logger.info(f"  ⏱ Load: {time.perf_counter() - t1:.2f}s")

    # ═══════════════════════════════════════════════════════════════
    # STEP 2: TRANSFORM JHU DATA
    # ═══════════════════════════════════════════════════════════════
    logger.info("┌─────────────────────────────────────┐")
    logger.info("│ STEP 2/6 — TRANSFORMING JHU DATA    │")
    logger.info("└─────────────────────────────────────┘")
    t2 = time.perf_counter()

    long_frames = transform_jhu_wide_to_long(jhu_dfs)
    jhu_aggregated = aggregate_by_country_date(long_frames)
    jhu_featured = compute_daily_features(jhu_aggregated)

    logger.info(f"  ⏱ Transform: {time.perf_counter() - t2:.2f}s")

    # ═══════════════════════════════════════════════════════════════
    # STEP 3: MERGE DATASETS
    # ═══════════════════════════════════════════════════════════════
    logger.info("┌─────────────────────────────────────┐")
    logger.info("│ STEP 3/6 — MERGING DATASETS         │")
    logger.info("└─────────────────────────────────────┘")
    t3 = time.perf_counter()

    merged = merge_all_datasets(jhu_featured, owid_df, mobility_df)

    logger.info(f"  ⏱ Merge: {time.perf_counter() - t3:.2f}s")

    # ═══════════════════════════════════════════════════════════════
    # STEP 4: HANDLE MISSING VALUES
    # ═══════════════════════════════════════════════════════════════
    logger.info("┌─────────────────────────────────────┐")
    logger.info("│ STEP 4/6 — HANDLING MISSING VALUES  │")
    logger.info("└─────────────────────────────────────┘")
    t4 = time.perf_counter()

    cleaned = handle_missing_values(merged)
    cleaned = optimize_memory(cleaned)

    logger.info(f"  ⏱ Clean: {time.perf_counter() - t4:.2f}s")

    # ═══════════════════════════════════════════════════════════════
    # STEP 5: DETECT ANOMALIES
    # ═══════════════════════════════════════════════════════════════
    logger.info("┌─────────────────────────────────────┐")
    logger.info("│ STEP 5/6 — DETECTING ANOMALIES      │")
    logger.info("└─────────────────────────────────────┘")
    t5 = time.perf_counter()

    anomalied = detect_anomalies(cleaned)
    final = smooth_spikes(anomalied)

    logger.info(f"  ⏱ Anomaly: {time.perf_counter() - t5:.2f}s")

    # ═══════════════════════════════════════════════════════════════
    # STEP 6: EXPORT
    # ═══════════════════════════════════════════════════════════════
    logger.info("┌─────────────────────────────────────┐")
    logger.info("│ STEP 6/6 — EXPORTING RESULTS        │")
    logger.info("└─────────────────────────────────────┘")
    t6 = time.perf_counter()

    export_master_dataset(final)
    export_country_datasets(final)

    logger.info(f"  ⏱ Export: {time.perf_counter() - t6:.2f}s")

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    total_time = time.perf_counter() - t_start
    
    summary = f"""
╔══════════════════════════════════════════════════════════════╗
║                  PIPELINE COMPLETE ✓                        ║
╠══════════════════════════════════════════════════════════════╣
║  Total time:     {total_time:>8.2f}s                                ║
║  Final shape:    {final.shape[0]:>8,} rows × {final.shape[1]:>3} cols                  ║
║  Countries:      {final['country'].nunique():>8,}                                ║
║  Date range:     {str(final['date'].min().date()):>10} → {str(final['date'].max().date()):>10}       ║
║  Memory:         {final.memory_usage(deep=True).sum() / 1e6:>8.1f} MB                             ║
║  Output dir:     {str(config.OUTPUT_DIR):<42}  ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(summary)
    logger.info(f"Pipeline completed in {total_time:.2f}s")

    return final


def main():
    parser = argparse.ArgumentParser(
        description="CodeCure — Epidemic Spread Prediction Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download real data from JHU, OWID, Google Mobility (requires internet)",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download all data (ignore cache)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        default=True,
        help="Use synthetic data for development (default)",
    )

    args = parser.parse_args()

    setup_logging()
    print_banner()

    use_synthetic = not args.download and not args.force_download

    try:
        df = run_pipeline(
            use_synthetic=use_synthetic,
            force_download=args.force_download,
        )
        return 0
    except Exception as e:
        logging.getLogger(__name__).exception(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
