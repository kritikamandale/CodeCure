"""
Output Verification Script
==========================
Validates that the pipeline outputs are correct and ML-ready.
"""

import os
import sys
import io
import time
from pathlib import Path

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import config


def verify():
    passed = 0
    failed = 0

    print("=" * 65)
    print("  VERIFICATION: Loading & Validating Pipeline Outputs")
    print("=" * 65)

    # ── 1. Load master parquet ──
    pq_path = config.OUTPUT_DIR / "master_dataset.parquet"
    csv_path = config.OUTPUT_DIR / "master_dataset.csv"

    if not pq_path.exists():
        print("\n[FAIL] master_dataset.parquet not found! Run the pipeline first.")
        return 1

    df = pd.read_parquet(pq_path)
    print(f"\n[1] Master dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"    dtypes: {dict(df.dtypes.value_counts())}")
    print(f"    Memory: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    passed += 1

    # ── 2. Null check ──
    nulls = df.isnull().sum().sum()
    status = "PASS" if nulls == 0 else "FAIL"
    print(f"\n[2] Null check: {nulls} nulls ({status})")
    if nulls == 0:
        passed += 1
    else:
        failed += 1

    # ── 3. Temporal order ──
    sorted_ok = df.groupby("country")["date"].apply(
        lambda x: x.is_monotonic_increasing
    ).all()
    status = "PASS" if sorted_ok else "FAIL"
    print(f"[3] Temporal order: {status}")
    if sorted_ok:
        passed += 1
    else:
        failed += 1

    # ── 4. Duplicate check ──
    dupes = df.duplicated(subset=["country", "date"]).sum()
    status = "PASS" if dupes == 0 else "FAIL"
    print(f"[4] Duplicate check: {dupes} duplicates ({status})")
    if dupes == 0:
        passed += 1
    else:
        failed += 1

    # ── 5. Target variable ──
    print(f"\n[5] Target variable (cases_7day_avg):")
    print(f"    min={df['cases_7day_avg'].min():.1f}")
    print(f"    max={df['cases_7day_avg'].max():.1f}")
    print(f"    mean={df['cases_7day_avg'].mean():.1f}")
    no_neg = (df["cases_7day_avg"] >= 0).all()
    status = "PASS" if no_neg else "FAIL"
    print(f"    Non-negative: {status}")
    if no_neg:
        passed += 1
    else:
        failed += 1

    # ── 6. Required features ──
    required = [
        "cases_7day_avg",
        "total_vaccinations",
        "total_tests_per_thousand",
        "stringency_index",
        "mobility_retail",
        "mobility_transit",
        "mobility_workplaces",
        "population_density",
        "median_age",
    ]
    missing = [f for f in required if f not in df.columns]
    status = "PASS" if not missing else "FAIL"
    print(f"\n[6] Required features: {'all present' if not missing else f'MISSING: {missing}'} ({status})")
    if not missing:
        passed += 1
    else:
        failed += 1

    # ── 7. Country files ──
    countries = ["united_states", "india", "brazil", "germany", "south_africa"]
    all_exist = all(
        (config.COUNTRY_OUTPUT_DIR / f"{c}.parquet").exists() for c in countries
    )
    status = "PASS" if all_exist else "FAIL"
    print(f"\n[7] Country files: {'all 5 present' if all_exist else 'MISSING'} ({status})")
    if all_exist:
        passed += 1
        for c in countries:
            cdf = pd.read_parquet(config.COUNTRY_OUTPUT_DIR / f"{c}.parquet")
            print(
                f"    {c}: {cdf.shape[0]:,} rows, "
                f"{cdf['date'].min().date()} -> {cdf['date'].max().date()}"
            )
    else:
        failed += 1

    # ── 8. Parquet vs CSV speed ──
    if csv_path.exists():
        t0 = time.perf_counter()
        _ = pd.read_csv(csv_path)
        csv_time = time.perf_counter() - t0
        t0 = time.perf_counter()
        _ = pd.read_parquet(pq_path)
        pq_time = time.perf_counter() - t0
        speedup = csv_time / pq_time if pq_time > 0 else float("inf")
        print(
            f"\n[8] Read speed: CSV={csv_time:.3f}s, Parquet={pq_time:.3f}s "
            f"({speedup:.1f}x faster)"
        )
        csv_size = csv_path.stat().st_size / 1e6
        pq_size = pq_path.stat().st_size / 1e6
        print(f"    Size: CSV={csv_size:.1f}MB, Parquet={pq_size:.1f}MB ({csv_size / pq_size:.1f}x smaller)")
        passed += 1

    # ── 9. Anomaly columns ──
    anomaly_cols = [c for c in df.columns if c.startswith("anomaly_") or c == "is_anomaly"]
    smoothed_cols = [c for c in df.columns if c.endswith("_smoothed")]
    print(f"\n[9] Anomaly flags: {anomaly_cols}")
    print(f"    Smoothed variants: {smoothed_cols}")
    if anomaly_cols and smoothed_cols:
        passed += 1
    else:
        failed += 1

    # ── 10. Column listing ──
    print(f"\n[10] All {len(df.columns)} columns:")
    for i, c in enumerate(df.columns):
        print(f"     {i + 1:2d}. {c} ({df[c].dtype})")

    # ── Summary ──
    print("\n" + "=" * 65)
    if failed == 0:
        print(f"  ALL {passed} CHECKS PASSED -- Dataset is ML-ready [OK]")
    else:
        print(f"  {passed} PASSED, {failed} FAILED — Review issues above")
    print("=" * 65)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(verify())
