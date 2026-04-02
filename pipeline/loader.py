"""
Data Loader Module
==================
Downloads and caches raw datasets from JHU, OWID, and Google Mobility.
Uses streaming downloads with progress bars and local caching.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


def _download_csv(url: str, dest: Path, chunk_size: int = 8192) -> Path:
    """Download a CSV with progress bar and cache locally."""
    if dest.exists():
        logger.info(f"  ✓ Cached: {dest.name}")
        return dest

    logger.info(f"  ↓ Downloading: {dest.name}")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            pbar.update(len(chunk))

    logger.info(f"  ✓ Saved: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def load_jhu_data(force_download: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Load Johns Hopkins COVID-19 time series (confirmed, deaths, recovered).
    Returns dict of DataFrames keyed by metric name.
    """
    logger.info("═══ Loading JHU Time Series ═══")
    result = {}

    for metric, url in config.JHU_URLS.items():
        dest = config.RAW_DATA_DIR / f"jhu_{metric}.csv"
        if force_download and dest.exists():
            dest.unlink()

        _download_csv(url, dest)
        df = pd.read_csv(dest, low_memory=False)
        logger.info(f"  {metric}: {df.shape[0]:,} rows × {df.shape[1]:,} cols")
        result[metric] = df

    return result


def load_owid_data(force_download: bool = False) -> pd.DataFrame:
    """
    Load Our World in Data COVID dataset.
    Selects only the feature columns defined in config.
    """
    logger.info("═══ Loading OWID Dataset ═══")
    dest = config.RAW_DATA_DIR / "owid_covid.csv"
    if force_download and dest.exists():
        dest.unlink()

    _download_csv(config.OWID_URL, dest)

    # Read only needed columns to save memory
    all_cols = pd.read_csv(dest, nrows=0).columns.tolist()
    usecols = [c for c in config.OWID_FEATURES if c in all_cols]
    missing = set(config.OWID_FEATURES) - set(usecols)
    if missing:
        logger.warning(f"  ⚠ OWID columns not found: {missing}")

    df = pd.read_csv(dest, usecols=usecols, low_memory=False)
    logger.info(f"  OWID: {df.shape[0]:,} rows × {df.shape[1]:,} cols")
    return df


def load_mobility_data(force_download: bool = False) -> pd.DataFrame:
    """
    Load Google Community Mobility Reports.
    Selects only the feature columns defined in config.
    """
    logger.info("═══ Loading Google Mobility ═══")
    dest = config.RAW_DATA_DIR / "google_mobility.csv"
    if force_download and dest.exists():
        dest.unlink()

    _download_csv(config.GOOGLE_MOBILITY_URL, dest)

    # Read only needed columns
    all_cols = pd.read_csv(dest, nrows=0).columns.tolist()
    usecols = [c for c in config.MOBILITY_FEATURES if c in all_cols]
    missing = set(config.MOBILITY_FEATURES) - set(usecols)
    if missing:
        logger.warning(f"  ⚠ Mobility columns not found: {missing}")

    df = pd.read_csv(dest, usecols=usecols, low_memory=False)
    logger.info(f"  Mobility: {df.shape[0]:,} rows × {df.shape[1]:,} cols")
    return df


def generate_synthetic_data() -> Dict:
    """
    Generate realistic synthetic data for development/testing
    when real data sources are unavailable.
    """
    import numpy as np

    logger.info("═══ Generating Synthetic Data ═══")
    np.random.seed(42)

    countries = config.TARGET_COUNTRIES
    dates = pd.date_range("2020-01-22", "2023-03-09", freq="D")
    n_days = len(dates)

    # ── Synthetic JHU data (already in long format for simplicity) ──
    jhu_records = []
    for country in countries:
        # Simulate cumulative cases with logistic growth + waves
        t = np.arange(n_days, dtype=np.float64)
        # Base logistic curve
        base = 1e7 / (1 + np.exp(-0.03 * (t - 200)))
        # Add waves
        wave1 = 2e6 * np.exp(-((t - 350) ** 2) / (2 * 50**2))
        wave2 = 3e6 * np.exp(-((t - 600) ** 2) / (2 * 60**2))
        wave3 = 4e6 * np.exp(-((t - 850) ** 2) / (2 * 40**2))
        confirmed = np.maximum(0, base + wave1 + wave2 + wave3)
        confirmed = np.round(confirmed).astype(np.int64)
        # Ensure monotonic
        confirmed = np.maximum.accumulate(confirmed)

        deaths = np.round(confirmed * np.random.uniform(0.01, 0.03)).astype(np.int64)
        deaths = np.maximum.accumulate(deaths)
        recovered = np.round(confirmed * np.random.uniform(0.85, 0.95)).astype(np.int64)
        recovered = np.minimum(recovered, confirmed - deaths)
        recovered = np.maximum.accumulate(np.maximum(0, recovered))

        for i, date in enumerate(dates):
            jhu_records.append({
                "country": country,
                "date": date,
                "confirmed": confirmed[i],
                "deaths": deaths[i],
                "recovered": recovered[i],
            })

    jhu_df = pd.DataFrame(jhu_records)

    # Convert to JHU wide format for each metric
    jhu_dfs = {}
    for metric in ["confirmed", "deaths", "recovered"]:
        pivot = jhu_df.pivot(index="country", columns="date", values=metric)
        pivot.columns = [f"{c.month}/{c.day}/{str(c.year)[-2:]}" if hasattr(c, "strftime") else str(c) for c in pivot.columns]
        pivot = pivot.reset_index()
        pivot.rename(columns={"country": "Country/Region"}, inplace=True)
        pivot.insert(0, "Province/State", "")
        pivot.insert(2, "Lat", 0.0)
        pivot.insert(3, "Long", 0.0)
        jhu_dfs[metric] = pivot

    # ── Synthetic OWID data ──
    owid_records = []
    for country in countries:
        vacc_total = 0
        tests_total = 0
        for i, date in enumerate(dates):
            day_idx = i
            # Vaccinations ramp up after day 300
            if day_idx > 300:
                new_vacc = int(np.random.poisson(50000 * min(1, (day_idx - 300) / 200)))
                vacc_total += new_vacc
            else:
                new_vacc = 0

            new_tests = int(np.random.poisson(max(5000, 50000 * min(1, day_idx / 200))))
            tests_total += new_tests

            owid_records.append({
                "location": country,
                "date": date.strftime("%Y-%m-%d"),
                "total_vaccinations": vacc_total if day_idx > 300 else None,
                "people_vaccinated": int(vacc_total * 0.6) if day_idx > 300 else None,
                "people_fully_vaccinated": int(vacc_total * 0.4) if day_idx > 320 else None,
                "total_boosters": int(vacc_total * 0.15) if day_idx > 500 else None,
                "new_vaccinations_smoothed": new_vacc if day_idx > 300 else None,
                "total_tests": tests_total,
                "new_tests": new_tests,
                "new_tests_smoothed": new_tests,
                "total_tests_per_thousand": round(tests_total / 3300, 2),
                "new_tests_per_thousand": round(new_tests / 330000, 4),
                "positive_rate": round(np.random.beta(2, 20), 4),
                "tests_per_case": round(1 / max(0.01, np.random.beta(2, 20)), 1),
                "stringency_index": round(np.clip(
                    40 + 30 * np.sin(day_idx / 100) + np.random.normal(0, 5), 0, 100
                ), 1),
                "population_density": {"United States": 36, "India": 464, "Brazil": 25, "Germany": 240, "South Africa": 49}.get(country, 50),
                "median_age": {"United States": 38.3, "India": 28.4, "Brazil": 33.5, "Germany": 45.7, "South Africa": 27.6}.get(country, 30),
                "aged_65_older": {"United States": 16.6, "India": 6.4, "Brazil": 9.1, "Germany": 21.5, "South Africa": 5.3}.get(country, 10),
                "gdp_per_capita": {"United States": 63544, "India": 6427, "Brazil": 14103, "Germany": 52559, "South Africa": 12295}.get(country, 10000),
                "hospital_beds_per_thousand": {"United States": 2.77, "India": 0.53, "Brazil": 2.11, "Germany": 8.0, "South Africa": 2.32}.get(country, 2),
                "life_expectancy": {"United States": 78.9, "India": 69.7, "Brazil": 75.9, "Germany": 81.3, "South Africa": 64.1}.get(country, 70),
                "human_development_index": {"United States": 0.926, "India": 0.645, "Brazil": 0.765, "Germany": 0.947, "South Africa": 0.709}.get(country, 0.7),
                "population": {"United States": 331002651, "India": 1380004385, "Brazil": 212559417, "Germany": 83783942, "South Africa": 59308690}.get(country, 1e8),
                "icu_patients": int(np.random.poisson(max(1, 500 * np.sin(day_idx / 150) + 300))) if day_idx > 50 else None,
                "hosp_patients": int(np.random.poisson(max(1, 2000 * np.sin(day_idx / 150) + 1500))) if day_idx > 50 else None,
                "reproduction_rate": round(np.clip(1.0 + 0.5 * np.sin(day_idx / 80) + np.random.normal(0, 0.1), 0.5, 3.0), 2),
            })

    owid_df = pd.DataFrame(owid_records)

    # ── Synthetic Mobility data ──
    mobility_records = []
    for country in countries:
        for i, date in enumerate(dates):
            day_idx = i
            # Mobility drops sharply at lockdowns, recovers gradually
            base_drop = -40 * np.exp(-((day_idx - 60) ** 2) / (2 * 30**2))
            seasonal = 5 * np.sin(2 * np.pi * day_idx / 365)
            recovery = min(0, -20 + day_idx * 0.03)

            mobility_records.append({
                "country_region": country,
                "date": date.strftime("%Y-%m-%d"),
                "retail_and_recreation_percent_change_from_baseline": int(base_drop + seasonal + recovery + np.random.normal(0, 3)),
                "grocery_and_pharmacy_percent_change_from_baseline": int(base_drop * 0.5 + seasonal + recovery + np.random.normal(0, 3)),
                "parks_percent_change_from_baseline": int(base_drop * 0.3 + seasonal * 2 + recovery + np.random.normal(0, 5)),
                "transit_stations_percent_change_from_baseline": int(base_drop * 1.2 + seasonal + recovery + np.random.normal(0, 3)),
                "workplaces_percent_change_from_baseline": int(base_drop * 1.1 + seasonal * 0.5 + recovery + np.random.normal(0, 3)),
                "residential_percent_change_from_baseline": int(-base_drop * 0.4 - seasonal * 0.3 - recovery * 0.5 + np.random.normal(0, 2)),
            })

    mobility_df = pd.DataFrame(mobility_records)

    logger.info(f"  Synthetic JHU: {len(jhu_records):,} records across {len(countries)} countries")
    logger.info(f"  Synthetic OWID: {owid_df.shape[0]:,} rows × {owid_df.shape[1]:,} cols")
    logger.info(f"  Synthetic Mobility: {mobility_df.shape[0]:,} rows × {mobility_df.shape[1]:,} cols")

    return {
        "jhu": jhu_dfs,
        "owid": owid_df,
        "mobility": mobility_df,
    }
