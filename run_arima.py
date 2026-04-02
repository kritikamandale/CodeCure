"""
CodeCure — ARIMA Statistical Baseline Forecasting Pipeline
============================================================
Complete pipeline for epidemic time-series forecasting using ARIMA:
  Task 1 → Stationarity Check (ADF test, determine d)
  Task 2 → Model Training   (auto_arima, seasonal, m=7)
  Task 3 → Forecasting      (test-set predictions + 7-day ahead)
  Task 4 → Evaluation       (RMSE, MAE, MAPE)
  Task 5 → Persistence      (serialize to arima_model.pkl)

Usage:
    python run_arima.py                          # Default: United States
    python run_arima.py --country "India"         # Specify country
    python run_arima.py --target daily_new_deaths  # Specify target column

Hard Constraints:
  • Strict temporal order — no shuffling at any stage
  • Zero data leakage — test set never influences training
  • Concise and efficient implementation
"""

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ─── Project Setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
import config

MODELS_DIR = config.ROOT_DIR / "models"
PLOTS_DIR = config.OUTPUT_DIR / "plots" / "arima"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ═════════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═════════════════════════════════════════════════════════════════════════════════
def setup_logging() -> None:
    """Configure structured logging to console and file."""
    log_format = "%(asctime)s │ %(levelname)-7s │ %(message)s"
    date_format = "%H:%M:%S"

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format, date_format))

    file_handler = logging.FileHandler(
        config.LOGS_DIR / "arima_pipeline.log", mode="w", encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))

    logging.basicConfig(level=logging.DEBUG, handlers=[console, file_handler])


# ═════════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════════
def load_series(country: str, target_col: str) -> pd.Series:
    """
    Load and prepare a univariate time series from the master dataset.
    
    Returns a pd.Series indexed by date, sorted chronologically, with NaNs
    forward-filled and any remaining NaNs dropped.
    """
    logger = logging.getLogger(__name__)

    parquet_path = config.OUTPUT_DIR / "master_dataset.parquet"
    csv_path = config.OUTPUT_DIR / "master_dataset.csv"

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        logger.info(f"Loaded master dataset from {parquet_path.name}")
    elif csv_path.exists():
        df = pd.read_csv(csv_path, parse_dates=["date"])
        logger.info(f"Loaded master dataset from {csv_path.name}")
    else:
        raise FileNotFoundError(
            "Master dataset not found. Run `python run_pipeline.py` first."
        )

    # Filter to requested country
    available = sorted(df["country"].unique())
    if country not in available:
        raise ValueError(
            f"Country '{country}' not found. Available: {available}"
        )

    df_country = df[df["country"] == country].copy()
    df_country["date"] = pd.to_datetime(df_country["date"])
    df_country.sort_values("date", inplace=True)
    df_country.set_index("date", inplace=True)

    if target_col not in df_country.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. "
            f"Available: {list(df_country.columns)}"
        )

    series = df_country[target_col].astype(float)

    # Handle missing values — forward fill then drop any leading NaNs
    series = series.ffill().dropna()

    # Clamp negatives to zero (daily counts can't be negative)
    series = series.clip(lower=0)

    # ── Zero-variance guard ───────────────────────────────────────────────
    # If the selected column has zero variance (e.g. smoothed to all zeros),
    # auto-fallback to a column with real signal.
    FALLBACK_TARGETS = ["confirmed", "deaths", "cases_7day_avg", "recovered"]
    if series.var() == 0:
        logger.warning(
            f"  ⚠ Target '{target_col}' has zero variance — all values identical."
        )
        original_target = target_col
        for fallback in FALLBACK_TARGETS:
            if fallback in df_country.columns and fallback != target_col:
                candidate = df_country[fallback].astype(float).ffill().dropna().clip(lower=0)
                if candidate.var() > 0:
                    series = candidate
                    target_col = fallback
                    logger.warning(
                        f"  ⟹  Auto-fallback: '{original_target}' → '{target_col}'"
                    )
                    break
        else:
            raise ValueError(
                f"No column with nonzero variance found for {country}. "
                f"Check your data pipeline output."
            )

    series.name = target_col

    logger.info(
        f"Series loaded: {country} | {target_col} | "
        f"{len(series)} observations | {series.index.min().date()} → {series.index.max().date()}"
    )

    return series


# ═════════════════════════════════════════════════════════════════════════════════
# TASK 1: STATIONARITY CHECK
# ═════════════════════════════════════════════════════════════════════════════════
def stationarity_check(series: pd.Series) -> dict:
    """
    Run the Augmented Dickey-Fuller test to determine stationarity and the
    required differencing order d.
    
    Returns dict with ADF results and recommended d.
    """
    logger = logging.getLogger(__name__)

    logger.info("┌─────────────────────────────────────┐")
    logger.info("│ TASK 1 — STATIONARITY CHECK (ADF)   │")
    logger.info("└─────────────────────────────────────┘")

    results = {}

    for d in range(3):  # Test d=0, 1, 2
        if d == 0:
            test_series = series
        else:
            test_series = series.diff(d).dropna()

        adf_stat, p_value, used_lag, n_obs, critical_values, icbest = adfuller(
            test_series, autolag="AIC"
        )

        results[d] = {
            "adf_statistic": adf_stat,
            "p_value": p_value,
            "used_lag": used_lag,
            "n_obs": n_obs,
            "critical_values": critical_values,
            "aic": icbest,
        }

        status = "✓ STATIONARY" if p_value < 0.05 else "✗ NON-STATIONARY"
        logger.info(
            f"  d={d}: ADF={adf_stat:>10.4f} | p={p_value:.6f} | {status}"
        )
        for key, val in critical_values.items():
            logger.info(f"         {key}: {val:.4f}")

    # Determine minimum d for stationarity
    recommended_d = 0
    for d in range(3):
        if results[d]["p_value"] < 0.05:
            recommended_d = d
            break
    else:
        recommended_d = 2  # Fallback if nothing works
        logger.warning("  ⚠ Series not stationary even at d=2, using d=2")

    logger.info(f"  ⟹  Recommended differencing order: d = {recommended_d}")

    return {
        "adf_results": results,
        "recommended_d": recommended_d,
    }


# ═════════════════════════════════════════════════════════════════════════════════
# TASK 2: MODEL TRAINING
# ═════════════════════════════════════════════════════════════════════════════════
def train_arima(
    series: pd.Series, recommended_d: int
) -> tuple:
    """
    Split 80/20 chronologically and train auto_arima with weekly seasonality.
    
    Returns (model, train_series, test_series).
    """
    logger = logging.getLogger(__name__)

    logger.info("┌─────────────────────────────────────┐")
    logger.info("│ TASK 2 — MODEL TRAINING (auto_arima)│")
    logger.info("└─────────────────────────────────────┘")

    # ── Strict chronological 80/20 split ──────────────────────────────────────
    n = len(series)
    split_idx = int(n * 0.80)
    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]

    logger.info(f"  Train: {len(train)} obs ({train.index.min().date()} → {train.index.max().date()})")
    logger.info(f"  Test:  {len(test)} obs ({test.index.min().date()} → {test.index.max().date()})")
    logger.info(f"  Split ratio: {len(train)/n:.1%} / {len(test)/n:.1%}")

    # ── Train auto_arima ──────────────────────────────────────────────────────
    logger.info("  Training auto_arima (seasonal=True, m=7) ...")
    logger.info("  This may take a few minutes depending on series length.")

    t_start = time.perf_counter()

    model = auto_arima(
        train,
        d=recommended_d,
        seasonal=True,
        m=7,                      # Weekly seasonality
        start_p=0, max_p=5,
        start_q=0, max_q=5,
        start_P=0, max_P=2,
        start_Q=0, max_Q=2,
        max_D=1,
        stepwise=True,            # Efficient stepwise search
        suppress_warnings=True,
        error_action="ignore",
        trace=False,
        n_fits=50,                # Cap search iterations
        information_criterion="aic",
    )

    elapsed = time.perf_counter() - t_start

    logger.info(f"  ⏱ Training completed in {elapsed:.1f}s")
    logger.info(f"  Best model: {model.summary().tables[0].as_text().split(chr(10))[1].strip() if hasattr(model.summary(), 'tables') else model.order}")
    logger.info(f"  Order (p,d,q):        {model.order}")
    logger.info(f"  Seasonal (P,D,Q,m):   {model.seasonal_order}")
    logger.info(f"  AIC: {model.aic():.2f}")
    logger.info(f"  BIC: {model.bic():.2f}")

    return model, train, test


# ═════════════════════════════════════════════════════════════════════════════════
# TASK 3: FORECASTING
# ═════════════════════════════════════════════════════════════════════════════════
def forecast(
    model, train: pd.Series, test: pd.Series
) -> tuple:
    """
    Generate predictions on the test set and produce a 7-day ahead forecast
    beyond the test set.
    
    Returns (test_predictions, future_forecast, confidence_intervals).
    """
    logger = logging.getLogger(__name__)

    logger.info("┌─────────────────────────────────────┐")
    logger.info("│ TASK 3 — FORECASTING                │")
    logger.info("└─────────────────────────────────────┘")

    n_test = len(test)

    # ── Test-set predictions ──────────────────────────────────────────────────
    logger.info(f"  Generating {n_test}-step test-set predictions ...")
    test_preds, test_ci = model.predict(
        n_periods=n_test, return_conf_int=True, alpha=0.05
    )

    # Convert to Series with proper date index
    test_preds = pd.Series(test_preds, index=test.index, name="predicted")
    test_ci = pd.DataFrame(
        test_ci, index=test.index, columns=["lower_95", "upper_95"]
    )

    # Clamp predictions to non-negative
    test_preds = test_preds.clip(lower=0)
    test_ci = test_ci.clip(lower=0)

    logger.info(f"  Test predictions: {n_test} values generated")

    # ── 7-day ahead future forecast ───────────────────────────────────────────
    logger.info("  Generating 7-day ahead forecast beyond test set ...")

    # Update the model with test data for the future forecast
    model.update(test)

    future_preds, future_ci = model.predict(
        n_periods=7, return_conf_int=True, alpha=0.05
    )

    # Create future date index
    last_date = test.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=7, freq="D"
    )

    future_preds = pd.Series(future_preds, index=future_dates, name="forecast")
    future_ci = pd.DataFrame(
        future_ci, index=future_dates, columns=["lower_95", "upper_95"]
    )

    # Clamp to non-negative
    future_preds = future_preds.clip(lower=0)
    future_ci = future_ci.clip(lower=0)

    logger.info("  7-day forecast:")
    for date, val in future_preds.items():
        lo = future_ci.loc[date, "lower_95"]
        hi = future_ci.loc[date, "upper_95"]
        logger.info(f"    {date.date()}: {val:>12,.1f}  [{lo:>12,.1f} — {hi:>12,.1f}]")

    return test_preds, test_ci, future_preds, future_ci


# ═════════════════════════════════════════════════════════════════════════════════
# TASK 4: EVALUATION
# ═════════════════════════════════════════════════════════════════════════════════
def evaluate(test: pd.Series, test_preds: pd.Series) -> dict:
    """
    Compute RMSE, MAE, and MAPE on the test set.
    
    Returns dict of metric name → value.
    """
    logger = logging.getLogger(__name__)

    logger.info("┌─────────────────────────────────────┐")
    logger.info("│ TASK 4 — EVALUATION METRICS         │")
    logger.info("└─────────────────────────────────────┘")

    actual = test.values.astype(float)
    predicted = test_preds.values.astype(float)

    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)

    # MAPE — handle zeros in actuals to avoid division by zero
    mask = actual > 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    else:
        mape = float("inf")

    metrics = {"RMSE": rmse, "MAE": mae, "MAPE": mape}

    logger.info(f"  RMSE:  {rmse:>14,.4f}")
    logger.info(f"  MAE:   {mae:>14,.4f}")
    logger.info(f"  MAPE:  {mape:>14.2f}%")
    logger.info(f"  Test samples: {len(actual)}")

    return metrics


# ═════════════════════════════════════════════════════════════════════════════════
# TASK 5: PERSISTENCE
# ═════════════════════════════════════════════════════════════════════════════════
def save_model(model, metrics: dict, country: str, target_col: str) -> Path:
    """
    Serialize the trained ARIMA model to arima_model.pkl.
    
    Also saves a metadata sidecar JSON with model parameters and metrics.
    """
    logger = logging.getLogger(__name__)

    logger.info("┌─────────────────────────────────────┐")
    logger.info("│ TASK 5 — MODEL PERSISTENCE          │")
    logger.info("└─────────────────────────────────────┘")

    model_path = MODELS_DIR / "arima_model.pkl"

    # Save model
    joblib.dump(model, model_path)
    model_size = model_path.stat().st_size / 1024

    logger.info(f"  Model saved: {model_path}")
    logger.info(f"  File size:   {model_size:.1f} KB")

    # Save metadata sidecar
    import json
    metadata = {
        "model_type": "SARIMAX",
        "country": country,
        "target": target_col,
        "order_pdq": list(model.order),
        "seasonal_order_PDQm": list(model.seasonal_order),
        "aic": float(model.aic()),
        "bic": float(model.bic()),
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
        "n_parameters": int(model.df_model()),
        "timestamp": pd.Timestamp.now().isoformat(),
    }

    meta_path = MODELS_DIR / "arima_model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"  Metadata saved: {meta_path}")

    return model_path


# ═════════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════════
def plot_results(
    train: pd.Series,
    test: pd.Series,
    test_preds: pd.Series,
    test_ci: pd.DataFrame,
    future_preds: pd.Series,
    future_ci: pd.DataFrame,
    metrics: dict,
    country: str,
    target_col: str,
) -> None:
    """Generate publication-quality forecast visualization."""
    logger = logging.getLogger(__name__)
    logger.info("  Generating forecast plot ...")

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(
        f"ARIMA Forecast — {country} | {target_col}",
        fontsize=16, fontweight="bold", y=0.98,
    )

    ax1 = axes[0]

    # Plot training data (last 90 days for clarity)
    train_tail = train.iloc[-90:]
    ax1.plot(train_tail.index, train_tail.values, color="#3498db", linewidth=1.2,
             label="Training Data", alpha=0.8)

    # Plot test actuals
    ax1.plot(test.index, test.values, color="#2c3e50", linewidth=1.5,
             label="Actual (Test)", alpha=0.9)

    # Plot test predictions
    ax1.plot(test_preds.index, test_preds.values, color="#e74c3c", linewidth=1.5,
             linestyle="--", label="ARIMA Prediction")

    # Confidence interval for test
    ax1.fill_between(
        test_ci.index, test_ci["lower_95"], test_ci["upper_95"],
        color="#e74c3c", alpha=0.15, label="95% CI (Test)"
    )

    # Plot future forecast
    ax1.plot(future_preds.index, future_preds.values, color="#27ae60", linewidth=2,
             linestyle="-.", marker="o", markersize=5, label="7-Day Forecast")

    # Confidence interval for future
    ax1.fill_between(
        future_ci.index, future_ci["lower_95"], future_ci["upper_95"],
        color="#27ae60", alpha=0.2, label="95% CI (Future)"
    )

    # Train/test split line
    ax1.axvline(x=test.index[0], color="#7f8c8d", linestyle=":", linewidth=1, alpha=0.7)
    ax1.text(test.index[0], ax1.get_ylim()[1] * 0.95, " ← Train | Test →",
             fontsize=9, color="#7f8c8d", ha="center")

    ax1.set_ylabel(target_col.replace("_", " ").title(), fontsize=12)
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    # Metrics annotation
    metrics_text = f"RMSE: {metrics['RMSE']:,.1f}  |  MAE: {metrics['MAE']:,.1f}  |  MAPE: {metrics['MAPE']:.1f}%"
    ax1.text(
        0.99, 0.02, metrics_text, transform=ax1.transAxes,
        fontsize=10, ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ecf0f1", alpha=0.9),
    )

    # ── Residuals subplot ─────────────────────────────────────────────────────
    ax2 = axes[1]
    residuals = test.values - test_preds.values
    ax2.bar(test.index, residuals, color=np.where(residuals >= 0, "#3498db", "#e74c3c"),
            alpha=0.6, width=1)
    ax2.axhline(y=0, color="#2c3e50", linewidth=0.8)
    ax2.set_ylabel("Residuals", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    plt.tight_layout()

    plot_path = PLOTS_DIR / f"arima_forecast_{country.lower().replace(' ', '_')}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"  Plot saved: {plot_path}")


# ═════════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════════
def run_arima_pipeline(country: str = "United States", target_col: str = "confirmed"):
    """Execute the complete ARIMA forecasting pipeline."""
    logger = logging.getLogger(__name__)
    t_start = time.perf_counter()

    banner = f"""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║     ██████╗ ██████╗ ██████╗ ███████╗ ██████╗██╗   ██╗██████╗ ███████╗║
║    ██╔════╝██╔═══██╗██╔══██╗██╔════╝██╔════╝██║   ██║██╔══██╗██╔════╝║
║    ██║     ██║   ██║██║  ██║█████╗  ██║     ██║   ██║██████╔╝█████╗  ║
║    ██║     ██║   ██║██║  ██║██╔══╝  ██║     ██║   ██║██╔══██╗██╔══╝  ║
║    ╚██████╗╚██████╔╝██████╔╝███████╗╚██████╗╚██████╔╝██║  ██║███████╗║
║     ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝║
║                                                                      ║
║     ARIMA Statistical Baseline — Forecasting Pipeline v1.0           ║
║     Country: {country:<53}║
║     Target:  {target_col:<53}║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    print(banner)

    # ── Load Data ─────────────────────────────────────────────────────────────
    series = load_series(country, target_col)
    # Update target_col in case load_series performed a fallback
    target_col = series.name

    # ── Task 1: Stationarity ──────────────────────────────────────────────────
    adf_info = stationarity_check(series)

    # ── Task 2: Train ─────────────────────────────────────────────────────────
    model, train, test = train_arima(series, adf_info["recommended_d"])

    # ── Task 3: Forecast ──────────────────────────────────────────────────────
    test_preds, test_ci, future_preds, future_ci = forecast(model, train, test)

    # ── Task 4: Evaluate ──────────────────────────────────────────────────────
    metrics = evaluate(test, test_preds)

    # ── Task 5: Save ──────────────────────────────────────────────────────────
    model_path = save_model(model, metrics, country, target_col)

    # ── Visualization ─────────────────────────────────────────────────────────
    plot_results(train, test, test_preds, test_ci, future_preds, future_ci,
                 metrics, country, target_col)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_time = time.perf_counter() - t_start
    summary = f"""
╔══════════════════════════════════════════════════════════════════════╗
║              ARIMA PIPELINE COMPLETE ✓                              ║
╠══════════════════════════════════════════════════════════════════════╣
║  Country:         {country:<49}║
║  Target:          {target_col:<49}║
║  Series length:   {len(series):<49}║
║  Model:           SARIMAX{str(model.order)}{str(model.seasonal_order):<34}║
║  AIC:             {model.aic():<49.2f}║
║  ────────────────────────────────────────────────────────────────── ║
║  RMSE:            {metrics['RMSE']:<49,.4f}║
║  MAE:             {metrics['MAE']:<49,.4f}║
║  MAPE:            {metrics['MAPE']:<48.2f}%║
║  ────────────────────────────────────────────────────────────────── ║
║  Model saved:     {str(model_path.name):<49}║
║  Total time:      {total_time:<48.1f}s║
╚══════════════════════════════════════════════════════════════════════╝
"""
    print(summary)

    return {
        "model": model,
        "metrics": metrics,
        "test_preds": test_preds,
        "future_preds": future_preds,
        "model_path": model_path,
    }


# ═════════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="CodeCure — ARIMA Statistical Baseline Forecasting Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--country", type=str, default="United States",
        help="Country to forecast (default: United States)",
    )
    parser.add_argument(
        "--target", type=str, default="confirmed",
        help="Target column to forecast (default: confirmed)",
    )

    args = parser.parse_args()

    setup_logging()

    try:
        result = run_arima_pipeline(country=args.country, target_col=args.target)
        return 0
    except Exception as e:
        logging.getLogger(__name__).exception(f"ARIMA pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
