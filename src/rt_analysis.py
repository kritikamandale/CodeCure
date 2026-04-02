"""
CodeCure - Effective Reproduction Number Tracker (Rt)
===================================================
Produces mathematical extraction of the real-time transmission 
momentum of COVID-19 to indicate structural wave-growths vs declines.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# Paths
ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "output" / "master_dataset.parquet"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data(country="India"):
    df = pd.read_parquet(DATA_PATH)
    india = df[df["country"] == country].sort_values("date").reset_index(drop=True)
    return india

def calculate_rt(df):
    """
    Computes effective reproduction momentum based on serial interval assumption.
    Rt ≈ cases_t / cases_(t-7)
    """
    # Use 7-day average natively calculated in pipeline, or back-fill
    if 'cases_7day_avg' in df.columns:
        cases = df['cases_7day_avg']
    else:
        # Fallback derivative computation
        raw_diff = df['confirmed'].diff().fillna(0).clip(lower=0)
        cases = raw_diff.rolling(7).mean()
        
    # Calculate cases(t-7) shifted
    cases_t = cases
    cases_t_minus_7 = cases.shift(7)
    
    # Calculate Rt proxy
    # We add 1.0 to denominator to prevent ZeroDivisionError during structural flatlines
    rt_raw = cases_t / (cases_t_minus_7 + 1.0)
    
    # 7-Day Rolling Average Applied to the Raw Rt ratio
    rt_smooth = rt_raw.rolling(window=7, min_periods=1).mean()
    
    # Uncertainty extraction: 14-day rolling standard deviation
    rt_std = rt_raw.rolling(window=14, min_periods=1).std().fillna(0)
    
    # Approximate 95% Confidence Bounds (Mean +/- 1.96*StDev)
    rt_lower = rt_smooth - (1.96 * rt_std)
    rt_upper = rt_smooth + (1.96 * rt_std)
    
    # Cleanup wild statistical outliers via clipping for Hackable Visual Canvas presentation
    rt_smooth = rt_smooth.clip(lower=0, upper=4.5)
    rt_lower = rt_lower.clip(lower=0, upper=4.5)
    rt_upper = rt_upper.clip(lower=0, upper=4.5)
    
    df['Rt'] = rt_smooth
    df['Rt_Lower'] = rt_lower
    df['Rt_Upper'] = rt_upper
    
    return df.dropna(subset=['Rt'])

def generate_rt_visualization(df):
    """
    Renders Hackathon visualization mapping tracking metrics to visual cues.
    """
    # Styling Setup
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 6))
    
    dates = df['date']
    rt = df['Rt']
    rt_lower = df['Rt_Lower']
    rt_upper = df['Rt_Upper']
    
    # Uncertainty Bounds Fill
    ax.fill_between(dates, rt_lower, rt_upper, color='#f39c12', alpha=0.3, label='95% Confidence Interval')
    
    # Main Rt Trajectory Line
    ax.plot(dates, rt, color='#f1c40f', linewidth=2.5, label='Effective Reproduction Number (Rt)')
    
    # Epidemic Threshold Anchor Line
    ax.axhline(y=1.0, color='#e74c3c', linestyle='--', linewidth=2, label='Epidemic Threshold (Rt = 1)')
    
    # Critical Anotations Setup
    # 1. Delta Variant Max Surge (Roughly around April 2021)
    peak_date = pd.to_datetime('2021-04-01')
    peak_idx = df.index[df['date'] >= peak_date].min()
    if pd.notna(peak_idx) and peak_idx in df.index:
        y_peak = df.loc[peak_idx, 'Rt']
        if not np.isnan(y_peak):
            ax.annotate("Delta Variant Surge\n(Explosive Growth)",
                xy=(df.loc[peak_idx, 'date'], y_peak),
                xytext=(-100, 40),
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='#2c3e50', alpha=0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='white'),
                color='white', fontweight='bold'
            )
            
    # 2. Omicron Variant Surge (Around January 2022)
    omi_date = pd.to_datetime('2022-01-05')
    omi_idx = df.index[df['date'] >= omi_date].min()
    if pd.notna(omi_idx) and omi_idx in df.index:
        y_omi = df.loc[omi_idx, 'Rt']
        if not np.isnan(y_omi):
            ax.annotate("Omicron Variant Tracking",
                xy=(df.loc[omi_idx, 'date'], y_omi),
                xytext=(40, 30),
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='#8e44ad', alpha=0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.2', color='white'),
                color='white', fontweight='bold'
            )
    
    # Current Output State text block
    latest_rt = rt.iloc[-1]
    status = "Active Growth" if latest_rt > 1 else "Decline Status"
    ax.text(0.02, 0.95, f"Current Rt: {latest_rt:.2f} ({status})",
            transform=ax.transAxes, fontsize=14, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#27ae60' if latest_rt < 1 else '#c0392b', alpha=0.7))
    
    # Format Visual Labels
    ax.set_title("Epidemic Early Warning: Effective Reproduction Number (Rt) — India", fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel("Rt Value", fontsize=12)
    ax.set_ylim(0, 4.5)
    
    # X-Axis Date Formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    ax.grid(color='white', alpha=0.1, linestyle='solid')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    plt.tight_layout()
    
    # File Extractor
    plot_path = OUTPUT_DIR / "rt_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Success! Visualization exported natively to: {plot_path}")

def main():
    print("┌──────────────────────────────────────────────┐")
    print("│ Phase: Rt Mathematical Proxy Calculations    │")
    print("└──────────────────────────────────────────────┘")
    
    df = load_data()
    print(f"  > Initialized Dataset array parameters: [{df.shape[0]} rows].")
    
    df_rt = calculate_rt(df)
    print("  > 95% Confidence Interval variances smoothed/calculated.")
    
    generate_rt_visualization(df_rt)
    
if __name__ == "__main__":
    main()
