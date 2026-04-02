"""
CodeCure - Epidemiological Phase Portrait Tracker
===============================================
Generates advanced mathematical Phase Portraits mapping $Log(NewCases)$
vs $Log(CumulativeCases)$ to determine if geographical viral spread
is trapped inside Exponential or Controlled boundaries!
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

# Paths
ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "output" / "master_dataset.parquet"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Custom Base Target Countries
TARGETS = ['United States', 'India', 'Brazil', 'Germany', 'South Africa']
COLORS = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f', '#9b59b6']

def process_country_trajectory(df, country_name):
    """
    Extracts the native logarithmic parameters corresponding specifically
    to the defined trajectory array calculations.
    """
    country_data = df[df["country"] == country_name].sort_values('date').copy()
    
    # 1. Total Cases (X-Axis)
    cumulative = country_data['confirmed'].values
    
    # 2. Weekly New Cases (Y-Axis) via rolling 7 day sum of daily differentiated values
    daily_cases = np.diff(cumulative, prepend=cumulative[0])
    daily_cases = np.clip(daily_cases, 0, None)
    
    weekly_cases = pd.Series(daily_cases).rolling(window=7, min_periods=1).sum().values
    
    # 3. Logarithmic Scaling avoiding mathematically invalid zero crashes
    x_log = np.log1p(cumulative)
    y_log = np.log1p(weekly_cases)
    
    # 4. Critical Visual Parameter: Filter Smoothing
    # Applying strong rolling mean filter removing daily scribble overlaps
    window_sz = 14
    x_smooth = pd.Series(x_log).rolling(window=window_sz, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
    y_smooth = pd.Series(y_log).rolling(window=window_sz, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
    
    return x_smooth, y_smooth, country_data['date'].values

def render_phase_portrait(df):
    """
    Renders Phase Canvas applying mathematical tracking arrows natively
    along the plotted output splines.
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Bounding Boxes
    ax.set_title("Epidemic Phase Portrait: Trajectory of Outbreak Control", fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel("Log(Total Cumulative Cases)", fontsize=13, fontweight='bold', labelpad=15)
    ax.set_ylabel("Log(New Weekly Cases)", fontsize=13, fontweight='bold', labelpad=15)
    
    for country, hex_color in zip(TARGETS, COLORS):
        X, Y, dates = process_country_trajectory(df, country)
        
        # We discard extreme origin bounds before tracking mathematically normalizes
        valid_idx = (X > 5) & (Y > 5)
        X, Y = X[valid_idx], Y[valid_idx]
        
        if len(X) < 10:
            continue
            
        # Draw Trajectory Plot line structurally
        ax.plot(X, Y, color=hex_color, alpha=0.8, linewidth=2.5, label=country)
        
        # Inject Directional Vector Arrows showing timeline flow across trajectory segments
        num_arrows = 4
        step = len(X) // num_arrows
        for i in range(step, len(X) - step, step):
            arrow = FancyArrowPatch((X[i], Y[i]), (X[i+1], Y[i+1]), 
                                    mutation_scale=20, color=hex_color, alpha=0.9,
                                    arrowstyle='-|>', zorder=5)
            ax.add_patch(arrow)
            
        # Mark Ending Position
        ax.scatter([X[-1]], [Y[-1]], color=hex_color, s=150, edgecolors='white', zorder=6)
        
    # Append Theoretical Background Reference Bounding Lines
    # 1. Uncontrolled Exponential Growth Bound (Diagonal Log Linear Track)
    # y = mx + b. Using rough anchor bounding
    x_vals = np.linspace(6, 20, 100)
    y_vals_exponential = x_vals - 1.5  # Constant Offset mimicking massive uniform explosive growth matrix
    
    ax.plot(x_vals, y_vals_exponential, color='white', linestyle='--', linewidth=2, alpha=0.4)
    # Annotation aligned to mathematical diagonal slope
    ax.annotate("Uncontrolled Exponential Spread (Linear)", 
               xy=(15, 13.5), xytext=(15.5, 14.5), color='white', alpha=0.7, 
               fontweight='bold', rotation=38)
    
    # 2. Complete Containment Region marker
    ax.fill_between(x_vals, 0, 5, color='#27ae60', alpha=0.1)
    ax.annotate("Controlled Epidemic Environment", xy=(7, 2), color='#2ecc71', alpha=0.8, fontsize=12, fontweight='bold')
    
    # Styling Map Structure Limits
    ax.set_xlim(5.5, 20)
    ax.set_ylim(0, 16)
    ax.grid(color='white', alpha=0.1)
    ax.legend(loc='lower right', framealpha=0.9, fontsize=12)
    
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "phase_portrait.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Executed. Publication-ready Phase Portrait exported to: {plot_path}")

def main():
    print("┌──────────────────────────────────────────────┐")
    print("│ Phase: Mathematical Epidemiological Portrait │")
    print("└──────────────────────────────────────────────┘")
    
    df = pd.read_parquet(DATA_PATH)
    render_phase_portrait(df)

if __name__ == "__main__":
    main()
