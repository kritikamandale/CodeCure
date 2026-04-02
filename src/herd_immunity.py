"""
CodeCure - Herd Immunity Mathematical Benchmark Tracker
=====================================================
Calculates final aggregate vaccination penetration metrics
across our modeled nations, assessing them against classical
epidemiological theoretical containment barriers defined
by Basic Reproduction variables (R0).
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths Setup
ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "output" / "master_dataset.parquet"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_vax_data():
    """Extracts the maximum historical vaccination penetration achieved."""
    df = pd.read_parquet(DATA_PATH)
    
    # We aggregate the absolute max of people_fully_vaccinated and population per country
    agg_df = df.groupby('country').agg({
        'people_fully_vaccinated': 'max',
        'population': 'max'
    }).reset_index()
    
    # Calculate native Percentage Penetration
    agg_df['vax_percent'] = (agg_df['people_fully_vaccinated'] / agg_df['population']) * 100
    
    # Clean NaN drops and sort descending structurally
    agg_df = agg_df.dropna(subset=['vax_percent'])
    agg_df = agg_df.sort_values(by='vax_percent', ascending=True).reset_index(drop=True)
    
    return agg_df

def calc_herd_immunity(R0):
    """Calculates Herd Threshold Percentage: 1 - 1/R0"""
    return (1 - (1/R0)) * 100

def generate_visualization(df):
    """
    Renders the Presentation Horizontal Bar Graph mapped against structural thresholds.
    """
    # Threshold Setup Array
    t_original = calc_herd_immunity(2.5) # Original Strain (~60%)
    t_delta = calc_herd_immunity(6.0)    # Delta Variant (~83%)
    t_omicron = calc_herd_immunity(12.0) # Omicron Base (~91%)
    
    # Structuring Canvas
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    countries = df['country']
    vax_rates = df['vax_percent']
    
    # We use Delta variant (83%) as the anchor for the Red vs Green hackathon logic requirement
    anchor = t_delta
    colors = ['#2ecc71' if v >= anchor else '#e74c3c' for v in vax_rates]
    
    # Generate Bars
    bars = ax.barh(countries, vax_rates, color=colors, height=0.6, alpha=0.85)
    
    # Generate Threshold Anchors
    ax.axvline(x=t_original, color='#f1c40f', linestyle=':', linewidth=2, zorder=0,
               label=f'Original Strain R0=2.5 ({t_original:.0f}%)')
    ax.axvline(x=t_delta, color='#e67e22', linestyle='--', linewidth=2, zorder=0,
               label=f'Delta Variant R0=6.0 ({t_delta:.0f}%)')
    ax.axvline(x=t_omicron, color='#9b59b6', linestyle='-.', linewidth=2, zorder=0,
               label=f'Omicron Variant R0=12.0 ({t_omicron:.0f}%)')
               
    # Annotate Values dynamically on top of bars
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + 1.0 if width < 90 else width - 6.0
        label_color = 'white'
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}%', va='center', ha='left' if width < 90 else 'right',
                color=label_color, fontweight='bold', fontsize=11)
                
    # Formalizing the Layout Canvas Aesthetics
    ax.set_title("Vulnerability Analysis: Vaccination Rates vs Herd Immunity Thresholds", 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xlabel("Fully Vaccinated Population (%)", fontsize=12, fontweight='bold', labelpad=15)
    
    # Toggling limits structurally to 100 max
    ax.set_xlim(0, 100)
    
    # Setting Spines to be minimalist
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    ax.grid(axis='x', color='white', alpha=0.1)
    
    # Pushing the Legend to bottom right outside plot array
    ax.legend(loc='lower right', framealpha=0.9, fontsize=10, bbox_to_anchor=(0.98, 0.05))
    
    plt.tight_layout()
    
    plot_path = OUTPUT_DIR / "herd_immunity.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Executed. Map visually encoded and exported to: {plot_path}")

def main():
    print("┌────────────────────────────────────────────────────────┐")
    print("│ Phase: Mathematical Herd Immunity Vulnerability Bounds │")
    print("└────────────────────────────────────────────────────────┘")
    
    df = load_vax_data()
    print("  > Base aggregate OWID parameters parsed.")
    
    generate_visualization(df)

if __name__ == "__main__":
    main()
