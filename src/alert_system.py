"""
CodeCure - Epidemic Early Warning & Alert System
================================================
Generates action-based epidemiological alert thresholds evaluating real-time 
growth parameters natively tying reproduction momentum (Rt) to doubling constraints.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Path Logic
ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "output" / "master_dataset.parquet"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def generate_alert_matrix():
    """
    Evaluates tracking matrices discovering highest warning boundaries 
    dynamically mapped across recent data blocks.
    """
    df = pd.read_parquet(DATA_PATH)
    
    # 1. We require the most current/latest reporting dates natively extracted
    idx = df.groupby('country')['date'].idxmax()
    latest = df.loc[idx].copy()
    
    # 2. Extract and establish constraint logic mappings natively
    # Some variables like `Rt` (reproduction_rate natively in our dataset) carry inherently
    # If not present or we need native proxy limits, we calculate them directly!
    
    alert_array = []
    
    for country in latest['country'].unique():
        country_df = df[df['country'] == country].sort_values('date').tail(8)
        
        # Calculate trailing cases array explicitly
        cases_today = country_df['cases_7day_avg'].values[-1]
        cases_7d_ago = country_df['cases_7day_avg'].values[0]
        
        # Growth Rate Logic mapping bounding limit protections
        growth_rate = cases_today / (cases_7d_ago + 1e-9)
        
        # Rt Calculation (If native representation missing, proxy map):
        # Taking actual tracking constants
        rt_val = latest.loc[latest['country'] == country, 'reproduction_rate'].values[0]
        if np.isnan(rt_val):
            rt_val = growth_rate # Simplistic mathematical fallback map
            
        # Doubling Time Calculation (Ln(2) / Ln(growth_rate_daily_equiv))
        # Growth_rate calculated above is weekly (7 days). Daily growth = exp(ln(W)/7)
        if growth_rate > 1.0:
            daily_growth = np.exp(np.log(growth_rate) / 7.0)
            doubling_time = np.log(2) / np.log(daily_growth)
        else:
            doubling_time = 999.0 # Declining or stable boundaries
            
        # Hard limits
        doubling_time = np.clip(doubling_time, 1, 999)
        
        alert_array.append({
            'country': country,
            'Rt': round(rt_val, 2),
            'growth_rate': round(growth_rate, 2),
            'doubling_time': round(doubling_time, 1)
        })
        
    alerts_df = pd.DataFrame(alert_array)
    
    # 3. Alert Conditions & Recommendations Structural Logic
    def assign_alert(row):
        rt = row['Rt']
        gr = row['growth_rate']
        dt = row['doubling_time']
        
        if rt > 1.2 and gr > 1.15 and dt < 14:
            return "🔴 Critical", "Immediate intervention required"
        elif rt > 1.0 and gr > 1.05:
            return "🟠 Warning", "Increase monitoring"
        elif rt > 1.0:
            return "🟡 Watch", "Observe trends"
        else:
            return "🟢 Safe", "Stable"
            
    alerts_df[['alert_level', 'recommendation']] = alerts_df.apply(
        lambda r: pd.Series(assign_alert(r)), axis=1
    )
    
    # Final Presentation Structuring
    
    # Custom mapping dictionary strictly scaling categorical limits appropriately
    level_rank = {"🔴 Critical": 1, "🟠 Warning": 2, "🟡 Watch": 3, "🟢 Safe": 4}
    alerts_df['rank'] = alerts_df['alert_level'].map(level_rank)
    
    # Organize structurally outputting highest critical threats to the top natively
    alerts_df = alerts_df.sort_values(['rank', 'Rt'], ascending=[True, False]).drop(columns=['rank']).reset_index(drop=True)
    
    # Terminal Display Array visually mapped to console
    print("\n┌────────────────────────────────────────────────────────┐")
    print("│         CODECURE EARLY WARNING ALERT SYSTEM            │")
    print("└────────────────────────────────────────────────────────┘\n")
    print(alerts_df.to_string(index=False))
    
    # Save constraints natively into structural outputs
    output_target = OUTPUT_DIR / "alerts.csv"
    alerts_df.to_csv(output_target, index=False)
    print(f"\n✅ Trigger Alert system generated. Payload stored locally: {output_target}")
    
def main():
    generate_alert_matrix()
    
if __name__ == "__main__":
    main()
