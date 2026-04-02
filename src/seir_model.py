"""
CodeCure - SEIR Compartmental Epidemic Model
============================================
Implements a mathematical ODE SEIR model optimized via Scipy 
to fit the real-world transmission rates of India's Delta Wave.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
from pathlib import Path

# Paths
ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "output" / "master_dataset.parquet"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── SEIR ODE System ─────────────────────────────────────────────────────────────
def seir_deriv(y, t, N, beta, gamma, sigma):
    """
    Differential equations for the SEIR model.
    S: Susceptible
    E: Exposed (incubating)
    I: Infectious
    R: Recovered/Removed
    """
    S, E, I, R, C = y
    
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    
    # Track Cumulative Cases strictly for new incidences
    dCdt = sigma * E
    
    return dSdt, dEdt, dIdt, dRdt, dCdt

# ─── Data Extraction & Preparation ───────────────────────────────────────────────
def load_wave_data(country="India", start_date="2021-02-15", end_date="2021-06-15"):
    """Isolates the data specifically corresponding to the massive Delta wave."""
    df = pd.read_parquet(DATA_PATH)
    wave_df = df[
        (df["country"] == country) & 
        (df["date"] >= start_date) & 
        (df["date"] <= end_date)
    ].sort_values("date")
    
    # We want daily incidences to fit against
    if 'cases_7day_avg' in wave_df.columns:
        daily_cases = wave_df['cases_7day_avg'].values
    else:
        # Calculate daily from cumulative confirmed if needed
        confirmed = wave_df['confirmed'].values
        daily_cases = np.diff(confirmed, prepend=confirmed[0])
        daily_cases = np.clip(daily_cases, 0, None) # Clamp negative corrects
        
    return wave_df['date'].values, daily_cases

# ─── Parameter Optimization ──────────────────────────────────────────────────────
def fit_seir_model(daily_cases, N, I0, E0, sigma):
    """
    Minimizes the Mean Squared Error to discover mathematical Beta and Gamma constants.
    """
    t = np.arange(len(daily_cases))
    
    # Initial conditions: Susceptibles = Pop - Intials
    S0 = N - I0 - E0
    R0 = 0
    C0 = I0
    y0 = [S0, E0, I0, R0, C0]
    
    def loss(params):
        beta, gamma = params
        # Constraints penalty (parameters must be strictly positive)
        if beta <= 0 or gamma <= 0:
            return np.inf
            
        # Simulate ODE
        solution = odeint(seir_deriv, y0, t, args=(N, beta, gamma, sigma))
        
        # Simulated daily cases = dC/dt = sigma * E
        E_sim = solution[:, 1]
        simulated_daily = sigma * E_sim
        
        # Calculate MSE Error specifically against actual smoothed daily cases
        mse = np.mean((simulated_daily - daily_cases)**2)
        return mse
        
    # Initial Guesses (beta, gamma)
    initial_guess = [0.4, 0.1]
    
    # Boundaries roughly [beta > 0, gamma between 1/14 and 1/3]
    bounds = [(0.01, 2.0), (0.05, 0.4)]
    
    print("Optimization engine starting (L-BFGS-B)...")
    res = minimize(loss, initial_guess, method="L-BFGS-B", bounds=bounds)
    print(f"Convergence Output: {res.message}")
    
    optimal_beta, optimal_gamma = res.x
    return optimal_beta, optimal_gamma, t, y0

# ─── Main Execution & Plotting ───────────────────────────────────────────────────
def main():
    print("┌──────────────────────────────────────────────┐")
    print("│ Phase: Mathematical SEIR Parameter Fitting   │")
    print("└──────────────────────────────────────────────┘")
    
    # Constants
    N_india = 1_393_000_000   # India approximate population
    sigma = 1.0 / 5.2         # Incubation period latency (~5.2 days literature avg)
    
    # 1. Load Data
    dates, actual_daily = load_wave_data()
    print(f"Targeting Delta Wave Window: {len(dates)} days.")
    
    # 2. Extract Initial Conditions
    I0 = actual_daily[0] * 5  # Rough approx of active cases vs incidence
    E0 = I0 * 3               # Latently exposed reservoir
    
    # 3. Fit ODE Parameters
    opt_beta, opt_gamma, time_steps, y0 = fit_seir_model(actual_daily, N_india, I0, E0, sigma)
    
    R0_calc = opt_beta / opt_gamma
    print("┌──────────────────────────────────────────────┐")
    print(f"│  Optimized Transmission Rate (β): {opt_beta:.4f}     │")
    print(f"│  Optimized Recovery Rate (γ)    : {opt_gamma:.4f}     │")
    print(f"│  Basic Reproduction Number (R0) : {R0_calc:.2f}      │")
    print("└──────────────────────────────────────────────┘")
    
    # 4. Generate Final Theoretical Simulation
    solution = odeint(seir_deriv, y0, time_steps, args=(N_india, opt_beta, opt_gamma, sigma))
    
    S_sim = solution[:, 0]
    E_sim = solution[:, 1]
    I_sim = solution[:, 2]
    R_sim = solution[:, 3]
    Simulated_Daily_Incidence = sigma * E_sim
    
    # 5. Visualization Map
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1.5, 1]})
    
    # Top Plot: Incidence comparison
    ax1 = axes[0]
    ax1.bar(dates, actual_daily, color='#3498db', alpha=0.6, label='Actual 7-Day Moving Avg (India)')
    ax1.plot(dates, Simulated_Daily_Incidence, color='#e74c3c', linewidth=3, 
             label=f'SEIR ODE Simulated Cases\n(R0 = {R0_calc:.2f})')
    
    ax1.set_title("Epidemic Modeling (SEIR): India 2021 Delta Wave Prediction Match", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Daily New Incidence")
    ax1.grid(color='white', alpha=0.1)
    ax1.legend(loc='upper right')
    
    # Format x-axis dates sparsely
    ax1.set_xticks(dates[::15])
    ax1.tick_params(axis='x', rotation=45)
    
    # Bottom Plot: The S-E-I-R curves overview
    ax2 = axes[1]
    
    # Note: Susceptible drops incredibly slowly as a percentage, so mapping E, I, R is better to visualize
    ax2.plot(dates, E_sim, color='#f39c12', linewidth=2, label='Exposed (E)')
    ax2.plot(dates, I_sim, color='#e74c3c', linewidth=2, label='Actively Infectious (I)')
    ax2.plot(dates, R_sim, color='#2ecc71', linewidth=2, label='Recovered (R)')
    
    ax2.set_title("Compartment Mathematical Trajectories (E-I-R)", fontsize=12)
    ax2.set_ylabel("Population Size")
    ax2.grid(color='white', alpha=0.1)
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    ax2.set_xticks(dates[::15])
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Export
    plot_path = OUTPUT_DIR / "seir_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Visualization successfully preserved to: {plot_path}")
    
if __name__ == "__main__":
    main()
