import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from pathlib import Path
from matplotlib.ticker import FuncFormatter

# Outputs Setup
ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def seir_sandbox_deriv(y, t, N, beta, gamma, sigma):
    """Native ODE Definitions isolating Active Infectious variables."""
    S, E, I, R = y
    
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    
    return dSdt, dEdt, dIdt, dRdt

def run_simulation(days, N, I0, E0, beta_base, gamma_base, sigma, 
                   stringency_level=0, vax_rate=0, compliance=100):
    """
    Executes ODE calculations adjusting boundaries natively.
    - Stringency & Compliance act recursively on Beta (Transmission efficiency)
    - Vax_Rate isolates chunks of population from Susceptible to Recovered 
    """
    
    # Policy Parameter Logic Engine
    # 1. Modify Transmission Bounds (Beta)
    reduction_factor = (stringency_level / 100.0) * (compliance / 100.0)
    adjusted_beta = beta_base * (1.0 - reduction_factor)
    
    # 2. Modify Susceptible Baseline (Vax Shift)
    vax_count = (vax_rate / 100.0) * N
    
    # We must ensure we don't accidentally remove currently sick people
    S0 = max(100.0, N - I0 - E0 - vax_count)
    R0 = vax_count
    
    y0 = [S0, E0, I0, R0]
    t = np.arange(days)
    
    # Process Mathematical Physics Sandbox Map
    solution = odeint(seir_sandbox_deriv, y0, t, args=(N, adjusted_beta, gamma_base, sigma))
    
    # Actively Infectious Trajectory curve is the metric we evaluate logically
    I_curve = solution[:, 2] 
    
    return t, I_curve, adjusted_beta

def evaluate_scenarios():
    st.set_page_config(page_title="Intervention Simulator", layout="wide")
    
    st.title("🦠 What-If Intervention Simulator")
    st.markdown("Forecast outbreak changes under different policy configurations.")
    
    # Base Constants derived natively from the Indian Delta Fit earlier!
    N = 1_393_000_000
    beta_baseline = 0.40
    gamma_baseline = 0.10
    sigma = 1 / 5.2
    
    I0 = 100_000 # Assuming outbreak has started structurally
    E0 = 300_000 
    
    # Create input controls in sidebar
    st.sidebar.header("Policy Controls")
    
    stringency_level = st.sidebar.slider("Stringency Level (0-100)", 0, 100, 50)
    vax_rate = st.sidebar.slider("Vaccination Rate (%)", 0.0, 100.0, 0.0, step=1.0)
    compliance = st.sidebar.slider("Population Compliance (%)", 0, 100, 80)
    days = st.sidebar.slider("Forecast Days", 10, 100, 30) # Forecast next 30 days
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Baseline Beta:** {beta_baseline}")
    st.sidebar.markdown(f"**Total Population:** {N:,}")
    
    # Apply dark mode style for matplotlib
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # ── Scenario 1: Current trajectory (Uncontrolled)
    t, I_uncontrolled, _ = run_simulation(days, N, I0, E0, beta_baseline, gamma_baseline, sigma,
                                          stringency_level=0, vax_rate=0, compliance=100)
    
    # ── Scenario 2: Lockdown scenario
    # High Stringency (75%), Moderate Compliance (80%), Low Vax (5%)
    t, I_lockdown, _ = run_simulation(days, N, I0, E0, beta_baseline, gamma_baseline, sigma,
                                      stringency_level=75, vax_rate=5, compliance=80)
    
    # ── Scenario 3: Vaccination boost scenario
    # Low Stringency (10%), High Vax (50%), High Compliance (90%)
    t, I_vax, _ = run_simulation(days, N, I0, E0, beta_baseline, gamma_baseline, sigma,
                                 stringency_level=10, vax_rate=50, compliance=90)
    
    # ── Scenario 4: User Custom Policy
    t, I_custom, adj_beta = run_simulation(days, N, I0, E0, beta_baseline, gamma_baseline, sigma,
                                 stringency_level=stringency_level, vax_rate=vax_rate, compliance=compliance)
    
    # ── Mapping Mechanics
    ax.plot(t, I_uncontrolled, color='#e74c3c', linewidth=2, linestyle=':', label="Current Trajectory (Base)")
    ax.plot(t, I_lockdown, color='#f39c12', linewidth=2, linestyle='--', label="Lockdown Scenario (75% Stringency)")
    ax.plot(t, I_vax, color='#3498db', linewidth=2, linestyle='-.', label="Vaccination Boost Scenario (50% Vax)")
    
    # Custom Policy Plot
    ax.plot(t, I_custom, color='#2ecc71', linewidth=4, label=f"Custom Policy (Effective Beta: {adj_beta:.2f})")
    
    # Formulating Logic Styling
    ax.set_title("Policy Intervention Sandbox: What-If Future Forecast", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Forecast Days from Implementation", fontsize=12, fontweight='bold')
    ax.set_ylabel("Active Infectious Population (I)", fontsize=12, fontweight='bold')
    
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x):,}"))
    
    ax.grid(color='white', alpha=0.1)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=12)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "simulation_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    st.pyplot(fig)
    
    st.success(f"Plot directly saved to `{output_path}` as requested!")
    
    st.markdown("### Methodology")
    st.markdown("""
    - **Stringency & Compliance:** Multiplying stringency by compliance calculates the reduction factor, which actively lowers the basic transmission rate (β).
    - **Vaccination:** Shifts a proportion of the Susceptible (S) population directly into the Recovered (R) category, factoring into structural immunity.
    """)

if __name__ == "__main__":
    evaluate_scenarios()
