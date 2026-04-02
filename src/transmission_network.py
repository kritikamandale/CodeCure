"""
CodeCure - Transmission Network Modeling Structure
==================================================
Maps physical outbreak correlations across borders. Heavy edge weights indicate
a massive parallel infection spread chronologically, often indicative of viral
transportation between geographical regions matching wave structures.
"""

import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
from pyvis.network import Network
from pathlib import Path

# Outputs Setup
ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# URL matching standard raw JHU confirmed dataset
JHU_URL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"

def build_correlation_matrix():
    """
    Downloads structural 50+ country vectors, calculating direct 7-day 
    smoothed correlation parameters tying transmission waves.
    """
    print("Loading global JHU matrices dynamically...")
    df = pd.read_csv(JHU_URL)
    
    # 1. Aggregate cases strictly by Country mathematically
    # Drop positional parameters
    clean_df = df.drop(columns=['Province/State', 'Lat', 'Long'])
    country_grouped = clean_df.groupby('Country/Region').sum()
    
    # Identify top 50 structural epicenters out of the 190 available to keep the graph aesthetically dense
    total_cases = country_grouped.iloc[:, -1]
    top_50 = total_cases.sort_values(ascending=False).head(50).index
    
    # Process the sequence timelines
    dataset = country_grouped.loc[top_50].T # Transpose making Date the index
    
    # Perform mathematical Daily Incidence diff structure
    daily_cases = dataset.diff().fillna(0).clip(lower=0)
    
    # Formulate chronological 14-day wave curves for Pearson scaling mathematically
    smoothed = daily_cases.rolling(window=14, min_periods=1).mean()
    
    # Correlation Matrix computation bounds
    print("Executing statistical Pearson correlations mapping node edges...")
    corr_matrix = smoothed.corr(method='pearson')
    
    return corr_matrix, total_cases.loc[top_50]

def render_interactive_graph(corr_matrix, cumulative_cases):
    """
    Builds the native standalone HTML physics engine utilizing PyVis
    rendering parameters against NetworkX models.
    """
    # 1. Initialize NetworkX
    G = nx.Graph()
    
    countries = corr_matrix.columns
    
    # Mathematical Node limits formulation
    # Size based on explicit logarithmic base calculations preventing the US from dwarfing visualization visually
    max_log = np.log1p(cumulative_cases.max())
    min_log = np.log1p(cumulative_cases.min())
    
    print("Generating structural Nodes and clustered Hub networks...")
    for c in countries:
        c_val = cumulative_cases[c]
        log_val = np.log1p(c_val)
        
        # Scale sizes strictly to PyVis bounds [10 -> 50 diameter bounds]
        size = 10 + 40 * ((log_val - min_log) / (max_log - min_log))
        
        # Group bounds for Continent visualization hooks dynamically
        group = 1 # We use simple groupings for pure hackathon aesthetic randomization based on case load rank
        
        G.add_node(c, size=size, title=f"Country: {c}<br>Total Historic Cases: {c_val:,.0f}", group=group, label=c)
        
    # Mathematical Edges formulation
    # Only establish literal connections if correlation > 0.82 representing undeniable identical outbreak vectors
    THRESHOLD = 0.85
    
    for i in range(len(countries)):
        for j in range(i+1, len(countries)):
            c1 = countries[i]
            c2 = countries[j]
            weight = corr_matrix.loc[c1, c2]
            
            if weight >= THRESHOLD:
                width = 3 * (weight ** 5) # Power scaler mapping strictly to visibility geometry
                G.add_edge(c1, c2, value=weight*100, width=width, title=f"Correlation Match: {weight*100:.1f}%")
                
    # 2. PyVis Exportation Layer
    print("Compiling interactive Physics graph engine...")
    net = Network(height='800px', width='100%', bgcolor='#1a1a2e', font_color='white', directed=False)
    net.from_nx(G)
    
    # Internal Node Repulsion mechanics (Barnes-Hut equations structure spreading identical hubs apart)
    net.repulsion(node_distance=150, central_gravity=0.1, spring_length=150, spring_strength=0.05, damping=0.9)
    
    plot_path = str(OUTPUT_DIR / "network_graph.html")
    
    # Explicitly avoid opening the browser in headless terminal structures
    net.save_graph(plot_path)
    print(f"✅ Success. Transmission layout correctly saved: {plot_path}")

def main():
    print("┌──────────────────────────────────────────────┐")
    print("│ Phase: Global Transmission Network           │")
    print("└──────────────────────────────────────────────┘")
    corr_matrix, cumulatives = build_correlation_matrix()
    render_interactive_graph(corr_matrix, cumulatives)

if __name__ == "__main__":
    main()
