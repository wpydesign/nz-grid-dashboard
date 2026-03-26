import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import plotly.graph_objects as go
from datetime import timedelta

# ==========================================
# 1. CONFIGURATION & CACHING
# ==========================================
st.set_page_config(layout="wide", page_title="NZ Grid Stress Monitor")

# Cache data loading to prevent re-running on every slider move
@st.cache_data(ttl=3600) # Refresh every hour
def load_grid_data():
    """
    Loads generation data from EMI.
    Focuses on supply-side only.
    """
    # Simplified loader for the skeleton - loads last 2 years for speed
    # In production, expand date range.
    print("Loading Supply Data...")
    base_url = "https://emi.ea.govt.nz/Wholesale/Datasets/Generation/Generation_MD/"
    # Get last 24 months
    months = pd.date_range(end=pd.Timestamp.now(), periods=24, freq='MS').strftime('%Y%m').tolist()
    
    all_data = []
    for m in months:
        try:
            url = f"{base_url}{m}_Generation_MD.csv"
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                df = pd.read_csv(io.StringIO(r.text), low_memory=False)
                # Aggregate to National Daily (Simplified)
                # Filter for relevant columns if needed, or sum all TP
                tp_cols = [c for c in df.columns if c.startswith('TP')]
                if tp_cols:
                    # Convert kWh to MWh (kWh / 1000) then sum
                    # Daily Total Generation = Sum of all TP
                    daily_gen = df[tp_cols].sum(axis=1) / 1000 # kWh -> MWh
                    # Extract Date
                    date_col = [c for c in df.columns if 'date' in c.lower()][0]
                    df['Date'] = pd.to_datetime(df[date_col])
                    
                    # Group by Date
                    day_sum = df.groupby('Date')[tp_cols].sum().sum(axis=1) / 1000
                    day_sum = day_sum.reset_index()
                    day_sum.columns = ['datetime', 'generation_mw']
                    all_data.append(day_sum)
        except Exception as e:
            pass
            
    if not all_data: return pd.DataFrame()
    
    df = pd.concat(all_data).drop_duplicates().sort_values('datetime').reset_index(drop=True)
    return df

# ==========================================
# 2. STRESS ENGINE (THE EDGE)
# ==========================================
def calculate_structural_stress(df, sensitivity_knob):
    """
    Calculates Stress Z-Score based purely on Supply.
    Knob adjusts the sensitivity threshold logic.
    """
    df = df.copy()
    
    # A. Baseline (Rolling 30d Mean)
    df['baseline'] = df['generation_mw'].rolling(30, min_periods=10).mean().shift(1)
    
    # B. Structural Metrics
    # 1. Deficit: How far below baseline?
    df['deficit'] = (df['baseline'] - df['generation_mw']).clip(lower=0) / df['baseline']
    
    # 2. Shock (Loss Rate): How fast did it drop?
    df['shock'] = (-df['generation_mw'].diff()).clip(lower=0) / df['baseline']
    
    # C. Raw Stress Score
    # Weighted sum: Deficit indicates persistent stress, Shock indicates sudden break.
    # This is the "Edge" logic.
    df['stress_raw'] = 10 * df['deficit'] + 5 * df['shock']
    
    # D. Normalization (Z-Score)
    # Rolling stats to handle seasonality
    r_mean = df['stress_raw'].rolling(30, min_periods=10).mean()
    r_std = df['stress_raw'].rolling(30, min_periods=10).std()
    
    df['stress_z'] = (df['stress_raw'] - r_mean) / r_std
    df['stress_z'] = df['stress_z'].fillna(0)
    
    # E. Knob Logic
    # Knob 1-10.
    # Higher Knob -> Lower Threshold -> More Sensitive.
    # Knob 1 (Low Sens) -> Threshold 3.0 (Only Systemic)
    # Knob 10 (High Sens) -> Threshold 1.0 (Subtle Stress)
    threshold = 4.0 - (sensitivity_knob * 0.3)
    
    df['is_alert'] = df['stress_z'] >= threshold
    
    return df, threshold

# ==========================================
# 3. REASONING MACHINE (RM)
# ==========================================
def generate_explanation(row):
    """
    Generates text explanation for a specific data point.
    """
    if not row['is_alert']:
        return None
        
    reasons = []
    causes = []
    observers = []
    
    # Analyze Causes
    if row['deficit'] > 0.10: # > 10% deficit
        reasons.append("Significant Supply Deficit")
        causes.append("Generation running persistently below seasonal baseline.")
        observers.append("Baseline Monitor")
        
    if row['shock'] > 0.05: # > 5% sudden drop
        reasons.append("Sudden Supply Shock")
        causes.append("Rapid drop in generation (potential plant trip or outage).")
        observers.append("Rate of Change Monitor")
        
    # Combine
    if not reasons: 
        reasons.append("Elevated Stress Pattern")
        causes.append("Complex interaction of supply factors.")
        observers.append("Composite Index")

    # Format Output
    text = f"""
    **Status:** STRESS EVENT DETECTED
    **Time:** {row['datetime'].strftime('%Y-%m-%d')}
    
    **What is happening:**  
    {', '.join(reasons)}
    
    **What causes it:**  
    {' '.join(causes)}
    
    **What observed it:**  
    {', '.join(observers)}
    """
    return text

# ==========================================
# 4. STREAMLIT INTERFACE
# ==========================================

st.title("NZ Grid Structural Stress Monitor")
st.markdown("### Detecting Supply-Side Regime Shifts")

# --- SIDEBAR / CONTROLS ---
with st.sidebar:
    st.header("Controls")
    # Sensitivity Knob
    sensitivity = st.slider(
        "Sensitivity Knob", 
        min_value=1, 
        max_value=10, 
        value=5,
        help="1 = Only Systemic Events, 10 = Subtle Build-ups"
    )
    
    st.markdown("---")
    st.markdown("Data updates hourly.")

# --- LOAD DATA ---
with st.spinner("Loading Grid Data..."):
    df_raw = load_grid_data()

if df_raw.empty:
    st.error("Could not load data from EMI.")
else:
    # --- CALCULATE STRESS ---
    df_final, current_threshold = calculate_structural_stress(df_raw, sensitivity)
    
    # --- MAIN CHART ---
    st.markdown(f"**Current Stress Threshold (Z > {current_threshold:.2f})**")
    
    # Plotly Chart
    fig = go.Figure()
    
    # 1. Stress Signal Line
    fig.add_trace(go.Scatter(
        x=df_final['datetime'], 
        y=df_final['stress_z'],
        mode='lines', 
        name='Stress Intensity (Z)',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # 2. Highlight Alert Zones
    df_alerts = df_final[df_final['is_alert']]
    fig.add_trace(go.Scatter(
        x=df_alerts['datetime'],
        y=df_alerts['stress_z'],
        mode='markers',
        name='Stress Event',
        marker=dict(color='red', size=8, symbol='circle')
    ))
    
    # 3. Threshold Line
    fig.add_hline(y=current_threshold, line_dash="dot", 
                  line_color="red", annotation_text="Threshold")
    
    fig.update_layout(
        title="Structural Stress Intensity Over Time",
        yaxis_title="Stress Z-Score",
        xaxis_title="Date",
        template="plotly_white",
        height=500,
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # --- EXPLANATION LAYER (BOTTOM) ---
    st.markdown("---")
    st.subheader("Reasoning Machine (RM)")
    
    # Get the latest data point
    latest = df_final.iloc[-1]
    
    # Generate Explanation
    explanation = generate_explanation(latest)
    
    if explanation:
        st.error(explanation) # Red box for alerts
    else:
        st.success(f"**Status: NORMAL** \n\nCurrent Stress Z-Score: {latest['stress_z']:.2f}. No structural anomalies detected.")
    
    # Hover details (Optional: Show details for selected point in chart)
    # Streamlit native interaction is limited, so we show the latest state primarily.
