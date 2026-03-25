import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import plotly.graph_objects as go

# --- CONFIGURATION ---
st.set_page_config(page_title="NZ Grid Stress Monitor", layout="wide")

# --- 1. DATA LOADING ---

@st.cache_data(ttl=3600)
def load_nz_data():
    """Loads last 90 days of NZ generation data."""
    now = pd.Timestamp.now()
    months_needed = []
    for i in range(3):
        m_date = now - pd.DateOffset(months=i)
        months_needed.append(m_date.strftime('%Y%m'))
    
    all_data = []
    base_url = "https://emi.ea.govt.nz/Wholesale/Datasets/Generation/Generation_MD/"
    
    for m in sorted(list(set(months_needed))):
        url = f"{base_url}{m}_Generation_MD.csv"
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                df = pd.read_csv(io.StringIO(r.text), low_memory=False)
                # Robust Date Handling
                date_col = next((c for c in df.columns if 'date' in c.lower()), None)
                if date_col: df['Trading_Date'] = pd.to_datetime(df[date_col])
                all_data.append(df)
        except: continue
            
    if not all_data: return None
    df = pd.concat(all_data, ignore_index=True)
    
    # Process to Hourly
    tp_cols = [c for c in df.columns if c.startswith('TP')]
    id_cols = [c for c in df.columns if c not in tp_cols]
    df_long = df.melt(id_vars=id_cols, value_vars=tp_cols, var_name='TP', value_name='kwh')
    
    df_long['TP_num'] = df_long['TP'].str.extract(r'(\d+)').astype(float)
    df_long['datetime'] = df_long['Trading_Date'] + pd.to_timedelta((df_long['TP_num']-1)*30, unit='m')
    df_long['mw'] = pd.to_numeric(df_long['kwh'], errors='coerce') / 500
    
    df_long['hour'] = df_long['datetime'].dt.floor('H')
    df_hourly = df_long.groupby('hour')['mw'].sum().reset_index()
    df_hourly.columns = ['datetime', 'generation_mw']
    
    return df_hourly

# --- 2. CORE LOGIC ---

def calculate_stress_signal(df, gen_col='generation_mw'):
    df = df.copy()
    df['baseline'] = df[gen_col].rolling(48, min_periods=24).mean().shift(24)
    df['deficit'] = (df['baseline'] - df[gen_col]).clip(lower=0) / df['baseline']
    df['loss_rate'] = (-df[gen_col].diff()).clip(lower=0) / df['baseline']
    df['stress'] = 10 * df['deficit'] + 5 * df['loss_rate']
    return df

def get_sensitivity_threshold(df, level):
    mean = df['stress'].mean()
    std = df['stress'].std()
    z = 3.5 - ((level - 1) * (3.0 / 9))
    return mean + (z * std)

def classify_event(avg_loss, avg_deficit):
    if avg_loss > 0.005:
        return "Acute (Trip/Outage)"
    else:
        return "Systemic (Fuel/Shortage)"

# --- 3. DASHBOARD UI ---

def main():
    st.title("🇳🇿 NZ Grid Stress Monitor")
    
    with st.spinner("Fetching latest grid data (Last 90 days)..."):
        df = load_nz_data()
    
    if df is None:
        st.error("Failed to load data.")
        return
        
    df = calculate_stress_signal(df)
    df = df.dropna(subset=['stress'])
    
    # SIDEBAR
    st.sidebar.header("Settings")
    sensitivity = st.sidebar.slider("Sensitivity", 1, 10, 5, 
                                    help="1=Low Noise (Traders), 10=High Recall (Analysts)")
    
    threshold = get_sensitivity_threshold(df, sensitivity)
    
    # TOP METRICS
    col1, col2, col3 = st.columns(3)
    latest = df.iloc[-1]
    
    with col1:
        st.metric("Current Stress", f"{latest['stress']:.2f}")
    with col2:
        status = "⚠️ STRESS ALERT" if latest['stress'] > threshold else "✅ NORMAL"
        st.metric("Status", status)
    with col3:
        st.metric("Threshold", f"{threshold:.2f}")
        
    # MAIN CHART
    st.header("Stress Signal vs Threshold")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['stress'], mode='lines', name='Stress', line=dict(color='#1f77b4')))
    fig.add_trace(go.Scatter(x=df['datetime'], y=[threshold]*len(df), mode='lines', name='Threshold', line=dict(color='red', dash='dash')))
    
    # Highlight Spikes
    spikes = df[df['stress'] > threshold]
    fig.add_trace(go.Scatter(x=spikes['datetime'], y=spikes['stress'], mode='markers', name='Alert', marker=dict(color='red', size=8)))
    
    fig.update_layout(template="plotly_white", height=400, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    
    # BACKTEST PANEL
    st.header("Stress Impact Analysis")
    st.markdown("Simulated analysis of what happens to prices when stress > threshold.")
    
    # Simulate Price Proxy (since we don't have live price API here)
    np.random.seed(42)
    df['price_proxy'] = 100 + (df['stress'] * 20) + np.random.normal(0, 10, len(df))
    
    high_stress = df[df['stress'] > threshold]
    normal_stress = df[df['stress'] <= threshold]
    
    if not high_stress.empty:
        avg_price_spike = high_stress['price_proxy'].mean()
        avg_price_norm = normal_stress['price_proxy'].mean()
        
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Avg Price (During Alert)", f"${avg_price_spike:.1f}")
        with c2:
            st.metric("Avg Price (Normal)", f"${avg_price_norm:.1f}")
            
        st.success(f"Price premium during stress events: {(avg_price_spike - avg_price_norm):.1f}%")
    else:
        st.info("No stress events detected at this sensitivity to analyze.")

if __name__ == "__main__":
    main()
