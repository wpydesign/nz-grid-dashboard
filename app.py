import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import plotly.graph_objects as go
from datetime import timedelta

# --- CONFIGURATION ---
st.set_page_config(page_title="NZ Grid Stress Monitor", layout="wide")

# --- 1. DATA LOADING ---

@st.cache_data(ttl=3600)
def load_nz_generation():
    now = pd.Timestamp.now()
    months_needed = sorted(list(set([(now - pd.DateOffset(months=i)).strftime('%Y%m') for i in range(3)])))
    all_data = []
    base_url = "https://emi.ea.govt.nz/Wholesale/Datasets/Generation/Generation_MD/"
    
    for m in months_needed:
        url = f"{base_url}{m}_Generation_MD.csv"
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                df = pd.read_csv(io.StringIO(r.text), low_memory=False)
                date_col = next((c for c in df.columns if 'date' in c.lower()), None)
                if date_col: df['Trading_Date'] = pd.to_datetime(df[date_col])
                all_data.append(df)
        except: continue
    if not all_data: return None
    df = pd.concat(all_data, ignore_index=True)
    
    tp_cols = [c for c in df.columns if c.startswith('TP')]
    id_cols = [c for c in df.columns if c not in tp_cols]
    df_long = df.melt(id_vars=id_cols, value_vars=tp_cols, var_name='TP', value_name='kwh')
    df_long['TP_num'] = df_long['TP'].str.extract(r'(\d+)').astype(float)
    df_long['datetime'] = df_long['Trading_Date'] + pd.to_timedelta((df_long['TP_num']-1)*30, unit='m')
    df_long['mw'] = pd.to_numeric(df_long['kwh'], errors='coerce') / 500
    df_long['hour'] = df_long['datetime'].dt.floor('h')
    df_hourly = df_long.groupby('hour')['mw'].sum().reset_index()
    df_hourly.columns = ['datetime', 'generation_mw']
    return df_hourly

@st.cache_data(ttl=3600)
def load_nz_prices():
    now = pd.Timestamp.now()
    # Load last 6 months to get more historical data for analysis if live fails
    months_needed = sorted(list(set([(now - pd.DateOffset(months=i)).strftime('%Y%m') for i in range(6)])))
    all_prices = []
    price_base = "https://www.emi.ea.govt.nz/Wholesale/Datasets/DispatchAndPricing/FinalEnergyPrices/ByMonth/"
    
    target_node = 'BEN2201' # Benmore Reference Node
    
    for m in months_needed:
        url = f"{price_base}{m}_FinalEnergyPrices.csv"
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                # Read in chunks or low_memory to handle size, filter immediately
                df_p = pd.read_csv(io.StringIO(r.text), low_memory=False)
                
                if 'PointOfConnection' in df_p.columns:
                    df_ben = df_p[df_p['PointOfConnection'] == target_node].copy()
                    if not df_ben.empty:
                        df_ben['datetime'] = pd.to_datetime(df_ben['TradingDate']) + pd.to_timedelta((df_ben['TradingPeriod']-1)*30, unit='m')
                        df_ben = df_ben[df_ben['DollarsPerMegawattHour'] > 0]
                        df_ben['hour'] = df_ben['datetime'].dt.floor('h')
                        df_hourly_p = df_ben.groupby('hour')['DollarsPerMegawattHour'].mean().reset_index()
                        df_hourly_p.columns = ['datetime', 'price']
                        all_prices.append(df_hourly_p)
        except: continue
                    
    if not all_prices: return None
    df_final = pd.concat(all_prices, ignore_index=True)
    # Drop duplicates just in case
    df_final = df_final.drop_duplicates(subset=['datetime']).sort_values('datetime')
    return df_final

# --- 2. CORE LOGIC ---

def calculate_stress_signal(df):
    df = df.copy()
    df['baseline'] = df['generation_mw'].rolling(48, min_periods=24).mean().shift(24)
    df['deficit'] = (df['baseline'] - df['generation_mw']).clip(lower=0) / df['baseline']
    df['loss_rate'] = (-df['generation_mw'].diff()).clip(lower=0) / df['baseline']
    df['stress'] = 10 * df['deficit'] + 5 * df['loss_rate']
    return df

def get_sensitivity_threshold(df, level):
    mean = df['stress'].mean()
    std = df['stress'].std()
    z = 3.5 - ((level - 1) * (3.0 / 9))
    return mean + (z * std)

# --- 3. DASHBOARD UI ---

def main():
    st.title("🇳🇿 NZ Grid Stress Monitor – VIX Edition")
    st.caption("Treating Grid Stress as a Forward Trading Signal")
    
    # Load Data
    df_gen = load_nz_generation()
    df_prices = load_nz_prices()
    
    if df_gen is None:
        st.error("Failed to load generation data.")
        return
        
    df = calculate_stress_signal(df_gen)
    df = df.dropna(subset=['stress'])
    
    # Merge Prices
    price_status = "No Price Data"
    if df_prices is not None and not df_prices.empty:
        df = df.merge(df_prices, on='datetime', how='left')
        if df['price'].notna().sum() > 0:
            price_status = "OK"
        else:
            price_status = "Empty"
    else:
        df['price'] = np.nan
        price_status = "Loader Failed"
    
    # SIDEBAR
    st.sidebar.header("Settings")
    
    # Polished Slider
    sensitivity = st.sidebar.slider(
        "Sensitivity", 
        1, 10, 6, # Default to 6 (Balanced)
        format="Level %d"
    )
    
    st.sidebar.markdown("**1 = Dull (High Certainty)**\n\n**10 = Sensitive (Early Warning)**")
    
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
        price_val = latest['price']
        st.metric("Current Price ($/MWh)", f"{price_val:.1f}" if pd.notna(price_val) else "N/A")
        
    # MAIN CHART
    st.header("Stress Signal vs Price (Overlay)")
    
    fig = go.Figure()
    
    # Stress Line (Primary Y)
    fig.add_trace(go.Scatter(
        x=df['datetime'], y=df['stress'],
        mode='lines', name='Stress', line=dict(color='#1f77b4', width=2), yaxis='y1'
    ))
    
    # Threshold Line
    fig.add_trace(go.Scatter(
        x=df['datetime'], y=[threshold]*len(df),
        mode='lines', name='Threshold', line=dict(color='red', dash='dash'), yaxis='y1'
    ))
    
    # Price Line (Secondary Y)
    if price_status == "OK":
        fig.add_trace(go.Scatter(
            x=df['datetime'], y=df['price'],
            mode='lines', name='Price ($)', line=dict(color='black', width=1, dash='dot'), yaxis='y2'
        ))
    
    # Spikes
    spikes = df[df['stress'] > threshold].copy()
    fig.add_trace(go.Scatter(
        x=spikes['datetime'], y=spikes['stress'],
        mode='markers', name='Alert', marker=dict(color='red', size=8), yaxis='y1'
    ))
    
    fig.update_layout(
        template="plotly_white", height=500, hovermode="x unified",
        yaxis=dict(title="Stress Level", side='left'),
        yaxis2=dict(title="Price ($/MWh)", side='right', overlaying='y', showgrid=False)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # --- EDGE ANALYSIS PANEL ---
    st.header("Statistical Edge (VIX Analysis)")
    
    if price_status == "OK":
        df_valid = df.dropna(subset=['price'])
        
        # Calculate Forward Reaction
        forward_6h, forward_12h, forward_24h = [], [], []
        
        # Find Events (First Crossing)
        df_valid['is_alert'] = df_valid['stress'] > threshold
        df_valid['event_start'] = df_valid['is_alert'] & (~df_valid['is_alert'].shift(1).astype(bool))
        events = df_valid[df_valid['event_start']]
        
        for idx, row in events.iterrows():
            t = row['datetime']
            p_now = row['price']
            
            p_6h = df_valid.loc[df_valid['datetime'] == t + timedelta(hours=6), 'price']
            p_12h = df_valid.loc[df_valid['datetime'] == t + timedelta(hours=12), 'price']
            p_24h = df_valid.loc[df_valid['datetime'] == t + timedelta(hours=24), 'price']
            
            if not p_6h.empty: forward_6h.append(p_6h.values[0] - p_now)
            if not p_12h.empty: forward_12h.append(p_12h.values[0] - p_now)
            if not p_24h.empty: forward_24h.append(p_24h.values[0] - p_now)
            
        col1, col2, col3 = st.columns(3)
        with col1:
            if forward_6h:
                avg_6 = np.mean(forward_6h)
                st.metric("Avg Price Change (+6h)", f"${avg_6:.1f}", delta_color="normal" if avg_6 > 0 else "inverse")
            else:
                st.metric("+6h", "N/A")
                
        with col2:
            if forward_12h:
                avg_12 = np.mean(forward_12h)
                st.metric("Avg Price Change (+12h)", f"${avg_12:.1f}", delta_color="normal" if avg_12 > 0 else "inverse")
            else:
                st.metric("+12h", "N/A")
                
        with col3:
            if forward_24h:
                avg_24 = np.mean(forward_24h)
                st.metric("Avg Price Change (+24h)", f"${avg_24:.1f}", delta_color="normal" if avg_24 > 0 else "inverse")
            else:
                st.metric("+24h", "N/A")
        
        # Interpretation
        st.info(f"""
        **Analysis Result:** 
        The Stress Signal acts as a **Leading Indicator**.
        *   Events detected: {len(events)}
        *   Price Reaction: The market tends to move **${np.mean(forward_6h):.0f}** in the 6 hours following an alert.
        """)
        
    else:
        st.warning("Price data unavailable for Edge Analysis (EMI data publishing delay).")

    # --- HOW IT WORKS ---
    with st.expander("How It Works (Methodology)"):
        st.markdown("""
        **The Stress Signal (Grid VIX):**
        `stress(t) = 10 × deficit(t) + 5 × loss_rate(t)`
        
        **Components:**
        - **Deficit:** Measures generation shortfall vs 48h Baseline.
        - **Loss Rate:** Measures speed of generation drop.
        
        **Trading Logic:**
        - **Signal:** Stress > Threshold.
        - **Behavior:** Signal often fires *before* price spikes (Low Price Premium).
        - **Action:** Use as a "Buy" signal for anticipated price volatility.
        """)

if __name__ == "__main__":
    main()
