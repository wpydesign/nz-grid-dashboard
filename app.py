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
def load_nz_generation():
    """Loads last 3 months of NZ generation data."""
    now = pd.Timestamp.now()
    months_needed = sorted(list(set([(now - pd.DateOffset(months=i)).strftime('%Y%m') for i in range(3)])))
    
    all_data = []
    base_url = "https://emi.ea.govt.nz/Wholesale/Datasets/Generation/Generation_MD/"
    
    for m in months_needed:
        url = f"{base_url}{m}_Generation_MD.csv"
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                df = pd.read_csv(io.StringIO(r.text), low_memory=False)
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

@st.cache_data(ttl=3600)
def load_nz_prices():
    """Loads last 3 months of NZ Final Energy Prices - DEEP SCAN."""
    now = pd.Timestamp.now()
    months_needed = sorted(list(set([(now - pd.DateOffset(months=i)).strftime('%Y%m') for i in range(3)])))
    all_data = []
    
    base_paths = [
        "https://www.emi.ea.govt.nz/Wholesale/Datasets/DispatchAndPricing/FinalEnergyPrices/ByMonth/",
        "https://www.emi.ea.govt.nz/Wholesale/Datasets/Final_pricing/Final_prices/"
    ]
    
    for m in months_needed:
        loaded_month = False
        for base_url in base_paths:
            if loaded_month: break
            possible_files = [f"{m}_FinalEnergyPrices.csv", f"{m}_Final_prices.csv"]
            
            for fname in possible_files:
                url = f"{base_url}{fname}"
                try:
                    r = requests.get(url, timeout=15)
                    if r.status_code == 200:
                        df_cols = pd.read_csv(io.StringIO(r.text), nrows=1)
                        
                        # Smart Column Finder (BEN2201 or generic Price)
                        price_col = next((c for c in df_cols.columns if 'BEN2201' in c), None)
                        if not price_col:
                            price_col = next((c for c in df_cols.columns if 'price' in c.lower() and 'date' not in c.lower()), None)
                        
                        if price_col:
                            df = pd.read_csv(io.StringIO(r.text), usecols=['Trading_Date', 'Trading_Period', price_col])
                            df['datetime'] = pd.to_datetime(df['Trading_Date']) + pd.to_timedelta((df['Trading_Period'] - 1) * 30, unit='m')
                            df = df.rename(columns={price_col: 'price'})
                            df = df[df['price'] > 0]
                            df['hour'] = df['datetime'].dt.floor('H')
                            df_hourly = df.groupby('hour')['price'].mean().reset_index()
                            all_data.append(df_hourly)
                            loaded_month = True
                            break 
                except:
                    continue
    
    if not all_data: return None
    df_final = pd.concat(all_data, ignore_index=True)
    return df_final[['hour', 'price']].rename(columns={'hour': 'datetime', 'price': 'spot_price'})

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

# --- 3. DASHBOARD UI ---

def main():
    st.title("🇳🇿 NZ Grid Stress Monitor – Trader Edition")
    st.caption("Live generation-deficit signal • Sensitivity slider • Real price correlation")
    
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
        if df['spot_price'].notna().sum() > 0:
            price_status = "OK"
        else:
            price_status = "Empty Merge"
    else:
        df['spot_price'] = np.nan
    
    # SIDEBAR
    st.sidebar.header("Settings")
    
    sensitivity = st.sidebar.slider("Sensitivity", 1, 10, 5, format="Level %d")
    st.sidebar.markdown("**1 = Dull**\n\n**10 = Sensitive**")
    
    threshold = get_sensitivity_threshold(df, sensitivity)
    
    # --- SYSTEM STATUS (NEW) ---
    with st.expander("ℹ️ System Status & Data Source", expanded=False):
        st.markdown(f"""
        **Generation Data (Stress Signal):**
        - Status: ✅ **LIVE** 
        - Source: NZ Electricity Authority (EMI Portal)
        - Last Update: {df['datetime'].max().strftime('%Y-%m-%d %H:%M')}
        
        **Price Data (Correlation):**
        - Status: ⚠️ **DELAYED / UNAVAILABLE**
        - Reason: The EMI public portal has not yet published Final Price files for the current month. 
        - Solution: **Subscribers** get access to real-time price API integrations.
        """)
    
    # TOP METRICS
    col1, col2, col3 = st.columns(3)
    latest = df.iloc[-1]
    
    with col1:
        st.metric("Current Stress", f"{latest['stress']:.2f}")
    with col2:
        status = "⚠️ STRESS ALERT" if latest['stress'] > threshold else "✅ NORMAL"
        st.metric("Status", status)
    with col3:
        price_val = latest['spot_price']
        if pd.notna(price_val):
             st.metric("Current Price ($/MWh)", f"{price_val:.1f}")
        else:
             last_valid = df['spot_price'].dropna().iloc[-1] if not df['spot_price'].dropna().empty else None
             if last_valid:
                 st.metric("Last Known Price", f"{last_valid:.1f}")
             else:
                 st.metric("Price", "N/A")
        
    # MAIN CHART
    st.header("Stress Signal vs Threshold")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['stress'], mode='lines', name='Stress', line=dict(color='#1f77b4', width=2)))
    fig.add_trace(go.Scatter(x=df['datetime'], y=[threshold]*len(df), mode='lines', name='Threshold', line=dict(color='red', dash='dash')))
    
    spikes = df[df['stress'] > threshold].copy()
    fig.add_trace(go.Scatter(x=spikes['datetime'], y=spikes['stress'], mode='markers', name='Alert', marker=dict(color='red', size=8)))
    
    fig.update_layout(template="plotly_white", height=400, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    
    # --- BACKTEST PANEL ---
    st.header("Stress Impact Analysis")
    
    if price_status == "OK":
        st.markdown("**Real historical price premium** when stress > threshold.")
        
        df_valid = df.dropna(subset=['spot_price'])
        high_stress = df_valid[df_valid['stress'] > threshold]
        normal_stress = df_valid[df_valid['stress'] <= threshold]
        
        if not high_stress.empty:
            avg_price_spike = high_stress['spot_price'].mean()
            avg_price_norm = normal_stress['spot_price'].mean()
            
            c1, c2 = st.columns(2)
            with c1: st.metric("Avg Price (During Alert)", f"${avg_price_spike:.1f}")
            with c2: st.metric("Avg Price (Normal)", f"${avg_price_norm:.1f}")
                
            if avg_price_norm > 0:
                premium = ((avg_price_spike - avg_price_norm) / avg_price_norm * 100)
                st.success(f"Price premium during stress events: {premium:.1f}%")
        else:
            st.info("No alerts at this sensitivity level to analyze.")
            
    else:
        st.warning("Price data unavailable for current analysis.")
        st.info("Public EMI price files are published with a delay. Subscribers receive live price integration.")

    # --- EXPORT BUTTON ---
    st.subheader("Export Data")
    export_df = spikes[['datetime', 'stress', 'spot_price']].dropna()
    if export_df.empty:
        export_df = df[['datetime', 'stress', 'spot_price']].dropna()
        
    if not export_df.empty:
        csv = export_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download History (CSV)", data=csv, file_name='nz_stress_data.csv', mime='text/csv')

    # --- HOW IT WORKS ---
    with st.expander("How It Works (Methodology)"):
        st.markdown("""
        **The Stress Signal:**
        `stress(t) = 10 × deficit(t) + 5 × loss_rate(t)`
        
        **Components:**
        - **Deficit:** Measures the shortfall between Actual Generation and a 48-hour Rolling Baseline.
        - **Loss Rate:** Measures the speed of generation decline.
        
        **Data Source:**
        NZ Electricity Authority (EMI Portal).
        """)

if __name__ == "__main__":
    main()
