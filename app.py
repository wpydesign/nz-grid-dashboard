import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import plotly.graph_objects as go

st.set_page_config(page_title="NZ Grid Stress Monitor", layout="wide")

# ==================== DATA LOADING ====================
@st.cache_data(ttl=3600)
def load_nz_generation():
    """Loads last 90 days of generation data."""
    now = pd.Timestamp.now()
    months_needed = [(now - pd.DateOffset(months=i)).strftime('%Y%m') for i in range(3)]
    all_data = []
    base_url = "https://emi.ea.govt.nz/Wholesale/Datasets/Generation/Generation_MD/"
    
    for m in sorted(set(months_needed)):
        url = f"{base_url}{m}_Generation_MD.csv"
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                df = pd.read_csv(io.StringIO(r.text), low_memory=False)
                date_col = next((c for c in df.columns if 'date' in c.lower()), None)
                if date_col:
                    df['Trading_Date'] = pd.to_datetime(df[date_col])
                all_data.append(df)
        except:
            continue
    if not all_data:
        return None
    df = pd.concat(all_data, ignore_index=True)
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
    """Loads last 30 days of Benmore spot prices — robust daily loader."""
    now = pd.Timestamp.now()
    dates_needed = [now - pd.DateOffset(days=i) for i in range(35)]   # extra buffer
    all_data = []
    base_url = "https://www.emi.ea.govt.nz/Wholesale/Datasets/DispatchAndPricing/FinalEnergyPrices/"
    
    for d in dates_needed:
        filename = d.strftime('%Y%m%d_FinalEnergyPrices.csv')
        url = f"{base_url}{filename}"
        try:
            r = requests.get(url, timeout=15)
            if r.status_code != 200:
                continue
            # Read only header first
            df_cols = pd.read_csv(io.StringIO(r.text), nrows=1)
            
            # Smart column detection for Benmore price
            price_col = None
            for candidate in ['BEN2201', 'BEN2201_Price', 'Price', 'BEN2202', 'Price_BEN']:
                if candidate in df_cols.columns:
                    price_col = candidate
                    break
            if not price_col:
                price_col = next((c for c in df_cols.columns if 'BEN' in c.upper()), None)
            
            if price_col:
                df = pd.read_csv(io.StringIO(r.text), usecols=['Trading_Date', 'Trading_Period', price_col])
                df['datetime'] = pd.to_datetime(df['Trading_Date']) + pd.to_timedelta((df['Trading_Period'] - 1) * 30, unit='m')
                df = df.rename(columns={price_col: 'price'})
                df = df[df['price'] > 0]
                df['hour'] = df['datetime'].dt.floor('H')
                df_hourly = df.groupby('hour')['price'].mean().reset_index()
                all_data.append(df_hourly)
        except:
            continue
    
    if not all_data:
        return None
    df_final = pd.concat(all_data, ignore_index=True)
    return df_final[['hour', 'price']].rename(columns={'hour': 'datetime', 'price': 'spot_price'})


# ==================== CORE LOGIC ====================
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


# ==================== DASHBOARD ====================
def main():
    st.title("🇳🇿 NZ Grid Stress Monitor – Trader Edition")
    st.caption("Live generation-deficit signal • Sensitivity slider • Real Benmore price correlation")

    with st.spinner("Fetching latest generation & price data..."):
        df_gen = load_nz_generation()
        df_prices = load_nz_prices()

    if df_gen is None:
        st.error("Failed to load generation data.")
        return

    df = calculate_stress_signal(df_gen)
    df = df.dropna(subset=['stress'])

    # Merge real prices
    if df_prices is not None:
        df = df.merge(df_prices, on='datetime', how='left')
    else:
        df['spot_price'] = np.nan
        st.warning("⚠️ Price data temporarily unavailable (EMI may be updating files).")

    # SIDEBAR – Your exact knob
    st.sidebar.header("Settings")
    sensitivity = st.sidebar.slider(
        "Sensitivity",
        1, 10, 5,
        format="Level %d",
        help="1 = Dull (very high certainty, few alerts) | 10 = Sensitive (more alerts, more noise)"
    )
    st.sidebar.markdown("**1 = Dull**  **10 = Sensitive**")

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
        price_val = latest.get('spot_price')
        st.metric("Current Benmore Price ($/MWh)", f"{price_val:.1f}" if pd.notna(price_val) else "N/A")

    # CHART
    st.header("Stress Signal vs Threshold")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['stress'], mode='lines', name='Stress', line=dict(color='#1f77b4', width=2)))
    fig.add_trace(go.Scatter(x=df['datetime'], y=[threshold]*len(df), mode='lines', name='Threshold', line=dict(color='red', dash='dash')))
    spikes = df[df['stress'] > threshold]
    fig.add_trace(go.Scatter(x=spikes['datetime'], y=spikes['stress'], mode='markers', name='Alert', marker=dict(color='red', size=8)))
    fig.update_layout(template="plotly_white", height=400, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # IMPACT ANALYSIS
    st.header("Stress Impact Analysis")
    st.markdown("**Real historical price premium** when stress > threshold (last 30 days)")
    if 'spot_price' in df.columns and not df['spot_price'].isna().all():
        high = df[df['stress'] > threshold]
        normal = df[df['stress'] <= threshold]
        if not high.empty:
            avg_spike = high['spot_price'].mean()
            avg_norm = normal['spot_price'].mean()
            c1, c2 = st.columns(2)
            with c1: st.metric("Avg Price (During Alert)", f"${avg_spike:.1f}")
            with c2: st.metric("Avg Price (Normal)", f"${avg_norm:.1f}")
            if avg_norm > 0:
                premium = ((avg_spike - avg_norm) / avg_norm * 100)
                st.success(f"Price premium during stress events: {premium:.1f}%")
        else:
            st.info("No alerts at this sensitivity level.")
    else:
        st.error("Price data unavailable for analysis.")

    # EXPORT
    st.subheader("Export Data")
    export_df = spikes[['datetime', 'stress', 'spot_price']].dropna()
    if not export_df.empty:
        csv = export_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Alert History (CSV)", csv, "nz_stress_alerts.csv", "text/csv")
    else:
        st.info("No alerts to export yet.")

    # HOW IT WORKS
    with st.expander("How It Works (Methodology)"):
        st.markdown("""
        **The Stress Signal:**  
        `stress(t) = 10 × deficit(t) + 5 × loss_rate(t)`  
        - **Deficit** = shortfall vs 48-hour rolling baseline  
        - **Loss Rate** = speed of generation decline  
        **Sensitivity knob:** 1 = Dull (high certainty)  10 = Sensitive (more noise)
        Data from Electricity Authority EMI Portal.
        """)

if __name__ == "__main__":
    main()
