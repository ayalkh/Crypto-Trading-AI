import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from crypto_ai.database.db import DatabaseManager

st.set_page_config(page_title="Data Viewer", page_icon="📊", layout="wide")

st.title("📊 Crypto Data Viewer")

# Initialize DB
db = DatabaseManager()

# Sidebar Controls
st.sidebar.header("Configuration")

# Get Symbols
symbols = db.get_symbols()
if not symbols:
    st.error("No data found in database. Please run data collection first.")
    st.stop()

symbol = st.sidebar.selectbox("Select Symbol", symbols)

# Get Timeframes
timeframes = db.get_timeframes(symbol)
timeframe = st.sidebar.selectbox("Select Timeframe", timeframes)

# Limit
limit = st.sidebar.slider("Number of candles", 50, 2000, 200)

# Load Data
if st.sidebar.button("Load Data", type="primary"):
    with st.spinner("Loading data..."):
        df = db.load_data(symbol, timeframe, limit)
        
        if not df.empty:
            # Calculate basic indicators for display
            df['MA20'] = df['close'].rolling(20).mean()
            df['MA50'] = df['close'].rolling(50).mean()
            
            # --- Plotting ---
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.03, subplot_titles=(f'{symbol} {timeframe} Price', 'Volume'),
                                row_width=[0.2, 0.7])

            # Candlestick
            fig.add_trace(go.Candlestick(x=df['timestamp'],
                            open=df['open'], high=df['high'],
                            low=df['low'], close=df['close'], name='OHLC'), 
                            row=1, col=1)

            # Moving Averages
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['MA20'], line=dict(color='orange', width=1), name='MA 20'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['MA50'], line=dict(color='blue', width=1), name='MA 50'), row=1, col=1)

            # Volume
            fig.add_trace(go.Bar(x=df['timestamp'], y=df['volume'], showlegend=False), row=2, col=1)

            fig.update_layout(xaxis_rangeslider_visible=False, height=800)
            st.plotly_chart(fig, use_container_width=True)
            
            # Data Table
            with st.expander("View Raw Data"):
                st.dataframe(df.sort_values('timestamp', ascending=False))
        else:
            st.warning("No data returned.")
else:
    st.info("Select parameters and click 'Load Data'")
