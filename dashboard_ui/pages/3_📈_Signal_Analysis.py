import sys; import os; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import sys
import pandas as pd
import plotly.graph_objects as go
import dashboard_utils as utils
import time

st.set_page_config(page_title="Signal Analysis", page_icon="📈", layout="wide")

st.title("📈 Signal Analysis & Market View")

tab1, tab2 = st.tabs(["📊 Market Charts", "📋 Run Analysis"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        symbols_opt = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "DOT/USDT"]
        symbol = st.selectbox("Select Symbol", symbols_opt, index=0)
    with col2:
        timeframes_opt = ["5m", "15m", "1h", "4h", "1d"]
        timeframe = st.selectbox("Select Timeframe", timeframes_opt, index=2)
        
    if st.button("Load Chart"):
        conn = utils.get_db_connection()
        if conn:
            try:
                # Find price table query from Database_viewer logic
                query = f"""
                    SELECT timestamp, open, high, low, close, volume
                    FROM price_data
                    WHERE symbol = '{symbol}' AND timeframe = '{timeframe}'
                    ORDER BY timestamp DESC
                    LIMIT 200
                """
                df = pd.read_sql_query(query, conn)
                
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values('timestamp')
                    
                    fig = go.Figure(data=[go.Candlestick(x=df['timestamp'],
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'])])
                        
                    fig.update_layout(title=f'{symbol} - {timeframe} Price Chart', xaxis_title='Date', yaxis_title='Price')
                    st.plotly_chart(fig, width="stretch")
                    
                    st.dataframe(df.tail(10), width="stretch")
                else:
                    st.warning(f"No data found for {symbol} {timeframe}")
            except Exception as e:
                st.error(f"Error loading data: {e}")
            finally:
                conn.close()
        else:
            st.error("Database connection failed.")

LOG_FILE = "dashboard_analysis.log"

with tab2:
    st.markdown("### 🔍 Run Unified Crypto Analyzer")
    st.info("Runs the `unified_crypto_analyzer.py` script to detect trading signals.")
    
    if st.button("🚀 Run Analysis"):
        cmd = [sys.executable, 'unified_crypto_analyzer.py']
        utils.run_process_async(cmd, LOG_FILE)
        st.success("Analysis started! Check logs below.")

    st.markdown(f"**Live Logs ({LOG_FILE})**")
    if st.checkbox("Autorefresh Analysis Logs", value=True):
        logs = utils.tail_log_file(LOG_FILE, max_lines=50)
        st.code("".join(logs), language="text")
        time.sleep(1)
        st.rerun()
    else:
        logs = utils.tail_log_file(LOG_FILE, max_lines=50)
        st.code("".join(logs), language="text")
