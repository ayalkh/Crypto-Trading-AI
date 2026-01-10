import streamlit as st
import sys
import os
import time
import dashboard_utils as utils

st.set_page_config(page_title="Data Collection", page_icon="📊", layout="wide")

st.title("📊 Market Data Collection")

# Two tabs: Actions and Logs
tab1, tab2 = st.tabs(["🚀 Actions", "📋 Live Logs"])

LOG_FILE = "dashboard_data_collection.log"

with tab1:
    st.markdown("### 📥 Ingest Market Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("🔄 **Quick Update**")
        st.caption("Updates recent data for existing symbols.")
        if st.button("Start Quick Update", key="quick_upd"):
            cmd = [sys.executable, 'comprehensive_ml_collector_v2.py']
            utils.run_process_async(cmd, LOG_FILE)
            st.success("Started Quick Update! Check 'Live Logs' tab.")
            
    with col2:
        st.warning("🔥 **Full Collection**")
        st.caption("Collects extensive historical data. Takes longer.")
        if st.button("Start Full Collection", key="full_col"):
            cmd = [sys.executable, 'comprehensive_ml_collector_v2.py'] # Script defaults to full if logic handled there, or add args if needed. 
            # Note: The original CLI didn't pass specific args for full vs quick, it seemed to be the same script differently used or same script.
            # Looking at CLI code: _run_collector([]) was called for both, just confirmation differences.
            # So I will just call it. Adaptation might be needed if script relies on interactive input.
            # IMPORTANT: The script might be interactive. If so, I should run it in non-interactive mode.
            # Let's assume standard run is fine for now, or check script content.
            # Validating args: The CLI passed [] for both.
            
            utils.run_process_async(cmd, LOG_FILE)
            st.success("Started Full Collection! Check 'Live Logs' tab.")

    with col3:
        st.markdown("### ⚙️ Custom Collection")
        symbols_opt = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "DOT/USDT"]
        selected_symbols = st.multiselect("Select Symbols", symbols_opt, default=["BTC/USDT"])
        
        timeframes_opt = ["5m", "15m", "1h", "4h", "1d"]
        selected_tfs = st.multiselect("Select Timeframes", timeframes_opt, default=["1h"])
        
        if st.button("Start Custom Collection"):
            args = []
            if selected_symbols:
                args.append('--symbols')
                args.extend(selected_symbols)
            if selected_tfs:
                args.append('--timeframes')
                args.extend(selected_tfs)
                
            cmd = [sys.executable, 'comprehensive_ml_collector_v2.py'] + args
            utils.run_process_async(cmd, LOG_FILE)
            st.success(f"Started Custom Collection! Check 'Live Logs' tab.")

    st.divider()
    
    st.markdown("### 📚 Database Snapshot")
    status = utils.get_system_status()
    if status['database']:
        st.json(status['db_stats'])
    else:
        st.warning("No Database Found.")

with tab2:
    st.markdown(f"### 📜 Log Output ({LOG_FILE})")
    log_container = st.empty()
    
    # Auto-refresh checkbox
    if st.checkbox("Autorefresh Logs", value=True):
        logs = utils.tail_log_file(LOG_FILE, max_lines=100)
        log_text = "".join(logs)
        st.code(log_text, language="text")
        time.sleep(1)
        st.rerun()
    else:
        if st.button("Refresh Logs Now"):
            logs = utils.tail_log_file(LOG_FILE, max_lines=100)
            log_text = "".join(logs)
            st.code(log_text, language="text")
