import streamlit as st
import sys
import os
import time
import dashboard_utils as utils

st.set_page_config(page_title="Model Training", page_icon="🧠", layout="wide")

st.title("🧠 Machine Learning Training")

tab1, tab2 = st.tabs(["🏋️ Training", "📋 Live Logs"])

LOG_FILE = "dashboard_ml_training.log"

with tab1:
    st.markdown("### 🚀 Train New Models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("⚡ **Quick Training**")
        st.caption("Trains models on 1h and 4h timeframes only.")
        if st.button("Start Quick Training"):
            # The CLI logic for Option 2 was just running optimized_ml_system.py directly.
            # Assuming it has defaults or handles args. 
            # CLI: _run_ml_training_direct('optimized_ml_system.py')
            cmd = [sys.executable, 'optimized_ml_system.py']
            utils.run_process_async(cmd, LOG_FILE)
            st.success("Started Quick Training! Check 'Live Logs' tab.")
            
    with col2:
        st.error("🚀 **Train All Models**")
        st.caption("Trains ALL models for ALL timeframes (5m, 15m, 1h, 4h, 1d). ~1-2 hours.")
        if st.button("Start Full Training"):
            # CLI: _run_ml_training([]) -> same script
            cmd = [sys.executable, 'optimized_ml_system.py']
            utils.run_process_async(cmd, LOG_FILE)
            st.success("Started Full Training! Check 'Live Logs' tab.")

    st.divider()
    
    st.markdown("### 📦 Model Inventory")
    status = utils.get_system_status()
    st.metric("Total Models", status['models'])
    
    # Show file list if possible
    if os.path.exists('ml_models'):
        files = os.listdir('ml_models')
        model_files = [f for f in files if f.endswith('.joblib') or f.endswith('.h5')]
        if model_files:
            st.dataframe(model_files, width="stretch", column_config={0: "Model Filename"})
        else:
            st.info("No models found.")
    else:
        st.info("ml_models directory does not exist.")

with tab2:
    st.markdown(f"### 📜 Log Output ({LOG_FILE})")
    
    if st.checkbox("Autorefresh Logs", value=True, key="mk_logs"):
        logs = utils.tail_log_file(LOG_FILE, max_lines=100)
        log_text = "".join(logs)
        st.code(log_text, language="text")
        time.sleep(2)
        st.rerun()
    else:
        if st.button("Refresh Logs Now"):
            logs = utils.tail_log_file(LOG_FILE, max_lines=100)
            log_text = "".join(logs)
            st.code(log_text, language="text")
