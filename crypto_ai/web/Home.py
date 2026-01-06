import streamlit as st
import pandas as pd
import os
import sys
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

st.set_page_config(
    page_title="Crypto AI Trading",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🤖 Crypto AI Control Center")

# Sidebar Status
st.sidebar.header("System Status")

if os.path.exists('logs/automation.pid'):
    st.sidebar.success("✅ Automation Running")
    if os.path.exists('logs/start_time.txt'):
        with open('logs/start_time.txt', 'r') as f:
            start_time = f.read().strip()
        st.sidebar.caption(f"Since: {start_time[:19]}")
else:
    st.sidebar.error("🛑 Automation Stopped")

# Main Dashboard
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Active Models", "5 (Demo)", "+1")
    
with col2:
    st.metric("Latest Signal", "BTC/USDT BUY", "+2.5%")

with col3:
    st.metric("System Health", "98%", "Stable")

st.markdown("---")

st.subheader("📋 Recent Activity")
try:
    if os.path.exists('logs/scheduler.log'):
        with open('logs/scheduler.log', 'r') as f:
            lines = f.readlines()[-10:]
            for line in reversed(lines):
                st.text(line.strip())
    else:
        st.info("No scheduler logs found yet.")
except Exception as e:
    st.error(f"Error reading logs: {e}")

st.markdown("---")
st.info("👈 Select a page from the sidebar to view data or manage ML models.")
