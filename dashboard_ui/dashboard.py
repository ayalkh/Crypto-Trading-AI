import sys; import os; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import pandas as pd
import os
import dashboard_utils as utils

st.set_page_config(
    page_title="Crypto Trading AI Control Center",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🚀 Crypto Trading AI Control Center")

# System Status Dashboard
st.markdown("### 🏥 System Health Overview")

status = utils.get_system_status()

col1, col2, col3, col4 = st.columns(4)

with col1:
    if status['database']:
        st.metric("Database", "Online", f"{status['db_stats'].get('records', 0):,} records")
    else:
        st.metric("Database", "Offline", "No DB Found")

with col2:
    st.metric("ML Models", f"{status['models']}", "Trained Models")

with col3:
    if status['automation']:
        st.metric("Automation", "Running", delta="Active 24/7", delta_color="normal")
    else:
        st.metric("Automation", "Stopped", delta="Inactive", delta_color="off")

with col4:
    if status['uptime']:
        st.metric("Uptime", status['uptime'])
    else:
        st.metric("Uptime", "0:00:00")

st.divider()

# Quick Actions / Overview
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📊 Recent Activity")
    # Show latest data points if DB exists
    conn = utils.get_db_connection()
    if conn:
        try:
            query = """
                SELECT symbol, timeframe, MAX(timestamp) as latest_update, close as current_price 
                FROM price_data 
                GROUP BY symbol, timeframe 
                ORDER BY latest_update DESC 
                LIMIT 5
            """
            df = pd.read_sql_query(query, conn)
            st.dataframe(df, width="stretch")
        except Exception as e:
            st.info(f"Could not load recent activity: {e}")
        finally:
            conn.close()
    else:
        st.info("Database not initialized. Go to 'Data Collection' to start.")

with col_right:
    st.subheader("🤖 Quick Actions")
    st.markdown("""
    - **📊 Data Collection**: Update market data.
    - **🧠 Model Training**: Train new ML models.
    - **📈 Analysis**: View signals and charts.
    - **⚙️ Settings**: Configure automation.
    """)
    
    if st.button("🔄 Refresh Status"):
        st.rerun()

st.sidebar.success("Select a page above ☝️")

st.markdown("---")
st.caption(f"Crypto Trading AI System | Local Time: {pd.Timestamp.now()}")
