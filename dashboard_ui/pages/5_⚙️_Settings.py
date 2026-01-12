import sys; import os; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import json
import dashboard_utils as utils

st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")

st.title("⚙️ System Settings")

st.markdown("### 🔧 Automation Configuration")

config = utils.load_config()

if config:
    # Convert to string for editing
    config_str = json.dumps(config, indent=4)
    new_config_str = st.text_area("Edit automation_config.json", value=config_str, height=400)
    
    if st.button("💾 Save Configuration"):
        try:
            new_config = json.loads(new_config_str)
            if utils.save_config(new_config):
                st.success("Configuration saved successfully!")
            else:
                st.error("Failed to save configuration.")
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
else:
    st.error("Could not load configuration file.")
    
st.divider()
st.markdown("### 🧹 System Maintenance")

if st.button("🗑️ Clear Dashboard Logs"):
    # Delete the temp dashboard logs
    logs_to_clear = ["dashboard_data_collection.log", "dashboard_ml_training.log", "dashboard_analysis.log"]
    import os
    count = 0
    for log in logs_to_clear:
        path = os.path.join("logs", log)
        if os.path.exists(path):
            os.remove(path)
            count += 1
    st.success(f"Cleared {count} log files.")
