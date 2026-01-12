import sys; import os; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import sys
import os
import signal
import time
import subprocess
import dashboard_utils as utils
from datetime import datetime

st.set_page_config(page_title="Automation Control", page_icon="🤖", layout="wide")

st.title("🤖 Automation Control Center")

status = utils.get_system_status()
pid_file = os.path.join('logs', 'automation.pid')

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Current Status")
    if status['automation']:
        st.success("✅ **Automation is RUNNING**")
        if status['uptime']:
            st.metric("Uptime", status['uptime'])
    else:
        st.error("🔴 **Automation is STOPPED**")

with col2:
    st.markdown("### Controls")
    if status['automation']:
        if st.button("🛑 STOP AUTOMATION", type="primary"):
            try:
                with open(pid_file, 'r') as f:
                    pid = int(f.read().strip())
                os.kill(pid, signal.SIGTERM)
                # Cleanup
                if os.path.exists(pid_file):
                    os.remove(pid_file)
                st.success("Stopped successfully!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Error stopping: {e}")
    else:
        if st.button("🚀 START 24/7 AUTOMATION"):
            try:
                scheduler_script = os.path.join('crypto_ai', 'automation', 'scheduler.py')
                # We need to detach this process properly so it survives
                # In Streamlit, this is tricky. We'll use subprocess.Popen
                
                os.makedirs('logs', exist_ok=True)
                
                # Using nohup-like behavior or just detached Popen
                # For simplicity in this demo environment:
                with open('logs/scheduler_stdout.log', 'w') as out, open('logs/scheduler_stderr.log', 'w') as err:
                    process = subprocess.Popen(
                        [sys.executable, scheduler_script],
                        stdout=out,
                        stderr=err,
                        # start_new_session=True # Unix only
                        preexec_fn=os.setsid if sys.platform != 'win32' else None
                    )
                
                with open(pid_file, 'w') as f:
                    f.write(str(process.pid))
                    
                with open('logs/start_time.txt', 'w') as f:
                    f.write(datetime.now().isoformat())
                    
                st.success(f"Started Automation (PID: {process.pid})")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Failed to start: {e}")

st.divider()

st.markdown("### 📜 Scheduler Logs")
log_file = "scheduler.log"

if st.checkbox("Autorefresh Scheduler Logs", value=True):
    logs = utils.tail_log_file(log_file, max_lines=50)
    st.code("".join(logs), language="text")
    time.sleep(2)
    st.rerun()
else:
    logs = utils.tail_log_file(log_file, max_lines=50)
    st.code("".join(logs), language="text")
