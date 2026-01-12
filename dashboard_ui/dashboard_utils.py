import os
import sys
import sqlite3
import pandas as pd
import json
import subprocess
import time
from datetime import datetime
import streamlit as st
import threading

# Constants
DB_PATH = 'data/ml_crypto_data.db'
CONFIG_PATH = 'automation_config.json'
LOGS_DIR = 'logs'
ML_MODELS_DIR = 'ml_models'

def get_db_connection():
    """Get connection to the SQLite database"""
    if not os.path.exists(DB_PATH):
        return None
    try:
        conn = sqlite3.connect(DB_PATH)
        return conn
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        return None

def get_system_status():
    """Get overall system status"""
    status = {
        'database': False,
        'db_stats': {},
        'models': 0,
        'automation': False,
        'uptime': None
    }
    
    # Check Database
    if os.path.exists(DB_PATH):
        status['database'] = True
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(DISTINCT symbol) FROM price_data")
            status['db_stats']['symbols'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM price_data")
            status['db_stats']['records'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT MAX(timestamp) FROM price_data")
            latest = cursor.fetchone()[0]
            status['db_stats']['latest_data'] = latest
            
            conn.close()
        except Exception:
            pass
            
    # Check Models
    if os.path.exists(ML_MODELS_DIR):
        # Count actual models, excluding aux files
        all_files = os.listdir(ML_MODELS_DIR)
        # Filter for actual model files (xgboost, lightgbm, catboost, h5)
        # Exclude scalers and feature lists
        models = [f for f in all_files if 
                 (f.endswith('.joblib') or f.endswith('.h5')) and 
                 not f.endswith('_scaler.joblib') and 
                 not f.endswith('_features.joblib')]
        status['models'] = len(models)
        
    # Check Automation
    pid_file = os.path.join(LOGS_DIR, 'automation.pid')
    if os.path.exists(pid_file):
        try:
            with open(pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            import psutil
            if psutil.pid_exists(pid):
                process = psutil.Process(pid)
                # Check if it looks like our scheduler
                # cmdline might be ['python', 'crypto_ai/automation/scheduler.py']
                cmdline = process.cmdline()
                if any('scheduler' in arg for arg in cmdline):
                    status['automation'] = True
                else:
                    # PID exists but might be reused by another process? 
                    # Or maybe prompt passed 'scheduler.py' relative path.
                    # Be lenient but careful.
                    pass
        except ImportError:
            # Fallback for systems without psutil (though reqs has it)
            try:
                os.kill(pid, 0)
                status['automation'] = True
            except OSError:
                pass
        except Exception:
            pass
            
    # Check Uptime - ONLY if automation is running
    # If not running, start_time is stale/irrelevant
    if status['automation']:
        start_time_file = os.path.join(LOGS_DIR, 'start_time.txt')
        if os.path.exists(start_time_file):
            try:
                with open(start_time_file, 'r') as f:
                    start_time_str = f.read().strip()
                    start_time = datetime.fromisoformat(start_time_str)
                    uptime = datetime.now() - start_time
                    status['uptime'] = str(uptime).split('.')[0] # Remove microseconds
            except:
                pass
    else:
        # If not running but files exist, they are stale. 
        # Optional: could clean them up here, but getter shouldn't mutate state ideally.
        pass
            
    return status

def load_config():
    """Load configuration details"""
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_config(config):
    """Save configuration details"""
    try:
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=4)
        return True
    except Exception as e:
        st.error(f"Failed to save config: {e}")
        return False

def tail_log_file(filename, max_lines=50):
    """Read specific log file"""
    log_path = os.path.join(LOGS_DIR, filename)
    if not os.path.exists(log_path):
        return ["No logs found."]
        
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
            return lines[-max_lines:]
    except Exception as e:
        return [f"Error reading log: {e}"]

def run_process_async(command, log_file):
    """Run a command asynchronously and pipe output to a log file"""
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = os.path.join(LOGS_DIR, log_file)
    
    with open(log_path, "w") as f:
        subprocess.Popen(
            command,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True
        )
