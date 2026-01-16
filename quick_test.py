#!/usr/bin/env python3
"""Quick automation component test"""
import os
import sys
import json
import sqlite3

print("=" * 50)
print("AUTOMATION SYSTEM QUICK TEST")
print("=" * 50)

# Test 1: Config
print("\n[1/4] CONFIG TEST")
try:
    with open('automation_config.json') as f:
        config = json.load(f)
    print("  ✓ Config loaded successfully")
    print(f"    - Data collection enabled: {config.get('data_collection', {}).get('enabled')}")
    print(f"    - Signal analysis enabled: {config.get('signal_analysis', {}).get('enabled')}")
except Exception as e:
    print(f"  ✗ Config error: {e}")

# Test 2: Database
print("\n[2/4] DATABASE TEST")
try:
    db_path = 'data/multi_timeframe_data.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM price_data')
    count = cursor.fetchone()[0]
    cursor.execute('SELECT MAX(timestamp) FROM price_data')
    latest = cursor.fetchone()[0]
    conn.close()
    print("  ✓ Database accessible")
    print(f"    - Records: {count:,}")
    print(f"    - Latest: {latest}")
except Exception as e:
    print(f"  ✗ Database error: {e}")

# Test 3: Scheduler module
print("\n[3/4] SCHEDULER MODULE TEST")
try:
    from crypto_ai.automation.scheduler import CryptoAutomationScheduler
    scheduler = CryptoAutomationScheduler()
    print("  ✓ Scheduler imports correctly")
    print(f"    - Config loaded: {bool(scheduler.config)}")
except Exception as e:
    print(f"  ✗ Scheduler error: {e}")

# Test 4: Core scripts exist
print("\n[4/4] CORE SCRIPTS TEST")
scripts = [
    ('comprehensive_ml_collector_v2.py', 'Data Collector'),
    ('unified_crypto_analyzer.py', 'Signal Analyzer'),
    ('crypto_ai/automation/scheduler.py', 'Scheduler'),
]
for script, name in scripts:
    if os.path.exists(script):
        print(f"  ✓ {name}: {script}")
    else:
        print(f"  ✗ {name}: NOT FOUND")

print("\n" + "=" * 50)
print("TEST COMPLETE")
print("=" * 50)
