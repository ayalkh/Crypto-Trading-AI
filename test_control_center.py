#!/usr/bin/env python3
"""
Minimal Test Control Center
Quick test to ensure system works
"""

import os
import sys
import subprocess
import json
from datetime import datetime

def test_basic_functionality():
    """Test basic system components"""
    print("🧪 TESTING BASIC FUNCTIONALITY")
    print("-" * 40)
    
    # Test 1: Check core files
    core_files = [
        'multi_timeframe_collector.py',
        'multi_timeframe_analyzer.py'
    ]
    
    print("📁 Checking core files:")
    for file in core_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} - MISSING!")
    
    # Test 2: Check directories
    directories = ['data', 'logs', 'alerts', 'backups']
    print(f"\n📂 Checking directories:")
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"   ✅ {directory}/")
        else:
            print(f"   ⚠️ {directory}/ - Creating...")
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"   ✅ {directory}/ - Created!")
            except Exception as e:
                print(f"   ❌ {directory}/ - Error: {e}")
    
    # Test 3: Check configuration
    config_file = 'automation_config.json'
    print(f"\n⚙️ Checking configuration:")
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            print(f"   ✅ {config_file} - Valid JSON")
        except Exception as e:
            print(f"   ❌ {config_file} - Invalid: {e}")
    else:
        print(f"   ⚠️ {config_file} - Missing, creating default...")
        create_default_config()
    
    # Test 4: Python imports
    print(f"\n🐍 Testing Python imports:")
    
    required_packages = [
        'pandas', 'numpy', 'sqlite3', 
        'subprocess', 'datetime', 'json'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - Not available!")

def create_default_config():
    """Create default configuration"""
    config = {
        "data_collection": {
            "enabled": True,
            "interval_minutes": 60,
            "symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT"],
            "timeframes": ["5m", "15m", "1h", "4h", "1d"]
        },
        "signal_analysis": {
            "enabled": True,
            "interval_minutes": 15,
            "confidence_threshold": 75
        },
        "alerts": {
            "enabled": True,
            "desktop": {"enabled": True},
            "log_file": {"enabled": True}
        }
    }
    
    try:
        with open('automation_config.json', 'w') as f:
            json.dump(config, f, indent=4)
        print(f"   ✅ automation_config.json - Created!")
    except Exception as e:
        print(f"   ❌ Error creating config: {e}")

def quick_menu():
    """Quick menu for testing"""
    while True:
        print(f"\n🎯 QUICK TEST MENU")
        print("-" * 25)
        print("1. 🧪 Run System Test")
        print("2. 📊 Test Data Collection") 
        print("3. 🔍 Test Signal Analysis")
        print("4. 🏃 Run Full Control Center")
        print("0. 🚪 Exit")
        
        choice = input("\nSelect option: ").strip()
        
        if choice == "1":
            test_basic_functionality()
            
        elif choice == "2":
            print("🔄 Testing data collection...")
            try:
                result = subprocess.run([
                    sys.executable, 'multi_timeframe_collector.py', '--diagnose'
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    print("✅ Data collector test passed!")
                    print("📋 Output preview:")
                    print(result.stdout[-300:])  # Last 300 chars
                else:
                    print("❌ Data collector test failed!")
                    if result.stderr:
                        print(f"Error: {result.stderr[:200]}")
                        
            except subprocess.TimeoutExpired:
                print("⏱️ Data collector test timed out")
            except Exception as e:
                print(f"❌ Error testing data collector: {e}")
                
        elif choice == "3":
            print("🔄 Testing signal analysis...")
            
            # Check if database exists first
            if not os.path.exists('data/multi_timeframe_data.db'):
                print("⚠️ No database found. Run data collection first!")
                continue
                
            try:
                result = subprocess.run([
                    sys.executable, 'multi_timeframe_analyzer.py'
                ], capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    print("✅ Signal analyzer test passed!")
                    print("📋 Output preview:")
                    print(result.stdout[-300:])  # Last 300 chars
                else:
                    print("❌ Signal analyzer test failed!")
                    if result.stderr:
                        print(f"Error: {result.stderr[:200]}")
                        
            except subprocess.TimeoutExpired:
                print("⏱️ Signal analyzer test timed out")
            except Exception as e:
                print(f"❌ Error testing signal analyzer: {e}")
                
        elif choice == "4":
            print("🚀 Starting full control center...")
            try:
                subprocess.run([sys.executable, 'crypto_control_center.py'])
            except Exception as e:
                print(f"❌ Error starting control center: {e}")
                print("💡 This test menu can help debug the issue!")
                
        elif choice == "0":
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice")
        
        input("\n⏸️ Press Enter to continue...")

def main():
    """Main function"""
    print("🧪 CRYPTO TRADING SYSTEM - QUICK TEST")
    print("=" * 50)
    print("🔧 Testing your system components...")
    print("=" * 50)
    
    # Run initial test
    test_basic_functionality()
    
    print(f"\n✅ Basic tests complete!")
    print(f"💡 Use the menu to test individual components")
    
    # Start menu
    quick_menu()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Test cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")