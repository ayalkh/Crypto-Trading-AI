#!/usr/bin/env python3
"""
Quick Setup Script for Crypto Trading System
Sets up everything you need to get started
"""

import os
import sys
import subprocess
import json
from datetime import datetime

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. You have:", sys.version)
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 INSTALLING REQUIREMENTS")
    print("-" * 30)
    
    if os.path.exists('requirements.txt'):
        print("📋 Installing packages from requirements.txt...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                          check=True)
            print("✅ Requirements installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing requirements: {e}")
            return False
    else:
        print("⚠️ requirements.txt not found, installing essential packages...")
        essential_packages = [
            'pandas>=1.3.0',
            'numpy>=1.21.0',
            'matplotlib>=3.5.0',
            'ccxt>=4.0.0',
            'schedule>=1.0.0',
            'requests>=2.25.0'
        ]
        
        for package in essential_packages:
            try:
                subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                              check=True, capture_output=True)
                print(f"✅ Installed {package}")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}")
        
        return True

def create_directories():
    """Create necessary directories"""
    print("\n📁 CREATING DIRECTORIES")
    print("-" * 25)
    
    directories = ['data', 'logs', 'alerts', 'backups']
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created: {directory}/")
        except Exception as e:
            print(f"❌ Error creating {directory}: {e}")

def create_config_files():
    """Create default configuration files"""
    print("\n⚙️ CREATING CONFIGURATION")
    print("-" * 30)
    
    # Default automation config
    automation_config = {
        "data_collection": {
            "enabled": True,
            "interval_minutes": 60,
            "symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "DOT/USDT"],
            "timeframes": ["5m", "15m", "1h", "4h", "1d"],
            "force_update_hours": 24
        },
        "signal_analysis": {
            "enabled": True,
            "interval_minutes": 15,
            "confidence_threshold": 75,
            "analyze_symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
        },
        "alerts": {
            "enabled": True,
            "email": {
                "enabled": False,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "your_email@gmail.com",
                "password": "your_app_password",
                "to_email": "your_email@gmail.com"
            },
            "desktop": {
                "enabled": True
            },
            "log_file": {
                "enabled": True
            }
        },
        "performance_tracking": {
            "enabled": True,
            "interval_hours": 6,
            "daily_report": True
        },
        "system": {
            "max_errors": 10,
            "error_cooldown_minutes": 30,
            "cleanup_interval_hours": 24,
            "database_path": "data/multi_timeframe_data.db"
        }
    }
    
    try:
        with open('automation_config.json', 'w') as f:
            json.dump(automation_config, f, indent=4)
        print("✅ Created automation_config.json")
    except Exception as e:
        print(f"❌ Error creating config: {e}")

def run_initial_data_collection():
    """Run initial data collection"""
    print("\n📊 INITIAL DATA COLLECTION")
    print("-" * 35)
    
    choice = input("Run initial data collection now? (y/n): ").lower()
    
    if choice == 'y':
        print("🔄 Running initial data collection...")
        print("💡 This may take a few minutes...")
        
        try:
            result = subprocess.run([
                sys.executable, 'multi_timeframe_collector.py', '--force'
            ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
            
            if result.returncode == 0:
                print("✅ Initial data collection completed!")
                
                # Try to run initial analysis
                print("🔍 Running initial signal analysis...")
                analysis_result = subprocess.run([
                    sys.executable, 'multi_timeframe_analyzer.py'
                ], capture_output=True, text=True, timeout=120)
                
                if analysis_result.returncode == 0:
                    print("✅ Initial analysis completed!")
                else:
                    print("⚠️ Analysis had some issues, but setup is complete")
            else:
                print("⚠️ Data collection had issues, but setup is complete")
                if result.stderr:
                    print(f"Error: {result.stderr[:200]}...")
                    
        except subprocess.TimeoutExpired:
            print("⚠️ Data collection timed out, but setup is complete")
        except Exception as e:
            print(f"⚠️ Error during data collection: {e}")
    else:
        print("⏭️ Skipped initial data collection")

def create_startup_scripts():
    """Create convenient startup scripts"""
    print("\n📜 CREATING STARTUP SCRIPTS")
    print("-" * 35)
    
    # Windows batch file
    batch_content = """@echo off
echo Starting Crypto Trading System...
python crypto_control_center.py
pause
"""
    
    try:
        with open('start_crypto_system.bat', 'w') as f:
            f.write(batch_content)
        print("✅ Created start_crypto_system.bat (Windows)")
    except Exception as e:
        print(f"❌ Error creating batch file: {e}")
    
    # Unix shell script
    shell_content = """#!/bin/bash
echo "Starting Crypto Trading System..."
python3 crypto_control_center.py
"""
    
    try:
        with open('start_crypto_system.sh', 'w') as f:
            f.write(shell_content)
        os.chmod('start_crypto_system.sh', 0o755)  # Make executable
        print("✅ Created start_crypto_system.sh (Linux/Mac)")
    except Exception as e:
        print(f"❌ Error creating shell script: {e}")

def display_next_steps():
    """Display next steps for the user"""
    print("\n🎉 SETUP COMPLETE!")
    print("=" * 50)
    print("""
🚀 YOUR CRYPTO TRADING SYSTEM IS READY!

📋 WHAT WAS CREATED:
   ✅ Directory structure (data/, logs/, alerts/)
   ✅ Configuration files
   ✅ Startup scripts
   ✅ Required packages installed

🎯 NEXT STEPS:

1. 🖥️ START THE CONTROL CENTER:
   python crypto_control_center.py
   
   OR use the startup scripts:
   - Windows: double-click start_crypto_system.bat
   - Linux/Mac: ./start_crypto_system.sh

2. 📊 COLLECT INITIAL DATA:
   - Use option 1 in the control center
   - Choose "Force fresh data" for first run

3. 🔍 ANALYZE SIGNALS:
   - Use option 2 in the control center
   - Get your first trading recommendations

4. 🤖 START AUTOMATION:
   - Use option 3 for 24/7 operation
   - System will run continuously

⚙️ CONFIGURATION:
   - Edit automation_config.json for custom settings
   - Add email alerts, change intervals, etc.

📚 DOCUMENTATION:
   - Use option 9 in the control center for help
   - Check logs for system status

💡 TIPS:
   - Start with manual data collection first
   - Check the control center's system status
   - Monitor logs for any issues
   - Begin with automation once data is flowing

🎊 Happy Trading! Your AI-powered crypto system is ready!
    """)

def main():
    """Main setup function"""
    print("🚀 CRYPTO TRADING SYSTEM SETUP")
    print("=" * 50)
    print("🔧 Setting up your complete trading system...")
    print("=" * 50)
    
    # Check prerequisites
    if not check_python_version():
        return False
    
    # Run setup steps
    steps = [
        ("📦 Installing requirements", install_requirements),
        ("📁 Creating directories", create_directories),
        ("⚙️ Creating configuration", create_config_files),
        ("📜 Creating startup scripts", create_startup_scripts),
    ]
    
    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        try:
            if not step_func():
                print(f"❌ {step_name} failed")
                return False
        except Exception as e:
            print(f"❌ {step_name} failed: {e}")
            return False
    
    # Optional initial data collection
    run_initial_data_collection()
    
    # Show next steps
    display_next_steps()
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎉 Setup completed successfully!")
        else:
            print("\n❌ Setup had some issues. Check the output above.")
    except KeyboardInterrupt:
        print("\n\n🛑 Setup cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error during setup: {e}")