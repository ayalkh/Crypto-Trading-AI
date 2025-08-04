"""
ML Integration Setup Script
Helps set up machine learning capabilities for your crypto trading system
"""
import os
import sys

if sys.platform.startswith('win'):
    try:
        # Try to set UTF-8 encoding for stdout
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, OSError):
        # If reconfigure doesn't work, try alternative
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')
import subprocess
import json
from datetime import datetime

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python version {version.major}.{version.minor} is compatible")
    return True

def install_ml_packages():
    """Install required ML packages"""
    packages = [
        'scikit-learn>=1.0.0',
        'tensorflow>=2.8.0',
        'xgboost>=1.5.0',
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'joblib>=1.0.0'
    ]
    
    print("🔧 Installing ML packages...")
    
    for package in packages:
        try:
            print(f"   Installing {package}...")
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', package
            ], check=True, capture_output=True)
            print(f"   ✅ {package.split('>=')[0]} installed")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to install {package}: {e}")
            return False
    
    print("✅ All ML packages installed successfully")
    return True

def create_ml_directories():
    """Create necessary directories for ML system"""
    directories = [
        'ml_models',
        'ml_predictions',
        'ml_backtest',
        'ml_logs'
    ]
    
    print("📁 Creating ML directories...")
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✅ {directory}/")
    
    print("✅ ML directories created")

def create_ml_config():
    """Create ML-enhanced configuration file"""
    config = {
        "data_collection": {
            "enabled": True,
            "interval_minutes": 60,
            "symbols": [
                "BTC/USDT",
                "ETH/USDT",
                "BNB/USDT",
                "ADA/USDT",
                "DOT/USDT"
            ],
            "timeframes": [
                "5m",
                "15m",
                "1h",
                "4h",
                "1d"
            ],
            "force_update_hours": 24
        },
        "signal_analysis": {
            "enabled": True,
            "interval_minutes": 15,
            "confidence_threshold": 75,
            "analyze_symbols": [
                "BTC/USDT",
                "ETH/USDT",
                "BNB/USDT"
            ]
        },
        "machine_learning": {
            "enabled": True,
            "model_training": {
                "enabled": True,
                "interval_hours": 24,
                "symbols": [
                    "BTC/USDT",
                    "ETH/USDT",
                    "BNB/USDT"
                ],
                "timeframes": [
                    "1h",
                    "4h",
                    "1d"
                ],
                "min_data_points": 100,
                "train_test_split": 0.8
            },
            "prediction": {
                "enabled": True,
                "interval_minutes": 30,
                "confidence_threshold": 0.6,
                "save_predictions": True,
                "prediction_horizon": 1
            },
            "model_types": {
                "traditional_ml": True,
                "deep_learning": True,
                "ensemble": True
            },
            "alerts": {
                "enabled": True,
                "high_confidence_threshold": 0.8,
                "significant_change_threshold": 0.05,
                "send_email": False,
                "send_desktop": True
            },
            "features": {
                "technical_indicators": True,
                "price_patterns": True,
                "volume_analysis": True,
                "time_features": True,
                "lag_features": True,
                "rolling_statistics": True
            }
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
            "daily_report": True,
            "include_ml_metrics": True
        },
        "system": {
            "max_errors": 10,
            "error_cooldown_minutes": 30,
            "cleanup_interval_hours": 24,
            "database_path": "data/multi_timeframe_data.db"
        }
    }
    
    config_file = "automation_config_ml.json"
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"✅ ML configuration created: {config_file}")
    return config_file

def create_startup_scripts():
    """Create convenient startup scripts"""
    
    # Python startup script
    startup_script = """#!/usr/bin/env python3
\"\"\"
Startup script for ML-Enhanced Crypto Trading Automation
\"\"\"
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_automation_scheduler import main

if __name__ == "__main__":
    main()
"""
    
    with open("start_ml_automation.py", 'w') as f:
        f.write(startup_script)
    
    # Batch file for Windows
    batch_script = """@echo off
echo 🧠 Starting ML-Enhanced Crypto Trading Automation...
echo ================================================

REM Set encoding
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

REM Check if virtual environment should be activated
if exist "venv\\Scripts\\activate.bat" (
    echo Activating virtual environment...
    call venv\\Scripts\\activate.bat
)

REM Start the automation system
python start_ml_automation.py

pause
"""
    
    with open("start_ml_automation.bat", 'w') as f:
        f.write(batch_script)
    
    # Shell script for Linux/Mac
    shell_script = """#!/bin/bash
echo "🧠 Starting ML-Enhanced Crypto Trading Automation..."
echo "================================================"

# Set encoding
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

# Check if virtual environment should be activated
if [ -f "venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Start the automation system
python3 start_ml_automation.py
"""
    
    with open("start_ml_automation.sh", 'w') as f:
        f.write(shell_script)
    
    # Make shell script executable on Unix systems
    if os.name != 'nt':
        os.chmod("start_ml_automation.sh", 0o755)
    
    print("✅ Startup scripts created:")
    print("   - start_ml_automation.py")
    print("   - start_ml_automation.bat (Windows)")
    print("   - start_ml_automation.sh (Linux/Mac)")

def test_ml_imports():
    """Test if ML packages can be imported"""
    print("🧪 Testing ML package imports...")
    
    packages_to_test = [
        ('numpy', 'np'),
        ('pandas', 'pd'),
        ('sklearn', None),
        ('tensorflow', 'tf'),
        ('xgboost', 'xgb'),
        ('joblib', None)
    ]
    
    all_good = True
    
    for package, alias in packages_to_test:
        try:
            if alias:
                exec(f"import {package} as {alias}")
            else:
                exec(f"import {package}")
            print(f"   ✅ {package}")
        except ImportError as e:
            print(f"   ❌ {package}: {e}")
            all_good = False
    
    if all_good:
        print("✅ All ML packages imported successfully")
    else:
        print("❌ Some ML packages failed to import")
    
    return all_good

def create_example_training_script():
    """Create example script for initial model training"""
    script_content = """#!/usr/bin/env python3
\"\"\"
Example script for initial ML model training
Run this after collecting some data to train your first models
\"\"\"
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_integration_system import CryptoMLSystem
from ml_enhanced_analyzer import MLEnhancedAnalyzer

def main():
    print("🎓 INITIAL ML MODEL TRAINING")
    print("=" * 40)
    
    # Initialize ML system
    ml_system = CryptoMLSystem()
    
    # Define symbols and timeframes to train
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    timeframes = ['1h', '4h']
    
    print(f"Training models for {len(symbols)} symbols and {len(timeframes)} timeframes...")
    
    for symbol in symbols:
        for timeframe in timeframes:
            print(f"\\n🔄 Training {symbol} {timeframe}...")
            
            try:
                # Train price prediction models
                if ml_system.train_price_prediction_models(symbol, timeframe):
                    print(f"   ✅ Price prediction models trained")
                else:
                    print(f"   ⚠️ Price prediction training failed")
                
                # Train direction prediction models  
                if ml_system.train_direction_prediction_models(symbol, timeframe):
                    print(f"   ✅ Direction prediction models trained")
                else:
                    print(f"   ⚠️ Direction prediction training failed")
                
                # Train LSTM model
                if ml_system.train_lstm_model(symbol, timeframe):
                    print(f"   ✅ LSTM model trained")
                else:
                    print(f"   ⚠️ LSTM training failed")
                    
            except Exception as e:
                print(f"   ❌ Training failed for {symbol} {timeframe}: {e}")
    
    print("\\n✅ Initial model training completed!")
    print("You can now run the ML-enhanced automation system.")

if __name__ == "__main__":
    main()
"""
    
    with open("train_initial_models.py", 'w') as f:
        f.write(script_content)
    
    print("✅ Example training script created: train_initial_models.py")

def main():
    """Main setup function"""
    print("🚀 ML INTEGRATION SETUP FOR CRYPTO TRADING")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install ML packages
    print("\n" + "=" * 30)
    if not install_ml_packages():
        print("❌ Package installation failed")
        return False
    
    # Create directories
    print("\n" + "=" * 30)
    create_ml_directories()
    
    # Create configuration
    print("\n" + "=" * 30)
    config_file = create_ml_config()
    
    # Create startup scripts
    print("\n" + "=" * 30)
    create_startup_scripts()
    
    # Create example training script
    print("\n" + "=" * 30)
    create_example_training_script()
    
    # Test imports
    print("\n" + "=" * 30)
    if not test_ml_imports():
        print("⚠️ Some packages may not work correctly")
    
    # Final instructions
    print("\n" + "🎉 ML INTEGRATION SETUP COMPLETE!" + "\n")
    print("=" * 50)
    print("NEXT STEPS:")
    print("=" * 50)
    print("1. Make sure your data collection is running and has some data")
    print("2. Run initial model training:")
    print("   python train_initial_models.py")
    print("")
    print("3. Start the ML-enhanced automation:")
    print("   python start_ml_automation.py")
    print("   OR")
    print("   Double-click start_ml_automation.bat (Windows)")
    print("   OR")
    print("   ./start_ml_automation.sh (Linux/Mac)")
    print("")
    print("4. Monitor the logs for ML predictions and alerts")
    print("")
    print("📁 Files created:")
    print("   - automation_config_ml.json (ML configuration)")
    print("   - start_ml_automation.py (Main startup script)")
    print("   - train_initial_models.py (Initial training)")
    print("   - Startup scripts for your OS")
    print("")
    print("🧠 ML Features enabled:")
    print("   - Automated model training (daily)")
    print("   - Real-time price predictions")
    print("   - Direction prediction (up/down)")
    print("   - LSTM time series analysis")
    print("   - Smart alerts based on ML confidence")
    print("   - Feature importance analysis")
    print("")
    print("Happy trading with AI! 🚀")
    
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Setup cancelled by user")
    except Exception as e:
        print(f"❌ Setup failed: {e}")