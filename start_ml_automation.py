"""
ML-Enhanced Crypto Trading Automation Starter
Python version - no batch file needed!
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
from datetime import datetime

def check_environment():
    """Check if environment is properly set up"""
    # Set UTF-8 encoding
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'
    
    # Suppress TensorFlow warnings for cleaner output
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    # For Windows, also try to set console encoding
    if sys.platform.startswith('win'):
        try:
            os.system('chcp 65001 > nul 2>&1')
        except:
            pass

def print_banner():
    """Print startup banner"""
    print("\n" + "█" * 60)
    print("███" + " " * 54 + "███")
    print("███    🧠 ML-ENHANCED CRYPTO TRADING AUTOMATION    ███")
    print("███" + " " * 18 + "Python Edition" + " " * 18 + "███")
    print("███" + " " * 54 + "███")
    print("█" * 60 + "\n")

def check_ml_packages():
    """Check if required ML packages are available"""
    print("🔍 Checking ML packages...")
    
    required_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'sklearn': 'scikit-learn',
        'tensorflow': 'tensorflow',
        'xgboost': 'xgboost',
        'joblib': 'joblib'
    }
    
    missing = []
    
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"   ✅ {package_name}")
        except ImportError:
            print(f"   ❌ {package_name} - not found")
            missing.append(package_name)
    
    if missing:
        print(f"\n⚠️ Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    print("✅ All ML packages available!")
    return True

def check_directories():
    """Create ML directories if they don't exist"""
    print("\n📁 Checking ML directories...")
    
    directories = ['ml_models', 'ml_predictions', 'ml_logs', 'ml_backtest']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"   ✅ Created {directory}/")
        else:
            print(f"   ✅ {directory}/ exists")

def check_data():
    """Check if we have data to work with"""
    print("\n📊 Checking for data...")
    
    db_path = "data/multi_timeframe_data.db"
    if os.path.exists(db_path):
        print(f"   ✅ Database found: {db_path}")
        return True
    else:
        print(f"   ⚠️ Database not found: {db_path}")
        print("   Run your data collector first: python multi_timeframe_collector.py")
        return False

def check_models():
    """Check if ML models exist"""
    print("\n🧠 Checking for ML models...")
    
    model_files = []
    if os.path.exists('ml_models'):
        model_files = [f for f in os.listdir('ml_models') 
                      if f.endswith('.joblib') or f.endswith('.h5')]
    
    if model_files:
        print(f"   ✅ Found {len(model_files)} trained models")
        return True
    else:
        print("   ⚠️ No trained models found")
        return False

def offer_training():
    """Offer to train models if none exist"""
    print("\n🎓 No ML models found.")
    
    choice = input("Would you like to train initial models now? (y/n): ").lower().strip()
    
    if choice in ['y', 'yes']:
        print("\n🔄 Training initial ML models...")
        try:
            # Check if training script exists
            if os.path.exists('train_initial_models.py'):
                result = subprocess.run([sys.executable, 'train_initial_models.py'], 
                                      capture_output=False)
                return result.returncode == 0
            else:
                print("❌ train_initial_models.py not found")
                return False
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
    else:
        print("⚠️ Starting without ML models. Models will be trained automatically.")
        return True

def start_automation():
    """Start the ML automation system"""
    print("\n🚀 Starting ML-Enhanced Trading Automation...")
    print("=" * 50)
    print("📊 Data Collection: Every 60 minutes")
    print("🧠 ML Predictions: Every 30 minutes")
    print("🎓 Model Training: Daily")
    print("🚨 Smart Alerts: High confidence only")
    print("=" * 50)
    print("🛑 Press Ctrl+C to stop the automation")
    print("=" * 50)
    
    try:
        # Import and run the ML automation system
        if os.path.exists('ml_automation_scheduler.py'):
            # Import the main function from the ML automation scheduler
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from ml_automation_scheduler import main
            main()
        else:
            print("❌ ml_automation_scheduler.py not found")
            print("Make sure all ML files are in your project directory")
            
    except KeyboardInterrupt:
        print("\n🛑 Automation stopped by user")
    except Exception as e:
        print(f"❌ Automation failed: {e}")

def main():
    """Main startup function"""
    # Setup environment
    check_environment()
    
    # Print banner
    print_banner()
    
    # Check if everything is ready
    print("🔧 Running startup checks...")
    
    # Check ML packages
    if not check_ml_packages():
        print("\n❌ Please install missing ML packages first")
        input("Press Enter to exit...")
        return
    
    # Create directories
    check_directories()
    
    # Check for data
    has_data = check_data()
    if not has_data:
        print("\n⚠️ No data found. You should run your data collector first.")
        choice = input("Continue anyway? (y/n): ").lower().strip()
        if choice not in ['y', 'yes']:
            return
    
    # Check for models
    has_models = check_models()
    if not has_models and has_data:
        if not offer_training():
            print("❌ Cannot continue without models or training")
            input("Press Enter to exit...")
            return
    
    print("\n✅ All checks passed!")
    
    # Start the automation
    start_automation()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Startup cancelled by user")
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        input("Press Enter to exit...")