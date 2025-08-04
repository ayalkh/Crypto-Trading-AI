"""
Clean Restart Script for ML Trading System
Fixes encoding issues and restarts the system cleanly
"""
import os
import sys
import subprocess
import time

def setup_clean_environment():
    """Setup clean environment variables"""
    print("🔧 Setting up clean environment...")
    
    # Force UTF-8 encoding everywhere
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'
    os.environ['PYTHONLEGACYWINDOWSFSENCODING'] = '0'
    os.environ['PYTHONLEGACYWINDOWSSTDIO'] = '0'
    
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    # For Windows - set console to UTF-8
    if sys.platform.startswith('win'):
        try:
            os.system('chcp 65001 > nul 2>&1')
        except:
            pass
    
    print("✅ Environment configured")

def kill_existing_processes():
    """Kill any existing automation processes"""
    print("🛑 Stopping any existing automation processes...")
    
    if sys.platform.startswith('win'):
        try:
            # Kill any existing python processes running our scripts
            subprocess.run(['taskkill', '/f', '/im', 'python.exe', '/fi', 'WINDOWTITLE eq ML*'], 
                         capture_output=True, check=False)
        except:
            pass
    
    print("✅ Existing processes stopped")

def start_ml_system():
    """Start the ML system with clean environment"""
    print("🚀 Starting ML system with clean environment...")
    
    try:
        # Start the ML automation in a new process with clean environment
        env = os.environ.copy()
        env.update({
            'PYTHONIOENCODING': 'utf-8',
            'PYTHONUTF8': '1',
            'PYTHONLEGACYWINDOWSFSENCODING': '0',
            'PYTHONLEGACYWINDOWSSTDIO': '0',
            'TF_CPP_MIN_LOG_LEVEL': '2',
            'TF_ENABLE_ONEDNN_OPTS': '0'
        })
        
        # Import and run directly (avoid subprocess encoding issues)
        print("✅ Importing ML automation system...")
        
        # Add current directory to path
        sys.path.insert(0, os.getcwd())
        
        # Import the main ML automation function
        from ml_automation_scheduler import MLEnhancedTradingAutomation
        
        print("✅ Starting ML-Enhanced Trading Automation...")
        automation = MLEnhancedTradingAutomation()
        automation.start()
        
    except KeyboardInterrupt:
        print("\n🛑 System stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        input("Press Enter to exit...")

def main():
    """Main restart function"""
    print("🔄 CLEAN ML SYSTEM RESTART")
    print("=" * 50)
    
    # Setup environment
    setup_clean_environment()
    
    # Kill existing processes
    kill_existing_processes()
    
    # Wait a moment
    print("⏳ Waiting 2 seconds...")
    time.sleep(2)
    
    # Start system
    start_ml_system()

if __name__ == "__main__":
    main()