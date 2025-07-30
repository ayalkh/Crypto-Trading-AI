"""
Start script for Enhanced Trading Automation
This script ensures proper encoding and environment setup
"""
import os
import sys
import subprocess

def setup_environment():
    """Setup proper environment for the automation system"""
    
    # Set UTF-8 encoding environment variables
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'
    
    # For Windows, also set console encoding
    if sys.platform.startswith('win'):
        try:
            # Try to set console to UTF-8
            os.system('chcp 65001 > nul')
        except:
            pass

def check_requirements():
    """Check if required files and packages exist"""
    required_files = [
        'enhanced_automation_scheduler.py',
        'multi_timeframe_collector.py', 
        'multi_timeframe_analyzer.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    # Check for required packages
    try:
        import schedule
        import pandas
        import sqlite3
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install with: pip install schedule pandas")
        return False
    
    return True

def main():
    """Main startup function"""
    print("🚀 CRYPTO TRADING AUTOMATION STARTUP")
    print("=" * 40)
    
    # Setup environment
    setup_environment()
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Cannot start automation - missing requirements")
        input("Press Enter to exit...")
        return
    
    print("✅ All requirements met")
    print("🔄 Starting automation system...")
    print("=" * 40)
    
    try:
        # Start the automation system with proper encoding
        result = subprocess.run([
            sys.executable, 'enhanced_automation_scheduler.py'
        ], env=dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1'))
        
    except KeyboardInterrupt:
        print("\n🛑 Automation stopped by user")
    except Exception as e:
        print(f"❌ Error starting automation: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()