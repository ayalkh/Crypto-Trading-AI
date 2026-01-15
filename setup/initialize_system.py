import os
import sys
import subprocess
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def run_step(step_name, command, description):
    """Run a system initialization step"""
    print(f"\n{'='*60}")
    print(f"🚀 STEP: {step_name}")
    print(f"📋 {description}")
    print(f"{'='*60}\n")
    
    try:
        # Use sys.executable to ensure we use the same python interpreter
        cmd = [sys.executable] + command
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output
        for line in iter(process.stdout.readline, ''):
            print(f"   {line.strip()}")
            
        process.wait()
        
        if process.returncode == 0:
            print(f"\n✅ {step_name} completed successfully!")
            return True
        else:
            print(f"\n❌ {step_name} failed with exit code {process.returncode}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error executing {step_name}: {e}")
        return False

def main():
    print("\n⚡ CRYPTO TRADING AI - SYSTEM INITIALIZATION ⚡")
    print("This script will set up your environment for the first time.")
    print("-" * 50)
    print("1. Collect Historical Data (for ML training)")
    print("2. Train Initial Machine Learning Models")
    print("3. Launch Control Center")
    print("-" * 50)
    
    confirm = input("\nStart initialization? (y/n): ").lower().strip()
    if confirm != 'y':
        print("Initialization cancelled.")
        sys.exit(0)

    # Step 1: Data Collection
    # Running without arguments uses defaults (BTC, ETH, etc.)
    if not run_step(
        "Data Collection", 
        ['collectors/comprehensive_ml_collector_v2.py'],
        "Collecting historical OHLCV data for major cryptocurrencies..."
    ):
        print("\n⚠️ Data collection failed. Cannot proceed with training.")
        sys.exit(1)

    # Step 2: Model Training
    if not run_step(
        "Model Training",
        ['optimized_ml_system.py'],
        "Training ML models (XGBoost, LightGBM, CatBoost) on collected data..."
    ):
        print("\n⚠️ Model training failed.")
        choice = input("Continue to Control Center anyway? (y/n): ").lower()
        if choice != 'y':
            sys.exit(1)

    # Step 3: Launch Control Center
    print("\n" + "="*60)
    print("🎉 INITIALIZATION COMPLETE!")
    print("🚀 Launching Control Center...")
    print("="*60 + "\n")
    time.sleep(2)
    
    try:
         # Replace current process with control center
        os.execv(sys.executable, [sys.executable, 'crypto_control_center.py'])
    except Exception as e:
        print(f"Failed to launch control center: {e}")

if __name__ == "__main__":
    main()
