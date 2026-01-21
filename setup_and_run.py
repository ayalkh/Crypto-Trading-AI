#!/usr/bin/env python3
"""
Crypto Trading AI - Complete Setup and Initialization
======================================================
This script runs the complete workflow from scratch:
1. Data Collection (collect_data.py)
2. Model Training (train_models.py)
3. Prediction Generation (generate_predictions.py)
4. Agent Analysis (run_agent.py)
5. Launch Control Center

Perfect for first-time setup or complete system refresh.
"""

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
    print(f"\n{'='*70}")
    print(f"🚀 STEP: {step_name}")
    print(f"📋 {description}")
    print(f"{'='*70}\n")
    
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
    print("\n" + "="*70)
    print("⚡ CRYPTO TRADING AI - COMPLETE SETUP & INITIALIZATION ⚡")
    print("="*70)
    print("\nThis script will run the complete workflow:")
    print("-" * 70)
    print("1. 📊 Collect Historical Data (for ML training)")
    print("2. 🧠 Train Machine Learning Models (CatBoost + XGBoost)")
    print("3. 🔮 Generate ML Predictions")
    print("4. 🤖 Run Agent Analysis")
    print("5. 🎮 Launch Control Center")
    print("-" * 70)
    print("\n⏱️  Estimated time: 10-30 minutes (depending on data volume)")
    print("💡 You can run individual steps later from the Control Center")
    
    confirm = input("\n🚀 Start complete setup? (y/n): ").lower().strip()
    if confirm != 'y':
        print("\n❌ Setup cancelled.")
        sys.exit(0)

    # Step 1: Data Collection
    print("\n" + "🔹"*35)
    print("PHASE 1/4: DATA COLLECTION")
    print("🔹"*35)
    
    if not run_step(
        "Data Collection", 
        ['collect_data.py'],
        "Collecting historical OHLCV data for major cryptocurrencies..."
    ):
        print("\n⚠️ Data collection failed. Cannot proceed with training.")
        print("💡 Check your internet connection and API access.")
        sys.exit(1)

    # Step 2: Model Training

    print("\n" + "🔹"*35)
    print("PHASE 2/4: MODEL TRAINING")
    print("🔹"*35)
    
    if not run_step(
        "Model Training",
        ['train_models.py', '--auto-run'],  # Added --auto-run flag
        "Training ML models (CatBoost + XGBoost) with feature selection..."
    ):
        print("\n⚠️ Model training failed.")
        choice = input("Continue to prediction generation anyway? (y/n): ").lower()
        if choice != 'y':
            print("\n❌ Setup aborted.")
            sys.exit(1)

    # Step 3: Generate Predictions
    print("\n" + "🔹"*35)
    print("PHASE 3/4: PREDICTION GENERATION")
    print("🔹"*35)
    
    if not run_step(
        "Prediction Generation",
        ['generate_predictions.py'],
        "Generating ML predictions and saving to database..."
    ):
        print("\n⚠️ Prediction generation failed.")
        choice = input("Continue to agent analysis anyway? (y/n): ").lower()
        if choice != 'y':
            print("\n❌ Setup aborted.")
            sys.exit(1)

    # Step 4: Run Agent Analysis
    print("\n" + "🔹"*35)
    print("PHASE 4/4: AGENT ANALYSIS")
    print("🔹"*35)
    
    if not run_step(
        "Agent Analysis",
        ['run_agent.py'],
        "Running agent to analyze market and identify opportunities..."
    ):
        print("\n⚠️ Agent analysis failed.")
        choice = input("Launch Control Center anyway? (y/n): ").lower()
        if choice != 'y':
            print("\n❌ Setup aborted.")
            sys.exit(1)

    # Final: Launch Control Center
    print("\n" + "="*70)
    print("🎉 SETUP COMPLETE!")
    print("="*70)
    print("\n✅ All systems initialized successfully!")
    print("📊 Data collected and stored")
    print("🧠 ML models trained and ready")
    print("🔮 Predictions generated")
    print("🤖 Agent analysis completed")
    print("\n🚀 Launching Control Center in 3 seconds...")
    print("="*70 + "\n")
    
    time.sleep(3)
    
    try:
         # Replace current process with control center
        os.execv(sys.executable, [sys.executable, 'control_center.py'])
    except Exception as e:
        print(f"❌ Failed to launch control center: {e}")
        print("\n💡 You can manually start it with:")
        print(f"   python control_center.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Setup interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error during setup: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
