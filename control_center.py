#!/usr/bin/env python3
"""
Crypto Trading Control Center - Updated for Unified ML System
Unified interface for your ML-powered trading system
"""

import os
import sys
import subprocess
import json
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import time
import threading

class CryptoControlCenter:
    def __init__(self):
        """Initialize the control center"""
        # Updated database path for unified system
        self.db_path = 'data/ml_crypto_data.db'
        self.config_path = 'automation_config.json'
        self.automation_process = None
        self.version = "3.0.0 - ML Edition"
    
    def display_banner(self):
        """Display the control center banner"""
        print("🚀 CRYPTO TRADING CONTROL CENTER - ML EDITION")
        print("=" * 65)
        print("🧠 AI-Powered ML Trading System | LightGBM + XGBoost + CatBoost")
        print("💡 Multi-Timeframe Analysis | ML Predictions | 24/7 Automation")
        print("=" * 65)

    def check_system_status(self):
        """Check the status of all system components"""
        print("\n📊 SYSTEM STATUS CHECK")
        print("-" * 45)
        
        status = {
            'database': False,
            'config': False,
            'ml_models': False,
            'data_fresh': False
        }
        
        # Check database
        if os.path.exists(self.db_path):
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Get symbol count
                cursor.execute("SELECT COUNT(DISTINCT symbol) FROM price_data")
                symbol_count = cursor.fetchone()[0]
                
                # Get latest data info
                cursor.execute("""
                    SELECT symbol, MAX(timestamp) as latest, COUNT(*) as records
                    FROM price_data 
                    GROUP BY symbol 
                    ORDER BY latest DESC 
                    LIMIT 5
                """)
                latest_data = cursor.fetchall()
                
                # Get total records
                cursor.execute("SELECT COUNT(*) FROM price_data")
                total_records = cursor.fetchone()[0]
                
                conn.close()
                
                print(f"✅ Database: {symbol_count} symbols, {total_records:,} total records")
                
                if latest_data:
                    print("   📈 Latest data per symbol:")
                    for symbol, latest, records in latest_data:
                        latest_dt = pd.to_datetime(latest)
                        hours_old = (datetime.now() - latest_dt).total_seconds() / 3600
                        
                        if hours_old < 2:
                            age_status = "🟢 Fresh"
                            status['data_fresh'] = True
                        elif hours_old < 24:
                            age_status = f"🟡 {hours_old:.0f}h old"
                        else:
                            age_status = f"🔴 {hours_old/24:.0f}d old"
                        
                        print(f"      {symbol}: {records:,} records, {latest} ({age_status})")
                
                status['database'] = True
                
            except Exception as e:
                print(f"❌ Database Error: {e}")
        else:
            print("❌ Database: Not found")
        
        # Check ML Models
        if os.path.exists('ml_models'):
            try:
                model_files = [f for f in os.listdir('ml_models') if f.endswith('.joblib')]
                gru_files = [f for f in os.listdir('ml_models') if f.endswith('.h5')]
                
                if model_files or gru_files:
                    print(f"✅ ML Models: {len(model_files)} ensemble models, {len(gru_files)} GRU models")
                    status['ml_models'] = True
                else:
                    print("⚠️ ML Models: Directory exists but no models found")
            except Exception as e:
                print(f"❌ ML Models: Error - {e}")
        else:
            print("⚠️ ML Models: Not trained yet")
        
        # Check configuration
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                print("✅ Configuration: Found and valid")
                print(f"   🔧 Data collection: {config.get('data_collection', {}).get('interval_minutes', 'N/A')} min intervals")
                print(f"   🔍 Signal analysis: {config.get('signal_analysis', {}).get('interval_minutes', 'N/A')} min intervals")
                status['config'] = True
            except Exception as e:
                print(f"❌ Configuration: Invalid JSON - {e}")
        else:
            print("⚠️ Configuration: Not found (will use defaults)")
        
        # Check automation status
        print(f"\n🤖 AUTOMATION STATUS:")
        
        if os.path.exists('logs/start_time.txt'):
            try:
                with open('logs/start_time.txt', 'r') as f:
                    start_time_str = f.read().strip()
                    start_time = datetime.fromisoformat(start_time_str)
                    uptime = datetime.now() - start_time
                    
                    days = uptime.days
                    hours, remainder = divmod(uptime.seconds, 3600)
                    minutes, _ = divmod(remainder, 60)
                    
                    print(f"🟢 Status: Running since {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"⏰ Uptime: {days}d {hours}h {minutes}m")
            except Exception as e:
                print(f"🔴 Status: Error reading start time - {e}")
        else:
            print("🔴 Status: Not running")
        
        # Overall system health
        print(f"\n🏥 SYSTEM HEALTH:")
        health_score = sum([
            status['database'] * 30,
            status['config'] * 20,
            status['ml_models'] * 30,
            status['data_fresh'] * 20
        ])
        
        if health_score >= 90:
            health_status = "🟢 EXCELLENT"
        elif health_score >= 70:
            health_status = "🟡 GOOD"
        elif health_score >= 50:
            health_status = "🟠 FAIR"
        else:
            health_status = "🔴 POOR"
        
        print(f"   Overall Health: {health_status} ({health_score}/100)")
        
        if not status['ml_models']:
            print(f"\n💡 TIP: Train ML models (Step 2) to enable predictions!")
        
        return status

    def display_menu(self):
        """Display the main menu"""
        print("\n🎯 CONTROL CENTER MENU")
        print("-" * 35)
        print("🛠️  MANUAL 4-STEP WORKFLOW:")
        print("   1. 📈 Step 1: Collect Market Data (collect_data.py)")
        print("   2. 🧠 Step 2: Train ML Models (train_models.py)")
        print("   3. 🔮 Step 3: Generate Predictions (generate_predictions.py)")
        print("   4. 🤖 Step 4: Run Autonomous Agent (run_agent.py)")
        print("")
        print("🤖 AUTOMATION & SCHEDULING:")
        print("   5. 🚀 Start 24/7 Scheduler (Background)")
        print("   6. 🛑 Stop Scheduler")
        print("   7. 📊 Scheduler Status")
        print("")
        print("⚙️ SYSTEM MANAGEMENT:")
        print("   8. 🔧 Configuration")
        print("   9. 📋 View Logs")
        print("   10. 🧹 System Cleanup")
        print("   11. 🔄 Quick System Test")
        print("")
        print("❓ HELP & EXIT:")
        print("   H. ❓ Help & Documentation")
        print("   0. 🚪 Exit")
        print("-" * 35)

    def collect_data(self):
        """Run data collection with unified ML collector"""
        print("\n📈 STEP 1: COLLECT MARKET DATA")
        print("-" * 40)
        
        print("Choose collection mode:")
        print("1. 🔄 Quick Update (recent data)")
        print("2. 🔥 Full Collection (comprehensive ML data)")
        print("3. 📊 Database Status")
        print("4. ⚙️ Custom Collection")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            print("🔄 Running quick data update...")
            self._run_collector([])
            
        elif choice == "2":
            print("🔥 Starting full comprehensive collection...")
            print("💡 This will collect extensive historical data for ML training")
            print("⏱️  Estimated time: 10-30 minutes")
            confirm = input("Continue? (y/n): ").lower()
            if confirm == 'y':
                self._run_collector([])
            else:
                print("❌ Collection cancelled")
                
        elif choice == "3":
            print("📊 Showing database status...")
            self._run_collector(['--status'])
            
        elif choice == "4":
            self._custom_collection()
            
        else:
            print("❌ Invalid choice")

    def _run_collector(self, args):
        """Run the unified ML collector"""
        cmd = [sys.executable, 'collect_data.py'] + args
        
        try:
            print(f"⚙️ Running: {' '.join(cmd)}")
            print("⏳ Please wait...")
            
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                encoding='utf-8',    
                errors='replace'    
            )
            
            for line in iter(process.stdout.readline, ''):
                print(f"   {line.rstrip()}")
            
            process.wait()
            
            if process.returncode == 0:
                print("✅ Data collection completed successfully!")
            else:
                print("❌ Data collection completed with errors")
                
        except Exception as e:
            print(f"❌ Error running data collection: {e}")

    def _custom_collection(self):
        """Custom data collection"""
        print("\n⚙️ CUSTOM DATA COLLECTION")
        print("-" * 35)
        
        default_symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "DOT/USDT"]
        
        print(f"Available symbols: {', '.join(default_symbols)}")
        symbols_input = input(f"Enter symbols (comma-separated) or press Enter for all: ").strip()
        if symbols_input:
            symbols = [s.strip() for s in symbols_input.split(',')]
            args = ['--symbols'] + symbols
        else:
            args = []
        
        all_timeframes = ["5m", "15m", "1h", "4h", "1d"]
        print(f"Available timeframes: {', '.join(all_timeframes)}")
        
        timeframes_input = input("Enter timeframes (comma-separated) or press Enter for all: ").strip()
        if timeframes_input:
            timeframes = [t.strip() for t in timeframes_input.split(',')]
            args.extend(['--timeframes'] + timeframes)
        
        print(f"\n📋 Custom collection starting...")
        self._run_collector(args)

    def generate_predictions(self):
        """Step 3: Generate Predictions"""
        print("\n🔮 STEP 3: GENERATE PREDICTIONS")
        print("-" * 40)
        
        if not os.path.exists(self.db_path):
            print("❌ No database found! Please collect data first (Step 1).")
            return
            
        print("Generating predictions from trained ML models...")
        self._run_script('generate_predictions.py')

    def _run_script(self, script_name, args=[]):
        """Helper to run a python script"""
        cmd = [sys.executable, script_name] + args
        
        try:
            print(f"⚙️ Running: {script_name}")
            print("⏳ Please wait...")
            
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                encoding='utf-8',   
                errors='replace'  
            )
            
            for line in iter(process.stdout.readline, ''):
                print(f"   {line.rstrip()}")
            
            process.wait()
            
            if process.returncode == 0:
                print(f"✅ {script_name} completed successfully!")
            else:
                print(f"❌ {script_name} completed with errors")
                
        except Exception as e:
            print(f"❌ Error running {script_name}: {e}")

    def run_agent_manual(self):
        """Step 4: Run Autonomous Agent"""
        print("\n🤖 STEP 4: RUN AUTONOMOUS AGENT")
        print("-" * 40)
        
        print("Running autonomous agent analysis...")
        self._run_script('run_agent.py')

    def train_ml_models(self):
        """Train ML models"""
        print("\n🧠 STEP 2: ML MODEL TRAINING")
        print("-" * 40)
        
        if not os.path.exists(self.db_path):
            print("❌ No database found! Please collect data first (Step 1).")
            return
        
        print("ML Training Options:")
        print("1. 🚀 Train All Models (All symbols + timeframes)")
        print("2. ⚡ Quick Training (1h and 4h only)")
        print("3. 🎯 Custom Training")
        print("4. 📊 View Model Status")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            print("\n🚀 TRAINING ALL MODELS")
            print("=" * 50)
            print("This will train:")
            print("• 5 symbols × 5 timeframes = 25 configurations")
            print("• ~6 models per configuration")
            print("• Total: ~150 models")
            print("\n⏱️  Estimated time: 1-2 hours")
            
            confirm = input("\nStart training? (y/n): ").lower()
            if confirm == 'y':
                self._run_ml_training([])
            else:
                print("❌ Training cancelled")
                
        elif choice == "2":
            print("\n⚡ QUICK TRAINING (1h and 4h)")
            print("=" * 50)
            print("This will train:")
            print("• 5 symbols × 2 timeframes = 10 configurations")
            print("• Total: ~60 models")
            print("\n⏱️  Estimated time: 20-40 minutes")
            
            confirm = input("\nStart training? (y/n): ").lower()
            if confirm == 'y':
                # Just run the existing train_models.py
                self._run_ml_training_direct('train_models.py')
            else:
                print("❌ Training cancelled")
                
        elif choice == "3":
            self._custom_ml_training()
            
        elif choice == "4":
            self.view_model_status()
            
        else:
            print("❌ Invalid choice")

    def _run_ml_training(self, args):
        """Run ML training"""
        cmd = [sys.executable, 'train_models.py'] + args
        
        try:
            print(f"⚙️ Starting ML training...")
            print("⏳ This will take a while. You can minimize this window.")
            print()
            
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                encoding='utf-8',
                errors='replace'
            )
            
            for line in iter(process.stdout.readline, ''):
                print(line.rstrip())
            
            process.wait()
            
            if process.returncode == 0:
                print("\n✅ ML training completed successfully!")
                print("💡 Run 'View Model Status' (option 4) to see trained models")
            else:
                print("\n❌ ML training completed with errors")
                
        except Exception as e:
            print(f"❌ Error running ML training: {e}")

    def _run_ml_training_direct(self, script):
        """Run ML training script directly"""
        cmd = [sys.executable, script]
        
        try:
            print(f"⚙️ Starting ML training...")
            print("⏳ This will take a while. You can minimize this window.")
            print()
            
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                encoding='utf-8',
                errors='replace'
            )
            
            for line in iter(process.stdout.readline, ''):
                print(line.rstrip())
            
            process.wait()
            
            if process.returncode == 0:
                print("\n✅ ML training completed successfully!")
            else:
                print("\n❌ ML training completed with errors")
                
        except Exception as e:
            print(f"❌ Error running ML training: {e}")

    def _custom_ml_training(self):
        """Custom ML training"""
        print("\n🎯 CUSTOM ML TRAINING")
        print("-" * 30)
        
        # This would require a custom training script
        # For now, redirect to full training
        print("💡 Custom training requires manual script modification")
        print("📝 Edit train_all_timeframes.py to customize symbols/timeframes")
        input("\nPress Enter to continue...")



    def view_model_status(self):
        """View ML model status"""
        print("\n📊 ML MODEL STATUS")
        print("-" * 40)
        
        if not os.path.exists('ml_models'):
            print("❌ ml_models directory not found")
            print("💡 Train models first using Step 2")
            return
        
        try:
            model_files = [f for f in os.listdir('ml_models') if f.endswith('.joblib')]
            gru_files = [f for f in os.listdir('ml_models') if f.endswith('.h5')]
            
            print(f"\n📈 MODEL INVENTORY:")
            print(f"   Ensemble Models (.joblib): {len(model_files)}")
            print(f"   GRU Models (.h5): {len(gru_files)}")
            print(f"   Total: {len(model_files) + len(gru_files)} models")
            
            if model_files:
                print(f"\n🔍 MODEL BREAKDOWN:")
                
                # Organize by symbol and timeframe
                symbol_tf_models = {}
                
                for model_file in model_files:
                    if '_price_' in model_file or '_direction_' in model_file:
                        parts = model_file.split('_')
                        if len(parts) >= 4:
                            symbol = parts[0] + '/' + parts[1]
                            timeframe = parts[2]
                            key = f"{symbol}_{timeframe}"
                            
                            if key not in symbol_tf_models:
                                symbol_tf_models[key] = {'price': 0, 'direction': 0}
                            
                            if 'price' in model_file:
                                symbol_tf_models[key]['price'] += 1
                            elif 'direction' in model_file:
                                symbol_tf_models[key]['direction'] += 1
                
                # Display organized results
                for key in sorted(symbol_tf_models.keys()):
                    symbol_tf = key
                    counts = symbol_tf_models[key]
                    
                    status = "✅" if counts['price'] >= 3 and counts['direction'] >= 3 else "⚠️"
                    print(f"   {status} {symbol_tf}: {counts['price']} price, {counts['direction']} direction models")
            
            # Show what's needed for complete coverage
            expected_symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'DOT/USDT']
            expected_timeframes = ['5m', '15m', '1h', '4h', '1d']
            
            print(f"\n📊 COVERAGE STATUS:")
            total_expected = len(expected_symbols) * len(expected_timeframes) * 6  # 6 models per config
            coverage_pct = (len(model_files) / total_expected) * 100 if total_expected > 0 else 0
            
            print(f"   Expected: {total_expected} models (full coverage)")
            print(f"   Current: {len(model_files)} models")
            print(f"   Coverage: {coverage_pct:.1f}%")
            
            if coverage_pct < 100:
                print(f"\n💡 TIP: Train all timeframes for 100% coverage (Step 2)")
            else:
                print(f"\n🎉 Complete model coverage achieved!")
                
        except Exception as e:
            print(f"❌ Error viewing model status: {e}")



    def start_automation(self):
        """Start the 24/7 automation system"""
        print("\n🤖 24/7 AUTOMATION SYSTEM")
        print("-" * 40)
        
        # Check if already running
        if os.path.exists('logs/automation.pid'):
            print("⚠️ Automation system appears to be already running (PID file exists)!")
            try:
                with open('logs/automation.pid', 'r') as f:
                    pid = f.read().strip()
                print(f"   PID: {pid}")
            except:
                pass
            
            print("\nOptions:")
            print("1. Restart (Stop & Start)")
            print("2. Cancel")
            
            choice = input("Select option (1-2): ").strip()
            
            if choice == "1":
                self.stop_automation()
                time.sleep(2)
            else:
                return

        print(f"\n🚀 Starting 24/7 automation system (APScheduler)...")
        print(f"💡 The system will run in the background")
        
        try:
            # Create logs dir
            os.makedirs('logs', exist_ok=True)
            
            # Helper to run scheduler
            scheduler_script = os.path.join('crypto_ai', 'automation', 'scheduler.py')
            
            # Start process detached
            if sys.platform == 'win32':
                process = subprocess.Popen(
                    [sys.executable, scheduler_script],
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            else:
                process = subprocess.Popen(
                    [sys.executable, scheduler_script],
                    stdout=open('logs/scheduler_stdout.log', 'w'),
                    stderr=open('logs/scheduler_stderr.log', 'w'),
                    start_new_session=True
                )
            
            # Save PID
            with open('logs/automation.pid', 'w') as f:
                f.write(str(process.pid))
            
            # Save start time
            with open('logs/start_time.txt', 'w') as f:
                f.write(datetime.now().isoformat())
                
            print(f"✅ Automation started! PID: {process.pid}")
            print("💡 Check logs/scheduler.log for activity")
            
        except Exception as e:
            print(f"❌ Failed to start automation: {e}")

    def stop_automation(self):
        """Stop the automation system"""
        print("\n🛑 STOPPING AUTOMATION")
        print("-" * 30)
        
        if not os.path.exists('logs/automation.pid'):
            print("ℹ️ Automation doesn't appear to be running (no PID file)")
            # Clean up stale start_time if exists
            if os.path.exists('logs/start_time.txt'):
               os.remove('logs/start_time.txt')
            return
        
        try:
            with open('logs/automation.pid', 'r') as f:
                pid = int(f.read().strip())
            
            print(f"Stopping process {pid}...")
            
            try:
                # Portable process killing
                import signal
                os.kill(pid, signal.SIGTERM)
                print("✅ Process terminated")
            except ProcessLookupError:
                print("⚠️ Process not found (already stopped?)")
            except Exception as e:
                print(f"❌ Error killing process: {e}")
                
            # Cleanup files
            if os.path.exists('logs/start_time.txt'):
                os.remove('logs/start_time.txt')
            
            if os.path.exists('logs/automation.pid'):
                os.remove('logs/automation.pid')
                
            print("✅ Stopped automation system")
                
        except Exception as e:
            print(f"❌ Error stopping automation: {e}")

    def automation_status(self):
        """Show detailed automation status"""
        print("\n📊 AUTOMATION STATUS")
        print("-" * 30)
        
        if os.path.exists('logs/start_time.txt'):
            try:
                with open('logs/start_time.txt', 'r') as f:
                    start_time_str = f.read().strip()
                    start_time = datetime.fromisoformat(start_time_str)
                    
                print(f"🟢 Status: RUNNING")
                print(f"⏰ Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                uptime = datetime.now() - start_time
                days = uptime.days
                hours, remainder = divmod(uptime.seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                
                print(f"📈 Uptime: {days}d {hours}h {minutes}m {seconds}s")
                
            except Exception as e:
                print(f"🔴 Status: ERROR - {e}")
        else:
            print(f"🔴 Status: NOT RUNNING")

    def configure_system(self):
        """Configure system settings"""
        print("\n⚙️ SYSTEM CONFIGURATION")
        print("-" * 35)
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                
                print("📄 CURRENT CONFIGURATION:")
                print("-" * 30)
                
                data_config = config.get('data_collection', {})
                print(f"📊 DATA COLLECTION:")
                print(f"   ✅ Enabled: {data_config.get('enabled', True)}")
                print(f"   ⏰ Interval: {data_config.get('interval_minutes', 60)} minutes")
                print(f"   📈 Symbols: {', '.join(data_config.get('symbols', []))}")
                
                signal_config = config.get('signal_analysis', {})
                print(f"\n🔍 SIGNAL ANALYSIS:")
                print(f"   ✅ Enabled: {signal_config.get('enabled', True)}")
                print(f"   ⏰ Interval: {signal_config.get('interval_minutes', 15)} minutes")
                
                print(f"\n⚙️ OPTIONS:")
                print("1. 📝 Edit Configuration File")
                print("2. 🔄 Reset to Defaults")
                print("0. ↩️  Back")
                
                choice = input("\nSelect (0-2): ").strip()
                
                if choice == "1":
                    self._edit_config_file()
                elif choice == "2":
                    self._reset_config()
                    
            except Exception as e:
                print(f"❌ Error reading configuration: {e}")
        else:
            print("📄 No configuration file found")
            choice = input("Create default configuration? (y/n): ").lower()
            if choice == 'y':
                self.create_default_config()

    def _edit_config_file(self):
        """Open configuration file for editing"""
        print(f"\n📝 Opening configuration file...")
        
        if os.name == 'nt':
            try:
                os.system(f'notepad {self.config_path}')
            except:
                print(f"💡 Please edit {self.config_path} manually")
        else:
            editors = ['nano', 'vim', 'vi']
            for editor in editors:
                try:
                    subprocess.run([editor, self.config_path])
                    break
                except FileNotFoundError:
                    continue

    def _reset_config(self):
        """Reset configuration to defaults"""
        print(f"\n🔄 RESET CONFIGURATION")
        print("-" * 25)
        
        confirm = input("Reset to defaults? (y/n): ").lower()
        
        if confirm == 'y':
            try:
                if os.path.exists(self.config_path):
                    backup = f"{self.config_path}.backup.{int(time.time())}"
                    os.rename(self.config_path, backup)
                
                self.create_default_config()
                print("✅ Configuration reset")
                
            except Exception as e:
                print(f"❌ Error: {e}")

    def create_default_config(self):
        """Create default configuration file"""
        default_config = {
            "data_collection": {
                "enabled": True,
                "interval_minutes": 60,
                "symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "DOT/USDT"],
                "timeframes": ["5m", "15m", "1h", "4h", "1d"]
            },
            "signal_analysis": {
                "enabled": True,
                "interval_minutes": 15,
                "use_ml": True
            },
            "alerts": {
                "enabled": True,
                "desktop": {"enabled": True},
                "log_file": {"enabled": True}
            },
            "system": {
                "database_path": "data/ml_crypto_data.db"
            }
        }
        
        try:
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
            print("✅ Default configuration created!")
        except Exception as e:
            print(f"❌ Error: {e}")

    def view_logs(self):
        """View system logs"""
        print("\n📋 SYSTEM LOGS")
        print("-" * 25)
        
        log_files = {
            '1': ('unified_ml_collector.log', 'Data Collection'),
            '2': ('unified_analyzer.log', 'Signal Analysis'),
            '3': ('automation.log', 'Automation')
        }
        
        print("📄 Available logs:")
        for key, (file_path, desc) in log_files.items():
            status = "✅" if os.path.exists(file_path) else "❌"
            print(f"   {key}. {status} {desc}")
        
        choice = input("\nSelect log (1-3): ").strip()
        
        if choice in log_files:
            file_path, desc = log_files[choice]
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                    
                    print(f"\n📄 {desc} - Last 20 lines:")
                    print("-" * 50)
                    for line in lines[-20:]:
                        print(line.rstrip())
                except Exception as e:
                    print(f"❌ Error: {e}")
            else:
                print(f"❌ Log file not found: {file_path}")

    def cleanup_system(self):
        """Clean up system files"""
        print("\n🧹 SYSTEM CLEANUP")
        print("-" * 25)
        
        print("Cleanup options:")
        print("1. 🗑️ Clear Old Logs")
        print("2. 📊 Show Disk Usage")
        
        choice = input("\nSelect (1-2): ").strip()
        
        if choice == "1":
            print("🗑️ Clearing old logs...")
            log_files = ['unified_ml_collector.log', 'unified_analyzer.log']
            
            for log_file in log_files:
                if os.path.exists(log_file):
                    size = os.path.getsize(log_file) / 1024
                    if size > 1024:  # > 1MB
                        backup = f"{log_file}.backup.{int(time.time())}"
                        os.rename(log_file, backup)
                        print(f"✅ Cleared {log_file} ({size:.1f} KB)")
            
        elif choice == "2":
            print("\n📊 DISK USAGE:")
            
            paths = [
                ('Database', self.db_path),
                ('ML Models', 'ml_models'),
                ('Logs', '.')
            ]
            
            for name, path in paths:
                if os.path.exists(path):
                    if os.path.isfile(path):
                        size = os.path.getsize(path) / (1024 * 1024)
                        print(f"   {name}: {size:.1f} MB")
                    else:
                        total = 0
                        for root, dirs, files in os.walk(path):
                            for f in files:
                                fp = os.path.join(root, f)
                                total += os.path.getsize(fp)
                        print(f"   {name}: {total/(1024*1024):.1f} MB")



    def quick_system_test(self):
        """Run quick system test"""
        print("\n🔄 QUICK SYSTEM TEST")
        print("-" * 30)
        
        tests = [
            ("Database", self._test_database),
            ("ML Models", self._test_ml_models),
            ("Collector Script", self._test_collector),
            ("Analyzer Script", self._test_analyzer)
        ]
        
        passed = 0
        
        for name, test_func in tests:
            print(f"\n🧪 Testing {name}...")
            try:
                if test_func():
                    print(f"   ✅ {name}: PASSED")
                    passed += 1
                else:
                    print(f"   ❌ {name}: FAILED")
            except Exception as e:
                print(f"   ❌ {name}: ERROR - {e}")
        
        print(f"\n📊 Results: {passed}/{len(tests)} passed")
        
        if passed == len(tests):
            print("🟢 System ready!")
        else:
            print("🔴 Some issues detected")

    def _test_database(self):
        """Test database"""
        if not os.path.exists(self.db_path):
            return False
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM price_data")
            cursor.fetchone()
            conn.close()
            return True
        except:
            return False

    def _test_ml_models(self):
        """Test ML models"""
        if not os.path.exists('ml_models'):
            return False
        files = os.listdir('ml_models')
        return len([f for f in files if f.endswith('.joblib')]) > 0

    def _test_collector(self):
        """Test collector script"""
        return os.path.exists('collect_data.py')

    def _test_analyzer(self):
        """Test analyzer script"""
        return os.path.exists('analyze_signals.py')

    def show_help(self):
        """Show help"""
        print("\n❓ CRYPTO TRADING SYSTEM HELP - ML EDITION")
        print("=" * 50)
        print("""
🚀 WELCOME TO YOUR ML-POWERED TRADING SYSTEM!

📊 MAIN FEATURES:
   • ML predictions with LightGBM, XGBoost, CatBoost
   • Technical analysis across 5 timeframes
   • Multi-symbol tracking (BTC, ETH, BNB, ADA, DOT)
   • Autonomous agent decision making

🧠 4-STEP WORKFLOW:
   1. 📈 Collect Data (Option 1)
      - Runs collect_data.py
      - Gathers market history for training
   
   2. 🧠 Train Models (Option 2)
      - Runs train_models.py
      - Trains ensemble models on data covers
   
   3. 🔮 Generate Predictions (Option 3)
      - Runs generate_predictions.py
      - Creates ML predictions for all symbols/timeframes
   
   4. 🤖 Run Agent (Option 4)
      - Runs run_agent.py
      - Analyzes signals and recommends trades

🤖 AUTOMATION:
   • Start 24/7 Scheduler (Option 5) covers steps 1-4 automatically.
   • View status with Option 7.

💡 TIPS:
   • More data = better models (collect 6+ months)
   • Train all timeframes for best coverage
   • Check "View Model Status" (Step 2, Option 4) regularly

For detailed documentation, see the README files.
        """)

    def run(self):
        """Main control center loop"""
        try:
            self.display_banner()
            self.check_system_status()
            
            while True:
                self.display_menu()
                choice = input("\nSelect option: ").strip().upper()
                
                if choice == "1":
                    self.collect_data()
                elif choice == "2":
                    self.train_ml_models()
                elif choice == "3":
                    self.generate_predictions()
                elif choice == "4":
                    self.run_agent_manual()
                elif choice == "5":
                    self.start_automation()
                elif choice == "6":
                    self.stop_automation()
                elif choice == "7":
                    self.automation_status()
                elif choice == "8":
                    self.configure_system()
                elif choice == "9":
                    self.view_logs()
                elif choice == "10":
                    self.cleanup_system()
                elif choice == "11":
                    self.quick_system_test()
                elif choice == "H":
                    self.show_help()
                elif choice == "0":
                    print("\n👋 Thank you for using Crypto Trading Control Center!")
                    print("🚀 Happy trading!")
                    break
                else:
                    print("❌ Invalid choice. Please select a valid option.")
                
                input("\n⏸️  Press Enter to continue...")
                
        except KeyboardInterrupt:
            print("\n\n🛑 Control Center interrupted by user")
            print("👋 Goodbye!")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")


def main():
    """Main function"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        if not os.path.exists('collect_data.py'):
            print("⚠️ Warning: Core system files not found")
            print("💡 Make sure you're in the right directory")
            print()
        
        control_center = CryptoControlCenter()
        control_center.run()
        
    except KeyboardInterrupt:
        print("\n\n👋 Control Center closed")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()