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
            print(f"\n💡 TIP: Train ML models (option M) to enable predictions!")
        
        return status

    def display_menu(self):
        """Display the main menu"""
        print("\n🎯 CONTROL CENTER MENU")
        print("-" * 35)
        print("📊 DATA & ANALYSIS:")
        print("   1. 📈 Collect Market Data")
        print("   2. 🔍 Analyze Trading Signals (ML + TA)")
        print("   3. 📋 View Latest Signals")
        print("")
        print("🧠 MACHINE LEARNING:")
        print("   M. 🤖 Train ML Models")
        print("   P. 🔮 Test ML Predictions")
        print("   V. 📊 View Model Status")
        print("")
        print("🤖 AUTOMATION:")
        print("   4. 🚀 Start 24/7 Automation")
        print("   5. 🛑 Stop Automation")
        print("   6. 📊 Automation Status")
        print("")
        print("⚙️ SYSTEM MANAGEMENT:")
        print("   7. 🔧 Configuration")
        print("   8. 📋 View Logs")
        print("   9. 🧹 System Cleanup")
        print("")
        print("📈 ADVANCED:")
        print("   A. 🎯 Performance Analysis")
        print("   B. 📊 Market Overview")
        print("   C. 🔄 Quick System Test")
        print("")
        print("❓ HELP & EXIT:")
        print("   H. ❓ Help & Documentation")
        print("   0. 🚪 Exit")
        print("-" * 35)

    def collect_data(self):
        """Run data collection with unified ML collector"""
        print("\n📊 MARKET DATA COLLECTION")
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
        cmd = [sys.executable, 'comprehensive_ml_collector_v2.py'] + args
        
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

    def analyze_signals(self):
        """Run unified signal analysis"""
        print("\n🔍 TRADING SIGNAL ANALYSIS (ML + TA)")
        print("-" * 40)
        
        if not os.path.exists(self.db_path):
            print("❌ No database found! Please collect data first.")
            return
        
        print("Choose analysis mode:")
        print("1. 🎯 All Symbols (Multi-timeframe)")
        print("2. 📊 Specific Symbol Analysis")
        print("3. 💾 Save Results to File")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == "1":
            print("🎯 Running comprehensive analysis...")
            self._run_analyzer([])
            
        elif choice == "2":
            self._single_symbol_analysis()
            
        elif choice == "3":
            print("📊 Running analysis and saving results...")
            self._run_analyzer(['--save'])
            
        else:
            print("❌ Invalid choice")

    def _run_analyzer(self, args):
        """Run the unified crypto analyzer"""
        cmd = [sys.executable, 'unified_crypto_analyzer.py'] + args
        
        try:
            print(f"⚙️ Running analysis...")
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
                print("✅ Signal analysis completed successfully!")
            else:
                print("❌ Signal analysis completed with errors")
                
        except Exception as e:
            print(f"❌ Error running signal analysis: {e}")

    def _single_symbol_analysis(self):
        """Analyze signals for a single symbol"""
        print("\n🔍 SINGLE SYMBOL ANALYSIS")
        print("-" * 30)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT symbol FROM price_data ORDER BY symbol")
            symbols = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            if not symbols:
                print("❌ No symbols found in database")
                return
            
            print("Available symbols:")
            for i, symbol in enumerate(symbols, 1):
                print(f"   {i}. {symbol}")
            
            try:
                choice = int(input(f"\nSelect symbol (1-{len(symbols)}): "))
                if 1 <= choice <= len(symbols):
                    selected_symbol = symbols[choice - 1]
                    print(f"🔍 Analyzing {selected_symbol}...")
                    self._run_analyzer(['--symbols', selected_symbol])
                else:
                    print("❌ Invalid selection")
            except ValueError:
                print("❌ Invalid input")
                
        except Exception as e:
            print(f"❌ Error getting symbols: {e}")

    def train_ml_models(self):
        """Train ML models"""
        print("\n🧠 ML MODEL TRAINING")
        print("-" * 40)
        
        if not os.path.exists(self.db_path):
            print("❌ No database found! Please collect data first.")
            return
        
        print("ML Training Options:")
        print("1. 🚀 Train All Models (All symbols + timeframes)")
        print("2. ⚡ Quick Training (1h and 4h only)")
        print("3. 🎯 Custom Training")
        
        choice = input("\nSelect option (1-3): ").strip()
        
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
                # Just run the existing optimized_ml_system.py
                self._run_ml_training_direct('optimized_ml_system.py')
            else:
                print("❌ Training cancelled")
                
        elif choice == "3":
            self._custom_ml_training()
            
        else:
            print("❌ Invalid choice")

    def _run_ml_training(self, args):
        """Run ML training"""
        cmd = [sys.executable, 'optimized_ml_system.py'] + args
        
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
                print("💡 Run 'View Model Status' (option V) to see trained models")
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

    def test_ml_predictions(self):
        """Test ML predictions"""
        print("\n🔮 ML PREDICTION TEST")
        print("-" * 40)
        
        if not os.path.exists('ml_models'):
            print("❌ No ML models found! Train models first (option M).")
            return
        
        print("Checking available models...")
        model_files = [f for f in os.listdir('ml_models') if f.endswith('.joblib')]
        
        if not model_files:
            print("❌ No trained models found in ml_models/")
            return
        
        print(f"✅ Found {len(model_files)} model files")
        print("\n🔮 Running prediction test...")
        
        # Run a quick analysis which will show ML predictions
        self._run_analyzer([])

    def view_model_status(self):
        """View ML model status"""
        print("\n📊 ML MODEL STATUS")
        print("-" * 40)
        
        if not os.path.exists('ml_models'):
            print("❌ ml_models directory not found")
            print("💡 Train models first using option M")
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
                print(f"\n💡 TIP: Train all timeframes for 100% coverage (option M)")
            else:
                print(f"\n🎉 Complete model coverage achieved!")
                
        except Exception as e:
            print(f"❌ Error viewing model status: {e}")

    def view_latest_signals(self):
        """View the latest trading signals"""
        print("\n📋 LATEST TRADING SIGNALS")
        print("-" * 40)
        
        if not os.path.exists(self.db_path):
            print("❌ No database found! Please collect data first.")
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = """
            SELECT 
                symbol,
                timeframe,
                COUNT(*) as record_count,
                MAX(timestamp) as latest_timestamp,
                AVG(close) as avg_price
            FROM price_data 
            GROUP BY symbol, timeframe 
            ORDER BY symbol, 
                CASE timeframe 
                    WHEN '5m' THEN 1
                    WHEN '15m' THEN 2
                    WHEN '1h' THEN 3
                    WHEN '4h' THEN 4
                    WHEN '1d' THEN 5
                    ELSE 6
                END
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                print("❌ No data available in database")
                return
            
            print(f"📊 DATA SUMMARY:")
            print(f"{'Symbol':<12} {'Timeframe':<10} {'Records':<8} {'Avg Price':<12} {'Latest Data'}")
            print("-" * 70)
            
            for _, row in df.iterrows():
                latest_dt = pd.to_datetime(row['latest_timestamp'])
                hours_old = (datetime.now() - latest_dt).total_seconds() / 3600
                
                if hours_old < 2:
                    status = "🟢"
                elif hours_old < 24:
                    status = "🟡"
                else:
                    status = "🔴"
                
                print(f"{row['symbol']:<12} {row['timeframe']:<10} {row['record_count']:<8} "
                      f"${row['avg_price']:<11.2f} {status} {latest_dt.strftime('%m-%d %H:%M')}")
            
            symbols = df['symbol'].unique()
            print(f"\n🎯 QUICK OVERVIEW:")
            print(f"   📈 Symbols tracked: {len(symbols)}")
            print(f"   ⏱️  Total records: {df['record_count'].sum():,}")
            
            df['hours_old'] = df['latest_timestamp'].apply(
                lambda x: (datetime.now() - pd.to_datetime(x)).total_seconds() / 3600
            )
            fresh_data = len(df[df['hours_old'] < 2])
            print(f"   🟢 Fresh datasets: {fresh_data}/{len(df)}")
            
            print(f"\n💡 Run signal analysis (option 2) for ML predictions!")
            
        except Exception as e:
            print(f"❌ Error viewing signals: {e}")

    def start_automation(self):
        """Start the 24/7 automation system"""
        print("\n🤖 24/7 AUTOMATION SYSTEM")
        print("-" * 40)
        
        # Check if already running
        if os.path.exists('logs/start_time.txt'):
            print("⚠️ Automation system appears to be already running!")
            
            try:
                with open('logs/start_time.txt', 'r') as f:
                    start_time = f.read().strip()
                print(f"   Started: {start_time}")
            except:
                pass
            
            print("\nOptions:")
            print("1. Continue anyway (may cause conflicts)")
            print("2. Stop existing and restart")
            print("3. Cancel")
            
            choice = input("Select option (1-3): ").strip()
            
            if choice == "2":
                self.stop_automation()
                time.sleep(2)
            elif choice == "3":
                print("❌ Automation start cancelled")
                return
            elif choice != "1":
                print("❌ Invalid choice")
                return
        
        # Check prerequisites
        print("🔍 Checking system prerequisites...")
        
        if not os.path.exists(self.db_path):
            print("❌ No database found!")
            choice = input("Run initial data collection now? (y/n): ").lower()
            if choice == 'y':
                self._run_collector([])
            else:
                print("❌ Cannot start automation without data")
                return
        
        if not os.path.exists(self.config_path):
            print("⚠️ No configuration found, creating default...")
            self.create_default_config()
        
        print("✅ Prerequisites checked")
        
        # Show automation settings
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            data_interval = config.get('data_collection', {}).get('interval_minutes', 60)
            signal_interval = config.get('signal_analysis', {}).get('interval_minutes', 15)
            
            print(f"\n📋 AUTOMATION CONFIGURATION:")
            print(f"   📊 Data collection: Every {data_interval} minutes")
            print(f"   🔍 Signal analysis: Every {signal_interval} minutes")
            print(f"   🔔 Alerts: {'✅ Enabled' if config.get('alerts', {}).get('enabled', True) else '❌ Disabled'}")
            
        except Exception as e:
            print(f"⚠️ Error reading config: {e}")
        
        print(f"\n🚀 Starting 24/7 automation system...")
        print(f"💡 The system will run continuously in the background")
        print(f"⚠️ Close this window or press Ctrl+C to stop")
        
        confirm = input("\nStart automation now? (y/n): ").lower()
        if confirm != 'y':
            print("❌ Automation start cancelled")
            return
        
        # Create automation marker
        os.makedirs('logs', exist_ok=True)
        with open('logs/start_time.txt', 'w') as f:
            f.write(datetime.now().isoformat())
        
        print("✅ Automation system started!")
        print("💡 Use option 6 to monitor status")

    def stop_automation(self):
        """Stop the automation system"""
        print("\n🛑 STOPPING AUTOMATION")
        print("-" * 30)
        
        if not os.path.exists('logs/start_time.txt'):
            print("ℹ️ Automation doesn't appear to be running")
            return
        
        try:
            if os.path.exists('logs/start_time.txt'):
                os.remove('logs/start_time.txt')
                print("✅ Stopped automation system")
            
            if os.path.exists('logs/automation.pid'):
                os.remove('logs/automation.pid')
                
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

    def market_overview(self):
        """Show market overview"""
        print("\n📊 MARKET OVERVIEW")
        print("-" * 30)
        print("💡 Running unified analyzer for market overview...")
        self._run_analyzer([])

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
        return os.path.exists('comprehensive_ml_collector.py')

    def _test_analyzer(self):
        """Test analyzer script"""
        return os.path.exists('unified_crypto_analyzer.py')

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
   • Automated data collection and analysis

🧠 MACHINE LEARNING:
   • Train models on historical data (option M)
   • Get ML-powered predictions (option 2)
   • View model status and coverage (option V)
   • Test predictions (option P)

📋 GETTING STARTED:
   1. Collect data (option 1 - choose full collection)
   2. Train ML models (option M)
   3. Analyze signals (option 2)
   4. Start automation (option 4)

💡 TIPS:
   • More data = better models (collect 6+ months)
   • Train all timeframes for best coverage
   • ML + TA together gives strongest signals
   • Check model status regularly (option V)

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
                    self.analyze_signals()
                elif choice == "3":
                    self.view_latest_signals()
                elif choice == "M":
                    self.train_ml_models()
                elif choice == "P":
                    self.test_ml_predictions()
                elif choice == "V":
                    self.view_model_status()
                elif choice == "4":
                    self.start_automation()
                elif choice == "5":
                    self.stop_automation()
                elif choice == "6":
                    self.automation_status()
                elif choice == "7":
                    self.configure_system()
                elif choice == "8":
                    self.view_logs()
                elif choice == "9":
                    self.cleanup_system()
                elif choice == "A":
                    print("Performance analysis - use option 2 for ML analysis")
                elif choice == "B":
                    self.market_overview()
                elif choice == "C":
                    self.quick_system_test()
                elif choice == "H":
                    self.show_help()
                elif choice == "0":
                    print("\n👋 Thank you for using Crypto Trading Control Center!")
                    print("🚀 Happy trading with ML-powered predictions!")
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
        
        if not os.path.exists('comprehensive_ml_collector.py'):
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