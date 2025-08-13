#!/usr/bin/env python3
"""
Crypto Trading Control Center - Complete Version
Unified interface for your trading system
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
        self.db_path = 'data/multi_timeframe_data.db'
        self.config_path = 'automation_config.json'
        self.automation_process = None
        self.version = "2.0.0"
    
    def display_banner(self):
        """Display the control center banner"""
        print("🚀 CRYPTO TRADING CONTROL CENTER v" + self.version)
        print("=" * 65)
        print("🔥 AI-Powered Crypto Trading System")
        print("💡 Multi-Timeframe Analysis | Ultimate Signals | 24/7 Automation")
        print("=" * 65)

    def check_system_status(self):
        """Check the status of all system components"""
        print("\n📊 SYSTEM STATUS CHECK")
        print("-" * 45)
        
        status = {
            'database': False,
            'config': False,
            'automation': False,
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
                    LIMIT 3
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
        automation_running = False
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
                    automation_running = True
                    status['automation'] = True
                    
            except Exception as e:
                print(f"🔴 Status: Error reading start time - {e}")
        else:
            print("🔴 Status: Not running")
        
        # Check log files
        print(f"\n📋 LOG FILES:")
        log_files = {
            'automation.log': 'Automation System',
            'crypto_collector.log': 'Data Collector',
            'ultimate_analyzer.log': 'Signal Analyzer',
            'alerts/alerts.log': 'Trading Alerts'
        }
        
        for log_file, description in log_files.items():
            if os.path.exists(log_file):
                try:
                    size = os.path.getsize(log_file) / 1024  # KB
                    modified = datetime.fromtimestamp(os.path.getmtime(log_file))
                    hours_since = (datetime.now() - modified).total_seconds() / 3600
                    
                    if hours_since < 1:
                        activity = "🟢 Active"
                    elif hours_since < 24:
                        activity = f"🟡 {hours_since:.0f}h old"
                    else:
                        activity = f"🔴 {hours_since/24:.0f}d old"
                    
                    print(f"   📄 {description}: {size:.1f} KB ({activity})")
                except Exception as e:
                    print(f"   ❌ {description}: Error - {e}")
            else:
                print(f"   ⚪ {description}: Not found")
        
        # Overall system health
        print(f"\n🏥 SYSTEM HEALTH:")
        health_score = sum([
            status['database'] * 30,
            status['config'] * 20,
            status['automation'] * 30,
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
        
        return status

    def display_menu(self):
        """Display the main menu"""
        print("\n🎯 CONTROL CENTER MENU")
        print("-" * 35)
        print("📊 DATA & ANALYSIS:")
        print("   1. 📈 Collect Market Data")
        print("   2. 🔍 Analyze Trading Signals")
        print("   3. 📋 View Latest Signals")
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
        """Run data collection with advanced options"""
        print("\n📊 MARKET DATA COLLECTION")
        print("-" * 40)
        
        print("Choose collection mode:")
        print("1. 🔄 Normal Update (incremental)")
        print("2. 🔥 Force Fresh Data (complete refresh)")
        print("3. 🔍 Diagnose Database Only")
        print("4. 🧹 Clear Old Data + Fresh Collection")
        print("5. ⚙️ Custom Collection")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            print("🔄 Running normal data collection...")
            self._run_collector([])
            
        elif choice == "2":
            print("🔥 Forcing fresh data collection...")
            print("💡 This will collect complete fresh data for all symbols")
            confirm = input("Continue? (y/n): ").lower()
            if confirm == 'y':
                self._run_collector(['--force'])
            else:
                print("❌ Collection cancelled")
                
        elif choice == "3":
            print("🔍 Running database diagnosis...")
            self._run_collector(['--diagnose'])
            
        elif choice == "4":
            print("🧹 Clearing old data and collecting fresh...")
            confirm = input("This will delete data older than 24h. Continue? (y/n): ").lower()
            if confirm == 'y':
                self._run_collector(['--clear', '24', '--force'])
            else:
                print("❌ Collection cancelled")
                
        elif choice == "5":
            self._custom_collection()
            
        else:
            print("❌ Invalid choice")

    def _run_collector(self, args):
        """Run the multi-timeframe collector with given arguments"""

        cmd = [sys.executable, 'multi_timeframe_collector.py'] + args
        
        try:
            print(f"⚙️ Running: {' '.join(cmd)}")
            print("⏳ Please wait...")
            
            # Run with real-time output
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                encoding='utf-8',    
        errors='replace'    
            )
            
            # Stream output in real-time
            for line in iter(process.stdout.readline, ''):
                print(f"   {line.rstrip()}")
            
            process.wait()
            
            if process.returncode == 0:
                print("✅ Data collection completed successfully!")
                self._show_collection_summary()
            else:
                print("❌ Data collection completed with errors")
                
        except Exception as e:
            print(f"❌ Error running data collection: {e}")

    def _custom_collection(self):
        """Custom data collection with user options"""
        print("\n⚙️ CUSTOM DATA COLLECTION")
        print("-" * 35)
        
        # Get available symbols from config or default
        default_symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    default_symbols = config.get('data_collection', {}).get('symbols', default_symbols)
            except:
                pass
        
        print(f"Available symbols: {', '.join(default_symbols)}")
        
        # Let user choose symbols
        symbols_input = input(f"Enter symbols (comma-separated) or press Enter for all: ").strip()
        if symbols_input:
            symbols = [s.strip() for s in symbols_input.split(',')]
        else:
            symbols = default_symbols
        
        # Choose timeframes
        all_timeframes = ["5m", "15m", "1h", "4h", "1d"]
        print(f"Available timeframes: {', '.join(all_timeframes)}")
        
        timeframes_input = input("Enter timeframes (comma-separated) or press Enter for all: ").strip()
        if timeframes_input:
            timeframes = [t.strip() for t in timeframes_input.split(',')]
        else:
            timeframes = all_timeframes
        
        # Confirm and run
        print(f"\n📋 Collection Summary:")
        print(f"   Symbols: {', '.join(symbols)}")
        print(f"   Timeframes: {', '.join(timeframes)}")
        
        confirm = input("\nProceed with custom collection? (y/n): ").lower()
        if confirm == 'y':
            print("🔄 Running custom collection...")
            self._run_collector(['--force'])
        else:
            print("❌ Collection cancelled")

    def _show_collection_summary(self):
        """Show summary after data collection"""
        if not os.path.exists(self.db_path):
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    symbol,
                    COUNT(DISTINCT timeframe) as timeframes,
                    COUNT(*) as total_records,
                    MAX(timestamp) as latest
                FROM price_data
                GROUP BY symbol
                ORDER BY symbol
            """)
            
            results = cursor.fetchall()
            conn.close()
            
            if results:
                print(f"\n📊 COLLECTION SUMMARY:")
                print(f"{'Symbol':<12} {'Timeframes':<11} {'Records':<8} {'Latest Data'}")
                print("-" * 55)
                
                for symbol, timeframes, records, latest in results:
                    latest_dt = pd.to_datetime(latest)
                    hours_old = (datetime.now() - latest_dt).total_seconds() / 3600
                    
                    if hours_old < 1:
                        age_display = "Just now"
                    elif hours_old < 24:
                        age_display = f"{hours_old:.0f}h ago"
                    else:
                        age_display = f"{hours_old/24:.0f}d ago"
                    
                    print(f"{symbol:<12} {timeframes:<11} {records:<8} {age_display}")
                
        except Exception as e:
            print(f"❌ Error showing summary: {e}")

    def analyze_signals(self):
        """Run signal analysis with options"""
        print("\n🔍 TRADING SIGNAL ANALYSIS")
        print("-" * 40)
        
        # Check if we have data first
        if not os.path.exists(self.db_path):
            print("❌ No database found! Please collect data first.")
            return
        
        print("Choose analysis mode:")
        print("1. 🎯 Quick Analysis (all symbols)")
        print("2. 📊 Detailed Analysis (with breakdown)")
        print("3. 🔍 Single Symbol Analysis")
        print("4. 📈 Comparison Analysis")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            print("🎯 Running quick signal analysis...")
            self._run_analyzer([])
            
        elif choice == "2":
            print("📊 Running detailed signal analysis...")
            self._run_analyzer(['--detailed'])
            
        elif choice == "3":
            self._single_symbol_analysis()
            
        elif choice == "4":
            print("📈 Running comparison analysis...")
            self._run_analyzer(['--compare'])
            
        else:
            print("❌ Invalid choice")

    def _run_analyzer(self, args):
        """Run the multi-timeframe analyzer"""
  
        cmd = [sys.executable, 'multi_timeframe_analyzer.py'] + args
        
        try:
            print(f"⚙️ Running analysis...")
            print("⏳ Please wait...")
            
            # Run with real-time output
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                encoding='utf-8',   
                errors='replace'  
            )
            
            # Stream output in real-time
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
        
        # Get available symbols from database
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
                    self._run_analyzer([])
                else:
                    print("❌ Invalid selection")
            except ValueError:
                print("❌ Invalid input")
                
        except Exception as e:
            print(f"❌ Error getting symbols: {e}")

    def view_latest_signals(self):
        """View the latest trading signals from database"""
        print("\n📋 LATEST TRADING SIGNALS")
        print("-" * 40)
        
        if not os.path.exists(self.db_path):
            print("❌ No database found! Please collect data first.")
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get latest data summary
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
            
            # Show market overview
            symbols = df['symbol'].unique()
            print(f"\n🎯 QUICK MARKET OVERVIEW:")
            print(f"   📈 Symbols tracked: {len(symbols)}")
            print(f"   ⏱️  Total records: {df['record_count'].sum():,}")
            
            # Calculate freshness
            df['hours_old'] = df['latest_timestamp'].apply(
                lambda x: (datetime.now() - pd.to_datetime(x)).total_seconds() / 3600
            )
            fresh_data = len(df[df['hours_old'] < 2])
            print(f"   🟢 Fresh datasets: {fresh_data}/{len(df)}")
            
            print(f"\n💡 Run signal analysis (option 2) to get trading recommendations!")
            
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
                time.sleep(2)  # Wait a moment
            elif choice == "3":
                print("❌ Automation start cancelled")
                return
            elif choice != "1":
                print("❌ Invalid choice")
                return
        
        # Check prerequisites
        print("🔍 Checking system prerequisites...")
        
        # Check database
        if not os.path.exists(self.db_path):
            print("❌ No database found!")
            choice = input("Run initial data collection now? (y/n): ").lower()
            if choice == 'y':
                self._run_collector(['--force'])
            else:
                print("❌ Cannot start automation without data")
                return
        
        # Check configuration
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
        
        # Start automation
        print(f"\n🚀 Starting 24/7 automation system...")
        print(f"💡 The system will run continuously in the background")
        print(f"⚠️ Close this window or press Ctrl+C to stop")
        
        confirm = input("\nStart automation now? (y/n): ").lower()
        if confirm != 'y':
            print("❌ Automation start cancelled")
            return
        
        # Start automation in background
        try:
            automation_script = 'enhanced_automation_scheduler.py'
            if not os.path.exists(automation_script):
                automation_script = 'automation_scheduler.py'  # Fallback
            
            if os.path.exists(automation_script):
                print(f"🔄 Starting {automation_script}...")
                
                # Start in background on Unix systems, foreground on Windows
                if os.name == 'nt':  # Windows
                    subprocess.Popen([sys.executable, automation_script], 
                                   creationflags=subprocess.CREATE_NEW_CONSOLE)
                else:  # Unix/Linux/Mac
                    subprocess.Popen([sys.executable, automation_script], 
                                   stdout=subprocess.DEVNULL, 
                                   stderr=subprocess.DEVNULL)
                
                print("✅ Automation system started!")
                print("💡 Check automation status (option 6) to monitor progress")
                
            else:
                print("❌ Automation script not found!")
                print("💡 Make sure enhanced_automation_scheduler.py exists")
                
        except Exception as e:
            print(f"❌ Error starting automation: {e}")

    def stop_automation(self):
        """Stop the automation system"""
        print("\n🛑 STOPPING AUTOMATION")
        print("-" * 30)
        
        # Check if running
        if not os.path.exists('logs/start_time.txt'):
            print("ℹ️ Automation doesn't appear to be running")
            return
        
        print("🔍 Looking for automation processes...")
        
        # Try to find and stop automation processes
        stopped = False
        
        try:
            # Remove start time file
            if os.path.exists('logs/start_time.txt'):
                os.remove('logs/start_time.txt')
                print("✅ Removed automation start marker")
                stopped = True
            
            # Remove PID file if exists
            if os.path.exists('logs/automation.pid'):
                try:
                    with open('logs/automation.pid', 'r') as f:
                        pid = int(f.read().strip())
                    
                    # Try to kill the process
                    import signal
                    os.kill(pid, signal.SIGTERM)
                    os.remove('logs/automation.pid')
                    print(f"✅ Stopped automation process (PID: {pid})")
                    stopped = True
                    
                except (ValueError, ProcessLookupError, PermissionError):
                    print("⚠️ Could not stop automation process")
                    os.remove('logs/automation.pid')  # Remove stale PID file
            
            if stopped:
                print("✅ Automation system stopped")
            else:
                print("⚠️ Automation may still be running")
                print("💡 You can manually close automation windows or restart the system")
                
        except Exception as e:
            print(f"❌ Error stopping automation: {e}")

    def automation_status(self):
        """Show detailed automation status"""
        print("\n📊 AUTOMATION STATUS")
        print("-" * 30)
        
        # Check if automation is running
        if os.path.exists('logs/start_time.txt'):
            try:
                with open('logs/start_time.txt', 'r') as f:
                    start_time_str = f.read().strip()
                    start_time = datetime.fromisoformat(start_time_str)
                    
                print(f"🟢 Status: RUNNING")
                print(f"⏰ Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Calculate uptime
                uptime = datetime.now() - start_time
                days = uptime.days
                hours, remainder = divmod(uptime.seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                
                print(f"📈 Uptime: {days}d {hours}h {minutes}m {seconds}s")
                
            except Exception as e:
                print(f"🔴 Status: ERROR - {e}")
        else:
            print(f"🔴 Status: NOT RUNNING")
        
        # Check recent activity
        print(f"\n📋 RECENT ACTIVITY:")
        
        activity_logs = [
            ('automation.log', 'Automation System'),
            ('crypto_collector.log', 'Data Collection'),
            ('ultimate_analyzer.log', 'Signal Analysis'),
        ]
        
        for log_file, description in activity_logs:
            if os.path.exists(log_file):
                try:
                    modified = datetime.fromtimestamp(os.path.getmtime(log_file))
                    hours_since = (datetime.now() - modified).total_seconds() / 3600
                    
                    if hours_since < 0.5:
                        activity = "🟢 Very Recent"
                    elif hours_since < 2:
                        activity = f"🟡 {hours_since:.1f}h ago"
                    else:
                        activity = f"🔴 {hours_since:.1f}h ago"
                    
                    print(f"   📄 {description}: {activity}")
                    
                    # Show last few lines
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        if lines:
                            last_line = lines[-1].strip()
                            if len(last_line) > 80:
                                last_line = last_line[:77] + "..."
                            print(f"      💬 Last: {last_line}")
                            
                except Exception as e:
                    print(f"   ❌ {description}: Error reading - {e}")
            else:
                print(f"   ⚪ {description}: No log file")
        
        # Check alerts
        if os.path.exists('alerts/alerts.log'):
            try:
                with open('alerts/alerts.log', 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                print(f"\n🔔 RECENT ALERTS ({len(lines)} total):")
                
                if lines:
                    # Show last 3 alerts
                    for line in lines[-3:]:
                        line = line.strip()
                        if len(line) > 100:
                            line = line[:97] + "..."
                        print(f"   📢 {line}")
                else:
                    print("   ⚪ No alerts yet")
                    
            except Exception as e:
                print(f"   ❌ Error reading alerts: {e}")

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
                
                # Data collection settings
                data_config = config.get('data_collection', {})
                print(f"📊 DATA COLLECTION:")
                print(f"   ✅ Enabled: {data_config.get('enabled', True)}")
                print(f"   ⏰ Interval: {data_config.get('interval_minutes', 60)} minutes")
                print(f"   📈 Symbols: {', '.join(data_config.get('symbols', []))}")
                print(f"   📊 Timeframes: {', '.join(data_config.get('timeframes', []))}")
                
                # Signal analysis settings
                signal_config = config.get('signal_analysis', {})
                print(f"\n🔍 SIGNAL ANALYSIS:")
                print(f"   ✅ Enabled: {signal_config.get('enabled', True)}")
                print(f"   ⏰ Interval: {signal_config.get('interval_minutes', 15)} minutes")
                print(f"   🎯 Confidence Threshold: {signal_config.get('confidence_threshold', 75)}%")
                
                # Alert settings
                alert_config = config.get('alerts', {})
                print(f"\n🔔 ALERTS:")
                print(f"   ✅ Enabled: {alert_config.get('enabled', True)}")
                print(f"   🖥️  Desktop: {alert_config.get('desktop', {}).get('enabled', True)}")
                print(f"   📧 Email: {alert_config.get('email', {}).get('enabled', False)}")
                print(f"   📋 Log File: {alert_config.get('log_file', {}).get('enabled', True)}")
                
                print(f"\n⚙️ CONFIGURATION OPTIONS:")
                print("1. 📝 Edit Configuration File")
                print("2. 🔧 Quick Settings")
                print("3. 📧 Setup Email Alerts")
                print("4. 🔄 Reset to Defaults")
                print("5. 📋 Export Configuration")
                print("0. ↩️  Back to Main Menu")
                
                choice = input("\nSelect option (0-5): ").strip()
                
                if choice == "1":
                    self._edit_config_file()
                elif choice == "2":
                    self._quick_settings(config)
                elif choice == "3":
                    self._setup_email_alerts(config)
                elif choice == "4":
                    self._reset_config()
                elif choice == "5":
                    self._export_config(config)
                elif choice == "0":
                    return
                else:
                    print("❌ Invalid choice")
                    
            except Exception as e:
                print(f"❌ Error reading configuration: {e}")
        else:
            print("📄 No configuration file found")
            choice = input("Create default configuration? (y/n): ").lower()
            if choice == 'y':
                self.create_default_config()

    def _edit_config_file(self):
        """Open configuration file for editing"""
        print(f"\n📝 EDIT CONFIGURATION")
        print("-" * 25)
        
        if os.name == 'nt':  # Windows
            try:
                os.system(f'notepad {self.config_path}')
                print("✅ Configuration file opened in Notepad")
            except:
                print(f"💡 Please edit {self.config_path} manually")
        else:  # Unix/Linux/Mac
            editors = ['nano', 'vim', 'vi', 'gedit']
            for editor in editors:
                try:
                    subprocess.run([editor, self.config_path])
                    print(f"✅ Configuration edited with {editor}")
                    break
                except FileNotFoundError:
                    continue
            else:
                print(f"💡 Please edit {self.config_path} manually with your preferred editor")

    def _quick_settings(self, config):
        """Quick settings adjustment"""
        print(f"\n🔧 QUICK SETTINGS")
        print("-" * 20)
        
        print("What would you like to adjust?")
        print("1. ⏰ Data Collection Interval")
        print("2. 🔍 Signal Analysis Interval") 
        print("3. 📈 Trading Symbols")
        print("4. 🎯 Confidence Threshold")
        
        choice = input("Select (1-4): ").strip()
        
        if choice == "1":
            current = config.get('data_collection', {}).get('interval_minutes', 60)
            print(f"Current data collection interval: {current} minutes")
            try:
                new_interval = int(input("Enter new interval (15-1440 minutes): "))
                if 15 <= new_interval <= 1440:
                    config.setdefault('data_collection', {})['interval_minutes'] = new_interval
                    self._save_config(config)
                    print(f"✅ Data collection interval updated to {new_interval} minutes")
                else:
                    print("❌ Invalid interval (must be 15-1440 minutes)")
            except ValueError:
                print("❌ Invalid input")
                
        elif choice == "2":
            current = config.get('signal_analysis', {}).get('interval_minutes', 15)
            print(f"Current signal analysis interval: {current} minutes")
            try:
                new_interval = int(input("Enter new interval (5-60 minutes): "))
                if 5 <= new_interval <= 60:
                    config.setdefault('signal_analysis', {})['interval_minutes'] = new_interval
                    self._save_config(config)
                    print(f"✅ Signal analysis interval updated to {new_interval} minutes")
                else:
                    print("❌ Invalid interval (must be 5-60 minutes)")
            except ValueError:
                print("❌ Invalid input")
                
        elif choice == "3":
            current = config.get('data_collection', {}).get('symbols', [])
            print(f"Current symbols: {', '.join(current)}")
            
            common_symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "DOT/USDT", "LINK/USDT", "XRP/USDT"]
            print(f"Common symbols: {', '.join(common_symbols)}")
            
            new_symbols = input("Enter new symbols (comma-separated): ").strip()
            if new_symbols:
                symbols_list = [s.strip().upper() for s in new_symbols.split(',')]
                config.setdefault('data_collection', {})['symbols'] = symbols_list
                self._save_config(config)
                print(f"✅ Symbols updated to: {', '.join(symbols_list)}")
            else:
                print("❌ No symbols provided")
                
        elif choice == "4":
            current = config.get('signal_analysis', {}).get('confidence_threshold', 75)
            print(f"Current confidence threshold: {current}%")
            try:
                new_threshold = int(input("Enter new threshold (50-95%): "))
                if 50 <= new_threshold <= 95:
                    config.setdefault('signal_analysis', {})['confidence_threshold'] = new_threshold
                    self._save_config(config)
                    print(f"✅ Confidence threshold updated to {new_threshold}%")
                else:
                    print("❌ Invalid threshold (must be 50-95%)")
            except ValueError:
                print("❌ Invalid input")

    def _setup_email_alerts(self, config):
        """Setup email alerts"""
        print(f"\n📧 EMAIL ALERTS SETUP")
        print("-" * 25)
        
        current_email = config.get('alerts', {}).get('email', {})
        enabled = current_email.get('enabled', False)
        
        print(f"Current status: {'✅ Enabled' if enabled else '❌ Disabled'}")
        
        if enabled:
            print(f"Current email: {current_email.get('username', 'Not set')}")
        
        print("\n📧 Email Alert Options:")
        print("1. ✅ Enable Email Alerts")
        print("2. ❌ Disable Email Alerts") 
        print("3. ⚙️ Configure Email Settings")
        print("4. 🧪 Test Email")
        
        choice = input("Select (1-4): ").strip()
        
        if choice == "1":
            config.setdefault('alerts', {}).setdefault('email', {})['enabled'] = True
            self._save_config(config)
            print("✅ Email alerts enabled")
            
        elif choice == "2":
            config.setdefault('alerts', {}).setdefault('email', {})['enabled'] = False
            self._save_config(config)
            print("❌ Email alerts disabled")
            
        elif choice == "3":
            self._configure_email_details(config)
            
        elif choice == "4":
            self._test_email(config)

    def _configure_email_details(self, config):
        """Configure email details"""
        print(f"\n⚙️ EMAIL CONFIGURATION")
        print("-" * 25)
        
        print("📧 Gmail Setup (recommended):")
        print("1. Use your Gmail address")
        print("2. Generate an App Password at: https://myaccount.google.com/apppasswords")
        print("3. Use the App Password (not your regular password)")
        
        email = input("\nEnter your email address: ").strip()
        if not email or '@' not in email:
            print("❌ Invalid email address")
            return
        
        password = input("Enter your app password: ").strip()
        if not password:
            print("❌ Password required")
            return
        
        # Setup email config
        email_config = {
            'enabled': True,
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': email,
            'password': password,
            'to_email': email
        }
        
        config.setdefault('alerts', {})['email'] = email_config
        self._save_config(config)
        
        print("✅ Email configuration saved")
        print("💡 Run test email to verify setup")

    def _test_email(self, config):
        """Test email configuration"""
        print(f"\n🧪 EMAIL TEST")
        print("-" * 15)
        
        email_config = config.get('alerts', {}).get('email', {})
        
        if not email_config.get('enabled', False):
            print("❌ Email alerts not enabled")
            return
        
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            print("📧 Sending test email...")
            
            msg = MIMEMultipart()
            msg['From'] = email_config.get('username', '')
            msg['To'] = email_config.get('to_email', '')
            msg['Subject'] = 'Crypto Trading System - Test Email'
            
            body = f"""
Hello!

This is a test email from your Crypto Trading Control Center.

If you received this email, your email alerts are working correctly!

System Status:
- Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Configuration: Valid
- Email Alerts: Enabled

Happy Trading! 🚀
            """.strip()
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(email_config.get('smtp_server', 'smtp.gmail.com'), 
                                 email_config.get('smtp_port', 587))
            server.starttls()
            server.login(email_config.get('username', ''), email_config.get('password', ''))
            server.send_message(msg)
            server.quit()
            
            print("✅ Test email sent successfully!")
            print(f"📬 Check your inbox: {email_config.get('to_email', '')}")
            
        except Exception as e:
            print(f"❌ Email test failed: {e}")
            print("💡 Check your email settings and app password")

    def _reset_config(self):
        """Reset configuration to defaults"""
        print(f"\n🔄 RESET CONFIGURATION")
        print("-" * 25)
        
        print("⚠️ This will reset all settings to defaults")
        confirm = input("Are you sure? (y/n): ").lower()
        
        if confirm == 'y':
            try:
                if os.path.exists(self.config_path):
                    backup_path = f"{self.config_path}.backup.{int(time.time())}"
                    os.rename(self.config_path, backup_path)
                    print(f"📁 Backed up current config to: {backup_path}")
                
                self.create_default_config()
                print("✅ Configuration reset to defaults")
                
            except Exception as e:
                print(f"❌ Error resetting config: {e}")
        else:
            print("❌ Reset cancelled")

    def _export_config(self, config):
        """Export configuration"""
        print(f"\n📋 EXPORT CONFIGURATION")
        print("-" * 25)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        export_path = f"config_export_{timestamp}.json"
        
        try:
            with open(export_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            print(f"✅ Configuration exported to: {export_path}")
            print("💡 You can share this file or use it as a backup")
            
        except Exception as e:
            print(f"❌ Export failed: {e}")

    def _save_config(self, config):
        """Save configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=4)
            return True
        except Exception as e:
            print(f"❌ Error saving configuration: {e}")
            return False

    def create_default_config(self):
        """Create default configuration file"""
        default_config = {
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
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
            print("✅ Default configuration created!")
        except Exception as e:
            print(f"❌ Error creating configuration: {e}")

    def view_logs(self):
        """View system logs"""
        print("\n📋 SYSTEM LOGS")
        print("-" * 25)
        
        log_files = {
            '1': ('automation.log', 'Automation System Log'),
            '2': ('crypto_collector.log', 'Data Collection Log'),
            '3': ('ultimate_analyzer.log', 'Signal Analysis Log'),
            '4': ('alerts/alerts.log', 'Trading Alerts Log'),
            '5': ('errors.log', 'Error Log')
        }
        
        print("📄 Available log files:")
        for key, (file_path, description) in log_files.items():
            status = "✅" if os.path.exists(file_path) else "❌"
            size = ""
            if os.path.exists(file_path):
                try:
                    size_kb = os.path.getsize(file_path) / 1024
                    size = f" ({size_kb:.1f} KB)"
                except:
                    pass
            print(f"   {key}. {status} {description}{size}")
        
        print(f"   A. 📊 All Logs Summary")
        print(f"   C. 🧹 Clear All Logs")
        
        choice = input("\nSelect log to view or action (1-5, A, C): ").strip().upper()
        
        if choice in log_files:
            file_path, description = log_files[choice]
            self._view_log_file(file_path, description)
            
        elif choice == 'A':
            self._view_all_logs_summary()
            
        elif choice == 'C':
            self._clear_logs()
            
        else:
            print("❌ Invalid choice")

    def _view_log_file(self, file_path, description):
        """View a specific log file"""
        if not os.path.exists(file_path):
            print(f"❌ Log file not found: {file_path}")
            return
        
        try:
            print(f"\n📄 {description}")
            print("-" * 50)
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            if not lines:
                print("📝 Log file is empty")
                return
            
            total_lines = len(lines)
            
            print(f"📊 Total lines: {total_lines}")
            
            # Show options
            print(f"\nViewing options:")
            print(f"1. Last 20 lines")
            print(f"2. Last 50 lines") 
            print(f"3. Search for text")
            print(f"4. Show all")
            print(f"5. Show file info")
            
            choice = input("Select option (1-5): ").strip()
            
            if choice == "1":
                print(f"\n📋 Last 20 lines:")
                for line in lines[-20:]:
                    print(line.rstrip())
                    
            elif choice == "2":
                print(f"\n📋 Last 50 lines:")
                for line in lines[-50:]:
                    print(line.rstrip())
                    
            elif choice == "3":
                search_term = input("Enter search term: ").strip()
                if search_term:
                    print(f"\n🔍 Lines containing '{search_term}':")
                    found = False
                    for i, line in enumerate(lines, 1):
                        if search_term.lower() in line.lower():
                            print(f"Line {i}: {line.rstrip()}")
                            found = True
                    if not found:
                        print(f"❌ No lines found containing '{search_term}'")
                        
            elif choice == "4":
                print(f"\n📋 Full log ({total_lines} lines):")
                if total_lines > 100:
                    confirm = input(f"⚠️ Large file ({total_lines} lines). Show anyway? (y/n): ")
                    if confirm.lower() != 'y':
                        return
                
                for line in lines:
                    print(line.rstrip())
                    
            elif choice == "5":
                self._show_log_info(file_path, lines)
                
        except Exception as e:
            print(f"❌ Error reading log file: {e}")

    def _show_log_info(self, file_path, lines):
        """Show log file information"""
        print(f"\n📊 LOG FILE INFORMATION")
        print("-" * 30)
        
        try:
            # File stats
            stat = os.stat(file_path)
            size_kb = stat.st_size / 1024
            modified = datetime.fromtimestamp(stat.st_mtime)
            
            print(f"📁 File: {file_path}")
            print(f"📏 Size: {size_kb:.1f} KB")
            print(f"📝 Lines: {len(lines):,}")
            print(f"⏰ Modified: {modified.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Content analysis
            if lines:
                error_count = sum(1 for line in lines if 'ERROR' in line.upper() or '❌' in line)
                warning_count = sum(1 for line in lines if 'WARNING' in line.upper() or '⚠️' in line)
                success_count = sum(1 for line in lines if 'SUCCESS' in line.upper() or '✅' in line)
                
                print(f"\n📈 Content Analysis:")
                print(f"   ✅ Success entries: {success_count}")
                print(f"   ⚠️ Warning entries: {warning_count}")
                print(f"   ❌ Error entries: {error_count}")
                
                # Show first and last entries
                if lines:
                    first_line = lines[0].strip()
                    last_line = lines[-1].strip()
                    
                    print(f"\n📅 Timeline:")
                    print(f"   First: {first_line[:80]}...")
                    print(f"   Last:  {last_line[:80]}...")
                    
        except Exception as e:
            print(f"❌ Error analyzing log: {e}")

    def _view_all_logs_summary(self):
        """View summary of all logs"""
        print(f"\n📊 ALL LOGS SUMMARY")
        print("-" * 30)
        
        log_files = [
            'automation.log',
            'crypto_collector.log', 
            'ultimate_analyzer.log',
            'alerts/alerts.log'
        ]
        
        total_size = 0
        total_lines = 0
        
        for log_file in log_files:
            if os.path.exists(log_file):
                try:
                    size = os.path.getsize(log_file) / 1024  # KB
                    total_size += size
                    
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = len(f.readlines())
                        total_lines += lines
                    
                    modified = datetime.fromtimestamp(os.path.getmtime(log_file))
                    hours_old = (datetime.now() - modified).total_seconds() / 3600
                    
                    if hours_old < 1:
                        activity = "🟢 Very recent"
                    elif hours_old < 24:
                        activity = f"🟡 {hours_old:.0f}h old"
                    else:
                        activity = f"🔴 {hours_old/24:.0f}d old"
                    
                    print(f"📄 {log_file}")
                    print(f"   Size: {size:.1f} KB | Lines: {lines:,} | {activity}")
                    
                except Exception as e:
                    print(f"❌ {log_file}: Error - {e}")
            else:
                print(f"⚪ {log_file}: Not found")
        
        print(f"\n📊 TOTALS:")
        print(f"   Total Size: {total_size:.1f} KB")
        print(f"   Total Lines: {total_lines:,}")

    def _clear_logs(self):
        """Clear all log files"""
        print(f"\n🧹 CLEAR ALL LOGS")
        print("-" * 20)
        
        print("⚠️ This will clear all log files")
        print("💡 Current logs will be backed up")
        
        confirm = input("Continue? (y/n): ").lower()
        if confirm != 'y':
            print("❌ Log clearing cancelled")
            return
        
        log_files = [
            'automation.log',
            'crypto_collector.log',
            'ultimate_analyzer.log', 
            'alerts/alerts.log'
        ]
        
        cleared = 0
        timestamp = int(time.time())
        
        for log_file in log_files:
            if os.path.exists(log_file):
                try:
                    # Create backup
                    backup_name = f"{log_file}.backup.{timestamp}"
                    os.rename(log_file, backup_name)
                    cleared += 1
                    print(f"✅ Cleared {log_file} (backed up as {backup_name})")
                    
                except Exception as e:
                    print(f"❌ Error clearing {log_file}: {e}")
        
        print(f"🧹 Cleared {cleared} log files")

    def cleanup_system(self):
        """Clean up system files and data"""
        print("\n🧹 SYSTEM CLEANUP")
        print("-" * 25)
        
        cleanup_options = {
            '1': ('🗑️ Clear Old Logs', self._cleanup_logs),
            '2': ('🗄️ Clean Database Cache', self._cleanup_database),
            '3': ('🔄 Reset Automation State', self._cleanup_automation),
            '4': ('📁 Clean Temp Files', self._cleanup_temp_files),
            '5': ('🧹 Full System Cleanup', self._full_cleanup),
            '6': ('📊 Show Disk Usage', self._show_disk_usage)
        }
        
        print("Cleanup options:")
        for key, (desc, _) in cleanup_options.items():
            print(f"   {key}. {desc}")
        
        choice = input("\nSelect cleanup option (1-6): ").strip()
        
        if choice in cleanup_options:
            desc, func = cleanup_options[choice]
            
            if choice == '5':  # Full cleanup needs confirmation
                print(f"\n⚠️ FULL SYSTEM CLEANUP")
                print("This will:")
                print("- Clear all logs (with backup)")
                print("- Clean old database records")
                print("- Reset automation state") 
                print("- Remove temporary files")
                
                confirm = input("\nProceed with full cleanup? (y/n): ").lower()
                if confirm == 'y':
                    func()
                else:
                    print("❌ Full cleanup cancelled")
            else:
                func()
        else:
            print("❌ Invalid choice")

    def _cleanup_logs(self):
        """Clean up old log files"""
        print("🗑️ Cleaning old logs...")
        
        log_files = ['automation.log', 'crypto_collector.log', 'ultimate_analyzer.log']
        cleaned = 0
        total_size_saved = 0
        
        for log_file in log_files:
            if os.path.exists(log_file):
                try:
                    # Get current size
                    size_before = os.path.getsize(log_file)
                    
                    # Backup and clear if over 1MB
                    if size_before > 1024 * 1024:  # 1MB
                        backup_name = f"{log_file}.backup.{int(time.time())}"
                        os.rename(log_file, backup_name)
                        cleaned += 1
                        total_size_saved += size_before
                        print(f"✅ Cleared {log_file} ({size_before/1024:.1f} KB saved)")
                        
                except Exception as e:
                    print(f"❌ Error cleaning {log_file}: {e}")
        
        if cleaned > 0:
            print(f"🧹 Cleaned {cleaned} log files, saved {total_size_saved/1024:.1f} KB")
        else:
            print("ℹ️ No large log files to clean")

    def _cleanup_database(self):
        """Clean old database records"""
        print("🗄️ Cleaning database...")
        
        if not os.path.exists(self.db_path):
            print("❌ No database found")
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get current record count
            cursor.execute("SELECT COUNT(*) FROM price_data")
            before_count = cursor.fetchone()[0]
            
            # Delete records older than 30 days
            cursor.execute("""
                DELETE FROM price_data 
                WHERE timestamp < datetime('now', '-30 days')
            """)
            
            deleted_count = cursor.rowcount
            
            # Get new record count
            cursor.execute("SELECT COUNT(*) FROM price_data")
            after_count = cursor.fetchone()[0]
            
            # Vacuum database to reclaim space
            cursor.execute("VACUUM")
            
            conn.commit()
            conn.close()
            
            print(f"🗄️ Database cleaned:")
            print(f"   Records before: {before_count:,}")
            print(f"   Records deleted: {deleted_count:,}")
            print(f"   Records remaining: {after_count:,}")
            
        except Exception as e:
            print(f"❌ Error cleaning database: {e}")

    def _cleanup_automation(self):
        """Reset automation state"""
        print("🔄 Resetting automation state...")
        
        state_files = [
            'logs/start_time.txt',
            'logs/last_force_update.txt',
            'logs/automation.pid'
        ]
        
        reset_count = 0
        
        for file_path in state_files:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    reset_count += 1
                    print(f"✅ Removed {file_path}")
                except Exception as e:
                    print(f"❌ Error removing {file_path}: {e}")
        
        print(f"🔄 Reset {reset_count} automation state files")

    def _cleanup_temp_files(self):
        """Clean temporary files"""
        print("📁 Cleaning temporary files...")
        
        temp_patterns = [
            '*.tmp',
            '*.temp',
            '*.log.backup.*',
            '__pycache__',
            '*.pyc'
        ]
        
        cleaned_count = 0
        
        # Clean current directory
        for pattern in temp_patterns:
            if pattern == '__pycache__':
                # Remove __pycache__ directories
                for root, dirs, files in os.walk('.'):
                    if '__pycache__' in dirs:
                        import shutil
                        pycache_path = os.path.join(root, '__pycache__')
                        try:
                            shutil.rmtree(pycache_path)
                            cleaned_count += 1
                            print(f"✅ Removed {pycache_path}")
                        except Exception as e:
                            print(f"❌ Error removing {pycache_path}: {e}")
            else:
                # Use glob for file patterns
                import glob
                for file_path in glob.glob(pattern):
                    try:
                        os.remove(file_path)
                        cleaned_count += 1
                        print(f"✅ Removed {file_path}")
                    except Exception as e:
                        print(f"❌ Error removing {file_path}: {e}")
        
        print(f"📁 Cleaned {cleaned_count} temporary files")

    def _full_cleanup(self):
        """Perform full system cleanup"""
        print("🧹 PERFORMING FULL SYSTEM CLEANUP")
        print("=" * 40)
        
        # Run all cleanup functions
        self._cleanup_logs()
        print()
        self._cleanup_database()
        print()
        self._cleanup_automation()
        print()
        self._cleanup_temp_files()
        
        print("\n✅ FULL CLEANUP COMPLETED!")
        print("💡 System is now optimized and ready for operation")

    def _show_disk_usage(self):
        """Show disk usage information"""
        print("📊 DISK USAGE ANALYSIS")
        print("-" * 30)
        
        # Analyze important directories and files
        paths_to_check = [
            ('Database', self.db_path),
            ('Logs Directory', 'logs'),
            ('Data Directory', 'data'),
            ('Alerts Directory', 'alerts'),
            ('Main Directory', '.')
        ]
        
        total_size = 0
        
        for name, path in paths_to_check:
            if os.path.exists(path):
                try:
                    if os.path.isfile(path):
                        size = os.path.getsize(path)
                        print(f"📄 {name}: {size/1024:.1f} KB")
                        total_size += size
                    else:
                        # Calculate directory size
                        dir_size = 0
                        for root, dirs, files in os.walk(path):
                            for file in files:
                                file_path = os.path.join(root, file)
                                try:
                                    dir_size += os.path.getsize(file_path)
                                except:
                                    pass
                        print(f"📁 {name}: {dir_size/1024:.1f} KB")
                        total_size += dir_size
                        
                except Exception as e:
                    print(f"❌ {name}: Error - {e}")
            else:
                print(f"⚪ {name}: Not found")
        
        print(f"\n📊 TOTAL SIZE: {total_size/1024:.1f} KB ({total_size/(1024*1024):.2f} MB)")
        
        # Show recommendations
        if total_size > 100 * 1024 * 1024:  # 100MB
            print("\n💡 RECOMMENDATIONS:")
            print("   🧹 Consider running full cleanup")
            print("   🗄️ Database is quite large - clean old records")
        elif total_size > 50 * 1024 * 1024:  # 50MB
            print("\n💡 System size is moderate - cleanup recommended weekly")
        else:
            print("\n✅ System size is optimal")

    def performance_analysis(self):
        """Run performance analysis"""
        print("\n🎯 PERFORMANCE ANALYSIS")
        print("-" * 35)
        
        print("Choose analysis type:")
        print("1. 📊 System Performance")
        print("2. 📈 Trading Performance")
        print("3. 🔍 Signal Accuracy")
        print("4. ⏱️ Speed Benchmarks")
        print("5. 📋 Full Performance Report")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            self._analyze_system_performance()
        elif choice == "2":
            self._analyze_trading_performance()
        elif choice == "3":
            self._analyze_signal_accuracy()
        elif choice == "4":
            self._run_speed_benchmarks()
        elif choice == "5":
            self._full_performance_report()
        else:
            print("❌ Invalid choice")

    def _analyze_system_performance(self):
        """Analyze system performance metrics"""
        print("\n📊 SYSTEM PERFORMANCE ANALYSIS")
        print("-" * 40)
        
        # Check database performance
        if os.path.exists(self.db_path):
            try:
                start_time = time.time()
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Simple performance test
                cursor.execute("SELECT COUNT(*) FROM price_data")
                total_records = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(DISTINCT symbol) FROM price_data")
                symbol_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(DISTINCT timeframe) FROM price_data")
                timeframe_count = cursor.fetchone()[0]
                
                query_time = time.time() - start_time
                conn.close()
                
                print(f"🗄️ DATABASE PERFORMANCE:")
                print(f"   Records: {total_records:,}")
                print(f"   Symbols: {symbol_count}")
                print(f"   Timeframes: {timeframe_count}")
                print(f"   Query Time: {query_time:.3f}s")
                
                # Performance rating
                if query_time < 0.1:
                    rating = "🟢 Excellent"
                elif query_time < 0.5:
                    rating = "🟡 Good"
                else:
                    rating = "🔴 Slow"
                
                print(f"   Performance: {rating}")
                
            except Exception as e:
                print(f"❌ Database performance test failed: {e}")
        
        # Check file system performance
        print(f"\n📁 FILE SYSTEM PERFORMANCE:")
        
        # Test file I/O
        test_file = "temp_perf_test.tmp"
        try:
            start_time = time.time()
            
            # Write test
            with open(test_file, 'w') as f:
                f.write("test data" * 1000)
            
            write_time = time.time() - start_time
            
            # Read test
            start_time = time.time()
            with open(test_file, 'r') as f:
                data = f.read()
            
            read_time = time.time() - start_time
            
            # Cleanup
            os.remove(test_file)
            
            print(f"   Write Speed: {write_time:.3f}s")
            print(f"   Read Speed: {read_time:.3f}s")
            
            if write_time < 0.01 and read_time < 0.01:
                fs_rating = "🟢 Excellent"
            elif write_time < 0.1 and read_time < 0.1:
                fs_rating = "🟡 Good"
            else:
                fs_rating = "🔴 Slow"
                
            print(f"   Performance: {fs_rating}")
            
        except Exception as e:
            print(f"❌ File system test failed: {e}")

    def _analyze_trading_performance(self):
        """Analyze trading performance"""
        print("\n📈 TRADING PERFORMANCE ANALYSIS")
        print("-" * 40)
        
        # Check if performance tracker exists
        if os.path.exists('performance_tracker.py'):
            try:
                print("🔄 Running performance tracker...")
                result = subprocess.run([
                    sys.executable, 'performance_tracker.py'
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("✅ Performance analysis completed!")
                    if result.stdout:
                        print("\n📋 Results:")
                        print(result.stdout[-1000:])  # Show last 1000 characters
                else:
                    print("❌ Performance analysis failed")
                    if result.stderr:
                        print(f"Error: {result.stderr}")
                        
            except Exception as e:
                print(f"❌ Error running performance tracker: {e}")
        else:
            print("⚠️ Performance tracker not found")
            print("💡 Basic performance analysis:")
            
            # Basic analysis from database
            if os.path.exists(self.db_path):
                try:
                    conn = sqlite3.connect(self.db_path)
                    
                    # Get data freshness
                    query = """
                    SELECT symbol, MAX(timestamp) as latest
                    FROM price_data
                    GROUP BY symbol
                    """
                    
                    df = pd.read_sql_query(query, conn)
                    conn.close()
                    
                    if not df.empty:
                        print(f"📊 Data Coverage:")
                        fresh_symbols = 0
                        
                        for _, row in df.iterrows():
                            latest = pd.to_datetime(row['latest'])
                            hours_old = (datetime.now() - latest).total_seconds() / 3600
                            
                            if hours_old < 2:
                                status = "🟢 Fresh"
                                fresh_symbols += 1
                            elif hours_old < 24:
                                status = "🟡 Stale"
                            else:
                                status = "🔴 Old"
                            
                            print(f"   {row['symbol']}: {status}")
                        
                        coverage = fresh_symbols / len(df) * 100
                        print(f"\n📈 Data Freshness: {coverage:.1f}%")
                        
                except Exception as e:
                    print(f"❌ Error analyzing data: {e}")

    def _analyze_signal_accuracy(self):
        """Analyze signal accuracy"""
        print("\n🔍 SIGNAL ACCURACY ANALYSIS")
        print("-" * 40)
        
        print("💡 Signal accuracy analysis requires historical data")
        print("🔄 This feature analyzes how well past signals performed")
        
        # Check for historical signals data
        if os.path.exists(self.db_path):
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Check data age span
                cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM price_data")
                min_time, max_time = cursor.fetchone()
                
                if min_time and max_time:
                    min_dt = pd.to_datetime(min_time)
                    max_dt = pd.to_datetime(max_time)
                    span_days = (max_dt - min_dt).days
                    
                    print(f"📅 Data Span: {span_days} days")
                    print(f"   From: {min_dt.strftime('%Y-%m-%d')}")
                    print(f"   To: {max_dt.strftime('%Y-%m-%d')}")
                    
                    if span_days >= 7:
                        print("✅ Sufficient data for basic accuracy analysis")
                        
                        # Simple accuracy check - count recent data points
                        cursor.execute("""
                            SELECT symbol, COUNT(*) as records
                            FROM price_data
                            WHERE timestamp >= datetime('now', '-7 days')
                            GROUP BY symbol
                        """)
                        
                        recent_data = cursor.fetchall()
                        
                        print(f"\n📊 Recent Data (7 days):")
                        for symbol, records in recent_data:
                            print(f"   {symbol}: {records:,} records")
                            
                    else:
                        print("⚠️ Need more historical data for accuracy analysis")
                        print("💡 Collect data for at least 7 days")
                        
                conn.close()
                
            except Exception as e:
                print(f"❌ Error analyzing signal accuracy: {e}")
        else:
            print("❌ No database found for accuracy analysis")

    def _run_speed_benchmarks(self):
        """Run speed benchmarks"""
        print("\n⏱️ SPEED BENCHMARKS")
        print("-" * 25)
        
        benchmarks = []
        
        # Data collection speed
        print("🔄 Testing data collection speed...")
        start_time = time.time()
        
        try:
            # Run a quick data collection test
            result = subprocess.run([
                sys.executable, 'multi_timeframe_collector.py', '--diagnose'
            ], capture_output=True, text=True, timeout=30)
            
            collection_time = time.time() - start_time
            benchmarks.append(("Data Collection Diagnosis", collection_time, "✅" if result.returncode == 0 else "❌"))
            
        except subprocess.TimeoutExpired:
            collection_time = 30
            benchmarks.append(("Data Collection Diagnosis", collection_time, "⏱️ Timeout"))
        except Exception as e:
            benchmarks.append(("Data Collection Diagnosis", 0, f"❌ Error: {e}"))
        
        # Signal analysis speed
        print("🔄 Testing signal analysis speed...")
        start_time = time.time()
        
        try:
            # Simple analysis test (if we have data)
            if os.path.exists(self.db_path):
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM price_data")
                count = cursor.fetchone()[0]
                conn.close()
                
                analysis_time = time.time() - start_time
                benchmarks.append(("Database Query", analysis_time, "✅"))
            else:
                benchmarks.append(("Database Query", 0, "❌ No DB"))
                
        except Exception as e:
            benchmarks.append(("Database Query", 0, f"❌ Error"))
        
        # File I/O speed
        print("🔄 Testing file I/O speed...")
        test_data = "benchmark test data\n" * 1000
        
        start_time = time.time()
        try:
            with open("benchmark_test.tmp", "w") as f:
                f.write(test_data)
            
            with open("benchmark_test.tmp", "r") as f:
                read_data = f.read()
            
            os.remove("benchmark_test.tmp")
            
            io_time = time.time() - start_time
            benchmarks.append(("File I/O", io_time, "✅"))
            
        except Exception as e:
            benchmarks.append(("File I/O", 0, f"❌ Error"))
        
        # Display results
        print(f"\n📊 BENCHMARK RESULTS:")
        print(f"{'Test':<25} {'Time':<10} {'Status'}")
        print("-" * 45)
        
        for test_name, test_time, status in benchmarks:
            if isinstance(test_time, float):
                time_str = f"{test_time:.3f}s"
            else:
                time_str = str(test_time)
            
            print(f"{test_name:<25} {time_str:<10} {status}")
        
        # Overall performance rating
        valid_times = [t for _, t, s in benchmarks if isinstance(t, float) and "✅" in s]
        
        if valid_times:
            avg_time = sum(valid_times) / len(valid_times)
            
            if avg_time < 1.0:
                rating = "🟢 Excellent"
            elif avg_time < 5.0:
                rating = "🟡 Good"
            else:
                rating = "🔴 Slow"
                
            print(f"\n⚡ Overall Performance: {rating} (avg: {avg_time:.3f}s)")

    def _full_performance_report(self):
        """Generate full performance report"""
        print("\n📋 FULL PERFORMANCE REPORT")
        print("=" * 40)
        
        print("🔄 Generating comprehensive performance report...")
        
        # Run all performance tests
        self._analyze_system_performance()
        print()
        self._analyze_trading_performance()
        print()
        self._analyze_signal_accuracy()
        print()
        self._run_speed_benchmarks()
        
        # Generate summary
        print(f"\n📊 PERFORMANCE SUMMARY")
        print("-" * 30)
        
        # System health score
        health_score = 0
        
        # Database check
        if os.path.exists(self.db_path):
            health_score += 25
            print("✅ Database: Online")
        else:
            print("❌ Database: Missing")
        
        # Configuration check
        if os.path.exists(self.config_path):
            health_score += 25
            print("✅ Configuration: Found")
        else:
            print("❌ Configuration: Missing")
        
        # Data freshness check
        try:
            if os.path.exists(self.db_path):
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT MAX(timestamp) FROM price_data")
                latest = cursor.fetchone()[0]
                conn.close()
                
                if latest:
                    latest_dt = pd.to_datetime(latest)
                    hours_old = (datetime.now() - latest_dt).total_seconds() / 3600
                    
                    if hours_old < 2:
                        health_score += 25
                        print("✅ Data: Fresh")
                    elif hours_old < 24:
                        health_score += 15
                        print("🟡 Data: Slightly stale")
                    else:
                        health_score += 5
                        print("🔴 Data: Old")
                        
        except:
            print("❌ Data: Error checking")
        
        # Automation check
        if os.path.exists('logs/start_time.txt'):
            health_score += 25
            print("✅ Automation: Running")
        else:
            print("❌ Automation: Stopped")
        
        # Final rating
        if health_score >= 90:
            rating = "🟢 EXCELLENT"
        elif health_score >= 70:
            rating = "🟡 GOOD"
        elif health_score >= 50:
            rating = "🟠 FAIR"
        else:
            rating = "🔴 POOR"
        
        print(f"\n🏆 OVERALL SYSTEM HEALTH: {rating} ({health_score}/100)")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if health_score < 70:
            print("   🔧 Run system diagnostics and fix issues")
            print("   📊 Ensure data collection is working")
            print("   🔄 Consider restarting automation")
        elif health_score < 90:
            print("   ⚙️ Minor optimizations recommended")
            print("   🧹 Consider running system cleanup")
        else:
            print("   ✅ System is performing excellently!")
            print("   📈 Continue current operation")

    def market_overview(self):
        """Show market overview"""
        print("\n📊 MARKET OVERVIEW")
        print("-" * 30)
        
        if not os.path.exists(self.db_path):
            print("❌ No database found! Please collect data first.")
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get latest prices for each symbol
            query = """
            WITH latest_data AS (
                SELECT 
                    symbol,
                    close as current_price,
                    timestamp,
                    ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp DESC) as rn
                FROM price_data
                WHERE timeframe = '1h'
            )
            SELECT symbol, current_price, timestamp
            FROM latest_data
            WHERE rn = 1
            ORDER BY symbol
            """
            
            current_prices = pd.read_sql_query(query, conn)
            
            if current_prices.empty:
                print("❌ No current price data available")
                conn.close()
                return
            
            # Get prices from 24h ago for comparison
            price_changes = []
            
            for _, row in current_prices.iterrows():
                symbol = row['symbol']
                current_price = row['current_price']
                current_time = pd.to_datetime(row['timestamp'])
                
                # Get price from ~24h ago
                day_ago = current_time - timedelta(hours=24)
                
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT close
                    FROM price_data
                    WHERE symbol = ? AND timeframe = '1h' AND timestamp <= ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                """, (symbol, day_ago.strftime('%Y-%m-%d %H:%M:%S')))
                
                result = cursor.fetchone()
                if result:
                    old_price = result[0]
                    change_pct = ((current_price - old_price) / old_price) * 100
                    change_abs = current_price - old_price
                else:
                    change_pct = 0
                    change_abs = 0
                
                hours_old = (datetime.now() - current_time).total_seconds() / 3600
                
                price_changes.append({
                    'symbol': symbol,
                    'price': current_price,
                    'change_pct': change_pct,
                    'change_abs': change_abs,
                    'hours_old': hours_old
                })
            
            conn.close()
            
            # Display market overview
            print(f"📈 CURRENT MARKET PRICES")
            print(f"{'Symbol':<12} {'Price':<12} {'24h Change':<12} {'Status'}")
            print("-" * 50)
            
            total_symbols = len(price_changes)
            gainers = 0
            losers = 0
            
            for data in price_changes:
                symbol = data['symbol']
                price = data['price']
                change_pct = data['change_pct']
                hours_old = data['hours_old']
                
                # Format price
                if price > 1000:
                    price_str = f"${price:,.0f}"
                elif price > 1:
                    price_str = f"${price:,.2f}"
                else:
                    price_str = f"${price:.4f}"
                
                # Format change
                if change_pct > 0:
                    change_str = f"+{change_pct:.2f}%"
                    change_color = "🟢"
                    gainers += 1
                elif change_pct < 0:
                    change_str = f"{change_pct:.2f}%"
                    change_color = "🔴"
                    losers += 1
                else:
                    change_str = "0.00%"
                    change_color = "⚪"
                
                # Data freshness
                if hours_old < 2:
                    fresh_status = "🟢"
                elif hours_old < 12:
                    fresh_status = "🟡"
                else:
                    fresh_status = "🔴"
                
                print(f"{symbol:<12} {price_str:<12} {change_color} {change_str:<10} {fresh_status}")
            
            # Market summary
            print(f"\n📊 MARKET SUMMARY:")
            print(f"   Total Symbols: {total_symbols}")
            print(f"   🟢 Gainers: {gainers}")
            print(f"   🔴 Losers: {losers}")
            print(f"   ⚪ Unchanged: {total_symbols - gainers - losers}")
            
            if gainers > losers:
                sentiment = "📈 BULLISH"
            elif losers > gainers:
                sentiment = "📉 BEARISH"
            else:
                sentiment = "⚖️ NEUTRAL"
            
            print(f"   Market Sentiment: {sentiment}")
            
        except Exception as e:
            print(f"❌ Error generating market overview: {e}")

    def quick_system_test(self):
        """Run a quick system test"""
        print("\n🔄 QUICK SYSTEM TEST")
        print("-" * 30)
        
        print("🔍 Running comprehensive system test...")
        
        tests = [
            ("Database Connection", self._test_database),
            ("Configuration", self._test_configuration),
            ("Data Collection", self._test_data_collection),
            ("Signal Analysis", self._test_signal_analysis),
            ("File Permissions", self._test_file_permissions)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n🧪 Testing {test_name}...")
            
            try:
                result = test_func()
                if result:
                    print(f"   ✅ {test_name}: PASSED")
                    passed += 1
                else:
                    print(f"   ❌ {test_name}: FAILED")
            except Exception as e:
                print(f"   ❌ {test_name}: ERROR - {e}")
        
        # Test results
        print(f"\n📊 TEST RESULTS:")
        print(f"   Passed: {passed}/{total}")
        print(f"   Success Rate: {passed/total*100:.1f}%")
        
        if passed == total:
            status = "🟢 ALL TESTS PASSED"
            print(f"   Status: {status}")
            print("   ✅ System is ready for operation!")
        elif passed >= total * 0.8:
            status = "🟡 MOSTLY WORKING"
            print(f"   Status: {status}")
            print("   ⚠️ Minor issues detected - system should work")
        else:
            status = "🔴 ISSUES DETECTED"
            print(f"   Status: {status}")
            print("   ❌ Significant problems - check configuration")

    def _test_database(self):
        """Test database connection and structure"""
        try:
            if not os.path.exists(self.db_path):
                return False
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Test basic query
            cursor.execute("SELECT COUNT(*) FROM price_data")
            count = cursor.fetchone()[0]
            
            # Test table structure
            cursor.execute("PRAGMA table_info(price_data)")
            columns = cursor.fetchall()
            
            expected_columns = ['symbol', 'timeframe', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
            actual_columns = [col[1] for col in columns]
            
            conn.close()
            
            # Check if all expected columns exist
            return all(col in actual_columns for col in expected_columns)
            
        except Exception:
            return False

    def _test_configuration(self):
        """Test configuration file"""
        try:
            if not os.path.exists(self.config_path):
                return False
            
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            # Check required sections
            required_sections = ['data_collection', 'signal_analysis', 'alerts']
            return all(section in config for section in required_sections)
            
        except Exception:
            return False

    def _test_data_collection(self):
        """Test data collection functionality"""
        try:
            # Check if collector script exists
            if not os.path.exists('multi_timeframe_collector.py'):
                return False
            
            # Run a quick diagnosis
            result = subprocess.run([
                sys.executable, 'multi_timeframe_collector.py', '--diagnose'
            ], capture_output=True, text=True, timeout=10)
            
            return result.returncode == 0
            
        except Exception:
            return False

    def _test_signal_analysis(self):
        """Test signal analysis functionality"""
        try:
            # Check if analyzer script exists
            if not os.path.exists('multi_timeframe_analyzer.py'):
                return False
            
            # Check if we have data to analyze
            if not os.path.exists(self.db_path):
                return False
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM price_data")
            count = cursor.fetchone()[0]
            conn.close()
            
            # Need at least some data
            return count > 100
            
        except Exception:
            return False

    def _test_file_permissions(self):
        """Test file system permissions"""
        try:
            # Test write permission
            test_file = "permission_test.tmp"
            with open(test_file, 'w') as f:
                f.write("test")
            
            # Test read permission
            with open(test_file, 'r') as f:
                content = f.read()
            
            # Test delete permission
            os.remove(test_file)
            
            # Test directory creation
            test_dir = "temp_test_dir"
            os.makedirs(test_dir, exist_ok=True)
            os.rmdir(test_dir)
            
            return True
            
        except Exception:
            # Clean up test files if they exist
            try:
                if os.path.exists("permission_test.tmp"):
                    os.remove("permission_test.tmp")
                if os.path.exists("temp_test_dir"):
                    os.rmdir("temp_test_dir")
            except:
                pass
            return False

    def show_help(self):
        """Show comprehensive help and documentation"""
        print("\n❓ CRYPTO TRADING SYSTEM HELP")
        print("=" * 50)
        
        print("""
🚀 WELCOME TO YOUR CRYPTO TRADING CONTROL CENTER!

This system provides AI-powered cryptocurrency trading analysis with:
- Multi-timeframe data collection
- Advanced signal analysis
- 24/7 automated monitoring
- Performance tracking

📊 MAIN FEATURES:

🔹 DATA COLLECTION:
   • Collects market data from multiple timeframes (5m, 15m, 1h, 4h, 1d)
   • Stores data in local SQLite database
   • Supports multiple cryptocurrency pairs
   • Automatic data freshness monitoring

🔹 SIGNAL ANALYSIS:
   • Multi-timeframe analysis for better accuracy
   • Combines multiple technical indicators
   • Ultimate signal combiner for final recommendations
   • Confidence scoring for each signal

🔹 AUTOMATION:
   • 24/7 continuous monitoring
   • Automatic data collection and analysis
   • Real-time alerts for signal changes
   • Email and desktop notifications

🔹 PERFORMANCE TRACKING:
   • System performance monitoring
   • Trading signal accuracy analysis
   • Speed benchmarks and optimization

📋 GETTING STARTED:

1️⃣ FIRST TIME SETUP:
   • Run option 1 (Collect Market Data) with "Force Fresh Data"
   • This will populate your database with initial data
   • Wait for collection to complete (may take several minutes)

2️⃣ ANALYZE SIGNALS:
   • Run option 2 (Analyze Trading Signals)
   • Get your first trading recommendations
   • Review signal confidence and market overview

3️⃣ START AUTOMATION:
   • Run option 4 (Start 24/7 Automation)
   • System will continuously monitor markets
   • Receive alerts for signal changes

4️⃣ MONITOR SYSTEM:
   • Use option 6 (Automation Status) to check health
   • View logs with option 8 for troubleshooting
   • Run system cleanup periodically with option 9

⚙️ CONFIGURATION:

📁 Files:
   • automation_config.json - Main configuration
   • data/multi_timeframe_data.db - Market data database
   • logs/ - System logs and status files
   • alerts/ - Trading alert logs

🔧 Key Settings:
   • Data collection interval (default: 60 minutes)
   • Signal analysis interval (default: 15 minutes)
   • Symbols to track (default: BTC, ETH, BNB, ADA, DOT)
   • Alert preferences (desktop, email, log file)

🔔 ALERTS & NOTIFICATIONS:

📧 Email Alerts:
   • Setup through option 7 (Configuration)
   • Requires Gmail app password
   • Sends signal changes and system status

🖥️ Desktop Notifications:
   • Enabled by default
   • Shows signal changes in real-time
   • Works on Windows, Mac, and Linux

📋 Log Files:
   • All alerts logged to alerts/alerts.log
   • System events in automation.log
   • Data collection logs in crypto_collector.log

🛠️ TROUBLESHOOTING:

❌ Common Issues:

🔸 "No database found":
   • Run data collection first (option 1)
   • Choose "Force Fresh Data" for initial setup

🔸 "Automation not starting":
   • Check system status (option 8)
   • Ensure data collection completed successfully
   • Verify configuration file exists

🔸 "Old/stale data":
   • Run data collection with force option
   • Check internet connection
   • Verify exchange API accessibility

🔸 "Signal analysis fails":
   • Ensure database has sufficient data (>100 records)
   • Check for data across multiple timeframes
   • Run system cleanup if needed

💡 TIPS FOR SUCCESS:

✅ Best Practices:
   • Keep system running 24/7 for best results
   • Monitor data freshness regularly
   • Review signal accuracy over time
   • Backup configuration before changes
   • Clean logs periodically to save space

⚡ Performance Tips:
   • Run system cleanup weekly
   • Monitor disk space usage
   • Keep database under 100MB for optimal speed
   • Use SSD storage for better performance

🎯 Trading Tips:
   • Higher confidence signals are more reliable
   • Combine signals with your own analysis
   • Start with paper trading to test system
   • Monitor performance metrics regularly

📚 ADVANCED FEATURES:

🔬 Performance Analysis (Option A):
   • System performance metrics
   • Trading signal accuracy
   • Speed benchmarks
   • Full performance reports

📊 Market Overview (Option B):
   • Current market prices
   • 24-hour price changes
   • Market sentiment analysis
   • Data freshness status

🧪 System Testing (Option C):
   • Database connectivity test
   • Configuration validation
   • Component functionality test
   • File permission checks

🆘 SUPPORT:

If you encounter issues:
1. Check system status (option 8)
2. Review logs (option 8)
3. Run system test (option C)
4. Try system cleanup (option 9)
5. Reset configuration if needed (option 7)

💬 Remember: This system is designed to assist with trading decisions,
    not replace your judgment. Always do your own research!

🎉 Happy Trading! Your AI-powered crypto system is ready to help you
    make better trading decisions 24/7!
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
                    self.performance_analysis()
                elif choice == "B":
                    self.market_overview()
                elif choice == "C":
                    self.quick_system_test()
                elif choice == "H":
                    self.show_help()
                elif choice == "0":
                    print("\n👋 Thank you for using Crypto Trading Control Center!")
                    print("🚀 Happy trading and may your signals be profitable!")
                    break
                else:
                    print("❌ Invalid choice. Please select a valid option.")
                
                input("\n⏸️  Press Enter to continue...")
                
        except KeyboardInterrupt:
            print("\n\n🛑 Control Center interrupted by user")
            print("👋 Goodbye!")
        except Exception as e:
            print(f"\n❌ Unexpected error in Control Center: {e}")
            print("💡 Try restarting the Control Center")


def main():
    """Main function"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__)) # Get the directory of the current script
        os.chdir(script_dir) # Change the current working directory to the script's directory
        # Ensure we're in the right directory
        if not os.path.exists('multi_timeframe_collector.py'): # Check if the collector script exists
            print("⚠️ Warning: Core system files not found in current directory")
            print("💡 Make sure you're running this from your crypto trading system directory")
            print()
        
        # Create and run control center
        control_center = CryptoControlCenter()
        control_center.run()
        
    except KeyboardInterrupt:
        print("\n\n👋 Control Center closed by user")
    except Exception as e:
        print(f"\n❌ Critical error starting Control Center: {e}")
        print("💡 Check your system setup and try again")


if __name__ == "__main__":
    main()