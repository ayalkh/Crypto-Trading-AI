"""
Enhanced 24/7 Crypto Trading Automation System
Compatible with your multi_timeframe_collector and multi_timeframe_analyzer
"""
import os
import sys

# Fix encoding issues on Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, OSError):
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')

import schedule
import time
import threading
import smtplib
import logging
import json
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sqlite3
import pandas as pd
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('automation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class EnhancedTradingAutomation:
    def __init__(self, config_file='automation_config.json'):
        """Initialize the enhanced trading automation system"""
        self.config_file = config_file
        self.config = self.load_config()
        self.running = False
        self.last_signals = {}
        self.alert_history = []
        self.error_count = 0
        
        # Create necessary directories
        os.makedirs('logs', exist_ok=True)
        os.makedirs('alerts', exist_ok=True)
        
        logging.info("🚀 Enhanced Trading Automation System initialized")
    
    def load_config(self):
        """Load configuration from JSON file"""
        default_config = {
            "data_collection": {
                "enabled": True,
                "interval_minutes": 60,
                "symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "DOT/USDT"],
                "timeframes": ["5m", "15m", "1h", "4h", "1d"],
                "force_update_hours": 24  # Force fresh data every 24 hours
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
                },
                "webhook": {
                    "enabled": False,
                    "url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
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
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                # Merge with defaults
                config = {**default_config, **user_config}
                logging.info(f"✅ Configuration loaded from {self.config_file}")
            except Exception as e:
                logging.error(f"❌ Error loading config: {e}, using defaults")
                config = default_config
        else:
            config = default_config
            self.save_config(config)
            logging.info(f"📄 Default configuration created at {self.config_file}")
        
        return config
    
    def save_config(self, config=None):
        """Save configuration to JSON file"""
        if config is None:
            config = self.config
        
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)
            logging.info(f"💾 Configuration saved to {self.config_file}")
        except Exception as e:
            logging.error(f"❌ Error saving config: {e}")
    
    def run_data_collection(self):
        """Run the multi-timeframe data collector"""
        try:
            logging.info("📊 Starting data collection...")
            
            # Check if we need to force update
            last_force_update = self.get_last_force_update()
            hours_since_force = (datetime.now() - last_force_update).total_seconds() / 3600
            
            if hours_since_force >= self.config['data_collection']['force_update_hours']:
                logging.info("🔄 Running forced data collection (fresh data)")
                result = subprocess.run([
                    sys.executable, 'multi_timeframe_collector.py', '--force'
                ], capture_output=True, text=True)
                self.set_last_force_update()
            else:
                logging.info("📈 Running normal data collection")
                result = subprocess.run([
                    sys.executable, 'multi_timeframe_collector.py'
                ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logging.info("✅ Data collection completed successfully")
                return True
            else:
                logging.error(f"❌ Data collection failed: {result.stderr}")
                return False
                
        except Exception as e:
            logging.error(f"❌ Error running data collection: {e}")
            return False
    
    def run_signal_analysis(self):
        """Run the multi-timeframe analyzer"""
        try:
            logging.info("🔍 Starting signal analysis...")
            
            # Run the analyzer
            result = subprocess.run([
                sys.executable, 'multi_timeframe_analyzer.py'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logging.info("✅ Signal analysis completed successfully")
                
                # Parse results and check for alerts
                self.check_for_new_signals()
                return True
            else:
                logging.error(f"❌ Signal analysis failed: {result.stderr}")
                return False
                
        except Exception as e:
            logging.error(f"❌ Error running signal analysis: {e}")
            return False
    
    def check_for_new_signals(self):
        """Check database for new trading signals"""
        try:
            # This is a simplified version - you might want to integrate more directly
            # with your analyzer results
            symbols = self.config['signal_analysis']['analyze_symbols']
            
            for symbol in symbols:
                # You could add logic here to check the latest signals from your database
                # and compare with previous signals to detect changes
                pass
                
        except Exception as e:
            logging.error(f"❌ Error checking signals: {e}")
    
    def get_last_force_update(self):
        """Get timestamp of last forced update"""
        try:
            with open('logs/last_force_update.txt', 'r') as f:
                timestamp_str = f.read().strip()
                return datetime.fromisoformat(timestamp_str)
        except:
            return datetime.now() - timedelta(hours=25)  # Force update on first run
    
    def set_last_force_update(self):
        """Set timestamp of last forced update"""
        try:
            with open('logs/last_force_update.txt', 'w') as f:
                f.write(datetime.now().isoformat())
        except Exception as e:
            logging.error(f"❌ Error saving force update timestamp: {e}")
    
    def send_alert(self, subject, message, alert_type="INFO"):
        """Send alert via configured methods"""
        alert_record = {
            'timestamp': datetime.now(),
            'subject': subject,
            'message': message,
            'type': alert_type
        }
        
        self.alert_history.append(alert_record)
        
        # Desktop notification
        if self.config['alerts']['desktop']['enabled']:
            self.send_desktop_notification(subject, message)
        
        # Email notification
        if self.config['alerts']['email']['enabled']:
            self.send_email_notification(subject, message)
        
        # Log file
        if self.config['alerts']['log_file']['enabled']:
            self.log_alert(subject, message)
    
    def send_desktop_notification(self, title, message):
        """Send desktop notification"""
        try:
            import platform
            
            if platform.system() == "Windows":
                try:
                    import win10toast
                    toaster = win10toast.ToastNotifier()
                    toaster.show_toast(title, message, duration=10)
                except ImportError:
                    print(f"\n🔔 NOTIFICATION: {title}: {message}\n")
            else:
                os.system(f'notify-send "{title}" "{message}"')
                
        except Exception as e:
            logging.error(f"❌ Desktop notification failed: {e}")
            print(f"\n🔔 NOTIFICATION: {title}: {message}\n")
    
    def send_email_notification(self, subject, message):
        """Send email notification"""
        try:
            email_config = self.config['alerts']['email']
            
            msg = MIMEMultipart()
            msg['From'] = email_config['username']
            msg['To'] = email_config['to_email']
            msg['Subject'] = f"Crypto Trading Alert: {subject}"
            
            msg.attach(MIMEText(message, 'plain'))
            
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            server.send_message(msg)
            server.quit()
            
            logging.info("📧 Email notification sent successfully")
            
        except Exception as e:
            logging.error(f"❌ Email notification failed: {e}")
    
    def log_alert(self, subject, message):
        """Log alert to file"""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_entry = f"[{timestamp}] {subject}: {message}\n"
            
            with open('alerts/alerts.log', 'a', encoding='utf-8') as f:
                f.write(log_entry)
                
        except Exception as e:
            logging.error(f"❌ Alert logging failed: {e}")
    
    def setup_schedule(self):
        """Set up the automation schedule"""
        config = self.config
        
        # Data collection
        if config['data_collection']['enabled']:
            interval = config['data_collection']['interval_minutes']
            schedule.every(interval).minutes.do(self.run_data_collection)
            logging.info(f"📅 Scheduled: Data collection every {interval} minutes")
        
        # Signal analysis
        if config['signal_analysis']['enabled']:
            interval = config['signal_analysis']['interval_minutes']
            schedule.every(interval).minutes.do(self.run_signal_analysis)
            logging.info(f"📅 Scheduled: Signal analysis every {interval} minutes")
        
        # Performance tracking
        if config['performance_tracking']['enabled']:
            interval = config['performance_tracking']['interval_hours']
            schedule.every(interval).hours.do(self.performance_tracking_job)
            logging.info(f"📅 Scheduled: Performance tracking every {interval} hours")
        
        # Daily cleanup
        schedule.every().day.at("03:00").do(self.cleanup_job)
        logging.info("📅 Scheduled: Daily cleanup at 3:00 AM")
    
    def performance_tracking_job(self):
        """Scheduled job for performance tracking"""
        try:
            logging.info("📊 Starting performance tracking...")
            
            # You could integrate your performance tracker here
            # For now, just log system status
            alert_count = len(self.alert_history)
            error_rate = self.error_count / max(alert_count, 1) * 100
            
            status_message = f"""
SYSTEM PERFORMANCE REPORT

Alert Count: {alert_count}
Error Rate: {error_rate:.1f}%
Uptime: {self.get_uptime()}
Database Size: {self.get_database_size()}

System Status: {'🟢 Healthy' if error_rate < 10 else '🟡 Warning' if error_rate < 25 else '🔴 Critical'}
            """.strip()
            
            self.send_alert("Performance Report", status_message, "PERFORMANCE")
            logging.info("✅ Performance tracking complete")
            
        except Exception as e:
            logging.error(f"❌ Performance tracking failed: {e}")
    
    def cleanup_job(self):
        """Daily cleanup job"""
        try:
            logging.info("🧹 Starting daily cleanup...")
            
            # Clean old logs
            log_files = ['automation.log', 'alerts/alerts.log']
            for log_file in log_files:
                if os.path.exists(log_file):
                    self.rotate_log_file(log_file)
            
            # Reset error count
            self.error_count = 0
            
            # Clean old alert history (keep last 1000)
            if len(self.alert_history) > 1000:
                self.alert_history = self.alert_history[-1000:]
            
            logging.info("✅ Daily cleanup complete")
            
        except Exception as e:
            logging.error(f"❌ Daily cleanup failed: {e}")
    
    def rotate_log_file(self, file_path, max_size_mb=10):
        """Rotate log file if it's too large"""
        try:
            if os.path.exists(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                if size_mb > max_size_mb:
                    backup_path = f"{file_path}.backup"
                    os.rename(file_path, backup_path)
                    logging.info(f"📁 Rotated log file: {file_path}")
        except Exception as e:
            logging.error(f"❌ Error rotating log file: {e}")
    
    def get_uptime(self):
        """Get system uptime"""
        try:
            with open('logs/start_time.txt', 'r') as f:
                start_time = datetime.fromisoformat(f.read().strip())
                uptime = datetime.now() - start_time
                days = uptime.days
                hours, remainder = divmod(uptime.seconds, 3600)
                return f"{days}d {hours}h"
        except:
            return "Unknown"
    
    def get_database_size(self):
        """Get database size"""
        try:
            db_path = self.config['system']['database_path']
            if os.path.exists(db_path):
                size_mb = os.path.getsize(db_path) / (1024 * 1024)
                return f"{size_mb:.1f} MB"
            return "0 MB"
        except:
            return "Unknown"
    
    def start(self):
        """Start the 24/7 automation system"""
        if self.running:
            logging.warning("⚠️ Automation system already running")
            return
        
        self.running = True
        
        # Save start time
        os.makedirs('logs', exist_ok=True)
        with open('logs/start_time.txt', 'w') as f:
            f.write(datetime.now().isoformat())
        
        logging.info("🚀 Starting Enhanced Trading Automation System")
        
        # Setup schedule
        self.setup_schedule()
        
        # Run initial data collection
        logging.info("🔄 Running initial data collection...")
        self.run_data_collection()
        
        # Send startup notification
        startup_message = f"""
ENHANCED CRYPTO TRADING AUTOMATION STARTED

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Status: System online and monitoring

Configuration:
- Data Collection: Every {self.config['data_collection']['interval_minutes']} minutes
- Signal Analysis: Every {self.config['signal_analysis']['interval_minutes']} minutes  
- Performance Tracking: Every {self.config['performance_tracking']['interval_hours']} hours

Symbols: {', '.join(self.config['signal_analysis']['analyze_symbols'])}
Timeframes: {', '.join(self.config['data_collection']['timeframes'])}

System is now running 24/7! 🚀
        """.strip()
        
        self.send_alert("System Started", startup_message, "SYSTEM")
        
        # Main automation loop
        try:
            while self.running:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds
                
        except KeyboardInterrupt:
            logging.info("🛑 Shutdown requested by user")
            self.stop()
        except Exception as e:
            logging.error(f"❌ Automation system error: {e}")
            self.send_alert("System Error", f"Automation system encountered an error: {e}", "ERROR")
            self.stop()
    
    def stop(self):
        """Stop the automation system"""
        if not self.running:
            logging.warning("⚠️ Automation system not running")
            return
        
        self.running = False
        logging.info("🛑 Stopping Enhanced Trading Automation System")
        
        # Send shutdown notification
        shutdown_message = f"""
ENHANCED CRYPTO TRADING AUTOMATION STOPPED

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Status: System offline

Statistics:
- Total Alerts: {len(self.alert_history)}
- Uptime: {self.get_uptime()}
- Errors: {self.error_count}

System has been safely shut down. ✅
        """.strip()
        
        self.send_alert("System Stopped", shutdown_message, "SYSTEM")
        
        # Clear schedule
        schedule.clear()
        logging.info("✅ Enhanced automation system stopped successfully")

def main():
    """Main function"""
    print("🚀 ENHANCED CRYPTO TRADING AUTOMATION")
    print("=" * 50)
    print("🔥 Automated Data Collection + Signal Analysis")
    print("=" * 50)
    
    # Create and start automation
    automation = EnhancedTradingAutomation()
    
    try:
        automation.start()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        automation.stop()

if __name__ == "__main__":
    main()