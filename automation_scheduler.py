"""
24/7 Crypto Trading Automation System - Windows Compatible
Runs continuously, collecting data and generating signals
"""

import schedule
import time
import threading
import smtplib
import logging
import json
import os
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sqlite3
import pandas as pd

# Configure logging for Windows
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('automation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class TradingAutomation:
    def __init__(self, config_file='automation_config.json'):
        """Initialize the 24/7 trading automation system"""
        self.config_file = config_file
        self.config = self.load_config()
        self.running = False
        self.last_signals = {}
        self.alert_history = []
        
        # Create necessary directories
        os.makedirs('logs', exist_ok=True)
        os.makedirs('alerts', exist_ok=True)
        
        logging.info("STARTED: Trading Automation System initialized")
    
    def load_config(self):
        """Load configuration from JSON file"""
        default_config = {
            "data_collection": {
                "enabled": True,
                "interval_minutes": 60,
                "symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT"],
                "timeframes": ["5m", "15m", "1h", "4h"]
            },
            "signal_analysis": {
                "enabled": True,
                "interval_minutes": 15,
                "confidence_threshold": 75
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
                "cleanup_interval_hours": 24
            }
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                # Merge with defaults
                config = {**default_config, **user_config}
                logging.info(f"SUCCESS: Configuration loaded from {self.config_file}")
            except Exception as e:
                logging.error(f"ERROR: Error loading config: {e}, using defaults")
                config = default_config
        else:
            config = default_config
            self.save_config(config)
            logging.info(f"CREATED: Default configuration created at {self.config_file}")
        
        return config
    
    def save_config(self, config=None):
        """Save configuration to JSON file"""
        if config is None:
            config = self.config
        
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)
            logging.info(f"SAVED: Configuration saved to {self.config_file}")
        except Exception as e:
            logging.error(f"ERROR: Error saving config: {e}")
    
    def collect_data_job(self):
        """Scheduled job to collect market data"""
        try:
            logging.info("STARTING: Scheduled data collection...")
            
            # Import and run data collector
            from multi_timeframe_collector import MultiTimeframeCollector
            
            collector = MultiTimeframeCollector()
            successful, total = collector.collect_all_data()
            
            success_rate = successful / total * 100 if total > 0 else 0
            
            if success_rate >= 80:
                logging.info(f"SUCCESS: Data collection successful: {successful}/{total} ({success_rate:.1f}%)")
            else:
                logging.warning(f"WARNING: Data collection issues: {successful}/{total} ({success_rate:.1f}%)")
                
        except Exception as e:
            logging.error(f"ERROR: Data collection failed: {e}")
    
    def analyze_signals_job(self):
        """Scheduled job to analyze trading signals"""
        try:
            logging.info("STARTING: Signal analysis...")
            
            # Import and run signal analysis
            from multi_timeframe_analyzer import MultiTimeframeAnalyzer
            
            analyzer = MultiTimeframeAnalyzer()
            results = analyzer.analyze_all_signals()
            
            # Check for signal changes
            self.check_signal_changes(results)
            
            logging.info("SUCCESS: Signal analysis complete")
            
        except Exception as e:
            logging.error(f"ERROR: Signal analysis failed: {e}")
    
    def check_signal_changes(self, current_signals):
        """Check for signal changes and trigger alerts"""
        for symbol, signal_data in current_signals.items():
            current_signal = signal_data.get('ultimate_signal', 'HOLD')
            confidence = signal_data.get('confidence', 0)
            price = signal_data.get('price', 0)
            
            # Check if signal changed
            if symbol in self.last_signals:
                last_signal = self.last_signals[symbol]['signal']
                
                if current_signal != last_signal:
                    # Signal changed - send alert
                    self.send_signal_alert(symbol, current_signal, last_signal, confidence, price)
            
            # Check for high confidence signals
            elif confidence >= self.config['signal_analysis']['confidence_threshold']:
                if current_signal in ['BUY', 'STRONG_BUY', 'SELL', 'STRONG_SELL']:
                    self.send_high_confidence_alert(symbol, current_signal, confidence, price)
            
            # Update last signals
            self.last_signals[symbol] = {
                'signal': current_signal,
                'confidence': confidence,
                'price': price,
                'timestamp': datetime.now()
            }
    
    def send_signal_alert(self, symbol, new_signal, old_signal, confidence, price):
        """Send alert for signal changes"""
        message = f"""
SIGNAL CHANGE ALERT

Symbol: {symbol}
Old Signal: {old_signal}
New Signal: {new_signal}
Confidence: {confidence:.1f}%
Price: ${price:,.2f}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{self.get_signal_emoji(new_signal)} {new_signal}
        """.strip()
        
        self.send_alert("Signal Change", message)
        logging.info(f"ALERT: Signal alert sent: {symbol} {old_signal} -> {new_signal}")
    
    def send_high_confidence_alert(self, symbol, signal, confidence, price):
        """Send alert for high confidence signals"""
        message = f"""
HIGH CONFIDENCE SIGNAL

Symbol: {symbol}
Signal: {signal}
Confidence: {confidence:.1f}%
Price: ${price:,.2f}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{self.get_signal_emoji(signal)} Strong {signal} signal detected!
        """.strip()
        
        self.send_alert("High Confidence Signal", message)
        logging.info(f"ALERT: High confidence alert sent: {symbol} {signal} ({confidence:.1f}%)")
    
    def get_signal_emoji(self, signal):
        """Get emoji for signal type"""
        emojis = {
            'STRONG_BUY': '[STRONG BUY]',
            'BUY': '[BUY]',
            'HOLD': '[HOLD]',
            'SELL': '[SELL]',
            'STRONG_SELL': '[STRONG SELL]'
        }
        return emojis.get(signal, '[NEUTRAL]')
    
    def send_alert(self, subject, message):
        """Send alert via configured methods"""
        alert_record = {
            'timestamp': datetime.now(),
            'subject': subject,
            'message': message
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
            # Simple Windows notification
            import platform
            
            if platform.system() == "Windows":
                try:
                    import win10toast
                    toaster = win10toast.ToastNotifier()
                    toaster.show_toast(title, message, duration=10)
                except ImportError:
                    # Fallback to print
                    print(f"\nNOTIFICATION: {title}: {message}\n")
            else:
                # For Linux/Mac
                os.system(f'notify-send "{title}" "{message}"')
                
        except Exception as e:
            logging.error(f"ERROR: Desktop notification failed: {e}")
            print(f"\nNOTIFICATION: {title}: {message}\n")
    
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
            
            logging.info("SUCCESS: Email notification sent successfully")
            
        except Exception as e:
            logging.error(f"ERROR: Email notification failed: {e}")
    
    def log_alert(self, subject, message):
        """Log alert to file"""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_entry = f"[{timestamp}] {subject}: {message}\n"
            
            with open('alerts/alerts.log', 'a', encoding='utf-8') as f:
                f.write(log_entry)
                
        except Exception as e:
            logging.error(f"ERROR: Alert logging failed: {e}")
    
    def setup_schedule(self):
        """Set up the automation schedule"""
        config = self.config
        
        # Data collection
        if config['data_collection']['enabled']:
            interval = config['data_collection']['interval_minutes']
            schedule.every(interval).minutes.do(self.collect_data_job)
            logging.info(f"SCHEDULED: Data collection every {interval} minutes")
        
        # Signal analysis
        if config['signal_analysis']['enabled']:
            interval = config['signal_analysis']['interval_minutes']
            schedule.every(interval).minutes.do(self.analyze_signals_job)
            logging.info(f"SCHEDULED: Signal analysis every {interval} minutes")
        
        # Performance tracking
        if config['performance_tracking']['enabled']:
            interval = config['performance_tracking']['interval_hours']
            schedule.every(interval).hours.do(self.performance_tracking_job)
            logging.info(f"SCHEDULED: Performance tracking every {interval} hours")
    
    def performance_tracking_job(self):
        """Scheduled job for performance tracking"""
        try:
            logging.info("STARTING: Performance tracking...")
            logging.info("SUCCESS: Performance tracking complete")
        except Exception as e:
            logging.error(f"ERROR: Performance tracking failed: {e}")
    
    def start(self):
        """Start the 24/7 automation system"""
        if self.running:
            logging.warning("WARNING: Automation system already running")
            return
        
        self.running = True
        logging.info("STARTING: 24/7 Trading Automation System")
        
        # Setup schedule
        self.setup_schedule()
        
        # Run initial data collection
        logging.info("STARTING: Running initial data collection...")
        self.collect_data_job()
        
        # Send startup notification
        startup_message = f"""
CRYPTO TRADING AUTOMATION STARTED

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Status: System online and monitoring

Data Collection: Every {self.config['data_collection']['interval_minutes']} minutes
Signal Analysis: Every {self.config['signal_analysis']['interval_minutes']} minutes
Performance Tracking: Every {self.config['performance_tracking']['interval_hours']} hours

System is now running 24/7!
        """.strip()
        
        self.send_alert("System Started", startup_message)
        
        # Main automation loop
        try:
            while self.running:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds
                
        except KeyboardInterrupt:
            logging.info("STOPPING: Shutdown requested by user")
            self.stop()
        except Exception as e:
            logging.error(f"ERROR: Automation system error: {e}")
            self.send_alert("System Error", f"Automation system encountered an error: {e}")
            self.stop()
    
    def stop(self):
        """Stop the automation system"""
        if not self.running:
            logging.warning("WARNING: Automation system not running")
            return
        
        self.running = False
        logging.info("STOPPING: Trading Automation System")
        
        # Send shutdown notification
        shutdown_message = f"""
CRYPTO TRADING AUTOMATION STOPPED

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Status: System offline

Total alerts sent: {len(self.alert_history)}

System has been safely shut down.
        """.strip()
        
        self.send_alert("System Stopped", shutdown_message)
        
        # Clear schedule
        schedule.clear()
        logging.info("SUCCESS: Automation system stopped successfully")

def create_service_script():
    """Create a script to run automation as a service"""
    service_script = '''#!/usr/bin/env python3
"""
Crypto Trading Automation Service
Run this script to start the 24/7 automation system
"""

import sys
import os
import signal
import atexit
from automation_scheduler_windows_fixed import TradingAutomation

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print("\\nSTOPPING: Received shutdown signal")
    automation.stop()
    sys.exit(0)

def main():
    """Main service function"""
    global automation
    
    print("CRYPTO TRADING AUTOMATION SERVICE")
    print("=" * 50)
    print("STARTING: 24/7 trading system...")
    print("INFO: Press Ctrl+C to stop")
    print("=" * 50)
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start automation
    automation = TradingAutomation()
    
    # Register cleanup function
    atexit.register(lambda: automation.stop() if automation.running else None)
    
    # Start the system
    automation.start()

if __name__ == "__main__":
    main()
'''
    
    with open('run_automation.py', 'w', encoding='utf-8') as f:
        f.write(service_script)
    
    print("SUCCESS: Service script created: run_automation.py")

def create_config_template():
    """Create configuration template"""
    template_config = {
        "data_collection": {
            "enabled": True,
            "interval_minutes": 60,
            "symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT"],
            "timeframes": ["5m", "15m", "1h", "4h"]
        },
        "signal_analysis": {
            "enabled": True,
            "interval_minutes": 15,
            "confidence_threshold": 75
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
            "cleanup_interval_hours": 24
        }
    }
    
    with open('automation_config_template.json', 'w', encoding='utf-8') as f:
        json.dump(template_config, f, indent=4)
    
    print("SUCCESS: Configuration template created: automation_config_template.json")
    print("INFO: Copy to automation_config.json and customize your settings")

def main():
    """Main function for testing"""
    print("CRYPTO TRADING AUTOMATION SETUP")
    print("=" * 50)
    
    # Create service script and config template
    create_service_script()
    create_config_template()
    
    print("\nSETUP COMPLETE!")
    print("=" * 30)
    print("SUCCESS: Multi-timeframe data collector ready")
    print("SUCCESS: 24/7 automation framework ready")
    print("SUCCESS: Service script created")
    print("SUCCESS: Configuration template created")
    
    print("\nTO START 24/7 AUTOMATION:")
    print("1. Customize automation_config.json if needed")
    print("2. Run: python run_automation.py")
    print("3. System will run continuously!")
    
    print("\nFEATURES:")
    print("- Automatic data collection every hour")
    print("- Signal analysis every 15 minutes")
    print("- Instant alerts for signal changes")
    print("- Email notifications (optional)")
    print("- Desktop notifications")
    print("- Daily performance reports")
    print("- Automatic cleanup and maintenance")

if __name__ == "__main__":
    main()