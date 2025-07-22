#!/usr/bin/env python3
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
    print("\nSTOPPING: Received shutdown signal")
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
