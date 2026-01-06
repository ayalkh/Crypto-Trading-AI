import os
import sys
import time
import logging
import json
import subprocess
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scheduler.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("CryptoScheduler")

class CryptoAutomationScheduler:
    def __init__(self, config_path='automation_config.json'):
        self.config_path = config_path
        self.config = self._load_config()
        self.scheduler = BackgroundScheduler()
        self._setup_listeners()
        
    def _load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}

    def _setup_listeners(self):
        def listener(event):
            if event.exception:
                logger.error(f"Job {event.job_id} failed: {event.exception}")
            else:
                logger.info(f"Job {event.job_id} executed successfully")
        
        self.scheduler.add_listener(listener, EVENT_JOB_ERROR | EVENT_JOB_EXECUTED)

    def run_data_collection(self):
        logger.info("Starting data collection job...")
        try:
            # Using subprocess to run the existing script
            # In a future refactor, this should call a python function directly
            cmd = [sys.executable, 'multi_timeframe_collector.py']
            
            # Check for force update logic (could be implemented here or in the script)
            # For now running standard collection
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Data collection completed successfully")
                logger.debug(result.stdout)
            else:
                logger.error(f"Data collection failed with code {result.returncode}")
                logger.error(result.stderr)
        except Exception as e:
            logger.error(f"Error in data collection job: {e}")

    def run_signal_analysis(self):
        logger.info("Starting signal analysis job...")
        try:
            cmd = [sys.executable, 'unified_crypto_analyzer.py']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Signal analysis completed successfully")
            else:
                logger.error(f"Signal analysis failed: {result.stderr}")
        except Exception as e:
            logger.error(f"Error in signal analysis job: {e}")

    def start(self):
        logger.info("Initializing Crypto Automation Scheduler...")
        
        # Data Collection
        if self.config.get('data_collection', {}).get('enabled', True):
            interval_mins = self.config.get('data_collection', {}).get('interval_minutes', 60)
            self.scheduler.add_job(
                self.run_data_collection,
                IntervalTrigger(minutes=interval_mins),
                id='data_collection',
                name='Data Collection',
                replace_existing=True,
                next_run_time=datetime.now() # Run immediately on start
            )
            logger.info(f"Scheduled Data Collection every {interval_mins} minutes")

        # Signal Analysis
        if self.config.get('signal_analysis', {}).get('enabled', True):
            interval_mins = self.config.get('signal_analysis', {}).get('interval_minutes', 15)
            self.scheduler.add_job(
                self.run_signal_analysis,
                IntervalTrigger(minutes=interval_mins),
                id='signal_analysis',
                name='Signal Analysis',
                replace_existing=True,
                next_run_time=datetime.now() # Run immediately on start
            )
            logger.info(f"Scheduled Signal Analysis every {interval_mins} minutes")

        self.scheduler.start()
        logger.info("Scheduler started. Press Ctrl+C to exit.")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping scheduler...")
            self.scheduler.shutdown()

if __name__ == "__main__":
    scheduler = CryptoAutomationScheduler()
    scheduler.start()
