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
            cmd = [sys.executable, 'collect_data.py']
            
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
            cmd = [sys.executable, 'analyze_signals.py']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Signal analysis completed successfully")
            else:
                logger.error(f"Signal analysis failed: {result.stderr}")
        except Exception as e:
            logger.error(f"Error in signal analysis job: {e}")

    def run_prediction_generation(self):
        logger.info("Starting prediction generation job...")
        try:
            cmd = [sys.executable, 'generate_predictions.py']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Prediction generation completed successfully")
            else:
                logger.error(f"Prediction generation failed: {result.stderr}")
                return False
            return True
        except Exception as e:
            logger.error(f"Error in prediction generation job: {e}")
            return False

    def run_model_training(self):
        logger.info("Starting daily model training...")
        try:
            # Run with --auto-run to avoid interactive prompts
            cmd = [sys.executable, 'train_models.py', '--auto-run']
            
            # This can take a while, so we increase timeout or just let it run
            # Using Popen to not block if we wanted, but run() is fine for a background thread
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Model training completed successfully")
            else:
                logger.error(f"Model training failed: {result.stderr}")
        except Exception as e:
            logger.error(f"Error in model training job: {e}")

    def run_trading_pipeline(self):
        """
        Runs the full trading pipeline:
        1. Collect Data
        2. Generate Predictions
        3. Analyze Signals (Agent)
        """
        logger.info("🚀 Starting Trading Pipeline...")
        
        # 1. Collect Data
        self.run_data_collection()
        
        # 2. Generate Predictions
        if not self.run_prediction_generation():
            logger.error("Skipping analysis due to prediction failure")
            return

        # 3. Analyze Signals / Run Agent
        # We run analyze_signals.py which handles the analysis part
        self.run_signal_analysis()
        
        # Optional: Run run_agent.py for CLI output logging
        # self.run_agent_reporting() 
        
        logger.info("✅ Trading Pipeline completed")

    def start(self):
        logger.info("Initializing Crypto Automation Scheduler...")
        
        # 1. Main Trading Pipeline (Data -> Predict -> Analyze)
        # Driven by the faster interval (usually signal analysis)
        if self.config.get('signal_analysis', {}).get('enabled', True):
            interval_mins = self.config.get('signal_analysis', {}).get('interval_minutes', 15)
            
            self.scheduler.add_job(
                self.run_trading_pipeline,
                IntervalTrigger(minutes=interval_mins),
                id='trading_pipeline',
                name='Trading Pipeline (Collect->Predict->Analyze)',
                replace_existing=True,
                next_run_time=datetime.now() # Run immediately on start
            )
            logger.info(f"Scheduled Trading Pipeline every {interval_mins} minutes")
            
        # 2. Daily Model Training (Retraining)
        # Schedule for 04:00 AM daily
        self.scheduler.add_job(
            self.run_model_training,
            CronTrigger(hour=4, minute=0),
            id='model_training',
            name='Daily Model Training',
            replace_existing=True
        )
        logger.info("Scheduled Daily Model Training at 04:00 AM")

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
