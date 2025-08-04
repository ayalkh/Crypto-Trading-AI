"""
ML-Enhanced 24/7 Crypto Trading Automation System
Integrates machine learning with your existing data collection and analysis
"""
import os
import sys

if sys.platform.startswith('win'):
    try:
        # Try to set UTF-8 encoding for stdout
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, OSError):
        # If reconfigure doesn't work, try alternative
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')
import schedule
import time
import threading
import logging
import json
import subprocess
from datetime import datetime, timedelta

# Import our ML components
try:
    from ml_integration_system import CryptoMLSystem
    from ml_enhanced_analyzer import MLEnhancedAnalyzer
    ML_INTEGRATION_AVAILABLE = True
except ImportError as e:
    ML_INTEGRATION_AVAILABLE = False
    logging.warning(f"⚠️ ML integration not available: {e}")

# Import the base automation system
from enhanced_automation_scheduler import EnhancedTradingAutomation

class MLEnhancedTradingAutomation(EnhancedTradingAutomation):
    def __init__(self, config_file='automation_config.json'):
        """Initialize ML-enhanced automation system"""
        super().__init__(config_file)
        
        # Initialize ML components
        if ML_INTEGRATION_AVAILABLE:
            self.ml_system = CryptoMLSystem(self.config['system']['database_path'])
            self.ml_analyzer = MLEnhancedAnalyzer(self.config['system']['database_path'])
        else:
            self.ml_system = None
            self.ml_analyzer = None
        
        # ML-specific configuration
        self.ml_config = self.config.get('machine_learning', {})
        
        logging.info("🧠 ML-Enhanced Trading Automation System initialized")
    
    def load_config(self):
        """Load configuration with ML settings"""
        config = super().load_config()
        
        # Add ML-specific default configuration
        ml_defaults = {
            "machine_learning": {
                "enabled": True,
                "model_training": {
                    "enabled": True,
                    "interval_hours": 24,  # Retrain models daily
                    "symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT"],
                    "timeframes": ["1h", "4h", "1d"]
                },
                "prediction": {
                    "enabled": True,
                    "interval_minutes": 30,  # Make predictions every 30 minutes
                    "confidence_threshold": 0.6,
                    "save_predictions": True
                },
                "model_types": {
                    "traditional_ml": True,  # Random Forest, XGBoost, etc.
                    "deep_learning": True,   # LSTM, CNN, etc.
                    "ensemble": True         # Combine multiple models
                },
                "alerts": {
                    "high_confidence_threshold": 0.8,
                    "significant_change_threshold": 0.05  # 5% predicted change
                }
            }
        }
        
        # Merge ML defaults with existing config
        if 'machine_learning' not in config:
            config['machine_learning'] = ml_defaults['machine_learning']
        else:
            # Deep merge ML config
            for key, value in ml_defaults['machine_learning'].items():
                if key not in config['machine_learning']:
                    config['machine_learning'][key] = value
        
        return config
    
    def setup_schedule(self):
        """Set up the automation schedule including ML tasks"""
        # Run base scheduling
        super().setup_schedule()
        
        if not ML_INTEGRATION_AVAILABLE:
            logging.warning("⚠️ ML integration not available, skipping ML scheduling")
            return
        
        ml_config = self.config['machine_learning']
        
        # Model training schedule
        if ml_config['model_training']['enabled']:
            interval_hours = ml_config['model_training']['interval_hours']
            schedule.every(interval_hours).hours.do(
                self.safe_run_job, self.ml_model_training_job, "ML Model Training"
            )
            logging.info(f"📅 Scheduled: ML model training every {interval_hours} hours")
        
        # Prediction schedule
        if ml_config['prediction']['enabled']:
            interval_minutes = ml_config['prediction']['interval_minutes']
            schedule.every(interval_minutes).minutes.do(
                self.safe_run_job, self.ml_prediction_job, "ML Predictions"
            )
            logging.info(f"📅 Scheduled: ML predictions every {interval_minutes} minutes")
        
        # Weekly model evaluation
        schedule.every().sunday.at("02:00").do(
            self.safe_run_job, self.ml_model_evaluation_job, "ML Model Evaluation"
        )
        logging.info("📅 Scheduled: Weekly ML model evaluation on Sundays at 2:00 AM")
    
    def safe_run_job(self, job_func, job_name):
        """Safely run a job with error handling"""
        try:
            logging.info(f"🔄 Starting {job_name}...")
            result = job_func()
            if result:
                logging.info(f"✅ {job_name} completed successfully")
            else:
                logging.warning(f"⚠️ {job_name} completed with issues")
        except Exception as e:
            logging.error(f"❌ {job_name} failed: {e}")
            self.error_count += 1
            self.send_alert(f"{job_name} Failed", f"Error: {str(e)}", "ERROR")
    
    def ml_model_training_job(self):
        """Scheduled job for ML model training"""
        try:
            logging.info("🎓 Starting ML model training...")
            
            if not self.ml_system:
                logging.error("❌ ML system not available")
                return False
            
            ml_config = self.config['machine_learning']['model_training']
            symbols = ml_config['symbols']
            timeframes = ml_config['timeframes']
            
            training_results = {
                'timestamp': datetime.now().isoformat(),
                'symbols_trained': 0,
                'models_trained': 0,
                'failures': []
            }
            
            for symbol in symbols:
                try:
                    logging.info(f"🎓 Training models for {symbol}")
                    
                    for timeframe in timeframes:
                        # Train traditional ML models
                        if self.config['machine_learning']['model_types']['traditional_ml']:
                            if self.ml_system.train_price_prediction_models(symbol, timeframe):
                                training_results['models_trained'] += 1
                            
                            if self.ml_system.train_direction_prediction_models(symbol, timeframe):
                                training_results['models_trained'] += 1
                        
                        # Train deep learning models
                        if self.config['machine_learning']['model_types']['deep_learning']:
                            if self.ml_system.train_lstm_model(symbol, timeframe):
                                training_results['models_trained'] += 1
                    
                    training_results['symbols_trained'] += 1
                    
                except Exception as e:
                    error_msg = f"Training failed for {symbol}: {str(e)}"
                    logging.error(f"❌ {error_msg}")
                    training_results['failures'].append(error_msg)
            
            # Send training report
            report_message = f"""ML MODEL TRAINING COMPLETED

Symbols Trained: {training_results['symbols_trained']}/{len(symbols)}
Models Trained: {training_results['models_trained']}
Failures: {len(training_results['failures'])}

Status: {'✅ Success' if len(training_results['failures']) == 0 else '⚠️ Partial Success' if training_results['models_trained'] > 0 else '❌ Failed'}"""
            
            if training_results['failures']:
                report_message += f"\n\nFailures:\n" + "\n".join(training_results['failures'][:5])
            
            self.send_alert("ML Training Report", report_message, "ML_TRAINING")
            
            logging.info("✅ ML model training completed")
            return True
            
        except Exception as e:
            logging.error(f"❌ ML model training job failed: {e}")
            return False
    
    def ml_prediction_job(self):
        """Scheduled job for ML predictions"""
        try:
            logging.info("🔮 Starting ML predictions...")
            
            if not self.ml_analyzer:
                logging.error("❌ ML analyzer not available")
                return False
            
            ml_config = self.config['machine_learning']
            symbols = ml_config['model_training']['symbols']
            timeframes = ml_config['model_training']['timeframes']
            
            # Run ML-enhanced analysis
            results = self.ml_analyzer.run_full_analysis(symbols, timeframes)
            
            # Check for high-confidence signals
            self.process_ml_alerts(results)
            
            logging.info("✅ ML predictions completed")
            return True
            
        except Exception as e:
            logging.error(f"❌ ML prediction job failed: {e}")
            return False
    
    def process_ml_alerts(self, results: dict):
        """Process ML results and send alerts for significant signals"""
        try:
            ml_config = self.config['machine_learning']['alerts']
            high_conf_threshold = ml_config['high_confidence_threshold']
            change_threshold = ml_config['significant_change_threshold']
            
            alerts_sent = 0
            
            for symbol, symbol_data in results.get('symbols', {}).items():
                if 'error' in symbol_data:
                    continue
                
                recommendation = symbol_data.get('recommendation', {})
                action = recommendation.get('action', 'HOLD')
                confidence = recommendation.get('confidence', 0)
                
                # Check for high confidence signals
                if confidence >= high_conf_threshold and action != 'HOLD':
                    # Check ML predictions for significant price changes
                    ml_predictions = symbol_data.get('ml_predictions', {})
                    significant_change = False
                    
                    for timeframe, predictions in ml_predictions.items():
                        if 'price_change' in predictions:
                            if abs(predictions['price_change']) >= change_threshold:
                                significant_change = True
                                break
                    
                    if significant_change:
                        alert_message = f"""HIGH CONFIDENCE ML SIGNAL: {symbol}

Action: {action}
Confidence: {confidence:.2f}
Risk Level: {recommendation.get('risk_level', 'UNKNOWN')}

Reasoning: {', '.join(recommendation.get('reasoning', []))}

Timeframe Analysis:"""
                        
                        for tf, pred in ml_predictions.items():
                            if 'predicted_change' in pred:
                                change_pct = pred['predicted_change'] * 100
                                alert_message += f"\n  {tf}: {change_pct:+.2f}% predicted"
                        
                        self.send_alert(f"ML Signal: {symbol} {action}", alert_message, "ML_SIGNAL")
                        alerts_sent += 1
            
            if alerts_sent > 0:
                logging.info(f"🚨 Sent {alerts_sent} ML-based alerts")
            
        except Exception as e:
            logging.error(f"❌ Error processing ML alerts: {e}")
    
    def ml_model_evaluation_job(self):
        """Weekly job to evaluate model performance"""
        try:
            logging.info("📊 Starting ML model evaluation...")
            
            if not self.ml_system:
                return False
            
            # Get feature importance for main models
            symbols = self.config['machine_learning']['model_training']['symbols']
            evaluation_report = "ML MODEL EVALUATION REPORT\n" + "="*40 + "\n"
            
            for symbol in symbols:
                evaluation_report += f"\n{symbol}:\n"
                
                # Get feature importance
                importance = self.ml_system.get_feature_importance(symbol, '1h')
                if importance:
                    evaluation_report += "  Top 5 Features:\n"
                    for i, (feature, imp) in enumerate(list(importance.items())[:5]):
                        evaluation_report += f"    {i+1}. {feature}: {imp:.4f}\n"
                else:
                    evaluation_report += "  No feature importance data available\n"
            
            self.send_alert("Weekly ML Evaluation", evaluation_report, "ML_EVALUATION")
            
            logging.info("✅ ML model evaluation completed")
            return True
            
        except Exception as e:
            logging.error(f"❌ ML model evaluation failed: {e}")
            return False
    
    def run_signal_analysis(self):
        """Enhanced signal analysis using ML - avoid subprocess encoding issues"""
        try:
            logging.info("🔍 Starting enhanced signal analysis with ML...")
            
            # Instead of subprocess, try to run original analysis directly if possible
            original_result = True  # Assume success for now
            
            try:
                # Try to run the original analyzer directly if it exists
                if os.path.exists('multi_timeframe_analyzer.py'):
                    # Use subprocess with better encoding handling
                    result = subprocess.run([
                        sys.executable, 'multi_timeframe_analyzer.py'
                    ], capture_output=True, text=True, timeout=300, 
                       encoding='utf-8', errors='replace',
                       env=dict(os.environ, 
                                PYTHONIOENCODING='utf-8',
                                PYTHONUTF8='1',
                                PYTHONLEGACYWINDOWSFSENCODING='0',
                                PYTHONLEGACYWINDOWSSTDIO='0'),
                       creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0)
                    
                    original_result = (result.returncode == 0)
                    if not original_result:
                        logging.warning(f"⚠️ Original analyzer failed: {result.stderr[:200] if result.stderr else 'Unknown error'}")
                else:
                    logging.info("ℹ️ Original analyzer not found, continuing with ML-only analysis")
                    
            except Exception as e:
                logging.warning(f"⚠️ Could not run original analyzer: {e}")
                original_result = True  # Continue anyway
            
            # If ML is available, also run ML analysis
            if ML_INTEGRATION_AVAILABLE and self.ml_analyzer:
                try:
                    symbols = self.config['signal_analysis']['analyze_symbols']
                    timeframes = ['15m', '1h', '4h']
                    
                    # Run ML-enhanced analysis
                    ml_results = self.ml_analyzer.run_full_analysis(symbols, timeframes)
                    
                    # Process any immediate alerts
                    self.process_ml_alerts(ml_results)
                    
                    logging.info("✅ ML-enhanced analysis completed")
                    
                except Exception as e:
                    logging.error(f"❌ ML analysis failed: {e}")
            
            return original_result
            
        except Exception as e:
            logging.error(f"❌ Enhanced signal analysis failed: {e}")
            return False
    
    def performance_tracking_job(self):
        """Enhanced performance tracking including ML metrics"""
        try:
            # Run base performance tracking
            base_result = super().performance_tracking_job()
            
            if ML_INTEGRATION_AVAILABLE:
                # Add ML-specific performance metrics
                ml_metrics = self.get_ml_performance_metrics()
                
                if ml_metrics:
                    ml_report = f"""
ML PERFORMANCE METRICS

Models Trained: {ml_metrics.get('models_trained', 0)}
Recent Predictions: {ml_metrics.get('recent_predictions', 0)}
Avg Prediction Confidence: {ml_metrics.get('avg_confidence', 0):.2f}
High Confidence Signals: {ml_metrics.get('high_confidence_signals', 0)}

ML System Status: {'🟢 Active' if ml_metrics.get('system_active', False) else '🔴 Inactive'}"""
                    
                    self.send_alert("ML Performance Report", ml_report, "ML_PERFORMANCE")
            
            return base_result
            
        except Exception as e:
            logging.error(f"❌ Enhanced performance tracking failed: {e}")
            return False
    
    def get_ml_performance_metrics(self) -> dict:
        """Get ML system performance metrics"""
        try:
            metrics = {
                'system_active': ML_INTEGRATION_AVAILABLE,
                'models_trained': 0,
                'recent_predictions': 0,
                'avg_confidence': 0,
                'high_confidence_signals': 0
            }
            
            # Count trained models
            if os.path.exists('ml_models'):
                model_files = [f for f in os.listdir('ml_models') if f.endswith('.joblib') or f.endswith('.h5')]
                metrics['models_trained'] = len(model_files)
            
            # Check recent predictions
            if os.path.exists('ml_predictions/latest_ml_analysis.json'):
                try:
                    with open('ml_predictions/latest_ml_analysis.json', 'r') as f:
                        latest_analysis = json.load(f)
                    
                    # Count predictions and calculate average confidence
                    total_confidence = 0
                    prediction_count = 0
                    high_conf_count = 0
                    
                    for symbol_data in latest_analysis.get('symbols', {}).values():
                        if 'recommendation' in symbol_data:
                            confidence = symbol_data['recommendation'].get('confidence', 0)
                            if confidence > 0:
                                total_confidence += confidence
                                prediction_count += 1
                                
                                if confidence >= 0.8:
                                    high_conf_count += 1
                    
                    if prediction_count > 0:
                        metrics['avg_confidence'] = total_confidence / prediction_count
                        metrics['recent_predictions'] = prediction_count
                        metrics['high_confidence_signals'] = high_conf_count
                        
                except Exception as e:
                    logging.warning(f"⚠️ Could not load latest predictions: {e}")
            
            return metrics
            
        except Exception as e:
            logging.error(f"❌ Error getting ML metrics: {e}")
            return {}

def main():
    """Main function with ML enhancement"""
    try:
        print("🧠 ML-ENHANCED CRYPTO TRADING AUTOMATION")
        print("=" * 60)
        print("🚀 Automated Data Collection + ML Analysis + Signal Generation")
        print("=" * 60)
        
        if not ML_INTEGRATION_AVAILABLE:
            print("⚠️ WARNING: ML integration not available")
            print("Install required packages: pip install scikit-learn tensorflow xgboost")
            print("=" * 60)
        
        # Create and start automation
        automation = MLEnhancedTradingAutomation()
        
        # Check if we should train models initially
        if len(sys.argv) > 1 and sys.argv[1] == '--train-models':
            print("🎓 Initial ML model training...")
            automation.ml_model_training_job()
            print("✅ Initial training completed")
        
        automation.start()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        if 'automation' in locals():
            automation.stop()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        logging.error(f"❌ Fatal error: {e}")

if __name__ == "__main__":
    main()