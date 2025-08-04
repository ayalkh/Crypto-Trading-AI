#!/usr/bin/env python3
"""
Initial ML Model Training Script
Train your first AI models on collected crypto data
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
import logging
from datetime import datetime

# Suppress TensorFlow warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our ML system
try:
    from ml_integration_system import CryptoMLSystem
    ML_AVAILABLE = True
except ImportError as e:
    print(f"❌ Cannot import ML system: {e}")
    ML_AVAILABLE = False

def setup_logging():
    """Setup logging for training"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ml_logs/training.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def print_banner():
    """Print training banner"""
    print("\n" + "🎓" * 20)
    print("🎓" + " " * 36 + "🎓")
    print("🎓    ML MODEL TRAINING FOR CRYPTO AI    🎓")
    print("🎓" + " " * 36 + "🎓")
    print("🎓" * 20 + "\n")

def check_data_availability(ml_system, symbols, timeframes):
    """Check if we have enough data for training"""
    print("📊 Checking data availability...")
    
    data_status = {}
    
    for symbol in symbols:
        data_status[symbol] = {}
        for timeframe in timeframes:
            df = ml_system.load_data(symbol, timeframe, days_back=30)
            data_count = len(df)
            data_status[symbol][timeframe] = data_count
            
            if data_count >= 100:
                status = "✅ Good"
            elif data_count >= 50:
                status = "⚠️ Limited"
            else:
                status = "❌ Insufficient"
            
            print(f"   {symbol} {timeframe}: {data_count} records - {status}")
    
    return data_status

def train_models_for_symbol(ml_system, symbol, timeframes):
    """Train all models for a specific symbol"""
    print(f"\n🎯 Training models for {symbol}")
    print("=" * 50)
    
    results = {
        'symbol': symbol,
        'timeframes': {},
        'total_models': 0,
        'successful_models': 0,
        'failed_models': 0
    }
    
    for timeframe in timeframes:
        print(f"\n📈 Training {symbol} {timeframe} models...")
        
        timeframe_results = {
            'price_prediction': False,
            'direction_prediction': False,
            'lstm_model': False,
            'training_time': None
        }
        
        start_time = datetime.now()
        
        # Train price prediction models
        print("   🔢 Training price prediction models...")
        try:
            if ml_system.train_price_prediction_models(symbol, timeframe):
                print("      ✅ Price prediction models trained successfully")
                timeframe_results['price_prediction'] = True
                results['successful_models'] += 1
            else:
                print("      ❌ Price prediction training failed")
                results['failed_models'] += 1
        except Exception as e:
            print(f"      ❌ Price prediction error: {str(e)[:100]}...")
            results['failed_models'] += 1
        
        results['total_models'] += 1
        
        # Train direction prediction models
        print("   🎯 Training direction prediction models...")
        try:
            if ml_system.train_direction_prediction_models(symbol, timeframe):
                print("      ✅ Direction prediction models trained successfully")
                timeframe_results['direction_prediction'] = True
                results['successful_models'] += 1
            else:
                print("      ❌ Direction prediction training failed")
                results['failed_models'] += 1
        except Exception as e:
            print(f"      ❌ Direction prediction error: {str(e)[:100]}...")
            results['failed_models'] += 1
        
        results['total_models'] += 1
        
        # Train LSTM model (this takes the longest)
        print("   🧠 Training LSTM model (this may take 10-30 minutes)...")
        try:
            if ml_system.train_lstm_model(symbol, timeframe):
                print("      ✅ LSTM model trained successfully")
                timeframe_results['lstm_model'] = True
                results['successful_models'] += 1
            else:
                print("      ❌ LSTM training failed")
                results['failed_models'] += 1
        except Exception as e:
            print(f"      ❌ LSTM error: {str(e)[:100]}...")
            results['failed_models'] += 1
        
        results['total_models'] += 1
        
        # Calculate training time
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        timeframe_results['training_time'] = training_time
        
        print(f"   ⏱️ Training completed in {training_time:.1f} seconds")
        
        results['timeframes'][timeframe] = timeframe_results
    
    return results

def print_training_summary(all_results):
    """Print a comprehensive training summary"""
    print("\n" + "📊" * 20)
    print("📊" + " " * 36 + "📊")
    print("📊         TRAINING SUMMARY         📊")
    print("📊" + " " * 36 + "📊")
    print("📊" * 20 + "\n")
    
    total_models = 0
    total_successful = 0
    total_failed = 0
    
    for result in all_results:
        symbol = result['symbol']
        successful = result['successful_models']
        failed = result['failed_models']
        total = result['total_models']
        
        total_models += total
        total_successful += successful
        total_failed += failed
        
        success_rate = (successful / total * 100) if total > 0 else 0
        
        print(f"🪙 {symbol}:")
        print(f"   Models trained: {total}")
        print(f"   Successful: {successful}")
        print(f"   Failed: {failed}")
        print(f"   Success rate: {success_rate:.1f}%")
        
        # Show timeframe details
        for timeframe, tf_data in result['timeframes'].items():
            models_status = []
            if tf_data['price_prediction']:
                models_status.append("💹 Price")
            if tf_data['direction_prediction']:
                models_status.append("🎯 Direction")
            if tf_data['lstm_model']:
                models_status.append("🧠 LSTM")
            
            time_str = f"{tf_data['training_time']:.1f}s" if tf_data['training_time'] else "N/A"
            print(f"   {timeframe}: {', '.join(models_status)} ({time_str})")
        
        print()
    
    # Overall summary
    overall_success_rate = (total_successful / total_models * 100) if total_models > 0 else 0
    
    print("🎯 OVERALL RESULTS:")
    print(f"   Total models: {total_models}")
    print(f"   Successful: {total_successful}")
    print(f"   Failed: {total_failed}")
    print(f"   Success rate: {overall_success_rate:.1f}%")
    
    if overall_success_rate >= 80:
        print("🎉 Excellent! Your AI models are ready to trade!")
    elif overall_success_rate >= 60:
        print("✅ Good! Most models trained successfully.")
    elif overall_success_rate >= 40:
        print("⚠️ Partial success. Some models may need more data.")
    else:
        print("❌ Many models failed. Check your data quality and try again.")
    
    # Show what's ready
    print(f"\n📁 Trained models saved in: ml_models/")
    print(f"📝 Training logs saved in: ml_logs/training.log")

def main():
    """Main training function"""
    print_banner()
    
    if not ML_AVAILABLE:
        print("❌ ML system not available. Please check your installation.")
        input("Press Enter to exit...")
        return
    
    # Setup logging
    setup_logging()
    
    # Configuration
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    timeframes = ['1h', '4h', '1d']
    
    print("🚀 INITIAL MODEL TRAINING")
    print("=" * 40)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Timeframes: {', '.join(timeframes)}")
    print(f"Total models to train: {len(symbols) * len(timeframes) * 3}")
    print("=" * 40)
    
    # Initialize ML system
    print("\n🔧 Initializing ML system...")
    try:
        ml_system = CryptoMLSystem()
        print("✅ ML system initialized")
    except Exception as e:
        print(f"❌ Failed to initialize ML system: {e}")
        input("Press Enter to exit...")
        return
    
    # Check data availability
    data_status = check_data_availability(ml_system, symbols, timeframes)
    
    # Ask user if they want to continue
    print("\n🤔 Do you want to proceed with training?")
    print("Note: This may take 30 minutes to 2 hours depending on your data.")
    choice = input("Continue? (y/n): ").lower().strip()
    
    if choice not in ['y', 'yes']:
        print("Training cancelled.")
        return
    
    # Start training
    print("\n🎓 Starting model training...")
    print("⏰ This will take a while. Grab a coffee! ☕")
    
    all_results = []
    start_time = datetime.now()
    
    for symbol in symbols:
        try:
            result = train_models_for_symbol(ml_system, symbol, timeframes)
            all_results.append(result)
        except Exception as e:
            print(f"❌ Failed to train models for {symbol}: {e}")
            # Continue with next symbol
            continue
    
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    # Print summary
    print_training_summary(all_results)
    
    print(f"\n⏰ Total training time: {total_time/60:.1f} minutes")
    print(f"📅 Training completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n🚀 Next steps:")
    print("1. Start the ML automation: python start_ml_automation.py")
    print("2. Monitor the predictions and alerts")
    print("3. Check ml_predictions/ for detailed results")
    
    print("\n🧠 Your AI trading models are now ready!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Training cancelled by user")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        logging.error(f"Training failed: {e}")
        input("Press Enter to exit...")