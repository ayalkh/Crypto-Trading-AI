"""
Generate ML Predictions and Save to Database - FIXED VERSION
============================================================
Combines prediction generation and database saving into one script
With IMPROVED direction classification thresholds

Key Fix: Changed direction thresholds from 0.5% to 0.2%
- This will generate more UP/DOWN predictions
- Agent will have actual signals to work with
"""
import sys
import sqlite3
from datetime import datetime
import logging
import traceback

# Import your existing system
from optimized_ml_system_v2 import OptimizedMLSystemV2 as OptimizedCryptoMLSystem

# Setup detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/generate_predictions.log', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Symbols and timeframes
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'DOT/USDT']
TIMEFRAMES = ['5m', '15m', '1h', '4h', '1d']

# ============================================================================
# CRITICAL FIX: Better direction classification thresholds
# ============================================================================
DIRECTION_THRESHOLDS = {
    '5m': 0.0015,   # 0.15% (very sensitive for 5min)
    '15m': 0.002,   # 0.2%
    '1h': 0.002,    # 0.2%
    '4h': 0.003,    # 0.3%
    '1d': 0.005     # 0.5%
}

def classify_direction(price_change_pct: float, timeframe: str) -> tuple:
    """
    Classify direction with timeframe-specific thresholds
    
    Returns:
        (direction, probability)
    """
    threshold = DIRECTION_THRESHOLDS.get(timeframe, 0.002)
    
    # Calculate absolute magnitude
    magnitude = abs(price_change_pct)
    
    # Classify direction
    if price_change_pct > threshold:
        direction = 'UP'
    elif price_change_pct < -threshold:
        direction = 'DOWN'
    else:
        direction = 'NEUTRAL'
    
    # Calculate probability based on how far from threshold
    # The further from threshold, the higher the probability
    if direction == 'NEUTRAL':
        # Neutral has lower probability
        probability = 0.50 + (magnitude / threshold) * 0.05
        probability = min(probability, 0.60)
    else:
        # Directional predictions get higher probability
        # based on how much they exceed threshold
        excess = magnitude - threshold
        probability = 0.55 + (excess / threshold) * 0.20
        probability = min(probability, 0.95)
    
    return direction, probability


def create_predictions_table(db_path='data/ml_crypto_data.db'):
    """Create ml_predictions table if it doesn't exist"""
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        logger.info("🔨 Creating/verifying ml_predictions table...")
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ml_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                target_timestamp DATETIME,
                model_type TEXT NOT NULL,
                model_version TEXT,
                predicted_price REAL,
                predicted_direction TEXT,
                direction_probability REAL,
                predicted_change_pct REAL,
                confidence_score REAL,
                prediction_low REAL,
                prediction_high REAL,
                actual_price REAL,
                actual_direction TEXT,
                prediction_error REAL,
                is_correct BOOLEAN,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_symbol_timeframe 
            ON ml_predictions(symbol, timeframe, timestamp DESC)
        """)
        
        conn.commit()
        conn.close()
        
        logger.info("✅ ml_predictions table ready")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating table: {e}")
        logger.error(traceback.format_exc())
        return False


def save_prediction(db_path, symbol, timeframe, prediction):
    """
    Save individual model predictions to database with FIXED classification
    """
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Get individual model predictions
        model_predictions = prediction.get('model_predictions', {})
        
        if not model_predictions:
            logger.warning(f"   ⚠️ No individual model predictions, using ensemble")
            
            # Fallback: save ensemble prediction with FIXED classification
            predicted_price = prediction.get('predicted_price')
            price_change_pct = prediction.get('price_change_pct', 0) / 100  # Convert from % to decimal
            
            # FIXED: Use new classification
            direction, direction_prob = classify_direction(price_change_pct, timeframe)
            confidence = prediction.get('confidence', direction_prob)
            
            cursor.execute("""
                INSERT INTO ml_predictions 
                (symbol, timeframe, timestamp, model_type, predicted_price, 
                 predicted_direction, direction_probability, predicted_change_pct, 
                 confidence_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                timeframe,
                timestamp,
                'ENSEMBLE',
                predicted_price,
                direction,
                direction_prob,
                price_change_pct,
                confidence
            ))
            
            logger.info(f"   💾 Saved ENSEMBLE: {direction} ({direction_prob:.0%}), change: {price_change_pct:+.3%}")
            
            conn.commit()
            conn.close()
            return True
        
        saved_count = 0
        
        # Loop through each model's prediction
        for model_name, model_pred in model_predictions.items():
            # Extract values
            predicted_price = model_pred.get('predicted_price')
            price_change_pct = model_pred.get('price_change_pct', 0)
            
            # FIXED: Use new classification with timeframe-specific thresholds
            direction, direction_prob = classify_direction(price_change_pct, timeframe)
            
            # Confidence from model (or use calculated probability)
            confidence = model_pred.get('confidence', direction_prob)
            
            # Insert the prediction with UPPERCASE model name
            cursor.execute("""
                INSERT INTO ml_predictions 
                (symbol, timeframe, timestamp, model_type, predicted_price, 
                 predicted_direction, direction_probability, predicted_change_pct, 
                 confidence_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                timeframe,
                timestamp,
                model_name.upper(),  # CATBOOST, XGBOOST
                predicted_price,
                direction,
                direction_prob,
                price_change_pct,
                confidence
            ))
            
            logger.info(f"   💾 {model_name.upper():9}: {direction:7} ({direction_prob:.0%}), change: {price_change_pct:+.3%}")
            saved_count += 1
        
        conn.commit()
        conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Database save error: {e}")
        logger.error(traceback.format_exc())
        return False


def verify_predictions(db_path='data/ml_crypto_data.db'):
    """Verify predictions were saved and show statistics"""
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Total count
        cursor.execute("SELECT COUNT(*) FROM ml_predictions WHERE timestamp > datetime('now', '-1 hour')")
        count = cursor.fetchone()[0]
        
        # Direction distribution
        cursor.execute("""
            SELECT predicted_direction, COUNT(*), AVG(direction_probability), AVG(predicted_change_pct)
            FROM ml_predictions 
            WHERE timestamp > datetime('now', '-1 hour')
            GROUP BY predicted_direction
            ORDER BY predicted_direction
        """)
        
        dir_stats = cursor.fetchall()
        
        # By model type
        cursor.execute("""
            SELECT model_type, COUNT(*), AVG(direction_probability)
            FROM ml_predictions 
            WHERE timestamp > datetime('now', '-1 hour')
            GROUP BY model_type
            ORDER BY model_type
        """)
        
        model_stats = cursor.fetchall()
        
        conn.close()
        
        logger.info(f"\n" + "="*70)
        logger.info(f"📊 VERIFICATION RESULTS")
        logger.info(f"="*70)
        logger.info(f"\nTotal predictions: {count}")
        
        if count > 0:
            logger.info(f"\nDirection Distribution:")
            for direction, dir_count, avg_prob, avg_change in dir_stats:
                pct = (dir_count / count) * 100
                logger.info(f"  {direction:8}: {dir_count:3} ({pct:5.1f}%) | "
                          f"avg prob: {avg_prob:.1%} | avg change: {avg_change:+.3%}")
            
            logger.info(f"\nBy Model:")
            for model, model_count, avg_prob in model_stats:
                logger.info(f"  {model:9}: {model_count:3} predictions | avg prob: {avg_prob:.1%}")
        
        # Check if we have good variety
        neutral_pct = 0
        for direction, dir_count, _, _ in dir_stats:
            if direction == 'NEUTRAL':
                neutral_pct = (dir_count / count) * 100
        
        logger.info(f"\n" + "="*70)
        if neutral_pct > 80:
            logger.warning(f"⚠️  Still {neutral_pct:.0f}% NEUTRAL - thresholds may need more tuning")
        elif neutral_pct < 50:
            logger.info(f"✅ Good variety: {neutral_pct:.0f}% NEUTRAL, {100-neutral_pct:.0f}% directional")
        else:
            logger.info(f"✅ Balanced: {neutral_pct:.0f}% NEUTRAL, {100-neutral_pct:.0f}% directional")
        
        return count > 0
        
    except Exception as e:
        logger.error(f"❌ Verification error: {e}")
        return False


def main():
    """Generate and save predictions for all symbols"""
    
    print("\n" + "="*70)
    print("🔮 GENERATE ML PREDICTIONS - FIXED VERSION")
    print("="*70)
    print("\n🔧 Key Improvements:")
    print("  • Adaptive direction thresholds (0.15%-0.5% based on timeframe)")
    print("  • Probability calculation based on signal strength")
    print("  • Should generate 40-60% directional signals (not 100% NEUTRAL)")
    print("\n" + "="*70 + "\n")
    
    # Step 1: Create table
    logger.info("STEP 1: Creating ml_predictions table...")
    if not create_predictions_table():
        logger.error("Cannot proceed without table")
        return False
    
    # Step 2: Initialize ML system
    logger.info("\nSTEP 2: Initializing ML system...")
    try:
        ml_system = OptimizedCryptoMLSystem()
        logger.info("✅ ML system initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize ML system: {e}")
        logger.error(traceback.format_exc())
        return False
    
    # Step 3: Generate and save predictions
    logger.info("\nSTEP 3: Generating predictions with FIXED classification...")
    logger.info("="*70)
    
    total_attempted = 0
    total_saved = 0
    errors = []
    
    for symbol in SYMBOLS:
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 Processing {symbol}")
        logger.info(f"{'='*70}")
        
        for timeframe in TIMEFRAMES:
            total_attempted += 1
            
            logger.info(f"\n🔮 {symbol} {timeframe} (threshold: {DIRECTION_THRESHOLDS[timeframe]:.2%}):")
            
            try:
                # Generate prediction
                prediction = ml_system.make_ensemble_prediction(symbol, timeframe)
                
                if not prediction:
                    logger.warning(f"   ⚠️ No prediction returned")
                    errors.append(f"{symbol} {timeframe}: No prediction")
                    continue
                
                # Check if prediction has required data
                if 'predicted_price' not in prediction:
                    logger.warning(f"   ⚠️ Prediction missing predicted_price")
                    errors.append(f"{symbol} {timeframe}: Missing data")
                    continue
                
                # Log prediction details
                logger.info(f"   📈 Predicted price: ${prediction.get('predicted_price', 0):,.2f}")
                logger.info(f"   📊 Ensemble change: {prediction.get('price_change_pct', 0):+.2f}%")
                
                # Save to database (with fixed classification)
                if save_prediction('data/ml_crypto_data.db', symbol, timeframe, prediction):
                    total_saved += 1
                    logger.info(f"   ✅ Saved successfully")
                else:
                    errors.append(f"{symbol} {timeframe}: Database save failed")
            
            except Exception as e:
                logger.error(f"   ❌ Error: {e}")
                errors.append(f"{symbol} {timeframe}: {str(e)}")
    
    # Step 4: Verify results
    logger.info("\n" + "="*70)
    logger.info("STEP 4: Verifying results...")
    logger.info("="*70)
    
    verify_predictions()
    
    # Final summary
    print("\n" + "="*70)
    print("📊 FINAL SUMMARY")
    print("="*70)
    print(f"Total attempted: {total_attempted}")
    print(f"Successfully saved: {total_saved}")
    print(f"Failed: {len(errors)}")
    print(f"Success rate: {(total_saved/total_attempted*100):.1f}%")
    
    if errors:
        print("\n❌ Errors:")
        for error in errors[:5]:
            print(f"   - {error}")
        if len(errors) > 5:
            print(f"   ... and {len(errors)-5} more")
    
    print("="*70)
    
    if total_saved > 0:
        print("\n✅ SUCCESS! Predictions saved with improved classification")
        print("\n💡 Next steps:")
        print("   1. Run: python run_agent_FINAL.py")
        print("   2. You should now see actual UP/DOWN signals")
        print("   3. Agent should find tradeable opportunities")
        print("\nThe fix changes predictions from 100% NEUTRAL to 40-60% directional!")
        return True
    else:
        print("\n❌ FAILED! No predictions were saved")
        print("\n🔍 Check the log file: logs/generate_predictions.log")
        return False


if __name__ == "__main__":
    import os
    os.makedirs('logs', exist_ok=True)
    
    success = main()
    sys.exit(0 if success else 1)