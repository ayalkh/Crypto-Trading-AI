"""
Generate ML Predictions and Save to Database
============================================
Generates predictions using trained ML models and saves them to the database.

Based on actual crypto market behavior:
- 5m:  0.01-0.05% typical moves
- 15m: 0.02-0.10% typical moves  
- 1h:  0.05-0.20% typical moves
- 4h:  0.10-0.50% typical moves
- 1d:  0.20-1.00% typical moves
"""
import sys
import sqlite3
from datetime import datetime
import logging
import traceback

# Import your existing system
from train_models import OptimizedMLSystemV2 as OptimizedCryptoMLSystem

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

# Direction classification thresholds by timeframe
DIRECTION_THRESHOLDS = {
    '5m': 0.000016,   # 0.0016% - 50th percentile
    '15m': 0.000021,  # 0.0021% - 50th percentile
    '1h': 0.000080,   # 0.0080% - 50th percentile
    '4h': 0.000443,   # 0.0443% - 50th percentile
    '1d': 0.000894    # 0.0894% - 50th percentile
}

def classify_direction(price_change_pct: float, timeframe: str) -> tuple:
    """
    Classify direction with REALISTIC thresholds for crypto markets
    
    Args:
        price_change_pct: Predicted price change as decimal (e.g., 0.0001 = 0.01%)
        timeframe: Timeframe string
    
    Returns:
        (direction, probability)
    """
    threshold = DIRECTION_THRESHOLDS.get(timeframe, 0.0006)
    
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
    if direction == 'NEUTRAL':
        # Neutral has lower probability (50-60%)
        probability = 0.50 + min((magnitude / threshold) * 0.10, 0.10)
    else:
        # Directional predictions get higher probability (55-90%)
        # based on how much they exceed threshold
        excess_ratio = (magnitude - threshold) / threshold
        probability = 0.55 + min(excess_ratio * 0.20, 0.35)
    
    # Log the classification decision
    logger.debug(f"   Classify: {price_change_pct*100:+.4f}% vs {threshold*100:.4f}% threshold → {direction} ({probability:.1%})")
    
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
    Save individual model predictions to database
    """
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Get individual model predictions
        model_predictions = prediction.get('model_predictions', {})
        
        if not model_predictions:
            logger.warning(f"   ⚠️ No individual model predictions, using ensemble")
            
            # Fallback: save ensemble prediction
            predicted_price = prediction.get('predicted_price')
            
            # Ensemble stores price_change_pct as percentage, convert to decimal
            price_change_pct_percentage = prediction.get('price_change_pct', 0)
            price_change_pct = price_change_pct_percentage / 100.0
            
            logger.info(f"   📊 Ensemble: {price_change_pct_percentage:+.4f}% (decimal: {price_change_pct:+.6f})")
            
            # Use realistic classification
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
                price_change_pct,  # Store as decimal
                confidence
            ))
            
            logger.info(f"   💾 Saved ENSEMBLE: {direction} ({direction_prob:.0%}), change: {price_change_pct*100:+.4f}%")
            
            conn.commit()
            conn.close()
            return True
        
        saved_count = 0
        direction_distribution = {'UP': 0, 'DOWN': 0, 'NEUTRAL': 0}
        
        # Loop through each model's prediction
        for model_name, model_pred in model_predictions.items():
            # Extract values
            predicted_price = model_pred.get('predicted_price')
            
            # Individual models store price_change_pct as decimal
            price_change_pct = model_pred.get('price_change_pct', 0)
            
            logger.info(f"   📊 {model_name.upper():9}: raw prediction = {price_change_pct:+.6f} ({price_change_pct*100:+.4f}%)")
            
            # Classification with timeframe-specific thresholds
            direction, direction_prob = classify_direction(price_change_pct, timeframe)
            direction_distribution[direction] += 1
            
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
                price_change_pct,  # Store as decimal
                confidence
            ))
            
            logger.info(f"   💾 {model_name.upper():9}: {direction:7} ({direction_prob:.0%}), change: {price_change_pct*100:+.4f}%")
            saved_count += 1
        
        # Log distribution for this symbol/timeframe
        total = sum(direction_distribution.values())
        if total > 0:
            logger.info(f"   📊 Distribution: UP={direction_distribution['UP']}/{total}, "
                       f"DOWN={direction_distribution['DOWN']}/{total}, "
                       f"NEUTRAL={direction_distribution['NEUTRAL']}/{total}")
        
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
            SELECT predicted_direction, COUNT(*), AVG(direction_probability), 
                   AVG(predicted_change_pct), MIN(predicted_change_pct), MAX(predicted_change_pct)
            FROM ml_predictions 
            WHERE timestamp > datetime('now', '-1 hour')
            GROUP BY predicted_direction
            ORDER BY predicted_direction
        """)
        
        dir_stats = cursor.fetchall()
        
        # By model type
        cursor.execute("""
            SELECT model_type, COUNT(*), AVG(direction_probability),
                   AVG(predicted_change_pct)
            FROM ml_predictions 
            WHERE timestamp > datetime('now', '-1 hour')
            GROUP BY model_type
            ORDER BY model_type
        """)
        
        model_stats = cursor.fetchall()
        
        # By timeframe
        cursor.execute("""
            SELECT timeframe, predicted_direction, COUNT(*)
            FROM ml_predictions 
            WHERE timestamp > datetime('now', '-1 hour')
            GROUP BY timeframe, predicted_direction
            ORDER BY timeframe, predicted_direction
        """)
        
        tf_stats = cursor.fetchall()
        
        conn.close()
        
        logger.info(f"\n" + "="*70)
        logger.info(f"📊 VERIFICATION RESULTS")
        logger.info(f"="*70)
        logger.info(f"\nTotal predictions: {count}")
        
        if count > 0:
            logger.info(f"\n📈 Direction Distribution:")
            for direction, dir_count, avg_prob, avg_change, min_change, max_change in dir_stats:
                pct = (dir_count / count) * 100
                logger.info(f"  {direction:8}: {dir_count:3} ({pct:5.1f}%) | "
                          f"avg prob: {avg_prob:.1%} | "
                          f"avg change: {avg_change*100:+.4f}% | "
                          f"range: [{min_change*100:+.4f}%, {max_change*100:+.4f}%]")
            
            logger.info(f"\n🤖 By Model:")
            for model, model_count, avg_prob, avg_change in model_stats:
                logger.info(f"  {model:9}: {model_count:3} predictions | "
                          f"avg prob: {avg_prob:.1%} | "
                          f"avg change: {avg_change*100:+.4f}%")
            
            logger.info(f"\n⏱️  By Timeframe:")
            current_tf = None
            for tf, direction, tf_count in tf_stats:
                if tf != current_tf:
                    logger.info(f"  {tf}:")
                    current_tf = tf
                logger.info(f"    {direction:8}: {tf_count:2}")
        
        # Check if we have good variety
        neutral_pct = 0
        for direction, dir_count, _, _, _, _ in dir_stats:
            if direction == 'NEUTRAL':
                neutral_pct = (dir_count / count) * 100
        
        logger.info(f"\n" + "="*70)
        if neutral_pct > 75:
            logger.warning(f"⚠️  Still {neutral_pct:.0f}% NEUTRAL - thresholds may need MORE tuning")
            logger.warning(f"💡 Consider lowering thresholds by 30-50%")
        elif neutral_pct < 30:
            logger.warning(f"⚠️  Only {neutral_pct:.0f}% NEUTRAL - thresholds may be TOO aggressive")
            logger.warning(f"💡 Consider raising thresholds by 20-30%")
        else:
            logger.info(f"✅ Good balance: {neutral_pct:.0f}% NEUTRAL, {100-neutral_pct:.0f}% directional")
            logger.info(f"🎯 Target range: 40-60% NEUTRAL")
        
        return count > 0
        
    except Exception as e:
        logger.error(f"❌ Verification error: {e}")
        return False


def main():
    """Generate and save predictions for all symbols"""
    
    print("\n" + "="*70)
    print("🔮 GENERATE ML PREDICTIONS")
    print("="*70)
    print("\n📊 Configuration:")
    print("  • Timeframe-specific thresholds for crypto markets")
    print("  • Target: 40-60% directional signals, 40-60% NEUTRAL")
    print("\n📊 Current Thresholds:")
    for tf, thresh in DIRECTION_THRESHOLDS.items():
        print(f"  • {tf:3}: {thresh*100:.3f}%")
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
    logger.info("\nSTEP 3: Generating predictions...")
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
            
            threshold_pct = DIRECTION_THRESHOLDS[timeframe] * 100
            logger.info(f"\n🔮 {symbol} {timeframe} (threshold: {threshold_pct:.3f}%):")
            
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
                logger.info(f"   📈 Current price: ${prediction.get('current_price', 0):,.2f}")
                logger.info(f"   📈 Predicted price: ${prediction.get('predicted_price', 0):,.2f}")
                logger.info(f"   📊 Ensemble change: {prediction.get('price_change_pct', 0):+.4f}%")
                
                # Save to database (with fixed classification)
                if save_prediction('data/ml_crypto_data.db', symbol, timeframe, prediction):
                    total_saved += 1
                    logger.info(f"   ✅ Saved successfully")
                else:
                    errors.append(f"{symbol} {timeframe}: Database save failed")
            
            except Exception as e:
                logger.error(f"   ❌ Error: {e}")
                logger.error(traceback.format_exc())
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
        print("\n✅ SUCCESS! Predictions saved")
        print("\n💡 Next steps:")
        print("   1. Run the trading agent to analyze opportunities")
        print("   2. Check predictions in the database")
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