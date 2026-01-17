"""
Save ML Predictions to Database
Runs your existing ML system and saves predictions to the database
so the agent can use them
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
        logging.FileHandler('logs/save_predictions.log', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Symbols and timeframes
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'DOT/USDT']
TIMEFRAMES = ['5m', '15m', '1h', '4h', '1d']

def verify_database(db_path='data/ml_crypto_data.db'):
    """Verify database exists and check tables"""
    import os
    
    if not os.path.exists(db_path):
        logger.error(f"❌ Database not found: {db_path}")
        return False
    
    logger.info(f"✅ Database found: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check existing tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    
    logger.info(f"📊 Existing tables: {tables}")
    
    conn.close()
    return True

def create_predictions_table(db_path='data/ml_crypto_data.db'):
    """Create ml_predictions table if it doesn't exist"""
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        logger.info("🔨 Creating ml_predictions table...")
        
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
        
        # Verify table was created
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ml_predictions'")
        if cursor.fetchone():
            logger.info("✅ ml_predictions table created successfully")
        else:
            logger.error("❌ Failed to create ml_predictions table")
            return False
        
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating table: {e}")
        logger.error(traceback.format_exc())
        return False

def save_prediction(db_path, symbol, timeframe, prediction):
    """Save individual model predictions to database"""
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # NEW: Get individual model predictions
        model_predictions = prediction.get('model_predictions', {})
        
        if not model_predictions:
            logger.warning(f"   ⚠️ No individual model predictions available")
            logger.warning(f"   📋 Available keys in prediction: {list(prediction.keys())}")
            logger.warning(f"   ⚠️ Falling back to ensemble prediction")
            
            # Fallback: save ensemble prediction
            predicted_price = prediction.get('predicted_price')
            price_change_pct = prediction.get('price_change_pct', 0)
            confidence = prediction.get('confidence', 0)
            
            if price_change_pct > 0.5:
                direction = 'UP'
            elif price_change_pct < -0.5:
                direction = 'DOWN'
            else:
                direction = 'NEUTRAL'
            
            direction_confidence = prediction.get('direction_confidence', confidence)
            
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
                'ensemble',
                predicted_price,
                direction,
                direction_confidence,
                price_change_pct,
                confidence
            ))
            
            conn.commit()
            conn.close()
            return True
        
        saved_count = 0
        
        # Loop through each model's prediction
        for model_name, model_pred in model_predictions.items():
            # Extract values
            predicted_price = model_pred.get('predicted_price')
            price_change_pct = model_pred.get('price_change_pct', 0)
            confidence = model_pred.get('confidence', 0)
            
            # Determine direction from price change
            if price_change_pct > 0.005:  # > 0.5%
                direction = 'UP'
            elif price_change_pct < -0.005:  # < -0.5%
                direction = 'DOWN'
            else:
                direction = 'NEUTRAL'
            
            # Direction probability (use confidence as proxy)
            direction_prob = confidence
            
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
                model_name.upper(),  # LIGHTGBM, XGBOOST, CATBOOST, GRU
                predicted_price,
                direction,
                direction_prob,
                price_change_pct,
                confidence
            ))
            
            saved_count += 1
        
        conn.commit()
        conn.close()
        
        logger.info(f"   💾 Saved {saved_count} model predictions to database")
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Database save error: {e}")
        logger.error(traceback.format_exc())
        return False

def verify_predictions_saved(db_path='data/ml_crypto_data.db'):
    """Verify predictions were actually saved"""
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM ml_predictions")
        count = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT symbol, timeframe, COUNT(*) 
            FROM ml_predictions 
            GROUP BY symbol, timeframe
            ORDER BY symbol, timeframe
        """)
        
        breakdown = cursor.fetchall()
        
        # Get count by model type
        cursor.execute("""
            SELECT model_type, COUNT(*) 
            FROM ml_predictions 
            GROUP BY model_type
            ORDER BY model_type
        """)
        
        model_breakdown = cursor.fetchall()
        
        conn.close()
        
        logger.info(f"\n📊 Verification Results:")
        logger.info(f"   Total predictions in database: {count}")
        
        if count > 0:
            logger.info(f"\n   By model type:")
            for model_type, model_count in model_breakdown:
                logger.info(f"      {model_type}: {model_count}")
            
            logger.info(f"\n   By symbol/timeframe:")
            for symbol, timeframe, tf_count in breakdown:
                logger.info(f"      {symbol} {timeframe}: {tf_count}")
        
        return count > 0
        
    except Exception as e:
        logger.error(f"❌ Verification error: {e}")
        return False

def main():
    """Generate and save predictions for all symbols"""
    
    print("\n" + "="*70)
    print("💾 SAVING ML PREDICTIONS TO DATABASE")
    print("="*70)
    print()
    
    # Step 1: Verify database
    logger.info("STEP 1: Verifying database...")
    if not verify_database():
        logger.error("Cannot proceed without database")
        return False
    
    # Step 2: Create table
    logger.info("\nSTEP 2: Creating ml_predictions table...")
    if not create_predictions_table():
        logger.error("Cannot proceed without table")
        return False
    
    # Step 3: Initialize ML system
    logger.info("\nSTEP 3: Initializing ML system...")
    try:
        ml_system = OptimizedCryptoMLSystem()
        logger.info("✅ ML system initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize ML system: {e}")
        logger.error(traceback.format_exc())
        return False
    
    # Step 4: Generate and save predictions
    logger.info("\nSTEP 4: Generating predictions...")
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
            
            logger.info(f"\n🔮 {symbol} {timeframe}:")
            
            try:
                # Generate prediction
                logger.info(f"   🔄 Generating prediction...")
                prediction = ml_system.make_ensemble_prediction(symbol, timeframe)
                
                if not prediction:
                    logger.warning(f"   ⚠️ No prediction returned")
                    errors.append(f"{symbol} {timeframe}: No prediction")
                    continue
                
                # Check if prediction has required data
                if 'predicted_price' not in prediction or prediction['predicted_price'] is None:
                    logger.warning(f"   ⚠️ Prediction missing predicted_price")
                    logger.info(f"   📋 Prediction keys: {list(prediction.keys())}")
                    errors.append(f"{symbol} {timeframe}: Missing predicted_price")
                    continue
                
                # Log prediction details
                logger.info(f"   📈 Predicted price: ${prediction.get('predicted_price', 0):,.2f}")
                logger.info(f"   📊 Price change: {prediction.get('price_change_pct', 0):+.2f}%")
                logger.info(f"   🎯 Confidence: {prediction.get('confidence', 0):.0%}")
                
                # Check if we have model_predictions
                if 'model_predictions' in prediction:
                    logger.info(f"   🤖 Individual models: {list(prediction['model_predictions'].keys())}")
                else:
                    logger.warning(f"   ⚠️ No model_predictions found in response")
                
                # Save to database
                if save_prediction('data/ml_crypto_data.db', symbol, timeframe, prediction):
                    total_saved += 1
                    logger.info(f"   ✅ Success!")
                else:
                    errors.append(f"{symbol} {timeframe}: Database save failed")
            
            except Exception as e:
                logger.error(f"   ❌ Error: {e}")
                logger.error(traceback.format_exc())
                errors.append(f"{symbol} {timeframe}: {str(e)}")
    
    # Step 5: Verify results
    logger.info("\n" + "="*70)
    logger.info("STEP 5: Verifying results...")
    logger.info("="*70)
    
    verify_predictions_saved()
    
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
        for error in errors[:10]:  # Show first 10 errors
            print(f"   - {error}")
        if len(errors) > 10:
            print(f"   ... and {len(errors)-10} more")
    
    print("="*70)
    
    if total_saved > 0:
        print("\n✅ SUCCESS! Predictions saved to database")
        print("\n💡 Next steps:")
        print("   1. Run: python check_db.py (to verify)")
        print("   2. Run: python test_agent.py")
        print("   3. Or run: python -m crypto_agent.agent")
        print("\nThe agent should now have ML predictions available!")
        return True
    else:
        print("\n❌ FAILED! No predictions were saved")
        print("\n🔍 Check the log file: logs/save_predictions.log")
        return False


if __name__ == "__main__":
    import os
    os.makedirs('logs', exist_ok=True)
    
    success = main()
    sys.exit(0 if success else 1)