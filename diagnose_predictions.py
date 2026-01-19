"""
Diagnostic Script: Check ML Predictions in Database
Shows exactly what the agent is seeing
"""

import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = "data/ml_crypto_data.db"

def diagnose_predictions():
    """Check what predictions exist and their values"""
    
    print("\n" + "="*80)
    print("🔍 ML PREDICTIONS DIAGNOSTIC")
    print("="*80)
    
    conn = sqlite3.connect(DB_PATH)
    
    # 1. Check which models exist
    print("\n📊 STEP 1: Which models exist in database?")
    print("-" * 80)
    
    query = """
        SELECT DISTINCT model_type, COUNT(*) as count
        FROM ml_predictions
        GROUP BY model_type
        ORDER BY count DESC
    """
    
    models = pd.read_sql_query(query, conn)
    
    if models.empty:
        print("❌ NO PREDICTIONS FOUND! Run generate_and_save_predictions.py first!")
        conn.close()
        return
    
    print(models.to_string(index=False))
    
    # 2. Check prediction values for specific symbols
    print("\n\n📊 STEP 2: Sample predictions (most recent)")
    print("-" * 80)
    
    for symbol in ['BNB/USDT', 'ADA/USDT', 'DOT/USDT']:
        for timeframe in ['1h', '4h']:
            query = f"""
                SELECT 
                    symbol,
                    timeframe,
                    model_type,
                    predicted_direction,
                    predicted_change_pct,
                    confidence_score,
                    datetime(timestamp) as time
                FROM ml_predictions
                WHERE symbol = '{symbol}' 
                  AND timeframe = '{timeframe}'
                ORDER BY timestamp DESC
                LIMIT 2
            """
            
            df = pd.read_sql_query(query, conn)
            
            if not df.empty:
                print(f"\n{symbol} {timeframe}:")
                for _, row in df.iterrows():
                    print(f"  {row['model_type']:10s} | "
                          f"Dir: {row['predicted_direction']:7s} | "
                          f"Change: {row['predicted_change_pct']:8.6f} | "  # Show many decimals
                          f"Conf: {row['confidence_score']:5.2%} | "
                          f"{row['time']}")
    
    # 3. Check prediction distribution
    print("\n\n📊 STEP 3: Prediction value distribution")
    print("-" * 80)
    
    query = """
        SELECT 
            model_type,
            timeframe,
            AVG(predicted_change_pct) as avg_change,
            MIN(predicted_change_pct) as min_change,
            MAX(predicted_change_pct) as max_change,
            AVG(ABS(predicted_change_pct)) as avg_abs_change
        FROM ml_predictions
        WHERE timestamp >= datetime('now', '-7 days')
        GROUP BY model_type, timeframe
        ORDER BY timeframe, model_type
    """
    
    stats = pd.read_sql_query(query, conn)
    
    print("\nPredicted price changes (last 7 days):")
    print(stats.to_string(index=False))
    
    # 4. Check direction distribution
    print("\n\n📊 STEP 4: Direction distribution")
    print("-" * 80)
    
    query = """
        SELECT 
            predicted_direction,
            COUNT(*) as count,
            ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM ml_predictions WHERE timestamp >= datetime('now', '-7 days')), 1) as percentage
        FROM ml_predictions
        WHERE timestamp >= datetime('now', '-7 days')
        GROUP BY predicted_direction
        ORDER BY count DESC
    """
    
    directions = pd.read_sql_query(query, conn)
    print(directions.to_string(index=False))
    
    # 5. Check if units are percentage or decimal
    print("\n\n📊 STEP 5: Unit analysis")
    print("-" * 80)
    
    query = """
        SELECT 
            predicted_change_pct,
            predicted_direction
        FROM ml_predictions
        WHERE ABS(predicted_change_pct) > 0
        ORDER BY timestamp DESC
        LIMIT 10
    """
    
    samples = pd.read_sql_query(query, conn)
    
    print("\nSample non-zero predictions:")
    print(samples.to_string(index=False))
    
    if not samples.empty:
        avg_abs = samples['predicted_change_pct'].abs().mean()
        
        print(f"\nAverage absolute prediction: {avg_abs:.6f}")
        
        if avg_abs > 1.0:
            print("⚠️  WARNING: Values > 1.0 suggest PERCENTAGE storage (e.g., 0.15 = 0.15%)")
            print("   Your thresholds expect DECIMAL (e.g., 0.0015 = 0.15%)")
            print("   FIX: Divide by 100 when reading from database")
        elif avg_abs > 0.1:
            print("⚠️  WARNING: Values > 0.1 suggest PERCENTAGE storage")
            print("   FIX: Divide by 100 when reading from database")
        elif avg_abs > 0.01:
            print("✅ Values in 0.01-0.1 range - likely correct DECIMAL storage")
            print("   0.01 = 1%, 0.001 = 0.1% ✓")
        else:
            print("✅ Values < 0.01 - looks like correct DECIMAL storage")
            print("   0.001 = 0.1%, 0.0001 = 0.01% ✓")
    
    conn.close()
    
    # 6. Summary and recommendations
    print("\n\n" + "="*80)
    print("💡 DIAGNOSTIC SUMMARY")
    print("="*80)
    
    print("\n1. Models in database:")
    for _, row in models.iterrows():
        print(f"   • {row['model_type']}: {row['count']} predictions")
    
    print("\n2. Config expects: catboost, xgboost")
    
    missing_models = []
    if 'lightgbm' in models['model_type'].values:
        print("   ⚠️  Found 'lightgbm' in DB but removed from config - OK")
    if 'gru' in models['model_type'].values:
        print("   ⚠️  Found 'gru' in DB but removed from config - OK")
    
    if 'catboost' not in models['model_type'].values:
        missing_models.append('catboost')
    if 'xgboost' not in models['model_type'].values:
        missing_models.append('xgboost')
    
    if missing_models:
        print(f"\n   ❌ MISSING MODELS: {missing_models}")
        print("      → Run training with these models!")
    else:
        print("\n   ✅ All required models present!")
    
    print("\n3. Next steps:")
    print("   → Replace crypto_agent/config.py with crypto_agent_config_FIXED.py")
    print("   → Run: python run_agent_FINAL.py")
    print("   → Check if you get BUY/SELL signals instead of HOLD")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    diagnose_predictions()