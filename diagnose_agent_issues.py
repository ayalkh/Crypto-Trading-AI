"""
Diagnose Agent Consensus Issues
Check why all consensus results are 0% confidence
"""

import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = 'data/ml_crypto_data.db'

def diagnose_predictions():
    """Analyze what's in the predictions table"""
    
    print("="*70)
    print("🔍 DIAGNOSING PREDICTION ISSUES")
    print("="*70 + "\n")
    
    conn = sqlite3.connect(DB_PATH)
    
    # Get latest predictions
    query = """
    SELECT 
        symbol,
        timeframe,
        model_type,
        predicted_direction,
        direction_probability,
        predicted_change_pct,
        confidence_score,
        timestamp
    FROM ml_predictions
    WHERE symbol = 'BTC/USDT'
    ORDER BY timestamp DESC
    LIMIT 20
    """
    
    df = pd.read_sql_query(query, conn)
    
    if df.empty:
        print("❌ No predictions found!")
        return
    
    print("📊 Latest 20 Predictions for BTC/USDT:")
    print("-" * 70)
    
    # Group by timeframe and model
    for timeframe in ['5m', '15m', '1h', '4h', '1d']:
        tf_data = df[df['timeframe'] == timeframe]
        
        if not tf_data.empty:
            print(f"\n{timeframe}:")
            
            for _, row in tf_data.head(4).iterrows():
                age = (datetime.now() - pd.to_datetime(row['timestamp'])).total_seconds() / 3600
                
                print(f"  {row['model_type']:12} | "
                      f"{row['predicted_direction']:8} | "
                      f"prob:{row['direction_probability']:5.1%} | "
                      f"change:{row['predicted_change_pct']:+7.3%} | "
                      f"conf:{row['confidence_score']:5.1%} | "
                      f"age:{age:5.1f}h")
    
    # Summary statistics
    print("\n" + "="*70)
    print("📈 SUMMARY STATISTICS")
    print("="*70 + "\n")
    
    latest_time = pd.to_datetime(df['timestamp'].max())
    oldest_time = pd.to_datetime(df['timestamp'].min())
    age_hours = (datetime.now() - latest_time).total_seconds() / 3600
    
    print(f"Latest prediction: {latest_time}")
    print(f"Age: {age_hours:.1f} hours")
    print(f"Oldest prediction: {oldest_time}")
    
    # Direction distribution
    print("\nDirection Distribution:")
    print(df['predicted_direction'].value_counts())
    
    # Average metrics
    print(f"\nAverage direction_probability: {df['direction_probability'].mean():.2%}")
    print(f"Average predicted_change_pct: {df['predicted_change_pct'].mean():+.3%}")
    print(f"Average confidence_score: {df['confidence_score'].mean():.2%}")
    
    # Check if changes are too small
    print("\n" + "="*70)
    print("🔍 DIAGNOSIS")
    print("="*70 + "\n")
    
    if age_hours > 24:
        print(f"❌ PROBLEM: Predictions are {age_hours:.0f} hours old!")
        print("   Solution: Run 'python generate_fresh_predictions.py'")
    
    avg_change = abs(df['predicted_change_pct'].mean())
    if avg_change < 0.001:  # Less than 0.1%
        print(f"❌ PROBLEM: Predicted changes are tiny ({avg_change:.4%})")
        print("   All predictions are essentially NEUTRAL")
        print("   Solution: Check if models are trained properly")
    
    neutral_pct = (df['predicted_direction'] == 'NEUTRAL').sum() / len(df)
    if neutral_pct > 0.7:
        print(f"❌ PROBLEM: {neutral_pct:.0%} predictions are NEUTRAL")
        print("   Models may need retraining with better features")
    
    avg_conf = df['confidence_score'].mean()
    if avg_conf < 0.3:
        print(f"❌ PROBLEM: Average confidence very low ({avg_conf:.1%})")
        print("   Models are very uncertain about their predictions")
    
    conn.close()

def check_model_files():
    """Check if model files exist and are recent"""
    
    print("\n" + "="*70)
    print("🔍 CHECKING MODEL FILES")
    print("="*70 + "\n")
    
    import os
    from pathlib import Path
    
    models_dir = Path('models')
    
    if not models_dir.exists():
        print("❌ Models directory doesn't exist!")
        return
    
    # Check for each symbol
    for symbol in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOTUSDT']:
        print(f"\n{symbol}:")
        
        symbol_dir = models_dir / symbol
        if not symbol_dir.exists():
            print("  ❌ No models directory")
            continue
        
        # Count model files
        model_files = list(symbol_dir.glob('**/*.pkl'))
        
        if not model_files:
            print("  ❌ No .pkl model files found")
            continue
        
        print(f"  ✅ Found {len(model_files)} model files")
        
        # Check most recent
        recent = max(model_files, key=lambda p: p.stat().st_mtime)
        age_days = (datetime.now().timestamp() - recent.stat().st_mtime) / 86400
        
        print(f"  📅 Most recent: {recent.name}")
        print(f"  ⏰ Age: {age_days:.1f} days")
        
        if age_days > 30:
            print(f"  ⚠️  Models are old - consider retraining")

if __name__ == "__main__":
    diagnose_predictions()
    check_model_files()
    
    print("\n" + "="*70)
    print("💡 RECOMMENDATIONS")
    print("="*70)
    print("\n1. Generate fresh predictions:")
    print("   python generate_fresh_predictions.py")
    print("\n2. If predictions are still weak, retrain models:")
    print("   python train_all_models.py")
    print("\n3. Then run agent again:")
    print("   python run_agent_FINAL.py")