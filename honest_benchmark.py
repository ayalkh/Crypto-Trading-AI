#!/usr/bin/env python3
"""
Honest Benchmark
Evaluates trained models ONLY on the last 15% of data (Test Set) 
to verify if the 85% accuracy is real or just overfitting to training data.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from optimized_ml_system import OptimizedCryptoMLSystem

def run_honest_benchmark():
    print("🕵️ HONEST BENCHMARK (Test Set Only)")
    print("=" * 60)
    
    ml_system = OptimizedCryptoMLSystem()
    results = []
    
    # Check just the top performers from previous run
    targets = [
        ('ETH/USDT', '1d'),
        ('BTC/USDT', '15m'),
        ('ADA/USDT', '4h'),
        ('ETH/USDT', '4h')
    ]
    
    for symbol, timeframe in targets:
        print(f"\nAnalyzing {symbol} {timeframe}...")
        
        # Load SAME data as training
        df = ml_system.load_data(symbol, timeframe)
        if df.empty: continue
            
        df = ml_system.create_features(df)
        
        # Prepare X, y
        actual_direction = (df['close'].shift(-1) > df['close']).astype(int)
        valid_idx = actual_direction.index[:-1]
        
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'target']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].iloc[:-1].values
        y_true = actual_direction.iloc[:-1].values
        
        # CRITICAL: Select only the last 15% (Test Set)
        # This matches the split logic in optimized_ml_system.py
        val_size = int(len(X) * 0.85)
        
        X_test = X[val_size:]
        y_test = y_true[val_size:]
        
        print(f"   Testing on last {len(y_test)} samples (out of {len(X)})")
        
        # Load XGBoost (since it was the winner)
        model_type = 'xgboost'
        safe_symbol = symbol.replace('/', '_')
        model_path = f"ml_models/{safe_symbol}_{timeframe}_direction_{model_type}.joblib"
        scaler_path = f"ml_models/{safe_symbol}_{timeframe}_direction_scaler.joblib"
        
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            
            X_test_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)
            
            acc = (y_pred == y_test).mean()
            
            print(f"   👉 Accuracy on Test Set: {acc:.2%}")
            
            if acc > 0.80:
                print("   ✅ RESULT: REAL! (High accuracy holds on unseen data)")
            elif acc > 0.55:
                print("   ⚠️ RESULT: OVERFIT (Good but drops significantly)")
            else:
                print("   ❌ RESULT: FAKE (Accuracy drops to random)")
                
            results.append({'config': f"{symbol} {timeframe}", 'accuracy': acc})

if __name__ == "__main__":
    run_honest_benchmark()
