"""
Diagnose Feature Mismatch
This script will show exactly what features are being created during training vs prediction
"""
import sys
import numpy as np
import pandas as pd
from optimized_ml_system_v2 import OptimizedMLSystemV2
import joblib

def diagnose():
    """Diagnose the feature mismatch issue"""
    
    print("\n" + "="*70)
    print("🔍 FEATURE MISMATCH DIAGNOSTIC")
    print("="*70 + "\n")
    
    symbol = 'BTC/USDT'
    timeframe = '5m'
    
    print(f"Testing: {symbol} {timeframe}\n")
    
    # Initialize system
    ml_system = OptimizedMLSystemV2()
    
    # Load data
    print("1️⃣ Loading data...")
    df = ml_system.load_data(symbol, timeframe, months_back=1)
    print(f"   Loaded {len(df)} candles\n")
    
    # Create features
    print("2️⃣ Creating features...")
    df_features = ml_system.create_features(df)
    
    if df_features.empty:
        print("   ❌ Failed to create features")
        return False
    
    exclude_cols = ['open', 'high', 'low', 'close', 'volume']
    feature_cols = [col for col in df_features.columns if col not in exclude_cols]
    
    print(f"   Total features created: {len(feature_cols)}")
    print(f"   Total rows: {len(df_features)}\n")
    
    # Check for saved models
    safe_symbol = symbol.replace('/', '_')
    
    model_path = f"ml_models/{safe_symbol}_{timeframe}_price_catboost.joblib"
    selector_path = f"ml_models/{safe_symbol}_{timeframe}_price_selector.joblib"
    scaler_path = f"ml_models/{safe_symbol}_{timeframe}_price_scaler.joblib"
    features_path = f"ml_models/{safe_symbol}_{timeframe}_price_features.joblib"
    
    print("3️⃣ Checking saved models...")
    
    import os
    if not os.path.exists(model_path):
        print(f"   ❌ Model not found: {model_path}")
        print("\n💡 You need to train the models first:")
        print("   python optimized_ml_system_v2.py")
        return False
    
    print(f"   ✅ Model found")
    
    if not os.path.exists(selector_path):
        print(f"   ❌ Selector not found: {selector_path}")
        return False
    
    print(f"   ✅ Selector found")
    
    if not os.path.exists(scaler_path):
        print(f"   ❌ Scaler not found: {scaler_path}")
        return False
    
    print(f"   ✅ Scaler found\n")
    
    # Load the selector
    print("4️⃣ Loading selector...")
    try:
        selector = joblib.load(selector_path)
        print(f"   ✅ Selector loaded")
        print(f"   Selector expects: {selector.k} features as OUTPUT")
        print(f"   Selector was trained on: {len(selector.scores_)} features as INPUT\n")
    except Exception as e:
        print(f"   ❌ Error loading selector: {e}")
        return False
    
    # Load the scaler
    print("5️⃣ Loading scaler...")
    try:
        scaler = joblib.load(scaler_path)
        print(f"   ✅ Scaler loaded")
        print(f"   Scaler expects: {scaler.n_features_in_} features\n")
    except Exception as e:
        print(f"   ❌ Error loading scaler: {e}")
        return False
    
    # Try to prepare features
    print("6️⃣ Preparing features for prediction...")
    try:
        X_latest = df_features[feature_cols].iloc[-1:].values
        print(f"   Created feature array with shape: {X_latest.shape}")
        print(f"   Number of features: {X_latest.shape[1]}\n")
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
        return False
    
    # Try to apply selector
    print("7️⃣ Applying feature selector...")
    try:
        X_selected = selector.transform(X_latest)
        print(f"   ✅ Selector transform successful")
        print(f"   Output shape: {X_selected.shape}")
        print(f"   Number of features after selection: {X_selected.shape[1]}\n")
    except Exception as e:
        print(f"   ❌ Selector transform FAILED")
        print(f"   Error: {e}")
        print(f"\n   🔍 ROOT CAUSE:")
        print(f"      - Selector was trained on {len(selector.scores_)} features")
        print(f"      - But you're giving it {X_latest.shape[1]} features")
        print(f"      - Difference: {len(selector.scores_) - X_latest.shape[1]} features missing!\n")
        
        # Find missing features
        if os.path.exists(features_path):
            saved_features = joblib.load(features_path)
            print(f"   📋 Saved feature list has {len(saved_features)} features")
            
            current_features = set(feature_cols)
            saved_features_set = set(saved_features) if isinstance(saved_features, list) else set()
            
            if saved_features_set:
                missing = saved_features_set - current_features
                extra = current_features - saved_features_set
                
                if missing:
                    print(f"\n   ⚠️ Features in saved list but NOT in current data:")
                    for feat in list(missing)[:10]:
                        print(f"      - {feat}")
                    if len(missing) > 10:
                        print(f"      ... and {len(missing)-10} more")
                
                if extra:
                    print(f"\n   ⚠️ Features in current data but NOT in saved list:")
                    for feat in list(extra)[:10]:
                        print(f"      - {feat}")
                    if len(extra) > 10:
                        print(f"      ... and {len(extra)-10} more")
        
        return False
    
    # Try to apply scaler
    print("8️⃣ Applying scaler...")
    try:
        X_scaled = scaler.transform(X_selected)
        print(f"   ✅ Scaler transform successful")
        print(f"   Final shape: {X_scaled.shape}\n")
    except Exception as e:
        print(f"   ❌ Scaler transform FAILED")
        print(f"   Error: {e}\n")
        return False
    
    print("="*70)
    print("✅ DIAGNOSIS COMPLETE - NO ERRORS FOUND")
    print("="*70)
    print("\nIf you're still getting errors, the issue may be:")
    print("  1. Different data being used during training vs prediction")
    print("  2. Database updated since training")
    print("  3. Need to retrain models\n")
    
    return True


if __name__ == "__main__":
    success = diagnose()
    
    if not success:
        print("\n" + "="*70)
        print("💡 RECOMMENDED FIX")
        print("="*70)
        print("\nThe feature count is different between training and prediction.")
        print("\nOption 1: Retrain models (RECOMMENDED)")
        print("   python optimized_ml_system_v2.py")
        print("\nOption 2: Check if create_features() changed since training")
        print("\n")
    
    sys.exit(0 if success else 1)