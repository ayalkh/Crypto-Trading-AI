"""
Baseline Benchmark Script
Runs current ML system and saves performance metrics for comparison.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import sys
import json
from datetime import datetime

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import sqlite3
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# Import the ML system
from optimized_ml_system import OptimizedCryptoMLSystem


def run_baseline_benchmark():
    """Run baseline benchmark and save results"""
    print("\n" + "="*70)
    print("📊 BASELINE BENCHMARK - Current ML System")
    print("="*70 + "\n")
    
    # Initialize ML system
    ml_system = OptimizedCryptoMLSystem()
    
    # Test configuration - use subset for quick benchmark
    symbols = ['BTC/USDT', 'ETH/USDT']
    timeframes = ['1h', '4h']
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'version': 'baseline',
        'models': {}
    }
    
    for symbol in symbols:
        for timeframe in timeframes:
            print(f"\n{'='*60}")
            print(f"🧪 Benchmarking: {symbol} {timeframe}")
            print(f"{'='*60}")
            
            key = f"{symbol}_{timeframe}"
            results['models'][key] = {}
            
            # Load data
            df = ml_system.load_data(symbol, timeframe)
            if df.empty or len(df) < 100:
                print(f"⚠️ Insufficient data for {symbol} {timeframe}")
                continue
            
            # Create features
            df_features = ml_system.create_features(df)
            if df_features.empty:
                print(f"⚠️ Feature creation failed for {symbol} {timeframe}")
                continue
            
            # Prepare data for price prediction
            X, y, feature_cols = ml_system.prepare_data(df_features, prediction_type='price')
            
            if len(X) < 100:
                print(f"⚠️ Too few samples: {len(X)}")
                continue
            
            # Split data
            train_size = int(len(X) * 0.70)
            val_size = int(len(X) * 0.85)
            
            X_train, y_train = X[:train_size], y[:train_size]
            X_val, y_val = X[train_size:val_size], y[train_size:val_size]
            X_test, y_test = X[val_size:], y[val_size:]
            
            # Scale
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            
            # Train and evaluate each model
            model_results = {}
            
            # LightGBM
            try:
                import lightgbm as lgb
                lgb_model = lgb.LGBMRegressor(
                    n_estimators=500, learning_rate=0.05, max_depth=7,
                    num_leaves=31, min_child_samples=20, subsample=0.8,
                    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
                    random_state=42, verbose=-1, force_col_wise=True
                )
                lgb_model.fit(X_train_scaled, y_train,
                             eval_set=[(X_val_scaled, y_val)],
                             callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
                
                pred = lgb_model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, pred)
                r2 = r2_score(y_test, pred)
                dir_acc = (np.sign(pred) == np.sign(y_test)).mean()
                
                model_results['lightgbm'] = {
                    'mse': float(mse),
                    'r2': float(r2),
                    'direction_accuracy': float(dir_acc)
                }
                print(f"✅ LightGBM - R²: {r2:.4f}, MSE: {mse:.6f}, Dir Acc: {dir_acc:.2%}")
            except Exception as e:
                print(f"❌ LightGBM failed: {e}")
            
            # XGBoost
            try:
                import xgboost as xgb
                xgb_model = xgb.XGBRegressor(
                    n_estimators=500, learning_rate=0.05, max_depth=6,
                    min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
                    gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
                    random_state=42, verbosity=0
                )
                xgb_model.fit(X_train_scaled, y_train,
                             eval_set=[(X_val_scaled, y_val)],
                             verbose=False)
                
                pred = xgb_model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, pred)
                r2 = r2_score(y_test, pred)
                dir_acc = (np.sign(pred) == np.sign(y_test)).mean()
                
                model_results['xgboost'] = {
                    'mse': float(mse),
                    'r2': float(r2),
                    'direction_accuracy': float(dir_acc)
                }
                print(f"✅ XGBoost - R²: {r2:.4f}, MSE: {mse:.6f}, Dir Acc: {dir_acc:.2%}")
            except Exception as e:
                print(f"❌ XGBoost failed: {e}")
            
            # CatBoost
            try:
                import catboost as cb
                cb_model = cb.CatBoostRegressor(
                    iterations=500, learning_rate=0.05, depth=6,
                    l2_leaf_reg=3, random_seed=42, verbose=False
                )
                cb_model.fit(X_train_scaled, y_train,
                            eval_set=(X_val_scaled, y_val),
                            early_stopping_rounds=50, verbose=False)
                
                pred = cb_model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, pred)
                r2 = r2_score(y_test, pred)
                dir_acc = (np.sign(pred) == np.sign(y_test)).mean()
                
                model_results['catboost'] = {
                    'mse': float(mse),
                    'r2': float(r2),
                    'direction_accuracy': float(dir_acc)
                }
                print(f"✅ CatBoost - R²: {r2:.4f}, MSE: {mse:.6f}, Dir Acc: {dir_acc:.2%}")
            except Exception as e:
                print(f"❌ CatBoost failed: {e}")
            
            results['models'][key] = {
                'num_features': len(feature_cols),
                'num_samples': len(X),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'model_results': model_results
            }
    
    # Calculate averages
    all_r2 = []
    all_mse = []
    all_dir_acc = []
    
    for key, data in results['models'].items():
        if 'model_results' in data:
            for model_name, metrics in data['model_results'].items():
                all_r2.append(metrics['r2'])
                all_mse.append(metrics['mse'])
                all_dir_acc.append(metrics['direction_accuracy'])
    
    if all_r2:
        results['summary'] = {
            'avg_r2': float(np.mean(all_r2)),
            'avg_mse': float(np.mean(all_mse)),
            'avg_direction_accuracy': float(np.mean(all_dir_acc))
        }
    
    # Save results
    output_path = 'ml_reports/baseline_benchmark.json'
    os.makedirs('ml_reports', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("📊 BASELINE BENCHMARK COMPLETE")
    print(f"{'='*70}")
    if 'summary' in results:
        print(f"\n📈 Average Metrics:")
        print(f"   R²: {results['summary']['avg_r2']:.4f}")
        print(f"   MSE: {results['summary']['avg_mse']:.6f}")
        print(f"   Direction Accuracy: {results['summary']['avg_direction_accuracy']:.2%}")
    print(f"\n💾 Results saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    run_baseline_benchmark()
