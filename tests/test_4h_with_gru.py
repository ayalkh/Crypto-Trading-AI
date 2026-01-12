"""
4H Timeframe with GRU Benchmark
Tests tree-based ensemble vs tree+GRU ensemble for 4h timeframe
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import sys
import warnings
import json
from datetime import datetime

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import sqlite3
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_regression

from crypto_ai.features import FeatureEngineer
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

try:
    from tensorflow.keras.models import load_model
    DL_AVAILABLE = True
except:
    DL_AVAILABLE = False
    print("⚠️ TensorFlow not available")

import joblib


def test_4h_with_gru(symbol, db_path='data/ml_crypto_data.db'):
    """Test 4h timeframe with and without GRU"""
    
    print(f"\n{'='*70}")
    print(f"🧪 Testing: {symbol} 4h")
    print(f"{'='*70}")
    
    # Load data
    conn = sqlite3.connect(db_path)
    query = """
    SELECT timestamp, open, high, low, close, volume
    FROM price_data 
    WHERE symbol = ? AND timeframe = '4h'
    AND timestamp >= datetime('now', '-365 days')
    ORDER BY timestamp
    """
    df = pd.read_sql_query(query, conn, params=(symbol,))
    conn.close()
    
    if df.empty or len(df) < 200:
        print("   ⚠️  Insufficient data")
        return None
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Create features
    fe = FeatureEngineer()
    df_features = fe.create_features(df)
    df_features.dropna(inplace=True)
    
    if len(df_features) < 200:
        print("   ⚠️  Insufficient data after feature engineering")
        return None
    
    results = {}
    
    # ========== TEST 1: DIRECTION PREDICTION ==========
    print("\n   🎯 Direction Prediction")
    
    df_dir = df_features.copy()
    df_dir['target'] = (df_dir['close'].shift(-1) > df_dir['close']).astype(int)
    df_dir.dropna(inplace=True)
    
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'target']
    feature_cols = [col for col in df_dir.columns if col not in exclude_cols]
    
    X = df_dir[feature_cols].values
    y = df_dir['target'].values
    
    # Split
    train_size = int(len(X) * 0.70)
    val_size = int(len(X) * 0.85)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:val_size], y[train_size:val_size]
    X_test, y_test = X[val_size:], y[val_size:]
    
    # Feature selection
    selector = SelectKBest(score_func=mutual_info_classif, k=min(50, len(feature_cols)))
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_val_sel = selector.transform(X_val)
    X_test_sel = selector.transform(X_test)
    
    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_sel)
    X_val_s = scaler.transform(X_val_sel)
    X_test_s = scaler.transform(X_test_sel)
    
    # Train tree-based models
    tree_models = []
    
    lgb_model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                   random_state=42, verbose=-1, force_col_wise=True)
    lgb_model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)],
                 callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    tree_models.append(lgb_model)
    
    xgb_model = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                  random_state=42, verbosity=0)
    xgb_model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False)
    tree_models.append(xgb_model)
    
    cb_model = cb.CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6,
                                     random_seed=42, verbose=False)
    cb_model.fit(X_train_s, y_train, eval_set=(X_val_s, y_val),
                early_stopping_rounds=50, verbose=False)
    tree_models.append(cb_model)
    
    # Tree-only ensemble
    tree_probas = [model.predict_proba(X_test_s)[:, 1] for model in tree_models]
    tree_avg_proba = np.mean(tree_probas, axis=0)
    tree_pred = (tree_avg_proba >= 0.5).astype(int)
    
    tree_acc = accuracy_score(y_test, tree_pred)
    tree_f1 = f1_score(y_test, tree_pred, zero_division=0)
    
    print(f"      Tree-only: Acc={tree_acc:.2%}, F1={tree_f1:.2%}")
    
    results['direction_tree_only'] = {
        'accuracy': tree_acc,
        'f1': tree_f1
    }
    
    # Try to load GRU for direction
    safe_symbol = symbol.replace('/', '_')
    gru_path = f"ml_models/{safe_symbol}_4h_gru.h5"
    
    if DL_AVAILABLE and os.path.exists(gru_path):
        print(f"      Loading GRU model...")
        
        try:
            gru_model = load_model(gru_path, compile=False)
            gru_scaler_path = f"ml_models/{safe_symbol}_4h_gru_scaler.joblib"
            gru_scaler = joblib.load(gru_scaler_path)
            
            # Prepare GRU sequences
            sequence_length = 60
            prices = df['close'].values.reshape(-1, 1)
            scaled_prices = gru_scaler.transform(prices)
            
            # Create sequences for test set
            # Align with test indices
            test_start_idx = val_size
            gru_test_preds = []
            
            for i in range(test_start_idx, len(scaled_prices)):
                if i >= sequence_length:
                    seq = scaled_prices[i-sequence_length:i, 0].reshape(1, sequence_length, 1)
                    pred = gru_model.predict(seq, verbose=0)[0, 0]
                    gru_test_preds.append(pred)
            
            # Convert GRU price predictions to direction
            if len(gru_test_preds) > 0:
                gru_test_preds = np.array(gru_test_preds[:len(y_test)])
                
                # Direction: compare predicted vs current
                current_prices = df['close'].iloc[val_size:val_size+len(gru_test_preds)].values
                current_scaled = gru_scaler.transform(current_prices.reshape(-1, 1)).flatten()
                
                gru_directions = (gru_test_preds > current_scaled).astype(float)
                
                # Ensemble: 45% LGB, 30% XGB, 15% CB, 10% GRU (original weights)
                if len(gru_directions) == len(tree_avg_proba):
                    combined_proba = (
                        0.45 * tree_probas[0][:len(gru_directions)] +
                        0.30 * tree_probas[1][:len(gru_directions)] +
                        0.15 * tree_probas[2][:len(gru_directions)] +
                        0.10 * gru_directions
                    )
                    combined_pred = (combined_proba >= 0.5).astype(int)
                    
                    combined_acc = accuracy_score(y_test[:len(gru_directions)], combined_pred)
                    combined_f1 = f1_score(y_test[:len(gru_directions)], combined_pred, zero_division=0)
                    
                    print(f"      Tree+GRU:  Acc={combined_acc:.2%}, F1={combined_f1:.2%}")
                    
                    results['direction_with_gru'] = {
                        'accuracy': combined_acc,
                        'f1': combined_f1,
                        'improvement': combined_acc - tree_acc
                    }
                    
        except Exception as e:
            print(f"      ⚠️  GRU loading failed: {e}")
    else:
        print(f"      ⚠️  GRU model not found at {gru_path}")
    
    # ========== TEST 2: PRICE PREDICTION ==========
    print("\n   📈 Price Prediction")
    
    df_price = df_features.copy()
    df_price['target'] = df_price['close'].shift(-1) / df_price['close'] - 1
    df_price.dropna(inplace=True)
    
    X = df_price[feature_cols].values
    y = df_price['target'].values
    
    # Split
    train_size = int(len(X) * 0.70)
    val_size = int(len(X) * 0.85)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:val_size], y[train_size:val_size]
    X_test, y_test = X[val_size:], y[val_size:]
    
    # Feature selection
    selector = SelectKBest(score_func=f_regression, k=min(50, len(feature_cols)))
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    
    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_sel)
    X_test_s = scaler.transform(X_test_sel)
    
    # Train tree model (just LightGBM for speed)
    lgb_reg = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                                random_state=42, verbose=-1, force_col_wise=True)
    lgb_reg.fit(X_train_s, y_train)
    
    tree_price_pred = lgb_reg.predict(X_test_s)
    tree_mae = np.mean(np.abs(y_test - tree_price_pred))
    tree_r2 = r2_score(y_test, tree_price_pred)
    
    print(f"      Tree-only: MAE={tree_mae:.4%}, R²={tree_r2:.4f}")
    
    results['price_tree_only'] = {
        'mae': tree_mae,
        'r2': tree_r2
    }
    
    return results


def run_4h_gru_benchmark():
    """Run 4h benchmark with GRU"""
    print("\n" + "="*80)
    print("🧠 4H TIMEFRAME WITH GRU BENCHMARK")
    print("Comparing: Tree-only vs Tree+GRU Ensemble")
    print("="*80)
    
    if not DL_AVAILABLE:
        print("\n❌ TensorFlow not available - cannot test GRU")
        return
    
    symbols = ['BTC/USDT', 'ETH/USDT']
    all_results = {}
    
    for symbol in symbols:
        result = test_4h_with_gru(symbol)
        if result:
            all_results[symbol] = result
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 SUMMARY: Does GRU Improve 4H Predictions?")
    print(f"{'='*80}\n")
    
    if all_results:
        print(f"{'Symbol':<15} {'Tree Acc':<12} {'Tree+GRU Acc':<15} {'Improvement':<12}")
        print("-" * 80)
        
        for symbol, results in all_results.items():
            tree_acc = results.get('direction_tree_only', {}).get('accuracy', 0)
            
            if 'direction_with_gru' in results:
                gru_acc = results['direction_with_gru']['accuracy']
                improvement = results['direction_with_gru']['improvement']
                emoji = "✅" if improvement > 0 else "❌"
                
                print(f"{symbol:<15} {tree_acc:<12.2%} {gru_acc:<15.2%} {improvement:+.2%} {emoji}")
            else:
                print(f"{symbol:<15} {tree_acc:<12.2%} {'N/A':<15} {'N/A':<12}")
        
        # Overall verdict
        improvements = [r['direction_with_gru']['improvement'] 
                       for r in all_results.values() 
                       if 'direction_with_gru' in r]
        
        if improvements:
            avg_improvement = np.mean(improvements)
            
            print(f"\nAverage Improvement: {avg_improvement:+.2%}")
            
            if avg_improvement > 0.02:
                print("\n✅ GRU HELPS! Adding GRU improves predictions by >2%")
            elif avg_improvement > 0:
                print("\n🟡 GRU HELPS SLIGHTLY (< 2% improvement)")
            else:
                print("\n❌ GRU DOESN'T HELP (or makes it worse)")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'results': {
            sym: {
                k: {m: float(v) for m, v in metrics.items()}
                for k, metrics in res.items()
            }
            for sym, res in all_results.items()
        }
    }
    
    output_path = 'ml_reports/4h_gru_benchmark.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    run_4h_gru_benchmark()
