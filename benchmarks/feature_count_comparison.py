"""
Feature Count Comparison Benchmark
Tests 20 vs 50 features to see if fewer features improve generalization
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from crypto_ai.features import FeatureEngineer
import lightgbm as lgb
import xgboost as xgb
import catboost as cb


def test_with_n_features(symbol, timeframe, n_features, db_path='data/ml_crypto_data.db'):
    """Test a single symbol/timeframe with specific number of features"""
    
    # Load data
    conn = sqlite3.connect(db_path)
    
    lookback_days = {
        '5m': 30, '15m': 60, '1h': 180, '4h': 365, '1d': 730
    }
    lookback = lookback_days.get(timeframe, 180)
    
    query = f"""
    SELECT timestamp, open, high, low, close, volume
    FROM price_data 
    WHERE symbol = ? AND timeframe = ? 
    AND timestamp >= datetime('now', '-{lookback} days')
    ORDER BY timestamp
    """
    df = pd.read_sql_query(query, conn, params=(symbol, timeframe))
    conn.close()
    
    if df.empty or len(df) < 200:
        return None
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Create features
    fe = FeatureEngineer()
    df_features = fe.create_features(df)
    df_features.dropna(inplace=True)
    
    if len(df_features) < 200:
        return None
    
    # Direction target
    df_features['target'] = (df_features['close'].shift(-1) > df_features['close']).astype(int)
    df_features.dropna(inplace=True)
    
    # Prepare features
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'target']
    feature_cols = [col for col in df_features.columns if col not in exclude_cols]
    
    X = df_features[feature_cols].values
    y = df_features['target'].values
    
    # Split
    train_size = int(len(X) * 0.70)
    val_size = int(len(X) * 0.85)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:val_size], y[train_size:val_size]
    X_test, y_test = X[val_size:], y[val_size:]
    
    # Feature selection
    selector = SelectKBest(score_func=mutual_info_classif, k=min(n_features, len(feature_cols)))
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_val_sel = selector.transform(X_val)
    X_test_sel = selector.transform(X_test)
    
    # Get selected feature names
    selected_mask = selector.get_support()
    selected_features = [name for name, sel in zip(feature_cols, selected_mask) if sel]
    
    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_sel)
    X_val_s = scaler.transform(X_val_sel)
    X_test_s = scaler.transform(X_test_sel)
    
    # Train ensemble
    models = []
    
    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        random_state=42, verbose=-1, force_col_wise=True
    )
    lgb_model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)],
                 callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    models.append(lgb_model)
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        random_state=42, verbosity=0
    )
    xgb_model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False)
    models.append(xgb_model)
    
    # CatBoost
    cb_model = cb.CatBoostClassifier(
        iterations=300, learning_rate=0.05, depth=6,
        random_seed=42, verbose=False
    )
    cb_model.fit(X_train_s, y_train, eval_set=(X_val_s, y_val),
                early_stopping_rounds=50, verbose=False)
    models.append(cb_model)
    
    # Ensemble prediction
    probas = [model.predict_proba(X_test_s)[:, 1] for model in models]
    avg_proba = np.mean(probas, axis=0)
    pred = (avg_proba >= 0.5).astype(int)
    
    # Calculate metrics
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'samples_test': len(X_test),
        'selected_features': selected_features[:10]  # Top 10 for reference
    }


def run_feature_count_comparison():
    """Compare 20 vs 50 features across all timeframes"""
    print("\n" + "="*80)
    print("🔬 FEATURE COUNT COMPARISON: 20 vs 50 Features")
    print("Hypothesis: Fewer features might reduce overfitting")
    print("="*80 + "\n")
    
    symbols = ['BTC/USDT', 'ETH/USDT']
    timeframes = ['5m', '15m', '1h', '4h', '1d']
    feature_counts = [20, 50]
    
    results = {}
    
    for n_features in feature_counts:
        print(f"\n{'='*70}")
        print(f"📊 Testing with {n_features} Features")
        print(f"{'='*70}")
        
        results[n_features] = {}
        
        for timeframe in timeframes:
            results[n_features][timeframe] = {}
            
            for symbol in symbols:
                print(f"   {timeframe:>4} {symbol:>12}...", end=' ')
                
                try:
                    result = test_with_n_features(symbol, timeframe, n_features)
                    
                    if result:
                        results[n_features][timeframe][symbol] = result
                        print(f"✅ {result['accuracy']:.2%}")
                    else:
                        print("⚠️  Skip")
                        
                except Exception as e:
                    print(f"❌ {str(e)[:30]}")
    
    # Comparison
    print(f"\n{'='*80}")
    print("📊 COMPARISON: 20 Features vs 50 Features")
    print(f"{'='*80}\n")
    
    print(f"{'Timeframe':<12} {'20 Feat Acc':<15} {'50 Feat Acc':<15} {'Difference':<15} {'Winner':<10}")
    print("-" * 80)
    
    comparison = {}
    
    for tf in timeframes:
        if tf in results[20] and tf in results[50]:
            # Calculate averages
            accs_20 = [r['accuracy'] for r in results[20][tf].values() if r]
            accs_50 = [r['accuracy'] for r in results[50][tf].values() if r]
            
            if accs_20 and accs_50:
                avg_20 = np.mean(accs_20)
                avg_50 = np.mean(accs_50)
                diff = avg_20 - avg_50
                
                winner = "20 ✅" if diff > 0 else "50 ✅" if diff < 0 else "Tie"
                
                comparison[tf] = {
                    'avg_20': avg_20,
                    'avg_50': avg_50,
                    'difference': diff
                }
                
                print(f"{tf:<12} {avg_20:<15.2%} {avg_50:<15.2%} {diff:+.2%}          {winner:<10}")
    
    # Overall winner
    print(f"\n{'='*80}")
    print("🏆 OVERALL WINNER")
    print(f"{'='*80}\n")
    
    all_20 = [comparison[tf]['avg_20'] for tf in comparison]
    all_50 = [comparison[tf]['avg_50'] for tf in comparison]
    
    overall_20 = np.mean(all_20)
    overall_50 = np.mean(all_50)
    overall_diff = overall_20 - overall_50
    
    print(f"Average across all timeframes:")
    print(f"  20 features: {overall_20:.2%}")
    print(f"  50 features: {overall_50:.2%}")
    print(f"  Difference:  {overall_diff:+.2%}")
    
    if abs(overall_diff) < 0.01:
        print(f"\n➖ NO SIGNIFICANT DIFFERENCE (< 1%)")
        print(f"   Feature count doesn't matter much")
    elif overall_diff > 0:
        print(f"\n✅ 20 FEATURES WIN by {overall_diff:.2%}")
        print(f"   Fewer features reduce overfitting!")
    else:
        print(f"\n✅ 50 FEATURES WIN by {abs(overall_diff):.2%}")
        print(f"   More features provide better signal")
    
    # Best configuration
    print(f"\n{'='*80}")
    print("🎯 BEST CONFIGURATION")
    print(f"{'='*80}\n")
    
    best_acc = 0
    best_config = None
    
    for n_feat in [20, 50]:
        for tf in timeframes:
            if tf in results[n_feat]:
                for sym, res in results[n_feat][tf].items():
                    if res and res['accuracy'] > best_acc:
                        best_acc = res['accuracy']
                        best_config = (n_feat, tf, sym)
    
    if best_config:
        n_feat, tf, sym = best_config
        print(f"Best: {sym} at {tf} with {n_feat} features")
        print(f"Accuracy: {best_acc:.2%}")
        print(f"F1 Score: {results[n_feat][tf][sym]['f1']:.2%}")
        
        if best_acc >= 0.60:
            print(f"\n🎉 This is PROMISING! (≥60% accuracy)")
        elif best_acc >= 0.55:
            print(f"\n🟡 This is MODERATE (55-60% accuracy)")
        else:
            print(f"\n⚠️  Still weak (< 55% accuracy)")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'results': {
            str(n): {
                tf: {
                    sym: {k: float(v) if isinstance(v, (np.floating, float)) else 
                          int(v) if isinstance(v, (np.integer, int)) else v
                          for k, v in metrics.items() if k != 'selected_features'}
                    for sym, metrics in symbols.items() if metrics
                }
                for tf, symbols in tfs.items()
            }
            for n, tfs in results.items()
        },
        'comparison': {
            tf: {k: float(v) for k, v in vals.items()}
            for tf, vals in comparison.items()
        },
        'overall': {
            '20_features': float(overall_20),
            '50_features': float(overall_50),
            'difference': float(overall_diff)
        }
    }
    
    output_path = 'ml_reports/feature_count_comparison.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    print(f"\n{'='*80}\n")
    
    return results, comparison


if __name__ == "__main__":
    run_feature_count_comparison()
