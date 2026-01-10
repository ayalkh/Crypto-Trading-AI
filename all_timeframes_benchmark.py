"""
All Timeframes Benchmark
Tests direction prediction across ALL timeframes: 5m, 15m, 1h, 4h, 1d
Hypothesis: Longer timeframes should be more predictable
"""
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


def test_timeframe(symbol, timeframe, db_path='data/ml_crypto_data.db', n_features=50):
    """Test a single symbol/timeframe combination"""
    
    # Load data
    conn = sqlite3.connect(db_path)
    
    # Adaptive lookback based on timeframe
    lookback_days = {
        '5m': 30,      # 1 month
        '15m': 60,     # 2 months
        '1h': 180,     # 6 months
        '4h': 365,     # 1 year
        '1d': 730      # 2 years
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
    
    # Split (70% train, 15% val, 15% test)
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
    
    # Ensemble prediction (average probabilities)
    probas = [model.predict_proba(X_test_s)[:, 1] for model in models]
    avg_proba = np.mean(probas, axis=0)
    pred = (avg_proba >= 0.5).astype(int)
    
    # Calculate metrics
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    
    # Class distribution
    up_pct = y_test.mean()
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'samples_train': len(X_train),
        'samples_test': len(X_test),
        'up_percentage': up_pct,
        'features_used': n_features
    }


def run_all_timeframes_benchmark():
    """Run benchmark across all timeframes"""
    print("\n" + "="*80)
    print("📊 ALL TIMEFRAMES BENCHMARK")
    print("Testing: 5m, 15m, 1h, 4h, 1d")
    print("Hypothesis: Longer timeframes should be more predictable")
    print("="*80 + "\n")
    
    symbols = ['BTC/USDT', 'ETH/USDT']
    timeframes = ['5m', '15m', '1h', '4h', '1d']
    
    results = {}
    
    for timeframe in timeframes:
        print(f"\n{'='*70}")
        print(f"⏱️  Testing Timeframe: {timeframe}")
        print(f"{'='*70}")
        
        results[timeframe] = {}
        
        for symbol in symbols:
            print(f"\n   🔍 {symbol}...", end=' ')
            
            try:
                result = test_timeframe(symbol, timeframe)
                
                if result:
                    results[timeframe][symbol] = result
                    print(f"✅ Acc: {result['accuracy']:.2%}, F1: {result['f1']:.2%}")
                else:
                    print("⚠️  Insufficient data")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
    
    # Summary by timeframe
    print(f"\n{'='*80}")
    print("📊 SUMMARY BY TIMEFRAME")
    print(f"{'='*80}\n")
    
    summary = {}
    
    print(f"{'Timeframe':<12} {'Avg Acc':<12} {'Avg F1':<12} {'Samples':<12} {'Verdict':<20}")
    print("-" * 80)
    
    for tf in timeframes:
        if tf in results and results[tf]:
            accs = [r['accuracy'] for r in results[tf].values()]
            f1s = [r['f1'] for r in results[tf].values()]
            samples = [r['samples_test'] for r in results[tf].values()]
            
            avg_acc = np.mean(accs)
            avg_f1 = np.mean(f1s)
            avg_samples = int(np.mean(samples))
            
            # Verdict
            if avg_acc >= 0.60:
                verdict = "🟢 Good"
            elif avg_acc >= 0.55:
                verdict = "🟡 Moderate"
            elif avg_acc >= 0.52:
                verdict = "🟠 Weak"
            else:
                verdict = "🔴 Random"
            
            summary[tf] = {
                'avg_accuracy': avg_acc,
                'avg_f1': avg_f1,
                'avg_samples': avg_samples
            }
            
            print(f"{tf:<12} {avg_acc:<12.2%} {avg_f1:<12.2%} {avg_samples:<12} {verdict:<20}")
    
    # Hypothesis test
    print(f"\n{'='*80}")
    print("🔬 HYPOTHESIS TEST: Do longer timeframes perform better?")
    print(f"{'='*80}\n")
    
    if '5m' in summary and '1d' in summary:
        acc_5m = summary['5m']['avg_accuracy']
        acc_1d = summary['1d']['avg_accuracy']
        improvement = acc_1d - acc_5m
        
        print(f"5-minute accuracy:  {acc_5m:.2%}")
        print(f"1-day accuracy:     {acc_1d:.2%}")
        print(f"Improvement:        {improvement:+.2%}")
        
        if improvement > 0.05:
            print(f"\n✅ HYPOTHESIS CONFIRMED: Daily timeframe is {improvement:.1%} better!")
        elif improvement > 0:
            print(f"\n🟡 HYPOTHESIS PARTIALLY CONFIRMED: Slight improvement of {improvement:.1%}")
        else:
            print(f"\n❌ HYPOTHESIS REJECTED: No improvement (or worse)")
    
    # Best timeframe
    print(f"\n{'='*80}")
    print("🏆 BEST PERFORMING TIMEFRAME")
    print(f"{'='*80}\n")
    
    best_tf = max(summary.keys(), key=lambda x: summary[x]['avg_accuracy'])
    best_acc = summary[best_tf]['avg_accuracy']
    
    print(f"Winner: {best_tf}")
    print(f"Accuracy: {best_acc:.2%}")
    print(f"F1 Score: {summary[best_tf]['avg_f1']:.2%}")
    
    if best_acc >= 0.60:
        print(f"\n🎉 This timeframe shows PROFITABLE potential!")
    elif best_acc >= 0.55:
        print(f"\n🟡 This timeframe shows MODERATE potential (needs improvement)")
    else:
        print(f"\n⚠️  Even the best timeframe is weak (< 55% accuracy)")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'results': {tf: {sym: {k: float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v 
                               for k, v in metrics.items()} 
                        for sym, metrics in symbols.items()} 
                   for tf, symbols in results.items()},
        'summary': {tf: {k: float(v) if isinstance(v, (np.floating, float)) else int(v) 
                        for k, v in metrics.items()} 
                   for tf, metrics in summary.items()}
    }
    
    output_path = 'ml_reports/all_timeframes_benchmark.json'
    os.makedirs('ml_reports', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    print(f"\n{'='*80}\n")
    
    return results, summary


if __name__ == "__main__":
    run_all_timeframes_benchmark()
