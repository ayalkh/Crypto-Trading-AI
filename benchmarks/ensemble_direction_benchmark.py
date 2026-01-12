"""
Ensemble Direction Benchmark
Uses voting ensemble of LightGBM + XGBoost + CatBoost for direction prediction.
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
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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


def run_ensemble_direction_benchmark():
    """Run ensemble direction benchmark"""
    print("\n" + "="*70)
    print("🎯 ENSEMBLE DIRECTION BENCHMARK")
    print("3-Model Voting Ensemble | 50 Features | Direction Focus")
    print("="*70 + "\n")
    
    db_path = 'data/ml_crypto_data.db'
    symbols = ['BTC/USDT', 'ETH/USDT']
    timeframes = ['1h', '4h']
    N_FEATURES = 50
    
    results = {
        'baseline_ensemble': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
        'enhanced_ensemble': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    }
    
    for symbol in symbols:
        for timeframe in timeframes:
            print(f"\n{'='*60}")
            print(f"🧪 Testing: {symbol} {timeframe}")
            print(f"{'='*60}")
            
            # Load data
            conn = sqlite3.connect(db_path)
            lookback = {'1h': 6, '4h': 12}[timeframe] * 30
            
            query = f"""
            SELECT timestamp, open, high, low, close, volume
            FROM price_data 
            WHERE symbol = ? AND timeframe = ? 
            AND timestamp >= datetime('now', '-{lookback} days')
            ORDER BY timestamp
            """
            df = pd.read_sql_query(query, conn, params=(symbol, timeframe))
            conn.close()
            
            if df.empty or len(df) < 100:
                continue
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Create features
            fe = FeatureEngineer()
            df_features = fe.create_features(df)
            df_features.dropna(inplace=True)
            
            # Direction target
            df_features['target'] = (df_features['close'].shift(-1) > df_features['close']).astype(int)
            df_features.dropna(inplace=True)
            
            # Baseline features
            baseline_cols = [
                'price_change', 'high_low_pct', 'close_open_pct',
                'ma_5', 'ma_10', 'ma_20', 'ma_50',
                'ma_5_ratio', 'ma_10_ratio', 'ma_20_ratio', 'ma_50_ratio',
                'rsi_14', 'rsi_21', 'rsi_28',
                'macd', 'macd_signal', 'macd_histogram',
                'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
                'atr', 'volume_sma', 'volume_ratio', 'price_volume', 'vwap',
                'volatility_5', 'volatility_10', 'volatility_20',
                'volatility_ratio_5', 'volatility_ratio_10', 'volatility_ratio_20',
                'momentum_5', 'momentum_10', 'momentum_20',
                'roc_5', 'roc_10', 'roc_20',
                'close_lag_1', 'close_lag_2', 'close_lag_3', 'close_lag_5', 'close_lag_10',
                'volume_lag_1', 'volume_lag_2', 'volume_lag_3', 'volume_lag_5', 'volume_lag_10',
                'price_change_lag_1', 'price_change_lag_2', 'price_change_lag_3', 
                'price_change_lag_5', 'price_change_lag_10',
                'close_mean_5', 'close_mean_10', 'close_mean_20',
                'close_std_5', 'close_std_10', 'close_std_20',
                'volume_mean_5', 'volume_mean_10', 'volume_mean_20',
                'hour', 'day_of_week', 'day_of_month', 'is_weekend'
            ]
            
            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'target']
            all_feature_cols = [col for col in df_features.columns if col not in exclude_cols]
            baseline_feature_cols = [col for col in baseline_cols if col in df_features.columns]
            
            y = df_features['target'].values
            
            # Split
            train_size = int(len(df_features) * 0.70)
            val_size = int(len(df_features) * 0.85)
            
            y_train, y_val, y_test = y[:train_size], y[train_size:val_size], y[val_size:]
            
            # ========== BASELINE ENSEMBLE ==========
            X_baseline = df_features[baseline_feature_cols].values
            X_train = X_baseline[:train_size]
            X_val = X_baseline[train_size:val_size]
            X_test = X_baseline[val_size:]
            
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)
            X_test_s = scaler.transform(X_test)
            
            # Train 3 models
            lgb_model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                           random_state=42, verbose=-1)
            lgb_model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)],
                         callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            
            xgb_model = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                          random_state=42, verbosity=0)
            xgb_model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False)
            
            cb_model = cb.CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6,
                                             random_seed=42, verbose=False)
            cb_model.fit(X_train_s, y_train, eval_set=(X_val_s, y_val),
                        early_stopping_rounds=50, verbose=False)
            
            # Ensemble voting (probability average)
            lgb_proba = lgb_model.predict_proba(X_test_s)[:, 1]
            xgb_proba = xgb_model.predict_proba(X_test_s)[:, 1]
            cb_proba = cb_model.predict_proba(X_test_s)[:, 1]
            
            avg_proba = (lgb_proba + xgb_proba + cb_proba) / 3
            pred = (avg_proba >= 0.5).astype(int)
            
            acc = accuracy_score(y_test, pred)
            prec = precision_score(y_test, pred, zero_division=0)
            rec = recall_score(y_test, pred, zero_division=0)
            f1 = f1_score(y_test, pred, zero_division=0)
            
            results['baseline_ensemble']['accuracy'].append(acc)
            results['baseline_ensemble']['precision'].append(prec)
            results['baseline_ensemble']['recall'].append(rec)
            results['baseline_ensemble']['f1'].append(f1)
            
            print(f"\n   📊 BASELINE ENSEMBLE ({len(baseline_feature_cols)} features)")
            print(f"      Accuracy:  {acc:.2%}")
            print(f"      F1 Score:  {f1:.2%}")
            
            # ========== ENHANCED ENSEMBLE ==========
            X_enhanced = df_features[all_feature_cols].values
            
            # Feature selection
            selector = SelectKBest(score_func=mutual_info_classif, k=min(N_FEATURES, len(all_feature_cols)))
            X_train_sel = selector.fit_transform(X_enhanced[:train_size], y_train)
            X_val_sel = selector.transform(X_enhanced[train_size:val_size])
            X_test_sel = selector.transform(X_enhanced[val_size:])
            
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train_sel)
            X_val_s = scaler.transform(X_val_sel)
            X_test_s = scaler.transform(X_test_sel)
            
            # Train 3 models
            lgb_model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                           random_state=42, verbose=-1)
            lgb_model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)],
                         callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            
            xgb_model = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                          random_state=42, verbosity=0)
            xgb_model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False)
            
            cb_model = cb.CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6,
                                             random_seed=42, verbose=False)
            cb_model.fit(X_train_s, y_train, eval_set=(X_val_s, y_val),
                        early_stopping_rounds=50, verbose=False)
            
            # Ensemble voting
            lgb_proba = lgb_model.predict_proba(X_test_s)[:, 1]
            xgb_proba = xgb_model.predict_proba(X_test_s)[:, 1]
            cb_proba = cb_model.predict_proba(X_test_s)[:, 1]
            
            avg_proba = (lgb_proba + xgb_proba + cb_proba) / 3
            pred = (avg_proba >= 0.5).astype(int)
            
            acc = accuracy_score(y_test, pred)
            prec = precision_score(y_test, pred, zero_division=0)
            rec = recall_score(y_test, pred, zero_division=0)
            f1 = f1_score(y_test, pred, zero_division=0)
            
            results['enhanced_ensemble']['accuracy'].append(acc)
            results['enhanced_ensemble']['precision'].append(prec)
            results['enhanced_ensemble']['recall'].append(rec)
            results['enhanced_ensemble']['f1'].append(f1)
            
            print(f"\n   🚀 ENHANCED ENSEMBLE ({N_FEATURES} selected features)")
            print(f"      Accuracy:  {acc:.2%}")
            print(f"      F1 Score:  {f1:.2%}")
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 ENSEMBLE DIRECTION COMPARISON")
    print(f"{'='*70}\n")
    
    summary = {}
    for version in ['baseline_ensemble', 'enhanced_ensemble']:
        if results[version]['accuracy']:
            summary[version] = {
                'avg_accuracy': np.mean(results[version]['accuracy']),
                'avg_precision': np.mean(results[version]['precision']),
                'avg_recall': np.mean(results[version]['recall']),
                'avg_f1': np.mean(results[version]['f1'])
            }
    
    print(f"{'Metric':<20} {'Baseline':<15} {'Enhanced':<15} {'Change':<15}")
    print("-" * 65)
    
    for metric in ['avg_accuracy', 'avg_precision', 'avg_recall', 'avg_f1']:
        base_val = summary['baseline_ensemble'][metric]
        enh_val = summary['enhanced_ensemble'][metric]
        change = enh_val - base_val
        emoji = "✅" if change > 0 else "❌" if change < 0 else "➖"
        
        metric_name = metric.replace('avg_', '').title()
        print(f"{metric_name:<20} {base_val:<15.2%} {enh_val:<15.2%} {change:+.2%} {emoji}")
    
    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'results': {k: {m: [float(v) for v in vals] for m, vals in v.items()} 
                   for k, v in results.items()},
        'summary': {k: {m: float(v) for m, v in metrics.items()} 
                   for k, metrics in summary.items()}
    }
    
    output_path = 'ml_reports/ensemble_direction_benchmark.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    return results, summary


if __name__ == "__main__":
    run_ensemble_direction_benchmark()
