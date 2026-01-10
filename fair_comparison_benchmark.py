"""
Fair Comparison Benchmark
Compares baseline vs enhanced using the same evaluation methodology (single holdout split).
This isolates the effect of new features + feature selection only.
"""
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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, mutual_info_regression

from crypto_ai.features import FeatureEngineer

import lightgbm as lgb
import xgboost as xgb
import catboost as cb


def run_fair_comparison():
    """Run fair comparison using same methodology for baseline and enhanced"""
    print("\n" + "="*70)
    print("📊 FAIR COMPARISON BENCHMARK")
    print("Same split methodology, comparing OLD vs NEW features")
    print("="*70 + "\n")
    
    db_path = 'data/ml_crypto_data.db'
    symbols = ['BTC/USDT', 'ETH/USDT']
    timeframes = ['1h', '4h']
    
    results = {
        'baseline': {'r2': [], 'mse': [], 'dir_acc': []},
        'enhanced_features_only': {'r2': [], 'mse': [], 'dir_acc': []},
        'enhanced_with_selection': {'r2': [], 'mse': [], 'dir_acc': []}
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
            
            # Create enhanced features
            fe = FeatureEngineer()
            df_features = fe.create_features(df)
            df_features.dropna(inplace=True)
            
            # Prepare target
            df_features['target'] = df_features['close'].shift(-1) / df_features['close'] - 1
            df_features.dropna(inplace=True)
            
            # Identify baseline vs new features
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
                'price_change_lag_1', 'price_change_lag_2', 'price_change_lag_3', 'price_change_lag_5', 'price_change_lag_10',
                'close_mean_5', 'close_mean_10', 'close_mean_20',
                'close_std_5', 'close_std_10', 'close_std_20',
                'volume_mean_5', 'volume_mean_10', 'volume_mean_20',
                'hour', 'day_of_week', 'day_of_month', 'is_weekend'
            ]
            
            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'target']
            all_feature_cols = [col for col in df_features.columns if col not in exclude_cols]
            
            # Get actual baseline features (intersection with what's available)
            baseline_feature_cols = [col for col in baseline_cols if col in df_features.columns]
            enhanced_feature_cols = all_feature_cols  # All features including new ones
            
            print(f"   Baseline features: {len(baseline_feature_cols)}")
            print(f"   Enhanced features: {len(enhanced_feature_cols)}")
            
            # Same split for all
            X_baseline = df_features[baseline_feature_cols].values
            X_enhanced = df_features[enhanced_feature_cols].values
            y = df_features['target'].values
            
            train_size = int(len(X_baseline) * 0.70)
            val_size = int(len(X_baseline) * 0.85)
            
            # BASELINE
            X_train = X_baseline[:train_size]
            X_val = X_baseline[train_size:val_size]
            X_test = X_baseline[val_size:]
            y_train, y_val, y_test = y[:train_size], y[train_size:val_size], y[val_size:]
            
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)
            X_test_s = scaler.transform(X_test)
            
            lgb_model = lgb.LGBMRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=7,
                random_state=42, verbose=-1, force_col_wise=True
            )
            lgb_model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)],
                         callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
            
            pred = lgb_model.predict(X_test_s)
            r2 = r2_score(y_test, pred)
            mse = mean_squared_error(y_test, pred)
            dir_acc = (np.sign(pred) == np.sign(y_test)).mean()
            
            results['baseline']['r2'].append(r2)
            results['baseline']['mse'].append(mse)
            results['baseline']['dir_acc'].append(dir_acc)
            print(f"   BASELINE      - R²: {r2:.4f}, MSE: {mse:.6f}, Dir Acc: {dir_acc:.2%}")
            
            # ENHANCED FEATURES ONLY
            X_train = X_enhanced[:train_size]
            X_val = X_enhanced[train_size:val_size]
            X_test = X_enhanced[val_size:]
            
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)
            X_test_s = scaler.transform(X_test)
            
            lgb_model = lgb.LGBMRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=7,
                random_state=42, verbose=-1, force_col_wise=True
            )
            lgb_model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)],
                         callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
            
            pred = lgb_model.predict(X_test_s)
            r2 = r2_score(y_test, pred)
            mse = mean_squared_error(y_test, pred)
            dir_acc = (np.sign(pred) == np.sign(y_test)).mean()
            
            results['enhanced_features_only']['r2'].append(r2)
            results['enhanced_features_only']['mse'].append(mse)
            results['enhanced_features_only']['dir_acc'].append(dir_acc)
            print(f"   ENHANCED      - R²: {r2:.4f}, MSE: {mse:.6f}, Dir Acc: {dir_acc:.2%}")
            
            # Get feature importance
            importance = dict(zip(enhanced_feature_cols, lgb_model.feature_importances_))
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            print(f"   Top 5 features: {', '.join([f[0] for f in top_features[:5]])}")
            
            # ENHANCED WITH SELECTION
            selector = SelectKBest(score_func=mutual_info_regression, k=min(50, len(enhanced_feature_cols)))
            X_train_sel = selector.fit_transform(X_enhanced[:train_size], y_train)
            X_val_sel = selector.transform(X_enhanced[train_size:val_size])
            X_test_sel = selector.transform(X_enhanced[val_size:])
            
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train_sel)
            X_val_s = scaler.transform(X_val_sel)
            X_test_s = scaler.transform(X_test_sel)
            
            lgb_model = lgb.LGBMRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=7,
                random_state=42, verbose=-1, force_col_wise=True
            )
            lgb_model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)],
                         callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
            
            pred = lgb_model.predict(X_test_s)
            r2 = r2_score(y_test, pred)
            mse = mean_squared_error(y_test, pred)
            dir_acc = (np.sign(pred) == np.sign(y_test)).mean()
            
            results['enhanced_with_selection']['r2'].append(r2)
            results['enhanced_with_selection']['mse'].append(mse)
            results['enhanced_with_selection']['dir_acc'].append(dir_acc)
            print(f"   ENHANCED+SEL  - R²: {r2:.4f}, MSE: {mse:.6f}, Dir Acc: {dir_acc:.2%}")
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 FINAL COMPARISON SUMMARY")
    print(f"{'='*70}\n")
    
    summary = {}
    for version in ['baseline', 'enhanced_features_only', 'enhanced_with_selection']:
        if results[version]['r2']:
            summary[version] = {
                'avg_r2': np.mean(results[version]['r2']),
                'avg_mse': np.mean(results[version]['mse']),
                'avg_dir_acc': np.mean(results[version]['dir_acc'])
            }
    
    print(f"{'Version':<25} {'Avg R²':<12} {'Avg MSE':<15} {'Dir Accuracy':<15}")
    print("-" * 70)
    
    for version, metrics in summary.items():
        print(f"{version:<25} {metrics['avg_r2']:<12.4f} {metrics['avg_mse']:<15.6f} {metrics['avg_dir_acc']:<15.2%}")
    
    # Calculate improvements
    if 'baseline' in summary and 'enhanced_features_only' in summary:
        r2_imp = summary['enhanced_features_only']['avg_r2'] - summary['baseline']['avg_r2']
        dir_imp = summary['enhanced_features_only']['avg_dir_acc'] - summary['baseline']['avg_dir_acc']
        
        print(f"\n📈 ENHANCEMENT IMPACT:")
        print(f"   R² change: {r2_imp:+.4f}")
        print(f"   Direction Accuracy change: {dir_imp:+.2%}")
    
    # Save detailed results
    output_path = 'ml_reports/fair_comparison.json'
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': {k: {m: [float(v) for v in vals] for m, vals in v.items()} 
                       for k, v in results.items()},
            'summary': {k: {m: float(v) for m, v in metrics.items()} 
                       for k, metrics in summary.items()}
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    
    return results, summary


if __name__ == "__main__":
    run_fair_comparison()
