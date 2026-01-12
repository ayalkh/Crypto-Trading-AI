"""
Direction-Focused ML Benchmark
Optimized for direction accuracy with 50 features selection.
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from crypto_ai.features import FeatureEngineer

import lightgbm as lgb
import xgboost as xgb
import catboost as cb


def run_direction_benchmark():
    """Run direction-focused benchmark with 50 features"""
    print("\n" + "="*70)
    print("🎯 DIRECTION-FOCUSED ML BENCHMARK")
    print("Feature Selection: 50 features | Target: Direction Accuracy")
    print("="*70 + "\n")
    
    db_path = 'data/ml_crypto_data.db'
    symbols = ['BTC/USDT', 'ETH/USDT']
    timeframes = ['1h', '4h']
    N_FEATURES = 50
    
    results = {
        'baseline': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
        'enhanced': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    }
    
    all_feature_importance = {}
    
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
            
            # DIRECTION target (1=UP, 0=DOWN)
            df_features['target'] = (df_features['close'].shift(-1) > df_features['close']).astype(int)
            df_features.dropna(inplace=True)
            
            # Baseline features (original ~66)
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
            baseline_feature_cols = [col for col in baseline_cols if col in df_features.columns]
            
            y = df_features['target'].values
            
            # Split
            train_size = int(len(df_features) * 0.70)
            val_size = int(len(df_features) * 0.85)
            
            y_train, y_val, y_test = y[:train_size], y[train_size:val_size], y[val_size:]
            
            print(f"   Samples - Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
            print(f"   Class balance - UP: {y_test.mean():.1%}, DOWN: {1-y_test.mean():.1%}")
            
            # ========== BASELINE (Original features) ==========
            X_baseline = df_features[baseline_feature_cols].values
            X_train = X_baseline[:train_size]
            X_val = X_baseline[train_size:val_size]
            X_test = X_baseline[val_size:]
            
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)
            X_test_s = scaler.transform(X_test)
            
            # LightGBM Classifier
            lgb_model = lgb.LGBMClassifier(
                n_estimators=500, learning_rate=0.05, max_depth=7,
                random_state=42, verbose=-1, force_col_wise=True
            )
            lgb_model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)],
                         callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
            
            pred = lgb_model.predict(X_test_s)
            acc = accuracy_score(y_test, pred)
            prec = precision_score(y_test, pred, zero_division=0)
            rec = recall_score(y_test, pred, zero_division=0)
            f1 = f1_score(y_test, pred, zero_division=0)
            
            results['baseline']['accuracy'].append(acc)
            results['baseline']['precision'].append(prec)
            results['baseline']['recall'].append(rec)
            results['baseline']['f1'].append(f1)
            
            print(f"\n   📊 BASELINE ({len(baseline_feature_cols)} features)")
            print(f"      Accuracy:  {acc:.2%}")
            print(f"      Precision: {prec:.2%}")
            print(f"      Recall:    {rec:.2%}")
            print(f"      F1 Score:  {f1:.2%}")
            
            # ========== ENHANCED (50 selected features) ==========
            X_enhanced = df_features[all_feature_cols].values
            
            # Feature selection using mutual information for classification
            selector = SelectKBest(score_func=mutual_info_classif, k=min(N_FEATURES, len(all_feature_cols)))
            X_train_sel = selector.fit_transform(X_enhanced[:train_size], y_train)
            X_val_sel = selector.transform(X_enhanced[train_size:val_size])
            X_test_sel = selector.transform(X_enhanced[val_size:])
            
            # Get selected feature names
            selected_mask = selector.get_support()
            selected_features = [name for name, sel in zip(all_feature_cols, selected_mask) if sel]
            scores = selector.scores_
            feature_scores = sorted(
                [(name, scores[i]) for i, name in enumerate(all_feature_cols) if selected_mask[i]],
                key=lambda x: x[1], reverse=True
            )
            
            # Track feature importance
            key = f"{symbol}_{timeframe}"
            all_feature_importance[key] = feature_scores[:15]
            
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train_sel)
            X_val_s = scaler.transform(X_val_sel)
            X_test_s = scaler.transform(X_test_sel)
            
            # LightGBM Classifier
            lgb_model = lgb.LGBMClassifier(
                n_estimators=500, learning_rate=0.05, max_depth=7,
                random_state=42, verbose=-1, force_col_wise=True
            )
            lgb_model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)],
                         callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
            
            pred = lgb_model.predict(X_test_s)
            acc = accuracy_score(y_test, pred)
            prec = precision_score(y_test, pred, zero_division=0)
            rec = recall_score(y_test, pred, zero_division=0)
            f1 = f1_score(y_test, pred, zero_division=0)
            
            results['enhanced']['accuracy'].append(acc)
            results['enhanced']['precision'].append(prec)
            results['enhanced']['recall'].append(rec)
            results['enhanced']['f1'].append(f1)
            
            print(f"\n   🚀 ENHANCED ({N_FEATURES} selected features)")
            print(f"      Accuracy:  {acc:.2%}")
            print(f"      Precision: {prec:.2%}")
            print(f"      Recall:    {rec:.2%}")
            print(f"      F1 Score:  {f1:.2%}")
            
            # Show top features
            print(f"\n   📈 Top 5 Features:")
            for name, score in feature_scores[:5]:
                new_marker = "✨NEW" if name in ['stoch_k', 'stoch_d', 'stoch_divergence', 'williams_r', 
                                                  'obv', 'obv_sma', 'obv_ratio', 'obv_change',
                                                  'ichimoku_tenkan', 'ichimoku_kijun', 'ichimoku_senkou_a',
                                                  'ichimoku_senkou_b', 'ichimoku_cloud_thickness', 
                                                  'ichimoku_cloud_position', 'ichimoku_tk_diff',
                                                  'adx', 'plus_di', 'minus_di', 'adx_trend_direction',
                                                  'market_regime', 'volatility_regime', 'trend_regime',
                                                  'volatility_cluster', 'trend_strength'] else ""
                print(f"      {name}: {score:.4f} {new_marker}")
    
    # ========== SUMMARY ==========
    print(f"\n{'='*70}")
    print("📊 FINAL DIRECTION ACCURACY COMPARISON")
    print(f"{'='*70}\n")
    
    summary = {}
    for version in ['baseline', 'enhanced']:
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
        base_val = summary['baseline'][metric]
        enh_val = summary['enhanced'][metric]
        change = enh_val - base_val
        emoji = "✅" if change > 0 else "❌" if change < 0 else "➖"
        
        metric_name = metric.replace('avg_', '').title()
        print(f"{metric_name:<20} {base_val:<15.2%} {enh_val:<15.2%} {change:+.2%} {emoji}")
    
    # Feature importance summary
    print(f"\n{'='*70}")
    print("📈 MOST IMPORTANT FEATURES (Across all tests)")
    print(f"{'='*70}\n")
    
    # Aggregate feature importance
    feature_counts = {}
    for key, features in all_feature_importance.items():
        for name, score in features:
            if name not in feature_counts:
                feature_counts[name] = {'count': 0, 'total_score': 0}
            feature_counts[name]['count'] += 1
            feature_counts[name]['total_score'] += score
    
    # Sort by count then score
    sorted_features = sorted(
        feature_counts.items(),
        key=lambda x: (x[1]['count'], x[1]['total_score']),
        reverse=True
    )[:15]
    
    new_indicators = ['stoch_k', 'stoch_d', 'stoch_divergence', 'williams_r', 
                      'obv', 'obv_sma', 'obv_ratio', 'obv_change',
                      'ichimoku_tenkan', 'ichimoku_kijun', 'ichimoku_senkou_a',
                      'ichimoku_senkou_b', 'ichimoku_cloud_thickness', 
                      'ichimoku_cloud_position', 'ichimoku_tk_diff',
                      'adx', 'plus_di', 'minus_di', 'adx_trend_direction',
                      'market_regime', 'volatility_regime', 'trend_regime',
                      'volatility_cluster', 'trend_strength']
    
    print(f"{'Rank':<6} {'Feature':<30} {'Appearances':<15} {'Is New?':<10}")
    print("-" * 65)
    for i, (name, data) in enumerate(sorted_features, 1):
        is_new = "✨ YES" if name in new_indicators else ""
        print(f"{i:<6} {name:<30} {data['count']}/4{'':<10} {is_new}")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'n_features_selected': N_FEATURES,
        'focus': 'direction_accuracy',
        'results': {k: {m: [float(v) for v in vals] for m, vals in v.items()} 
                   for k, v in results.items()},
        'summary': {k: {m: float(v) for m, v in metrics.items()} 
                   for k, metrics in summary.items()},
        'top_features': [(name, data['count']) for name, data in sorted_features]
    }
    
    output_path = 'ml_reports/direction_benchmark.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    
    return results, summary


if __name__ == "__main__":
    run_direction_benchmark()
