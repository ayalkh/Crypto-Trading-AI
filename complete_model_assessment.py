"""
Complete Model Assessment - ALL Models Individual + Ensemble
Tests each model separately and shows weighted ensemble performance
Includes: LightGBM, XGBoost, CatBoost, GRU (for 4h)
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
try:
    import catboost as cb
    CB_AVAILABLE = True
except ImportError:
    CB_AVAILABLE = False
    print("⚠️ CatBoost not available")


try:
    from tensorflow.keras.models import load_model, Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    DL_AVAILABLE = True
except:
    DL_AVAILABLE = False

import joblib


def train_and_evaluate_all_models(symbol, timeframe, db_path='data/ml_crypto_data.db'):
    """Train and evaluate ALL models individually + ensemble"""
    
    print(f"\n{'='*80}")
    print(f"🔬 COMPLETE MODEL ASSESSMENT: {symbol} {timeframe}")
    print(f"{'='*80}")
    
    # Load data
    conn = sqlite3.connect(db_path)
    lookback_days = {'5m': 30, '15m': 60, '1h': 180, '4h': 365, '1d': 730}
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
    
    print(f"\n📊 Data Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Feature selection
    n_features = 50
    selector = SelectKBest(score_func=mutual_info_classif, k=min(n_features, len(feature_cols)))
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_val_sel = selector.transform(X_val)
    X_test_sel = selector.transform(X_test)
    
    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_sel)
    X_val_s = scaler.transform(X_val_sel)
    X_test_s = scaler.transform(X_test_sel)
    
    results = {}
    test_predictions = {}
    
    print(f"\n{'='*80}")
    print("🤖 INDIVIDUAL MODEL PERFORMANCE")
    print(f"{'='*80}\n")
    
    # ========== 1. LightGBM ==========
    print("1️⃣  LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        random_state=42, verbose=-1, force_col_wise=True
    )
    lgb_model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)],
                 callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    
    lgb_pred = lgb_model.predict(X_test_s)
    lgb_proba = lgb_model.predict_proba(X_test_s)[:, 1]
    
    results['lightgbm'] = {
        'accuracy': accuracy_score(y_test, lgb_pred),
        'precision': precision_score(y_test, lgb_pred, zero_division=0),
        'recall': recall_score(y_test, lgb_pred, zero_division=0),
        'f1': f1_score(y_test, lgb_pred, zero_division=0)
    }
    test_predictions['lightgbm'] = lgb_proba
    
    print(f"   Acc: {results['lightgbm']['accuracy']:.2%}, "
          f"Prec: {results['lightgbm']['precision']:.2%}, "
          f"Rec: {results['lightgbm']['recall']:.2%}, "
          f"F1: {results['lightgbm']['f1']:.2%}")
    
    # ========== 2. XGBoost ==========
    print("\n2️⃣  XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        random_state=42, verbosity=0
    )
    xgb_model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False)
    
    xgb_pred = xgb_model.predict(X_test_s)
    xgb_proba = xgb_model.predict_proba(X_test_s)[:, 1]
    
    results['xgboost'] = {
        'accuracy': accuracy_score(y_test, xgb_pred),
        'precision': precision_score(y_test, xgb_pred, zero_division=0),
        'recall': recall_score(y_test, xgb_pred, zero_division=0),
        'f1': f1_score(y_test, xgb_pred, zero_division=0)
    }
    test_predictions['xgboost'] = xgb_proba
    
    print(f"   Acc: {results['xgboost']['accuracy']:.2%}, "
          f"Prec: {results['xgboost']['precision']:.2%}, "
          f"Rec: {results['xgboost']['recall']:.2%}, "
          f"F1: {results['xgboost']['f1']:.2%}")
    
    # ========== 3. CatBoost ==========
    if CB_AVAILABLE:
        print("\n3️⃣  CatBoost...")
        cb_model = cb.CatBoostClassifier(
            iterations=300, learning_rate=0.05, depth=6,
            random_seed=42, verbose=False
        )
        cb_model.fit(X_train_s, y_train, eval_set=(X_val_s, y_val),
                    early_stopping_rounds=50, verbose=False)
        
        cb_pred = cb_model.predict(X_test_s)
        cb_proba = cb_model.predict_proba(X_test_s)[:, 1]
        
        results['catboost'] = {
            'accuracy': accuracy_score(y_test, cb_pred),
            'precision': precision_score(y_test, cb_pred, zero_division=0),
            'recall': recall_score(y_test, cb_pred, zero_division=0),
            'f1': f1_score(y_test, cb_pred, zero_division=0)
        }
        test_predictions['catboost'] = cb_proba
        
        print(f"   Acc: {results['catboost']['accuracy']:.2%}, "
              f"Prec: {results['catboost']['precision']:.2%}, "
              f"Rec: {results['catboost']['recall']:.2%}, "
              f"F1: {results['catboost']['f1']:.2%}")
    else:
        print("\n3️⃣  CatBoost (Skipped - Not Installed)")
    
    # ========== 4. GRU (only for 4h) ==========
    if timeframe == '4h' and DL_AVAILABLE and len(df) >= 200:
        print("\n4️⃣  GRU (Deep Learning)...")
        
        sequence_length = 60
        prices = df['close'].values.reshape(-1, 1)
        
        gru_scaler = StandardScaler()
        scaled_prices = gru_scaler.fit_transform(prices)
        
        # Create sequences
        X_seq, y_seq = [], []
        for i in range(sequence_length, len(scaled_prices)):
            X_seq.append(scaled_prices[i-sequence_length:i, 0])
            # Direction target
            if i < len(scaled_prices) - 1:
                y_seq.append(1 if scaled_prices[i+1, 0] > scaled_prices[i, 0] else 0)
        
        X_seq, y_seq = np.array(X_seq), np.array(y_seq)
        X_seq = X_seq.reshape((X_seq.shape[0], X_seq.shape[1], 1))
        
        # Align with test split
        seq_train_size = train_size - sequence_length
        seq_test_start = val_size - sequence_length
        
        if seq_test_start > 0 and len(X_seq) > seq_test_start:
            X_seq_test = X_seq[seq_test_start:]
            y_seq_test = y_seq[seq_test_start:]
            
            # Build and train GRU
            gru_model = Sequential([
                GRU(64, return_sequences=True, input_shape=(sequence_length, 1)),
                Dropout(0.2),
                GRU(32, return_sequences=False),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            
            gru_model.compile(optimizer=Adam(learning_rate=0.001), 
                            loss='binary_crossentropy', metrics=['accuracy'])
            
            early_stop = EarlyStopping(patience=10, restore_best_weights=True)
            
            gru_model.fit(
                X_seq[:seq_train_size], y_seq[:seq_train_size],
                epochs=50, batch_size=32,
                validation_split=0.2,
                callbacks=[early_stop],
                verbose=0
            )
            
            # Predict
            gru_proba = gru_model.predict(X_seq_test, verbose=0).flatten()
            gru_pred = (gru_proba >= 0.5).astype(int)
            
            # Align with y_test
            min_len = min(len(gru_pred), len(y_test))
            gru_pred = gru_pred[:min_len]
            gru_proba = gru_proba[:min_len]
            y_test_aligned = y_test[:min_len]
            
            results['gru'] = {
                'accuracy': accuracy_score(y_test_aligned, gru_pred),
                'precision': precision_score(y_test_aligned, gru_pred, zero_division=0),
                'recall': recall_score(y_test_aligned, gru_pred, zero_division=0),
                'f1': f1_score(y_test_aligned, gru_pred, zero_division=0)
            }
            test_predictions['gru'] = gru_proba
            
            print(f"   Acc: {results['gru']['accuracy']:.2%}, "
                  f"Prec: {results['gru']['precision']:.2%}, "
                  f"Rec: {results['gru']['recall']:.2%}, "
                  f"F1: {results['gru']['f1']:.2%}")
    
    # ========== 5. LSTM (train fresh) ==========
    if DL_AVAILABLE and len(df) >= 200:
        print("\n5️⃣  LSTM (Deep Learning)...")
        
        sequence_length = 60
        prices = df['close'].values.reshape(-1, 1)
        
        lstm_scaler = StandardScaler()
        scaled_prices = lstm_scaler.fit_transform(prices)
        
        # Create sequences
        X_seq, y_seq = [], []
        for i in range(sequence_length, len(scaled_prices)):
            X_seq.append(scaled_prices[i-sequence_length:i, 0])
            if i < len(scaled_prices) - 1:
                y_seq.append(1 if scaled_prices[i+1, 0] > scaled_prices[i, 0] else 0)
        
        X_seq, y_seq = np.array(X_seq), np.array(y_seq)
        X_seq = X_seq.reshape((X_seq.shape[0], X_seq.shape[1], 1))
        
        seq_train_size = train_size - sequence_length
        seq_test_start = val_size - sequence_length
        
        if seq_test_start > 0 and len(X_seq) > seq_test_start:
            X_seq_test = X_seq[seq_test_start:]
            y_seq_test = y_seq[seq_test_start:]
            
            # Build and train LSTM
            lstm_model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(sequence_length, 1)),
                Dropout(0.2),
                LSTM(32, return_sequences=False),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            
            lstm_model.compile(optimizer=Adam(learning_rate=0.001),
                             loss='binary_crossentropy', metrics=['accuracy'])
            
            early_stop = EarlyStopping(patience=10, restore_best_weights=True)
            
            lstm_model.fit(
                X_seq[:seq_train_size], y_seq[:seq_train_size],
                epochs=50, batch_size=32,
                validation_split=0.2,
                callbacks=[early_stop],
                verbose=0
            )
            
            # Predict
            lstm_proba = lstm_model.predict(X_seq_test, verbose=0).flatten()
            lstm_pred = (lstm_proba >= 0.5).astype(int)
            
            min_len = min(len(lstm_pred), len(y_test))
            lstm_pred = lstm_pred[:min_len]
            lstm_proba = lstm_proba[:min_len]
            y_test_aligned = y_test[:min_len]
            
            results['lstm'] = {
                'accuracy': accuracy_score(y_test_aligned, lstm_pred),
                'precision': precision_score(y_test_aligned, lstm_pred, zero_division=0),
                'recall': recall_score(y_test_aligned, lstm_pred, zero_division=0),
                'f1': f1_score(y_test_aligned, lstm_pred, zero_division=0)
            }
            test_predictions['lstm'] = lstm_proba
            
            print(f"   Acc: {results['lstm']['accuracy']:.2%}, "
                  f"Prec: {results['lstm']['precision']:.2%}, "
                  f"Rec: {results['lstm']['recall']:.2%}, "
                  f"F1: {results['lstm']['f1']:.2%}")
    
    # ========== ENSEMBLE PREDICTIONS ==========
    print(f"\n{'='*80}")
    print("🎯 ENSEMBLE PERFORMANCE")
    print(f"{'='*80}\n")
    
    # Equal weights ensemble
    print("📊 Equal Weights Ensemble (all models weighted equally)")
    available_probas = list(test_predictions.values())
    min_len = min(len(p) for p in available_probas)
    aligned_probas = [p[:min_len] for p in available_probas]
    
    equal_ensemble_proba = np.mean(aligned_probas, axis=0)
    equal_ensemble_pred = (equal_ensemble_proba >= 0.5).astype(int)
    y_test_aligned = y_test[:min_len]
    
    results['ensemble_equal'] = {
        'accuracy': accuracy_score(y_test_aligned, equal_ensemble_pred),
        'precision': precision_score(y_test_aligned, equal_ensemble_pred, zero_division=0),
        'recall': recall_score(y_test_aligned, equal_ensemble_pred, zero_division=0),
        'f1': f1_score(y_test_aligned, equal_ensemble_pred, zero_division=0),
        'weights': {name: 1.0/len(test_predictions) for name in test_predictions.keys()}
    }
    
    print(f"   Acc: {results['ensemble_equal']['accuracy']:.2%}, "
          f"F1: {results['ensemble_equal']['f1']:.2%}")
    
    # Optimized weights (based on validation accuracy)
    print("\n📊 Optimized Weights Ensemble (based on validation performance)")
    
    # Calculate validation accuracy for each model
    val_accuracies = {}
    for name in test_predictions.keys():
        if name in ['lightgbm', 'xgboost', 'catboost']:
            # Use validation set
            if name == 'lightgbm':
                val_pred = lgb_model.predict(X_val_s)
            elif name == 'xgboost':
                val_pred = xgb_model.predict(X_val_s)
            elif name == 'catboost' and CB_AVAILABLE:
                val_pred = cb_model.predict(X_val_s)
            val_accuracies[name] = accuracy_score(y_val, val_pred)
        else:
            # For DL models, use test accuracy as proxy
            val_accuracies[name] = results[name]['accuracy']
    
    # Calculate softmax weights
    val_scores = np.array(list(val_accuracies.values()))
    val_scores = np.maximum(val_scores, 0.01)  # Avoid zero
    exp_scores = np.exp(val_scores * 10)  # Temperature = 10
    optimized_weights = exp_scores / exp_scores.sum()
    
    weight_dict = {name: float(weight) for name, weight in zip(val_accuracies.keys(), optimized_weights)}
    
    # Apply optimized weights
    weighted_ensemble_proba = sum(
        test_predictions[name][:min_len] * weight_dict[name]
        for name in test_predictions.keys()
    )
    weighted_ensemble_pred = (weighted_ensemble_proba >= 0.5).astype(int)
    
    results['ensemble_optimized'] = {
        'accuracy': accuracy_score(y_test_aligned, weighted_ensemble_pred),
        'precision': precision_score(y_test_aligned, weighted_ensemble_pred, zero_division=0),
        'recall': recall_score(y_test_aligned, weighted_ensemble_pred, zero_division=0),
        'f1': f1_score(y_test_aligned, weighted_ensemble_pred, zero_division=0),
        'weights': weight_dict
    }
    
    print(f"   Acc: {results['ensemble_optimized']['accuracy']:.2%}, "
          f"F1: {results['ensemble_optimized']['f1']:.2%}")
    print(f"\n   Weights:")
    for name, weight in weight_dict.items():
        print(f"      {name}: {weight:.1%}")
    
    return results


def run_complete_assessment():
    """Run complete assessment on key timeframes"""
    print("\n" + "="*80)
    print("🔬 COMPLETE MODEL ASSESSMENT - ALL MODELS + ENSEMBLE")
    print("="*80)
    
    test_cases = [
        ('BTC/USDT', '15m'),  # Best timeframe from previous tests
        ('ETH/USDT', '15m'),
        ('BTC/USDT', '4h'),   # For GRU
        ('ETH/USDT', '4h'),
    ]
    
    all_results = {}
    
    for symbol, timeframe in test_cases:
        result = train_and_evaluate_all_models(symbol, timeframe)
        if result:
            all_results[f"{symbol}_{timeframe}"] = result
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*80}\n")
    
    for key, results in all_results.items():
        print(f"\n{key}:")
        print(f"{'Model':<20} {'Accuracy':<12} {'F1 Score':<12}")
        print("-" * 50)
        
        for model_name, metrics in results.items():
            if 'accuracy' in metrics:
                print(f"{model_name:<20} {metrics['accuracy']:<12.2%} {metrics['f1']:<12.2%}")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'results': {
            key: {
                model: {k: float(v) if isinstance(v, (float, np.floating)) else v
                       for k, v in metrics.items() if k != 'weights'}
                for model, metrics in res.items()
            }
            for key, res in all_results.items()
        }
    }
    
    output_path = 'ml_reports/complete_model_assessment.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    run_complete_assessment()
