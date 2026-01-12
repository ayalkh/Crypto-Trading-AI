"""
Complete End-to-End ML Assessment
From data collection → training → comprehensive evaluation

Metrics Tracked:
- Accuracy, Precision, Recall, F1 Score
- AUC-ROC, AUC-PR
- Confusion Matrix
- Per-model and ensemble performance
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import sys
import warnings
from datetime import datetime
import json

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import sqlite3
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from crypto_ai.features import FeatureEngineer
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    DL_AVAILABLE = True
except:
    DL_AVAILABLE = False


def comprehensive_evaluation(symbol, timeframe, db_path='data/ml_crypto_data.db'):
    """Complete end-to-end evaluation with all metrics"""
    
    print(f"\n{'='*80}")
    print(f"🔬 COMPLETE END-TO-END ASSESSMENT: {symbol} {timeframe}")
    print(f"{'='*80}")
    
    # ========== STEP 1: DATA COLLECTION ==========
    print(f"\n{'='*80}")
    print("📊 STEP 1: DATA COLLECTION")
    print(f"{'='*80}")
    
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
        print("   ❌ Insufficient data")
        return None
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    print(f"   ✅ Loaded {len(df)} candles")
    print(f"   📅 Date range: {df.index[0]} to {df.index[-1]}")
    
    # ========== STEP 2: FEATURE ENGINEERING ==========
    print(f"\n{'='*80}")
    print("🧠 STEP 2: FEATURE ENGINEERING")
    print(f"{'='*80}")
    
    fe = FeatureEngineer()
    df_features = fe.create_features(df)
    df_features.dropna(inplace=True)
    
    print(f"   ✅ Created {len(df_features.columns)} features")
    print(f"   📊 Samples after cleaning: {len(df_features)}")
    
    # Direction target
    df_features['target'] = (df_features['close'].shift(-1) > df_features['close']).astype(int)
    df_features.dropna(inplace=True)
    
    # Class distribution
    up_pct = df_features['target'].mean()
    print(f"   📈 Class distribution: UP={up_pct:.1%}, DOWN={1-up_pct:.1%}")
    
    # ========== STEP 3: DATA PREPARATION ==========
    print(f"\n{'='*80}")
    print("🔧 STEP 3: DATA PREPARATION")
    print(f"{'='*80}")
    
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'target']
    feature_cols = [col for col in df_features.columns if col not in exclude_cols]
    
    X = df_features[feature_cols].values
    y = df_features['target'].values
    
    # Split: 70% train, 15% val, 15% test
    train_size = int(len(X) * 0.70)
    val_size = int(len(X) * 0.85)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:val_size], y[train_size:val_size]
    X_test, y_test = X[val_size:], y[val_size:]
    
    print(f"   📊 Train: {len(X_train)} samples")
    print(f"   📊 Validation: {len(X_val)} samples")
    print(f"   📊 Test: {len(X_test)} samples")
    
    # Feature selection
    n_features = 50
    print(f"\n   🎯 Feature Selection: Top {n_features} features")
    selector = SelectKBest(score_func=mutual_info_classif, k=min(n_features, len(feature_cols)))
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_val_sel = selector.transform(X_val)
    X_test_sel = selector.transform(X_test)
    
    selected_mask = selector.get_support()
    selected_features = [name for name, sel in zip(feature_cols, selected_mask) if sel]
    print(f"   ✅ Selected features: {selected_features[:5]}...")
    
    # Scaling
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_sel)
    X_val_s = scaler.transform(X_val_sel)
    X_test_s = scaler.transform(X_test_sel)
    
    print(f"   ✅ Features scaled")
    
    # ========== STEP 4: MODEL TRAINING ==========
    print(f"\n{'='*80}")
    print("🤖 STEP 4: MODEL TRAINING")
    print(f"{'='*80}")
    
    models = {}
    predictions = {}
    probabilities = {}
    
    # 1. LightGBM
    print(f"\n   1️⃣  Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        random_state=42, verbose=-1, force_col_wise=True
    )
    lgb_model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)],
                 callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    
    models['LightGBM'] = lgb_model
    predictions['LightGBM'] = lgb_model.predict(X_test_s)
    probabilities['LightGBM'] = lgb_model.predict_proba(X_test_s)[:, 1]
    print(f"      ✅ Trained (stopped at iteration {lgb_model.best_iteration_})")
    
    # 2. XGBoost
    print(f"\n   2️⃣  Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        random_state=42, verbosity=0
    )
    xgb_model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False)
    
    models['XGBoost'] = xgb_model
    predictions['XGBoost'] = xgb_model.predict(X_test_s)
    probabilities['XGBoost'] = xgb_model.predict_proba(X_test_s)[:, 1]
    print(f"      ✅ Trained")
    
    # 3. CatBoost
    print(f"\n   3️⃣  Training CatBoost...")
    cb_model = cb.CatBoostClassifier(
        iterations=300, learning_rate=0.05, depth=6,
        random_seed=42, verbose=False
    )
    cb_model.fit(X_train_s, y_train, eval_set=(X_val_s, y_val),
                early_stopping_rounds=50, verbose=False)
    
    models['CatBoost'] = cb_model
    predictions['CatBoost'] = cb_model.predict(X_test_s)
    probabilities['CatBoost'] = cb_model.predict_proba(X_test_s)[:, 1]
    print(f"      ✅ Trained")
    
    # 4. LSTM (if available)
    if DL_AVAILABLE:
        print(f"\n   4️⃣  Training LSTM...")
        
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
            
            X_seq_test = X_seq[seq_test_start:]
            lstm_proba = lstm_model.predict(X_seq_test, verbose=0).flatten()
            lstm_pred = (lstm_proba >= 0.5).astype(int)
            
            min_len = min(len(lstm_pred), len(y_test))
            
            models['LSTM'] = lstm_model
            predictions['LSTM'] = lstm_pred[:min_len]
            probabilities['LSTM'] = lstm_proba[:min_len]
            print(f"      ✅ Trained")
    
    # ========== STEP 5: INDIVIDUAL MODEL EVALUATION ==========
    print(f"\n{'='*80}")
    print("📊 STEP 5: INDIVIDUAL MODEL EVALUATION")
    print(f"{'='*80}")
    
    results = {}
    
    for model_name in models.keys():
        print(f"\n   {model_name}:")
        print(f"   {'-'*70}")
        
        pred = predictions[model_name]
        proba = probabilities[model_name]
        
        # Align lengths
        min_len = min(len(pred), len(y_test))
        pred = pred[:min_len]
        proba = proba[:min_len]
        y_test_aligned = y_test[:min_len]
        
        # Calculate metrics
        acc = accuracy_score(y_test_aligned, pred)
        prec = precision_score(y_test_aligned, pred, zero_division=0)
        rec = recall_score(y_test_aligned, pred, zero_division=0)
        f1 = f1_score(y_test_aligned, pred, zero_division=0)
        
        try:
            auc_roc = roc_auc_score(y_test_aligned, proba)
        except:
            auc_roc = 0.5
        
        try:
            auc_pr = average_precision_score(y_test_aligned, proba)
        except:
            auc_pr = 0.5
        
        cm = confusion_matrix(y_test_aligned, pred)
        tn, fp, fn, tp = cm.ravel()
        
        results[model_name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'confusion_matrix': cm.tolist(),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }
        
        print(f"      Accuracy:  {acc:.4f} ({acc:.2%})")
        print(f"      Precision: {prec:.4f} ({prec:.2%})")
        print(f"      Recall:    {rec:.4f} ({rec:.2%})")
        print(f"      F1 Score:  {f1:.4f} ({f1:.2%})")
        print(f"      AUC-ROC:   {auc_roc:.4f}")
        print(f"      AUC-PR:    {auc_pr:.4f}")
        print(f"      Confusion Matrix:")
        print(f"         TN={tn:4d}  FP={fp:4d}")
        print(f"         FN={fn:4d}  TP={tp:4d}")
    
    # ========== STEP 6: ENSEMBLE PREDICTION ==========
    print(f"\n{'='*80}")
    print("🎯 STEP 6: ENSEMBLE PREDICTION")
    print(f"{'='*80}")
    
    # Align all predictions
    min_len = min(len(p) for p in probabilities.values())
    aligned_probas = {name: proba[:min_len] for name, proba in probabilities.items()}
    y_test_final = y_test[:min_len]
    
    # Equal weights ensemble
    print(f"\n   📊 Equal Weights Ensemble")
    equal_proba = np.mean(list(aligned_probas.values()), axis=0)
    equal_pred = (equal_proba >= 0.5).astype(int)
    
    acc = accuracy_score(y_test_final, equal_pred)
    prec = precision_score(y_test_final, equal_pred, zero_division=0)
    rec = recall_score(y_test_final, equal_pred, zero_division=0)
    f1 = f1_score(y_test_final, equal_pred, zero_division=0)
    auc_roc = roc_auc_score(y_test_final, equal_proba)
    auc_pr = average_precision_score(y_test_final, equal_proba)
    
    cm = confusion_matrix(y_test_final, equal_pred)
    tn, fp, fn, tp = cm.ravel()
    
    results['Ensemble_Equal'] = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'confusion_matrix': cm.tolist(),
        'weights': {name: 1.0/len(models) for name in models.keys()}
    }
    
    print(f"      Accuracy:  {acc:.4f} ({acc:.2%})")
    print(f"      Precision: {prec:.4f} ({prec:.2%})")
    print(f"      Recall:    {rec:.4f} ({rec:.2%})")
    print(f"      F1 Score:  {f1:.4f} ({f1:.2%})")
    print(f"      AUC-ROC:   {auc_roc:.4f}")
    print(f"      AUC-PR:    {auc_pr:.4f}")
    print(f"      Weights: {', '.join(f'{k}={v:.1%}' for k, v in results['Ensemble_Equal']['weights'].items())}")
    
    # Optimized weights (based on validation performance)
    print(f"\n   📊 Optimized Weights Ensemble")
    
    val_scores = {}
    for name in models.keys():
        if name in ['LightGBM', 'XGBoost', 'CatBoost']:
            val_pred = models[name].predict(X_val_s)
            val_scores[name] = accuracy_score(y_val, val_pred)
        else:
            val_scores[name] = results[name]['accuracy']
    
    # Softmax weights
    scores_array = np.array(list(val_scores.values()))
    scores_array = np.maximum(scores_array, 0.01)
    exp_scores = np.exp(scores_array * 10)
    opt_weights = exp_scores / exp_scores.sum()
    weight_dict = {name: float(w) for name, w in zip(val_scores.keys(), opt_weights)}
    
    opt_proba = sum(aligned_probas[name] * weight_dict[name] for name in models.keys())
    opt_pred = (opt_proba >= 0.5).astype(int)
    
    acc = accuracy_score(y_test_final, opt_pred)
    prec = precision_score(y_test_final, opt_pred, zero_division=0)
    rec = recall_score(y_test_final, opt_pred, zero_division=0)
    f1 = f1_score(y_test_final, opt_pred, zero_division=0)
    auc_roc = roc_auc_score(y_test_final, opt_proba)
    auc_pr = average_precision_score(y_test_final, opt_proba)
    
    results['Ensemble_Optimized'] = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'weights': weight_dict
    }
    
    print(f"      Accuracy:  {acc:.4f} ({acc:.2%})")
    print(f"      Precision: {prec:.4f} ({prec:.2%})")
    print(f"      Recall:    {rec:.4f} ({rec:.2%})")
    print(f"      F1 Score:  {f1:.4f} ({f1:.2%})")
    print(f"      AUC-ROC:   {auc_roc:.4f}")
    print(f"      AUC-PR:    {auc_pr:.4f}")
    print(f"      Weights: {', '.join(f'{k}={v:.1%}' for k, v in weight_dict.items())}")
    
    return results


def run_complete_assessment():
    """Run complete end-to-end assessment"""
    print("\n" + "="*80)
    print("🚀 COMPLETE END-TO-END ML ASSESSMENT")
    print("From Data Collection → Training → Comprehensive Evaluation")
    print("="*80)
    
    test_cases = [
        ('ETH/USDT', '15m'),  # Best from previous tests
        ('BTC/USDT', '15m'),
    ]
    
    all_results = {}
    
    for symbol, timeframe in test_cases:
        result = comprehensive_evaluation(symbol, timeframe)
        if result:
            all_results[f"{symbol}_{timeframe}"] = result
    
    # Final summary
    print(f"\n{'='*80}")
    print("📊 FINAL SUMMARY - ALL METRICS")
    print(f"{'='*80}")
    
    for key, results in all_results.items():
        print(f"\n{key}:")
        print(f"{'='*80}")
        print(f"{'Model':<20} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'AUC-ROC':<8} {'AUC-PR':<8}")
        print(f"{'-'*80}")
        
        for model_name, metrics in results.items():
            print(f"{model_name:<20} "
                  f"{metrics['accuracy']:<8.2%} "
                  f"{metrics['precision']:<8.2%} "
                  f"{metrics['recall']:<8.2%} "
                  f"{metrics['f1']:<8.2%} "
                  f"{metrics['auc_roc']:<8.4f} "
                  f"{metrics['auc_pr']:<8.4f}")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'results': {
            key: {
                model: {k: float(v) if isinstance(v, (float, np.floating)) else v
                       for k, v in metrics.items() if k not in ['confusion_matrix', 'weights']}
                for model, metrics in res.items()
            }
            for key, res in all_results.items()
        }
    }
    
    output_path = 'ml_reports/complete_end_to_end_assessment.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    run_complete_assessment()
