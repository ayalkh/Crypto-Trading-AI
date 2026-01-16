"""
Enhanced Model Assessment - With Hyperparameter Tuning
Tests each model with Optuna optimization and feature selection to top 50
Includes: LightGBM, XGBoost, CatBoost optimized
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
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna not available. Install: pip install optuna")

import joblib


def tune_lightgbm(X_train, y_train, X_val, y_val, n_trials=15):
    """Tune LightGBM hyperparameters using Optuna"""
    if not OPTUNA_AVAILABLE:
        return {
            'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 6,
            'num_leaves': 31, 'reg_alpha': 0.1, 'reg_lambda': 0.1
        }
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'random_state': 42, 'verbose': -1, 'force_col_wise': True
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
        return accuracy_score(y_val, model.predict(X_val))
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {**study.best_params, 'random_state': 42, 'verbose': -1, 'force_col_wise': True}


def tune_xgboost(X_train, y_train, X_val, y_val, n_trials=15):
    """Tune XGBoost hyperparameters using Optuna"""
    if not OPTUNA_AVAILABLE:
        return {
            'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 6,
            'subsample': 0.8, 'colsample_bytree': 0.8
        }
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42, 'verbosity': 0
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return accuracy_score(y_val, model.predict(X_val))
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {**study.best_params, 'random_state': 42, 'verbosity': 0}


def tune_catboost(X_train, y_train, X_val, y_val, n_trials=15):
    """Tune CatBoost hyperparameters using Optuna"""
    if not OPTUNA_AVAILABLE or not CB_AVAILABLE:
        return {
            'iterations': 300, 'learning_rate': 0.05, 'depth': 6,
            'l2_leaf_reg': 3.0
        }
    
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'depth': trial.suggest_int('depth', 3, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            'random_seed': 42, 'verbose': False
        }
        model = cb.CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val),
                  early_stopping_rounds=30, verbose=False)
        return accuracy_score(y_val, model.predict(X_val))
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {**study.best_params, 'random_seed': 42, 'verbose': False}


def train_and_evaluate_enhanced(symbol, timeframe, db_path='data/ml_crypto_data.db', n_trials=15):
    """Train and evaluate models with hyperparameter tuning"""
    
    print(f"\n{'='*80}")
    print(f"🔬 ENHANCED MODEL ASSESSMENT (with Optuna): {symbol} {timeframe}")
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
    
    # Create features using enhanced FeatureEngineer 
    fe = FeatureEngineer()
    df_features = fe.create_features(df)
    df_features.dropna(inplace=True)
    
    if len(df_features) < 200:
        print("   ⚠️  Insufficient data after feature engineering")
        return None
    
    print(f"   📊 Features generated: {len(df_features.columns)}")
    
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
    
    print(f"   📊 Data Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Feature selection to top 50
    n_features = 50
    selector = SelectKBest(score_func=mutual_info_classif, k=min(n_features, len(feature_cols)))
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_val_sel = selector.transform(X_val)
    X_test_sel = selector.transform(X_test)
    
    print(f"   📊 Selected top {X_train_sel.shape[1]} features")
    
    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_sel)
    X_val_s = scaler.transform(X_val_sel)
    X_test_s = scaler.transform(X_test_sel)
    
    results = {}
    test_predictions = {}
    
    print(f"\n{'='*80}")
    print(f"🤖 TUNED MODEL PERFORMANCE (Optuna trials: {n_trials})")
    print(f"{'='*80}\n")
    
    # ========== 1. LightGBM with tuning ==========
    print("1️⃣  LightGBM (tuning hyperparameters)...")
    lgb_params = tune_lightgbm(X_train_s, y_train, X_val_s, y_val, n_trials)
    print(f"   Best params: lr={lgb_params.get('learning_rate', 0.05):.3f}, depth={lgb_params.get('max_depth', 6)}")
    
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)],
                 callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    
    lgb_pred = lgb_model.predict(X_test_s)
    lgb_proba = lgb_model.predict_proba(X_test_s)[:, 1]
    
    results['lightgbm'] = {
        'accuracy': accuracy_score(y_test, lgb_pred),
        'precision': precision_score(y_test, lgb_pred, zero_division=0),
        'recall': recall_score(y_test, lgb_pred, zero_division=0),
        'f1': f1_score(y_test, lgb_pred, zero_division=0),
        'best_params': {k: v for k, v in lgb_params.items() if k not in ['random_state', 'verbose', 'force_col_wise']}
    }
    test_predictions['lightgbm'] = lgb_proba
    
    print(f"   Acc: {results['lightgbm']['accuracy']:.2%}, "
          f"Prec: {results['lightgbm']['precision']:.2%}, "
          f"Rec: {results['lightgbm']['recall']:.2%}, "
          f"F1: {results['lightgbm']['f1']:.2%}")
    
    # ========== 2. XGBoost with tuning ==========
    print("\n2️⃣  XGBoost (tuning hyperparameters)...")
    xgb_params = tune_xgboost(X_train_s, y_train, X_val_s, y_val, n_trials)
    print(f"   Best params: lr={xgb_params.get('learning_rate', 0.05):.3f}, depth={xgb_params.get('max_depth', 6)}")
    
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False)
    
    xgb_pred = xgb_model.predict(X_test_s)
    xgb_proba = xgb_model.predict_proba(X_test_s)[:, 1]
    
    results['xgboost'] = {
        'accuracy': accuracy_score(y_test, xgb_pred),
        'precision': precision_score(y_test, xgb_pred, zero_division=0),
        'recall': recall_score(y_test, xgb_pred, zero_division=0),
        'f1': f1_score(y_test, xgb_pred, zero_division=0),
        'best_params': {k: v for k, v in xgb_params.items() if k not in ['random_state', 'verbosity']}
    }
    test_predictions['xgboost'] = xgb_proba
    
    print(f"   Acc: {results['xgboost']['accuracy']:.2%}, "
          f"Prec: {results['xgboost']['precision']:.2%}, "
          f"Rec: {results['xgboost']['recall']:.2%}, "
          f"F1: {results['xgboost']['f1']:.2%}")
    
    # ========== 3. CatBoost with tuning ==========
    if CB_AVAILABLE:
        print("\n3️⃣  CatBoost (tuning hyperparameters)...")
        cb_params = tune_catboost(X_train_s, y_train, X_val_s, y_val, n_trials)
        print(f"   Best params: lr={cb_params.get('learning_rate', 0.05):.3f}, depth={cb_params.get('depth', 6)}")
        
        cb_model = cb.CatBoostClassifier(**cb_params)
        cb_model.fit(X_train_s, y_train, eval_set=(X_val_s, y_val),
                    early_stopping_rounds=50, verbose=False)
        
        cb_pred = cb_model.predict(X_test_s)
        cb_proba = cb_model.predict_proba(X_test_s)[:, 1]
        
        results['catboost'] = {
            'accuracy': accuracy_score(y_test, cb_pred),
            'precision': precision_score(y_test, cb_pred, zero_division=0),
            'recall': recall_score(y_test, cb_pred, zero_division=0),
            'f1': f1_score(y_test, cb_pred, zero_division=0),
            'best_params': {k: v for k, v in cb_params.items() if k not in ['random_seed', 'verbose']}
        }
        test_predictions['catboost'] = cb_proba
        
        print(f"   Acc: {results['catboost']['accuracy']:.2%}, "
              f"Prec: {results['catboost']['precision']:.2%}, "
              f"Rec: {results['catboost']['recall']:.2%}, "
              f"F1: {results['catboost']['f1']:.2%}")
    else:
        print("\n3️⃣  CatBoost (Skipped - Not Installed)")
    
    # ========== ENSEMBLE PREDICTIONS ==========
    print(f"\n{'='*80}")
    print("🎯 ENSEMBLE PERFORMANCE")
    print(f"{'='*80}\n")
    
    # Equal weights ensemble
    available_probas = list(test_predictions.values())
    min_len = min(len(p) for p in available_probas)
    aligned_probas = [p[:min_len] for p in available_probas]
    
    equal_ensemble_proba = np.mean(aligned_probas, axis=0)
    equal_ensemble_pred = (equal_ensemble_proba >= 0.5).astype(int)
    y_test_aligned = y_test[:min_len]
    
    results['ensemble'] = {
        'accuracy': accuracy_score(y_test_aligned, equal_ensemble_pred),
        'precision': precision_score(y_test_aligned, equal_ensemble_pred, zero_division=0),
        'recall': recall_score(y_test_aligned, equal_ensemble_pred, zero_division=0),
        'f1': f1_score(y_test_aligned, equal_ensemble_pred, zero_division=0)
    }
    
    print(f"📊 Ensemble: Acc: {results['ensemble']['accuracy']:.2%}, F1: {results['ensemble']['f1']:.2%}")
    
    return results


def run_enhanced_assessment():
    """Run enhanced assessment on key timeframes"""
    print("\n" + "="*80)
    print("🔬 ENHANCED MODEL ASSESSMENT - WITH OPTUNA HYPERPARAMETER TUNING")
    print("="*80)
    print(f"   Using new feature engineering (116+ features)")
    print(f"   Feature selection: Top 50 features")
    print(f"   Hyperparameter tuning: Optuna (15 trials per model)")
    
    test_cases = [
        ('BTC/USDT', '15m'),
        ('ETH/USDT', '15m'),
        ('BTC/USDT', '4h'),
        ('ETH/USDT', '4h'),
    ]
    
    all_results = {}
    
    for symbol, timeframe in test_cases:
        result = train_and_evaluate_enhanced(symbol, timeframe, n_trials=15)
        if result:
            all_results[f"{symbol}_{timeframe}"] = result
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 FINAL SUMMARY - ENHANCED (Tuned)")
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
        'config': {
            'hyperparameter_tuning': True,
            'n_optuna_trials': 15,
            'feature_selection': 50
        },
        'results': {
            key: {
                model: {k: float(v) if isinstance(v, (float, np.floating)) else v
                       for k, v in metrics.items()}
                for model, metrics in res.items()
            }
            for key, res in all_results.items()
        }
    }
    
    output_path = 'ml_reports/enhanced_assessment_v2.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    print(f"\n{'='*80}\n")
    
    return all_results


if __name__ == "__main__":
    run_enhanced_assessment()
