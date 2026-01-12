"""
Enhanced ML System with Feature Selection & Hyperparameter Tuning
Improvements:
1. Feature importance ranking and selection using SelectKBest
2. Optuna-based hyperparameter optimization
3. TimeSeriesSplit cross-validation
"""
import os
import sys
import warnings
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import sqlite3
import joblib
import logging

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

from crypto_ai.features import FeatureEngineer

# ML Libraries
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna not available. Install: pip install optuna")


class EnhancedCryptoMLSystem:
    """
    Enhanced ML system with:
    - Feature importance analysis & selection
    - Optuna hyperparameter tuning
    - TimeSeriesSplit cross-validation
    """
    
    def __init__(self, db_path='data/ml_crypto_data.db', use_feature_selection=True,
                 use_hyperparameter_tuning=True, n_optuna_trials=20):
        """Initialize the enhanced ML system"""
        self.db_path = db_path
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.feature_importance = {}
        self.best_params = {}
        
        self.use_feature_selection = use_feature_selection
        self.use_hyperparameter_tuning = use_hyperparameter_tuning
        self.n_optuna_trials = n_optuna_trials
        self.n_features_to_select = 40  # Top N features to keep
        
        self.feature_engineer = FeatureEngineer()
        
        # Create directories
        os.makedirs('ml_models', exist_ok=True)
        os.makedirs('ml_reports', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        self._setup_logging()
        
        logging.info("🧠 Enhanced Crypto ML System initialized")
        logging.info(f"   Feature Selection: {use_feature_selection}")
        logging.info(f"   Hyperparameter Tuning: {use_hyperparameter_tuning}")
        if use_hyperparameter_tuning:
            logging.info(f"   Optuna Trials: {n_optuna_trials}")
    
    def _setup_logging(self):
        """Setup logging"""
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        logging.root.addHandler(console_handler)
        logging.root.setLevel(logging.INFO)
    
    def load_data(self, symbol: str, timeframe: str, months_back: int = None) -> pd.DataFrame:
        """Load data from database"""
        try:
            if not os.path.exists(self.db_path):
                logging.error(f"❌ Database not found: {self.db_path}")
                return pd.DataFrame()
            
            conn = sqlite3.connect(self.db_path)
            
            if months_back is None:
                lookback_config = {
                    '5m': 1, '15m': 2, '1h': 6, '4h': 12, '1d': 24
                }
                months_back = lookback_config.get(timeframe, 6)
            
            days_back = months_back * 30
            
            query = """
            SELECT timestamp, open, high, low, close, volume
            FROM price_data 
            WHERE symbol = ? AND timeframe = ? 
            AND timestamp >= datetime('now', '-{} days')
            ORDER BY timestamp
            """.format(days_back)
            
            df = pd.read_sql_query(query, conn, params=(symbol, timeframe))
            conn.close()
            
            if df.empty:
                logging.warning(f"⚠️ No data found for {symbol} {timeframe}")
                return pd.DataFrame()
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            logging.info(f"📊 Loaded {len(df)} candles for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            logging.error(f"❌ Error loading data: {e}")
            return pd.DataFrame()
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features using enhanced FeatureEngineer"""
        if df.empty:
            return df
        
        df_features = self.feature_engineer.create_features(df)
        
        initial_len = len(df_features)
        df_features.dropna(inplace=True)
        dropped = initial_len - len(df_features)
        
        if dropped > 0:
            logging.info(f"ℹ️ Dropped {dropped} rows with NaN values")
        
        logging.info(f"✅ Created {len(df_features.columns)} features from {len(df_features)} samples")
        return df_features
    
    def prepare_data(self, df: pd.DataFrame, prediction_type: str = 'price') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for ML models"""
        if df.empty:
            return np.array([]), np.array([]), []
        
        if prediction_type == 'price':
            df['target'] = df['close'].shift(-1) / df['close'] - 1
        elif prediction_type == 'direction':
            df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'target']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].values
        y = df['target'].values
        
        # Remove NaN rows
        valid_idx = ~np.isnan(y)
        X = X[valid_idx]
        y = y[valid_idx]
        
        valid_idx = ~np.isnan(X).any(axis=1)
        X = X[valid_idx]
        y = y[valid_idx]
        
        logging.info(f"📊 Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y, feature_cols
    
    def select_features(self, X: np.ndarray, y: np.ndarray, feature_names: List[str],
                       n_features: int = 40) -> Tuple[np.ndarray, List[str], Dict]:
        """
        Select top features using SelectKBest with mutual information
        Returns transformed X, selected feature names, and importance scores
        """
        if not self.use_feature_selection or len(feature_names) <= n_features:
            importance = {name: 1.0 for name in feature_names}
            return X, feature_names, importance
        
        logging.info(f"🔍 Selecting top {n_features} features from {len(feature_names)}...")
        
        # Use mutual information for feature selection
        selector = SelectKBest(score_func=mutual_info_regression, k=n_features)
        X_selected = selector.fit_transform(X, y)
        
        # Get feature scores
        scores = selector.scores_
        feature_scores = dict(zip(feature_names, scores))
        
        # Get selected feature mask
        selected_mask = selector.get_support()
        selected_features = [name for name, selected in zip(feature_names, selected_mask) if selected]
        
        # Sort by importance
        sorted_features = sorted(
            [(name, feature_scores[name]) for name in selected_features],
            key=lambda x: x[1], reverse=True
        )
        
        # Log top 10 features
        logging.info("📊 Top 10 features:")
        for name, score in sorted_features[:10]:
            logging.info(f"   {name}: {score:.4f}")
        
        importance = {name: score for name, score in sorted_features}
        
        return X_selected, selected_features, importance
    
    def _tune_lightgbm_params(self, X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Tune LightGBM hyperparameters using Optuna"""
        if not OPTUNA_AVAILABLE or not self.use_hyperparameter_tuning:
            return {
                'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 7,
                'num_leaves': 31, 'min_child_samples': 20, 'subsample': 0.8,
                'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1
            }
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 800),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'num_leaves': trial.suggest_int('num_leaves', 15, 63),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1.0, log=True),
            }
            
            model = lgb.LGBMRegressor(**params, random_state=42, verbose=-1, force_col_wise=True)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                     callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            
            pred = model.predict(X_val)
            return mean_squared_error(y_val, pred)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_optuna_trials, show_progress_bar=False)
        
        logging.info(f"✨ LightGBM best MSE: {study.best_value:.6f}")
        return study.best_params
    
    def _tune_xgboost_params(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Tune XGBoost hyperparameters using Optuna"""
        if not OPTUNA_AVAILABLE or not self.use_hyperparameter_tuning:
            return {
                'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 6,
                'min_child_weight': 3, 'subsample': 0.8, 'colsample_bytree': 0.8,
                'gamma': 0.1, 'reg_alpha': 0.1, 'reg_lambda': 1.0
            }
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 800),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0.01, 1.0, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            }
            
            model = xgb.XGBRegressor(**params, random_state=42, verbosity=0)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            pred = model.predict(X_val)
            return mean_squared_error(y_val, pred)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_optuna_trials, show_progress_bar=False)
        
        logging.info(f"✨ XGBoost best MSE: {study.best_value:.6f}")
        return study.best_params
    
    def _tune_catboost_params(self, X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Tune CatBoost hyperparameters using Optuna"""
        if not OPTUNA_AVAILABLE or not self.use_hyperparameter_tuning:
            return {
                'iterations': 500, 'learning_rate': 0.05, 'depth': 6, 'l2_leaf_reg': 3
            }
        
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 100, 800),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            }
            
            model = cb.CatBoostRegressor(**params, random_seed=42, verbose=False)
            model.fit(X_train, y_train, eval_set=(X_val, y_val),
                     early_stopping_rounds=50, verbose=False)
            
            pred = model.predict(X_val)
            return mean_squared_error(y_val, pred)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_optuna_trials, show_progress_bar=False)
        
        logging.info(f"✨ CatBoost best MSE: {study.best_value:.6f}")
        return study.best_params
    
    def train_with_cv(self, symbol: str, timeframe: str) -> Dict:
        """
        Train models with TimeSeriesSplit cross-validation and hyperparameter tuning
        Returns metrics dictionary
        """
        logging.info(f"\n{'='*60}")
        logging.info(f"🚀 Training Enhanced Models: {symbol} {timeframe}")
        logging.info(f"{'='*60}\n")
        
        # Load and prepare data
        df = self.load_data(symbol, timeframe)
        if df.empty or len(df) < 100:
            logging.error(f"❌ Insufficient data for {symbol} {timeframe}")
            return {}
        
        df_features = self.create_features(df)
        if df_features.empty:
            return {}
        
        X, y, feature_cols = self.prepare_data(df_features, prediction_type='price')
        if len(X) < 100:
            return {}
        
        # Feature selection
        if self.use_feature_selection:
            X, feature_cols, importance = self.select_features(
                X, y, feature_cols, n_features=self.n_features_to_select
            )
            self.feature_importance[f"{symbol}_{timeframe}"] = importance
        
        # TimeSeriesSplit cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        all_results = {
            'lightgbm': {'r2': [], 'mse': [], 'dir_acc': []},
            'xgboost': {'r2': [], 'mse': [], 'dir_acc': []},
            'catboost': {'r2': [], 'mse': [], 'dir_acc': []}
        }
        
        fold = 0
        for train_idx, test_idx in tscv.split(X):
            fold += 1
            logging.info(f"\n📂 Fold {fold}/{tscv.n_splits}")
            
            X_train_full, X_test = X[train_idx], X[test_idx]
            y_train_full, y_test = y[train_idx], y[test_idx]
            
            # Split train into train/val for tuning (80/20)
            val_size = int(len(X_train_full) * 0.8)
            X_train, X_val = X_train_full[:val_size], X_train_full[val_size:]
            y_train, y_val = y_train_full[:val_size], y_train_full[val_size:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            
            # Only tune on first fold
            if fold == 1 and self.use_hyperparameter_tuning:
                logging.info("🔧 Tuning hyperparameters (first fold only)...")
                
                if LIGHTGBM_AVAILABLE:
                    self.best_params['lightgbm'] = self._tune_lightgbm_params(
                        X_train_scaled, y_train, X_val_scaled, y_val
                    )
                if XGBOOST_AVAILABLE:
                    self.best_params['xgboost'] = self._tune_xgboost_params(
                        X_train_scaled, y_train, X_val_scaled, y_val
                    )
                if CATBOOST_AVAILABLE:
                    self.best_params['catboost'] = self._tune_catboost_params(
                        X_train_scaled, y_train, X_val_scaled, y_val
                    )
            
            # Scale full train set for final training
            X_train_full_scaled = scaler.fit_transform(X_train_full)
            X_test_scaled = scaler.transform(X_test)
            
            # Train LightGBM
            if LIGHTGBM_AVAILABLE:
                params = self.best_params.get('lightgbm', {
                    'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 7
                })
                model = lgb.LGBMRegressor(**params, random_state=42, verbose=-1, force_col_wise=True)
                model.fit(X_train_full_scaled, y_train_full)
                
                pred = model.predict(X_test_scaled)
                r2 = r2_score(y_test, pred)
                mse = mean_squared_error(y_test, pred)
                dir_acc = (np.sign(pred) == np.sign(y_test)).mean()
                
                all_results['lightgbm']['r2'].append(r2)
                all_results['lightgbm']['mse'].append(mse)
                all_results['lightgbm']['dir_acc'].append(dir_acc)
                
                logging.info(f"   LightGBM - R²: {r2:.4f}, MSE: {mse:.6f}, Dir Acc: {dir_acc:.2%}")
            
            # Train XGBoost
            if XGBOOST_AVAILABLE:
                params = self.best_params.get('xgboost', {
                    'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 6
                })
                model = xgb.XGBRegressor(**params, random_state=42, verbosity=0)
                model.fit(X_train_full_scaled, y_train_full)
                
                pred = model.predict(X_test_scaled)
                r2 = r2_score(y_test, pred)
                mse = mean_squared_error(y_test, pred)
                dir_acc = (np.sign(pred) == np.sign(y_test)).mean()
                
                all_results['xgboost']['r2'].append(r2)
                all_results['xgboost']['mse'].append(mse)
                all_results['xgboost']['dir_acc'].append(dir_acc)
                
                logging.info(f"   XGBoost - R²: {r2:.4f}, MSE: {mse:.6f}, Dir Acc: {dir_acc:.2%}")
            
            # Train CatBoost
            if CATBOOST_AVAILABLE:
                params = self.best_params.get('catboost', {
                    'iterations': 500, 'learning_rate': 0.05, 'depth': 6
                })
                model = cb.CatBoostRegressor(**params, random_seed=42, verbose=False)
                model.fit(X_train_full_scaled, y_train_full)
                
                pred = model.predict(X_test_scaled)
                r2 = r2_score(y_test, pred)
                mse = mean_squared_error(y_test, pred)
                dir_acc = (np.sign(pred) == np.sign(y_test)).mean()
                
                all_results['catboost']['r2'].append(r2)
                all_results['catboost']['mse'].append(mse)
                all_results['catboost']['dir_acc'].append(dir_acc)
                
                logging.info(f"   CatBoost - R²: {r2:.4f}, MSE: {mse:.6f}, Dir Acc: {dir_acc:.2%}")
        
        # Average CV results
        final_results = {}
        for model_name, metrics in all_results.items():
            if metrics['r2']:
                final_results[model_name] = {
                    'r2': float(np.mean(metrics['r2'])),
                    'mse': float(np.mean(metrics['mse'])),
                    'direction_accuracy': float(np.mean(metrics['dir_acc']))
                }
                logging.info(f"\n📊 {model_name.upper()} CV Average: R²={final_results[model_name]['r2']:.4f}, "
                           f"Dir Acc={final_results[model_name]['direction_accuracy']:.2%}")
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'num_features': len(feature_cols),
            'num_samples': len(X),
            'cv_folds': tscv.n_splits,
            'feature_selection': self.use_feature_selection,
            'hyperparameter_tuning': self.use_hyperparameter_tuning,
            'model_results': final_results,
            'best_params': self.best_params
        }


def run_enhanced_benchmark():
    """Run enhanced benchmark and compare with baseline"""
    print("\n" + "="*70)
    print("🚀 ENHANCED BENCHMARK - ML System with Improvements")
    print("="*70 + "\n")
    
    # Initialize enhanced system
    ml_system = EnhancedCryptoMLSystem(
        use_feature_selection=True,
        use_hyperparameter_tuning=True,
        n_optuna_trials=15  # Fewer trials for speed, increase for better tuning
    )
    
    symbols = ['BTC/USDT', 'ETH/USDT']
    timeframes = ['1h', '4h']
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'version': 'enhanced',
        'models': {}
    }
    
    for symbol in symbols:
        for timeframe in timeframes:
            key = f"{symbol}_{timeframe}"
            result = ml_system.train_with_cv(symbol, timeframe)
            if result:
                results['models'][key] = result
    
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
    output_path = 'ml_reports/enhanced_benchmark.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("📊 ENHANCED BENCHMARK COMPLETE")
    print(f"{'='*70}")
    if 'summary' in results:
        print(f"\n📈 Average Metrics:")
        print(f"   R²: {results['summary']['avg_r2']:.4f}")
        print(f"   MSE: {results['summary']['avg_mse']:.6f}")
        print(f"   Direction Accuracy: {results['summary']['avg_direction_accuracy']:.2%}")
    print(f"\n💾 Results saved to: {output_path}")
    
    # Load baseline and compare
    baseline_path = 'ml_reports/baseline_benchmark.json'
    if os.path.exists(baseline_path):
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)
        
        if 'summary' in baseline and 'summary' in results:
            print(f"\n{'='*70}")
            print("📊 COMPARISON: BASELINE vs ENHANCED")
            print(f"{'='*70}")
            
            r2_change = results['summary']['avg_r2'] - baseline['summary']['avg_r2']
            mse_change = results['summary']['avg_mse'] - baseline['summary']['avg_mse']
            dir_change = results['summary']['avg_direction_accuracy'] - baseline['summary']['avg_direction_accuracy']
            
            print(f"\n{'Metric':<25} {'Baseline':<15} {'Enhanced':<15} {'Change':<15}")
            print("-" * 70)
            print(f"{'R²':<25} {baseline['summary']['avg_r2']:<15.4f} {results['summary']['avg_r2']:<15.4f} {r2_change:+.4f} {'✅' if r2_change > 0 else '❌'}")
            print(f"{'MSE':<25} {baseline['summary']['avg_mse']:<15.6f} {results['summary']['avg_mse']:<15.6f} {mse_change:+.6f} {'✅' if mse_change < 0 else '❌'}")
            print(f"{'Direction Accuracy':<25} {baseline['summary']['avg_direction_accuracy']:<15.2%} {results['summary']['avg_direction_accuracy']:<15.2%} {dir_change:+.2%} {'✅' if dir_change > 0 else '❌'}")
    
    return results


if __name__ == "__main__":
    run_enhanced_benchmark()
