"""
Optimized ML/DL Integration System for Crypto Trading
Implementation: LightGBM + XGBoost + CatBoost + GRU Ensemble
Database: ml_crypto_data.db (unified database)
Strategy: Best practices for crypto price prediction
Enhanced with dual logging (console + file)
"""
import os
import sys

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, OSError):
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')

import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import sqlite3
import joblib
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
from crypto_ai.features import FeatureEngineer

# Note: Logging will be configured in OptimizedCryptoMLSystem.__init__
# to support dual output (console + file)

# ML Libraries
try:
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, precision_score, recall_score, f1_score
    from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_regression
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("❌ Scikit-learn not available. Install: pip install scikit-learn")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM not available. Install: pip install lightgbm")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available. Install: pip install xgboost")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️ CatBoost not available. Install: pip install catboost")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import GRU, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False
    print("⚠️ TensorFlow not available. Install: pip install tensorflow")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna not available. Install: pip install optuna")

# GPU utilities
try:
    from crypto_ai.gpu_utils import (
        detect_gpu_availability, get_xgboost_gpu_params, 
        get_catboost_gpu_params, get_lightgbm_gpu_params,
        configure_tensorflow_gpu, XGBOOST_GPU_AVAILABLE, 
        CATBOOST_GPU_AVAILABLE, LIGHTGBM_GPU_AVAILABLE, TENSORFLOW_GPU_AVAILABLE
    )
    GPU_UTILS_AVAILABLE = True
except ImportError:
    GPU_UTILS_AVAILABLE = False
    XGBOOST_GPU_AVAILABLE = False
    CATBOOST_GPU_AVAILABLE = False
    LIGHTGBM_GPU_AVAILABLE = False
    TENSORFLOW_GPU_AVAILABLE = False


class OptimizedCryptoMLSystem:
    """
    Optimized ML system using ensemble of:
    - LightGBM (Primary - Best for tabular data)
    - XGBoost (Secondary - Industry standard)
    - CatBoost (Tertiary - Robust against overfitting)
    - GRU (Deep Learning - For 4h timeframe)
    """
    
    
    def __init__(self, db_path='data/ml_crypto_data.db', n_features=50, enable_tuning=False, n_trials=50):
        """Initialize the optimized ML system
        
        Args:
            db_path: Path to database
            n_features: Number of features to select (default: 50)
            enable_tuning: Enable hyperparameter tuning with Optuna
            n_trials: Number of Optuna trials for hyperparameter tuning
        """
        self.db_path = db_path
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.feature_engineer = FeatureEngineer()
        self.n_features = n_features
        self.enable_tuning = enable_tuning
        self.n_trials = n_trials
        self.feature_selectors = {}  # Store feature selectors
        self.model_performance = {}  # Store validation performance for dynamic weights
        
        # Create directories
        os.makedirs('ml_models', exist_ok=True)
        os.makedirs('ml_predictions', exist_ok=True)
        os.makedirs('ml_reports', exist_ok=True)
        os.makedirs('logs', exist_ok=True)  # For log files
        
        # Setup dual logging (console + file)
        self.log_file = self._setup_logging()
        
        logging.info("🧠 Optimized Crypto ML System initialized")
        logging.info(f"📁 Database: {self.db_path}")
        logging.info(f"📝 Log file: {self.log_file}")
        logging.info("✨ Enhanced Features:")
        logging.info(f"   • Feature Selection: Top {self.n_features} features")
        logging.info(f"   • Hyperparameter Tuning: {'Enabled' if self.enable_tuning else 'Disabled'}")
        logging.info(f"   • Dynamic Model Weights: Enabled")
        logging.info(f"✅ LightGBM: {LIGHTGBM_AVAILABLE}")
        logging.info(f"✅ XGBoost: {XGBOOST_AVAILABLE}")
        logging.info(f"✅ CatBoost: {CATBOOST_AVAILABLE}")
        logging.info(f"✅ TensorFlow: {DL_AVAILABLE}")
        logging.info(f"✅ Optuna: {OPTUNA_AVAILABLE}")
        
        # Detect and log GPU availability
        if GPU_UTILS_AVAILABLE:
            detect_gpu_availability(log_results=True)
            if DL_AVAILABLE:
                configure_tensorflow_gpu()
    
    def _setup_logging(self):
        """Setup logging to both console and file"""
        # Create timestamped log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'logs/ml_training_{timestamp}.log'
        
        # Remove existing handlers to avoid duplicates
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler (captures everything)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Console handler (for screen output)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Add handlers to root logger
        logging.root.addHandler(file_handler)
        logging.root.addHandler(console_handler)
        logging.root.setLevel(logging.DEBUG)
        
        return log_file
    
    def load_data(self, symbol: str, timeframe: str, months_back: int = None) -> pd.DataFrame:
        """
        Load data from unified database
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe ('5m', '15m', '1h', '4h', '1d')
            months_back: Months of data to load (None = all available)
        """
        try:
            if not os.path.exists(self.db_path):
                logging.error(f"❌ Database not found: {self.db_path}")
                return pd.DataFrame()
            
            conn = sqlite3.connect(self.db_path)
            
            # Optimized lookback periods per timeframe
            if months_back is None:
                lookback_config = {
                    '5m': 1,    # 1 month
                    '15m': 2,   # 2 months
                    '1h': 6,    # 6 months
                    '4h': 12,   # 12 months
                    '1d': 24    # 24 months
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
            
            logging.info(f"📊 Loaded {len(df)} candles for {symbol} {timeframe} ({months_back} months)")
            return df
            
        except Exception as e:
            logging.error(f"❌ Error loading data: {e}")
            return pd.DataFrame()
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive feature set optimized for gradient boosting
        DELEGATED to FeatureEngineer class for consistency
        """
        if df.empty:
            return df
            
        logging.info("🧠 Generating features using centralized FeatureEngineer...")
        df_features = self.feature_engineer.create_features(df)
        
        # Remove any rows with NaN values created during feature engineering
        initial_len = len(df)
        df_features.dropna(inplace=True)
        dropped = initial_len - len(df_features)
        
        if dropped > 0:
            logging.info(f"ℹ️ Dropped {dropped} rows with NaN values")
            
        logging.info(f"✅ Created {len(df_features.columns)} features from {len(df_features)} samples")
        return df_features
    
    # _add_rsi, _add_macd, _add_bollinger_bands, _add_atr are no longer needed here as they are in FeatureEngineer
    # Removed to cleanup code
    
    def prepare_data(self, df: pd.DataFrame, prediction_type: str = 'price') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for ML models
        
        Args:
            df: DataFrame with features
            prediction_type: 'price' or 'direction'
        
        Returns:
            X: Feature array
            y: Target array
            feature_names: List of feature names
        """
        if df.empty:
            return np.array([]), np.array([]), []
        
        # Create target variable
        if prediction_type == 'price':
            # Predict next candle's price change (%)
            df['target'] = df['close'].shift(-1) / df['close'] - 1
        elif prediction_type == 'direction':
            # Predict if next candle will be up (1) or down (0)
            df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        else:
            raise ValueError(f"Unknown prediction_type: {prediction_type}")
        
        # Exclude OHLCV and target from features
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'target']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Prepare X and y
        X = df[feature_cols].values
        y = df['target'].values
        
        # Remove rows with NaN in target
        valid_idx = ~np.isnan(y)
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Remove rows with NaN in features
        valid_idx = ~np.isnan(X).any(axis=1)
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Log target statistics
        if prediction_type == 'price':
            logging.info(f"📊 Target stats - Mean: {y.mean():.4%}, Std: {y.std():.4%}, Min: {y.min():.4%}, Max: {y.max():.4%}")
        else:
            logging.info(f"📊 Target distribution - UP: {(y==1).sum()} ({(y==1).mean():.1%}), DOWN: {(y==0).sum()} ({(y==0).mean():.1%})")
        
        logging.info(f"📊 Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y, feature_cols
    
    def train_ensemble(self, symbol: str, timeframe: str):
        """
        Train ensemble of models:
        - LightGBM (Primary)
        - XGBoost (Secondary)
        - CatBoost (Tertiary)
        - GRU (for 4h only)
        """
        logging.info(f"\n{'='*60}")
        logging.info(f"🚀 Training Ensemble for {symbol} {timeframe}")
        logging.info(f"{'='*60}\n")
        
        # Load data
        df = self.load_data(symbol, timeframe)
        if df.empty or len(df) < 100:
            logging.error(f"❌ Insufficient data for {symbol} {timeframe}")
            return False
        
        # Create features
        df_features = self.create_features(df)
        if df_features.empty:
            logging.error(f"❌ Failed to create features for {symbol} {timeframe}")
            return False
        
        # Train both price and direction models
        success_price = self._train_price_models(symbol, timeframe, df_features)
        success_direction = self._train_direction_models(symbol, timeframe, df_features)
        
        # Train GRU for 4h timeframe
        if timeframe == '4h' and DL_AVAILABLE:
            success_gru = self._train_gru_model(symbol, timeframe, df)
        else:
            success_gru = False
        
        return success_price or success_direction or success_gru
    
    def _train_price_models(self, symbol: str, timeframe: str, df: pd.DataFrame) -> bool:
        """Train price prediction models (regression) with feature selection and hyperparameter tuning"""
        logging.info(f"\n📈 Training Price Prediction Models")
        logging.info("-" * 60)
        
        # Prepare data
        X, y, feature_cols = self.prepare_data(df, prediction_type='price')
        
        if len(X) == 0:
            logging.error("❌ No valid data for price prediction")
            return False
        
        # Time series split (70% train, 15% validation, 15% test)
        train_size = int(len(X) * 0.70)
        val_size = int(len(X) * 0.85)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:val_size]
        y_val = y[train_size:val_size]
        X_test = X[val_size:]
        y_test = y[val_size:]
        
        logging.info(f"📊 Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # ========== FEATURE SELECTION ==========
        logging.info(f"\n🎯 Selecting top {self.n_features} features using f_regression...")
        selector = SelectKBest(score_func=f_regression, k=min(self.n_features, len(feature_cols)))
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_val_selected = selector.transform(X_val)
        X_test_selected = selector.transform(X_test)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        selected_features = [name for name, sel in zip(feature_cols, selected_mask) if sel]
        
        logging.info(f"✅ Selected {len(selected_features)} features")
        logging.info(f"   Top 5: {selected_features[:5]}")
        
        # Store feature selector
        self.feature_selectors[f"{symbol}_{timeframe}_price"] = selector
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_val_scaled = scaler.transform(X_val_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        
        # Store scaler and feature columns
        self.scalers[f"{symbol}_{timeframe}_price"] = scaler
        self.feature_columns = selected_features
        
        models = {}
        val_scores = {}  # Track validation R² for dynamic weights
        
        # ========== 1. LightGBM (PRIMARY) ==========
        if LIGHTGBM_AVAILABLE:
            logging.info("\n🌟 Training LightGBM (Primary)...")
            
            if self.enable_tuning and OPTUNA_AVAILABLE:
                logging.info(f"   🔧 Tuning hyperparameters ({self.n_trials} trials)...")
                
                def objective(trial):
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                        'max_depth': trial.suggest_int('max_depth', 3, 12),
                        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
                        'random_state': 42,
                        'verbose': -1,
                        'force_col_wise': True
                    }
                    model = lgb.LGBMRegressor(**params)
                    model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)],
                             callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
                    pred_val = model.predict(X_val_scaled)
                    return r2_score(y_val, pred_val)
                
                study = optuna.create_study(direction='maximize', study_name='lgb_price')
                study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
                best_params = study.best_params
                logging.info(f"   ✅ Best R²: {study.best_value:.4f}")
                lgb_model = lgb.LGBMRegressor(**best_params, verbose=-1, force_col_wise=True)
            else:
                lgb_model = lgb.LGBMRegressor(
                    n_estimators=500, learning_rate=0.05, max_depth=7, num_leaves=31,
                    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1, force_col_wise=True
                )
            
            lgb_model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)],
                         callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
            models['lightgbm'] = lgb_model
            
            # Evaluate
            pred_val = lgb_model.predict(X_val_scaled)
            pred_test = lgb_model.predict(X_test_scaled)
            r2_val = r2_score(y_val, pred_val)
            r2_test = r2_score(y_test, pred_test)
            mse_test = mean_squared_error(y_test, pred_test)
            direction_acc = (np.sign(pred_test) == np.sign(y_test)).mean()
            
            val_scores['lightgbm'] = r2_val
            logging.info(f"✅ LightGBM - Val R²: {r2_val:.4f}, Test R²: {r2_test:.4f}, MSE: {mse_test:.6f}, Dir Acc: {direction_acc:.2%}")
        
        # ========== 2. XGBoost (SECONDARY) ==========
        if XGBOOST_AVAILABLE:
            logging.info("\n🔥 Training XGBoost (Secondary)...")
            
            if self.enable_tuning and OPTUNA_AVAILABLE:
                logging.info(f"   🔧 Tuning hyperparameters ({self.n_trials} trials)...")
                
                def objective(trial):
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                        'max_depth': trial.suggest_int('max_depth', 3, 12),
                        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
                        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
                        'random_state': 42,
                        'verbosity': 0
                    }
                    model = xgb.XGBRegressor(**params)
                    model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
                    pred_val = model.predict(X_val_scaled)
                    return r2_score(y_val, pred_val)
                
                study = optuna.create_study(direction='maximize', study_name='xgb_price')
                study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
                best_params = study.best_params
                logging.info(f"   ✅ Best R²: {study.best_value:.4f}")
                xgb_model = xgb.XGBRegressor(**best_params, verbosity=0)
            else:
                xgb_model = xgb.XGBRegressor(
                    n_estimators=500, learning_rate=0.05, max_depth=6, min_child_weight=3,
                    subsample=0.8, colsample_bytree=0.8, gamma=0.1,
                    reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0
                )
            
            xgb_model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
            models['xgboost'] = xgb_model
            
            # Evaluate
            pred_val = xgb_model.predict(X_val_scaled)
            pred_test = xgb_model.predict(X_test_scaled)
            r2_val = r2_score(y_val, pred_val)
            r2_test = r2_score(y_test, pred_test)
            mse_test = mean_squared_error(y_test, pred_test)
            direction_acc = (np.sign(pred_test) == np.sign(y_test)).mean()
            
            val_scores['xgboost'] = r2_val
            logging.info(f"✅ XGBoost - Val R²: {r2_val:.4f}, Test R²: {r2_test:.4f}, MSE: {mse_test:.6f}, Dir Acc: {direction_acc:.2%}")
        
        # ========== 3. CatBoost (TERTIARY) ==========
        if CATBOOST_AVAILABLE:
            logging.info("\n🐱 Training CatBoost (Tertiary)...")
            
            if self.enable_tuning and OPTUNA_AVAILABLE:
                logging.info(f"   🔧 Tuning hyperparameters ({self.n_trials} trials)...")
                
                def objective(trial):
                    params = {
                        'iterations': trial.suggest_int('iterations', 100, 1000),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                        'depth': trial.suggest_int('depth', 3, 10),
                        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
                        'random_seed': 42,
                        'verbose': False
                    }
                    model = cb.CatBoostRegressor(**params)
                    model.fit(X_train_scaled, y_train, eval_set=(X_val_scaled, y_val),
                             early_stopping_rounds=50, verbose=False)
                    pred_val = model.predict(X_val_scaled)
                    return r2_score(y_val, pred_val)
                
                study = optuna.create_study(direction='maximize', study_name='cb_price')
                study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
                best_params = study.best_params
                logging.info(f"   ✅ Best R²: {study.best_value:.4f}")
                cb_model = cb.CatBoostRegressor(**best_params, verbose=False)
            else:
                cb_model = cb.CatBoostRegressor(
                    iterations=500, learning_rate=0.05, depth=6, l2_leaf_reg=3,
                    random_seed=42, verbose=False
                )
            
            cb_model.fit(X_train_scaled, y_train, eval_set=(X_val_scaled, y_val),
                        early_stopping_rounds=50, verbose=False)
            models['catboost'] = cb_model
            
            # Evaluate
            pred_val = cb_model.predict(X_val_scaled)
            pred_test = cb_model.predict(X_test_scaled)
            r2_val = r2_score(y_val, pred_val)
            r2_test = r2_score(y_test, pred_test)
            mse_test = mean_squared_error(y_test, pred_test)
            direction_acc = (np.sign(pred_test) == np.sign(y_test)).mean()
            
            val_scores['catboost'] = r2_val
            logging.info(f"✅ CatBoost - Val R²: {r2_val:.4f}, Test R²: {r2_test:.4f}, MSE: {mse_test:.6f}, Dir Acc: {direction_acc:.2%}")
        
        # ========== CALCULATE DYNAMIC WEIGHTS ==========
        if val_scores:
            # Use softmax to convert R² scores to weights
            scores_array = np.array(list(val_scores.values()))
            # Clip negative R² to 0
            scores_array = np.maximum(scores_array, 0)
            # Apply softmax with temperature
            exp_scores = np.exp(scores_array * 5)  # Temperature = 5 to amplify differences
            dynamic_weights = exp_scores / exp_scores.sum()
            
            weight_dict = {name: float(weight) for name, weight in zip(val_scores.keys(), dynamic_weights)}
            self.model_performance[f"{symbol}_{timeframe}_price"] = {
                'val_scores': val_scores,
                'dynamic_weights': weight_dict
            }
            
            logging.info(f"\n⚖️  Dynamic Weights (based on validation R²):")
            for name, weight in weight_dict.items():
                logging.info(f"   {name}: {weight:.2%} (R²: {val_scores[name]:.4f})")
        
        # Save all models
        for model_name, model in models.items():
            safe_symbol = symbol.replace('/', '_')
            model_path = f"ml_models/{safe_symbol}_{timeframe}_price_{model_name}.joblib"
            joblib.dump(model, model_path)
            self.models[f"{symbol}_{timeframe}_price_{model_name}"] = model
            logging.info(f"💾 Saved: {model_path}")
        
        # Save scaler, selector, and feature columns
        safe_symbol = symbol.replace('/', '_')
        scaler_path = f"ml_models/{safe_symbol}_{timeframe}_price_scaler.joblib"
        joblib.dump(scaler, scaler_path)
        
        selector_path = f"ml_models/{safe_symbol}_{timeframe}_price_selector.joblib"
        joblib.dump(selector, selector_path)
        
        features_path = f"ml_models/{safe_symbol}_{timeframe}_price_features.joblib"
        joblib.dump(selected_features, features_path)
        
        # Save performance metrics
        perf_path = f"ml_models/{safe_symbol}_{timeframe}_price_performance.joblib"
        joblib.dump(self.model_performance.get(f"{symbol}_{timeframe}_price", {}), perf_path)
        
        logging.info(f"\n✅ Price models trained successfully!")
        return len(models) > 0

    
    def _train_direction_models(self, symbol: str, timeframe: str, df: pd.DataFrame) -> bool:
        """Train direction prediction models (classification) with feature selection and hyperparameter tuning"""
        logging.info(f"\n🎯 Training Direction Prediction Models")
        logging.info("-" * 60)
        
        # Prepare data
        X, y, feature_cols = self.prepare_data(df, prediction_type='direction')
        
        if len(X) == 0:
            logging.error("❌ No valid data for direction prediction")
            return False
        
        # Time series split
        train_size = int(len(X) * 0.70)
        val_size = int(len(X) * 0.85)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:val_size]
        y_val = y[train_size:val_size]
        X_test = X[val_size:]
        y_test = y[val_size:]
        
        logging.info(f"📊 Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # ========== FEATURE SELECTION ==========
        logging.info(f"\n🎯 Selecting top {self.n_features} features using mutual_info_classif...")
        selector = SelectKBest(score_func=mutual_info_classif, k=min(self.n_features, len(feature_cols)))
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_val_selected = selector.transform(X_val)
        X_test_selected = selector.transform(X_test)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        selected_features = [name for name, sel in zip(feature_cols, selected_mask) if sel]
        
        logging.info(f"✅ Selected {len(selected_features)} features")
        logging.info(f"   Top 5: {selected_features[:5]}")
        
        # Store feature selector
        self.feature_selectors[f"{symbol}_{timeframe}_direction"] = selector
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_val_scaled = scaler.transform(X_val_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        
        # Store scaler
        self.scalers[f"{symbol}_{timeframe}_direction"] = scaler
        
        models = {}
        val_scores = {}  # Track validation accuracy for dynamic weights
        
        # ========== 1. LightGBM Classifier ==========
        if LIGHTGBM_AVAILABLE:
            logging.info("\n🌟 Training LightGBM Classifier...")
            
            if self.enable_tuning and OPTUNA_AVAILABLE:
                logging.info(f"   🔧 Tuning hyperparameters ({self.n_trials} trials)...")
                
                def objective(trial):
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                        'max_depth': trial.suggest_int('max_depth', 3, 12),
                        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'random_state': 42,
                        'verbose': -1
                    }
                    model = lgb.LGBMClassifier(**params)
                    model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)],
                             callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
                    pred_val = model.predict(X_val_scaled)
                    return accuracy_score(y_val, pred_val)
                
                study = optuna.create_study(direction='maximize', study_name='lgb_direction')
                study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
                best_params = study.best_params
                logging.info(f"   ✅ Best Accuracy: {study.best_value:.2%}")
                lgb_model = lgb.LGBMClassifier(**best_params, verbose=-1)
            else:
                lgb_model = lgb.LGBMClassifier(
                    n_estimators=500, learning_rate=0.05, max_depth=7, num_leaves=31,
                    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                    random_state=42, verbose=-1
                )
            
            lgb_model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)],
                         callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            models['lightgbm'] = lgb_model
            
            # Evaluate
            pred_val = lgb_model.predict(X_val_scaled)
            pred_test = lgb_model.predict(X_test_scaled)
            acc_val = accuracy_score(y_val, pred_val)
            acc_test = accuracy_score(y_test, pred_test)
            prec = precision_score(y_test, pred_test, zero_division=0)
            rec = recall_score(y_test, pred_test, zero_division=0)
            f1 = f1_score(y_test, pred_test, zero_division=0)
            
            val_scores['lightgbm'] = acc_val
            logging.info(f"✅ LightGBM - Val Acc: {acc_val:.2%}, Test Acc: {acc_test:.2%}, Prec: {prec:.2%}, Rec: {rec:.2%}, F1: {f1:.2%}")
        
        # ========== 2. XGBoost Classifier ==========
        if XGBOOST_AVAILABLE:
            logging.info("\n🔥 Training XGBoost Classifier...")
            
            if self.enable_tuning and OPTUNA_AVAILABLE:
                logging.info(f"   🔧 Tuning hyperparameters ({self.n_trials} trials)...")
                
                def objective(trial):
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                        'max_depth': trial.suggest_int('max_depth', 3, 12),
                        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'random_state': 42,
                        'verbosity': 0
                    }
                    model = xgb.XGBClassifier(**params)
                    model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
                    pred_val = model.predict(X_val_scaled)
                    return accuracy_score(y_val, pred_val)
                
                study = optuna.create_study(direction='maximize', study_name='xgb_direction')
                study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
                best_params = study.best_params
                logging.info(f"   ✅ Best Accuracy: {study.best_value:.2%}")
                xgb_model = xgb.XGBClassifier(**best_params, verbosity=0)
            else:
                xgb_model = xgb.XGBClassifier(
                    n_estimators=500, learning_rate=0.05, max_depth=6, min_child_weight=3,
                    subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
                )
            
            xgb_model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
            models['xgboost'] = xgb_model
            
            # Evaluate
            pred_val = xgb_model.predict(X_val_scaled)
            pred_test = xgb_model.predict(X_test_scaled)
            acc_val = accuracy_score(y_val, pred_val)
            acc_test = accuracy_score(y_test, pred_test)
            prec = precision_score(y_test, pred_test, zero_division=0)
            rec = recall_score(y_test, pred_test, zero_division=0)
            f1 = f1_score(y_test, pred_test, zero_division=0)
            
            val_scores['xgboost'] = acc_val
            logging.info(f"✅ XGBoost - Val Acc: {acc_val:.2%}, Test Acc: {acc_test:.2%}, Prec: {prec:.2%}, Rec: {rec:.2%}, F1: {f1:.2%}")
        
        # ========== 3. CatBoost Classifier ==========
        if CATBOOST_AVAILABLE:
            logging.info("\n🐱 Training CatBoost Classifier...")
            
            if self.enable_tuning and OPTUNA_AVAILABLE:
                logging.info(f"   🔧 Tuning hyperparameters ({self.n_trials} trials)...")
                
                def objective(trial):
                    params = {
                        'iterations': trial.suggest_int('iterations', 100, 1000),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                        'depth': trial.suggest_int('depth', 3, 10),
                        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
                        'random_seed': 42,
                        'verbose': False
                    }
                    model = cb.CatBoostClassifier(**params)
                    model.fit(X_train_scaled, y_train, eval_set=(X_val_scaled, y_val),
                             early_stopping_rounds=50, verbose=False)
                    pred_val = model.predict(X_val_scaled)
                    return accuracy_score(y_val, pred_val)
                
                study = optuna.create_study(direction='maximize', study_name='cb_direction')
                study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
                best_params = study.best_params
                logging.info(f"   ✅ Best Accuracy: {study.best_value:.2%}")
                cb_model = cb.CatBoostClassifier(**best_params, verbose=False)
            else:
                cb_model = cb.CatBoostClassifier(
                    iterations=500, learning_rate=0.05, depth=6, l2_leaf_reg=3,
                    random_seed=42, verbose=False
                )
            
            cb_model.fit(X_train_scaled, y_train, eval_set=(X_val_scaled, y_val),
                        early_stopping_rounds=50, verbose=False)
            models['catboost'] = cb_model
            
            # Evaluate
            pred_val = cb_model.predict(X_val_scaled)
            pred_test = cb_model.predict(X_test_scaled)
            acc_val = accuracy_score(y_val, pred_val)
            acc_test = accuracy_score(y_test, pred_test)
            prec = precision_score(y_test, pred_test, zero_division=0)
            rec = recall_score(y_test, pred_test, zero_division=0)
            f1 = f1_score(y_test, pred_test, zero_division=0)
            
            val_scores['catboost'] = acc_val
            logging.info(f"✅ CatBoost - Val Acc: {acc_val:.2%}, Test Acc: {acc_test:.2%}, Prec: {prec:.2%}, Rec: {rec:.2%}, F1: {f1:.2%}")
        
        # ========== CALCULATE DYNAMIC WEIGHTS ==========
        if val_scores:
            # Use softmax to convert accuracy scores to weights
            scores_array = np.array(list(val_scores.values()))
            # Apply softmax with temperature
            exp_scores = np.exp(scores_array * 10)  # Temperature = 10 for classification
            dynamic_weights = exp_scores / exp_scores.sum()
            
            weight_dict = {name: float(weight) for name, weight in zip(val_scores.keys(), dynamic_weights)}
            self.model_performance[f"{symbol}_{timeframe}_direction"] = {
                'val_scores': val_scores,
                'dynamic_weights': weight_dict
            }
            
            logging.info(f"\n⚖️  Dynamic Weights (based on validation accuracy):")
            for name, weight in weight_dict.items():
                logging.info(f"   {name}: {weight:.2%} (Acc: {val_scores[name]:.2%})")
        
        # Save all models
        for model_name, model in models.items():
            safe_symbol = symbol.replace('/', '_')
            model_path = f"ml_models/{safe_symbol}_{timeframe}_direction_{model_name}.joblib"
            joblib.dump(model, model_path)
            self.models[f"{symbol}_{timeframe}_direction_{model_name}"] = model
            logging.info(f"💾 Saved: {model_path}")
        
        # Save scaler, selector, and performance metrics
        safe_symbol = symbol.replace('/', '_')
        scaler_path = f"ml_models/{safe_symbol}_{timeframe}_direction_scaler.joblib"
        joblib.dump(scaler, scaler_path)
        
        selector_path = f"ml_models/{safe_symbol}_{timeframe}_direction_selector.joblib"
        joblib.dump(selector, selector_path)
        
        features_path = f"ml_models/{safe_symbol}_{timeframe}_direction_features.joblib"
        joblib.dump(selected_features, features_path)
        
        perf_path = f"ml_models/{safe_symbol}_{timeframe}_direction_performance.joblib"
        joblib.dump(self.model_performance.get(f"{symbol}_{timeframe}_direction", {}), perf_path)
        
        logging.info(f"\n✅ Direction models trained successfully!")
        return len(models) > 0

    
    def _train_gru_model(self, symbol: str, timeframe: str, df: pd.DataFrame) -> bool:
        """Train GRU model for time series (4h timeframe only)"""
        if not DL_AVAILABLE:
            logging.warning("⚠️ TensorFlow not available, skipping GRU training")
            return False
        
        logging.info(f"\n🧠 Training GRU Model (Deep Learning)")
        logging.info("-" * 60)
        
        sequence_length = 60
        
        if len(df) < sequence_length * 3:
            logging.warning(f"⚠️ Insufficient data for GRU (need {sequence_length*3}, have {len(df)})")
            return False
        
        # Prepare price sequences
        prices = df['close'].values.reshape(-1, 1)
        
        # Scale data
        scaler = StandardScaler()
        scaled_prices = scaler.fit_transform(prices)
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(scaled_prices)):
            X.append(scaled_prices[i-sequence_length:i, 0])
            y.append(scaled_prices[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        logging.info(f"📊 GRU sequences - Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Build GRU model
        model = Sequential([
            GRU(128, return_sequences=True, input_shape=(sequence_length, 1)),
            Dropout(0.2),
            GRU(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        # Train
        early_stopping = EarlyStopping(patience=15, restore_best_weights=True)
        safe_symbol = symbol.replace('/', '_')
        model_path = f"ml_models/{safe_symbol}_{timeframe}_gru.h5"
        checkpoint = ModelCheckpoint(model_path, save_best_only=True)
        
        logging.info("🔄 Training GRU (this may take 10-20 minutes)...")
        
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping, checkpoint],
            verbose=0
        )
        
        # Evaluate
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
        logging.info(f"✅ GRU - Test Loss: {test_loss:.6f}, MAE: {test_mae:.6f}")
        
        # Save scaler
        scaler_path = f"ml_models/{safe_symbol}_{timeframe}_gru_scaler.joblib"
        joblib.dump(scaler, scaler_path)
        
        # Store in memory
        self.models[f"{symbol}_{timeframe}_gru"] = model
        self.scalers[f"{symbol}_{timeframe}_gru"] = scaler
        
        logging.info(f"💾 Saved GRU model: {model_path}")
        return True
    
    def make_ensemble_prediction(self, symbol: str, timeframe: str) -> Dict:
        """
        Make ensemble predictions combining all models
        
        Weights by timeframe (OPTIMIZED - Priority 1):
        - 1h: 50% LightGBM, 30% XGBoost, 20% CatBoost
        - 4h: 45% LightGBM, 30% XGBoost, 15% CatBoost, 10% GRU
        """
        logging.info(f"\n{'='*60}")
        logging.info(f"🔮 Ensemble Prediction: {symbol} {timeframe}")
        logging.info(f"{'='*60}\n")
        
        # PRIORITY 1 FIX: Adaptive lookback based on timeframe
        lookback_map = {
            '5m': 1, '15m': 1, '1h': 2, '4h': 3, '1d': 6
        }
        months_back = lookback_map.get(timeframe, 1)
        logging.info(f"📅 Using {months_back} months lookback for {timeframe}")
        
        # Load recent data with appropriate lookback
        df = self.load_data(symbol, timeframe, months_back=months_back)
        if df.empty or len(df) < 50:
            logging.error("❌ Insufficient data for prediction")
            return {}
        
        # Create features
        df_features = self.create_features(df)
        if df_features.empty:
            logging.error("❌ Failed to create features")
            return {}
        
        predictions = {
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': datetime.now().isoformat(),
            'current_price': float(df['close'].iloc[-1])
        }
        
        # Load models
        self._load_models(symbol, timeframe)
        
        # Get latest features
        exclude_cols = ['open', 'high', 'low', 'close', 'volume']
        available_features = [col for col in df_features.columns if col not in exclude_cols]
        
        if not available_features:
            logging.error("❌ No features available")
            return predictions
        
        X_latest = df_features[available_features].iloc[-1:].values
        
        # PRIORITY 1 FIX: Optimized ensemble weights
        if timeframe == '4h':
            weight_config = {'lightgbm': 0.45, 'xgboost': 0.30, 'catboost': 0.15, 'gru': 0.10}
        else:
            weight_config = {'lightgbm': 0.50, 'xgboost': 0.30, 'catboost': 0.20}
        
        # Price predictions
        price_preds = []
        price_weights = []
        
        # LightGBM
        if f"{symbol}_{timeframe}_price_lightgbm" in self.models:
            scaler = self.scalers.get(f"{symbol}_{timeframe}_price")
            if scaler is not None:
                X_scaled = scaler.transform(X_latest)
                pred = self.models[f"{symbol}_{timeframe}_price_lightgbm"].predict(X_scaled)[0]
                price_preds.append(pred)
                price_weights.append(weight_config['lightgbm'])
                logging.info(f"📊 LightGBM price: {pred:+.4%} (weight: {weight_config['lightgbm']:.0%})")
        
        # XGBoost
        if f"{symbol}_{timeframe}_price_xgboost" in self.models:
            scaler = self.scalers.get(f"{symbol}_{timeframe}_price")
            if scaler is not None:
                X_scaled = scaler.transform(X_latest)
                pred = self.models[f"{symbol}_{timeframe}_price_xgboost"].predict(X_scaled)[0]
                price_preds.append(pred)
                price_weights.append(weight_config['xgboost'])
                logging.info(f"📊 XGBoost price: {pred:+.4%} (weight: {weight_config['xgboost']:.0%})")
        
        # CatBoost
        if f"{symbol}_{timeframe}_price_catboost" in self.models:
            scaler = self.scalers.get(f"{symbol}_{timeframe}_price")
            if scaler is not None:
                X_scaled = scaler.transform(X_latest)
                pred = self.models[f"{symbol}_{timeframe}_price_catboost"].predict(X_scaled)[0]
                price_preds.append(pred)
                price_weights.append(weight_config['catboost'])
                logging.info(f"📊 CatBoost price: {pred:+.4%} (weight: {weight_config['catboost']:.0%})")
        
        # GRU (4h only)
        if timeframe == '4h' and f"{symbol}_{timeframe}_gru" in self.models:
            sequence_length = 60
            if len(df) >= sequence_length:
                scaler = self.scalers.get(f"{symbol}_{timeframe}_gru")
                if scaler is not None:
                    prices = df['close'].values[-sequence_length:].reshape(-1, 1)
                    scaled_prices = scaler.transform(prices)
                    X_gru = scaled_prices.reshape(1, sequence_length, 1)
                    
                    try:
                        gru_pred_scaled = self.models[f"{symbol}_{timeframe}_gru"].predict(X_gru, verbose=0)[0][0]
                        gru_pred_price = scaler.inverse_transform([[gru_pred_scaled]])[0][0]
                        gru_pred_change = gru_pred_price / df['close'].iloc[-1] - 1
                        
                        price_preds.append(gru_pred_change)
                        price_weights.append(weight_config.get('gru', 0.10))
                        logging.info(f"🧠 GRU price: {gru_pred_change:+.4%} (weight: {weight_config.get('gru', 0.10):.0%})")
                    except Exception as e:
                        logging.warning(f"⚠️ GRU prediction failed: {e}")
        
        # Calculate ensemble price prediction
        if price_preds:
            # Normalize weights
            total_weight = sum(price_weights)
            normalized_weights = [w/total_weight for w in price_weights]
            
            ensemble_price_change = sum(p * w for p, w in zip(price_preds, normalized_weights))
            
            # PRIORITY 1 FIX: Clip unrealistic predictions
            prediction_limits = {
                '5m': 0.02, '15m': 0.03, '1h': 0.05, '4h': 0.10, '1d': 0.15
            }
            max_change = prediction_limits.get(timeframe, 0.05)
            original_pred = ensemble_price_change
            ensemble_price_change = np.clip(ensemble_price_change, -max_change, max_change)
            
            if abs(original_pred) != abs(ensemble_price_change):
                logging.warning(f"⚠️ Clipped prediction from {original_pred:+.4%} to {ensemble_price_change:+.4%}")
            
            predictions['price_change_pct'] = float(ensemble_price_change * 100)
            predictions['predicted_price'] = float(df['close'].iloc[-1] * (1 + ensemble_price_change))
            
            # Use improved confidence calculation
            direction_votes_for_conf = direction_votes if 'direction_votes' in locals() else {'UP': 0, 'DOWN': 0}
            predictions['confidence'] = self.calculate_confidence(price_preds, direction_votes_for_conf)
            
            logging.info(f"\n💡 Ensemble Price Change: {ensemble_price_change:+.4%}")
            logging.info(f"💰 Predicted Price: ${predictions['predicted_price']:.2f}")
            logging.info(f"📊 Confidence: {predictions['confidence']:.2%}")
        
        # Direction predictions
        direction_votes = {'UP': 0, 'DOWN': 0}
        
        # All models use scaled data
        for model_type in ['lightgbm', 'xgboost', 'catboost']:
            model_key = f"{symbol}_{timeframe}_direction_{model_type}"
            if model_key in self.models:
                scaler = self.scalers.get(f"{symbol}_{timeframe}_direction")
                if scaler is not None:
                    X_scaled = scaler.transform(X_latest)
                    direction_pred = self.models[model_key].predict(X_scaled)[0]
                    direction_votes['UP' if direction_pred == 1 else 'DOWN'] += 1
        
        if sum(direction_votes.values()) > 0:
            predictions['direction'] = max(direction_votes, key=direction_votes.get)
            predictions['direction_confidence'] = direction_votes[predictions['direction']] / sum(direction_votes.values())
            
            logging.info(f"🎯 Direction: {predictions['direction']} (confidence: {predictions['direction_confidence']:.2%})")
        
        logging.info(f"\n✅ Ensemble prediction complete!\n")
        return predictions
    
    def calculate_confidence(self, price_preds: list, direction_votes: dict) -> float:
        """
        Calculate prediction confidence based on model agreement
        
        PRIORITY 1 FIX: Improved confidence scoring
        
        Args:
            price_preds: List of price predictions from models
            direction_votes: Dictionary of direction votes
        
        Returns:
            Confidence score between 0 and 0.95
        """
        # Direction agreement component
        total_votes = sum(direction_votes.values())
        if total_votes == 0:
            direction_confidence = 0.5
        else:
            max_votes = max(direction_votes.values())
            direction_confidence = max_votes / total_votes
        
        # Price prediction variance component
        if len(price_preds) > 0:
            std = np.std(price_preds)
            variance_confidence = 1.0 / (1.0 + std * 100)
        else:
            variance_confidence = 0.5
        
        # Combined confidence (70% direction, 30% variance)
        confidence = (direction_confidence * 0.7 + variance_confidence * 0.3)
        
        return float(np.clip(confidence, 0.0, 0.95))
    
    def _load_models(self, symbol: str, timeframe: str):
        """Load all available models, scalers, selectors, and performance metrics for a symbol/timeframe"""
        safe_symbol = symbol.replace('/', '_')
        model_types = ['price', 'direction']
        model_names = ['lightgbm', 'xgboost', 'catboost']
        
        for model_type in model_types:
            # Load feature selector and performance metrics
            selector_path = f"ml_models/{safe_symbol}_{timeframe}_{model_type}_selector.joblib"
            perf_path = f"ml_models/{safe_symbol}_{timeframe}_{model_type}_performance.joblib"
            
            if os.path.exists(selector_path):
                try:
                    selector = joblib.load(selector_path)
                    self.feature_selectors[f"{symbol}_{timeframe}_{model_type}"] = selector
                except Exception as e:
                    logging.warning(f"⚠️ Failed to load selector {selector_path}: {e}")
            
            if os.path.exists(perf_path):
                try:
                    perf = joblib.load(perf_path)
                    self.model_performance[f"{symbol}_{timeframe}_{model_type}"] = perf
                except Exception as e:
                    logging.warning(f"⚠️ Failed to load performance {perf_path}: {e}")
            
            # Load models and scalers
            for model_name in model_names:
                model_path = f"ml_models/{safe_symbol}_{timeframe}_{model_type}_{model_name}.joblib"
                scaler_path = f"ml_models/{safe_symbol}_{timeframe}_{model_type}_scaler.joblib"
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    try:
                        model = joblib.load(model_path)
                        scaler = joblib.load(scaler_path)
                        
                        self.models[f"{symbol}_{timeframe}_{model_type}_{model_name}"] = model
                        self.scalers[f"{symbol}_{timeframe}_{model_type}"] = scaler
                    except Exception as e:
                        logging.warning(f"⚠️ Failed to load {model_path}: {e}")
        
        # Load GRU
        gru_path = f"ml_models/{safe_symbol}_{timeframe}_gru.h5"
        gru_scaler_path = f"ml_models/{safe_symbol}_{timeframe}_gru_scaler.joblib"
        
        if os.path.exists(gru_path) and os.path.exists(gru_scaler_path) and DL_AVAILABLE:
            try:
                model = load_model(gru_path, compile=False)
                model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
                scaler = joblib.load(gru_scaler_path)
                
                self.models[f"{symbol}_{timeframe}_gru"] = model
                self.scalers[f"{symbol}_{timeframe}_gru"] = scaler
            except Exception as e:
                logging.warning(f"⚠️ Failed to load GRU: {e}")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimized Crypto ML Training System')
    parser.add_argument('--symbol', type=str, help='Specific symbol to train (e.g., BTC/USDT)')
    parser.add_argument('--timeframe', type=str, help='Specific timeframe to train (e.g., 1h, 4h)')
    parser.add_argument('--n-features', type=int, default=50, help='Number of features to select (default: 50)')
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning with Optuna')
    parser.add_argument('--n-trials', type=int, default=50, help='Number of Optuna trials (default: 50)')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🧠 OPTIMIZED CRYPTO ML TRAINING SYSTEM")
    print("="*70)
    print("\nStrategy: LightGBM + XGBoost + CatBoost + GRU Ensemble")
    print("Database: ml_crypto_data.db")
    print(f"Feature Selection: Top {args.n_features} features")
    print(f"Hyperparameter Tuning: {'Enabled' if args.tune else 'Disabled'}")
    if args.tune:
        print(f"Optuna Trials: {args.n_trials}")
    print("\n" + "="*70 + "\n")
    
    # Initialize system with enhanced features
    ml_system = OptimizedCryptoMLSystem(
        n_features=args.n_features,
        enable_tuning=args.tune,
        n_trials=args.n_trials
    )
    
    # Determine symbols and timeframes to train
    if args.symbol and args.timeframe:
        symbols = [args.symbol]
        timeframes = [args.timeframe]
    elif args.symbol:
        symbols = [args.symbol]
        timeframes = ['5m', '15m', '1h', '4h', '1d']
    elif args.timeframe:
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'DOT/USDT']
        timeframes = [args.timeframe]
    else:
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'DOT/USDT']
        timeframes = ['5m', '15m', '1h', '4h', '1d']
    
    print("\n📋 Training Plan:")
    print(f"   Symbols: {', '.join(symbols)}")
    print(f"   Timeframes: {', '.join(timeframes)}")
    print(f"   Models per symbol/timeframe: 6-7 (LightGBM, XGBoost, CatBoost x2, GRU for 4h)")
    print(f"   Total models: ~{len(symbols) * len(timeframes) * 6} models\n")
    
    # Track progress
    total_trained = 0
    total_failed = 0
    start_time = datetime.now()
    
    # Training loop
    for symbol in symbols:
        for timeframe in timeframes:
            print(f"\n{'='*70}")
            print(f"🎯 Processing: {symbol} {timeframe}")
            print(f"{'='*70}")
            
            success = ml_system.train_ensemble(symbol, timeframe)
            
            if success:
                total_trained += 1
                print(f"\n✅ Training completed for {symbol} {timeframe}")
                
                # Make a test prediction
                print(f"\n🔮 Making test prediction...")
                prediction = ml_system.make_ensemble_prediction(symbol, timeframe)
                
                if prediction:
                    print(f"\n📊 Test Prediction Results:")
                    print(f"   Current Price: ${prediction.get('current_price', 0):.2f}")
                    if 'predicted_price' in prediction:
                        print(f"   Predicted Price: ${prediction['predicted_price']:.2f}")
                        print(f"   Expected Change: {prediction['price_change_pct']:+.2f}%")
                    if 'direction' in prediction:
                        print(f"   Direction: {prediction['direction']} ({prediction['direction_confidence']:.0%} confidence)")
            else:
                total_failed += 1
                print(f"\n❌ Training failed for {symbol} {timeframe}")
            
            print("\n" + "-"*70)
    
    # Final summary
    duration = datetime.now() - start_time
    
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"   Total Successful: {total_trained}")
    print(f"   Total Failed: {total_failed}")
    print(f"   Success Rate: {total_trained/(total_trained+total_failed)*100:.1f}%")
    print(f"   Duration: {duration}")
    print(f"\n📁 Models saved in: ml_models/")
    print(f"📝 Logs saved in: {ml_system.log_file}")
    print("💡 Next step: Integrate predictions into unified_crypto_analyzer.py")
    print("\n")


if __name__ == "__main__":
    main()