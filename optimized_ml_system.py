"""
Optimized ML/DL Integration System for Crypto Trading
Implementation: LightGBM + XGBoost + CatBoost + GRU Ensemble
Database: ml_crypto_data.db (unified database)
Strategy: Best practices for crypto price prediction
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# ML Libraries
try:
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.error("❌ Scikit-learn not available. Install: pip install scikit-learn")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("⚠️ LightGBM not available. Install: pip install lightgbm")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("⚠️ XGBoost not available. Install: pip install xgboost")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logging.warning("⚠️ CatBoost not available. Install: pip install catboost")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import GRU, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False
    logging.warning("⚠️ TensorFlow not available. Install: pip install tensorflow")


class OptimizedCryptoMLSystem:
    """
    Optimized ML system using ensemble of:
    - LightGBM (Primary - Best for tabular data)
    - XGBoost (Secondary - Industry standard)
    - CatBoost (Tertiary - Robust against overfitting)
    - GRU (Deep Learning - For 4h timeframe)
    """
    
    def __init__(self, db_path='data/ml_crypto_data.db'):
        """Initialize the optimized ML system"""
        self.db_path = db_path
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        
        # Create directories
        os.makedirs('ml_models', exist_ok=True)
        os.makedirs('ml_predictions', exist_ok=True)
        os.makedirs('ml_reports', exist_ok=True)
        
        logging.info("🧠 Optimized Crypto ML System initialized")
        logging.info(f"📁 Database: {self.db_path}")
        logging.info(f"✅ LightGBM: {LIGHTGBM_AVAILABLE}")
        logging.info(f"✅ XGBoost: {XGBOOST_AVAILABLE}")
        logging.info(f"✅ CatBoost: {CATBOOST_AVAILABLE}")
        logging.info(f"✅ TensorFlow: {DL_AVAILABLE}")
    
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
        
        Features (70+ total):
        1. Price features (3)
        2. Moving averages (8)
        3. Technical indicators (15+)
        4. Volume indicators (4)
        5. Volatility features (9)
        6. Momentum features (6)
        7. Lag features (15)
        8. Rolling statistics (9)
        9. Time features (4)
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # 1. Basic price features
        df['price_change'] = df['close'].pct_change()
        df['high_low_pct'] = (df['high'] - df['low']) / df['low']
        df['close_open_pct'] = (df['close'] - df['open']) / df['open']
        
        # 2. Moving averages (optimized windows)
        for window in [5, 10, 20, 50]:
            df[f'ma_{window}'] = df['close'].rolling(window).mean()
            df[f'ma_{window}_ratio'] = df['close'] / df[f'ma_{window}']
        
        # 3. Technical indicators
        self._add_rsi(df, windows=[14, 21, 28])
        self._add_macd(df)
        self._add_bollinger_bands(df)
        self._add_atr(df)
        
        # 4. Volume indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['price_volume'] = df['close'] * df['volume']
        df['vwap'] = (df['price_volume'].rolling(20).sum() / 
                      df['volume'].rolling(20).sum())
        
        # 5. Volatility features
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['price_change'].rolling(window).std()
            df[f'volatility_ratio_{window}'] = (
                df[f'volatility_{window}'] / 
                df[f'volatility_{window}'].rolling(50).mean()
            )
        
        # 6. Momentum features
        for window in [5, 10, 20]:
            df[f'momentum_{window}'] = df['close'] / df['close'].shift(window) - 1
            df[f'roc_{window}'] = df['close'].pct_change(window)
        
        # 7. Lag features (important for time series)
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            df[f'price_change_lag_{lag}'] = df['price_change'].shift(lag)
        
        # 8. Rolling statistics
        for window in [5, 10, 20]:
            df[f'close_mean_{window}'] = df['close'].rolling(window).mean()
            df[f'close_std_{window}'] = df['close'].rolling(window).std()
            df[f'volume_mean_{window}'] = df['volume'].rolling(window).mean()
        
        # 9. Time features (categorical for CatBoost)
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        # Drop NaN values
        initial_len = len(df)
        df.dropna(inplace=True)
        dropped = initial_len - len(df)
        
        if dropped > 0:
            logging.info(f"ℹ️ Dropped {dropped} rows with NaN values")
        
        logging.info(f"✅ Created {len(df.columns)} features from {len(df)} samples")
        return df
    
    def _add_rsi(self, df: pd.DataFrame, windows: List[int] = [14, 21, 28]):
        """Add RSI indicators with multiple windows"""
        for window in windows:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
    
    def _add_macd(self, df: pd.DataFrame):
        """Add MACD indicator"""
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    def _add_bollinger_bands(self, df: pd.DataFrame, window: int = 20):
        """Add Bollinger Bands"""
        rolling_mean = df['close'].rolling(window).mean()
        rolling_std = df['close'].rolling(window).std()
        df['bb_upper'] = rolling_mean + (rolling_std * 2)
        df['bb_lower'] = rolling_mean - (rolling_std * 2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    def _add_atr(self, df: pd.DataFrame, window: int = 14):
        """Add Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(window).mean()
    
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
        """Train price prediction models (regression)"""
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
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Store scaler and feature columns
        self.scalers[f"{symbol}_{timeframe}_price"] = scaler
        self.feature_columns = feature_cols
        
        models = {}
        
        # 1. LightGBM (PRIMARY) ⭐⭐⭐⭐⭐
        if LIGHTGBM_AVAILABLE:
            logging.info("\n🌟 Training LightGBM (Primary)...")
            lgb_model = lgb.LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=7,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbose=-1,
                force_col_wise=True  # Better for wide datasets
            )
            lgb_model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]  # Increased patience
            )
            models['lightgbm'] = lgb_model
            
            # Evaluate
            pred_train = lgb_model.predict(X_train_scaled)
            pred = lgb_model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, pred)
            r2 = r2_score(y_test, pred)
            r2_train = r2_score(y_train, pred_train)
            direction_acc = (np.sign(pred) == np.sign(y_test)).mean()
            
            logging.info(f"✅ LightGBM - Train R²: {r2_train:.4f}, Test R²: {r2:.4f}, MSE: {mse:.6f}, Dir Acc: {direction_acc:.2%}")
        
        # 2. XGBoost (SECONDARY) ⭐⭐⭐⭐⭐
        if XGBOOST_AVAILABLE:
            logging.info("\n🔥 Training XGBoost (Secondary)...")
            xgb_model = xgb.XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbosity=0
            )
            xgb_model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False
            )
            models['xgboost'] = xgb_model
            
            # Evaluate
            pred = xgb_model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, pred)
            r2 = r2_score(y_test, pred)
            direction_acc = (np.sign(pred) == np.sign(y_test)).mean()
            
            logging.info(f"✅ XGBoost - MSE: {mse:.6f}, R²: {r2:.4f}, Dir Acc: {direction_acc:.2%}")
        
        # 3. CatBoost (TERTIARY) ⭐⭐⭐⭐
        if CATBOOST_AVAILABLE:
            logging.info("\n🐱 Training CatBoost (Tertiary)...")
            
            # Simplified: Use scaled data like other models
            # CatBoost is robust enough to work without explicit categorical features
            cb_model = cb.CatBoostRegressor(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=3,
                random_seed=42,
                verbose=False
            )
            
            cb_model.fit(
                X_train_scaled, y_train,
                eval_set=(X_val_scaled, y_val),
                early_stopping_rounds=50,
                verbose=False
            )
            models['catboost'] = cb_model
            
            # Evaluate
            pred = cb_model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, pred)
            r2 = r2_score(y_test, pred)
            direction_acc = (np.sign(pred) == np.sign(y_test)).mean()
            
            logging.info(f"✅ CatBoost - MSE: {mse:.6f}, R²: {r2:.4f}, Dir Acc: {direction_acc:.2%}")
        
        # Save all models
        for model_name, model in models.items():
            # Replace / with _ for safe filenames
            safe_symbol = symbol.replace('/', '_')
            model_path = f"ml_models/{safe_symbol}_{timeframe}_price_{model_name}.joblib"
            joblib.dump(model, model_path)
            self.models[f"{symbol}_{timeframe}_price_{model_name}"] = model
            logging.info(f"💾 Saved: {model_path}")
        
        # Save scaler
        safe_symbol = symbol.replace('/', '_')
        scaler_path = f"ml_models/{safe_symbol}_{timeframe}_price_scaler.joblib"
        joblib.dump(scaler, scaler_path)
        
        # Save feature columns
        features_path = f"ml_models/{safe_symbol}_{timeframe}_price_features.joblib"
        joblib.dump(feature_cols, features_path)
        
        logging.info(f"\n✅ Price models trained successfully!")
        return len(models) > 0
    
    def _train_direction_models(self, symbol: str, timeframe: str, df: pd.DataFrame) -> bool:
        """Train direction prediction models (classification)"""
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
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Store scaler
        self.scalers[f"{symbol}_{timeframe}_direction"] = scaler
        
        models = {}
        
        # 1. LightGBM Classifier
        if LIGHTGBM_AVAILABLE:
            logging.info("\n🌟 Training LightGBM Classifier...")
            lgb_model = lgb.LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=7,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )
            lgb_model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            models['lightgbm'] = lgb_model
            
            # Evaluate
            pred = lgb_model.predict(X_test_scaled)
            acc = accuracy_score(y_test, pred)
            logging.info(f"✅ LightGBM - Accuracy: {acc:.2%}")
        
        # 2. XGBoost Classifier
        if XGBOOST_AVAILABLE:
            logging.info("\n🔥 Training XGBoost Classifier...")
            xgb_model = xgb.XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0
            )
            xgb_model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False
            )
            models['xgboost'] = xgb_model
            
            # Evaluate
            pred = xgb_model.predict(X_test_scaled)
            acc = accuracy_score(y_test, pred)
            logging.info(f"✅ XGBoost - Accuracy: {acc:.2%}")
        
        # 3. CatBoost Classifier
        if CATBOOST_AVAILABLE:
            logging.info("\n🐱 Training CatBoost Classifier...")
            
            # Simplified: Use scaled data like other models
            cb_model = cb.CatBoostClassifier(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=3,
                random_seed=42,
                verbose=False
            )
            
            cb_model.fit(
                X_train_scaled, y_train,
                eval_set=(X_val_scaled, y_val),
                early_stopping_rounds=50,
                verbose=False
            )
            models['catboost'] = cb_model
            
            # Evaluate
            pred = cb_model.predict(X_test_scaled)
            acc = accuracy_score(y_test, pred)
            logging.info(f"✅ CatBoost - Accuracy: {acc:.2%}")
        
        # Save all models
        for model_name, model in models.items():
            safe_symbol = symbol.replace('/', '_')
            model_path = f"ml_models/{safe_symbol}_{timeframe}_direction_{model_name}.joblib"
            joblib.dump(model, model_path)
            self.models[f"{symbol}_{timeframe}_direction_{model_name}"] = model
            logging.info(f"💾 Saved: {model_path}")
        
        # Save scaler
        safe_symbol = symbol.replace('/', '_')
        scaler_path = f"ml_models/{safe_symbol}_{timeframe}_direction_scaler.joblib"
        joblib.dump(scaler, scaler_path)
        
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
        
        Weights by timeframe:
        - 1h: 50% LightGBM, 30% XGBoost, 20% CatBoost
        - 4h: 35% LightGBM, 25% XGBoost, 15% CatBoost, 25% GRU
        """
        logging.info(f"\n{'='*60}")
        logging.info(f"🔮 Ensemble Prediction: {symbol} {timeframe}")
        logging.info(f"{'='*60}\n")
        
        # Load recent data
        df = self.load_data(symbol, timeframe, months_back=1)
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
                price_weights.append(0.50 if timeframe != '4h' else 0.35)
                logging.info(f"📊 LightGBM price: {pred:+.4%}")
        
        # XGBoost
        if f"{symbol}_{timeframe}_price_xgboost" in self.models:
            scaler = self.scalers.get(f"{symbol}_{timeframe}_price")
            if scaler is not None:
                X_scaled = scaler.transform(X_latest)
                pred = self.models[f"{symbol}_{timeframe}_price_xgboost"].predict(X_scaled)[0]
                price_preds.append(pred)
                price_weights.append(0.30 if timeframe != '4h' else 0.25)
                logging.info(f"📊 XGBoost price: {pred:+.4%}")
        
        # CatBoost (uses scaled data like LightGBM/XGBoost)
        if f"{symbol}_{timeframe}_price_catboost" in self.models:
            scaler = self.scalers.get(f"{symbol}_{timeframe}_price")
            if scaler is not None:
                X_scaled = scaler.transform(X_latest)
                pred = self.models[f"{symbol}_{timeframe}_price_catboost"].predict(X_scaled)[0]
                price_preds.append(pred)
                price_weights.append(0.20 if timeframe != '4h' else 0.15)
                logging.info(f"📊 CatBoost price: {pred:+.4%}")
        
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
                        price_weights.append(0.25)
                        logging.info(f"🧠 GRU price: {gru_pred_change:+.4%}")
                    except Exception as e:
                        logging.warning(f"⚠️ GRU prediction failed: {e}")
        
        # Calculate ensemble price prediction
        if price_preds:
            # Normalize weights
            total_weight = sum(price_weights)
            normalized_weights = [w/total_weight for w in price_weights]
            
            ensemble_price_change = sum(p * w for p, w in zip(price_preds, normalized_weights))
            predictions['price_change_pct'] = float(ensemble_price_change * 100)
            predictions['predicted_price'] = float(df['close'].iloc[-1] * (1 + ensemble_price_change))
            predictions['confidence'] = float(min(0.95, 1.0 - np.std(price_preds)))
            
            logging.info(f"\n💡 Ensemble Price Change: {ensemble_price_change:+.4%}")
            logging.info(f"💰 Predicted Price: ${predictions['predicted_price']:.2f}")
        
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
    
    def _load_models(self, symbol: str, timeframe: str):
        """Load all available models for a symbol/timeframe"""
        safe_symbol = symbol.replace('/', '_')
        model_types = ['price', 'direction']
        model_names = ['lightgbm', 'xgboost', 'catboost']
        
        for model_type in model_types:
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
    print("\n" + "="*70)
    print("🧠 OPTIMIZED CRYPTO ML TRAINING SYSTEM")
    print("="*70)
    print("\nStrategy: LightGBM + XGBoost + CatBoost + GRU Ensemble")
    print("Database: ml_crypto_data.db")
    print("\n" + "="*70 + "\n")
    
    # Initialize system
    ml_system = OptimizedCryptoMLSystem()
    
    # Symbols and timeframes
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'DOT/USDT']
    timeframes = ['1h', '4h']  # Focus on best performing timeframes
    
    print("\n📋 Training Plan:")
    print(f"   Symbols: {', '.join(symbols)}")
    print(f"   Timeframes: {', '.join(timeframes)}")
    print(f"   Models per symbol/timeframe: 6-7 (LightGBM, XGBoost, CatBoost x2, GRU for 4h)")
    print(f"   Total models: ~{len(symbols) * len(timeframes) * 6} models\n")
    
    input("Press ENTER to start training... ")
    
    # Training loop
    for symbol in symbols:
        for timeframe in timeframes:
            print(f"\n{'='*70}")
            print(f"🎯 Processing: {symbol} {timeframe}")
            print(f"{'='*70}")
            
            success = ml_system.train_ensemble(symbol, timeframe)
            
            if success:
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
                print(f"\n❌ Training failed for {symbol} {timeframe}")
            
            print("\n" + "-"*70)
    
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print("="*70)
    print("\n📁 Models saved in: ml_models/")
    print("💡 Next step: Integrate predictions into unified_crypto_analyzer.py")
    print("\n")


if __name__ == "__main__":
    main()