"""
Enhanced ML/DL Integration System for Crypto Trading
Integrates with your existing multi_timeframe_collector and analyzer
"""
import os
import sys

if sys.platform.startswith('win'):
    try:
        # Try to set UTF-8 encoding for stdout
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, OSError):
        # If reconfigure doesn't work, try alternative
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import sqlite3
import joblib
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional

# ML/DL Libraries
try:
    import sklearn
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import mean_squared_error, r2_score, classification_report
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("❌ Scikit-learn not available. Install with: pip install scikit-learn")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Conv1D, MaxPooling1D, Flatten
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False
    logging.warning("❌ TensorFlow not available. Install with: pip install tensorflow")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("❌ XGBoost not available. Install with: pip install xgboost")

class CryptoMLSystem:
    def __init__(self, db_path='data/multi_timeframe_data.db'):
        """Initialize the ML system"""
        self.db_path = db_path
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        
        # Auto-detect the correct table name
        self.table_name = self.detect_market_data_table()
        
        # Create ML models directory
        os.makedirs('ml_models', exist_ok=True)
        os.makedirs('ml_predictions', exist_ok=True)
        
        logging.info("🧠 Crypto ML System initialized")
    
    def detect_market_data_table(self):
        """Automatically detect the correct market data table"""
        try:
            if not os.path.exists(self.db_path):
                logging.warning(f"⚠️ Database not found: {self.db_path}")
                return "price_data"  # Default fallback
                
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [table[0] for table in cursor.fetchall()]
            
            # Look for tables with OHLCV structure
            for table in tables:
                try:
                    cursor.execute(f"PRAGMA table_info({table})")
                    columns = [col[1].lower() for col in cursor.fetchall()]
                    
                    # Check row count
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    row_count = cursor.fetchone()[0]
                    
                    # Check if this looks like market data
                    has_ohlcv = all(col in columns for col in ['open', 'high', 'low', 'close'])
                    has_timestamp = any(col in columns for col in ['timestamp', 'time', 'date'])
                    has_symbol = any(col in columns for col in ['symbol', 'pair'])
                    
                    if has_ohlcv and has_timestamp and row_count > 0:
                        logging.info(f"✅ Detected market data table: {table} ({row_count} rows)")
                        conn.close()
                        return table
                except:
                    continue
            
            conn.close()
            logging.warning("⚠️ No suitable market data table found, using default")
            return "price_data"  # Default fallback
            
        except Exception as e:
            logging.error(f"❌ Error detecting table: {e}")
            return "price_data"  # Default fallback
    
    def load_data(self, symbol: str, timeframe: str, days_back: int = 30) -> pd.DataFrame:
        """Load data from your existing database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = f"""
            SELECT timestamp, open, high, low, close, volume
            FROM {self.table_name} 
            WHERE symbol = ? AND timeframe = ? 
            AND timestamp >= datetime('now', '-{days_back} days')
            ORDER BY timestamp
            """
            
            df = pd.read_sql_query(query, conn, params=(symbol, timeframe))
            conn.close()
            
            if df.empty:
                logging.warning(f"⚠️ No data found for {symbol} {timeframe} in table {self.table_name}")
                return pd.DataFrame()
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            logging.info(f"📊 Loaded {len(df)} records for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            logging.error(f"❌ Error loading data: {e}")
            return pd.DataFrame()
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive features for ML models"""
        if df.empty:
            return df
        
        df = df.copy()
        
        # Basic price features
        df['price_change'] = df['close'].pct_change()
        df['high_low_pct'] = (df['high'] - df['low']) / df['low']
        df['close_open_pct'] = (df['close'] - df['open']) / df['open']
        
        # Technical indicators
        self._add_moving_averages(df)
        self._add_rsi(df)
        self._add_macd(df)
        self._add_bollinger_bands(df)
        self._add_volume_indicators(df)
        self._add_volatility_features(df)
        self._add_momentum_features(df)
        
        # Time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            df[f'price_change_lag_{lag}'] = df['price_change'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'close_mean_{window}'] = df['close'].rolling(window).mean()
            df[f'close_std_{window}'] = df['close'].rolling(window).std()
            df[f'volume_mean_{window}'] = df['volume'].rolling(window).mean()
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        logging.info(f"✅ Created {len(df.columns)} features")
        return df
    
    def _add_moving_averages(self, df: pd.DataFrame):
        """Add moving average features"""
        for window in [5, 10, 20, 50]:
            df[f'ma_{window}'] = df['close'].rolling(window).mean()
            df[f'ma_{window}_ratio'] = df['close'] / df[f'ma_{window}']
    
    def _add_rsi(self, df: pd.DataFrame, window: int = 14):
        """Add RSI indicator"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
    
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
    
    def _add_volume_indicators(self, df: pd.DataFrame):
        """Add volume-based indicators"""
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['price_volume'] = df['close'] * df['volume']
        df['vwap'] = df['price_volume'].rolling(20).mean() / df['volume'].rolling(20).mean()
    
    def _add_volatility_features(self, df: pd.DataFrame):
        """Add volatility features"""
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['price_change'].rolling(window).std()
            df[f'volatility_ratio_{window}'] = df[f'volatility_{window}'] / df[f'volatility_{window}'].rolling(50).mean()
    
    def _add_momentum_features(self, df: pd.DataFrame):
        """Add momentum features"""
        for window in [5, 10, 20]:
            df[f'momentum_{window}'] = df['close'] / df['close'].shift(window) - 1
            df[f'roc_{window}'] = df['close'].pct_change(window)
    
    def prepare_ml_data(self, df: pd.DataFrame, target_column: str = 'price_change', 
                       prediction_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for ML models"""
        if df.empty:
            return np.array([]), np.array([])
        
        # Create target variable (future price movement)
        if target_column == 'price_change':
            df['target'] = df['close'].shift(-prediction_horizon) / df['close'] - 1
        elif target_column == 'direction':
            df['target'] = (df['close'].shift(-prediction_horizon) > df['close']).astype(int)
        else:
            df['target'] = df[target_column].shift(-prediction_horizon)
        
        # Select features (exclude target and price columns)
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'target']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Store feature columns for later use
        self.feature_columns = feature_cols
        
        # Prepare X and y
        X = df[feature_cols].values
        y = df['target'].values
        
        # Remove NaN values
        valid_idx = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_idx]
        y = y[valid_idx]
        
        logging.info(f"📊 Prepared ML data: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def train_price_prediction_models(self, symbol: str, timeframe: str = '1h'):
        """Train models to predict future price movements"""
        logging.info(f"🧠 Training price prediction models for {symbol} {timeframe}")
        
        # Load and prepare data
        df = self.load_data(symbol, timeframe, days_back=90)
        if df.empty:
            return False
        
        df = self.create_features(df)
        X, y = self.prepare_ml_data(df, target_column='price_change')
        
        if len(X) == 0:
            logging.error("❌ No valid data for training")
            return False
        
        # Split data (time series split)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Store scaler
        self.scalers[f"{symbol}_{timeframe}"] = scaler
        
        models_to_train = {}
        
        # Random Forest
        if ML_AVAILABLE:
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf_model.fit(X_train_scaled, y_train)
            models_to_train['random_forest'] = rf_model
        
        # XGBoost
        if XGBOOST_AVAILABLE:
            xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
            xgb_model.fit(X_train_scaled, y_train)
            models_to_train['xgboost'] = xgb_model
        
        # Gradient Boosting
        if ML_AVAILABLE:
            gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            gb_model.fit(X_train_scaled, y_train)
            models_to_train['gradient_boosting'] = gb_model
        
        # Evaluate models
        best_model = None
        best_score = float('inf')
        
        for name, model in models_to_train.items():
            predictions = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            logging.info(f"📊 {name}: MSE={mse:.6f}, R2={r2:.4f}")
            
            if mse < best_score:
                best_score = mse
                best_model = (name, model)
        
        # Save best model
        if best_model:
            model_key = f"{symbol}_{timeframe}_price"
            self.models[model_key] = best_model[1]
            
            # Save to disk
            model_path = f"ml_models/{model_key}_{best_model[0]}.joblib"
            joblib.dump(best_model[1], model_path)
            
            scaler_path = f"ml_models/{model_key}_scaler.joblib"
            joblib.dump(scaler, scaler_path)
            
            logging.info(f"✅ Best model ({best_model[0]}) saved: {model_path}")
            return True
        
        return False
    
    def train_direction_prediction_models(self, symbol: str, timeframe: str = '1h'):
        """Train models to predict price direction (up/down)"""
        logging.info(f"🎯 Training direction prediction models for {symbol} {timeframe}")
        
        # Load and prepare data
        df = self.load_data(symbol, timeframe, days_back=90)
        if df.empty:
            return False
        
        df = self.create_features(df)
        X, y = self.prepare_ml_data(df, target_column='direction')
        
        if len(X) == 0:
            return False
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models_to_train = {}
        
        # Random Forest Classifier
        if ML_AVAILABLE:
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf_model.fit(X_train_scaled, y_train)
            models_to_train['random_forest'] = rf_model
        
        # Logistic Regression
        if ML_AVAILABLE:
            lr_model = LogisticRegression(random_state=42, max_iter=1000)
            lr_model.fit(X_train_scaled, y_train)
            models_to_train['logistic_regression'] = lr_model
        
        # XGBoost Classifier
        if XGBOOST_AVAILABLE:
            xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
            xgb_model.fit(X_train_scaled, y_train)
            models_to_train['xgboost'] = xgb_model
        
        # Evaluate models
        best_model = None
        best_accuracy = 0
        
        for name, model in models_to_train.items():
            predictions = model.predict(X_test_scaled)
            accuracy = (predictions == y_test).mean()
            
            logging.info(f"🎯 {name}: Accuracy={accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = (name, model)
        
        # Save best model
        if best_model:
            model_key = f"{symbol}_{timeframe}_direction"
            self.models[model_key] = best_model[1]
            
            # Save to disk
            model_path = f"ml_models/{model_key}_{best_model[0]}.joblib"
            joblib.dump(best_model[1], model_path)
            
            scaler_path = f"ml_models/{model_key}_scaler.joblib"
            joblib.dump(self.scalers[f"{symbol}_{timeframe}"], scaler_path)
            
            logging.info(f"✅ Best direction model ({best_model[0]}) saved: {model_path}")
            return True
        
        return False
    
    def train_lstm_model(self, symbol: str, timeframe: str = '1h', sequence_length: int = 60):
        """Train LSTM model for time series prediction"""
        if not DL_AVAILABLE:
            logging.error("❌ TensorFlow not available for LSTM training")
            return False
        
        logging.info(f"🧠 Training LSTM model for {symbol} {timeframe}")
        
        # Load data
        df = self.load_data(symbol, timeframe, days_back=180)
        if df.empty or len(df) < sequence_length * 2:
            logging.error("❌ Insufficient data for LSTM training")
            return False
        
        # Prepare price data for LSTM
        prices = df['close'].values.reshape(-1, 1)
        
        # Scale data
        scaler = MinMaxScaler()
        scaled_prices = scaler.fit_transform(prices)
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(scaled_prices)):
            X.append(scaled_prices[i-sequence_length:i, 0])
            y.append(scaled_prices[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # Train model
        early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
        model_path = f"ml_models/{symbol}_{timeframe}_lstm.h5"
        checkpoint = ModelCheckpoint(model_path, save_best_only=True)
        
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping, checkpoint],
            verbose=0
        )
        
        # Save scaler
        scaler_path = f"ml_models/{symbol}_{timeframe}_lstm_scaler.joblib"
        joblib.dump(scaler, scaler_path)
        
        # Store in memory
        self.models[f"{symbol}_{timeframe}_lstm"] = model
        self.scalers[f"{symbol}_{timeframe}_lstm"] = scaler
        
        logging.info(f"✅ LSTM model trained and saved: {model_path}")
        return True
    
    def make_predictions(self, symbol: str, timeframe: str = '1h') -> Dict:
        """Make predictions using trained models"""
        logging.info(f"🔮 Making predictions for {symbol} {timeframe}")
        
        # Load recent data
        df = self.load_data(symbol, timeframe, days_back=7)
        if df.empty:
            logging.warning(f"⚠️ No data available for {symbol} {timeframe}")
            return {}
        
        df_features = self.create_features(df)
        if df_features.empty:
            logging.warning(f"⚠️ No features could be created for {symbol} {timeframe}")
            return {}
        
        predictions = {}
        
        try:
            # Price prediction
            price_model_key = f"{symbol}_{timeframe}_price"
            if price_model_key in self.models:
                # Make sure we have the same features as during training
                scaler_key = f"{symbol}_{timeframe}"
                if scaler_key in self.scalers:
                    # Get the latest data point
                    if len(df_features) > 0:
                        # Use all available feature columns from the dataframe
                        exclude_cols = ['open', 'high', 'low', 'close', 'volume']
                        available_features = [col for col in df_features.columns if col not in exclude_cols]
                        
                        if len(available_features) > 0:
                            X_latest = df_features[available_features].iloc[-1:].values
                            
                            if X_latest.shape[1] > 0:  # Make sure we have features
                                scaler = self.scalers[scaler_key]
                                X_scaled = scaler.transform(X_latest)
                                pred = self.models[price_model_key].predict(X_scaled)[0]
                                predictions['price_change'] = pred
                                predictions['predicted_price'] = df['close'].iloc[-1] * (1 + pred)
                                logging.info(f"✅ Price prediction: {pred:.4f}")
                            else:
                                logging.warning(f"⚠️ No valid features for price prediction")
                        else:
                            logging.warning(f"⚠️ No feature columns available for price prediction")
                    else:
                        logging.warning(f"⚠️ No data points available for price prediction")
                else:
                    logging.warning(f"⚠️ No scaler found for {scaler_key}")
            else:
                logging.info(f"ℹ️ No price model found for {price_model_key}")
        
        except Exception as e:
            logging.error(f"❌ Price prediction failed: {e}")
        
        try:
            # Direction prediction
            direction_model_key = f"{symbol}_{timeframe}_direction"
            if direction_model_key in self.models:
                scaler_key = f"{symbol}_{timeframe}"
                if scaler_key in self.scalers:
                    if len(df_features) > 0:
                        exclude_cols = ['open', 'high', 'low', 'close', 'volume']
                        available_features = [col for col in df_features.columns if col not in exclude_cols]
                        
                        if len(available_features) > 0:
                            X_latest = df_features[available_features].iloc[-1:].values
                            
                            if X_latest.shape[1] > 0:
                                scaler = self.scalers[scaler_key]
                                X_scaled = scaler.transform(X_latest)
                                
                                direction_pred = self.models[direction_model_key].predict(X_scaled)[0]
                                direction_prob = self.models[direction_model_key].predict_proba(X_scaled)[0]
                                
                                predictions['direction'] = 'UP' if direction_pred == 1 else 'DOWN'
                                predictions['direction_probability'] = max(direction_prob)
                                logging.info(f"✅ Direction prediction: {predictions['direction']} ({predictions['direction_probability']:.4f})")
                            else:
                                logging.warning(f"⚠️ No valid features for direction prediction")
                        else:
                            logging.warning(f"⚠️ No feature columns available for direction prediction")
                    else:
                        logging.warning(f"⚠️ No data points available for direction prediction")
            else:
                logging.info(f"ℹ️ No direction model found for {direction_model_key}")
                
        except Exception as e:
            logging.error(f"❌ Direction prediction failed: {e}")
        
        # Skip LSTM prediction for now due to Keras compatibility issues
        try:
            lstm_model_key = f"{symbol}_{timeframe}_lstm"
            if lstm_model_key in self.models and DL_AVAILABLE:
                sequence_length = 60
                if len(df) >= sequence_length:
                    prices = df['close'].values[-sequence_length:].reshape(-1, 1)
                    scaler = self.scalers.get(lstm_model_key)
                    
                    if scaler is not None:
                        scaled_prices = scaler.transform(prices)
                        X_lstm = scaled_prices.reshape(1, sequence_length, 1)
                        
                        try:
                            lstm_pred = self.models[lstm_model_key].predict(X_lstm, verbose=0)[0][0]
                            lstm_price = scaler.inverse_transform([[lstm_pred]])[0][0]
                            
                            predictions['lstm_predicted_price'] = lstm_price
                            predictions['lstm_price_change'] = lstm_price / df['close'].iloc[-1] - 1
                            logging.info(f"✅ LSTM prediction: {predictions['lstm_price_change']:.4f}")
                        except Exception as lstm_e:
                            logging.warning(f"⚠️ LSTM prediction failed (compatibility issue): {lstm_e}")
                    else:
                        logging.warning(f"⚠️ No LSTM scaler found")
                else:
                    logging.warning(f"⚠️ Insufficient data for LSTM (need {sequence_length}, have {len(df)})")
            else:
                logging.info(f"ℹ️ No LSTM model found or TensorFlow not available")
                
        except Exception as e:
            logging.warning(f"⚠️ LSTM prediction skipped: {e}")
        
        if predictions:
            logging.info(f"✅ Generated {len(predictions)} predictions for {symbol} {timeframe}")
        else:
            logging.warning(f"⚠️ No predictions generated for {symbol} {timeframe}")
        
        return predictions
    
    def load_models(self, symbol: str, timeframe: str):
        """Load saved models from disk with better error handling"""
        model_files_info = [
            ('price', ['random_forest', 'xgboost', 'gradient_boosting']),
            ('direction', ['random_forest', 'logistic_regression', 'xgboost'])
        ]
        
        # Load traditional ML models
        for model_type, model_names in model_files_info:
            for model_name in model_names:
                model_path = f"ml_models/{symbol}_{timeframe}_{model_type}_{model_name}.joblib"
                scaler_path = f"ml_models/{symbol}_{timeframe}_{model_type}_scaler.joblib"
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    try:
                        model = joblib.load(model_path)
                        scaler = joblib.load(scaler_path)
                        
                        # Use the most recent model for each type
                        self.models[f"{symbol}_{timeframe}_{model_type}"] = model
                        self.scalers[f"{symbol}_{timeframe}"] = scaler
                        
                        logging.info(f"✅ Loaded {model_type} model: {model_name}")
                        break  # Use the first model found for each type
                        
                    except Exception as e:
                        logging.error(f"❌ Failed to load {model_path}: {e}")
                        continue
        
        # Load LSTM model with better error handling
        lstm_path = f"ml_models/{symbol}_{timeframe}_lstm.h5"
        lstm_scaler_path = f"ml_models/{symbol}_{timeframe}_lstm_scaler.joblib"
        
        if os.path.exists(lstm_path) and os.path.exists(lstm_scaler_path) and DL_AVAILABLE:
            try:
                # Try to load LSTM model with custom objects handling
                from tensorflow.keras.models import load_model
                import tensorflow.keras.metrics as metrics
                
                # Define custom objects to handle compatibility issues
                custom_objects = {
                    'mse': metrics.MeanSquaredError(),
                    'mean_squared_error': metrics.MeanSquaredError()
                }
                
                model = load_model(lstm_path, custom_objects=custom_objects, compile=False)
                scaler = joblib.load(lstm_scaler_path)
                
                # Recompile the model with current TensorFlow version
                from tensorflow.keras.optimizers import Adam
                model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
                
                self.models[f"{symbol}_{timeframe}_lstm"] = model
                self.scalers[f"{symbol}_{timeframe}_lstm"] = scaler
                
                logging.info("✅ Loaded LSTM model with compatibility fix")
                
            except Exception as e:
                logging.error(f"❌ Failed to load LSTM model: {e}")
                logging.info("ℹ️ LSTM predictions will be skipped due to compatibility issues")
        elif os.path.exists(lstm_path):
            logging.warning("⚠️ LSTM model found but TensorFlow not available")
        else:
            logging.info("ℹ️ No LSTM model found")
    
    def get_feature_importance(self, symbol: str, timeframe: str) -> Dict:
        """Get feature importance from trained models"""
        importance_dict = {}
        
        # Price model feature importance
        price_model_key = f"{symbol}_{timeframe}_price"
        if price_model_key in self.models:
            model = self.models[price_model_key]
            if hasattr(model, 'feature_importances_'):
                if len(self.feature_columns) == len(model.feature_importances_):
                    importance_pairs = list(zip(self.feature_columns, model.feature_importances_))
                    importance_pairs.sort(key=lambda x: x[1], reverse=True)
                    importance_dict = dict(importance_pairs)
        
        return importance_dict


def main():
    """Example usage of the ML system"""
    # Initialize ML system
    ml_system = CryptoMLSystem()
    
    symbols = ['BTC/USDT', 'ETH/USDT']
    timeframes = ['1h', '4h']
    
    for symbol in symbols:
        for timeframe in timeframes:
            print(f"\n🚀 Training models for {symbol} {timeframe}")
            
            # Train price prediction models
            if ml_system.train_price_prediction_models(symbol, timeframe):
                print(f"✅ Price prediction models trained for {symbol} {timeframe}")
            
            # Train direction prediction models
            if ml_system.train_direction_prediction_models(symbol, timeframe):
                print(f"✅ Direction prediction models trained for {symbol} {timeframe}")
            
            # Train LSTM model (if TensorFlow available)
            if DL_AVAILABLE:
                if ml_system.train_lstm_model(symbol, timeframe):
                    print(f"✅ LSTM model trained for {symbol} {timeframe}")
            
            # Make predictions
            predictions = ml_system.make_predictions(symbol, timeframe)
            if predictions:
                print(f"🔮 Predictions for {symbol} {timeframe}:")
                for key, value in predictions.items():
                    print(f"   {key}: {value}")
            
            # Get feature importance
            importance = ml_system.get_feature_importance(symbol, timeframe)
            if importance:
                print(f"📊 Top 5 most important features:")
                for i, (feature, imp) in enumerate(list(importance.items())[:5]):
                    print(f"   {i+1}. {feature}: {imp:.4f}")

if __name__ == "__main__":
    main()