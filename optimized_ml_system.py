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
from crypto_ai.database.db import DatabaseManager

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


class OptimizedCryptoMLSystem:
    """
    Optimized ML system using ensemble of:
    - LightGBM (Primary - Best for tabular data)
    - XGBoost (Secondary - Industry standard)
    - CatBoost (Tertiary - Robust against overfitting)
    - GRU (Deep Learning - For 4h timeframe)
    """
    
    
    def __init__(self, db_path='data/ml_crypto_data.db', n_features=50, enable_tuning=True, n_trials=20):
        """Initialize the optimized ML system
        
        Args:
            db_path: Path to database
            n_features: Number of features to select (default: 50)
            enable_tuning: Enable hyperparameter tuning with Optuna
            n_trials: Number of Optuna trials for hyperparameter tuning
        """
        self.db_path = db_path
        self.db_manager = DatabaseManager(db_path)
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

        # Check for GPU
        self.gpu_available = self._check_gpu_availability()
        logging.info(f"🚀 GPU Acceleration: {'Enabled' if self.gpu_available else 'Disabled (Not found)'}")

    def _check_gpu_availability(self) -> bool:
        """Check if NVIDIA GPU is available"""
        try:
            # Method 1: Check nvidia-smi
            import subprocess
            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                return True
            
            # Method 2: Check TensorFlow
            if DL_AVAILABLE:
                if len(tf.config.list_physical_devices('GPU')) > 0:
                    return True
                    
            return False
        except Exception:
            return False
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
    
    def load_data(self, symbol: str, timeframe: str, months_back: int = None, include_correlation: bool = True) -> pd.DataFrame:
        """
        Load data from unified database with optional correlation data
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe ('5m', '15m', '1h', '4h', '1d')
            months_back: Months of data to load (None = all available)
            include_correlation: Whether to load a benchmark symbol for correlation
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
            
            # 1. Load Main Symbol
            query = """
            SELECT timestamp, open, high, low, close, volume
            FROM price_data 
            WHERE symbol = ? AND timeframe = ? 
            AND timestamp >= datetime('now', '-{} days')
            ORDER BY timestamp
            """.format(days_back)
            
            df = pd.read_sql_query(query, conn, params=(symbol, timeframe))
            
            if df.empty:
                conn.close()
                logging.warning(f"⚠️ No data found for {symbol} {timeframe}")
                return pd.DataFrame()
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # 2. Load Correlation Symbol (if enabled)
            if include_correlation:
                # Select benchmark: ETH if symbol is BTC, otherwise BTC
                corr_symbol = 'ETH/USDT' if 'BTC' in symbol else 'BTC/USDT'
                
                df_corr = pd.read_sql_query(query, conn, params=(corr_symbol, timeframe))
                if not df_corr.empty:
                    df_corr['timestamp'] = pd.to_datetime(df_corr['timestamp'])
                    df_corr.set_index('timestamp', inplace=True)
                    
                    # Merge specific columns
                    df_corr = df_corr[['close', 'volume']].rename(
                        columns={'close': 'corr_close', 'volume': 'corr_volume'}
                    )
                    
                    # Merge closest timestamps (asof merge or direct join if aligned)
                    df = df.join(df_corr, how='left')
                    
                    # Forward fill correlation data (up to a limit)
                    df['corr_close'] = df['corr_close'].ffill(limit=5)
                    df['corr_volume'] = df['corr_volume'].ffill(limit=5)
                    
                    logging.info(f"➕ Added correlation data from {corr_symbol}")

            # 3. Load Advanced Features (Order Book & Funding)
            # Fetch for the same period (days_back converted to hours for OB)
            try:
                # Order Book
                df_ob = self.db_manager.load_order_book_data(symbol, hours=days_back*24)
                if not df_ob.empty:
                    # Rename columns to avoid collision and clarify source
                    df_ob = df_ob.drop(columns=['id', 'symbol']).rename(columns={
                        'bid_volume_depth': 'ob_bid_vol',
                        'ask_volume_depth': 'ob_ask_vol',
                        'spread_pct': 'ob_spread',
                        'imbalance_ratio': 'ob_imbalance'
                    })
                    # Sort for merge_asof
                    df_ob = df_ob.sort_values('timestamp')
                    df = df.sort_index() # Ensure main df is sorted
                    
                    # Merge using asof (backward search)
                    df = pd.merge_asof(df, df_ob, left_index=True, right_on='timestamp', 
                                     direction='backward', tolerance=pd.Timedelta('1h'))
                    
                    # Set index back to timestamp and clean
                    if 'timestamp' in df.columns:
                        df.set_index('timestamp', inplace=True)
                    logging.info(f"➕ Added Order Book features ({len(df_ob)} records)")
                
                # Funding Data
                # Force 365 days to ensure we catch backfilled history
                df_funding = self.db_manager.load_funding_data(symbol, days=max(days_back, 365))
                if not df_funding.empty:
                    df_funding = df_funding.drop(columns=['id', 'symbol'])
                    df_funding = df_funding.sort_values('timestamp')
                    
                    df = pd.merge_asof(df, df_funding, left_index=True, right_on='timestamp',
                                     direction='backward', tolerance=pd.Timedelta('4h'))
                    
                    if 'timestamp' in df.columns:
                        df.set_index('timestamp', inplace=True)
                    logging.info(f"➕ Added Funding features ({len(df_funding)} records)")
                
                # Arbitrage Data (External Prices - Force 365 days)
                df_ext = self.db_manager.load_external_price_data(symbol, days=max(days_back, 365))
                if not df_ext.empty:
                    # Rename for clarity
                    df_ext = df_ext[['timestamp', 'close_price']].rename(columns={'close_price': 'ext_price'})
                    df_ext = df_ext.sort_values('timestamp')
                    
                    df = pd.merge_asof(df, df_ext, left_index=True, right_on='timestamp',
                                     direction='backward', tolerance=pd.Timedelta('30m'))
                    
                    if 'timestamp' in df.columns:
                        df.set_index('timestamp', inplace=True)
                    logging.info(f"➕ Added Arbitrage features ({len(df_ext)} records)")
                    
            except Exception as e:
                logging.warning(f"⚠️ Error merging advanced features: {e}")

            conn.close()
            
            logging.info(f"📊 Loaded {len(df)} candles for {symbol} {timeframe} ({months_back} months)")
            return df
            
        except Exception as e:
            logging.error(f"❌ Error loading data: {e}")
            return pd.DataFrame()

    def load_sentiment(self, symbol: str, months_back: int = None) -> pd.DataFrame:
        """Load sentiment data from database"""
        try:
            if not os.path.exists(self.db_path):
                return pd.DataFrame()
                
            conn = sqlite3.connect(self.db_path)
            
            # Default to match price lookback
            if months_back is None:
                months_back = 6
            
            days_back = months_back * 30
            
            # Check if table exists first
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sentiment_data'")
            if not cursor.fetchone():
                conn.close()
                return pd.DataFrame()
            
            query = """
            SELECT *
            FROM sentiment_data 
            WHERE symbol = ? 
            AND timestamp >= datetime('now', '-{} days')
            ORDER BY timestamp
            """.format(days_back)
            
            # Note: Stored symbol typically includes pair (BTC/USDT)
            df = pd.read_sql_query(query, conn, params=(symbol,))
            conn.close()
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                # Don't set index yet, feature engineer expects column or index handling
                
            return df
            
        except Exception as e:
            logging.warning(f"⚠️ Could not load sentiment: {e}")
            return pd.DataFrame()
    
    def create_features(self, df: pd.DataFrame, sentiment_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Create comprehensive feature set optimized for gradient boosting
        with outlier removal and correlation features
        """
        if df.empty:
            return df
            
        logging.info("🧠 Generating features using centralized FeatureEngineer...")
        df_features = self.feature_engineer.create_features(df, sentiment_df)
        
        # --- DATA QUALITY IMPROVEMENTS ---
        
        # 1. Add Correlation Features (if loaded)
        if 'corr_close' in df.columns:
            # Correlation Score (30-period rolling correlation)
            df_features['corr_btc_eth'] = df['close'].rolling(30).corr(df['corr_close'])
            # Relative Strength (Price Ratio)
            df_features['rel_strength'] = df['close'] / df['corr_close']
            # Divergence (Difference in % change)
            df_features['corr_divergence'] = df['close'].pct_change() - df['corr_close'].pct_change()
            
            logging.info("✨ Generated correlation features (corr_btc_eth, rel_strength, divergence)")

        # 2. Outlier Removal (Clip extreme % changes)
        # We don't want to drop the rows (loss of data), but clip the artifacts
        # Using 99.9th percentile to catch flash crashes/API errors
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        
        # Only clip columns that look like returns/changes (small values)
        # Assuming most features are normalized or small ratios
        # Simple heuristic: Clip single-candle returns > 20%
        if 'close_open_pct' in df_features.columns:
            mask_outlier = df_features['close_open_pct'].abs() > 0.20
            outliers = mask_outlier.sum()
            if outliers > 0:
                logging.info(f"🧹 Clipped {outliers} extreme outliers (>20% candle change)")
                df_features.loc[mask_outlier, 'close_open_pct'] = \
                    df_features.loc[mask_outlier, 'close_open_pct'].clip(-0.20, 0.20)
        
        # Remove any rows with NaN values created during feature engineering
        initial_len = len(df)
        
        # DEBUG: Check which columns have NaNs
        nan_cols = df_features.columns[df_features.isna().any()].tolist()
        if nan_cols:
            logging.warning(f"⚠️ Columns with NaNs: {nan_cols}")
            # Check counts
            nan_counts = df_features[nan_cols].isna().sum().to_dict()
            logging.warning(f"⚠️ NaN Counts: {nan_counts}")
            
            # Check if any column is ALL NaNs
            if df_features.shape[0] > 0:
                all_nan = [c for c in nan_cols if df_features[c].isna().all()]
                if all_nan:
                    logging.error(f"❌ Columns with 100% NaNs: {all_nan}")
                    logging.info(f"🔧 Removing {len(all_nan)} columns with 100% NaNs")

        # CRITICAL FIX: Drop columns that are 100% NaN BEFORE dropping rows
        # This prevents the row-wise dropna from removing all training samples
        df_features = df_features.dropna(axis=1, how='all')
        logging.info(f"📊 Features after removing 100% NaN columns: {len(df_features.columns)}")
        
        # Now drop rows with NaN values in remaining columns
        df_features.dropna(inplace=True)
        dropped = initial_len - len(df_features)
        
        if dropped > 0:
            logging.info(f"ℹ️ Dropped {dropped} rows with NaN values")
            
        logging.info(f"✅ Created {len(df_features.columns)} features from {len(df_features)} samples")
        return df_features
    
    # _add_rsi, _add_macd, _add_bollinger_bands, _add_atr are no longer needed here as they are in FeatureEngineer
    # Removed to cleanup code
    
    def prepare_data(self, df: pd.DataFrame, prediction_type: str = 'price', threshold: float = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for ML models with Threshold Labeling
        
        Args:
            df: DataFrame with features
            prediction_type: 'price' or 'direction'
            threshold: % change threshold for labelling (e.g. 0.002 for 0.2%)
        
        Returns:
            X: Feature array
            y: Target array
            feature_names: List of feature names
        """
        if df.empty:
            return np.array([]), np.array([]), []
        
        # Create target variable
        future_return = df['close'].shift(-1) / df['close'] - 1
        
        if prediction_type == 'price':
            # Regression: Predict exact % change
            df['target'] = future_return
            
        elif prediction_type == 'direction':
            # Classification: 3-Class System (Threshold Labeling)
            # 0: Hold (Neutral/Noise)
            # 1: Buy (Significant Up)
            # 2: Sell (Significant Down)
            
            if threshold is None:
                # Default thresholds based on typical volatility
                # 1h: 0.2%, 4h: 0.5%, 1d: 1.0%
                threshold = 0.002 # Default 0.2%
            
            conditions = [
                (future_return > threshold),       # Buy
                (future_return < -threshold)       # Sell
            ]
            choices = [1, 2] # 1=Buy, 2=Sell
            
            # Default is 0 (Hold)
            df['target'] = np.select(conditions, choices, default=0)
            
            # Log threshold info
            logging.info(f"🎯 Threshold Labeling applied: >{threshold:.1%} = Buy, <-{threshold:.1%} = Sell")
            
        else:
            raise ValueError(f"Unknown prediction_type: {prediction_type}")
        
        # Exclude OHLCV, target, and correlation columns from features
        # Also exclude raw advanced columns that are sparse or non-stationary
        # CRITICAL: Include raw columns that might be 100% NaN if coverage too low
        exclude_cols = [
            'open', 'high', 'low', 'close', 'volume', 'target', 'corr_close', 'corr_volume',
            # Raw sparse columns (often 100% NaN)
            'ob_imbalance', 'ob_spread', 'funding_rate', 'open_interest',
            'best_bid_price', 'best_ask_price', 'ext_price', 
            'ob_bid_vol', 'ob_ask_vol', 'ob_base_volume', 'ob_quote_volume',
            'liquidations_long', 'liquidations_short',
            # Derived OB features (might be missing if coverage <10%)
            'ob_imbalance_ma12', 'ob_imbalance_delta', 'ob_spread_zscore',
            # Derived Funding features (might be missing if coverage <5%)
            'funding_rate_zscore', 'funding_cum_3d',
            # Derived OI features (might be missing if coverage <5%)
            'oi_pct_change', 'oi_sentiment'
        ]
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
            # Multi-class stats
            counts = pd.Series(y).value_counts().sort_index()
            total = len(y)
            logging.info(f"📊 Classes: Hold(0): {counts.get(0,0)} ({counts.get(0,0)/total:.1%}), "
                         f"Buy(1): {counts.get(1,0)} ({counts.get(1,0)/total:.1%}), "
                         f"Sell(2): {counts.get(2,0)} ({counts.get(2,0)/total:.1%})")
        
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
            
        # Load sentiment data (optional)
        sentiment_df = self.load_sentiment(symbol)
        if not sentiment_df.empty:
            logging.info(f"🧠 Loaded {len(sentiment_df)} sentiment records")
        
        # Create features
        df_features = self.create_features(df, sentiment_df)
        if df_features.empty:
            logging.error(f"❌ Failed to create features for {symbol} {timeframe}")
            return False
        
        # Train both price and direction models
        success_price = self._train_price_models(symbol, timeframe, df_features)
        success_direction = self._train_direction_models(symbol, timeframe, df_features)
        
        # Train GRU for 4h timeframe
        # Train GRU (Deep Learning) - Available for all timeframes if TF is installed
        if DL_AVAILABLE:
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
            else:
                lgb_params = {
                    'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 7, 'num_leaves': 31,
                    'min_child_samples': 20, 'subsample': 0.8, 'colsample_bytree': 0.8,
                    'reg_alpha': 0.1, 'reg_lambda': 0.1, 'random_state': 42, 'verbose': -1
                }
                
                # Enable GPU if available
                if self.gpu_available:
                    lgb_params.update({'device': 'gpu'})
                    # Note: LightGBM needs OpenCL/CUDA. If it crashes, user might need to reinstall.
                    
                lgb_model = lgb.LGBMRegressor(**lgb_params)
            
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
            else:
                xgb_params = {
                    'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 6, 'min_child_weight': 3,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.1,
                    'reg_alpha': 0.1, 'reg_lambda': 1.0, 'random_state': 42, 'verbosity': 0
                }
                
                # Enable GPU if available
                if self.gpu_available:
                     xgb_params.update({'device': 'cuda'}) # For XGBoost >= 2.0
                
                xgb_model = xgb.XGBRegressor(**xgb_params)
            
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
            else:
                cb_params = {
                    'iterations': 500, 'learning_rate': 0.05, 'depth': 6, 'l2_leaf_reg': 3,
                    'random_seed': 42, 'verbose': False
                }
                
                # Enable GPU if available
                if self.gpu_available:
                    cb_params.update({'task_type': 'GPU'})
                    
                cb_model = cb.CatBoostRegressor(**cb_params)
            
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
        """Train direction prediction models (MULTI-CLASS Classification)"""
        logging.info(f"\n🎯 Training Direction Prediction Models (3-Class: Hold/Buy/Sell)")
        logging.info("-" * 60)
        
        # Set threshold based on timeframe
        thresholds = {'5m': 0.001, '15m': 0.002, '1h': 0.003, '4h': 0.006, '1d': 0.015}
        threshold = thresholds.get(timeframe, 0.003)
        
        # Prepare data (3 classes)
        X, y, feature_cols = self.prepare_data(df, prediction_type='direction', threshold=threshold)
        
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
        
        # ========== 1. LightGBM Classifier (Multi-class) ==========
        if LIGHTGBM_AVAILABLE:
            logging.info("\n🌟 Training LightGBM Classifier...")
            
            if self.enable_tuning and OPTUNA_AVAILABLE:
                logging.info(f"   🔧 Tuning hyperparameters ({self.n_trials} trials)...")
                
                def objective(trial):
                    params = {
                        'objective': 'multiclass',
                        'num_class': 3,
                        'metric': 'multi_logloss',
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
                best_params.update({'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss'})
                
                logging.info(f"   ✅ Best Accuracy: {study.best_value:.2%}")
            else:
                lgb_params = {
                    'objective': 'multiclass', 'num_class': 3,
                    'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 7, 'num_leaves': 31,
                    'min_child_samples': 20, 'subsample': 0.8, 'colsample_bytree': 0.8,
                    'random_state': 42, 'verbose': -1
                }
                
                # Enable GPU if available
                if self.gpu_available:
                    lgb_params.update({'device': 'gpu'})
                    
                lgb_model = lgb.LGBMClassifier(**lgb_params)
            
            lgb_model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)],
                         callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            models['lightgbm'] = lgb_model
            
            # Evaluate using Weighted F1 (better for imbalanced classes)
            pred_val = lgb_model.predict(X_val_scaled)
            pred_test = lgb_model.predict(X_test_scaled)
            
            acc_val = accuracy_score(y_val, pred_val)
            acc_test = accuracy_score(y_test, pred_test)
            f1 = f1_score(y_test, pred_test, average='weighted', zero_division=0)
            
            # Confusion Matrix summary (Precision per class)
            report = classification_report(y_test, pred_test, output_dict=True, zero_division=0)
            prec_buy = report.get('1', {}).get('precision', 0)
            prec_sell = report.get('2', {}).get('precision', 0)
            
            val_scores['lightgbm'] = acc_val
            logging.info(f"✅ LightGBM - Val Acc: {acc_val:.2%}, Test Acc: {acc_test:.2%}, Weighted F1: {f1:.2%}")
            logging.info(f"   Precision - Buy: {prec_buy:.2%}, Sell: {prec_sell:.2%}")
        
        # ========== 2. XGBoost Classifier (Multi-class) ==========
        if XGBOOST_AVAILABLE:
            logging.info("\n🔥 Training XGBoost Classifier...")
            
            if self.enable_tuning and OPTUNA_AVAILABLE:
                logging.info(f"   🔧 Tuning hyperparameters ({self.n_trials} trials)...")
                
                def objective(trial):
                    params = {
                        'objective': 'multi:softprob',
                        'num_class': 3,
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
                best_params.update({'objective': 'multi:softprob', 'num_class': 3})
                
                logging.info(f"   ✅ Best Accuracy: {study.best_value:.2%}")
                # Add GPU parameter to best_params if available
                if self.gpu_available:
                    best_params.update({'tree_method': 'gpu_hist', 'device': 'cuda'})
                xgb_model = xgb.XGBClassifier(**best_params, verbosity=0)
            else:
                xgb_params = {
                    'objective': 'multi:softprob', 'num_class': 3,
                    'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 6, 'min_child_weight': 3,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'verbosity': 0
                }
                
                # Enable GPU if available
                if self.gpu_available:
                    xgb_params.update({'tree_method': 'gpu_hist', 'device': 'cuda'})
                    
                xgb_model = xgb.XGBClassifier(**xgb_params)
            
            xgb_model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
            models['xgboost'] = xgb_model
            
            # Evaluate
            pred_val = xgb_model.predict(X_val_scaled)
            pred_test = xgb_model.predict(X_test_scaled)
            
            acc_val = accuracy_score(y_val, pred_val)
            acc_test = accuracy_score(y_test, pred_test)
            f1 = f1_score(y_test, pred_test, average='weighted', zero_division=0)
            
            val_scores['xgboost'] = acc_val
            logging.info(f"✅ XGBoost - Val Acc: {acc_val:.2%}, Test Acc: {acc_test:.2%}, Weighted F1: {f1:.2%}")
        
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
            else:
                cb_params = {
                    'iterations': 500, 'learning_rate': 0.05, 'depth': 6, 'l2_leaf_reg': 3,
                    'random_seed': 42, 'verbose': False
                }
                
                # Enable GPU if available
                if self.gpu_available:
                    cb_params.update({'task_type': 'GPU'})
                    
                cb_model = cb.CatBoostClassifier(**cb_params)
            
            cb_model.fit(X_train_scaled, y_train, eval_set=(X_val_scaled, y_val),
                        early_stopping_rounds=50, verbose=False)
            models['catboost'] = cb_model
            
            # Evaluate
            pred_val = cb_model.predict(X_val_scaled)
            pred_test = cb_model.predict(X_test_scaled)
            acc_val = accuracy_score(y_val, pred_val)
            acc_test = accuracy_score(y_test, pred_test)
            f1 = f1_score(y_test, pred_test, average='weighted', zero_division=0)
            
            val_scores['catboost'] = acc_val
            logging.info(f"✅ CatBoost - Val Acc: {acc_val:.2%}, Test Acc: {acc_test:.2%}, Weighted F1: {f1:.2%}")
        
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
        """Train GRU model for time series"""
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
            'current_price': float(df['close'].iloc[-1]),
            'model_predictions': {}
        }
        
        # Load models
        self._load_models(symbol, timeframe)
        
        # Get latest features for tabular models
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'target', 'corr_close', 'corr_volume']
        available_features = [col for col in df_features.columns if col not in exclude_cols]
        
        if not available_features:
            logging.error("❌ No features available")
            return predictions
        
        X_latest = df_features[available_features].iloc[-1:].values
        
        weight_config = {}
        
        if f"{symbol}_{timeframe}_price" in self.model_performance:
            perf_data = self.model_performance[f"{symbol}_{timeframe}_price"]
            if 'dynamic_weights' in perf_data:
                logging.info(f"⚖️ Using dynamic weights for price: {perf_data['dynamic_weights']}")
                # Merge dynamic weights with existing keys to ensure we have all required models
                # (Dynamic weights might only contain valid ones, but we need structure)
                for k, v in perf_data['dynamic_weights'].items():
                    weight_config[k] = v

        if f"{symbol}_{timeframe}_gru" in self.models and 'gru' not in weight_config:
             # Ensure GRU has a weight if loaded but not dynamically tuned
             weight_config['gru'] = 0.10

        # 1. Price predictions (Regression)
        price_preds = []
        price_weights = []
        
        # Get price scaler and selector
        selector_price = self.feature_selectors.get(f"{symbol}_{timeframe}_price")
        scaler_price = self.scalers.get(f"{symbol}_{timeframe}_price")
        
        if selector_price and scaler_price:
            try:
                # Transform features: Selection then Scaling
                X_selected = selector_price.transform(X_latest)
                X_scaled = scaler_price.transform(X_selected)
                
                # LightGBM Price
                if f"{symbol}_{timeframe}_price_lightgbm" in self.models:
                    pred = self.models[f"{symbol}_{timeframe}_price_lightgbm"].predict(X_scaled)[0]
                    price_preds.append(pred)
                    weight = weight_config.get('lightgbm', 0.33)
                    price_weights.append(weight)
                    predictions['model_predictions']['lightgbm'] = {
                        'predicted_price': float(df['close'].iloc[-1] * (1 + pred)),
                        'price_change_pct': float(pred),
                        'confidence': float(weight)
                    }
                    logging.debug(f"📊 LightGBM price: {pred:+.4%}")
                
                # XGBoost Price
                if f"{symbol}_{timeframe}_price_xgboost" in self.models:
                    pred = self.models[f"{symbol}_{timeframe}_price_xgboost"].predict(X_scaled)[0]
                    price_preds.append(pred)
                    weight = weight_config.get('xgboost', 0.33)
                    price_weights.append(weight)
                    predictions['model_predictions']['xgboost'] = {
                        'predicted_price': float(df['close'].iloc[-1] * (1 + pred)),
                        'price_change_pct': float(pred),
                        'confidence': float(weight)
                    }
                    logging.debug(f"📊 XGBoost price: {pred:+.4%}")
                
                # CatBoost Price
                if f"{symbol}_{timeframe}_price_catboost" in self.models:
                    pred = self.models[f"{symbol}_{timeframe}_price_catboost"].predict(X_scaled)[0]
                    price_preds.append(pred)
                    weight = weight_config.get('catboost', 0.33)
                    price_weights.append(weight)
                    predictions['model_predictions']['catboost'] = {
                        'predicted_price': float(df['close'].iloc[-1] * (1 + pred)),
                        'price_change_pct': float(pred),
                        'confidence': float(weight)
                    }
                    logging.debug(f"📊 CatBoost price: {pred:+.4%}")
            except Exception as e:
                logging.error(f"⚠️ Price model prediction failed: {e}")
        
        # GRU (Deep Learning)
        if f"{symbol}_{timeframe}_gru" in self.models:
            self._predict_gru_price(df, symbol, timeframe, price_preds, price_weights, weight_config, predictions['model_predictions'])

        # Calculate ensemble price prediction
        if price_preds:
            total_weight = sum(price_weights)
            normalized_weights = [w/total_weight for w in price_weights]
            ensemble_price_change = sum(p * w for p, w in zip(price_preds, normalized_weights))
            
            # Clip unrealistic predictions
            prediction_limits = {'5m': 0.02, '15m': 0.03, '1h': 0.05, '4h': 0.10, '1d': 0.15}
            max_change = prediction_limits.get(timeframe, 0.05)
            ensemble_price_change = np.clip(ensemble_price_change, -max_change, max_change)
            
            predictions['price_change_pct'] = float(ensemble_price_change * 100)
            predictions['predicted_price'] = float(df['close'].iloc[-1] * (1 + ensemble_price_change))
            
        # 2. Direction predictions (Soft Voting Implementation)
        direction_probs = []
        direction_weights = []
        
        # Dynamic weights for direction
        dir_weight_config = weight_config.copy() # Default to fallback
        if f"{symbol}_{timeframe}_direction" in self.model_performance:
             perf_data = self.model_performance[f"{symbol}_{timeframe}_direction"]
             if 'dynamic_weights' in perf_data:
                logging.info(f"⚖️ Using dynamic weights for direction: {perf_data['dynamic_weights']}")
                for k, v in perf_data['dynamic_weights'].items():
                    dir_weight_config[k] = v

        # Get direction selector
        selector_dir = self.feature_selectors.get(f"{symbol}_{timeframe}_direction")
        
        # Breakdown to be populated by _get_model_prob or manually here
        predictions['model_breakdown'] = {}

        if selector_dir:
            try:
                # Transform features using selector
                X_dir_selected = selector_dir.transform(X_latest)
                
                # Get probabilities from each model
                for model_name in ['lightgbm', 'xgboost', 'catboost']:
                     prob_dict = self._get_model_prob(symbol, timeframe, model_name, X_dir_selected, direction_probs, direction_weights, dir_weight_config)
                     if prob_dict is not None:
                         # Determine model's vote
                         if prob_dict['buy'] > prob_dict['sell'] and prob_dict['buy'] > prob_dict['hold']:
                             vote = 'UP'
                         elif prob_dict['sell'] > prob_dict['buy'] and prob_dict['sell'] > prob_dict['hold']:
                             vote = 'DOWN'
                         else:
                             vote = 'NEUTRAL'
                             
                         predictions['model_breakdown'][model_name] = {
                             'weight': float(dir_weight_config.get(model_name, 0)),
                             'prob_buy': float(prob_dict['buy']),
                             'prob_sell': float(prob_dict['sell']),
                             'prediction': vote
                         }

            except Exception as e:
                logging.error(f"⚠️ Direction model prediction failed: {e}")
            
        # Calculate weighted probabilities
        if direction_probs:
            total_dir_weight = sum(direction_weights)
            norm_dir_weights = [w/total_dir_weight for w in direction_weights]
            
            # Aggregate probabilities
            weighted_buy = sum(p['buy'] * w for p, w in zip(direction_probs, norm_dir_weights))
            weighted_sell = sum(p['sell'] * w for p, w in zip(direction_probs, norm_dir_weights))
            weighted_hold = sum(p['hold'] * w for p, w in zip(direction_probs, norm_dir_weights))
            
            # Decision Logic
            # We want to be confident.
            # If Buy is strongest signal
            if weighted_buy > weighted_sell and weighted_buy > weighted_hold:
                predictions['direction'] = 'UP'
                combined_conf = weighted_buy
            # If Sell is strongest
            elif weighted_sell > weighted_buy and weighted_sell > weighted_hold:
                predictions['direction'] = 'DOWN'
                combined_conf = weighted_sell
            # Else Hold/Neutral
            else:
                predictions['direction'] = 'NEUTRAL'
                combined_conf = weighted_hold
                
            predictions['direction_confidence'] = float(combined_conf)
            predictions['confidence'] = float(combined_conf)
            
            logging.info(f"🎯 Ensemble: {predictions['direction']} (Buy: {weighted_buy:.2%}, Sell: {weighted_sell:.2%}, Hold: {weighted_hold:.2%})")

        logging.info(f"✅ Ensemble prediction complete!\n")
        return predictions

    def _get_model_prob(self, symbol, timeframe, model_name, X_latest_selected, probs_list, weights_list, weight_config):
        """Helper to get probability from a specific model (3-class: Hold/Buy/Sell)"""
        model_key = f"{symbol}_{timeframe}_direction_{model_name}"
        if model_key in self.models:
            scaler = self.scalers.get(f"{symbol}_{timeframe}_direction")
            if scaler is not None:
                # Expects feature-selected input
                X_scaled = scaler.transform(X_latest_selected)
                
                # predict_proba returns [prob_hold, prob_buy, prob_sell]
                try:
                    probs = self.models[model_key].predict_proba(X_scaled)[0]
                    
                    # Handle cases where model might not have seen all classes (rare but possible)
                    # We assume classes are [0, 1, 2] if initialized correctly with num_class=3
                    # But sklearn/joblib load might be tricky if one class was never seen.
                    # Strict validation:
                    if len(probs) == 3:
                        prob_hold, prob_buy, prob_sell = probs[0], probs[1], probs[2]
                    elif len(probs) == 2:
                        # Fallback if binary (old model or weird split)
                        # Assume 0/1
                        prob_hold = probs[0]
                        prob_buy = probs[1]
                        prob_sell = 0.0
                    else:
                        prob_hold, prob_buy, prob_sell = 1.0, 0.0, 0.0 # Fail safe
                    
                    prob_dict = {'hold': prob_hold, 'buy': prob_buy, 'sell': prob_sell}
                    
                    probs_list.append(prob_dict)
                    weights_list.append(weight_config.get(model_name, 0.33))
                    return prob_dict
                except Exception as e:
                    logging.warning(f"⚠️ Prob extraction error for {model_name}: {e}")
                    return None
        return None

    def _predict_gru_price(self, df, symbol, timeframe, price_preds, price_weights, weight_config, model_predictions=None):
        """Helper for GRU prediction to keep main method clean"""
        sequence_length = 60
        if len(df) >= sequence_length:
            scaler = self.scalers.get(f"{symbol}_{timeframe}_gru")
            if scaler is not None:
                try:
                    prices = df['close'].values[-sequence_length:].reshape(-1, 1)
                    scaled_prices = scaler.transform(prices)
                    X_gru = scaled_prices.reshape(1, sequence_length, 1)
                    
                    gru_pred_scaled = self.models[f"{symbol}_{timeframe}_gru"].predict(X_gru, verbose=0)[0][0]
                    gru_pred_price = scaler.inverse_transform([[gru_pred_scaled]])[0][0]
                    gru_pred_change = gru_pred_price / df['close'].iloc[-1] - 1
                    
                    price_preds.append(gru_pred_change)
                    weight = weight_config.get('gru', 0.10)
                    price_weights.append(weight)
                    
                    if model_predictions is not None:
                        model_predictions['gru'] = {
                            'predicted_price': float(gru_pred_price),
                            'price_change_pct': float(gru_pred_change),
                            'confidence': float(weight)
                        }
                except Exception as e:
                    logging.warning(f"⚠️ GRU prediction failed: {e}")
    
    def calculate_confidence(self, price_preds: list, direction_votes: dict) -> float:
        """
        Legacy method kept for interface compatibility, but logic moved to probability-based approach above.
        """
        return 0.0
    
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
    print(f"   Models per symbol/timeframe: 6-7 (LightGBM, XGBoost, CatBoost x2, GRU)")
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