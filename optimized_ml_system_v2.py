"""
Optimized ML System v2.0 - Enhanced with Feature Selection & Hyperparameter Tuning
============================================================================
Models: CatBoost + XGBoost (2 models only - optimized based on empirical data)
Features: SelectKBest with top 50 features (reduced from 71)
Tuning: Optuna hyperparameter optimization (one-time per symbol/timeframe)
Database: ml_crypto_data.db (unified database)

Changes from v1:
- Reduced from 4 models to 2 (CatBoost + XGBoost)
- Added intelligent feature selection (mutual_info_classif)
- Added Optuna hyperparameter tuning
- Saved optimal parameters to config for fast retraining
- 50% reduction in training time and storage
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import sqlite3
import joblib
import json
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional

# ML Libraries
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

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
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna not available. Install: pip install optuna")

try:
    from crypto_ai.gpu_utils import (
        detect_gpu_availability, get_xgboost_gpu_params, 
        get_catboost_gpu_params, XGBOOST_GPU_AVAILABLE, 
        CATBOOST_GPU_AVAILABLE
    )
    GPU_UTILS_AVAILABLE = True
except ImportError:
    GPU_UTILS_AVAILABLE = False
    XGBOOST_GPU_AVAILABLE = False
    CATBOOST_GPU_AVAILABLE = False


class OptimizedMLSystemV2:
    """
    Enhanced ML system with:
    - 2 models only (CatBoost + XGBoost)
    - Feature selection (top 50 features)
    - Hyperparameter optimization (Optuna)
    """
    
    def __init__(self, db_path='data/ml_crypto_data.db'):
        """Initialize the optimized ML system v2"""
        self.db_path = db_path
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.selected_features = {}
        self.optimal_params = {}
        
        # Priority features from collaborator
        self.priority_features = [
            'stoch_k', 'stoch_d', 'stoch_divergence', 'williams_r',
            'obv', 'obv_sma', 'obv_ratio', 'obv_change',
            'ichimoku_tenkan', 'ichimoku_kijun', 'ichimoku_senkou_a', 'ichimoku_senkou_b',
            'ichimoku_cloud_thickness', 'ichimoku_cloud_position', 'ichimoku_tk_diff',
            'plus_di', 'minus_di', 'adx', 'adx_trend_direction',
            'market_regime', 'ob_imbalance_ma12', 'ob_imbalance_delta', 'ob_spread_zscore',
            'funding_rate_zscore', 'funding_cum_3d', 'oi_pct_change', 'oi_sentiment',
            'arb_exch_delta_pct', 'arb_delta_zscore', 'ext_price_roc', 'arb_roc_divergence',
            'corr_btc_eth', 'rel_strength', 'corr_divergence'
        ]
        
        # Create directories
        os.makedirs('ml_models', exist_ok=True)
        os.makedirs('ml_configs', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # Setup logging
        self.log_file = self._setup_logging()
        
        logging.info("🧠 Optimized ML System v2.0 initialized")
        logging.info(f"📁 Database: {self.db_path}")
        logging.info(f"📝 Log file: {self.log_file}")
        logging.info("✨ Enhancements:")
        logging.info("   • 2 models only: CatBoost + XGBoost")
        logging.info("   • Feature selection: Top 50 features")
        logging.info("   • Hyperparameter optimization: Optuna")
        logging.info("   • 50% faster training, 50% less storage")
        logging.info(f"✅ XGBoost: {XGBOOST_AVAILABLE}")
        logging.info(f"✅ CatBoost: {CATBOOST_AVAILABLE}")
        logging.info(f"✅ Optuna: {OPTUNA_AVAILABLE}")
        
        if GPU_UTILS_AVAILABLE:
            detect_gpu_availability()
    
    def _setup_logging(self):
        """Setup logging to both console and file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'logs/ml_training_v2_{timestamp}.log'
        
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        logging.root.addHandler(file_handler)
        logging.root.addHandler(console_handler)
        logging.root.setLevel(logging.DEBUG)
        
        return log_file
    
    def load_data(self, symbol: str, timeframe: str, months_back: int = None) -> pd.DataFrame:
        """Load data from unified database with adaptive lookback"""
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
            
            logging.info(f"📊 Loaded {len(df)} candles for {symbol} {timeframe} ({months_back} months)")
            return df
            
        except Exception as e:
            logging.error(f"❌ Error loading data: {e}")
            return pd.DataFrame()
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive feature set
        Includes both standard features + priority features from collaborator
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # === 1. Basic price features ===
        df['price_change'] = df['close'].pct_change()
        df['high_low_pct'] = (df['high'] - df['low']) / df['low']
        df['close_open_pct'] = (df['close'] - df['open']) / df['open']
        
        # === 2. Moving averages ===
        for window in [5, 10, 20, 50]:
            df[f'ma_{window}'] = df['close'].rolling(window).mean()
            df[f'ma_{window}_ratio'] = df['close'] / df[f'ma_{window}']
        
        # === 3. RSI (multiple periods) ===
        for period in [14, 21, 28]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # === 4. MACD ===
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # === 5. Bollinger Bands ===
        rolling_mean = df['close'].rolling(20).mean()
        rolling_std = df['close'].rolling(20).std()
        df['bb_upper'] = rolling_mean + (rolling_std * 2)
        df['bb_lower'] = rolling_mean - (rolling_std * 2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # === 6. ATR ===
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        # === 7. Volume indicators ===
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['price_volume'] = df['close'] * df['volume']
        df['vwap'] = (df['price_volume'].rolling(20).sum() / 
                      df['volume'].rolling(20).sum())
        
        # === 8. OBV (On-Balance Volume) ===
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv_sma'] = df['obv'].rolling(20).mean()
        df['obv_ratio'] = df['obv'] / df['obv_sma']
        df['obv_change'] = df['obv'].pct_change()
        
        # === 9. Stochastic Oscillator ===
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        df['stoch_divergence'] = df['stoch_k'] - df['stoch_d']
        
        # === 10. Williams %R ===
        df['williams_r'] = -100 * ((high_14 - df['close']) / (high_14 - low_14))
        
        # === 11. ADX and Directional Indicators ===
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        tr = true_range
        atr_14 = tr.rolling(14).mean()
        df['plus_di'] = 100 * (plus_dm.rolling(14).mean() / atr_14)
        df['minus_di'] = 100 * (minus_dm.rolling(14).mean() / atr_14)
        dx = 100 * np.abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        df['adx'] = dx.rolling(14).mean()
        df['adx_trend_direction'] = np.where(df['plus_di'] > df['minus_di'], 1, -1)
        
        # === 12. Ichimoku Cloud ===
        high_9 = df['high'].rolling(9).max()
        low_9 = df['low'].rolling(9).min()
        df['ichimoku_tenkan'] = (high_9 + low_9) / 2
        
        high_26 = df['high'].rolling(26).max()
        low_26 = df['low'].rolling(26).min()
        df['ichimoku_kijun'] = (high_26 + low_26) / 2
        
        df['ichimoku_senkou_a'] = ((df['ichimoku_tenkan'] + df['ichimoku_kijun']) / 2).shift(26)
        
        high_52 = df['high'].rolling(52).max()
        low_52 = df['low'].rolling(52).min()
        df['ichimoku_senkou_b'] = ((high_52 + low_52) / 2).shift(26)
        
        df['ichimoku_cloud_thickness'] = df['ichimoku_senkou_a'] - df['ichimoku_senkou_b']
        df['ichimoku_cloud_position'] = np.where(df['close'] > df['ichimoku_senkou_a'], 1,
                                                  np.where(df['close'] < df['ichimoku_senkou_b'], -1, 0))
        df['ichimoku_tk_diff'] = df['ichimoku_tenkan'] - df['ichimoku_kijun']
        
        # === 13. Volatility features ===
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['price_change'].rolling(window).std()
        
        # === 14. Momentum features ===
        for window in [5, 10, 20]:
            df[f'momentum_{window}'] = df['close'] / df['close'].shift(window) - 1
            df[f'roc_{window}'] = df['close'].pct_change(window)
        
        # === 15. Lag features ===
        for lag in [1, 2, 3, 5]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        
        # === 16. Market regime (placeholder - will be enhanced) ===
        df['market_regime'] = np.where(df['ma_20'] > df['ma_50'], 1, -1)
        
        # === 17. Correlations (placeholder for BTC correlation) ===
        df['corr_btc_eth'] = df['close'].rolling(20).corr(df['close'])  # Simplified
        df['rel_strength'] = df['close'] / df['ma_50']
        df['corr_divergence'] = df['rsi_14'] - 50
        
        # === 18. Order book / funding rate placeholders ===
        # These would require external data - using proxies
        df['ob_imbalance_ma12'] = df['volume'].rolling(12).mean()
        df['ob_imbalance_delta'] = df['volume'].diff()
        df['ob_spread_zscore'] = (df['high'] - df['low'] - df['atr']) / df['atr'].std()
        df['funding_rate_zscore'] = 0  # Placeholder
        df['funding_cum_3d'] = 0  # Placeholder
        df['oi_pct_change'] = df['volume'].pct_change()
        df['oi_sentiment'] = np.where(df['close'] > df['ma_20'], 1, -1)
        
        # === 19. Arbitrage indicators (proxies) ===
        df['arb_exch_delta_pct'] = df['price_change']
        df['arb_delta_zscore'] = (df['price_change'] - df['price_change'].rolling(20).mean()) / df['price_change'].rolling(20).std()
        df['ext_price_roc'] = df['roc_5']
        df['arb_roc_divergence'] = df['roc_5'] - df['roc_10']
        
        # Drop NaN values
        initial_len = len(df)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        dropped = initial_len - len(df)
        
        if dropped > 0:
            logging.info(f"ℹ️ Dropped {dropped} rows with NaN values")
        
        logging.info(f"✅ Created {len(df.columns)} features from {len(df)} samples")
        return df
    
    def select_features(self, X: np.ndarray, y: np.ndarray, feature_names: List[str],
                       task_type: str = 'classification', n_features: int = 50) -> Tuple[np.ndarray, List[str], object]:
        """
        Select top N features using SelectKBest
        
        Args:
            X: Feature array
            y: Target array
            feature_names: List of feature names
            task_type: 'classification' or 'regression'
            n_features: Number of features to select
        
        Returns:
            X_selected: Selected features
            selected_names: Names of selected features
            selector: Fitted selector object
        """
        logging.info(f"🔍 Selecting top {n_features} features from {len(feature_names)}")
        
        # Choose scoring function
        if task_type == 'classification':
            score_func = mutual_info_classif
        else:
            score_func = mutual_info_regression
        
        # Fit selector
        selector = SelectKBest(score_func=score_func, k=min(n_features, len(feature_names)))
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        selected_names = [name for name, selected in zip(feature_names, selected_mask) if selected]
        
        # Get feature scores
        scores = selector.scores_
        feature_scores = list(zip(feature_names, scores))
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        logging.info(f"✅ Selected {len(selected_names)} features")
        logging.info(f"📊 Top 10 features:")
        for i, (name, score) in enumerate(feature_scores[:10], 1):
            logging.info(f"   {i}. {name}: {score:.4f}")
        
        return X_selected, selected_names, selector
    
    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                                 model_type: str, task: str = 'classification',
                                 n_trials: int = 10) -> Dict:
        """
        Optimize hyperparameters using Optuna
        
        Args:
            X_train: Training features
            y_train: Training target
            model_type: 'catboost' or 'xgboost'
            task: 'classification' or 'regression'
            n_trials: Number of optimization trials
        
        Returns:
            best_params: Dictionary of optimal parameters
        """
        if not OPTUNA_AVAILABLE:
            logging.warning("⚠️ Optuna not available, using default parameters")
            return self._get_default_params(model_type, task)
        
        logging.info(f"🔧 Optimizing {model_type} hyperparameters ({n_trials} trials)...")
        
        def objective(trial):
            if model_type == 'catboost':
                params = {
                    'iterations': trial.suggest_int('iterations', 100, 500),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'depth': trial.suggest_int('depth', 4, 10),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                    'random_seed': 42,
                    'verbose': False
                }
                # Add GPU params
                if GPU_UTILS_AVAILABLE:
                    params.update(get_catboost_gpu_params())
                
                if task == 'classification':
                    model = cb.CatBoostClassifier(**params)
                else:
                    model = cb.CatBoostRegressor(**params)
            
            elif model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0.0, 0.5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
                    'random_state': 42,
                    'verbosity': 0
                }
                # Add GPU params
                if GPU_UTILS_AVAILABLE:
                    params.update(get_xgboost_gpu_params())
                
                if task == 'classification':
                    model = xgb.XGBClassifier(**params)
                else:
                    model = xgb.XGBRegressor(**params)
            
            # Cross-validation score
            tss = TimeSeriesSplit(n_splits=3)
            scores = cross_val_score(model, X_train, y_train, cv=tss, scoring='accuracy' if task == 'classification' else 'r2')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        best_params = study.best_params
        best_score = study.best_value
        
        logging.info(f"✅ Best score: {best_score:.4f}")
        logging.info(f"📊 Best parameters: {best_params}")
        
        return best_params
    
    def _get_default_params(self, model_type: str, task: str) -> Dict:
        """Get default parameters if Optuna is not available"""
        if model_type == 'catboost':
            params = {
                'iterations': 500,
                'learning_rate': 0.05,
                'depth': 6,
                'l2_leaf_reg': 3,
                'random_seed': 42,
                'verbose': False
            }
            if GPU_UTILS_AVAILABLE:
                params.update(get_catboost_gpu_params())
            return params
        elif model_type == 'xgboost':
            params = {
                'n_estimators': 500,
                'learning_rate': 0.05,
                'max_depth': 6,
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'verbosity': 0
            }
            if GPU_UTILS_AVAILABLE:
                params.update(get_xgboost_gpu_params())
            return params
    
    def save_optimal_params(self, symbol: str, timeframe: str, model_type: str, task: str, params: Dict):
        """Save optimal parameters to config file"""
        safe_symbol = symbol.replace('/', '_')
        config_file = f"ml_configs/{safe_symbol}_{timeframe}_{task}_{model_type}_params.json"
        
        with open(config_file, 'w') as f:
            json.dump(params, f, indent=2)
        
        logging.info(f"💾 Saved optimal parameters: {config_file}")
    
    def load_optimal_params(self, symbol: str, timeframe: str, model_type: str, task: str) -> Optional[Dict]:
        """Load optimal parameters from config file"""
        safe_symbol = symbol.replace('/', '_')
        config_file = f"ml_configs/{safe_symbol}_{timeframe}_{task}_{model_type}_params.json"
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                params = json.load(f)
            logging.info(f"📂 Loaded saved parameters: {config_file}")
            return params
        return None
    
    def train_ensemble(self, symbol: str, timeframe: str, optimize: bool = False):
        """
        Train 2-model ensemble: CatBoost + XGBoost
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            optimize: Whether to run Optuna optimization (slow)
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
        success_price = self._train_models(symbol, timeframe, df_features, 'price', optimize)
        success_direction = self._train_models(symbol, timeframe, df_features, 'direction', optimize)
        
        return success_price or success_direction
    
    def _train_models(self, symbol: str, timeframe: str, df: pd.DataFrame, task: str, optimize: bool) -> bool:
        """Train both CatBoost and XGBoost for a specific task"""
        logging.info(f"\n📈 Training {task.upper()} Models")
        logging.info("-" * 60)
        
        # Prepare data
        if task == 'price':
            df['target'] = df['close'].shift(-1) / df['close'] - 1
            task_type = 'regression'
        else:  # direction
            df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            task_type = 'classification'
        
        # Exclude OHLCV and target
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'target']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].values
        y = df['target'].values
        
        # Remove NaN
        valid_idx = ~np.isnan(y)
        X = X[valid_idx]
        y = y[valid_idx]
        valid_idx = ~np.isnan(X).any(axis=1)
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) == 0:
            logging.error(f"❌ No valid data for {task}")
            return False
        
        logging.info(f"📊 Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Feature selection
        X_selected, selected_names, selector = self.select_features(
            X, y, feature_cols, task_type, n_features=50
        )
        
        # Time series split
        train_size = int(len(X_selected) * 0.70)
        val_size = int(len(X_selected) * 0.85)
        
        X_train = X_selected[:train_size]
        y_train = y[:train_size]
        X_val = X_selected[train_size:val_size]
        y_val = y[train_size:val_size]
        X_test = X_selected[val_size:]
        y_test = y[val_size:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        logging.info(f"📊 Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Store components
        safe_symbol = symbol.replace('/', '_')
        self.scalers[f"{safe_symbol}_{timeframe}_{task}"] = scaler
        self.feature_selectors[f"{safe_symbol}_{timeframe}_{task}"] = selector
        self.selected_features[f"{safe_symbol}_{timeframe}_{task}"] = selected_names
        
        models = {}
        
        # === Train CatBoost ===
        if CATBOOST_AVAILABLE:
            logging.info("\n🐱 Training CatBoost...")
            
            # Load or optimize parameters
            if optimize:
                best_params = self.optimize_hyperparameters(X_train_scaled, y_train, 'catboost', task_type)
                self.save_optimal_params(symbol, timeframe, 'catboost', task, best_params)
            else:
                best_params = self.load_optimal_params(symbol, timeframe, 'catboost', task)
                if best_params is None:
                    best_params = self._get_default_params('catboost', task_type)
            
            # Ensure GPU params are included
            if GPU_UTILS_AVAILABLE:
                best_params.update(get_catboost_gpu_params())
            
            # Train model
            if task_type == 'classification':
                cb_model = cb.CatBoostClassifier(**best_params)
            else:
                cb_model = cb.CatBoostRegressor(**best_params)
            
            cb_model.fit(
                X_train_scaled, y_train,
                eval_set=(X_val_scaled, y_val),
                early_stopping_rounds=50,
                verbose=False
            )
            models['catboost'] = cb_model
            
            # Evaluate
            pred = cb_model.predict(X_test_scaled)
            if task_type == 'classification':
                acc = accuracy_score(y_test, pred)
                logging.info(f"✅ CatBoost - Accuracy: {acc:.2%}")
            else:
                mse = mean_squared_error(y_test, pred)
                r2 = r2_score(y_test, pred)
                direction_acc = (np.sign(pred) == np.sign(y_test)).mean()
                logging.info(f"✅ CatBoost - MSE: {mse:.6f}, R²: {r2:.4f}, Dir Acc: {direction_acc:.2%}")
        
        # === Train XGBoost ===
        if XGBOOST_AVAILABLE:
            logging.info("\n🔥 Training XGBoost...")
            
            # Load or optimize parameters
            if optimize:
                best_params = self.optimize_hyperparameters(X_train_scaled, y_train, 'xgboost', task_type)
                self.save_optimal_params(symbol, timeframe, 'xgboost', task, best_params)
            else:
                best_params = self.load_optimal_params(symbol, timeframe, 'xgboost', task)
                if best_params is None:
                    best_params = self._get_default_params('xgboost', task_type)
            
            # Ensure GPU params are included
            if GPU_UTILS_AVAILABLE:
                best_params.update(get_xgboost_gpu_params())
            
            # Train model
            if task_type == 'classification':
                xgb_model = xgb.XGBClassifier(**best_params)
            else:
                xgb_model = xgb.XGBRegressor(**best_params)
            
            xgb_model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False
            )
            models['xgboost'] = xgb_model
            
            # Evaluate
            pred = xgb_model.predict(X_test_scaled)
            if task_type == 'classification':
                acc = accuracy_score(y_test, pred)
                logging.info(f"✅ XGBoost - Accuracy: {acc:.2%}")
            else:
                mse = mean_squared_error(y_test, pred)
                r2 = r2_score(y_test, pred)
                direction_acc = (np.sign(pred) == np.sign(y_test)).mean()
                logging.info(f"✅ XGBoost - MSE: {mse:.6f}, R²: {r2:.4f}, Dir Acc: {direction_acc:.2%}")
        
        # Save all models and components
        for model_name, model in models.items():
            model_path = f"ml_models/{safe_symbol}_{timeframe}_{task}_{model_name}.joblib"
            joblib.dump(model, model_path)
            logging.info(f"💾 Saved: {model_path}")
        
        # Save scaler, selector, and feature names
        joblib.dump(scaler, f"ml_models/{safe_symbol}_{timeframe}_{task}_scaler.joblib")
        joblib.dump(selector, f"ml_models/{safe_symbol}_{timeframe}_{task}_selector.joblib")
        joblib.dump(selected_names, f"ml_models/{safe_symbol}_{timeframe}_{task}_features.joblib")
        
        logging.info(f"\n✅ {task.capitalize()} models trained successfully!")
        return len(models) > 0
    
    def make_ensemble_prediction(self, symbol: str, timeframe: str) -> Dict:
        """Make ensemble predictions using trained models"""
        logging.info(f"\n{'='*60}")
        logging.info(f"🔮 Ensemble Prediction: {symbol} {timeframe}")
        logging.info(f"{'='*60}\n")
        
        # Load data
        lookback_map = {'5m': 1, '15m': 1, '1h': 2, '4h': 3, '1d': 6}
        months_back = lookback_map.get(timeframe, 1)
        
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
        
        # Load models and make predictions
        safe_symbol = symbol.replace('/', '_')
        
        # Price predictions
        price_preds = []
        weights = {'catboost': 0.55, 'xgboost': 0.45}
        model_predictions = {}
        
        for model_name in ['catboost', 'xgboost']:
            model_path = f"ml_models/{safe_symbol}_{timeframe}_price_{model_name}.joblib"
            scaler_path = f"ml_models/{safe_symbol}_{timeframe}_price_scaler.joblib"
            selector_path = f"ml_models/{safe_symbol}_{timeframe}_price_selector.joblib"
            features_path = f"ml_models/{safe_symbol}_{timeframe}_price_features.joblib"
            
            if all(os.path.exists(p) for p in [model_path, scaler_path, selector_path, features_path]):
                try:
                    model = joblib.load(model_path)
                    scaler = joblib.load(scaler_path)
                    selector = joblib.load(selector_path)
                    feature_names = joblib.load(features_path)
                    
                    # Prepare features
                    exclude_cols = ['open', 'high', 'low', 'close', 'volume']
                    available_features = [col for col in df_features.columns if col not in exclude_cols]
                    X_latest = df_features[available_features].iloc[-1:].values
                    
                    # Select features and scale
                    X_selected = selector.transform(X_latest)
                    X_scaled = scaler.transform(X_selected)
                    
                    # Predict
                    pred = model.predict(X_scaled)[0]
                    price_preds.append(pred)
                    
                    logging.info(f"📊 {model_name.capitalize()} price: {pred:+.4%} (weight: {weights[model_name]:.0%})")
                    
                    # Store individual prediction
                    model_predictions[model_name] = {
                        'predicted_price': float(df['close'].iloc[-1] * (1 + pred)),
                        'price_change_pct': float(pred),
                        'confidence': float(weights[model_name])
                    }
                except Exception as e:
                    logging.warning(f"⚠️ Error loading {model_name}: {e}")
        
        # Calculate ensemble price prediction
        if price_preds:
            ensemble_price_change = sum(p * weights[m] for p, m in zip (price_preds, ['catboost', 'xgboost']))
            
            # Clip predictions
            prediction_limits = {'5m': 0.02, '15m': 0.03, '1h': 0.05, '4h': 0.10, '1d': 0.15}
            max_change = prediction_limits.get(timeframe, 0.05)
            ensemble_price_change = np.clip(ensemble_price_change, -max_change, max_change)
            
            predictions['price_change_pct'] = float(ensemble_price_change * 100)
            predictions['predicted_price'] = float(df['close'].iloc[-1] * (1 + ensemble_price_change))
            
            logging.info(f"\n💡 Ensemble Price Change: {ensemble_price_change:+.4%}")
            logging.info(f"💰 Predicted Price: ${predictions['predicted_price']:.2f}")
        
        # Direction predictions
        direction_votes = {'UP': 0, 'DOWN': 0}
        
        for model_name in ['catboost', 'xgboost']:
            model_path = f"ml_models/{safe_symbol}_{timeframe}_direction_{model_name}.joblib"
            scaler_path = f"ml_models/{safe_symbol}_{timeframe}_direction_scaler.joblib"
            selector_path = f"ml_models/{safe_symbol}_{timeframe}_direction_selector.joblib"
            
            if all(os.path.exists(p) for p in [model_path, scaler_path, selector_path]):
                try:
                    model = joblib.load(model_path)
                    scaler = joblib.load(scaler_path)
                    selector = joblib.load(selector_path)
                    
                    exclude_cols = ['open', 'high', 'low', 'close', 'volume']
                    available_features = [col for col in df_features.columns if col not in exclude_cols]
                    X_latest = df_features[available_features].iloc[-1:].values
                    
                    X_selected = selector.transform(X_latest)
                    X_scaled = scaler.transform(X_selected)
                    
                    direction_pred = model.predict(X_scaled)[0]
                    direction_votes['UP' if direction_pred == 1 else 'DOWN'] += 1
                except Exception as e:
                    logging.warning(f"⚠️ Error loading {model_name} direction model: {e}")
        
        if sum(direction_votes.values()) > 0:
            predictions['direction'] = max(direction_votes, key=direction_votes.get)
            predictions['direction_confidence'] = direction_votes[predictions['direction']] / sum(direction_votes.values())
            predictions['confidence'] = float(predictions['direction_confidence'])
            
            logging.info(f"🎯 Direction: {predictions['direction']} (confidence: {predictions['direction_confidence']:.2%})")
        
        predictions['model_predictions'] = model_predictions
        
        logging.info(f"\n✅ Ensemble prediction complete!\n")
        return predictions


def main():
    """Main execution"""
    print("\n" + "="*70)
    print("🧠 OPTIMIZED ML TRAINING SYSTEM V2.0")
    print("="*70)
    print("\nEnhancements:")
    print("  • 2 models only: CatBoost + XGBoost (50% faster)")
    print("  • Feature selection: Top 50 features")
    print("  • Hyperparameter optimization: Optuna")
    print("  • 50% reduction in storage and training time")
    print("\n" + "="*70 + "\n")
    
    # Initialize system
    ml_system = OptimizedMLSystemV2()
    
    # Symbols and timeframes
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'DOT/USDT']
    timeframes = ['5m', '15m', '1h', '4h', '1d']
    
    print("\n📋 Training Plan:")
    print(f"   Symbols: {', '.join(symbols)}")
    print(f"   Timeframes: {', '.join(timeframes)}")
    print(f"   Models per symbol/timeframe: 4 (2 price + 2 direction)")
    print(f"   Total models: {len(symbols) * len(timeframes) * 4} models")
    print(f"   (50% reduction from previous 8 models per combo)\n")
    
    # Ask about optimization
    optimize = input("Run hyperparameter optimization? (y/N): ").strip().lower() == 'y'
    if optimize:
        print("⏰ Warning: Optimization will take 2-3 hours total")
    else:
        print("⚡ Using default parameters (fast training)")
    
    input("\nPress ENTER to start training... ")
    
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
            
            success = ml_system.train_ensemble(symbol, timeframe, optimize=optimize)
            
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
    print(f"📁 Configs saved in: ml_configs/")
    print(f"📝 Logs saved in: {ml_system.log_file}")
    print("\n💡 Next step: Run predictions or integrate with trading agent")
    print("\n")


if __name__ == "__main__":
    main()