"""
Configuration for Crypto Trading Agent v2.0
Updated for 2-model ensemble (CatBoost + XGBoost)
"""

# Database configuration
DB_PATH = 'data/ml_crypto_data.db'

# Agent behavior settings
AGENT_CONFIG = {
    'name': 'CryptoTradingAdvisor',
    'version': '2.0.0',  # Updated version
    'max_iterations': 10,
    'temperature': 0.7,
}

# Model weights by timeframe (UPDATED - 2 models only)
# Based on empirical testing:
# - CatBoost: 51.3% avg direction accuracy (best performer)
# - XGBoost: 50.7% avg direction accuracy (stable second)
MODEL_WEIGHTS = {
    '5m': {
        'catboost': 0.55,  # Primary model
        'xgboost': 0.45    # Secondary model
    },
    '15m': {
        'catboost': 0.55,
        'xgboost': 0.45
    },
    '1h': {
        'catboost': 0.55,
        'xgboost': 0.45
    },
    '4h': {
        'catboost': 0.55,
        'xgboost': 0.45
    },
    '1d': {
        'catboost': 0.55,
        'xgboost': 0.45
    }
}

# Trade quality scoring weights (total = 100%)
QUALITY_WEIGHTS = {
    'model_consensus': 15,      # How many models agree
    'historical_winrate': 15,   # Past performance at this confidence
    'timeframe_alignment': 15,  # Multi-TF agreement
    'model_performance': 10,    # Recent model accuracy
    'technical_confirmation': 15, # TA indicators support
    'signal_strength': 10,      # Signal vs noise
    'data_freshness': 10,       # How recent is data
    'trading_frequency': 5,     # Avoid overtrading
    'btc_correlation': 5        # Independent or following BTC
}

# Signal thresholds
SIGNAL_THRESHOLDS = {
    'strong_buy': 80,
    'buy': 65,
    'hold_upper': 55,
    'hold_lower': 45,
    'sell': 35,
    'strong_sell': 20
}

# Market regime detection
REGIME_CONFIG = {
    'trend_lookback_days': 7,
    'volatility_window': 20,
    'high_volatility_threshold': 0.05,  # 5% daily moves
    'low_volatility_threshold': 0.02    # 2% daily moves
}

# Position sizing recommendations (% of portfolio)
POSITION_SIZING = {
    'quality_90_plus': (5, 7),   # (min%, max%)
    'quality_80_89': (4, 6),
    'quality_70_79': (3, 5),
    'quality_60_69': (2, 4),
    'quality_below_60': (1, 2)
}

# Available symbols and timeframes
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'DOT/USDT']
TIMEFRAMES = ['5m', '15m', '1h', '4h', '1d']

# Timeframe weights for multi-timeframe analysis
TIMEFRAME_WEIGHTS = {
    '5m': 10,
    '15m': 15,
    '1h': 30,
    '4h': 30,
    '1d': 15
}

# Prediction clipping limits
PREDICTION_LIMITS = {
    '5m': 0.02,   # 2% max
    '15m': 0.03,  # 3% max
    '1h': 0.05,   # 5% max
    '4h': 0.10,   # 10% max
    '1d': 0.15    # 15% max
}

# Feature Selection Configuration (NEW)
FEATURE_SELECTION = {
    'enabled': True,
    'n_features': 50,  # Select top 50 features
    'method': 'mutual_info_classif'  # For classification tasks
}

# Hyperparameter Optimization Configuration (NEW)
OPTUNA_CONFIG = {
    'enabled': False,  # Enable for first-time training, disable for fast retraining
    'n_trials': 10,    # Number of optimization trials
    'timeout': 3600    # Max optimization time per model (seconds)
}

# Model-specific optimal parameters (saved after Optuna tuning)
# These will be loaded from ml_configs/ directory if available

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'logs/crypto_agent.log'
}


PRIORITY_FEATURES = [
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