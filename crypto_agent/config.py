"""
Configuration for Crypto Trading Agent
Centralized settings for easy modification
"""

# Database configuration
DB_PATH = 'data/ml_crypto_data.db'

# Agent behavior settings
AGENT_CONFIG = {
    'name': 'CryptoTradingAdvisor',
    'version': '1.0.0',
    'max_iterations': 10,  # Max conversation turns
    'temperature': 0.7,    # LLM creativity (0-1)
}

# Model weights by timeframe (for Smart Consensus Analyzer)
MODEL_WEIGHTS = {
    '5m': {
        'lightgbm': 0.50,
        'xgboost': 0.30,
        'catboost': 0.20,
        'gru': 0.00
    },
    '15m': {
        'lightgbm': 0.50,
        'xgboost': 0.30,
        'catboost': 0.20,
        'gru': 0.00
    },
    '1h': {
        'lightgbm': 0.50,
        'xgboost': 0.30,
        'catboost': 0.20,
        'gru': 0.00
    },
    '4h': {
        'lightgbm': 0.45,
        'xgboost': 0.30,
        'catboost': 0.15,
        'gru': 0.10  # GRU only for 4h
    },
    '1d': {
        'lightgbm': 0.50,
        'xgboost': 0.30,
        'catboost': 0.20,
        'gru': 0.00
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

# Prediction clipping limits (from Priority 1 fixes)
PREDICTION_LIMITS = {
    '5m': 0.02,   # 2% max
    '15m': 0.03,  # 3% max
    '1h': 0.05,   # 5% max
    '4h': 0.10,   # 10% max
    '1d': 0.15    # 15% max
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'logs/crypto_agent.log'
}