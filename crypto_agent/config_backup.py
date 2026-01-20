"""
Configuration for Crypto Trading Agent - FIXED VERSION
Matches actual 2-model setup (catboost + xgboost)
"""

# Database configuration
DB_PATH = 'data/ml_crypto_data.db'

# Agent behavior settings
AGENT_CONFIG = {
    'name': 'CryptoTradingAdvisor',
    'version': '1.0.0',
    'max_iterations': 10,
    'temperature': 0.7,
}

# ============================================================================
# CRITICAL FIX: Model weights by timeframe
# UPDATED: UPPERCASE to match database storage (CATBOOST, XGBOOST)
# ============================================================================
MODEL_WEIGHTS = {
    '5m': {
        'CATBOOST': 0.55,  # CatBoost slightly favored
        'XGBOOST': 0.45
    },
    '15m': {
        'CATBOOST': 0.55,
        'XGBOOST': 0.45
    },
    '1h': {
        'CATBOOST': 0.52,
        'XGBOOST': 0.48
    },
    '4h': {
        'CATBOOST': 0.58,  # CatBoost better on longer timeframes
        'XGBOOST': 0.42
    },
    '1d': {
        'CATBOOST': 0.60,
        'XGBOOST': 0.40
    }
}

# Trade quality scoring weights (total = 100)
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

# ============================================================================
# CRITICAL FIX: Signal thresholds for price change (in DECIMAL form)
# OLD: Used quality scores (80, 65, etc.) - WRONG!
# NEW: Use actual price change thresholds calibrated for crypto
# ============================================================================
SIGNAL_THRESHOLDS = {
    '5m': {
        'strong_buy': 0.0004,   # 0.04% for 5m
        'buy': 0.0002,          # 0.02%
        'sell': -0.0002,
        'strong_sell': -0.0004
    },
    '15m': {
        'strong_buy': 0.0005,   # 0.05%
        'buy': 0.00025,         # 0.025%
        'sell': -0.00025,
        'strong_sell': -0.0005
    },
    '1h': {
        'strong_buy': 0.0008,   # 0.08%
        'buy': 0.0003,          # 0.03%
        'sell': -0.0003,
        'strong_sell': -0.0008
    },
    '4h': {
        'strong_buy': 0.0015,   # 0.15%
        'buy': 0.0006,          # 0.06%
        'sell': -0.0006,
        'strong_sell': -0.0015
    },
    '1d': {
        'strong_buy': 0.0030,   # 0.30%
        'buy': 0.0012,          # 0.12%
        'sell': -0.0012,
        'strong_sell': -0.0030
    }
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

# Prediction clipping limits (max allowed price change per timeframe)
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

# ============================================================================
# KEY CHANGES MADE:
# ============================================================================
# 1. MODEL_WEIGHTS: Removed lightgbm and gru (you don't have these models)
#    - Now only catboost + xgboost with proper weight distribution
#
# 2. SIGNAL_THRESHOLDS: Changed from quality scores to price change decimals
#    - OLD: {'strong_buy': 80, 'buy': 65, ...} ← These were quality scores!
#    - NEW: Timeframe-specific price change thresholds in decimal form
#    - Example: 0.0006 = 0.06% price change
#
# These fixes ensure:
# ✅ Only actual models get weighted (no missing model issue)
# ✅ Thresholds match your model predictions (0.01-0.2% range)
# ✅ Different sensitivity per timeframe (5m more sensitive than 4h)
# ============================================================================