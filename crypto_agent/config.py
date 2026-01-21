"""
Configuration for Crypto Trading Agent
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

# Model weights by timeframe
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

# Signal thresholds for price change predictions
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

# Fibonacci and leverage configuration

# Fibonacci levels for technical analysis
FIBONACCI_LEVELS = {
    'retracement': [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0],
    'extension': [1.236, 1.382, 1.618, 2.0, 2.618]
}

# Take profit distribution (must sum to 100%)
TP_DISTRIBUTION = {
    'tp1': 30,  # Close 30% at TP1 (Fib 0.382)
    'tp2': 40,  # Close 40% at TP2 (Fib 0.618)
    'tp3': 30   # Close 30% at TP3 (Fib 1.0)
}

# Leverage settings (1x to 20x)
LEVERAGE_CONFIG = {
    'min_leverage': 1,
    'max_leverage': 20,
    
    # Base leverage by confidence level
    'confidence_tiers': {
        'very_high': (8, 15),   # confidence >= 75%: suggest 8-15x
        'high': (5, 10),         # confidence 60-75%: suggest 5-10x
        'moderate': (3, 5),      # confidence 50-60%: suggest 3-5x
        'low': (1, 3)            # confidence < 50%: suggest 1-3x
    },
    
    # Quality score multipliers (adjust leverage based on quality)
    'quality_multipliers': {
        'excellent': 1.2,    # quality >= 80: boost leverage by 20%
        'good': 1.0,         # quality >= 70: normal leverage
        'fair': 0.8,         # quality >= 60: reduce leverage by 20%
        'poor': 0.5          # quality < 60: reduce leverage by 50%
    },
    
    # Safety caps
    'max_recommended': 10,  # Never suggest more than 10x (safety)
    'conservative_cap': 5   # For risk-averse traders
}

# Risk management settings
RISK_MANAGEMENT = {
    # Default stop loss percentages by timeframe
    'default_stop_loss_pct': {
        '5m': 0.015,   # 1.5%
        '15m': 0.020,  # 2.0%
        '1h': 0.025,   # 2.5%
        '4h': 0.035,   # 3.5%
        '1d': 0.050    # 5.0%
    },
    
    # Maximum position size
    'max_position_size_pct': 6.0,
    
    # Minimum acceptable risk/reward ratio
    'min_risk_reward_ratio': 1.5,
    
    # Fibonacci-based TP levels (as ratios to use for calculation)
    # These represent how far beyond the stop loss distance to set each TP
    'fibonacci_tp_multipliers': {
        'tp1': 0.382,   # 38.2% Fibonacci retracement (~1.5R)
        'tp2': 0.618,   # 61.8% Golden ratio (~2.5R)
        'tp3': 1.0      # 100% Fibonacci extension (~4R)
    }
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
