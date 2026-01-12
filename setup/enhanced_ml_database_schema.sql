-- Enhanced ML-Ready Database Schema for Crypto Trading
-- Optimized for LSTM, GRU, and advanced ML models
-- Supports long-term data storage and feature engineering

-- ============================================================================
-- CORE PRICE DATA TABLE (Enhanced)
-- ============================================================================
CREATE TABLE IF NOT EXISTS price_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    
    -- OHLCV (Open, High, Low, Close, Volume)
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(20, 8) NOT NULL,
    
    -- Additional Price Metrics
    vwap DECIMAL(20, 8),              -- Volume-Weighted Average Price
    number_of_trades INTEGER,          -- Number of trades in this candle
    taker_buy_volume DECIMAL(20, 8),  -- Aggressive buy volume
    taker_sell_volume DECIMAL(20, 8), -- Aggressive sell volume
    
    -- Calculated Features (stored for efficiency)
    price_change_pct DECIMAL(10, 6),  -- Percentage change
    high_low_range DECIMAL(20, 8),    -- High - Low
    body_size DECIMAL(20, 8),         -- abs(Close - Open)
    upper_wick DECIMAL(20, 8),        -- High - max(Open, Close)
    lower_wick DECIMAL(20, 8),        -- min(Open, Close) - Low
    
    -- Metadata
    data_quality VARCHAR(20),         -- 'COMPLETE', 'PARTIAL', 'ESTIMATED'
    source VARCHAR(50),               -- Data source (e.g., 'binance', 'coinbase')
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    UNIQUE(symbol, timeframe, timestamp)
);

-- Indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_symbol_timeframe_timestamp 
    ON price_data(symbol, timeframe, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_timestamp 
    ON price_data(timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_symbol_timestamp 
    ON price_data(symbol, timestamp DESC);


-- ============================================================================
-- TECHNICAL INDICATORS TABLE (Pre-calculated for ML)
-- ============================================================================
CREATE TABLE IF NOT EXISTS technical_indicators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    
    -- Moving Averages
    sma_5 DECIMAL(20, 8),
    sma_10 DECIMAL(20, 8),
    sma_20 DECIMAL(20, 8),
    sma_50 DECIMAL(20, 8),
    sma_100 DECIMAL(20, 8),
    sma_200 DECIMAL(20, 8),
    
    ema_5 DECIMAL(20, 8),
    ema_10 DECIMAL(20, 8),
    ema_20 DECIMAL(20, 8),
    ema_50 DECIMAL(20, 8),
    ema_100 DECIMAL(20, 8),
    ema_200 DECIMAL(20, 8),
    
    -- Momentum Indicators
    rsi_14 DECIMAL(10, 4),
    rsi_7 DECIMAL(10, 4),
    rsi_21 DECIMAL(10, 4),
    
    macd_line DECIMAL(20, 8),
    macd_signal DECIMAL(20, 8),
    macd_histogram DECIMAL(20, 8),
    
    stochastic_k DECIMAL(10, 4),
    stochastic_d DECIMAL(10, 4),
    
    -- Volatility Indicators
    bb_upper DECIMAL(20, 8),
    bb_middle DECIMAL(20, 8),
    bb_lower DECIMAL(20, 8),
    bb_width DECIMAL(20, 8),
    bb_position DECIMAL(10, 4),
    
    atr_14 DECIMAL(20, 8),
    atr_7 DECIMAL(20, 8),
    
    -- Volume Indicators
    obv DECIMAL(30, 2),                -- On-Balance Volume
    cmf DECIMAL(10, 6),                -- Chaikin Money Flow
    mfi DECIMAL(10, 4),                -- Money Flow Index
    volume_sma_20 DECIMAL(20, 8),
    volume_ratio DECIMAL(10, 4),
    
    -- Trend Indicators
    adx DECIMAL(10, 4),                -- Average Directional Index
    cci DECIMAL(10, 4),                -- Commodity Channel Index
    
    -- Metadata
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    UNIQUE(symbol, timeframe, timestamp),
    FOREIGN KEY(symbol, timeframe, timestamp) 
        REFERENCES price_data(symbol, timeframe, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_tech_symbol_timeframe_timestamp 
    ON technical_indicators(symbol, timeframe, timestamp DESC);


-- ============================================================================
-- MARKET FEATURES TABLE (ML-specific features)
-- ============================================================================
CREATE TABLE IF NOT EXISTS market_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    
    -- Lag Features (for time series models)
    price_lag_1 DECIMAL(20, 8),
    price_lag_2 DECIMAL(20, 8),
    price_lag_3 DECIMAL(20, 8),
    price_lag_5 DECIMAL(20, 8),
    price_lag_10 DECIMAL(20, 8),
    price_lag_20 DECIMAL(20, 8),
    
    volume_lag_1 DECIMAL(20, 8),
    volume_lag_2 DECIMAL(20, 8),
    volume_lag_3 DECIMAL(20, 8),
    volume_lag_5 DECIMAL(20, 8),
    
    -- Rolling Statistics
    returns_mean_5 DECIMAL(10, 6),
    returns_mean_10 DECIMAL(10, 6),
    returns_mean_20 DECIMAL(10, 6),
    
    returns_std_5 DECIMAL(10, 6),
    returns_std_10 DECIMAL(10, 6),
    returns_std_20 DECIMAL(10, 6),
    
    volume_mean_5 DECIMAL(20, 8),
    volume_mean_20 DECIMAL(20, 8),
    volume_std_5 DECIMAL(20, 8),
    volume_std_20 DECIMAL(20, 8),
    
    -- Momentum Features
    momentum_5 DECIMAL(10, 6),
    momentum_10 DECIMAL(10, 6),
    momentum_20 DECIMAL(10, 6),
    
    roc_5 DECIMAL(10, 6),
    roc_10 DECIMAL(10, 6),
    roc_20 DECIMAL(10, 6),
    
    -- Volatility Features
    volatility_5 DECIMAL(10, 6),
    volatility_10 DECIMAL(10, 6),
    volatility_20 DECIMAL(10, 6),
    volatility_ratio_5_20 DECIMAL(10, 4),
    
    -- Price Pattern Features
    higher_high BOOLEAN,
    higher_low BOOLEAN,
    lower_high BOOLEAN,
    lower_low BOOLEAN,
    
    -- Time-based Features
    hour_of_day INTEGER,
    day_of_week INTEGER,
    day_of_month INTEGER,
    is_weekend BOOLEAN,
    is_month_end BOOLEAN,
    
    -- Market Regime
    trend_strength DECIMAL(10, 4),     -- -1 to 1 (bearish to bullish)
    volatility_regime VARCHAR(20),     -- 'LOW', 'MEDIUM', 'HIGH'
    
    -- Metadata
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(symbol, timeframe, timestamp),
    FOREIGN KEY(symbol, timeframe, timestamp) 
        REFERENCES price_data(symbol, timeframe, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_features_symbol_timeframe_timestamp 
    ON market_features(symbol, timeframe, timestamp DESC);


-- ============================================================================
-- MULTI-TIMEFRAME CORRELATION TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS timeframe_correlations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol VARCHAR(20) NOT NULL,
    primary_timeframe VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    
    -- Correlation with other timeframes
    correlation_5m DECIMAL(10, 6),
    correlation_15m DECIMAL(10, 6),
    correlation_1h DECIMAL(10, 6),
    correlation_4h DECIMAL(10, 6),
    correlation_1d DECIMAL(10, 6),
    
    -- Trend alignment (percentage of timeframes in same direction)
    trend_alignment_score DECIMAL(10, 4),
    
    -- Dominant timeframe trend
    dominant_trend VARCHAR(20),        -- 'BULLISH', 'BEARISH', 'NEUTRAL'
    
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(symbol, primary_timeframe, timestamp)
);


-- ============================================================================
-- ML PREDICTIONS TABLE (Store model predictions)
-- ============================================================================
CREATE TABLE IF NOT EXISTS ml_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,      -- When prediction was made
    target_timestamp TIMESTAMP NOT NULL, -- What time is being predicted
    
    -- Model Information
    model_type VARCHAR(50) NOT NULL,   -- 'LSTM', 'GRU', 'RandomForest', etc.
    model_version VARCHAR(50),
    
    -- Predictions
    predicted_price DECIMAL(20, 8),
    predicted_direction VARCHAR(10),   -- 'UP', 'DOWN', 'NEUTRAL'
    direction_probability DECIMAL(10, 6),
    
    predicted_change_pct DECIMAL(10, 6),
    
    confidence_score DECIMAL(10, 6),   -- Model confidence
    
    -- Prediction Range (for uncertainty)
    prediction_low DECIMAL(20, 8),
    prediction_high DECIMAL(20, 8),
    
    -- Actual Outcome (filled in later)
    actual_price DECIMAL(20, 8),
    actual_direction VARCHAR(10),
    prediction_error DECIMAL(20, 8),
    is_correct BOOLEAN,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY(symbol, timeframe, timestamp) 
        REFERENCES price_data(symbol, timeframe, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_predictions_symbol_timeframe 
    ON ml_predictions(symbol, timeframe, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_predictions_model 
    ON ml_predictions(model_type, symbol, timeframe);


-- ============================================================================
-- MODEL PERFORMANCE TABLE (Track model accuracy)
-- ============================================================================
CREATE TABLE IF NOT EXISTS model_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_type VARCHAR(50) NOT NULL,
    model_version VARCHAR(50),
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    
    -- Performance Metrics
    accuracy DECIMAL(10, 6),
    precision_score DECIMAL(10, 6),
    recall DECIMAL(10, 6),
    f1_score DECIMAL(10, 6),
    
    mae DECIMAL(20, 8),                -- Mean Absolute Error
    mse DECIMAL(20, 8),                -- Mean Squared Error
    rmse DECIMAL(20, 8),               -- Root Mean Squared Error
    r2_score DECIMAL(10, 6),
    
    -- Trading Performance
    win_rate DECIMAL(10, 6),
    profit_factor DECIMAL(10, 4),
    sharpe_ratio DECIMAL(10, 4),
    
    -- Data Statistics
    training_samples INTEGER,
    test_samples INTEGER,
    training_period_start TIMESTAMP,
    training_period_end TIMESTAMP,
    
    -- Metadata
    trained_at TIMESTAMP,
    last_evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(model_type, model_version, symbol, timeframe, trained_at)
);


-- ============================================================================
-- DATA COLLECTION STATUS (Enhanced tracking)
-- ============================================================================
CREATE TABLE IF NOT EXISTS collection_status (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    
    -- Status Information
    last_update TIMESTAMP NOT NULL,
    records_count INTEGER NOT NULL,
    status VARCHAR(50) NOT NULL,
    last_timestamp TIMESTAMP,
    
    -- Data Range
    earliest_timestamp TIMESTAMP,
    latest_timestamp TIMESTAMP,
    expected_records INTEGER,         -- Expected number based on timeframe
    completeness_pct DECIMAL(10, 2), -- % of expected records present
    
    -- Data Quality
    missing_records INTEGER,
    duplicate_records INTEGER,
    error_count INTEGER,
    last_error TEXT,
    
    -- Collection Metadata
    collection_method VARCHAR(50),    -- 'API', 'BACKFILL', 'REALTIME'
    next_scheduled_update TIMESTAMP,
    
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(symbol, timeframe)
);


-- ============================================================================
-- MARKET EVENTS TABLE (for context)
-- ============================================================================
CREATE TABLE IF NOT EXISTS market_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP NOT NULL,
    event_type VARCHAR(50) NOT NULL,  -- 'HALVING', 'FORK', 'LISTING', 'NEWS', etc.
    symbol VARCHAR(20),                -- NULL for market-wide events
    
    title TEXT NOT NULL,
    description TEXT,
    impact_level VARCHAR(20),         -- 'HIGH', 'MEDIUM', 'LOW'
    
    source VARCHAR(100),
    source_url TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_events_timestamp 
    ON market_events(timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_events_symbol 
    ON market_events(symbol, timestamp DESC);


-- ============================================================================
-- VIEWS FOR CONVENIENT QUERYING
-- ============================================================================

-- Complete market data view (price + indicators + features)
CREATE VIEW IF NOT EXISTS complete_market_data AS
SELECT 
    p.*,
    ti.rsi_14, ti.macd_line, ti.macd_signal, ti.bb_upper, ti.bb_lower,
    ti.atr_14, ti.obv, ti.adx,
    mf.returns_mean_20, mf.volatility_20, mf.trend_strength,
    mf.hour_of_day, mf.day_of_week, mf.is_weekend
FROM price_data p
LEFT JOIN technical_indicators ti 
    ON p.symbol = ti.symbol 
    AND p.timeframe = ti.timeframe 
    AND p.timestamp = ti.timestamp
LEFT JOIN market_features mf 
    ON p.symbol = mf.symbol 
    AND p.timeframe = mf.timeframe 
    AND p.timestamp = mf.timestamp;


-- ML-ready data view (for model training)
CREATE VIEW IF NOT EXISTS ml_ready_data AS
SELECT 
    p.symbol,
    p.timeframe,
    p.timestamp,
    p.open, p.high, p.low, p.close, p.volume,
    p.price_change_pct,
    
    -- Technical Indicators
    ti.rsi_14, ti.macd_histogram, ti.bb_position,
    ti.atr_14, ti.adx,
    
    -- Features
    mf.price_lag_1, mf.price_lag_5, mf.price_lag_10,
    mf.returns_mean_20, mf.returns_std_20,
    mf.volatility_20, mf.momentum_20,
    mf.hour_of_day, mf.day_of_week, mf.is_weekend,
    mf.trend_strength, mf.volatility_regime,
    
    -- Multi-timeframe
    tc.trend_alignment_score,
    tc.dominant_trend
    
FROM price_data p
LEFT JOIN technical_indicators ti USING(symbol, timeframe, timestamp)
LEFT JOIN market_features mf USING(symbol, timeframe, timestamp)
LEFT JOIN timeframe_correlations tc USING(symbol, timeframe, timestamp)
WHERE p.data_quality = 'COMPLETE';