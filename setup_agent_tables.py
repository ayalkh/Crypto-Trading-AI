"""
Database Setup Script for Crypto Trading Agent
Creates all necessary tables for full agent functionality
"""

import sqlite3
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_agent_database(db_path: str = "data/ml_crypto_data.db"):
    """
    Create all tables needed for full agent functionality
    
    Tables created:
    1. model_performance - Track model accuracy and win rates
    2. technical_indicators - Store RSI, MACD, BB, etc.
    3. agent_recommendations - Track agent's past recommendations
    4. prediction_outcomes - Track prediction vs actual results
    """
    
    logger.info("=" * 70)
    logger.info("🔧 Setting up Crypto Trading Agent Database")
    logger.info("=" * 70)
    
    # Ensure database directory exists
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    logger.info(f"📂 Connected to database: {db_path}")
    
    # ========================================================================
    # TABLE 1: Model Performance Tracking
    # ========================================================================
    logger.info("\n📊 Creating model_performance table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_performance (
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            model_type TEXT NOT NULL,
            accuracy REAL,
            precision_score REAL,
            recall REAL,
            f1_score REAL,
            win_rate REAL,
            avg_return REAL,
            sharpe_ratio REAL,
            max_drawdown REAL,
            total_trades INTEGER,
            winning_trades INTEGER,
            losing_trades INTEGER,
            last_evaluated_at TEXT,
            training_samples INTEGER,
            test_samples INTEGER,
            evaluation_period_days INTEGER DEFAULT 30,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, timeframe, model_type)
        )
    """)
    
    # Add index for faster lookups
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_model_perf_lookup 
        ON model_performance(symbol, timeframe, last_evaluated_at)
    """)
    
    logger.info("   ✅ model_performance table created")
    
    # ========================================================================
    # TABLE 2: Technical Indicators
    # ========================================================================
    logger.info("\n📈 Creating technical_indicators table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS technical_indicators (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            
            -- Trend Indicators
            sma_20 REAL,
            sma_50 REAL,
            sma_200 REAL,
            ema_12 REAL,
            ema_26 REAL,
            
            -- Momentum Indicators
            rsi_14 REAL,
            stoch_k REAL,
            stoch_d REAL,
            
            -- MACD
            macd_line REAL,
            macd_signal REAL,
            macd_histogram REAL,
            
            -- Bollinger Bands
            bb_upper REAL,
            bb_middle REAL,
            bb_lower REAL,
            bb_width REAL,
            
            -- Volatility
            atr_14 REAL,
            atr_percent REAL,
            
            -- Volume
            volume_sma_20 REAL,
            volume_ratio REAL,
            
            -- Trend Strength
            adx REAL,
            adx_di_plus REAL,
            adx_di_minus REAL,
            
            -- Ichimoku
            ichimoku_conversion REAL,
            ichimoku_base REAL,
            ichimoku_span_a REAL,
            ichimoku_span_b REAL,
            
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            
            UNIQUE(symbol, timeframe, timestamp)
        )
    """)
    
    # Add indexes for faster queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_indicators_lookup 
        ON technical_indicators(symbol, timeframe, timestamp DESC)
    """)
    
    logger.info("   ✅ technical_indicators table created")
    
    # ========================================================================
    # TABLE 3: Agent Recommendations
    # ========================================================================
    logger.info("\n🤖 Creating agent_recommendations table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS agent_recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            
            -- Recommendation Details
            recommendation TEXT NOT NULL,  -- BUY, SELL, HOLD
            confidence REAL NOT NULL,
            quality_score REAL NOT NULL,
            quality_grade TEXT,
            
            -- Price Information
            price_at_recommendation REAL NOT NULL,
            predicted_price REAL,
            predicted_change_pct REAL,
            
            -- Risk Management
            should_trade BOOLEAN NOT NULL,
            position_size_min REAL,
            position_size_max REAL,
            stop_loss REAL,
            take_profit REAL,
            
            -- Analysis Components
            consensus_direction TEXT,
            consensus_confidence REAL,
            model_agreement_pct REAL,
            timeframe_alignment_pct REAL,
            
            -- Market Context
            market_regime TEXT,
            market_regime_confidence REAL,
            risk_level TEXT,
            volatility_percentile REAL,
            
            -- Reasoning
            key_factors TEXT,  -- JSON array of key decision factors
            risk_factors TEXT,  -- JSON array of risk concerns
            strengths TEXT,    -- JSON array of strengths
            
            -- Timestamps
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            target_timestamp TEXT,  -- When prediction is for
            
            -- Outcome Tracking (filled in later)
            actual_price REAL,
            actual_direction TEXT,
            outcome TEXT,  -- WIN, LOSS, NEUTRAL
            profit_loss_pct REAL,
            evaluated_at TEXT
        )
    """)
    
    # Add indexes
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_recommendations_lookup 
        ON agent_recommendations(symbol, timeframe, created_at DESC)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_recommendations_outcome 
        ON agent_recommendations(should_trade, outcome, created_at)
    """)
    
    logger.info("   ✅ agent_recommendations table created")
    
    # ========================================================================
    # TABLE 4: Prediction Outcomes (for detailed tracking)
    # ========================================================================
    logger.info("\n🎯 Creating prediction_outcomes table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recommendation_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            model_type TEXT,
            
            -- Prediction
            predicted_direction TEXT NOT NULL,
            predicted_price REAL,
            predicted_change_pct REAL,
            prediction_confidence REAL,
            
            -- Actual Outcome
            actual_direction TEXT,
            actual_price REAL,
            actual_change_pct REAL,
            
            -- Result
            direction_correct BOOLEAN,
            price_error_pct REAL,
            outcome TEXT,  -- WIN, LOSS, NEUTRAL
            
            -- Timestamps
            predicted_at TEXT NOT NULL,
            target_timestamp TEXT NOT NULL,
            evaluated_at TEXT,
            
            -- Performance Metrics
            hours_to_target INTEGER,
            actual_hours_held INTEGER,
            
            FOREIGN KEY (recommendation_id) REFERENCES agent_recommendations(id)
        )
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_outcomes_lookup 
        ON prediction_outcomes(symbol, timeframe, predicted_at DESC)
    """)
    
    logger.info("   ✅ prediction_outcomes table created")
    
    # ========================================================================
    # TABLE 5: Agent Performance Metrics (aggregate stats)
    # ========================================================================
    logger.info("\n📊 Creating agent_performance_metrics table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS agent_performance_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            period_start TEXT NOT NULL,
            period_end TEXT NOT NULL,
            
            -- Overall Stats
            total_recommendations INTEGER,
            total_trades INTEGER,
            
            -- Performance
            win_rate REAL,
            avg_confidence REAL,
            avg_quality_score REAL,
            
            -- By Recommendation Type
            buy_recommendations INTEGER,
            sell_recommendations INTEGER,
            hold_recommendations INTEGER,
            buy_win_rate REAL,
            sell_win_rate REAL,
            
            -- By Symbol
            best_performing_symbol TEXT,
            worst_performing_symbol TEXT,
            
            -- By Timeframe
            best_performing_timeframe TEXT,
            worst_performing_timeframe TEXT,
            
            -- Quality Thresholds
            high_quality_trades INTEGER,  -- quality >= 70
            high_quality_win_rate REAL,
            
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            
            UNIQUE(period_start, period_end)
        )
    """)
    
    logger.info("   ✅ agent_performance_metrics table created")
    
    # ========================================================================
    # Commit changes
    # ========================================================================
    conn.commit()
    
    # ========================================================================
    # Verify tables were created
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("🔍 Verifying database setup...")
    logger.info("=" * 70)
    
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' 
        ORDER BY name
    """)
    
    tables = [row[0] for row in cursor.fetchall()]
    
    required_tables = [
        'model_performance',
        'technical_indicators', 
        'agent_recommendations',
        'prediction_outcomes',
        'agent_performance_metrics'
    ]
    
    for table in required_tables:
        if table in tables:
            logger.info(f"   ✅ {table}")
        else:
            logger.error(f"   ❌ {table} - MISSING!")
    
    # ========================================================================
    # Get table counts
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("📊 Current table statistics:")
    logger.info("=" * 70)
    
    for table in tables:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            logger.info(f"   {table}: {count:,} records")
        except Exception as e:
            logger.info(f"   {table}: Error - {e}")
    
    # Close connection
    conn.close()
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ Database setup complete!")
    logger.info("=" * 70)
    logger.info("\nYour agent now has full functionality:")
    logger.info("  ✅ Model performance tracking")
    logger.info("  ✅ Technical indicators support")
    logger.info("  ✅ Recommendation history")
    logger.info("  ✅ Prediction outcome tracking")
    logger.info("  ✅ Performance metrics")
    logger.info("\nRun test_agent.py again to verify!")
    logger.info("=" * 70 + "\n")

if __name__ == "__main__":
    setup_agent_database()