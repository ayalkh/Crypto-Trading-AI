"""
Database interface for Crypto Trading Agent
Handles all database queries and data retrieval
"""
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from .config import DB_PATH

logger = logging.getLogger(__name__)


class AgentDatabase:
    """Database interface for the trading agent"""
    
    def __init__(self, db_path: str = DB_PATH):
        """Initialize database connection"""
        self.db_path = db_path
        self._create_ml_predictions_table()  # FIX: Create table
        logger.info(f"📊 Database initialized: {db_path}")
    
    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    

    
    def _create_ml_predictions_table(self):
        """Create ml_predictions table if it doesn't exist"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ml_predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        target_timestamp DATETIME,
                        model_type TEXT NOT NULL,
                        predicted_price REAL,
                        predicted_direction TEXT,
                        direction_probability REAL,
                        predicted_change_pct REAL,
                        confidence_score REAL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_predictions_symbol_timeframe 
                    ON ml_predictions(symbol, timeframe, timestamp DESC)
                """)
                
                conn.commit()
        except Exception as e:
            logger.error(f"Error creating ml_predictions table: {e}")

    def get_latest_price(self, symbol: str, timeframe: str = '1h') -> Optional[float]:
        """Get latest price for a symbol"""
        try:
            with self.get_connection() as conn:
                query = """
                SELECT close 
                FROM price_data 
                WHERE symbol = ? AND timeframe = ?
                ORDER BY timestamp DESC 
                LIMIT 1
                """
                result = pd.read_sql_query(query, conn, params=(symbol, timeframe))
                
                if not result.empty:
                    return float(result['close'].iloc[0])
                return None
                
        except Exception as e:
            logger.error(f"❌ Error getting latest price: {e}")
            return None
    
    def get_ml_predictions(self, symbol: str, timeframe: str, 
                        limit: int = 1) -> pd.DataFrame:
        """
        Get ML predictions for a symbol/timeframe
        
        Returns most recent predictions from all models
        """
        try:
            with self.get_connection() as conn:
                query = """
                SELECT 
                    model_type,
                    predicted_price,
                    predicted_direction,
                    direction_probability,
                    predicted_change_pct,
                    confidence_score,
                    timestamp,
                    target_timestamp
                FROM ml_predictions
                WHERE symbol = ? AND timeframe = ?
                ORDER BY timestamp DESC
                LIMIT 50
                """
                
                df = pd.read_sql_query(
                    query, 
                    conn, 
                    params=(symbol, timeframe)
                )
                
                if df.empty:
                    logger.warning(f"⚠️ No ML predictions found for {symbol} {timeframe}")
                    return pd.DataFrame()
                
                # Get latest prediction from each model type
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
                latest_predictions = df.sort_values('timestamp', ascending=False).groupby('model_type').first().reset_index()
                
                logger.info(f"📊 Found {len(latest_predictions)} model predictions for {symbol} {timeframe}")
                logger.debug(f"   Models: {latest_predictions['model_type'].tolist()}")
                
                return latest_predictions
                
        except Exception as e:
            logger.error(f"❌ Error getting ML predictions: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def get_model_performance(self, symbol: str, timeframe: str,
                            days_back: int = 30) -> pd.DataFrame:
        """
        Get recent model performance metrics
        
        Args:
            symbol: Trading pair
            timeframe: Timeframe
            days_back: How many days of performance to retrieve
            
        Returns:
            DataFrame with model performance metrics
        """
        try:
            with self.get_connection() as conn:
                cutoff_date = datetime.now() - timedelta(days=days_back)
                
                query = """
                SELECT 
                    model_type,
                    accuracy,
                    win_rate,
                    last_evaluated_at,
                    training_samples,
                    test_samples
                FROM model_performance
                WHERE symbol = ? 
                  AND timeframe = ?
                  AND last_evaluated_at >= ?
                ORDER BY last_evaluated_at DESC
                """
                
                df = pd.read_sql_query(
                    query,
                    conn,
                    params=(symbol, timeframe, cutoff_date.strftime('%Y-%m-%d'))
                )
                
                return df
                
        except Exception as e:
            logger.error(f"❌ Error getting model performance: {e}")
            return pd.DataFrame()
    
    def get_historical_signals(self, symbol: str, timeframe: str,
                              days_back: int = 30) -> pd.DataFrame:
        """
        Get historical signals and their outcomes
        
        This would query agent_recommendations table (to be created)
        For now, returns empty DataFrame
        """
        try:
            with self.get_connection() as conn:
                # Check if table exists
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='agent_recommendations'
                """)
                
                if not cursor.fetchone():
                    logger.info("ℹ️ agent_recommendations table doesn't exist yet")
                    return pd.DataFrame()
                
                cutoff_date = datetime.now() - timedelta(days=days_back)
                
                query = """
                SELECT 
                    timestamp,
                    recommendation,
                    confidence,
                    quality_score,
                    entry_price,
                    outcome_4h,
                    return_4h
                FROM agent_recommendations
                WHERE symbol = ? 
                  AND timeframe = ?
                  AND timestamp >= ?
                ORDER BY timestamp DESC
                """
                
                df = pd.read_sql_query(
                    query,
                    conn,
                    params=(symbol, timeframe, cutoff_date.strftime('%Y-%m-%d %H:%M:%S'))
                )
                
                return df
                
        except Exception as e:
            logger.warning(f"⚠️ Could not get historical signals: {e}")
            return pd.DataFrame()
    
    def get_technical_indicators(self, symbol: str, timeframe: str,
                                limit: int = 20) -> pd.DataFrame:
        """
        Get recent technical indicators
        
        Args:
            symbol: Trading pair
            timeframe: Timeframe
            limit: Number of recent candles to retrieve
            
        Returns:
            DataFrame with technical indicators
        """
        try:
            with self.get_connection() as conn:
                query = """
                SELECT 
                    timestamp,
                    rsi_14,
                    macd_line,
                    macd_signal,
                    bb_upper,
                    bb_lower,
                    atr_14,
                    adx
                FROM technical_indicators
                WHERE symbol = ? AND timeframe = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """
                
                df = pd.read_sql_query(
                    query,
                    conn,
                    params=(symbol, timeframe, limit)
                )
                
                if df.empty:
                    logger.warning(f"⚠️ No technical indicators found for {symbol} {timeframe}")
                
                return df
                
        except Exception as e:
            logger.warning(f"⚠️ Technical indicators table may not exist: {e}")
            return pd.DataFrame()
    
    def get_price_history(self, symbol: str, timeframe: str,
                         hours_back: int = 168) -> pd.DataFrame:
        """
        Get price history for a symbol
        
        Args:
            symbol: Trading pair
            timeframe: Timeframe
            hours_back: Hours of history to retrieve (default 7 days)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            with self.get_connection() as conn:
                cutoff_time = datetime.now() - timedelta(hours=hours_back)
                
                query = """
                SELECT 
                    timestamp,
                    open,
                    high,
                    low,
                    close,
                    volume
                FROM price_data
                WHERE symbol = ? 
                  AND timeframe = ?
                  AND timestamp >= ?
                ORDER BY timestamp ASC
                """
                
                df = pd.read_sql_query(
                    query,
                    conn,
                    params=(symbol, timeframe, cutoff_time.strftime('%Y-%m-%d %H:%M:%S'))
                )
                
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                return df
                
        except Exception as e:
            logger.error(f"❌ Error getting price history: {e}")
            return pd.DataFrame()
    
    def get_all_symbols_latest_prices(self) -> Dict[str, float]:
        """Get latest prices for all symbols"""
        try:
            with self.get_connection() as conn:
                query = """
                SELECT DISTINCT 
                    symbol,
                    (SELECT close 
                     FROM price_data p2 
                     WHERE p2.symbol = p1.symbol 
                       AND p2.timeframe = '1h'
                     ORDER BY timestamp DESC 
                     LIMIT 1) as latest_price
                FROM price_data p1
                WHERE timeframe = '1h'
                """
                
                df = pd.read_sql_query(query, conn)
                
                prices = {}
                for _, row in df.iterrows():
                    if row['latest_price'] is not None:
                        prices[row['symbol']] = float(row['latest_price'])
                
                return prices
                
        except Exception as e:
            logger.error(f"❌ Error getting all symbols prices: {e}")
            return {}
    
    def save_agent_recommendation(self, recommendation: Dict) -> bool:
        """
        Save agent recommendation to database
        
        Creates agent_recommendations table if it doesn't exist
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create table if it doesn't exist
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS agent_recommendations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME NOT NULL,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        recommendation TEXT NOT NULL,
                        confidence REAL,
                        quality_score INTEGER,
                        entry_price REAL,
                        model_predictions TEXT,
                        reasoning TEXT,
                        market_regime TEXT,
                        price_1h REAL,
                        price_4h REAL,
                        price_24h REAL,
                        outcome_1h TEXT,
                        return_1h REAL,
                        outcome_4h TEXT,
                        return_4h REAL,
                        outcome_24h TEXT,
                        return_24h REAL,
                        followed BOOLEAN DEFAULT 0,
                        actual_return REAL
                    )
                """)
                
                # Insert recommendation
                cursor.execute("""
                    INSERT INTO agent_recommendations 
                    (timestamp, symbol, timeframe, recommendation, confidence, 
                     quality_score, entry_price, model_predictions, reasoning, market_regime)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    recommendation.get('timestamp', datetime.now()),
                    recommendation['symbol'],
                    recommendation['timeframe'],
                    recommendation['recommendation'],
                    recommendation.get('confidence'),
                    recommendation.get('quality_score'),
                    recommendation.get('entry_price'),
                    str(recommendation.get('model_predictions', {})),
                    recommendation.get('reasoning'),
                    recommendation.get('market_regime')
                ))
                
                conn.commit()
                logger.info(f"💾 Saved recommendation for {recommendation['symbol']} {recommendation['timeframe']}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error saving recommendation: {e}")
            return False
    
    def update_recommendation_outcome(self, recommendation_id: int,
                                     outcome_period: str,
                                     price: float,
                                     outcome: str,
                                     return_pct: float) -> bool:
        """
        Update the outcome of a recommendation after time has passed
        
        Args:
            recommendation_id: ID of the recommendation
            outcome_period: '1h', '4h', or '24h'
            price: Price at outcome time
            outcome: 'WIN', 'LOSS', or 'NEUTRAL'
            return_pct: Return percentage
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute(f"""
                    UPDATE agent_recommendations
                    SET price_{outcome_period} = ?,
                        outcome_{outcome_period} = ?,
                        return_{outcome_period} = ?
                    WHERE id = ?
                """, (price, outcome, return_pct, recommendation_id))
                
                conn.commit()
                logger.info(f"📊 Updated outcome for recommendation #{recommendation_id}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error updating recommendation outcome: {e}")
            return False