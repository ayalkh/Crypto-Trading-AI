import sqlite3
import pandas as pd
import os
import logging

class DatabaseManager:
    """
    Centralized database access for Crypto AI.
    """
    def __init__(self, db_path='data/ml_crypto_data.db'):
        # Ensure path is absolute or correct relative to execution context
        # In this project structure, 'data' is at the root
        self.db_path = db_path
        self.create_sentiment_table()

    def get_connection(self):
        if not os.path.exists(self.db_path):
            # Try to find it relative to project root if running from elsewhere
            if os.path.exists(os.path.join(os.getcwd(), self.db_path)):
                self.db_path = os.path.join(os.getcwd(), self.db_path)
            else:
                raise FileNotFoundError(f"Database not found at {self.db_path}")
        return sqlite3.connect(self.db_path)

    def get_symbols(self):
        try:
            with self.get_connection() as conn:
                query = "SELECT DISTINCT symbol FROM price_data ORDER BY symbol"
                return pd.read_sql_query(query, conn)['symbol'].tolist()
        except Exception as e:
            logging.error(f"Error getting symbols: {e}")
            return []

    def get_timeframes(self, symbol=None):
        try:
            with self.get_connection() as conn:
                if symbol:
                    query = "SELECT DISTINCT timeframe FROM price_data WHERE symbol = ? ORDER BY timeframe"
                    params = (symbol,)
                else:
                    query = "SELECT DISTINCT timeframe FROM price_data ORDER BY timeframe"
                    params = ()
                return pd.read_sql_query(query, conn, params=params)['timeframe'].tolist()
        except Exception as e:
            logging.error(f"Error getting timeframes: {e}")
            return []

    def load_data(self, symbol, timeframe, limit=500):
        try:
            with self.get_connection() as conn:
                query = """
                SELECT timestamp, open, high, low, close, volume 
                FROM price_data 
                WHERE symbol = ? AND timeframe = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
                """
                df = pd.read_sql_query(query, conn, params=(symbol, timeframe, limit))
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                # Sort ascending for plotting
                return df.sort_values('timestamp')
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            return pd.DataFrame()

    def create_sentiment_table(self):
        """Create sentiment data table if it doesn't exist"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS sentiment_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT,
                    twitter_score REAL,
                    twitter_volume INTEGER,
                    reddit_score REAL,
                    reddit_volume INTEGER,
                    composite_score REAL
                )
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_sentiment_symbol_time ON sentiment_data (symbol, timestamp)")
                conn.commit()
                logging.info("✅ Sentiment table checked/created")
        except Exception as e:
            logging.error(f"Error creating sentiment table: {e}")

    def save_sentiment(self, data: dict):
        """Save sentiment data to database"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                INSERT INTO sentiment_data (
                    timestamp, symbol, twitter_score, twitter_volume,
                    reddit_score, reddit_volume, composite_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    data['timestamp'], data['symbol'],
                    data.get('twitter_score', 0), data.get('twitter_volume', 0),
                    data.get('reddit_score', 0), data.get('reddit_volume', 0),
                    data.get('composite_score', 0)
                ))
                conn.commit()
                # logging.info(f"💾 Saved sentiment for {data['symbol']}")
        except Exception as e:
            logging.error(f"Error saving sentiment: {e}")

    def load_sentiment(self, symbol, hours=24):
        """Load recent sentiment data"""
        try:
            with self.get_connection() as conn:
                query = """
                SELECT * FROM sentiment_data 
                WHERE symbol = ? AND timestamp >= datetime('now', ?) 
                ORDER BY timestamp ASC
                """
                df = pd.read_sql_query(query, conn, params=(symbol, f'-{hours} hours'))
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
        except Exception as e:
            logging.error(f"Error loading sentiment: {e}")
            return pd.DataFrame()
