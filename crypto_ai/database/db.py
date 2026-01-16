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
