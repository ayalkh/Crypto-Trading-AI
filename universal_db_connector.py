"""
Universal Database Connector
Works with both SQLite (local) and Neon PostgreSQL (cloud)
Automatically detects which database to use
"""
import os
import sys
import json
import sqlite3
from datetime import datetime, timedelta
import pandas as pd

# Try to import PostgreSQL driver
try:
    import psycopg2
    from psycopg2 import extras
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    print("⚠️ psycopg2 not installed. Install with: pip install psycopg2-binary")


class UniversalDatabaseConnector:
    """
    Universal database connector that works with both SQLite and Neon PostgreSQL
    Automatically uses cloud database if available, falls back to SQLite
    """
    
    def __init__(self, prefer_cloud=True):
        """
        Initialize connector
        
        Args:
            prefer_cloud: If True, prefer cloud database when available
        """
        self.db_type = None
        self.connection = None
        self.config = None
        self.prefer_cloud = prefer_cloud
        
        # Detect and connect to database
        self._detect_and_connect()
    
    def _detect_and_connect(self):
        """Detect available database and connect"""
        
        # Try cloud database first if preferred
        if self.prefer_cloud and POSTGRES_AVAILABLE:
            if self._try_neon_connection():
                print("✅ Connected to Neon PostgreSQL (cloud)")
                self.db_type = 'neon'
                return
        
        # Fall back to SQLite
        if self._try_sqlite_connection():
            print("✅ Connected to SQLite (local)")
            self.db_type = 'sqlite'
            return
        
        # If cloud not preferred but available, try it now
        if not self.prefer_cloud and POSTGRES_AVAILABLE:
            if self._try_neon_connection():
                print("✅ Connected to Neon PostgreSQL (cloud)")
                self.db_type = 'neon'
                return
        
        raise ConnectionError("❌ Could not connect to any database!")
    
    def _try_neon_connection(self):
        """Try to connect to Neon PostgreSQL"""
        try:
            # Load config
            config_path = 'neon_config.json'
            if not os.path.exists(config_path):
                return False
            
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            self.config = config_data['connection']
            
            # Get password from environment variable
            password = os.environ.get('NEON_PASSWORD')
            if not password:
                print("⚠️ NEON_PASSWORD environment variable not set")
                return False
            
            # Add password to config
            self.config['password'] = password
            
            # Try to connect
            self.connection = psycopg2.connect(**self.config)
            
            # Test connection
            cursor = self.connection.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            
            return True
            
        except Exception as e:
            print(f"⚠️ Neon connection failed: {e}")
            return False
    
    def _try_sqlite_connection(self):
        """Try to connect to SQLite"""
        try:
            sqlite_path = 'data/multi_timeframe_data.db'
            
            if not os.path.exists(sqlite_path):
                return False
            
            self.connection = sqlite3.connect(sqlite_path)
            self.config = {'path': sqlite_path}
            
            return True
            
        except Exception as e:
            print(f"⚠️ SQLite connection failed: {e}")
            return False
    
    def execute_query(self, query, params=None):
        """
        Execute a query and return results
        
        Args:
            query: SQL query string
            params: Query parameters (tuple or dict)
        
        Returns:
            List of tuples with results
        """
        cursor = self.connection.cursor()
        
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            # For SELECT queries, fetch results
            if query.strip().upper().startswith('SELECT'):
                results = cursor.fetchall()
                cursor.close()
                return results
            else:
                # For INSERT/UPDATE/DELETE, commit and return rowcount
                self.connection.commit()
                rowcount = cursor.rowcount
                cursor.close()
                return rowcount
                
        except Exception as e:
            print(f"❌ Query error: {e}")
            self.connection.rollback()
            cursor.close()
            raise
    
    def load_dataframe(self, query, params=None):
        """
        Execute query and return pandas DataFrame
        
        Args:
            query: SQL query string
            params: Query parameters
        
        Returns:
            pandas DataFrame
        """
        return pd.read_sql_query(query, self.connection, params=params)
    
    def insert_dataframe(self, df, table_name, if_exists='append'):
        """
        Insert DataFrame into database
        
        Args:
            df: pandas DataFrame
            table_name: Name of table to insert into
            if_exists: What to do if table exists ('append', 'replace', 'fail')
        """
        if self.db_type == 'sqlite':
            df.to_sql(table_name, self.connection, if_exists=if_exists, index=False)
        
        elif self.db_type == 'neon':
            # For PostgreSQL, use more efficient method
            if if_exists == 'replace':
                # Drop and recreate table
                cursor = self.connection.cursor()
                cursor.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
                self.connection.commit()
            
            # Use COPY for efficient bulk insert
            from io import StringIO
            
            # Prepare data
            output = StringIO()
            df.to_csv(output, sep='\t', header=False, index=False)
            output.seek(0)
            
            # Insert data
            cursor = self.connection.cursor()
            
            try:
                # Create columns list
                columns = ', '.join(df.columns)
                
                # Use COPY for fast insert
                cursor.copy_from(output, table_name, columns=df.columns.tolist(), null='')
                self.connection.commit()
                cursor.close()
                
            except Exception as e:
                # Fall back to standard insert
                print(f"⚠️ COPY failed, using standard insert: {e}")
                df.to_sql(table_name, self.connection, if_exists=if_exists, 
                         index=False, method='multi')
    
    def get_available_symbols(self):
        """Get list of available symbols"""
        query = "SELECT DISTINCT symbol FROM price_data ORDER BY symbol"
        results = self.execute_query(query)
        return [row[0] for row in results]
    
    def get_available_timeframes(self, symbol=None):
        """Get available timeframes for a symbol"""
        if symbol:
            query = "SELECT DISTINCT timeframe FROM price_data WHERE symbol = %s ORDER BY timeframe"
            params = (symbol,)
        else:
            query = "SELECT DISTINCT timeframe FROM price_data ORDER BY timeframe"
            params = None
        
        # Adjust for SQLite parameter style
        if self.db_type == 'sqlite' and symbol:
            query = query.replace('%s', '?')
        
        results = self.execute_query(query, params)
        return [row[0] for row in results]
    
    def load_crypto_data(self, symbol, timeframe='1h', limit_hours=168):
        """
        Load crypto data for a symbol and timeframe
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Data timeframe (e.g., '1h', '4h', '1d')
            limit_hours: Number of hours of data to retrieve
        
        Returns:
            pandas DataFrame with OHLCV data
        """
        hours_ago = datetime.now() - timedelta(hours=limit_hours)
        
        query = """
        SELECT timestamp, open, high, low, close, volume
        FROM price_data 
        WHERE symbol = %s AND timeframe = %s AND timestamp >= %s
        ORDER BY timestamp ASC
        """
        
        # Adjust for SQLite parameter style
        if self.db_type == 'sqlite':
            query = query.replace('%s', '?')
        
        params = (symbol, timeframe, hours_ago.strftime('%Y-%m-%d %H:%M:%S'))
        
        df = self.load_dataframe(query, params)
        
        if df.empty:
            return df
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Ensure proper data types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any rows with NaN values
        df = df.dropna()
        
        return df
    
    def insert_price_data(self, symbol, timeframe, df):
        """
        Insert price data into database
        
        Args:
            symbol: Trading pair
            timeframe: Data timeframe
            df: DataFrame with columns: timestamp, open, high, low, close, volume
        """
        # Add symbol and timeframe columns
        df_insert = df.copy()
        df_insert['symbol'] = symbol
        df_insert['timeframe'] = timeframe
        
        # Reorder columns
        columns = ['symbol', 'timeframe', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
        df_insert = df_insert[columns]
        
        if self.db_type == 'sqlite':
            # SQLite: Simple insert
            df_insert.to_sql('price_data', self.connection, if_exists='append', index=False)
        
        elif self.db_type == 'neon':
            # PostgreSQL: Use ON CONFLICT to handle duplicates
            cursor = self.connection.cursor()
            
            insert_query = """
                INSERT INTO price_data (symbol, timeframe, timestamp, open, high, low, close, volume)
                VALUES %s
                ON CONFLICT (symbol, timeframe, timestamp) DO NOTHING
            """
            
            # Convert DataFrame to list of tuples
            data = [tuple(row) for row in df_insert.values]
            
            # Execute batch insert
            extras.execute_values(cursor, insert_query, data)
            self.connection.commit()
            cursor.close()
    
    def get_data_info(self):
        """Get information about available data"""
        query = """
        SELECT 
            symbol,
            timeframe,
            COUNT(*) as record_count,
            MAX(timestamp) as latest_timestamp,
            MIN(timestamp) as earliest_timestamp
        FROM price_data
        GROUP BY symbol, timeframe
        ORDER BY symbol, timeframe
        """
        
        return self.load_dataframe(query)
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# Convenience function
def get_db_connection(prefer_cloud=True):
    """
    Get a database connection
    
    Args:
        prefer_cloud: If True, prefer cloud database when available
    
    Returns:
        UniversalDatabaseConnector instance
    """
    return UniversalDatabaseConnector(prefer_cloud=prefer_cloud)


# Example usage
if __name__ == "__main__":
    print("🔌 Universal Database Connector - Test")
    print("=" * 50)
    
    # Test connection
    with get_db_connection() as db:
        print(f"\n✅ Connected to {db.db_type.upper()} database")
        
        # Get available symbols
        symbols = db.get_available_symbols()
        print(f"\n📊 Available symbols: {symbols}")
        
        # Get data info
        info = db.get_data_info()
        
        if not info.empty:
            print(f"\n📈 Data Summary:")
            print(info.to_string(index=False))
        else:
            print("\n⚠️ No data in database")
        
        # Test loading data
        if symbols:
            test_symbol = symbols[0]
            timeframes = db.get_available_timeframes(test_symbol)
            
            if timeframes:
                test_timeframe = timeframes[0]
                print(f"\n🔍 Testing data load: {test_symbol} {test_timeframe}")
                
                df = db.load_crypto_data(test_symbol, test_timeframe, limit_hours=24)
                print(f"   Loaded {len(df)} records")
                
                if not df.empty:
                    print(f"   Latest: {df['timestamp'].max()}")
                    print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    print("\n✅ Database connector test complete!")