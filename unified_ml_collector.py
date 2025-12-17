"""
Unified Comprehensive ML Data Collector
Combines extensive historical data collection with ML-optimized database storage

Features:
- Collects 1-24 months of historical data across multiple timeframes
- Stores data in enhanced ML-ready database schema
- Supports both SQLite (local) and PostgreSQL (Neon cloud)
- Includes technical indicators and feature engineering
- Comprehensive status tracking and diagnostics
"""
import os
import sys
import ccxt
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import time
import logging
import argparse
import json

# Fix Windows console encoding
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, OSError):
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')
    
    try:
        os.system('chcp 65001 > nul')
    except:
        pass

# Try to import PostgreSQL support
try:
    import psycopg2
    from psycopg2 import extras
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('unified_ml_collector.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)


class UnifiedMLCollector:
    """
    Unified collector for comprehensive ML-ready cryptocurrency data
    """
    
    def __init__(self, db_type='sqlite', db_path='data/ml_crypto_data.db'):
        """
        Initialize the unified ML data collector
        
        Args:
            db_type: 'sqlite' or 'postgres'
            db_path: Path to SQLite database (if using SQLite)
        """
        self.db_type = db_type
        self.db_path = db_path
        self.connection = None
        self.exchange = ccxt.binance()
        
        # Trading symbols to collect
        self.symbols = [
            'BTC/USDT',
            'ETH/USDT',
            'BNB/USDT',
            'ADA/USDT',
            'DOT/USDT'
        ]
        
        # Comprehensive timeframe configuration
        # Each timeframe specifies how much history to collect and expected sample count
        self.timeframes = {
            '5m':  {'months_back': 1,  'ml_samples': 8640,  'description': '1 month of 5-min candles'},
            '15m': {'months_back': 2,  'ml_samples': 5760,  'description': '2 months of 15-min candles'},
            '1h':  {'months_back': 6,  'ml_samples': 4320,  'description': '6 months of hourly candles'},
            '4h':  {'months_back': 12, 'ml_samples': 2190,  'description': '12 months of 4-hour candles'},
            '1d':  {'months_back': 24, 'ml_samples': 730,   'description': '24 months of daily candles'}
        }
        
        # Create data directory
        os.makedirs('data', exist_ok=True)
        
        # Initialize database connection
        self._init_database_connection()
        
        # Initialize database schema
        self._init_database_schema()
        
        logging.info("🧠 Unified ML Data Collector initialized")
        logging.info(f"📊 Database: {self.db_type.upper()}")
        logging.info(f"💾 Target: {sum(config['ml_samples'] for config in self.timeframes.values())} total candles per symbol")
    
    def _init_database_connection(self):
        """Initialize database connection based on type"""
        if self.db_type == 'sqlite':
            self.connection = sqlite3.connect(self.db_path)
            logging.info(f"✅ Connected to SQLite: {self.db_path}")
        
        elif self.db_type == 'postgres':
            if not POSTGRES_AVAILABLE:
                raise ImportError("psycopg2 not installed. Install with: pip install psycopg2-binary")
            
            # Load Neon configuration
            config_path = 'neon_config.json'
            if not os.path.exists(config_path):
                raise FileNotFoundError("neon_config.json not found")
            
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            db_config = config_data['connection']
            
            # Get password from environment
            password = os.environ.get('NEON_PASSWORD')
            if not password:
                raise ValueError("NEON_PASSWORD environment variable not set")
            
            db_config['password'] = password
            
            self.connection = psycopg2.connect(**db_config)
            logging.info("✅ Connected to Neon PostgreSQL")
        
        else:
            raise ValueError(f"Invalid db_type: {self.db_type}")
    
    def _init_database_schema(self):
        """Initialize the enhanced ML database schema"""
        logging.info("🗄️ Initializing enhanced ML database schema...")
        
        schema_file = 'enhanced_ml_database_schema.sql'
        
        # Try to use schema file first
        schema_success = False
        if os.path.exists(schema_file):
            try:
                self._execute_schema_file(schema_file)
                # Verify that price_data table was created
                if self._verify_table_exists('price_data'):
                    schema_success = True
                    logging.info("✅ Schema file executed successfully")
                else:
                    logging.warning("⚠️ Schema file executed but tables not found")
            except Exception as e:
                logging.warning(f"⚠️ Schema file execution failed: {e}")
        else:
            logging.warning("⚠️ Schema file not found")
        
        # If schema file failed, create basic schema
        if not schema_success:
            logging.info("🔧 Creating basic schema as fallback...")
            self._create_basic_schema()
            
            # Verify basic schema worked
            if not self._verify_table_exists('price_data'):
                raise RuntimeError("❌ Failed to create database schema!")
    
    def _verify_table_exists(self, table_name: str) -> bool:
        """Verify that a table exists in the database"""
        try:
            cursor = self.connection.cursor()
            
            if self.db_type == 'sqlite':
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name=?
                """, (table_name,))
            else:
                cursor.execute("""
                    SELECT tablename FROM pg_tables 
                    WHERE tablename = %s
                """, (table_name,))
            
            result = cursor.fetchone()
            cursor.close()
            
            return result is not None
            
        except Exception as e:
            logging.warning(f"⚠️ Error verifying table {table_name}: {e}")
            return False
    
    def _execute_schema_file(self, schema_file):
        """Execute SQL schema file"""
        try:
            with open(schema_file, 'r', encoding='utf-8') as f:
                schema_sql = f.read()
            
            cursor = self.connection.cursor()
            
            # Remove comments and split into statements
            lines = []
            for line in schema_sql.split('\n'):
                # Skip comment lines
                if line.strip().startswith('--'):
                    continue
                lines.append(line)
            
            clean_sql = '\n'.join(lines)
            
            # Split by semicolons
            statements = clean_sql.split(';')
            
            executed_count = 0
            for statement in statements:
                statement = statement.strip()
                
                if not statement:
                    continue
                
                try:
                    # Adjust SQL for database type
                    if self.db_type == 'sqlite':
                        # SQLite-specific adjustments
                        statement = statement.replace('VARCHAR(20)', 'TEXT')
                        statement = statement.replace('VARCHAR(50)', 'TEXT')
                        statement = statement.replace('VARCHAR(100)', 'TEXT')
                        statement = statement.replace('VARCHAR(10)', 'TEXT')
                        statement = statement.replace('VARCHAR', 'TEXT')
                        statement = statement.replace('DECIMAL(20, 8)', 'REAL')
                        statement = statement.replace('DECIMAL(10, 6)', 'REAL')
                        statement = statement.replace('DECIMAL(10, 4)', 'REAL')
                        statement = statement.replace('DECIMAL(10, 2)', 'REAL')
                        statement = statement.replace('DECIMAL(30, 2)', 'REAL')
                        statement = statement.replace('BOOLEAN', 'INTEGER')
                        # Fix AUTOINCREMENT syntax
                        statement = statement.replace('INTEGER PRIMARY KEY', 'INTEGER PRIMARY KEY AUTOINCREMENT')
                    
                    cursor.execute(statement)
                    executed_count += 1
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    # Only log if it's not a harmless error
                    if 'view' not in statement.lower() and 'already exists' not in error_msg:
                        logging.warning(f"Schema statement warning: {e}")
            
            self.connection.commit()
            cursor.close()
            
            logging.info(f"✅ Executed {executed_count} schema statements")
            
        except Exception as e:
            logging.error(f"❌ Error executing schema file: {e}")
            raise
    
    def _create_basic_schema(self):
        """Create basic schema if full schema fails"""
        cursor = self.connection.cursor()
        
        # Adjust data types for database
        if self.db_type == 'sqlite':
            decimal_type = 'REAL'
            text_type = 'TEXT'
            timestamp_type = 'DATETIME'
            primary_key = 'INTEGER PRIMARY KEY AUTOINCREMENT'
        else:
            decimal_type = 'DECIMAL(20, 8)'
            text_type = 'VARCHAR(20)'
            timestamp_type = 'TIMESTAMP'
            primary_key = 'SERIAL PRIMARY KEY'
        
        # Essential price_data table
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS price_data (
                id {primary_key},
                symbol {text_type} NOT NULL,
                timeframe {text_type} NOT NULL,
                timestamp {timestamp_type} NOT NULL,
                open {decimal_type} NOT NULL,
                high {decimal_type} NOT NULL,
                low {decimal_type} NOT NULL,
                close {decimal_type} NOT NULL,
                volume {decimal_type} NOT NULL,
                price_change_pct {decimal_type},
                data_quality {text_type} DEFAULT 'COMPLETE',
                created_at {timestamp_type} DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timeframe, timestamp)
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_symbol_timeframe_timestamp 
            ON price_data(symbol, timeframe, timestamp)
        ''')
        
        # Collection status table
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS collection_status (
                id {primary_key},
                symbol {text_type} NOT NULL,
                timeframe {text_type} NOT NULL,
                last_update {timestamp_type},
                records_count INTEGER,
                status {text_type},
                earliest_timestamp {timestamp_type},
                latest_timestamp {timestamp_type},
                completeness_pct {decimal_type},
                UNIQUE(symbol, timeframe)
            )
        ''')
        
        self.connection.commit()
        cursor.close()
        
        logging.info("✅ Basic database schema created")

    
    def collect_historical_data(self, symbol: str, timeframe: str, months_back: int) -> pd.DataFrame:
        """
        Collect extended historical data from exchange
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1h', '1d')
            months_back: How many months of history to collect
        
        Returns:
            DataFrame with historical OHLCV data
        """
        logging.info(f"📥 Collecting {months_back} months of {symbol} {timeframe} data...")
        
        all_candles = []
        
        # Calculate timestamps
        end_time = datetime.now()
        start_time = end_time - timedelta(days=months_back * 30)
        
        # Convert to milliseconds
        since = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        
        # Determine batch size based on timeframe
        timeframe_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '1h': 60,
            '4h': 240, '1d': 1440
        }
        
        minutes = timeframe_minutes.get(timeframe, 60)
        limit = 1000  # Max candles per request for Binance
        
        current_since = since
        batch_num = 0
        
        try:
            while current_since < end_ms:
                batch_num += 1
                
                # Fetch batch
                candles = self.exchange.fetch_ohlcv(
                    symbol, 
                    timeframe, 
                    since=current_since, 
                    limit=limit
                )
                
                if not candles:
                    break
                
                all_candles.extend(candles)
                
                # Update since for next batch
                current_since = candles[-1][0] + 1
                
                # Progress update
                progress_pct = ((current_since - since) / (end_ms - since)) * 100
                logging.info(f"   📊 Batch {batch_num}: {len(all_candles)} candles ({progress_pct:.1f}% complete)")
                
                # Rate limiting
                time.sleep(0.5)
                
                # Safety break
                if batch_num > 100:
                    logging.warning("⚠️ Reached batch limit, stopping")
                    break
            
            # Convert to DataFrame
            if all_candles:
                df = pd.DataFrame(
                    all_candles,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['symbol'] = symbol
                df['timeframe'] = timeframe
                
                # Calculate additional metrics
                df['price_change_pct'] = df['close'].pct_change() * 100
                df['high_low_range'] = df['high'] - df['low']
                df['body_size'] = abs(df['close'] - df['open'])
                
                logging.info(f"✅ Collected {len(df)} candles for {symbol} {timeframe}")
                logging.info(f"   📅 Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                
                return df
            else:
                logging.warning(f"⚠️ No data collected for {symbol} {timeframe}")
                return pd.DataFrame()
                
        except Exception as e:
            logging.error(f"❌ Error collecting {symbol} {timeframe}: {e}")
            return pd.DataFrame()
    
    def save_to_database(self, df: pd.DataFrame) -> int:
        """
        Save DataFrame to database with conflict handling
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            Number of records saved
        """
        if df.empty:
            return 0
        
        try:
            # Prepare data
            df_save = df[[
                'symbol', 'timeframe', 'timestamp',
                'open', 'high', 'low', 'close', 'volume',
                'price_change_pct'
            ]].copy()
            
            df_save['data_quality'] = 'COMPLETE'
            
            # Get record count before insert
            records_before = self._get_record_count(
                df['symbol'].iloc[0], df['timeframe'].iloc[0]
            )
            
            if self.db_type == 'sqlite':
                # SQLite: Use INSERT OR IGNORE for duplicates
                df_save.to_sql('price_data', self.connection, 
                              if_exists='append', index=False)
            
            elif self.db_type == 'postgres':
                # PostgreSQL: Use ON CONFLICT DO NOTHING
                cursor = self.connection.cursor()
                
                insert_query = """
                    INSERT INTO price_data 
                    (symbol, timeframe, timestamp, open, high, low, close, volume, 
                     price_change_pct, data_quality)
                    VALUES %s
                    ON CONFLICT (symbol, timeframe, timestamp) DO NOTHING
                """
                
                # Convert DataFrame to list of tuples
                data = [tuple(row) for row in df_save.values]
                
                # Execute batch insert
                extras.execute_values(cursor, insert_query, data)
                self.connection.commit()
                cursor.close()
            
            # Get record count after insert
            records_after = self._get_record_count(
                df['symbol'].iloc[0], df['timeframe'].iloc[0]
            )
            
            records_saved = records_after - records_before
            
            logging.info(f"💾 Saved {records_saved} new records to database")
            
            return records_saved
            
        except Exception as e:
            logging.error(f"❌ Database save error: {e}")
            if self.db_type == 'postgres':
                self.connection.rollback()
            return 0
    
    def _get_record_count(self, symbol: str, timeframe: str) -> int:
        """Get current record count for symbol/timeframe"""
        cursor = self.connection.cursor()
        
        if self.db_type == 'sqlite':
            query = "SELECT COUNT(*) FROM price_data WHERE symbol = ? AND timeframe = ?"
        else:
            query = "SELECT COUNT(*) FROM price_data WHERE symbol = %s AND timeframe = %s"
        
        cursor.execute(query, (symbol, timeframe))
        count = cursor.fetchone()[0]
        cursor.close()
        
        return count
    
    def update_collection_status(self, symbol: str, timeframe: str):
        """Update collection status table"""
        try:
            cursor = self.connection.cursor()
            
            # Get statistics
            if self.db_type == 'sqlite':
                stats_query = """
                    SELECT 
                        COUNT(*) as count,
                        MIN(timestamp) as earliest,
                        MAX(timestamp) as latest
                    FROM price_data
                    WHERE symbol = ? AND timeframe = ?
                """
            else:
                stats_query = """
                    SELECT 
                        COUNT(*) as count,
                        MIN(timestamp) as earliest,
                        MAX(timestamp) as latest
                    FROM price_data
                    WHERE symbol = %s AND timeframe = %s
                """
            
            cursor.execute(stats_query, (symbol, timeframe))
            result = cursor.fetchone()
            count, earliest, latest = result
            
            if count > 0:
                # Calculate completeness
                expected_records = self.timeframes[timeframe]['ml_samples']
                completeness = min((count / expected_records) * 100, 100)
                
                # Update status
                if self.db_type == 'sqlite':
                    status_query = """
                        INSERT OR REPLACE INTO collection_status 
                        (symbol, timeframe, last_update, records_count, status, 
                         earliest_timestamp, latest_timestamp, completeness_pct)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """
                else:
                    status_query = """
                        INSERT INTO collection_status 
                        (symbol, timeframe, last_update, records_count, status, 
                         earliest_timestamp, latest_timestamp, completeness_pct)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (symbol, timeframe) DO UPDATE SET
                            last_update = EXCLUDED.last_update,
                            records_count = EXCLUDED.records_count,
                            status = EXCLUDED.status,
                            earliest_timestamp = EXCLUDED.earliest_timestamp,
                            latest_timestamp = EXCLUDED.latest_timestamp,
                            completeness_pct = EXCLUDED.completeness_pct
                    """
                
                cursor.execute(status_query, (
                    symbol, timeframe, datetime.now(), count, 'COMPLETE',
                    earliest, latest, completeness
                ))
                
                self.connection.commit()
                logging.info(f"📊 Status updated: {count} records ({completeness:.1f}% complete)")
            
            cursor.close()
            
        except Exception as e:
            logging.error(f"❌ Status update error: {e}")
            if self.db_type == 'postgres':
                self.connection.rollback()
    
    def collect_all_data(self):
        """Main method to collect all comprehensive data for ML"""
        start_time = datetime.now()
        
        logging.info("🚀 STARTING COMPREHENSIVE DATA COLLECTION FOR ML")
        logging.info("=" * 70)
        
        total_symbols = len(self.symbols)
        total_timeframes = len(self.timeframes)
        total_collections = total_symbols * total_timeframes
        
        completed = 0
        total_records = 0
        
        for symbol_idx, symbol in enumerate(self.symbols, 1):
            logging.info(f"\n{'='*70}")
            logging.info(f"📊 SYMBOL {symbol_idx}/{total_symbols}: {symbol}")
            logging.info(f"{'='*70}")
            
            for tf_idx, (timeframe, config) in enumerate(self.timeframes.items(), 1):
                logging.info(f"\n⏱️  Timeframe {tf_idx}/{total_timeframes}: {timeframe}")
                logging.info(f"   Target: {config['ml_samples']} candles ({config['months_back']} months)")
                
                # Collect historical data
                df = self.collect_historical_data(
                    symbol, 
                    timeframe, 
                    config['months_back']
                )
                
                if not df.empty:
                    # Save to database
                    records_saved = self.save_to_database(df)
                    total_records += records_saved
                    
                    # Update status
                    self.update_collection_status(symbol, timeframe)
                    
                    completed += 1
                else:
                    logging.warning(f"⚠️ No data collected for {symbol} {timeframe}")
                
                # Progress
                progress = (completed / total_collections) * 100
                logging.info(f"📈 Overall Progress: {completed}/{total_collections} ({progress:.1f}%)")
                
                # Rate limiting between timeframes
                time.sleep(1)
            
            # Longer pause between symbols
            if symbol_idx < total_symbols:
                logging.info("\n⏸️  Pausing 5 seconds before next symbol...")
                time.sleep(5)
        
        # Final summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logging.info("\n" + "="*70)
        logging.info("🎉 COMPREHENSIVE DATA COLLECTION COMPLETE!")
        logging.info("="*70)
        logging.info(f"✅ Completed: {completed}/{total_collections} collections")
        logging.info(f"💾 Total records: {total_records:,}")
        logging.info(f"⏱️  Duration: {duration/60:.1f} minutes")
        logging.info(f"📊 Average: {total_records/duration:.1f} records/second")
        logging.info("="*70)
        
        # Display final status
        self.display_status()
    
    def display_status(self):
        """Display comprehensive database status"""
        logging.info("\n📊 DATABASE STATUS")
        logging.info("="*90)
        
        try:
            cursor = self.connection.cursor()
            
            if self.db_type == 'sqlite':
                query = """
                    SELECT 
                        symbol,
                        timeframe,
                        records_count,
                        earliest_timestamp,
                        latest_timestamp,
                        completeness_pct,
                        status
                    FROM collection_status
                    ORDER BY symbol, timeframe
                """
            else:
                query = """
                    SELECT 
                        symbol,
                        timeframe,
                        records_count,
                        earliest_timestamp,
                        latest_timestamp,
                        completeness_pct,
                        status
                    FROM collection_status
                    ORDER BY symbol, timeframe
                """
            
            cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
            
            if not results:
                logging.info("⚠️ No status data available")
                return
            
            # Display table
            logging.info(f"{'Symbol':<12} {'TF':<6} {'Records':<10} {'Earliest':<20} {'Latest':<20} {'Complete':<10} {'Status'}")
            logging.info("-"*90)
            
            total_records = 0
            
            for row in results:
                symbol = row[0]
                timeframe = row[1]
                records = row[2]
                earliest = pd.to_datetime(row[3]).strftime('%Y-%m-%d %H:%M') if row[3] else 'N/A'
                latest = pd.to_datetime(row[4]).strftime('%Y-%m-%d %H:%M') if row[4] else 'N/A'
                completeness = row[5] if row[5] else 0
                status = row[6]
                
                total_records += records if records else 0
                
                # Status emoji
                if completeness >= 90:
                    status_emoji = "🟢"
                elif completeness >= 70:
                    status_emoji = "🟡"
                else:
                    status_emoji = "🔴"
                
                logging.info(f"{symbol:<12} {timeframe:<6} {records:<10,} {earliest:<20} {latest:<20} {completeness:>6.1f}% {status_emoji:>3}")
            
            logging.info("-"*90)
            logging.info(f"TOTAL RECORDS: {total_records:,}")
            logging.info("="*90)
            
            # ML readiness assessment
            self._assess_ml_readiness(results)
            
        except Exception as e:
            logging.error(f"❌ Status display error: {e}")
    
    def _assess_ml_readiness(self, status_results):
        """Assess if data is ready for ML models"""
        logging.info("\n🧠 ML READINESS ASSESSMENT")
        logging.info("="*70)
        
        ml_ready = True
        issues = []
        
        for row in status_results:
            symbol = row[0]
            timeframe = row[1]
            records = row[2] if row[2] else 0
            
            # Minimum requirements for LSTM/GRU
            min_required = {
                '5m': 1000,
                '15m': 1000,
                '1h': 2000,
                '4h': 1000,
                '1d': 500
            }
            
            required = min_required.get(timeframe, 500)
            
            if records < required:
                ml_ready = False
                issues.append(f"{symbol} {timeframe}: {records} < {required} (need {required-records} more)")
        
        if ml_ready:
            logging.info("✅ DATA IS READY FOR ML TRAINING!")
            logging.info("   • All symbols have sufficient data")
            logging.info("   • LSTM/GRU models can be trained")
            logging.info("   • Recommended: Start with 1h and 4h timeframes")
        else:
            logging.info("⚠️ DATA NEEDS MORE COLLECTION")
            logging.info("   Issues found:")
            for issue in issues[:5]:
                logging.info(f"   - {issue}")
            
            if len(issues) > 5:
                logging.info(f"   ... and {len(issues)-5} more")
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
            logging.info("🔌 Database connection closed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Unified Comprehensive ML Data Collector',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect to SQLite (local)
  python unified_ml_collector.py
  
  # Collect to PostgreSQL (Neon cloud)
  python unified_ml_collector.py --postgres
  
  # Show status only
  python unified_ml_collector.py --status
  
  # Collect specific symbols
  python unified_ml_collector.py --symbols BTC/USDT ETH/USDT
  
  # Collect specific timeframes
  python unified_ml_collector.py --timeframes 1h 4h 1d
        """
    )
    
    parser.add_argument('--postgres', action='store_true',
                       help='Use PostgreSQL (Neon) instead of SQLite')
    parser.add_argument('--status', action='store_true',
                       help='Show database status only')
    parser.add_argument('--symbols', nargs='+',
                       help='Specific symbols to collect (default: all)')
    parser.add_argument('--timeframes', nargs='+',
                       help='Specific timeframes (default: all)')
    parser.add_argument('--db-path', default='data/ml_crypto_data.db',
                       help='SQLite database path (default: data/ml_crypto_data.db)')
    
    args = parser.parse_args()
    
    # Determine database type
    db_type = 'postgres' if args.postgres else 'sqlite'
    
    print("🚀 UNIFIED ML DATA COLLECTOR")
    print("=" * 70)
    print(f"📊 Database: {db_type.upper()}")
    print()
    
    try:
        # Initialize collector
        with UnifiedMLCollector(db_type=db_type, db_path=args.db_path) as collector:
            
            if args.status:
                # Just show status
                collector.display_status()
            else:
                # Override symbols if specified
                if args.symbols:
                    collector.symbols = args.symbols
                    print(f"🎯 Collecting symbols: {', '.join(args.symbols)}")
                
                # Override timeframes if specified
                if args.timeframes:
                    collector.timeframes = {
                        tf: config for tf, config in collector.timeframes.items()
                        if tf in args.timeframes
                    }
                    print(f"⏱️  Collecting timeframes: {', '.join(args.timeframes)}")
                
                print()
                
                # Run collection
                collector.collect_all_data()
        
        print("\n✅ Collection process complete!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Collection interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()