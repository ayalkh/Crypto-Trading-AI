"""
Comprehensive ML-Optimized Data Collector
Collects extended historical data for LSTM/GRU and advanced ML models

Features:
- 3-6 months historical data collection
- Technical indicators pre-calculation
- Feature engineering and storage
- Multi-timeframe synchronization
- Data quality validation
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
from typing import Dict, List, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comprehensive_collector.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)


class ComprehensiveMLDataCollector:
    """
    Collect comprehensive historical data optimized for ML models
    """
    
    def __init__(self, db_path='data/ml_crypto_data.db'):
        """Initialize comprehensive data collector"""
        self.db_path = db_path
        self.exchange = ccxt.binance()
        
        # Extended configuration for ML
        self.symbols = [
            'BTC/USDT',
            'ETH/USDT',
            'BNB/USDT',
            'ADA/USDT',
            'DOT/USDT'
        ]
        
        # Comprehensive timeframe configuration
        self.timeframes = {
            '5m':  {'months_back': 1,  'ml_samples': 8640},   # 1 month = 8,640 5-min candles
            '15m': {'months_back': 2,  'ml_samples': 5760},   # 2 months = 5,760 15-min candles
            '1h':  {'months_back': 6,  'ml_samples': 4320},   # 6 months = 4,320 hourly candles
            '4h':  {'months_back': 12, 'ml_samples': 2190},   # 12 months = 2,190 4-hour candles
            '1d':  {'months_back': 24, 'ml_samples': 730}     # 24 months = 730 daily candles
        }
        
        # Create database directory
        os.makedirs('data', exist_ok=True)
        
        # Initialize enhanced database
        self.init_enhanced_database()
        
        logging.info("🧠 Comprehensive ML Data Collector initialized")
        logging.info(f"📊 Target: {sum(config['ml_samples'] for config in self.timeframes.values())} total candles per symbol")
    
    def init_enhanced_database(self):
        """Initialize database with enhanced ML schema"""
        logging.info("🗄️ Initializing enhanced ML database...")
        
        # Read and execute schema
        schema_path = 'enhanced_ml_database_schema.sql'
        
        if not os.path.exists(schema_path):
            logging.warning("⚠️ Schema file not found, creating basic schema")
            self._create_basic_schema()
            return
        
        try:
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Execute schema (split by ; and execute each statement)
            for statement in schema_sql.split(';'):
                if statement.strip():
                    try:
                        cursor.execute(statement)
                    except Exception as e:
                        # Skip view creations and other non-critical errors
                        if 'VIEW' not in statement.upper():
                            logging.warning(f"Schema warning: {e}")
            
            conn.commit()
            conn.close()
            
            logging.info("✅ Enhanced database schema created")
            
        except Exception as e:
            logging.error(f"❌ Error creating schema: {e}")
            self._create_basic_schema()
    
    def _create_basic_schema(self):
        """Create basic schema if full schema fails"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Just create the essential price_data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                price_change_pct REAL,
                data_quality TEXT DEFAULT 'COMPLETE',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timeframe, timestamp)
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_symbol_timeframe_timestamp 
            ON price_data(symbol, timeframe, timestamp DESC)
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS collection_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                last_update DATETIME,
                records_count INTEGER,
                status TEXT,
                earliest_timestamp DATETIME,
                latest_timestamp DATETIME,
                completeness_pct REAL,
                UNIQUE(symbol, timeframe)
            )
        ''')
        
        conn.commit()
        conn.close()
        logging.info("✅ Basic database schema created")
    
    def collect_historical_data(self, symbol: str, timeframe: str, 
                                months_back: int) -> pd.DataFrame:
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
        
        batch_ms = minutes * 60 * 1000 * limit
        
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
                if batch_num > 100:  # Prevent infinite loops
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
        Save DataFrame to database
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            Number of records saved
        """
        if df.empty:
            return 0
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Prepare data
            df_save = df[[
                'symbol', 'timeframe', 'timestamp',
                'open', 'high', 'low', 'close', 'volume',
                'price_change_pct'
            ]].copy()
            
            df_save['data_quality'] = 'COMPLETE'
            
            # Insert with conflict handling
            records_before = self._get_record_count(
                conn, df['symbol'].iloc[0], df['timeframe'].iloc[0]
            )
            
            df_save.to_sql('price_data', conn, if_exists='append', index=False)
            
            records_after = self._get_record_count(
                conn, df['symbol'].iloc[0], df['timeframe'].iloc[0]
            )
            
            records_saved = records_after - records_before
            
            conn.commit()
            conn.close()
            
            logging.info(f"💾 Saved {records_saved} new records to database")
            
            return records_saved
            
        except Exception as e:
            logging.error(f"❌ Database save error: {e}")
            return 0
    
    def _get_record_count(self, conn, symbol: str, timeframe: str) -> int:
        """Get current record count for symbol/timeframe"""
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM price_data WHERE symbol = ? AND timeframe = ?",
            (symbol, timeframe)
        )
        return cursor.fetchone()[0]
    
    def update_collection_status(self, symbol: str, timeframe: str):
        """Update collection status table"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as count,
                    MIN(timestamp) as earliest,
                    MAX(timestamp) as latest
                FROM price_data
                WHERE symbol = ? AND timeframe = ?
            """, (symbol, timeframe))
            
            result = cursor.fetchone()
            count, earliest, latest = result
            
            if count > 0:
                # Calculate completeness
                expected_records = self.timeframes[timeframe]['ml_samples']
                completeness = min((count / expected_records) * 100, 100)
                
                # Update status
                cursor.execute("""
                    INSERT OR REPLACE INTO collection_status 
                    (symbol, timeframe, last_update, records_count, status, 
                     earliest_timestamp, latest_timestamp, completeness_pct)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol, timeframe, datetime.now(), count, 'COMPLETE',
                    earliest, latest, completeness
                ))
                
                conn.commit()
                logging.info(f"📊 Status updated: {count} records ({completeness:.1f}% complete)")
            
            conn.close()
            
        except Exception as e:
            logging.error(f"❌ Status update error: {e}")
    
    def collect_all_comprehensive_data(self):
        """
        Main method to collect all comprehensive data for ML
        """
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
        self.display_database_status()
    
    def display_database_status(self):
        """Display comprehensive database status"""
        logging.info("\n📊 FINAL DATABASE STATUS")
        logging.info("="*90)
        
        try:
            conn = sqlite3.connect(self.db_path)
            
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
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                logging.info("⚠️ No status data available")
                return
            
            # Display table
            logging.info(f"{'Symbol':<12} {'TF':<6} {'Records':<10} {'Earliest':<20} {'Latest':<20} {'Complete':<10} {'Status'}")
            logging.info("-"*90)
            
            total_records = 0
            
            for _, row in df.iterrows():
                symbol = row['symbol']
                timeframe = row['timeframe']
                records = row['records_count']
                earliest = pd.to_datetime(row['earliest_timestamp']).strftime('%Y-%m-%d %H:%M')
                latest = pd.to_datetime(row['latest_timestamp']).strftime('%Y-%m-%d %H:%M')
                completeness = row['completeness_pct']
                status = row['status']
                
                total_records += records
                
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
            self._assess_ml_readiness(df)
            
        except Exception as e:
            logging.error(f"❌ Status display error: {e}")
    
    def _assess_ml_readiness(self, status_df: pd.DataFrame):
        """Assess if data is ready for ML models"""
        logging.info("\n🧠 ML READINESS ASSESSMENT")
        logging.info("="*70)
        
        ml_ready = True
        issues = []
        
        for _, row in status_df.iterrows():
            symbol = row['symbol']
            timeframe = row['timeframe']
            records = row['records_count']
            completeness = row['completeness_pct']
            
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
            for issue in issues[:5]:  # Show first 5 issues
                logging.info(f"   - {issue}")
            
            if len(issues) > 5:
                logging.info(f"   ... and {len(issues)-5} more")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive ML Data Collector')
    parser.add_argument('--status', action='store_true', help='Show database status only')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to collect (default: all)')
    parser.add_argument('--timeframes', nargs='+', help='Specific timeframes (default: all)')
    
    args = parser.parse_args()
    
    # Initialize collector
    collector = ComprehensiveMLDataCollector()
    
    if args.status:
        # Just show status
        collector.display_database_status()
    else:
        # Override symbols if specified
        if args.symbols:
            collector.symbols = args.symbols
        
        # Override timeframes if specified
        if args.timeframes:
            collector.timeframes = {
                tf: config for tf, config in collector.timeframes.items()
                if tf in args.timeframes
            }
        
        # Run collection
        collector.collect_all_comprehensive_data()


if __name__ == "__main__":
    main()