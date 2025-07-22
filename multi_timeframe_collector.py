"""
Enhanced Multi-Timeframe Crypto Data Collector
Includes diagnostics, fresh data forcing, and better update handling
"""

import ccxt
import pandas as pd
import time
import os
import sqlite3
from datetime import datetime, timedelta
import json
import logging
import sys
import argparse

# Fix Windows console encoding issues
if sys.platform.startswith('win'):
    try:
        os.system('chcp 65001 > nul')
    except:
        pass

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_collector.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

class EnhancedMultiTimeframeCollector:
    def __init__(self):
        """Initialize the enhanced multi-timeframe data collector"""
        self.exchange = ccxt.binance()
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        
        # Enhanced timeframe configuration
        self.timeframes = {
            '5m': {'limit': 288, 'days': 1, 'update_interval': 300},      # 5 minutes
            '15m': {'limit': 672, 'days': 7, 'update_interval': 900},     # 15 minutes  
            '1h': {'limit': 168, 'days': 7, 'update_interval': 3600},     # 1 hour
            '4h': {'limit': 168, 'days': 28, 'update_interval': 14400},   # 4 hours
            '1d': {'limit': 90, 'days': 90, 'update_interval': 86400}     # 1 day
        }
        
        # Database setup
        self.db_path = 'data/multi_timeframe_data.db'
        os.makedirs('data', exist_ok=True)
        self.init_database()
        
        # Status tracking
        self.last_update = {}
        self.error_count = 0
        self.max_errors = 5
        
        logging.info("🚀 Enhanced Multi-Timeframe Collector initialized")
    
    def init_database(self):
        """Initialize SQLite database for multi-timeframe data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create comprehensive table for all timeframes
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
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timeframe, timestamp)
            )
        ''')
        
        # Create index for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_symbol_timeframe_timestamp 
            ON price_data(symbol, timeframe, timestamp)
        ''')
        
        # Create status table for tracking updates
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS collection_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                last_update DATETIME NOT NULL,
                records_count INTEGER NOT NULL,
                status TEXT NOT NULL,
                last_timestamp DATETIME,
                UNIQUE(symbol, timeframe)
            )
        ''')
        
        conn.commit()
        conn.close()
        logging.info("✅ Database initialized")
    
    def diagnose_database(self):
        """Diagnose current database state"""
        print("\n🔍 DIAGNOSING DATABASE")
        print("=" * 50)
        
        if not os.path.exists(self.db_path):
            print("❌ Database file doesn't exist!")
            return False
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check data freshness
            cursor.execute("""
                SELECT symbol, timeframe, COUNT(*), MAX(timestamp), MIN(timestamp)
                FROM price_data 
                GROUP BY symbol, timeframe
                ORDER BY symbol, timeframe
            """)
            
            results = cursor.fetchall()
            
            if not results:
                print("⚠️ No data found in database")
                conn.close()
                return False
            
            print(f"\n📈 Current Data Status:")
            print(f"{'Symbol':<12} {'Timeframe':<10} {'Count':<8} {'Latest':<20} {'Age'}")
            print("-" * 70)
            
            fresh_count = 0
            total_count = len(results)
            
            for row in results:
                symbol, timeframe, count, latest, oldest = row
                latest_dt = pd.to_datetime(latest)
                hours_old = (datetime.now() - latest_dt).total_seconds() / 3600
                
                if hours_old < 2:
                    status = "🟢 Fresh"
                    fresh_count += 1
                elif hours_old < 24:
                    status = f"🟡 {hours_old:.0f}h old"
                else:
                    status = f"🔴 {hours_old/24:.1f}d old"
                
                print(f"{symbol:<12} {timeframe:<10} {count:<8} {latest:<20} {status}")
            
            conn.close()
            
            print(f"\n📊 Summary: {fresh_count}/{total_count} datasets are fresh")
            
            if fresh_count == 0:
                print("⚠️ All data is stale - recommended to force fresh collection")
                return False
            elif fresh_count < total_count:
                print("⚠️ Some data is stale - consider updating")
                return True
            else:
                print("✅ All data is fresh")
                return True
            
        except Exception as e:
            print(f"❌ Database error: {e}")
            return False
    
    def clear_old_data(self, hours_to_keep=24):
        """Clear old data to force fresh collection"""
        print(f"\n🗑️ CLEARING DATA OLDER THAN {hours_to_keep} HOURS")
        print("=" * 50)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_time = datetime.now() - timedelta(hours=hours_to_keep)
            
            # Delete old data
            cursor.execute("""
                DELETE FROM price_data 
                WHERE timestamp < ?
            """, (cutoff_time.strftime('%Y-%m-%d %H:%M:%S'),))
            
            deleted_count = cursor.rowcount
            print(f"🗑️ Deleted {deleted_count} old records")
            
            # Update collection status to force refresh
            cursor.execute("DELETE FROM collection_status")
            print("🔄 Reset collection status")
            
            conn.commit()
            conn.close()
            
            print("✅ Database cleanup complete")
            return True
            
        except Exception as e:
            print(f"❌ Cleanup error: {e}")
            return False
    
    def force_fresh_collection(self):
        """Force collection of completely fresh data"""
        print(f"\n🚀 FORCING FRESH DATA COLLECTION")
        print("=" * 50)
        
        total_success = 0
        total_attempts = 0
        
        for symbol in self.symbols:
            print(f"\n📊 Collecting fresh data for {symbol}...")
            
            for timeframe, config in self.timeframes.items():
                total_attempts += 1
                limit = config['limit']
                
                try:
                    print(f"  📈 Fetching {timeframe} data (limit: {limit})...")
                    
                    # Fetch fresh data from exchange
                    ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                    
                    if not ohlcv:
                        print(f"    ⚠️ No data received for {symbol} {timeframe}")
                        continue
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df['symbol'] = symbol
                    df['timeframe'] = timeframe
                    
                    # Save to database (replace existing)
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    
                    # Delete existing data for this symbol/timeframe
                    cursor.execute("""
                        DELETE FROM price_data 
                        WHERE symbol = ? AND timeframe = ?
                    """, (symbol, timeframe))
                    
                    # Insert fresh data
                    df[['symbol', 'timeframe', 'timestamp', 'open', 'high', 'low', 'close', 'volume']].to_sql(
                        'price_data', conn, if_exists='append', index=False
                    )
                    
                    # Update status
                    cursor.execute("""
                        INSERT OR REPLACE INTO collection_status 
                        (symbol, timeframe, last_update, records_count, status, last_timestamp)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (symbol, timeframe, datetime.now(), len(df), 'SUCCESS', df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')))
                    
                    conn.commit()
                    conn.close()
                    
                    latest_time = df['timestamp'].max()
                    minutes_ago = (datetime.now() - latest_time).total_seconds() / 60
                    print(f"    ✅ {len(df)} records saved - Latest: {latest_time} ({minutes_ago:.0f}m ago)")
                    total_success += 1
                    
                    # Rate limiting
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"    ❌ Error collecting {symbol} {timeframe}: {e}")
                    self.error_count += 1
        
        print(f"\n🎉 Fresh collection complete: {total_success}/{total_attempts} successful")
        return total_success > 0
    
    def collect_timeframe_data(self, symbol, timeframe, force_update=False):
        """Collect data for a specific symbol and timeframe"""
        try:
            config = self.timeframes[timeframe]
            limit = config['limit']
            
            # Check if we need to update (unless forcing)
            if not force_update:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT last_timestamp FROM collection_status 
                    WHERE symbol = ? AND timeframe = ?
                """, (symbol, timeframe))
                
                result = cursor.fetchone()
                if result:
                    last_timestamp = pd.to_datetime(result[0])
                    hours_since_update = (datetime.now() - last_timestamp).total_seconds() / 3600
                    update_interval_hours = config['update_interval'] / 3600
                    
                    if hours_since_update < update_interval_hours:
                        logging.info(f"⏭️ Skipping {symbol} {timeframe} - updated {hours_since_update:.1f}h ago")
                        conn.close()
                        return True
                
                conn.close()
            
            logging.info(f"📊 Collecting {symbol} {timeframe} data (limit: {limit})")
            
            # Fetch data from exchange
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv:
                logging.warning(f"⚠️ No data received for {symbol} {timeframe}")
                return False
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            df['timeframe'] = timeframe
            
            # Save to database
            records_saved = self.save_to_database(df, replace_existing=force_update)
            
            # Update status
            self.update_collection_status(symbol, timeframe, len(df), 'SUCCESS', df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S'))
            
            latest_time = df['timestamp'].max()
            minutes_ago = (datetime.now() - latest_time).total_seconds() / 60
            logging.info(f"✅ {symbol} {timeframe}: {records_saved} records saved - Latest: {latest_time} ({minutes_ago:.0f}m ago)")
            return True
            
        except Exception as e:
            logging.error(f"❌ Error collecting {symbol} {timeframe}: {e}")
            self.update_collection_status(symbol, timeframe, 0, f'ERROR: {str(e)}', None)
            self.error_count += 1
            return False
    
    def save_to_database(self, df, replace_existing=False):
        """Save DataFrame to database with conflict handling"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            if replace_existing:
                # Delete existing data first
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM price_data 
                    WHERE symbol = ? AND timeframe = ?
                """, (df['symbol'].iloc[0], df['timeframe'].iloc[0]))
            
            # Insert new data
            records_before = self.get_record_count(df['symbol'].iloc[0], df['timeframe'].iloc[0])
            
            df[['symbol', 'timeframe', 'timestamp', 'open', 'high', 'low', 'close', 'volume']].to_sql(
                'price_data', conn, if_exists='append', index=False, method='multi'
            )
            
            records_after = self.get_record_count(df['symbol'].iloc[0], df['timeframe'].iloc[0])
            records_added = records_after - records_before if not replace_existing else len(df)
            
            conn.commit()
            return records_added
            
        except Exception as e:
            logging.error(f"Database error: {e}")
            conn.rollback()
            return 0
        finally:
            conn.close()
    
    def get_record_count(self, symbol, timeframe):
        """Get current record count for symbol/timeframe"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT COUNT(*) FROM price_data WHERE symbol = ? AND timeframe = ?",
            (symbol, timeframe)
        )
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def update_collection_status(self, symbol, timeframe, records_count, status, last_timestamp):
        """Update collection status in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO collection_status 
            (symbol, timeframe, last_update, records_count, status, last_timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (symbol, timeframe, datetime.now(), records_count, status, last_timestamp))
        
        conn.commit()
        conn.close()
    
    def collect_all_data(self, force_update=False):
        """Collect data for all symbols and timeframes"""
        start_time = datetime.now()
        mode = "FORCED UPDATE" if force_update else "NORMAL COLLECTION"
        logging.info(f"🔄 Starting {mode} at {start_time}")
        
        total_collections = 0
        successful_collections = 0
        
        for symbol in self.symbols:
            for timeframe in self.timeframes.keys():
                total_collections += 1
                
                if self.collect_timeframe_data(symbol, timeframe, force_update):
                    successful_collections += 1
                
                # Rate limiting - be nice to the API
                time.sleep(0.5)
                
                # Check error threshold
                if self.error_count >= self.max_errors:
                    logging.error(f"❌ Too many errors ({self.error_count}), stopping collection")
                    break
            
            if self.error_count >= self.max_errors:
                break
            
            # Longer pause between symbols
            time.sleep(1)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logging.info(f"🎉 Collection complete in {duration:.1f}s")
        logging.info(f"📊 Success rate: {successful_collections}/{total_collections}")
        
        # Reset error count on successful collection
        if successful_collections > total_collections * 0.8:  # 80% success rate
            self.error_count = 0
        
        return successful_collections, total_collections
    
    def display_status(self):
        """Display current collection status"""
        print(f"\n📊 MULTI-TIMEFRAME DATA STATUS")
        print("=" * 70)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT symbol, timeframe, records_count, last_update, status, last_timestamp
                FROM collection_status
                ORDER BY symbol, timeframe
            """)
            
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                print("⚠️ No collection status available")
                return
            
            print(f"{'Symbol':<12} {'Timeframe':<10} {'Records':<8} {'Last Update':<20} {'Data Age':<15} {'Status'}")
            print("-" * 90)
            
            total_records = 0
            fresh_count = 0
            
            for row in results:
                symbol, timeframe, records, last_update, status, last_timestamp = row
                total_records += records
                
                last_update_dt = pd.to_datetime(last_update)
                update_age = datetime.now() - last_update_dt
                
                if last_timestamp:
                    data_dt = pd.to_datetime(last_timestamp)
                    data_age = datetime.now() - data_dt
                    
                    if data_age.total_seconds() < 7200:  # 2 hours
                        data_status = "🟢 Fresh"
                        fresh_count += 1
                    elif data_age.total_seconds() < 86400:  # 24 hours
                        data_status = f"🟡 {data_age.total_seconds()/3600:.0f}h old"
                    else:
                        data_status = f"🔴 {data_age.days}d old"
                else:
                    data_status = "❓ Unknown"
                
                status_emoji = "✅" if status == 'SUCCESS' else "❌"
                
                print(f"{symbol:<12} {timeframe:<10} {records:<8} "
                      f"{last_update_dt.strftime('%Y-%m-%d %H:%M'):<20} "
                      f"{data_status:<15} {status_emoji}")
            
            print(f"\n📈 Summary:")
            print(f"   Total Records: {total_records:,}")
            print(f"   Fresh Datasets: {fresh_count}/{len(results)}")
            
            if fresh_count == len(results):
                print("   ✅ All data is fresh and ready for analysis!")
            elif fresh_count > 0:
                print("   ⚠️ Some data needs updating")
            else:
                print("   🔴 All data is stale - run with --force to refresh")
                
        except Exception as e:
            print(f"❌ Error displaying status: {e}")

def main():
    """Main function with command line options"""
    parser = argparse.ArgumentParser(description='Enhanced Multi-Timeframe Crypto Data Collector')
    parser.add_argument('--force', action='store_true', 
                       help='Force fresh data collection (ignores update intervals)')
    parser.add_argument('--diagnose', action='store_true', 
                       help='Diagnose database and show current status')
    parser.add_argument('--clear', type=int, metavar='HOURS',
                       help='Clear data older than specified hours before collecting')
    parser.add_argument('--status', action='store_true',
                       help='Show current data status only')
    
    args = parser.parse_args()
    
    print("🚀 ENHANCED MULTI-TIMEFRAME CRYPTO DATA COLLECTOR")
    print("=" * 60)
    
    collector = EnhancedMultiTimeframeCollector()
    
    # Handle different modes
    if args.status:
        collector.display_status()
        return
    
    if args.diagnose:
        collector.diagnose_database()
        return
    
    if args.clear:
        collector.clear_old_data(args.clear)
        args.force = True  # Force collection after clearing
    
    # Show current status first
    collector.diagnose_database()
    
    if args.force:
        print("\n🚀 FORCING FRESH DATA COLLECTION...")
        collector.force_fresh_collection()
    else:
        print("\n🔄 STARTING NORMAL DATA COLLECTION...")
        successful, total = collector.collect_all_data()
        
        if successful < total * 0.5:  # Less than 50% success
            print("\n⚠️ Low success rate detected. Consider running with --force")
    
    # Display final status
    collector.display_status()
    
    print(f"\n🎉 Collection process complete!")
    print(f"💡 Your analysis scripts now have fresh data to work with!")

if __name__ == "__main__":
    main()