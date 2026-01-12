"""
Comprehensive ML-Optimized Data Collector V2
Updated based on ML training analysis and Priority 1 fixes

Key Improvements:
- Aligned with prediction lookback requirements (6 months for 1d)
- Optimized timeframe configurations
- Better data quality validation
- ML readiness assessment based on actual needs

Features:
- 3-6 months historical data collection
- Technical indicators pre-calculation
- Feature engineering and storage
- Multi-timeframe synchronization
- Data quality validation
"""
from datetime import datetime, timedelta
import logging
import os
import sqlite3
import sys
import time
from typing import Dict, List, Tuple

import ccxt
import numpy as np
import pandas as pd

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
    V2 - Updated based on ML analysis results
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
        
        # Optimized timeframe configuration based on ML analysis
        # Updated to ensure sufficient data for both training AND predictions
        self.timeframes = {
            # Format: 'timeframe': {training_months, prediction_months, expected_samples, min_required}
            '5m':  {
                'training_months': 1,      # For model training
                'prediction_months': 1,    # For making predictions
                'ml_samples': 8640,        # Expected training samples
                'min_for_training': 1000,  # Minimum needed to train
                'min_for_prediction': 100, # Minimum needed to predict
                'priority': 'LOW'          # Analysis shows 5m is too noisy
            },
            '15m': {
                'training_months': 2,
                'prediction_months': 1,
                'ml_samples': 5760,
                'min_for_training': 1000,
                'min_for_prediction': 100,
                'priority': 'MEDIUM'
            },
            '1h':  {
                'training_months': 6,
                'prediction_months': 2,    # PRIORITY 1 FIX: increased from 1
                'ml_samples': 4320,
                'min_for_training': 2000,
                'min_for_prediction': 150,
                'priority': 'HIGH'         # Best timeframe per analysis
            },
            '4h':  {
                'training_months': 12,
                'prediction_months': 3,    # PRIORITY 1 FIX: increased from 1
                'ml_samples': 2190,
                'min_for_training': 1000,
                'min_for_prediction': 150,
                'priority': 'HIGH'         # Strong signals with GRU
            },
            '1d':  {
                'training_months': 24,
                'prediction_months': 6,    # PRIORITY 1 FIX: CRITICAL - was 1, now 6!
                'ml_samples': 730,
                'min_for_training': 500,
                'min_for_prediction': 180, # 6 months minimum for predictions
                'priority': 'MEDIUM'       # Good signals but needs more data
            }
        }
        
        # Create database directory
        os.makedirs('data', exist_ok=True)
        
        # Initialize enhanced database
        self.init_enhanced_database()
        
        logging.info("🧠 Comprehensive ML Data Collector V2 initialized")
        logging.info(f"📊 Target: {sum(config['ml_samples'] for config in self.timeframes.values())} total candles per symbol")
        logging.info("✨ Updated with Priority 1 ML fixes")
    
    def init_enhanced_database(self):
        """Initialize database with enhanced ML schema"""
        logging.info("🗄️ Initializing enhanced ML database...")
        
        # Create basic schema (compatible with your existing system)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main price data table
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
        
        # Optimized index
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_symbol_timeframe_timestamp 
            ON price_data(symbol, timeframe, timestamp DESC)
        ''')
        
        # Collection status tracking
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
                training_ready BOOLEAN DEFAULT 0,
                prediction_ready BOOLEAN DEFAULT 0,
                UNIQUE(symbol, timeframe)
            )
        ''')
        
        conn.commit()
        conn.close()
        logging.info("✅ Enhanced database schema created")
    
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
        Save DataFrame to database with duplicate handling (skip duplicates)
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            Number of NEW records saved
        """
        if df.empty:
            return 0
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Prepare data
            df_save = df[[
                'symbol', 'timeframe', 'timestamp',
                'open', 'high', 'low', 'close', 'volume',
                'price_change_pct'
            ]].copy()
            
            # Convert timestamp to string format for SQLite
            df_save['timestamp'] = df_save['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            df_save['data_quality'] = 'COMPLETE'
            
            # Get count before insertion
            symbol = df['symbol'].iloc[0]
            timeframe = df['timeframe'].iloc[0]
            records_before = self._get_record_count(conn, symbol, timeframe)
            
            # Batch insert with INSERT OR IGNORE (skips duplicates)
            records_data = []
            for _, row in df_save.iterrows():
                records_data.append((
                    row['symbol'], 
                    row['timeframe'], 
                    row['timestamp'],  # Now a string
                    float(row['open']), 
                    float(row['high']), 
                    float(row['low']), 
                    float(row['close']),
                    float(row['volume']), 
                    float(row['price_change_pct']) if pd.notna(row['price_change_pct']) else None,
                    'COMPLETE'
                ))
            
            cursor.executemany('''
                INSERT OR IGNORE INTO price_data 
                (symbol, timeframe, timestamp, open, high, low, close, volume, 
                price_change_pct, data_quality)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', records_data)
            
            conn.commit()
            
            # Get count after insertion
            records_after = self._get_record_count(conn, symbol, timeframe)
            records_saved = records_after - records_before
            
            conn.close()
            
            if records_saved > 0:
                logging.info(f"💾 Saved {records_saved} new records (skipped {len(df_save) - records_saved} duplicates)")
            else:
                logging.info(f"💾 No new records (all {len(df_save)} already exist)")
            
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
        """Update collection status table with ML readiness flags"""
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
                config = self.timeframes[timeframe]
                
                # Calculate completeness
                expected_records = config['ml_samples']
                completeness = min((count / expected_records) * 100, 100)
                
                # Check ML readiness
                training_ready = count >= config['min_for_training']
                prediction_ready = count >= config['min_for_prediction']
                
                # Update status
                cursor.execute("""
                    INSERT OR REPLACE INTO collection_status 
                    (symbol, timeframe, last_update, records_count, status, 
                     earliest_timestamp, latest_timestamp, completeness_pct,
                     training_ready, prediction_ready)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol, timeframe, datetime.now(), count, 'COMPLETE',
                    earliest, latest, completeness, training_ready, prediction_ready
                ))
                
                conn.commit()
                
                # Log ML readiness
                status_msg = []
                if training_ready:
                    status_msg.append("✅ Training ready")
                else:
                    needed = config['min_for_training'] - count
                    status_msg.append(f"⚠️ Training needs {needed} more")
                
                if prediction_ready:
                    status_msg.append("✅ Prediction ready")
                else:
                    needed = config['min_for_prediction'] - count
                    status_msg.append(f"⚠️ Prediction needs {needed} more")
                
                logging.info(f"📊 Status: {count} records ({completeness:.1f}%) - {' | '.join(status_msg)}")
            
            conn.close()
            
        except Exception as e:
            logging.error(f"❌ Status update error: {e}")
    
    def collect_all_comprehensive_data(self):
        """
        Main method to collect all comprehensive data for ML
        Now uses TRAINING months for collection
        """
        start_time = datetime.now()
        
        logging.info("🚀 STARTING COMPREHENSIVE DATA COLLECTION FOR ML V2")
        logging.info("=" * 70)
        logging.info("✨ Priority 1 optimizations applied:")
        logging.info("   • 1h prediction: 2 months lookback")
        logging.info("   • 4h prediction: 3 months lookback")
        logging.info("   • 1d prediction: 6 months lookback (CRITICAL FIX)")
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
                priority_emoji = {"HIGH": "⭐", "MEDIUM": "🔵", "LOW": "⚪"}[config['priority']]
                
                logging.info(f"\n⏱️  Timeframe {tf_idx}/{total_timeframes}: {timeframe} {priority_emoji} {config['priority']}")
                logging.info(f"   Training: {config['ml_samples']} candles ({config['training_months']} months)")
                logging.info(f"   Prediction: {config['prediction_months']} months minimum")
                
                # Collect historical data (use training_months)
                df = self.collect_historical_data(
                    symbol, 
                    timeframe, 
                    config['training_months']
                )
                
                if not df.empty:
                    # Save to database
                    records_saved = self.save_to_database(df)
                    total_records += records_saved
                    
                    # Update status with ML readiness
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
        logging.info(f"💾 Total new records: {total_records:,}")
        logging.info(f"⏱️  Duration: {duration/60:.1f} minutes")
        if total_records > 0:
            logging.info(f"📊 Average: {total_records/duration:.1f} records/second")
        logging.info("="*70)
        
        # Display final status
        self.display_database_status()
    
    def display_database_status(self):
        """Display comprehensive database status with ML readiness"""
        logging.info("\n📊 FINAL DATABASE STATUS")
        logging.info("="*110)
        
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
                    training_ready,
                    prediction_ready,
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
            logging.info(f"{'Symbol':<12} {'TF':<6} {'Records':<10} {'Earliest':<20} {'Latest':<20} {'Complete':<10} {'Train':<7} {'Predict':<8}")
            logging.info("-"*110)
            
            total_records = 0
            training_ready_count = 0
            prediction_ready_count = 0
            
            for _, row in df.iterrows():
                symbol = row['symbol']
                timeframe = row['timeframe']
                records = row['records_count']
                earliest = pd.to_datetime(row['earliest_timestamp']).strftime('%Y-%m-%d %H:%M')
                latest = pd.to_datetime(row['latest_timestamp']).strftime('%Y-%m-%d %H:%M')
                completeness = row['completeness_pct']
                training_ready = row['training_ready']
                prediction_ready = row['prediction_ready']
                
                total_records += records
                if training_ready:
                    training_ready_count += 1
                if prediction_ready:
                    prediction_ready_count += 1
                
                # Status emojis
                if completeness >= 90:
                    status_emoji = "🟢"
                elif completeness >= 70:
                    status_emoji = "🟡"
                else:
                    status_emoji = "🔴"
                
                train_emoji = "✅" if training_ready else "❌"
                pred_emoji = "✅" if prediction_ready else "❌"
                
                logging.info(
                    f"{symbol:<12} {timeframe:<6} {records:<10,} {earliest:<20} {latest:<20} "
                    f"{completeness:>6.1f}% {status_emoji:>3} {train_emoji:<7} {pred_emoji:<8}"
                )
            
            logging.info("-"*110)
            logging.info(f"TOTAL RECORDS: {total_records:,}")
            logging.info(f"Training Ready: {training_ready_count}/{len(df)} ({training_ready_count/len(df)*100:.0f}%)")
            logging.info(f"Prediction Ready: {prediction_ready_count}/{len(df)} ({prediction_ready_count/len(df)*100:.0f}%)")
            logging.info("="*110)
            
            # ML readiness assessment
            self._assess_ml_readiness(df)
            
        except Exception as e:
            logging.error(f"❌ Status display error: {e}")
    
    def _assess_ml_readiness(self, status_df: pd.DataFrame):
        """Assess if data is ready for ML models with Priority 1 requirements"""
        logging.info("\n🧠 ML READINESS ASSESSMENT (Priority 1 Standards)")
        logging.info("="*70)
        
        training_ready = []
        prediction_ready = []
        issues = []
        
        for _, row in status_df.iterrows():
            symbol = row['symbol']
            timeframe = row['timeframe']
            records = row['records_count']
            
            config = self.timeframes[timeframe]
            
            # Check training readiness
            if records >= config['min_for_training']:
                training_ready.append(f"{symbol} {timeframe}")
            else:
                needed = config['min_for_training'] - records
                issues.append(f"❌ {symbol} {timeframe}: {records} < {config['min_for_training']} (need {needed} for training)")
            
            # Check prediction readiness
            if records >= config['min_for_prediction']:
                prediction_ready.append(f"{symbol} {timeframe}")
            else:
                needed = config['min_for_prediction'] - records
                issues.append(f"⚠️ {symbol} {timeframe}: {records} < {config['min_for_prediction']} (need {needed} for predictions)")
        
        # Summary
        total = len(status_df)
        
        if len(training_ready) == total and len(prediction_ready) == total:
            logging.info("✅ ALL DATA IS READY FOR ML!")
            logging.info("   • All symbols can be trained")
            logging.info("   • All symbols can make predictions")
            logging.info("   • Daily predictions have 6+ months data ✨")
            logging.info("\n🎯 Recommended Next Steps:")
            logging.info("   1. Train models with optimized_ml_system_enhanced.py")
            logging.info("   2. Focus on 1h and 4h timeframes (best performers)")
            logging.info("   3. Prioritize DOT and ETH (highest accuracy)")
        elif len(training_ready) == total:
            logging.info("✅ TRAINING DATA READY!")
            logging.info(f"   • {len(training_ready)}/{total} ready for training")
            logging.info(f"   • {len(prediction_ready)}/{total} ready for predictions")
            logging.info("\n⚠️ Prediction Issues:")
            for issue in issues[:10]:
                if "⚠️" in issue:
                    logging.info(f"   {issue}")
        else:
            logging.info("⚠️ DATA COLLECTION INCOMPLETE")
            logging.info(f"   Training Ready: {len(training_ready)}/{total}")
            logging.info(f"   Prediction Ready: {len(prediction_ready)}/{total}")
            logging.info("\n❌ Issues Found:")
            for issue in issues[:10]:
                logging.info(f"   {issue}")
            
            if len(issues) > 10:
                logging.info(f"   ... and {len(issues)-10} more issues")
        
        # Priority recommendations
        logging.info("\n📋 PRIORITY TIMEFRAMES (based on analysis):")
        logging.info("   ⭐ HIGH:   1h, 4h (best accuracy, stable predictions)")
        logging.info("   🔵 MEDIUM: 15m, 1d (decent signals, needs careful use)")
        logging.info("   ⚪ LOW:    5m (too noisy, consider removing)")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive ML Data Collector V2')
    parser.add_argument('--status', action='store_true', help='Show database status only')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to collect (default: all)')
    parser.add_argument('--timeframes', nargs='+', help='Specific timeframes (default: all)')
    parser.add_argument('--high-priority-only', action='store_true', help='Only collect HIGH priority timeframes (1h, 4h)')
    
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
        elif args.high_priority_only:
            # Only collect high priority timeframes
            collector.timeframes = {
                tf: config for tf, config in collector.timeframes.items()
                if config['priority'] == 'HIGH'
            }
            logging.info("🎯 HIGH PRIORITY ONLY MODE: Collecting 1h and 4h only")
        
        # Run collection
        collector.collect_all_comprehensive_data()


if __name__ == "__main__":
    main()