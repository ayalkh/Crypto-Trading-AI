"""
Quick Timestamp Fix for Enhanced Collector
Fixes the timestamp format issue in the enhanced collector
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import ccxt
import time
import os

def force_fresh_data_simple():
    """Simple force fresh data collection with proper timestamp handling"""
    print("🚀 SIMPLE FRESH DATA COLLECTION")
    print("=" * 50)
    
    db_path = 'data/multi_timeframe_data.db'
    exchange = ccxt.binance()
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    
    timeframes = {
        '5m': 100,   # Reduced for faster collection
        '15m': 100,
        '1h': 100,
        '4h': 100,
        '1d': 90
    }
    
    total_success = 0
    total_attempts = 0
    
    for symbol in symbols:
        print(f"\n📊 Collecting {symbol}...")
        
        for timeframe, limit in timeframes.items():
            total_attempts += 1
            
            try:
                print(f"  📈 Fetching {timeframe} data...")
                
                # Fetch data
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                
                if not ohlcv:
                    print(f"    ⚠️ No data received")
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['symbol'] = symbol
                df['timeframe'] = timeframe
                
                # Save to database
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Delete existing data
                cursor.execute("""
                    DELETE FROM price_data 
                    WHERE symbol = ? AND timeframe = ?
                """, (symbol, timeframe))
                
                # Insert new data with proper timestamp conversion
                for _, row in df.iterrows():
                    cursor.execute("""
                        INSERT INTO price_data 
                        (symbol, timeframe, timestamp, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row['symbol'], 
                        row['timeframe'], 
                        row['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),  # Convert to string
                        row['open'], 
                        row['high'], 
                        row['low'], 
                        row['close'], 
                        row['volume']
                    ))
                
                # Update status
                latest_timestamp = df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')
                cursor.execute("""
                    INSERT OR REPLACE INTO collection_status 
                    (symbol, timeframe, last_update, records_count, status, last_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (symbol, timeframe, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), len(df), 'SUCCESS', latest_timestamp))
                
                conn.commit()
                conn.close()
                
                minutes_ago = (datetime.now() - df['timestamp'].max()).total_seconds() / 60
                print(f"    ✅ {len(df)} records saved - Latest: {df['timestamp'].max()} ({minutes_ago:.0f}m ago)")
                total_success += 1
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
    
    print(f"\n🎉 Collection complete: {total_success}/{total_attempts} successful")
    return total_success > 0

def verify_fresh_data():
    """Verify the fresh data"""
    print(f"\n✅ VERIFYING FRESH DATA")
    print("=" * 50)
    
    try:
        conn = sqlite3.connect('data/multi_timeframe_data.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT symbol, timeframe, COUNT(*), MAX(timestamp)
            FROM price_data 
            GROUP BY symbol, timeframe
            ORDER BY symbol, timeframe
        """)
        
        results = cursor.fetchall()
        print(f"{'Symbol':<12} {'Timeframe':<10} {'Count':<8} {'Latest':<20} {'Freshness'}")
        print("-" * 70)
        
        fresh_count = 0
        total_count = 0
        
        for row in results:
            symbol, timeframe, count, latest = row
            total_count += 1
            
            latest_dt = pd.to_datetime(latest)
            minutes_old = (datetime.now() - latest_dt).total_seconds() / 60
            
            if minutes_old < 60:  # Less than 1 hour old
                status = f"🟢 {minutes_old:.0f}m old"
                fresh_count += 1
            elif minutes_old < 1440:  # Less than 1 day old
                status = f"🟡 {minutes_old/60:.1f}h old"
            else:
                status = f"🔴 {minutes_old/1440:.1f}d old"
            
            print(f"{symbol:<12} {timeframe:<10} {count:<8} {latest:<20} {status}")
        
        conn.close()
        
        print(f"\n📊 Summary: {fresh_count}/{total_count} datasets are fresh")
        return fresh_count > 0
        
    except Exception as e:
        print(f"❌ Verification error: {e}")
        return False

def main():
    """Main function"""
    print("🔧 TIMESTAMP FIX & FRESH DATA COLLECTION")
    print("=" * 50)
    
    # Force fresh data with proper timestamp handling
    if force_fresh_data_simple():
        print("✅ Fresh data collection successful!")
    else:
        print("❌ Fresh data collection failed!")
    
    # Verify the results
    verify_fresh_data()
    
    print(f"\n🎉 Process complete!")
    print(f"💡 You can now run your analysis scripts with fresh data!")

if __name__ == "__main__":
    main()