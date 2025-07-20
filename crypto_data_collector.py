"""
Crypto Trading AI - Real Data Collector
Collects live cryptocurrency data from Binance
"""

import ccxt
import pandas as pd
import time
import os
from datetime import datetime

def main():
    """Main function to collect crypto data"""
    print("🚀 CRYPTO TRADING AI - LIVE DATA COLLECTOR")
    print("=" * 55)
    
    print("🚀 Initializing Crypto Data Collector...")
    
    # Initialize Binance exchange (no API key needed for public data)
    try:
        exchange = ccxt.binance()
        print("✅ Connected to Binance exchange")
    except Exception as e:
        print(f"❌ Error connecting to Binance: {e}")
        return
    
    # Our target cryptocurrencies
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    
    # Create data folder if it doesn't exist
    os.makedirs('data', exist_ok=True)
    print("✅ Data folder ready")
    
    # Step 1: Get current prices
    print("\n💰 Getting current prices...")
    print("Symbol   | Price      | 24h Change | Volume")
    print("-" * 45)
    
    for symbol in symbols:
        try:
            ticker = exchange.fetch_ticker(symbol)
            print(f"{symbol:8} | ${ticker['last']:8,.2f} | {ticker['percentage']:+6.2f}% | {ticker['baseVolume']:,.0f}")
            time.sleep(0.5)
        except Exception as e:
            print(f"❌ Error getting {symbol}: {e}")
    
    # Step 2: Collect historical data
    print(f"\n📊 Collecting historical data...")
    
    for symbol in symbols:
        try:
            print(f"\nCollecting 7 days of hourly data for {symbol}...")
            
            # Get 7 days of hourly data (7 * 24 = 168 hours)
            ohlcv = exchange.fetch_ohlcv(symbol, '1h', limit=168)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            df['timeframe'] = '1h'
            
            # Save to CSV
            filename = f"data/{symbol.replace('/', '_')}_1h_7days.csv"
            df.to_csv(filename, index=False)
            
            print(f"✅ Saved {len(df)} records to {filename}")
            print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"   Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
            
            time.sleep(1)  # Be nice to the API
            
        except Exception as e:
            print(f"❌ Error collecting {symbol}: {e}")
    
    print(f"\n🎉 Data Collection Complete!")
    print(f"📁 Check your 'data' folder for CSV files")
    print(f"\n💡 Next step: Run analysis_notebook.py to analyze your data")

if __name__ == "__main__":
    main()