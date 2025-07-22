"""
Simple Test Script for Multi-Timeframe Analysis
Tests the connection between collector and analyzer
"""

import pandas as pd
import sqlite3
from datetime import datetime

def test_database_connection():
    """Test if we can connect to the multi-timeframe database"""
    print("TESTING: Database connection...")
    
    try:
        # Connect to the database created by multi_timeframe_collector
        conn = sqlite3.connect('data/multi_timeframe_data.db')
        
        # Check what data we have
        query = """
            SELECT symbol, timeframe, COUNT(*) as records, 
                   MIN(timestamp) as earliest, 
                   MAX(timestamp) as latest
            FROM price_data 
            GROUP BY symbol, timeframe 
            ORDER BY symbol, timeframe
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        print("SUCCESS: Database connection working!")
        print("\nAVAILABLE DATA:")
        print("=" * 60)
        print(df.to_string(index=False))
        
        return True
        
    except Exception as e:
        print(f"ERROR: Database connection failed: {e}")
        return False

def test_data_loading():
    """Test loading data for a specific symbol and timeframe"""
    print("\nTESTING: Data loading for BTC 1h...")
    
    try:
        # This is how the analyzer loads data
        conn = sqlite3.connect('data/multi_timeframe_data.db')
        
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM price_data 
            WHERE symbol = 'BTC/USDT' AND timeframe = '1h'
            ORDER BY timestamp DESC
            LIMIT 10
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            print("SUCCESS: Data loading working!")
            print("\nLATEST 10 RECORDS (BTC 1h):")
            print("=" * 50)
            print(df.to_string(index=False))
            return True
        else:
            print("WARNING: No data found for BTC/USDT 1h")
            return False
            
    except Exception as e:
        print(f"ERROR: Data loading failed: {e}")
        return False

def test_all_timeframes():
    """Test data availability for all timeframes"""
    print("\nTESTING: All timeframes data...")
    
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    timeframes = ['5m', '15m', '1h', '4h', '1d']
    
    try:
        conn = sqlite3.connect('data/multi_timeframe_data.db')
        
        print("\nDATA AVAILABILITY CHECK:")
        print("=" * 60)
        print(f"{'Symbol':<10} {'Timeframe':<10} {'Records':<10} {'Status'}")
        print("-" * 60)
        
        total_available = 0
        
        for symbol in symbols:
            for timeframe in timeframes:
                query = """
                    SELECT COUNT(*) as count
                    FROM price_data 
                    WHERE symbol = ? AND timeframe = ?
                """
                
                result = pd.read_sql_query(query, conn, params=[symbol, timeframe])
                count = result['count'].iloc[0]
                
                status = "OK" if count > 50 else "LOW" if count > 0 else "MISSING"
                print(f"{symbol:<10} {timeframe:<10} {count:<10} {status}")
                
                if count > 50:
                    total_available += 1
        
        conn.close()
        
        print(f"\nSUMMARY: {total_available}/{len(symbols)*len(timeframes)} timeframes have sufficient data")
        
        return total_available > 10  # Need at least 10 good timeframes
        
    except Exception as e:
        print(f"ERROR: Timeframe check failed: {e}")
        return False

def create_simple_analyzer():
    """Create a simplified analyzer that works with your data"""
    analyzer_code = '''
"""
Simple Multi-Timeframe Analyzer
Uses data from multi_timeframe_collector database
"""

import pandas as pd
import sqlite3
from datetime import datetime

class SimpleMultiAnalyzer:
    def __init__(self):
        self.db_path = 'data/multi_timeframe_data.db'
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        self.timeframes = ['5m', '15m', '1h', '4h', '1d']
    
    def get_data(self, symbol, timeframe, limit=100):
        """Get data for specific symbol and timeframe"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM price_data 
            WHERE symbol = ? AND timeframe = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """
        
        df = pd.read_sql_query(query, conn, params=[symbol, timeframe, limit])
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not rsi.empty else 50
    
    def analyze_symbol(self, symbol):
        """Analyze a symbol across multiple timeframes"""
        print(f"\\nANALYZING: {symbol}")
        print("-" * 40)
        
        timeframe_scores = {}
        
        for timeframe in self.timeframes:
            df = self.get_data(symbol, timeframe, 50)
            
            if len(df) > 20:
                current_price = df['close'].iloc[-1]
                rsi = self.calculate_rsi(df['close'])
                
                # Simple scoring based on RSI
                if rsi < 30:
                    score = 80  # Oversold - Buy signal
                elif rsi > 70:
                    score = 20  # Overbought - Sell signal
                else:
                    score = 50  # Neutral
                
                timeframe_scores[timeframe] = {
                    'price': current_price,
                    'rsi': rsi,
                    'score': score,
                    'records': len(df)
                }
                
                print(f"{timeframe:3}: Price=${current_price:8.2f} RSI={rsi:5.1f} Score={score:3.0f} Records={len(df):3d}")
            else:
                print(f"{timeframe:3}: Insufficient data ({len(df)} records)")
        
        # Calculate combined score
        if timeframe_scores:
            total_score = sum(data['score'] for data in timeframe_scores.values())
            avg_score = total_score / len(timeframe_scores)
            
            if avg_score >= 70:
                signal = "BUY"
            elif avg_score <= 30:
                signal = "SELL"
            else:
                signal = "HOLD"
            
            print(f"COMBINED: Score={avg_score:.1f} Signal={signal}")
            
            return {
                'symbol': symbol,
                'signal': signal,
                'score': avg_score,
                'timeframes': timeframe_scores
            }
        
        return None
    
    def analyze_all(self):
        """Analyze all symbols"""
        print("MULTI-TIMEFRAME ANALYSIS")
        print("=" * 50)
        
        results = {}
        
        for symbol in self.symbols:
            result = self.analyze_symbol(symbol)
            if result:
                results[symbol] = result
        
        # Summary
        print(f"\\nSUMMARY:")
        print("-" * 30)
        for symbol, result in results.items():
            signal = result['signal']
            score = result['score']
            print(f"{symbol}: {signal} (Score: {score:.1f})")
        
        return results

if __name__ == "__main__":
    analyzer = SimpleMultiAnalyzer()
    results = analyzer.analyze_all()
'''
    
    with open('simple_multi_analyzer.py', 'w', encoding='utf-8') as f:
        f.write(analyzer_code)
    
    print("SUCCESS: Created simple_multi_analyzer.py")

def main():
    """Main test function"""
    print("MULTI-TIMEFRAME DATA CONNECTION TEST")
    print("=" * 50)
    
    # Test 1: Database connection
    db_ok = test_database_connection()
    
    if not db_ok:
        print("\nERROR: Database not found!")
        print("SOLUTION: Run 'python multi_timeframe_collector.py' first")
        return
    
    # Test 2: Data loading
    data_ok = test_data_loading()
    
    # Test 3: All timeframes
    timeframes_ok = test_all_timeframes()
    
    # Create simple analyzer
    create_simple_analyzer()
    
    print(f"\nTEST RESULTS:")
    print("=" * 30)
    print(f"Database Connection: {'OK' if db_ok else 'FAILED'}")
    print(f"Data Loading: {'OK' if data_ok else 'FAILED'}")
    print(f"Timeframes Available: {'OK' if timeframes_ok else 'LIMITED'}")
    
    if db_ok and data_ok:
        print(f"\nNEXT STEPS:")
        print("1. Run: python simple_multi_analyzer.py")
        print("2. Or use the full multi_timeframe_analyzer.py")
        print("3. Data is ready for analysis!")
    else:
        print(f"\nTROUBLESHOOTING:")
        print("1. Make sure multi_timeframe_collector.py ran successfully")
        print("2. Check if data/multi_timeframe_data.db exists")
        print("3. Re-run the collector if needed")

if __name__ == "__main__":
    main()