"""
Integrated Ultimate Multi-Timeframe Signal Analyzer
Combines Multi-Timeframe Analysis + Ultimate Signal Combiner for Maximum Accuracy
"""
import sys
import os

# Fix Windows encoding issues with emojis
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, OSError):
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import warnings
import logging
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no charts)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ultimate_analyzer.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class DatabaseManager:
    def __init__(self, db_path='data/multi_timeframe_data.db'):
        """Initialize database connection"""
        self.db_path = db_path
    
    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def get_available_symbols(self):
        """Get list of available symbols from database"""
        try:
            with self.get_connection() as conn:
                query = "SELECT DISTINCT symbol FROM price_data ORDER BY symbol"
                symbols = pd.read_sql_query(query, conn)['symbol'].tolist()
                return symbols
        except Exception as e:
            print(f"❌ Error getting symbols: {e}")
            return []
    
    def get_available_timeframes(self, symbol=None):
        """Get available timeframes"""
        try:
            with self.get_connection() as conn:
                if symbol:
                    query = "SELECT DISTINCT timeframe FROM price_data WHERE symbol = ? ORDER BY timeframe"
                    params = (symbol,)
                else:
                    query = "SELECT DISTINCT timeframe FROM price_data ORDER BY timeframe"
                    params = ()
                
                timeframes = pd.read_sql_query(query, conn, params=params)['timeframe'].tolist()
                return timeframes
        except Exception as e:
            print(f"❌ Error getting timeframes: {e}")
            return []
    
    def load_crypto_data(self, symbol, timeframe='1h', limit_hours=168):
        """Load crypto data from database"""
        try:
            with self.get_connection() as conn:
                # Calculate the timestamp limit using datetime
                hours_ago = datetime.now() - timedelta(hours=limit_hours)
                
                query = """
                SELECT timestamp, open, high, low, close, volume
                FROM price_data 
                WHERE symbol = ? AND timeframe = ? AND timestamp >= ?
                ORDER BY timestamp ASC
                """
                
                df = pd.read_sql_query(
                    query, 
                    conn, 
                    params=(symbol, timeframe, hours_ago.strftime('%Y-%m-%d %H:%M:%S'))
                )
                
                if df.empty:
                    return None
                
                # Convert timestamp to datetime (already in datetime format from DB)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Ensure proper data types
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                return df
                
        except Exception as e:
            logging.error(f"Error loading data for {symbol}: {e}")
            return None

class UltimateSignalCombiner:
    def __init__(self):
        """Initialize the Ultimate Signal Combiner"""
        self.confidence_threshold_buy = 60
        self.confidence_threshold_sell = 60
        self.strong_threshold = 80
        
        # Signal weights
        self.weights = {
            'trade_bulls': 30,
            'rsi': 20,
            'macd': 20,
            'bollinger': 15,
            'volume': 15
        }
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        macd_histogram = macd_line - macd_signal
        
        return {
            'macd_line': macd_line,
            'macd_signal': macd_signal,
            'macd_histogram': macd_histogram
        }
    
    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }
    
    def calculate_trade_bulls_score(self, df):
        """Calculate Trade Bulls strategy score"""
        scores = pd.Series(50, index=df.index)
        
        if 'signal' in df.columns:
            scores[df['signal'] == 'STRONG_BUY'] = 100
            scores[df['signal'] == 'BUY'] = 75
            scores[df['signal'] == 'SELL'] = 25
            scores[df['signal'] == 'STRONG_SELL'] = 0
            scores[df['signal'] == 'HOLD'] = 50
        else:
            # Use price momentum as proxy
            df['price_change_5'] = df['close'].pct_change(5) * 100
            scores[df['price_change_5'] > 2] = 75
            scores[(df['price_change_5'] > 0) & (df['price_change_5'] <= 2)] = 60
            scores[(df['price_change_5'] < 0) & (df['price_change_5'] >= -2)] = 40
            scores[df['price_change_5'] < -2] = 25
        
        return scores
    
    def calculate_rsi_score(self, df):
        """Calculate RSI-based score (0-100)"""
        scores = pd.Series(50, index=df.index)
        
        if 'rsi' not in df.columns:
            df['rsi'] = self.calculate_rsi(df['close'])
        
        if 'rsi' in df.columns:
            rsi = df['rsi']
            scores[rsi <= 20] = 100
            scores[(rsi > 20) & (rsi <= 30)] = 80
            scores[(rsi > 30) & (rsi <= 40)] = 65
            scores[(rsi > 40) & (rsi <= 60)] = 50
            scores[(rsi > 60) & (rsi <= 70)] = 35
            scores[(rsi > 70) & (rsi <= 80)] = 20
            scores[rsi > 80] = 0
        
        return scores
    
    def calculate_macd_score(self, df):
        """Calculate MACD-based score (0-100)"""
        scores = pd.Series(50, index=df.index)
        
        if not all(col in df.columns for col in ['macd_line', 'macd_signal_line', 'macd_histogram']):
            macd_data = self.calculate_macd(df['close'])
            df['macd_line'] = macd_data['macd_line']
            df['macd_signal_line'] = macd_data['macd_signal']
            df['macd_histogram'] = macd_data['macd_histogram']
        
        if all(col in df.columns for col in ['macd_line', 'macd_signal_line', 'macd_histogram']):
            macd_line = df['macd_line']
            signal_line = df['macd_signal_line']
            histogram = df['macd_histogram']
            
            macd_bullish = macd_line > signal_line
            histogram_increasing = histogram > histogram.shift(1)
            macd_cross_up = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
            macd_cross_down = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
            macd_above_zero = macd_line > 0
            
            scores[macd_cross_up & macd_above_zero] = 90
            scores[macd_cross_up & ~macd_above_zero] = 75
            scores[macd_bullish & histogram_increasing & macd_above_zero] = 70
            scores[macd_bullish & macd_above_zero] = 60
            scores[macd_bullish & ~macd_above_zero] = 55
            
            scores[macd_cross_down & ~macd_above_zero] = 10
            scores[macd_cross_down & macd_above_zero] = 25
            scores[~macd_bullish & ~histogram_increasing & ~macd_above_zero] = 30
            scores[~macd_bullish & ~macd_above_zero] = 40
            scores[~macd_bullish & macd_above_zero] = 45
        
        return scores
    
    def calculate_bollinger_score(self, df):
        """Calculate Bollinger Bands score (0-100)"""
        scores = pd.Series(50, index=df.index)
        
        if not all(col in df.columns for col in ['bb_upper', 'bb_lower', 'bb_position', 'bb_width']):
            bb_data = self.calculate_bollinger_bands(df['close'])
            df['bb_upper'] = bb_data['upper']
            df['bb_lower'] = bb_data['lower']
            df['bb_middle'] = bb_data['middle']
            df['bb_position'] = ((df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])) * 100
            df['bb_width'] = ((df['bb_upper'] - df['bb_lower']) / df['bb_middle']) * 100
        
        if all(col in df.columns for col in ['bb_upper', 'bb_lower', 'bb_position', 'bb_width']):
            bb_position = df['bb_position']
            bb_width = df['bb_width']
            
            squeeze = bb_width < bb_width.rolling(20).mean() * 0.8
            
            scores[bb_position <= 5] = 90
            scores[(bb_position > 5) & (bb_position <= 20)] = 75
            scores[(bb_position > 20) & (bb_position <= 40)] = 60
            scores[(bb_position > 40) & (bb_position <= 60)] = 50
            scores[(bb_position > 60) & (bb_position <= 80)] = 40
            scores[(bb_position > 80) & (bb_position <= 95)] = 25
            scores[bb_position > 95] = 10
            
            squeeze_breakout_up = squeeze.shift(1) & (bb_position > 80)
            squeeze_breakout_down = squeeze.shift(1) & (bb_position < 20)
            
            scores[squeeze_breakout_up] = 95
            scores[squeeze_breakout_down] = 5
        
        return scores
    
    def calculate_volume_score(self, df):
        """Calculate volume confirmation score (0-100)"""
        scores = pd.Series(50, index=df.index)
        
        if 'volume' in df.columns:
            volume = df['volume']
            price_change = df['close'].pct_change()
            
            volume_ma_long = volume.rolling(30).mean()
            volume_ratio = volume / volume_ma_long
            
            price_up = price_change > 0
            price_down = price_change < 0
            high_volume = volume_ratio > 1.5
            very_high_volume = volume_ratio > 2.0
            low_volume = volume_ratio < 0.7
            
            scores[price_up & very_high_volume] = 90
            scores[price_up & high_volume] = 75
            scores[price_up & ~low_volume] = 60
            scores[price_up & low_volume] = 45
            
            scores[price_down & very_high_volume] = 10
            scores[price_down & high_volume] = 25
            scores[price_down & ~low_volume] = 40
            scores[price_down & low_volume] = 55
            
            scores[(abs(price_change) < 0.005)] = 50
        
        return scores
    
    def combine_all_signals(self, df):
        """Combine all signal sources into ultimate signal"""
        # Calculate individual scores
        trade_bulls_score = self.calculate_trade_bulls_score(df)
        rsi_score = self.calculate_rsi_score(df)
        macd_score = self.calculate_macd_score(df)
        bollinger_score = self.calculate_bollinger_score(df)
        volume_score = self.calculate_volume_score(df)
        
        # Store individual scores
        df['score_trade_bulls'] = trade_bulls_score
        df['score_rsi'] = rsi_score
        df['score_macd'] = macd_score
        df['score_bollinger'] = bollinger_score
        df['score_volume'] = volume_score
        
        # Calculate weighted combined score
        combined_score = (
            trade_bulls_score * self.weights['trade_bulls'] / 100 +
            rsi_score * self.weights['rsi'] / 100 +
            macd_score * self.weights['macd'] / 100 +
            bollinger_score * self.weights['bollinger'] / 100 +
            volume_score * self.weights['volume'] / 100
        )
        
        df['combined_score'] = combined_score
        
        # Generate ultimate signals
        ultimate_signals = pd.Series('HOLD', index=df.index)
        ultimate_signals[combined_score >= self.strong_threshold] = 'STRONG_BUY'
        ultimate_signals[(combined_score >= self.confidence_threshold_buy) & 
                        (combined_score < self.strong_threshold)] = 'BUY'
        ultimate_signals[(combined_score <= (100 - self.confidence_threshold_sell)) & 
                        (combined_score > (100 - self.strong_threshold))] = 'SELL'
        ultimate_signals[combined_score <= (100 - self.strong_threshold)] = 'STRONG_SELL'
        
        df['ultimate_signal'] = ultimate_signals
        
        # Calculate confidence
        df['confidence'] = np.where(
            combined_score >= 50,
            combined_score,
            100 - combined_score
        )
        
        return df

class IntegratedUltimateAnalyzer:
    def __init__(self, db_path='data/multi_timeframe_data.db'):
        """Initialize the integrated analyzer"""
        self.db_manager = DatabaseManager(db_path)
        self.ultimate_combiner = UltimateSignalCombiner()
        
        # Timeframe weights for multi-timeframe combination
        self.timeframe_weights = {
            '5m': 10,   # Short-term precision
            '15m': 20,  # Entry timing
            '1h': 30,   # Main trading timeframe
            '4h': 25,   # Trend confirmation
            '1d': 15    # Major trend context
        }
        
        # Multi-timeframe thresholds
        self.mtf_strong_buy_threshold = 80
        self.mtf_buy_threshold = 65
        self.mtf_sell_threshold = 35
        self.mtf_strong_sell_threshold = 20
        
        logging.info("🚀 Integrated Ultimate Analyzer initialized")
    
    def analyze_timeframe(self, symbol, timeframe):
        """Analyze signals for a specific symbol and timeframe"""
        try:
            # Load data for this timeframe
            df = self.db_manager.load_crypto_data(symbol, timeframe, limit_hours=168)
            
            if df is None or df.empty or len(df) < 50:
                logging.warning(f"Insufficient data for {symbol} {timeframe}")
                return None
            
            # Apply Ultimate Signal Combiner
            df = self.ultimate_combiner.combine_all_signals(df)
            
            # Get latest values
            latest = df.iloc[-1]
            
            return {
                'timeframe': timeframe,
                'symbol': symbol,
                'timestamp': latest['timestamp'],
                'price': latest['close'],
                'ultimate_signal': latest['ultimate_signal'],
                'combined_score': latest['combined_score'],
                'confidence': latest['confidence'],
                'rsi': latest.get('rsi', 50),
                'macd_line': latest.get('macd_line', 0),
                'bb_position': latest.get('bb_position', 50),
                'volume_ratio': latest.get('volume', 1) / df['volume'].rolling(20).mean().iloc[-1] if 'volume' in df.columns else 1,
                'individual_scores': {
                    'trade_bulls': latest['score_trade_bulls'],
                    'rsi': latest['score_rsi'],
                    'macd': latest['score_macd'],
                    'bollinger': latest['score_bollinger'],
                    'volume': latest['score_volume']
                }
            }
            
        except Exception as e:
            logging.error(f"Error analyzing {symbol} {timeframe}: {e}")
            return None
    
    def combine_timeframe_signals(self, timeframe_results):
        """Combine Ultimate Signals from multiple timeframes"""
        if not timeframe_results:
            return None
        
        # Calculate weighted score using combined_score from each timeframe
        total_weight = 0
        weighted_score = 0
        
        for tf_result in timeframe_results:
            timeframe = tf_result['timeframe']
            weight = self.timeframe_weights.get(timeframe, 20)
            score = tf_result['combined_score']
            
            weighted_score += score * weight
            total_weight += weight
        
        mtf_combined_score = weighted_score / total_weight if total_weight > 0 else 50
        
        # Determine multi-timeframe signal
        if mtf_combined_score >= self.mtf_strong_buy_threshold:
            mtf_signal = 'STRONG_BUY'
        elif mtf_combined_score >= self.mtf_buy_threshold:
            mtf_signal = 'BUY'
        elif mtf_combined_score <= self.mtf_strong_sell_threshold:
            mtf_signal = 'STRONG_SELL'
        elif mtf_combined_score <= self.mtf_sell_threshold:
            mtf_signal = 'SELL'
        else:
            mtf_signal = 'HOLD'
        
        # Calculate multi-timeframe confidence
        mtf_confidence = max(mtf_combined_score, 100 - mtf_combined_score)
        
        # Signal consensus analysis
        signals = [result['ultimate_signal'] for result in timeframe_results]
        signal_counts = {signal: signals.count(signal) for signal in set(signals)}
        dominant_signal = max(signal_counts, key=signal_counts.get)
        signal_consensus = signal_counts[dominant_signal] / len(signals) * 100
        
        # Get latest price (use shortest timeframe for most recent)
        latest_price = min(timeframe_results, key=lambda x: x['timeframe'])['price']
        
        return {
            'multi_timeframe_signal': mtf_signal,
            'mtf_combined_score': mtf_combined_score,
            'mtf_confidence': mtf_confidence,
            'price': latest_price,
            'dominant_signal': dominant_signal,
            'signal_consensus': signal_consensus,
            'timeframe_breakdown': {
                result['timeframe']: {
                    'signal': result['ultimate_signal'],
                    'score': result['combined_score'],
                    'confidence': result['confidence']
                } for result in timeframe_results
            },
            'timestamp': datetime.now()
        }
    
    def analyze_symbol_all_timeframes(self, symbol):
        """Analyze a symbol across all available timeframes"""
        logging.info(f"🔍 Analyzing {symbol} across all timeframes")
        
        available_timeframes = self.db_manager.get_available_timeframes(symbol)
        timeframe_results = []
        
        for timeframe in available_timeframes:
            if timeframe in self.timeframe_weights:  # Only analyze configured timeframes
                result = self.analyze_timeframe(symbol, timeframe)
                if result:
                    timeframe_results.append(result)
                    logging.info(f"✅ {symbol} {timeframe}: {result['ultimate_signal']} "
                               f"(Score: {result['combined_score']:.1f})")
        
        if timeframe_results:
            combined_result = self.combine_timeframe_signals(timeframe_results)
            if combined_result:
                combined_result['symbol'] = symbol
                combined_result['timeframe_results'] = timeframe_results
                
                logging.info(f"🎯 {symbol} Multi-Timeframe: {combined_result['multi_timeframe_signal']} "
                           f"({combined_result['mtf_confidence']:.1f}%)")
                
                return combined_result
        
        logging.warning(f"⚠️ No valid timeframe results for {symbol}")
        return None
    
    def analyze_all_symbols(self):
        """Analyze all available symbols"""
        logging.info("🔍 Starting integrated multi-timeframe ultimate analysis")
        
        symbols = self.db_manager.get_available_symbols()
        if not symbols:
            logging.error("No symbols found in database")
            return {}
        
        results = {}
        
        for symbol in symbols:
            try:
                result = self.analyze_symbol_all_timeframes(symbol)
                if result:
                    results[symbol] = result
            except Exception as e:
                logging.error(f"❌ Error analyzing {symbol}: {e}")
        
        logging.info(f"✅ Integrated analysis complete: {len(results)} symbols")
        return results
    
    def display_ultimate_results(self, results):
        """Display comprehensive ultimate analysis results"""
        print(f"\n🚀 INTEGRATED ULTIMATE MULTI-TIMEFRAME ANALYSIS")
        print("=" * 70)
        print(f"⏰ Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🔥 Ultimate Signals + Multi-Timeframe Confirmation")
        print("=" * 70)
        
        if not results:
            print("❌ No analysis results available")
            return
        
        for symbol, result in results.items():
            mtf_signal = result['multi_timeframe_signal']
            mtf_confidence = result['mtf_confidence']
            mtf_score = result['mtf_combined_score']
            price = result['price']
            dominant_signal = result['dominant_signal']
            consensus = result['signal_consensus']
            
            # Signal emoji
            signal_emoji = {
                'STRONG_BUY': '🟢🟢',
                'BUY': '🟢',
                'HOLD': '🟡',
                'SELL': '🔴',
                'STRONG_SELL': '🔴🔴'
            }.get(mtf_signal, '⚪')
            
            print(f"\n📊 {symbol}")
            print(f"   {signal_emoji} Multi-Timeframe Signal: {mtf_signal}")
            print(f"   🎯 MTF Confidence: {mtf_confidence:.1f}%")
            print(f"   📈 MTF Score: {mtf_score:.1f}/100")
            print(f"   💰 Price: ${price:,.2f}")
            print(f"   🏆 Dominant Signal: {dominant_signal}")
            print(f"   🤝 Signal Consensus: {consensus:.1f}%")
            
            # Timeframe breakdown
            print(f"   🔬 Timeframe Analysis:")
            breakdown = result['timeframe_breakdown']
            for tf in ['5m', '15m', '1h', '4h', '1d']:
                if tf in breakdown:
                    tf_data = breakdown[tf]
                    weight = self.timeframe_weights.get(tf, 0)
                    tf_emoji = {
                        'STRONG_BUY': '🟢🟢',
                        'BUY': '🟢',
                        'HOLD': '🟡',
                        'SELL': '🔴',
                        'STRONG_SELL': '🔴🔴'
                    }.get(tf_data['signal'], '⚪')
                    
                    print(f"      {tf:3}: {tf_emoji} {tf_data['signal']:10} "
                          f"Score: {tf_data['score']:5.1f} "
                          f"Conf: {tf_data['confidence']:5.1f}% "
                          f"(Weight: {weight:2d}%)")
        
        # Market summary
        signals = [result['multi_timeframe_signal'] for result in results.values()]
        buy_signals = sum(1 for s in signals if s in ['BUY', 'STRONG_BUY'])
        sell_signals = sum(1 for s in signals if s in ['SELL', 'STRONG_SELL'])
        hold_signals = sum(1 for s in signals if s == 'HOLD')
        
        print(f"\n🌍 ULTIMATE MARKET SUMMARY:")
        print(f"   🟢 Buy Signals: {buy_signals}")
        print(f"   🔴 Sell Signals: {sell_signals}")
        print(f"   🟡 Hold Signals: {hold_signals}")
        
        if buy_signals > sell_signals:
            sentiment = "📈 BULLISH"
        elif sell_signals > buy_signals:
            sentiment = "📉 BEARISH"
        else:
            sentiment = "⚖️ NEUTRAL"
        
        print(f"   🌍 Overall Sentiment: {sentiment}")
        
        # Best opportunities
        print(f"\n🎯 TOP OPPORTUNITIES:")
        
        buy_opportunities = [(symbol, result) for symbol, result in results.items() 
                           if result['multi_timeframe_signal'] in ['STRONG_BUY', 'BUY']]
        sell_opportunities = [(symbol, result) for symbol, result in results.items() 
                            if result['multi_timeframe_signal'] in ['STRONG_SELL', 'SELL']]
        
        if buy_opportunities:
            best_buy = max(buy_opportunities, key=lambda x: x[1]['mtf_confidence'])
            print(f"   🟢 Best BUY: {best_buy[0]} - {best_buy[1]['multi_timeframe_signal']} "
                  f"({best_buy[1]['mtf_confidence']:.1f}% confidence)")
        
        if sell_opportunities:
            best_sell = max(sell_opportunities, key=lambda x: x[1]['mtf_confidence'])
            print(f"   🔴 Best SELL: {best_sell[0]} - {best_sell[1]['multi_timeframe_signal']} "
                  f"({best_sell[1]['mtf_confidence']:.1f}% confidence)")
        
        if not buy_opportunities and not sell_opportunities:
            print(f"   ⏳ All markets in HOLD - Wait for clearer signals")

def main():
    """Main function"""
    print("🚀 INTEGRATED ULTIMATE MULTI-TIMEFRAME SIGNAL ANALYZER")
    print("=" * 60)
    print("🔥 Ultimate Signals + Multi-Timeframe Analysis = Maximum Accuracy")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = IntegratedUltimateAnalyzer()
    
    # Show database status
    symbols = analyzer.db_manager.get_available_symbols()
    timeframes = analyzer.db_manager.get_available_timeframes()
    
    if not symbols:
        print("❌ No data found in database!")
        print("💡 Run the multi-timeframe data collector first!")
        return
    
    print(f"📊 Database contains {len(symbols)} symbols and {len(timeframes)} timeframes")
    print(f"💡 Available symbols: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")
    print(f"⏱️  Available timeframes: {', '.join(timeframes)}")
    
    # Analyze all symbols
    results = analyzer.analyze_all_symbols()
    
    # Display results
    if results:
        analyzer.display_ultimate_results(results)
    else:
        print("❌ No analysis results available")
        print("💡 Make sure your database has sufficient data!")
    
    print(f"\n🎉 Integrated Ultimate Analysis Complete!")

if __name__ == "__main__":
    main()