"""
Ultimate Crypto Trading Signals - Combined System (NO CHARTS VERSION)
Merges Trade Bulls Strategy + Technical Indicators for Maximum Accuracy
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Disable matplotlib to prevent charts
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

class UltimateSignalCombiner:
    def __init__(self):
        """Initialize the Ultimate Signal Combiner"""
        self.confidence_threshold_buy = 60  # 60% confidence to buy
        self.confidence_threshold_sell = 60  # 60% confidence to sell
        self.strong_threshold = 80  # 80% for strong signals
        
        # Signal weights (how much each system contributes)
        self.weights = {
            'trade_bulls': 30,      # Trade Bulls strategy
            'rsi': 20,              # RSI signals
            'macd': 20,             # MACD signals
            'bollinger': 15,        # Bollinger Bands
            'volume': 15            # Volume confirmation
        }
        
        # Signal storage
        self.signal_history = []
        self.alerts = []
    
    def calculate_trade_bulls_score(self, df):
        """Calculate Trade Bulls strategy score"""
        scores = pd.Series(0, index=df.index)
        
        if 'signal' in df.columns:
            scores[df['signal'] == 'STRONG_BUY'] = 100
            scores[df['signal'] == 'BUY'] = 75
            scores[df['signal'] == 'SELL'] = 25
            scores[df['signal'] == 'STRONG_SELL'] = 0
            scores[df['signal'] == 'HOLD'] = 50
        else:
            df['price_change_5'] = df['close'].pct_change(5) * 100
            scores[df['price_change_5'] > 2] = 75
            scores[(df['price_change_5'] > 0) & (df['price_change_5'] <= 2)] = 60
            scores[(df['price_change_5'] < 0) & (df['price_change_5'] >= -2)] = 40
            scores[df['price_change_5'] < -2] = 25
        
        return scores
    
    def calculate_rsi_score(self, df):
        """Calculate RSI-based score (0-100)"""
        scores = pd.Series(50, index=df.index)
        
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
        print("🔄 Combining all signal sources...")
        
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
        
        print("✅ Signal combination complete!")
        return df
    
    def generate_trading_alerts(self, df):
        """Generate actionable trading alerts"""
        alerts = []
        
        for i in range(1, len(df)):
            current_signal = df['ultimate_signal'].iloc[i]
            prev_signal = df['ultimate_signal'].iloc[i-1]
            confidence = df['confidence'].iloc[i]
            price = df['close'].iloc[i]
            timestamp = df['timestamp'].iloc[i]
            
            if current_signal != prev_signal and current_signal != 'HOLD':
                alert = {
                    'timestamp': timestamp,
                    'signal': current_signal,
                    'price': price,
                    'confidence': confidence,
                    'type': 'SIGNAL_CHANGE',
                    'prev_signal': prev_signal
                }
                
                if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
                    alert['resistance'] = df['bb_upper'].iloc[i]
                    alert['support'] = df['bb_lower'].iloc[i]
                
                alerts.append(alert)
            
            elif confidence >= 85 and current_signal in ['STRONG_BUY', 'STRONG_SELL']:
                alerts.append({
                    'timestamp': timestamp,
                    'signal': current_signal,
                    'price': price,
                    'confidence': confidence,
                    'type': 'HIGH_CONFIDENCE',
                    'message': f'Very high confidence {current_signal.lower()} signal'
                })
        
        return alerts
    
    def generate_ultimate_report(self, df, symbol='BTC_USDT'):
        """Generate comprehensive ultimate trading report"""
        print(f"\n🚀 ULTIMATE CRYPTO SIGNALS REPORT - {symbol}")
        print("=" * 70)
        
        current_price = df['close'].iloc[-1]
        current_signal = df['ultimate_signal'].iloc[-1]
        combined_score = df['combined_score'].iloc[-1]
        confidence = df['confidence'].iloc[-1]
        
        # Signal strength emoji
        signal_emojis = {
            'STRONG_BUY': '🟢🟢',
            'BUY': '🟢',
            'SELL': '🔴',
            'STRONG_SELL': '🔴🔴',
            'HOLD': '🟡'
        }
        signal_emoji = signal_emojis.get(current_signal, '🟡')
        
        print(f"💰 Current Price: ${current_price:,.2f}")
        print(f"🎯 Ultimate Signal: {signal_emoji} {current_signal}")
        print(f"📊 Combined Score: {combined_score:.1f}/100")
        print(f"🔥 Confidence: {confidence:.1f}%")
        
        # Individual component analysis
        print(f"\n🔬 SIGNAL BREAKDOWN:")
        components = [
            ('Trade Bulls', df['score_trade_bulls'].iloc[-1], self.weights['trade_bulls']),
            ('RSI', df['score_rsi'].iloc[-1], self.weights['rsi']),
            ('MACD', df['score_macd'].iloc[-1], self.weights['macd']),
            ('Bollinger Bands', df['score_bollinger'].iloc[-1], self.weights['bollinger']),
            ('Volume', df['score_volume'].iloc[-1], self.weights['volume'])
        ]
        
        for name, score, weight in components:
            contribution = score * weight / 100
            status = "🟢" if score >= 60 else "🔴" if score <= 40 else "🟡"
            print(f"   {status} {name:15}: {score:5.1f}/100 (Weight: {weight:2d}%) → {contribution:5.1f}")
        
        # Recent signal history
        print(f"\n📈 RECENT SIGNAL HISTORY (Last 24 Hours):")
        recent_signals = df['ultimate_signal'].tail(24)
        signal_changes = []
        
        for i in range(1, len(recent_signals)):
            if recent_signals.iloc[i] != recent_signals.iloc[i-1]:
                timestamp = df['timestamp'].iloc[-24+i]
                signal = recent_signals.iloc[i]
                price = df['close'].iloc[-24+i]
                signal_changes.append((timestamp, signal, price))
        
        if signal_changes:
            for timestamp, signal, price in signal_changes[-5:]:
                print(f"   📅 {timestamp}: {signal} at ${price:,.2f}")
        else:
            print(f"   ⚪ No signal changes - steady {current_signal}")
        
        # Performance metrics
        print(f"\n📊 SIGNAL PERFORMANCE:")
        total_signals = len(df[df['ultimate_signal'] != 'HOLD'])
        buy_signals = len(df[df['ultimate_signal'].isin(['BUY', 'STRONG_BUY'])])
        sell_signals = len(df[df['ultimate_signal'].isin(['SELL', 'STRONG_SELL'])])
        
        print(f"   Total Active Signals: {total_signals}")
        print(f"   Buy Signals: {buy_signals} ({buy_signals/len(df)*100:.1f}%)")
        print(f"   Sell Signals: {sell_signals} ({sell_signals/len(df)*100:.1f}%)")
        print(f"   Average Confidence: {df['confidence'].mean():.1f}%")
        
        # Trading recommendations
        print(f"\n💡 ULTIMATE TRADING RECOMMENDATIONS:")
        
        if current_signal in ['STRONG_BUY', 'BUY']:
            print(f"   ✅ {current_signal}: Multiple systems align for buying")
            print(f"   🎯 Entry Zone: ${current_price:,.2f}")
            
            if 'bb_lower' in df.columns:
                stop_loss = df['bb_lower'].iloc[-1]
                take_profit = df['bb_upper'].iloc[-1]
                print(f"   🛑 Stop Loss: ${stop_loss:,.2f} ({((current_price-stop_loss)/current_price*100):.1f}% risk)")
                print(f"   🚀 Take Profit: ${take_profit:,.2f} ({((take_profit-current_price)/current_price*100):.1f}% target)")
            
            position_size = "Large" if confidence >= 80 else "Medium" if confidence >= 65 else "Small"
            print(f"   📊 Suggested Position: {position_size} ({confidence:.0f}% confidence)")
        
        elif current_signal in ['STRONG_SELL', 'SELL']:
            print(f"   ❌ {current_signal}: Multiple systems align for selling")
            print(f"   🎯 Exit Zone: ${current_price:,.2f}")
            
            if 'bb_upper' in df.columns:
                stop_loss = df['bb_upper'].iloc[-1]
                take_profit = df['bb_lower'].iloc[-1]
                print(f"   🛑 Stop Loss: ${stop_loss:,.2f}")
                print(f"   🚀 Target: ${take_profit:,.2f}")
        
        else:
            print(f"   ⏳ HOLD: Wait for stronger signal alignment")
            print(f"   👀 Watch for score above {self.confidence_threshold_buy} (BUY) or below {100-self.confidence_threshold_sell} (SELL)")
        
        # Risk assessment
        print(f"\n⚠️ RISK ASSESSMENT:")
        risk_level = "HIGH" if confidence < 60 else "MEDIUM" if confidence < 80 else "LOW"
        print(f"   🎲 Signal Risk: {risk_level}")
        print(f"   📈 Trend Strength: {confidence:.0f}%")
        
        if 'bb_width' in df.columns:
            volatility = df['bb_width'].iloc[-1]
            vol_status = "HIGH" if volatility > 5 else "MEDIUM" if volatility > 2 else "LOW"
            print(f"   🌊 Volatility: {vol_status} ({volatility:.2f}%)")

def load_and_analyze_ultimate_signals(symbol='BTC_USDT'):
    """Load data and generate ultimate signals - NO CHARTS VERSION"""
    filename = f"data/{symbol}_1h_7days.csv"
    
    try:
        df = pd.read_csv(filename)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"✅ Loaded {len(df)} records for {symbol}")
        
        # Try to add existing analysis if available
        try:
            from trade_bulls_strategy import TradeBullsStrategy
            trade_bulls = TradeBullsStrategy()
            df, _ = trade_bulls.analyze_crypto(df)
            print("✅ Trade Bulls analysis integrated")
        except:
            print("⚠️ Trade Bulls strategy not available, using basic signals")
        
        try:
            from technical_indicators import TechnicalIndicators
            tech_indicators = TechnicalIndicators()
            df = tech_indicators.analyze_crypto_indicators(df)
            print("✅ Technical indicators integrated")
        except:
            print("⚠️ Technical indicators not available, using basic analysis")
        
        # Apply Ultimate Signal Combiner
        ultimate = UltimateSignalCombiner()
        df = ultimate.combine_all_signals(df)
        
        # Generate alerts
        alerts = ultimate.generate_trading_alerts(df)
        
        # Generate report (NO CHARTS)
        ultimate.generate_ultimate_report(df, symbol)
        
        # Display recent alerts
        if alerts:
            print(f"\n🔔 RECENT TRADING ALERTS:")
            for alert in alerts[-3:]:
                timestamp = alert['timestamp'].strftime('%Y-%m-%d %H:%M')
                print(f"   📅 {timestamp}: {alert['signal']} at ${alert['price']:.2f} (Confidence: {alert['confidence']:.1f}%)")
        
        return df, alerts, ultimate
        
    except FileNotFoundError:
        print(f"❌ File not found: {filename}")
        print("💡 Run crypto_data_collector.py first!")
        return None, None, None
    except Exception as e:
        print(f"❌ Error analyzing {symbol}: {e}")
        return None, None, None

def compare_ultimate_signals():
    """Compare ultimate signals across all cryptocurrencies - NO CHARTS"""
    print(f"\n🔥 ULTIMATE SIGNALS COMPARISON")
    print("=" * 70)
    
    symbols = ['BTC_USDT', 'ETH_USDT', 'BNB_USDT']
    results = []
    
    for symbol in symbols:
        print(f"\n📊 Analyzing {symbol}...")
        df, alerts, ultimate = load_and_analyze_ultimate_signals(symbol)
        
        if df is not None:
            current_price = df['close'].iloc[-1]
            current_signal = df['ultimate_signal'].iloc[-1]
            combined_score = df['combined_score'].iloc[-1]
            confidence = df['confidence'].iloc[-1]
            
            results.append({
                'Symbol': symbol.replace('_USDT', ''),
                'Price': f"${current_price:,.2f}",
                'Signal': current_signal,
                'Score': f"{combined_score:.1f}",
                'Confidence': f"{confidence:.1f}%",
                'Trade Bulls': f"{df['score_trade_bulls'].iloc[-1]:.0f}",
                'RSI': f"{df['score_rsi'].iloc[-1]:.0f}",
                'MACD': f"{df['score_macd'].iloc[-1]:.0f}",
                'Bollinger': f"{df['score_bollinger'].iloc[-1]:.0f}",
                'Volume': f"{df['score_volume'].iloc[-1]:.0f}"
            })
    
    if results:
        # Create comparison table
        comparison_df = pd.DataFrame(results)
        print(f"\n📊 ULTIMATE SIGNALS SUMMARY:")
        print(comparison_df.to_string(index=False))
        
        # Find best opportunities
        print(f"\n🎯 TOP TRADING OPPORTUNITIES:")
        
        buy_opportunities = [r for r in results if r['Signal'] in ['STRONG_BUY', 'BUY']]
        sell_opportunities = [r for r in results if r['Signal'] in ['STRONG_SELL', 'SELL']]
        
        if buy_opportunities:
            best_buy = max(buy_opportunities, key=lambda x: float(x['Confidence'].replace('%', '')))
            print(f"   🟢 Best BUY: {best_buy['Symbol']} - {best_buy['Signal']} ({best_buy['Confidence']} confidence)")
        
        if sell_opportunities:
            best_sell = max(sell_opportunities, key=lambda x: float(x['Confidence'].replace('%', '')))
            print(f"   🔴 Best SELL: {best_sell['Symbol']} - {best_sell['Signal']} ({best_sell['Confidence']} confidence)")
        
        if not buy_opportunities and not sell_opportunities:
            print(f"   ⏳ All markets in HOLD - Wait for clearer signals")
        
        # Market sentiment
        signals = [r['Signal'] for r in results]
        bullish_count = sum(1 for s in signals if s in ['BUY', 'STRONG_BUY'])
        bearish_count = sum(1 for s in signals if s in ['SELL', 'STRONG_SELL'])
        
        print(f"\n🌍 OVERALL MARKET SENTIMENT:")
        if bullish_count > bearish_count:
            print(f"   📈 BULLISH: {bullish_count}/{len(results)} cryptos showing buy signals")
        elif bearish_count > bullish_count:
            print(f"   📉 BEARISH: {bearish_count}/{len(results)} cryptos showing sell signals")
        else:
            print(f"   ⚖️ NEUTRAL: Mixed signals across cryptos")

def main():
    """Main function - NO CHARTS VERSION"""
    print("🚀 ULTIMATE CRYPTO SIGNALS - COMBINED SYSTEM (NO CHARTS)")
    print("=" * 65)
    print("🔥 Trade Bulls + Technical Indicators = Maximum Accuracy")
    print("=" * 65)
    
    # Run analysis for all cryptos
    symbols = ['BTC_USDT', 'ETH_USDT', 'BNB_USDT']
    
    # Individual analysis
    for symbol in symbols:
        print(f"\n{'='*25} {symbol} {'='*25}")
        load_and_analyze_ultimate_signals(symbol)
    
    # Comparison
    compare_ultimate_signals()
    
    print(f"\n🎉 Ultimate Signals Analysis Complete!")
    print(f"🔥 You now have the most advanced crypto trading signals!")
    print(f"💡 NO CHARTS = Faster execution and cleaner output!")

if __name__ == "__main__":
    main()