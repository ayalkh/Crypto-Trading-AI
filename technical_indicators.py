"""
Advanced Technical Indicators for Crypto Trading AI
RSI, MACD, Bollinger Bands, Volume Analysis + Combined Signals
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TechnicalIndicators:
    def __init__(self):
        """Initialize Technical Indicators with default parameters"""
        # RSI settings
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.rsi_extreme_overbought = 80
        self.rsi_extreme_oversold = 20
        
        # MACD settings
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        
        # Bollinger Bands settings
        self.bb_period = 20
        self.bb_std_dev = 2
        
        # Volume settings
        self.volume_sma_short = 10
        self.volume_sma_long = 30
        self.volume_spike_threshold = 1.5  # 50% above average
        
        # Signal storage
        self.signals = []
    
    def calculate_rsi(self, prices, period=None):
        """Calculate Relative Strength Index (RSI)"""
        if period is None:
            period = self.rsi_period
        
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate rolling averages
        avg_gains = gains.rolling(window=period, min_periods=1).mean()
        avg_losses = losses.rolling(window=period, min_periods=1).mean()
        
        # Calculate RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, prices, fast=None, slow=None, signal=None):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        if fast is None:
            fast = self.macd_fast
        if slow is None:
            slow = self.macd_slow
        if signal is None:
            signal = self.macd_signal
        
        # Calculate EMAs
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # Signal line
        signal_line = macd_line.ewm(span=signal).mean()
        
        # Histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_bollinger_bands(self, prices, period=None, std_dev=None):
        """Calculate Bollinger Bands"""
        if period is None:
            period = self.bb_period
        if std_dev is None:
            std_dev = self.bb_std_dev
        
        # Middle band (SMA)
        middle_band = prices.rolling(window=period).mean()
        
        # Standard deviation
        std = prices.rolling(window=period).std()
        
        # Upper and lower bands
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        # Band width and position
        band_width = ((upper_band - lower_band) / middle_band) * 100
        band_position = ((prices - lower_band) / (upper_band - lower_band)) * 100
        
        return upper_band, middle_band, lower_band, band_width, band_position
    
    def analyze_volume(self, df):
        """Comprehensive volume analysis"""
        volume = df['volume']
        prices = df['close']
        
        # Volume moving averages
        volume_sma_short = volume.rolling(window=self.volume_sma_short).mean()
        volume_sma_long = volume.rolling(window=self.volume_sma_long).mean()
        
        # Volume ratio
        volume_ratio = volume / volume_sma_long
        
        # Price-Volume Trend (PVT)
        price_change_pct = prices.pct_change()
        pvt = (price_change_pct * volume).cumsum()
        
        # On-Balance Volume (OBV)
        obv = []
        obv_value = 0
        
        for i in range(len(df)):
            if i == 0:
                obv_value = volume.iloc[i]
            else:
                if prices.iloc[i] > prices.iloc[i-1]:
                    obv_value += volume.iloc[i]
                elif prices.iloc[i] < prices.iloc[i-1]:
                    obv_value -= volume.iloc[i]
                # If price unchanged, OBV stays same
            obv.append(obv_value)
        
        obv = pd.Series(obv, index=volume.index)
        
        # Volume spikes
        volume_spikes = volume_ratio > self.volume_spike_threshold
        
        # Accumulation/Distribution Line
        money_flow_multiplier = ((prices - df['low']) - (df['high'] - prices)) / (df['high'] - df['low'])
        money_flow_multiplier = money_flow_multiplier.fillna(0)
        money_flow_volume = money_flow_multiplier * volume
        ad_line = money_flow_volume.cumsum()
        
        return {
            'volume_sma_short': volume_sma_short,
            'volume_sma_long': volume_sma_long,
            'volume_ratio': volume_ratio,
            'volume_spikes': volume_spikes,
            'pvt': pvt,
            'obv': obv,
            'ad_line': ad_line
        }
    
    def generate_rsi_signals(self, rsi):
        """Generate RSI-based trading signals"""
        signals = pd.Series('HOLD', index=rsi.index)
        
        # Strong signals
        signals[rsi <= self.rsi_extreme_oversold] = 'STRONG_BUY'
        signals[rsi >= self.rsi_extreme_overbought] = 'STRONG_SELL'
        
        # Regular signals
        signals[(rsi > self.rsi_extreme_oversold) & (rsi <= self.rsi_oversold)] = 'BUY'
        signals[(rsi < self.rsi_extreme_overbought) & (rsi >= self.rsi_overbought)] = 'SELL'
        
        return signals
    
    def generate_macd_signals(self, macd_line, signal_line, histogram):
        """Generate MACD-based trading signals"""
        signals = pd.Series('HOLD', index=macd_line.index)
        
        # MACD crossovers
        macd_crossover_up = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        macd_crossover_down = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
        
        # Zero line crossovers
        zero_cross_up = (macd_line > 0) & (macd_line.shift(1) <= 0)
        zero_cross_down = (macd_line < 0) & (macd_line.shift(1) >= 0)
        
        # Histogram analysis
        histogram_increasing = histogram > histogram.shift(1)
        histogram_decreasing = histogram < histogram.shift(1)
        
        # Combined signals
        signals[macd_crossover_up & (macd_line > 0)] = 'STRONG_BUY'
        signals[macd_crossover_up & (macd_line <= 0)] = 'BUY'
        signals[macd_crossover_down & (macd_line < 0)] = 'STRONG_SELL'
        signals[macd_crossover_down & (macd_line >= 0)] = 'SELL'
        
        return signals
    
    def generate_bollinger_signals(self, prices, upper_band, lower_band, band_position):
        """Generate Bollinger Bands signals"""
        signals = pd.Series('HOLD', index=prices.index)
        
        # Price touching bands
        touching_upper = prices >= upper_band * 0.995  # 99.5% of upper band
        touching_lower = prices <= lower_band * 1.005  # 100.5% of lower band
        
        # Band squeeze (low volatility)
        band_squeeze = band_position.rolling(window=10).std() < 5
        
        # Breakouts after squeeze
        squeeze_breakout_up = band_squeeze.shift(1) & (band_position > 80)
        squeeze_breakout_down = band_squeeze.shift(1) & (band_position < 20)
        
        # Signals
        signals[touching_lower & ~band_squeeze] = 'BUY'
        signals[touching_upper & ~band_squeeze] = 'SELL'
        signals[squeeze_breakout_up] = 'STRONG_BUY'
        signals[squeeze_breakout_down] = 'STRONG_SELL'
        
        return signals
    
    def generate_volume_signals(self, volume_data, price_change):
        """Generate volume-based signals"""
        signals = pd.Series('HOLD', index=price_change.index)
        
        volume_spikes = volume_data['volume_spikes']
        volume_ratio = volume_data['volume_ratio']
        
        # Price up with high volume
        bullish_volume = (price_change > 0) & volume_spikes
        bearish_volume = (price_change < 0) & volume_spikes
        
        # Volume confirmation signals
        signals[bullish_volume & (volume_ratio > 2)] = 'STRONG_BUY'
        signals[bullish_volume & (volume_ratio > 1.5)] = 'BUY'
        signals[bearish_volume & (volume_ratio > 2)] = 'STRONG_SELL'
        signals[bearish_volume & (volume_ratio > 1.5)] = 'SELL'
        
        return signals
    
    def combine_signals(self, rsi_signals, macd_signals, bb_signals, volume_signals):
        """Combine all signals into final recommendation"""
        combined_signals = pd.Series('HOLD', index=rsi_signals.index)
        
        # Signal scoring system
        signal_scores = pd.Series(0, index=rsi_signals.index)
        
        # RSI scores
        signal_scores[rsi_signals == 'STRONG_BUY'] += 3
        signal_scores[rsi_signals == 'BUY'] += 2
        signal_scores[rsi_signals == 'SELL'] -= 2
        signal_scores[rsi_signals == 'STRONG_SELL'] -= 3
        
        # MACD scores
        signal_scores[macd_signals == 'STRONG_BUY'] += 3
        signal_scores[macd_signals == 'BUY'] += 2
        signal_scores[macd_signals == 'SELL'] -= 2
        signal_scores[macd_signals == 'STRONG_SELL'] -= 3
        
        # Bollinger Bands scores
        signal_scores[bb_signals == 'STRONG_BUY'] += 3
        signal_scores[bb_signals == 'BUY'] += 2
        signal_scores[bb_signals == 'SELL'] -= 2
        signal_scores[bb_signals == 'STRONG_SELL'] -= 3
        
        # Volume scores (confirmation)
        signal_scores[volume_signals == 'STRONG_BUY'] += 2
        signal_scores[volume_signals == 'BUY'] += 1
        signal_scores[volume_signals == 'SELL'] -= 1
        signal_scores[volume_signals == 'STRONG_SELL'] -= 2
        
        # Final signals based on combined scores
        combined_signals[signal_scores >= 6] = 'STRONG_BUY'
        combined_signals[(signal_scores >= 3) & (signal_scores < 6)] = 'BUY'
        combined_signals[(signal_scores <= -3) & (signal_scores > -6)] = 'SELL'
        combined_signals[signal_scores <= -6] = 'STRONG_SELL'
        
        return combined_signals, signal_scores
    
    def analyze_crypto_indicators(self, df):
        """Main function to analyze all technical indicators"""
        print(f"🔬 Calculating Advanced Technical Indicators...")
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Calculate all indicators
        df['rsi'] = self.calculate_rsi(df['close'])
        
        macd_line, signal_line, histogram = self.calculate_macd(df['close'])
        df['macd_line'] = macd_line
        df['macd_signal_line'] = signal_line
        df['macd_histogram'] = histogram
        
        upper_bb, middle_bb, lower_bb, bb_width, bb_position = self.calculate_bollinger_bands(df['close'])
        df['bb_upper'] = upper_bb
        df['bb_middle'] = middle_bb
        df['bb_lower'] = lower_bb
        df['bb_width'] = bb_width
        df['bb_position'] = bb_position
        
        # Volume analysis
        volume_data = self.analyze_volume(df)
        for key, value in volume_data.items():
            df[f'volume_{key}'] = value
        
        # Generate individual signals
        df['rsi_signal'] = self.generate_rsi_signals(df['rsi'])
        df['macd_signal'] = self.generate_macd_signals(df['macd_line'], df['macd_signal_line'], df['macd_histogram'])
        df['bb_signal'] = self.generate_bollinger_signals(df['close'], df['bb_upper'], df['bb_lower'], df['bb_position'])
        df['volume_signal'] = self.generate_volume_signals(volume_data, df['close'].pct_change())
        
        # Combine all signals
        combined_signal, signal_scores = self.combine_signals(
            df['rsi_signal'], df['macd_signal'], df['bb_signal'], df['volume_signal']
        )
        df['combined_signal'] = combined_signal
        df['signal_score'] = signal_scores
        
        print(f"✅ Technical indicators calculated successfully!")
        
        return df
    
    def create_indicators_chart(self, df, symbol='BTC_USDT'):
        """Create comprehensive indicators chart"""
        # Create figure with proper spacing
        fig = plt.figure(figsize=(16, 18))
        
        # Clean symbol name for title
        clean_symbol = symbol.replace('_USDT', '/USDT')

        
        # Create 4 subplots with proper spacing
        gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1, 1], hspace=0.4)
        
        # 1. Price with Bollinger Bands and Signals
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(df['timestamp'], df['close'], label='Price', color='black', linewidth=2)
        ax1.plot(df['timestamp'], df['bb_upper'], label='BB Upper', color='red', alpha=0.8, linewidth=1.5)
        ax1.plot(df['timestamp'], df['bb_middle'], label='BB Middle', color='orange', alpha=0.8, linewidth=1.5)
        ax1.plot(df['timestamp'], df['bb_lower'], label='BB Lower', color='green', alpha=0.8, linewidth=1.5)
        ax1.fill_between(df['timestamp'], df['bb_upper'], df['bb_lower'], alpha=0.1, color='blue')
        
        # Add combined signals
        buy_signals = df[df['combined_signal'].isin(['BUY', 'STRONG_BUY'])]
        sell_signals = df[df['combined_signal'].isin(['SELL', 'STRONG_SELL'])]
        
        if not buy_signals.empty:
            ax1.scatter(buy_signals['timestamp'], buy_signals['close'], 
                       color='green', marker='^', s=120, label='BUY Signal', 
                       zorder=5, edgecolors='darkgreen', linewidth=1)
        
        if not sell_signals.empty:
            ax1.scatter(sell_signals['timestamp'], sell_signals['close'], 
                       color='red', marker='v', s=120, label='SELL Signal', 
                       zorder=5, edgecolors='darkred', linewidth=1)
        
        ax1.set_title('Price Action with Bollinger Bands & Trading Signals', fontsize=14, pad=10)
        ax1.set_ylabel('Price (USDT)', fontsize=12)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. RSI
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(df['timestamp'], df['rsi'], label='RSI', color='purple', linewidth=2)
        ax2.axhline(y=self.rsi_overbought, color='red', linestyle='--', 
                   alpha=0.8, label='Overbought (70)', linewidth=1.5)
        ax2.axhline(y=self.rsi_oversold, color='green', linestyle='--', 
                   alpha=0.8, label='Oversold (30)', linewidth=1.5)
        ax2.axhline(y=self.rsi_extreme_overbought, color='darkred', linestyle='-', 
                   alpha=0.8, label='Extreme OB (80)', linewidth=1)
        ax2.axhline(y=self.rsi_extreme_oversold, color='darkgreen', linestyle='-', 
                   alpha=0.8, label='Extreme OS (20)', linewidth=1)
        
        # Fill RSI zones
        ax2.fill_between(df['timestamp'], 0, self.rsi_extreme_oversold, alpha=0.2, color='darkgreen')
        ax2.fill_between(df['timestamp'], self.rsi_extreme_oversold, self.rsi_oversold, alpha=0.15, color='green')
        ax2.fill_between(df['timestamp'], self.rsi_overbought, self.rsi_extreme_overbought, alpha=0.15, color='red')
        ax2.fill_between(df['timestamp'], self.rsi_extreme_overbought, 100, alpha=0.2, color='darkred')
        
        ax2.set_title('RSI - Relative Strength Index', fontsize=14, pad=10)
        ax2.set_ylabel('RSI Value', fontsize=12)
        ax2.set_ylim(0, 100)
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # 3. MACD
        ax3 = fig.add_subplot(gs[2])
        ax3.plot(df['timestamp'], df['macd_line'], label='MACD Line', 
                color='blue', linewidth=2)
        ax3.plot(df['timestamp'], df['macd_signal_line'], label='Signal Line', 
                color='red', linewidth=2)
        
        # MACD Histogram with better colors
        histogram_colors = ['green' if x >= 0 else 'red' for x in df['macd_histogram']]
        ax3.bar(df['timestamp'], df['macd_histogram'], label='Histogram', 
               alpha=0.7, color=histogram_colors, width=0.5)
        
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.6, linewidth=1)
        
        ax3.set_title('MACD - Moving Average Convergence Divergence', fontsize=14, pad=10)
        ax3.set_ylabel('MACD Value', fontsize=12)
        ax3.legend(loc='upper left', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
    
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        # Show the chart
        plt.show()
        
        print(f"📊 Technical indicators chart created for {clean_symbol}")
    
    def generate_trading_report(self, df, symbol='BTC_USDT'):
        """Generate comprehensive trading report"""
        print(f"\n📊 TECHNICAL INDICATORS REPORT - {symbol}")
        print("=" * 60)
        
        current_price = df['close'].iloc[-1]
        current_rsi = df['rsi'].iloc[-1]
        current_signal = df['combined_signal'].iloc[-1]
        signal_score = df['signal_score'].iloc[-1]
        
        print(f"💰 Current Price: ${current_price:,.2f}")
        print(f"🎯 Combined Signal: {current_signal}")
        print(f"📊 Signal Strength: {signal_score:.1f}/12")
        
        # RSI Analysis
        print(f"\n📈 RSI ANALYSIS:")
        print(f"   Current RSI: {current_rsi:.1f}")
        if current_rsi >= self.rsi_extreme_overbought:
            rsi_status = "🔴 Extremely Overbought - Strong Sell Signal"
        elif current_rsi >= self.rsi_overbought:
            rsi_status = "🟠 Overbought - Sell Signal"
        elif current_rsi <= self.rsi_extreme_oversold:
            rsi_status = "🟢 Extremely Oversold - Strong Buy Signal"
        elif current_rsi <= self.rsi_oversold:
            rsi_status = "🟡 Oversold - Buy Signal"
        else:
            rsi_status = "⚪ Neutral Zone"
        print(f"   Status: {rsi_status}")
        
        # MACD Analysis
        current_macd = df['macd_line'].iloc[-1]
        current_macd_signal_line = df['macd_signal_line'].iloc[-1]
        current_histogram = df['macd_histogram'].iloc[-1]
        
        print(f"\n📊 MACD ANALYSIS:")
        print(f"   MACD Line: {current_macd:.4f}")
        print(f"   Signal Line: {current_macd_signal_line:.4f}")
        print(f"   Histogram: {current_histogram:.4f}")
        
        if current_macd > current_macd_signal_line:
            macd_status = "🟢 Bullish (MACD > Signal)"
        else:
            macd_status = "🔴 Bearish (MACD < Signal)"
        print(f"   Status: {macd_status}")
        
        # Bollinger Bands Analysis
        current_bb_position = df['bb_position'].iloc[-1]
        current_bb_width = df['bb_width'].iloc[-1]
        
        print(f"\n📏 BOLLINGER BANDS ANALYSIS:")
        print(f"   Band Position: {current_bb_position:.1f}%")
        print(f"   Band Width: {current_bb_width:.2f}%")
        
        if current_bb_position >= 95:
            bb_status = "🔴 Near Upper Band - Potential Resistance"
        elif current_bb_position <= 5:
            bb_status = "🟢 Near Lower Band - Potential Support"
        elif current_bb_width < 2:
            bb_status = "🟡 Band Squeeze - Breakout Expected"
        else:
            bb_status = "⚪ Normal Range"
        print(f"   Status: {bb_status}")
        
        # Volume Analysis
        current_volume_ratio = df['volume_volume_ratio'].iloc[-1]
        volume_spike = df['volume_volume_spikes'].iloc[-1]
        
        print(f"\n📊 VOLUME ANALYSIS:")
        print(f"   Volume Ratio: {current_volume_ratio:.2f}x")
        print(f"   Volume Spike: {'🔥 YES' if volume_spike else '❌ No'}")
        
        if current_volume_ratio > 2:
            volume_status = "🔥 Very High Volume"
        elif current_volume_ratio > 1.5:
            volume_status = "📈 High Volume"
        elif current_volume_ratio < 0.5:
            volume_status = "📉 Low Volume"
        else:
            volume_status = "⚪ Normal Volume"
        print(f"   Status: {volume_status}")
        
        # Signal Summary
        print(f"\n🎯 SIGNAL SUMMARY:")
        recent_signals = df['combined_signal'].tail(24)  # Last 24 hours
        signal_counts = recent_signals.value_counts()
        
        print(f"   Last 24 Hours Signal Distribution:")
        for signal, count in signal_counts.items():
            print(f"   {signal}: {count} times")
        
        # Recommendations
        print(f"\n💡 TRADING RECOMMENDATIONS:")
        if current_signal in ['STRONG_BUY', 'BUY']:
            print(f"   ✅ Consider BUYING - Multiple indicators align")
            print(f"   🎯 Entry: ${current_price:,.2f}")
            print(f"   🛑 Stop Loss: ${df['bb_lower'].iloc[-1]:,.2f} (BB Lower)")
            print(f"   🚀 Target: ${df['bb_upper'].iloc[-1]:,.2f} (BB Upper)")
        
        elif current_signal in ['STRONG_SELL', 'SELL']:
            print(f"   ❌ Consider SELLING - Multiple indicators align")
            print(f"   🎯 Exit: ${current_price:,.2f}")
            print(f"   🛑 Stop Loss: ${df['bb_upper'].iloc[-1]:,.2f} (BB Upper)")
            print(f"   🚀 Target: ${df['bb_lower'].iloc[-1]:,.2f} (BB Lower)")
        
        else:
            print(f"   ⏳ HOLD - Wait for clearer signals")
            print(f"   👀 Watch for RSI extreme levels or MACD crossovers")
        
        print(f"\n⚠️ Risk Management:")
        print(f"   📊 Signal Confidence: {abs(signal_score)/12*100:.1f}%")
        print(f"   🎲 Position Size: {'Large' if abs(signal_score) > 8 else 'Medium' if abs(signal_score) > 5 else 'Small'}")

def analyze_crypto_with_indicators(symbol='BTC_USDT'):
    """Load crypto data and apply technical indicators analysis"""
    filename = f"data/{symbol}_1h_7days.csv"
    
    try:
        df = pd.read_csv(filename)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"✅ Loaded {len(df)} records for {symbol}")
        
        # Initialize indicators
        indicators = TechnicalIndicators()
        
        # Analyze with all indicators
        df_analyzed = indicators.analyze_crypto_indicators(df)
        
        # Create visualization
        indicators.create_indicators_chart(df_analyzed, symbol)
        
        # Generate trading report
        indicators.generate_trading_report(df_analyzed, symbol)
        
        return df_analyzed, indicators
        
    except FileNotFoundError:
        print(f"❌ File not found: {filename}")
        print("💡 Run crypto_data_collector.py first!")
        return None, None

def main():
    """Main function to run technical indicators analysis"""
    print("📊 ADVANCED TECHNICAL INDICATORS ANALYSIS")
    print("=" * 55)
    print("🔬 RSI | MACD | Bollinger Bands | Volume Analysis")
    print("=" * 55)
    
    symbols = ['BTC_USDT', 'ETH_USDT', 'BNB_USDT']
    
    for symbol in symbols:
        print(f"\n📈 Analyzing {symbol}...")
        df, indicators = analyze_crypto_with_indicators(symbol)
        
        if df is not None:
            current_signal = df['combined_signal'].iloc[-1]
            signal_score = df['signal_score'].iloc[-1]
            current_rsi = df['rsi'].iloc[-1]
            
            print(f"🎯 {symbol} Summary:")
            print(f"   Signal: {current_signal} (Score: {signal_score:.1f})")
            print(f"   RSI: {current_rsi:.1f}")
            print(f"   Price: ${df['close'].iloc[-1]:,.2f}")
    
    print(f"\n🎉 Technical indicators analysis complete!")
    print(f"🔥 Now you have professional-grade trading signals!")

if __name__ == "__main__":
    main()