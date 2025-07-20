"""
Trade Bulls v1.2 Strategy - Python Implementation
Translated from Pine Script for Crypto Trading AI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TradeBullsStrategy:
    def __init__(self):
        """Initialize the Trade Bulls strategy"""
        # Block Order settings
        self.bo_enabled = True
        self.bo_sensitivity = 0.28  # 28% converted to decimal
        self.ob_mitigation_type = "Close"  # "Close" or "Wick"
        
        # Buy/Sell settings
        self.bs_enabled = True
        self.bs_type = "Atr"
        self.bs_size = 1.0
        self.bs_max_sequence = 3
        self.fibonacci_sequence = [1, 1, 2, 3, 5, 8, 13]
        
        # Channel settings
        self.ch_enabled = True
        self.ch_length = 100
        self.ch_deviation = 2.0
        
        # Support/Resistance settings
        self.sr_enabled = True
        self.sr_left_bars = 15
        self.sr_right_bars = 15
        self.sr_volume_threshold = 20
        
        # Storage for signals and levels
        self.signals = []
        self.order_blocks = []
        self.support_levels = []
        self.resistance_levels = []
    
    def calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close_prev = np.abs(df['high'] - df['close'].shift(1))
        low_close_prev = np.abs(df['low'] - df['close'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def detect_order_blocks(self, df):
        """Detect Block Orders (Supply/Demand Zones)"""
        if not self.bo_enabled:
            return df
        
        # Calculate rate of change
        df['price_change_rate'] = (df['open'] - df['open'].shift(4)) / df['open'].shift(4) * 100
        
        # Detect significant moves
        df['bullish_ob'] = df['price_change_rate'] > (self.bo_sensitivity * 100)
        df['bearish_ob'] = df['price_change_rate'] < -(self.bo_sensitivity * 100)
        
        # Find recent red/green candles for order blocks
        df['is_red_candle'] = df['close'] < df['open']
        df['is_green_candle'] = df['close'] > df['open']
        
        order_blocks = []
        
        for i in range(len(df)):
            if df['bullish_ob'].iloc[i]:
                # Look for recent red candle (4-15 bars back)
                for j in range(4, min(16, i)):
                    if df['is_red_candle'].iloc[i-j]:
                        ob = {
                            'type': 'bullish',
                            'bar_index': i-j,
                            'high': df['high'].iloc[i-j],
                            'low': df['low'].iloc[i-j],
                            'active': True
                        }
                        order_blocks.append(ob)
                        break
            
            elif df['bearish_ob'].iloc[i]:
                # Look for recent green candle (4-15 bars back)
                for j in range(4, min(16, i)):
                    if df['is_green_candle'].iloc[i-j]:
                        ob = {
                            'type': 'bearish',
                            'bar_index': i-j,
                            'high': df['high'].iloc[i-j],
                            'low': df['low'].iloc[i-j],
                            'active': True
                        }
                        order_blocks.append(ob)
                        break
        
        self.order_blocks = order_blocks
        return df
    
    def calculate_fibonacci_levels(self, current_price, avg_price, atr_value, fib_level):
        """Calculate Fibonacci-based distance"""
        fib_multiplier = self.fibonacci_sequence[min(fib_level, len(self.fibonacci_sequence)-1)]
        return atr_value * self.bs_size * fib_multiplier
    
    def generate_buy_sell_signals(self, df):
        """Generate Buy/Sell signals using Fibonacci sequence"""
        if not self.bs_enabled:
            return df
        
        # Calculate ATR
        df['atr'] = self.calculate_atr(df, 200)
        
        # Initialize variables
        df['fib_level'] = 1
        df['avg_price'] = df['close'].iloc[0]
        df['signal'] = 'HOLD'
        df['stop_loss'] = np.nan
        df['take_profit'] = np.nan
        
        fib_level = 1
        avg_price = df['close'].iloc[0]
        position = 0  # 0 = neutral, 1 = long, -1 = short
        
        for i in range(1, len(df)):
            current_price = df['close'].iloc[i]
            atr_value = df['atr'].iloc[i]
            
            if pd.isna(atr_value):
                atr_value = df['atr'].dropna().iloc[0] if not df['atr'].dropna().empty else 100
            
            # Calculate Fibonacci distance
            fib_distance = self.calculate_fibonacci_levels(current_price, avg_price, atr_value, fib_level)
            
            # Check if price moved beyond Fibonacci level
            if abs(current_price - avg_price) > fib_distance:
                fib_level = min(fib_level + 1, self.bs_max_sequence + 1)
                
                # Determine trend direction
                if current_price > avg_price:
                    # Bullish trend
                    if position != 1:
                        df.loc[df.index[i], 'signal'] = 'BUY'
                        position = 1
                    avg_price = current_price
                else:
                    # Bearish trend
                    if position != -1:
                        df.loc[df.index[i], 'signal'] = 'SELL'
                        position = -1
                    avg_price = current_price
                
                # Reset Fibonacci level if max reached
                if fib_level > self.bs_max_sequence + 1:
                    fib_level = 1
            
            # Calculate stop loss and take profit
            if position == 1:  # Long position
                df.loc[df.index[i], 'stop_loss'] = avg_price - fib_distance
                df.loc[df.index[i], 'take_profit'] = avg_price + fib_distance
            elif position == -1:  # Short position
                df.loc[df.index[i], 'stop_loss'] = avg_price + fib_distance
                df.loc[df.index[i], 'take_profit'] = avg_price - fib_distance
            
            df.loc[df.index[i], 'fib_level'] = fib_level
            df.loc[df.index[i], 'avg_price'] = avg_price
        
        return df
    
    def calculate_channel(self, df):
        """Calculate Linear Regression Channel"""
        if not self.ch_enabled or len(df) < self.ch_length:
            return df
        
        df['channel_upper'] = np.nan
        df['channel_lower'] = np.nan
        df['channel_middle'] = np.nan
        df['channel_trend'] = np.nan
        
        for i in range(self.ch_length, len(df)):
            # Get data window
            window_data = df['close'].iloc[i-self.ch_length:i]
            x = np.arange(len(window_data))
            
            # Linear regression
            slope, intercept = np.polyfit(x, window_data, 1)
            
            # Calculate deviation
            regression_line = slope * x + intercept
            deviation = np.sqrt(np.mean((window_data - regression_line) ** 2))
            
            # Current regression values
            current_middle = slope * (self.ch_length - 1) + intercept
            
            df.loc[df.index[i], 'channel_middle'] = current_middle
            df.loc[df.index[i], 'channel_upper'] = current_middle + deviation * self.ch_deviation
            df.loc[df.index[i], 'channel_lower'] = current_middle - deviation * self.ch_deviation
            df.loc[df.index[i], 'channel_trend'] = slope
        
        return df
    
    def find_support_resistance(self, df):
        """Find Support and Resistance levels"""
        if not self.sr_enabled:
            return df
        
        # Calculate pivot highs and lows
        df['pivot_high'] = df['high'].rolling(window=self.sr_left_bars + self.sr_right_bars + 1, center=True).apply(
            lambda x: x.iloc[self.sr_left_bars] if x.iloc[self.sr_left_bars] == x.max() else np.nan
        )
        
        df['pivot_low'] = df['low'].rolling(window=self.sr_left_bars + self.sr_right_bars + 1, center=True).apply(
            lambda x: x.iloc[self.sr_left_bars] if x.iloc[self.sr_left_bars] == x.min() else np.nan
        )
        
        # Volume confirmation
        df['volume_ema_short'] = df['volume'].ewm(span=5).mean()
        df['volume_ema_long'] = df['volume'].ewm(span=10).mean()
        df['volume_delta'] = 100 * (df['volume_ema_short'] - df['volume_ema_long']) / df['volume_ema_long']
        
        # Forward fill pivot levels
        df['resistance_level'] = df['pivot_high'].fillna(method='ffill')
        df['support_level'] = df['pivot_low'].fillna(method='ffill')
        
        return df
    
    def generate_alerts(self, df):
        """Generate trading alerts"""
        alerts = []
        
        for i in range(1, len(df)):
            current_price = df['close'].iloc[i]
            prev_price = df['close'].iloc[i-1]
            
            # Buy/Sell signals
            if df['signal'].iloc[i] == 'BUY':
                alerts.append({
                    'timestamp': df['timestamp'].iloc[i],
                    'type': 'BUY_SIGNAL',
                    'price': current_price,
                    'stop_loss': df['stop_loss'].iloc[i],
                    'take_profit': df['take_profit'].iloc[i],
                    'confidence': 'HIGH'
                })
            
            elif df['signal'].iloc[i] == 'SELL':
                alerts.append({
                    'timestamp': df['timestamp'].iloc[i],
                    'type': 'SELL_SIGNAL',
                    'price': current_price,
                    'stop_loss': df['stop_loss'].iloc[i],
                    'take_profit': df['take_profit'].iloc[i],
                    'confidence': 'HIGH'
                })
            
            # Support/Resistance breaks with volume confirmation
            if not pd.isna(df['resistance_level'].iloc[i]) and df['volume_delta'].iloc[i] > self.sr_volume_threshold:
                if current_price > df['resistance_level'].iloc[i] and prev_price <= df['resistance_level'].iloc[i]:
                    alerts.append({
                        'timestamp': df['timestamp'].iloc[i],
                        'type': 'RESISTANCE_BREAK',
                        'price': current_price,
                        'level': df['resistance_level'].iloc[i],
                        'confidence': 'MEDIUM'
                    })
            
            if not pd.isna(df['support_level'].iloc[i]) and df['volume_delta'].iloc[i] > self.sr_volume_threshold:
                if current_price < df['support_level'].iloc[i] and prev_price >= df['support_level'].iloc[i]:
                    alerts.append({
                        'timestamp': df['timestamp'].iloc[i],
                        'type': 'SUPPORT_BREAK',
                        'price': current_price,
                        'level': df['support_level'].iloc[i],
                        'confidence': 'MEDIUM'
                    })
        
        return alerts
    
    def analyze_crypto(self, df):
        """Main analysis function - applies full Trade Bulls strategy"""
        print(f"🚀 Applying Trade Bulls v1.2 Strategy...")
        
        # Ensure timestamp column
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.to_datetime(df.index)
        
        # Apply all strategy components
        df = self.detect_order_blocks(df)
        df = self.generate_buy_sell_signals(df)
        df = self.calculate_channel(df)
        df = self.find_support_resistance(df)
        
        # Generate alerts
        alerts = self.generate_alerts(df)
        
        # Current market analysis
        current_signal = df['signal'].iloc[-1]
        current_price = df['close'].iloc[-1]
        
        print(f"✅ Strategy analysis complete!")
        print(f"📊 Current Signal: {current_signal}")
        print(f"💰 Current Price: ${current_price:,.2f}")
        print(f"🔔 Generated {len(alerts)} alerts")
        
        return df, alerts
    
    def create_strategy_chart(self, df, symbol='BTC_USDT'):
        """Create comprehensive strategy chart"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle(f'Trade Bulls v1.2 Strategy - {symbol}', fontsize=16, fontweight='bold')
        
        # 1. Price with signals and channels
        ax1 = axes[0]
        ax1.plot(df['timestamp'], df['close'], label='Price', color='black', linewidth=1)
        
        # Buy/Sell signals
        buy_signals = df[df['signal'] == 'BUY']
        sell_signals = df[df['signal'] == 'SELL']
        
        if not buy_signals.empty:
            ax1.scatter(buy_signals['timestamp'], buy_signals['close'], 
                       color='green', marker='^', s=100, label='BUY', zorder=5)
        
        if not sell_signals.empty:
            ax1.scatter(sell_signals['timestamp'], sell_signals['close'], 
                       color='red', marker='v', s=100, label='SELL', zorder=5)
        
        # Channel lines
        if 'channel_upper' in df.columns:
            ax1.plot(df['timestamp'], df['channel_upper'], color='blue', alpha=0.5, label='Channel Upper')
            ax1.plot(df['timestamp'], df['channel_lower'], color='blue', alpha=0.5, label='Channel Lower')
            ax1.fill_between(df['timestamp'], df['channel_upper'], df['channel_lower'], 
                           alpha=0.1, color='blue')
        
        ax1.set_title('Price Action with Trade Bulls Signals')
        ax1.set_ylabel('Price (USDT)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Support/Resistance levels
        ax2 = axes[1]
        ax2.plot(df['timestamp'], df['close'], color='black', alpha=0.7)
        
        if 'resistance_level' in df.columns:
            ax2.plot(df['timestamp'], df['resistance_level'], color='red', linewidth=2, 
                    label='Resistance', alpha=0.7)
        
        if 'support_level' in df.columns:
            ax2.plot(df['timestamp'], df['support_level'], color='green', linewidth=2, 
                    label='Support', alpha=0.7)
        
        ax2.set_title('Support & Resistance Levels')
        ax2.set_ylabel('Price (USDT)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Volume with delta
        ax3 = axes[2]
        colors = ['red' if df['close'].iloc[i] < df['open'].iloc[i] else 'green' 
                 for i in range(len(df))]
        ax3.bar(df['timestamp'], df['volume'], color=colors, alpha=0.6)
        
        if 'volume_delta' in df.columns:
            ax3_twin = ax3.twinx()
            ax3_twin.plot(df['timestamp'], df['volume_delta'], color='purple', 
                         label='Volume Delta %')
            ax3_twin.axhline(y=self.sr_volume_threshold, color='orange', 
                           linestyle='--', label='Volume Threshold')
            ax3_twin.legend()
        
        ax3.set_title('Volume Analysis')
        ax3.set_ylabel('Volume')
        ax3.set_xlabel('Date & Time')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"📈 Trade Bulls strategy chart created!")

def load_and_analyze_crypto(symbol='BTC_USDT'):
    """Load crypto data and apply Trade Bulls strategy"""
    filename = f"data/{symbol}_1h_7days.csv"
    
    try:
        df = pd.read_csv(filename)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"✅ Loaded {len(df)} records for {symbol}")
        
        # Initialize strategy
        strategy = TradeBullsStrategy()
        
        # Analyze with Trade Bulls strategy
        df_analyzed, alerts = strategy.analyze_crypto(df)
        
        # Create visualization
        strategy.create_strategy_chart(df_analyzed, symbol)
        
        # Display recent alerts
        if alerts:
            print(f"\n🔔 RECENT ALERTS:")
            for alert in alerts[-5:]:  # Show last 5 alerts
                print(f"   {alert['type']}: ${alert['price']:.2f} at {alert['timestamp']}")
        
        return df_analyzed, alerts, strategy
        
    except FileNotFoundError:
        print(f"❌ File not found: {filename}")
        print("💡 Run crypto_data_collector.py first!")
        return None, None, None

def main():
    """Main function to run Trade Bulls strategy"""
    print("🐂 TRADE BULLS v1.2 STRATEGY - PYTHON EDITION")
    print("=" * 55)
    
    symbols = ['BTC_USDT', 'ETH_USDT', 'BNB_USDT']
    
    for symbol in symbols:
        print(f"\n📊 Analyzing {symbol} with Trade Bulls Strategy...")
        df, alerts, strategy = load_and_analyze_crypto(symbol)
        
        if df is not None:
            current_signal = df['signal'].iloc[-1]
            current_price = df['close'].iloc[-1]
            
            print(f"🎯 {symbol} Analysis:")
            print(f"   Current Signal: {current_signal}")
            print(f"   Current Price: ${current_price:,.2f}")
            
            if not df[df['signal'] != 'HOLD'].empty:
                last_signal = df[df['signal'] != 'HOLD'].iloc[-1]
                print(f"   Last Action: {last_signal['signal']} at ${last_signal['close']:.2f}")
    
    print(f"\n🎉 Trade Bulls analysis complete!")
    print(f"🚀 Your Pine Script strategy is now running in Python!")

if __name__ == "__main__":
    main()