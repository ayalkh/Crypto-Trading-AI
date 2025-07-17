"""
Crypto Trading AI - Data Analysis
Analyze collected cryptocurrency data and create insights
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

def load_crypto_data(symbol='BTC_USDT'):
    """Load cryptocurrency data from CSV file"""
    filename = f"data/{symbol}_1h_7days.csv"
    
    if not os.path.exists(filename):
        print(f"❌ File not found: {filename}")
        print("💡 Run crypto_data_collector.py first!")
        return None
    
    try:
        df = pd.read_csv(filename)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')  # Ensure chronological order
        
        print(f"✅ Loaded {len(df)} records for {symbol}")
        return df
    except Exception as e:
        print(f"❌ Error loading {symbol}: {e}")
        return None

def calculate_basic_indicators(df):
    """Calculate basic technical indicators"""
    if df is None or len(df) < 20:
        return df
    
    # Simple Moving Averages
    df['sma_12'] = df['close'].rolling(window=12).mean()  # 12-hour SMA
    df['sma_24'] = df['close'].rolling(window=24).mean()  # 24-hour SMA
    
    # Price change
    df['price_change'] = df['close'].pct_change()
    df['price_change_24h'] = df['close'].pct_change(periods=24)
    
    # Volatility (rolling standard deviation)
    df['volatility'] = df['close'].rolling(window=24).std()
    
    # High-Low spread
    df['hl_spread'] = ((df['high'] - df['low']) / df['close']) * 100
    
    # Volume moving average
    df['volume_sma'] = df['volume'].rolling(window=12).mean()
    
    return df

def analyze_crypto_detailed(symbol='BTC_USDT'):
    """Perform detailed analysis of a cryptocurrency"""
    print(f"\n📊 DETAILED ANALYSIS: {symbol}")
    print("=" * 50)
    
    # Load and prepare data
    df = load_crypto_data(symbol)
    if df is None:
        return None
    
    df = calculate_basic_indicators(df)
    
    # Basic statistics
    current_price = df['close'].iloc[-1]
    start_price = df['close'].iloc[0]
    change_7d = ((current_price - start_price) / start_price) * 100
    
    # 24-hour change
    change_24h = df['price_change_24h'].iloc[-1] * 100 if not pd.isna(df['price_change_24h'].iloc[-1]) else 0
    
    print(f"📈 PRICE ANALYSIS:")
    print(f"   Current Price:     ${current_price:,.2f}")
    print(f"   7-day change:      {change_7d:+.2f}%")
    print(f"   24-hour change:    {change_24h:+.2f}%")
    print(f"   Highest (7d):      ${df['high'].max():,.2f}")
    print(f"   Lowest (7d):       ${df['low'].min():,.2f}")
    print(f"   Average price:     ${df['close'].mean():,.2f}")
    
    # Volatility analysis
    current_volatility = df['volatility'].iloc[-1]
    avg_volatility = df['volatility'].mean()
    
    print(f"\n📊 VOLATILITY ANALYSIS:")
    print(f"   Current volatility: ${current_volatility:.2f}")
    print(f"   Average volatility: ${avg_volatility:.2f}")
    print(f"   Max daily spread:   {df['hl_spread'].max():.2f}%")
    print(f"   Avg daily spread:   {df['hl_spread'].mean():.2f}%")
    
    # Volume analysis
    current_volume = df['volume'].iloc[-1]
    avg_volume = df['volume'].mean()
    volume_trend = "📈 Above average" if current_volume > avg_volume else "📉 Below average"
    
    print(f"\n💹 VOLUME ANALYSIS:")
    print(f"   Current volume:     {current_volume:,.0f}")
    print(f"   Average volume:     {avg_volume:,.0f}")
    print(f"   Volume trend:       {volume_trend}")
    print(f"   Max volume (7d):    {df['volume'].max():,.0f}")
    
    # Technical indicators
    current_sma12 = df['sma_12'].iloc[-1]
    current_sma24 = df['sma_24'].iloc[-1]
    
    if not pd.isna(current_sma12) and not pd.isna(current_sma24):
        trend = "🟢 Bullish" if current_price > current_sma12 > current_sma24 else "🔴 Bearish"
        print(f"\n📈 TECHNICAL INDICATORS:")
        print(f"   12-hour SMA:        ${current_sma12:.2f}")
        print(f"   24-hour SMA:        ${current_sma24:.2f}")
        print(f"   Short-term trend:   {trend}")
    
    # Recent performance
    print(f"\n⏰ RECENT PERFORMANCE:")
    recent_hours = min(24, len(df))
    recent_df = df.tail(recent_hours)
    best_hour = recent_df.loc[recent_df['close'].idxmax()]
    worst_hour = recent_df.loc[recent_df['close'].idxmin()]
    
    print(f"   Best hour:          ${best_hour['close']:.2f} at {best_hour['timestamp']}")
    print(f"   Worst hour:         ${worst_hour['close']:.2f} at {worst_hour['timestamp']}")
    
    return df

def create_comprehensive_chart(symbol='BTC_USDT'):
    """Create comprehensive price and volume charts"""
    df = load_crypto_data(symbol)
    if df is None:
        return
    
    df = calculate_basic_indicators(df)
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle(f'{symbol.replace("_", "/")} - Comprehensive Analysis', fontsize=16, fontweight='bold')
    
    # 1. Price chart with moving averages
    ax1 = axes[0]
    ax1.plot(df['timestamp'], df['close'], label='Price', color='#FF6B35', linewidth=2)
    
    if not df['sma_12'].isna().all():
        ax1.plot(df['timestamp'], df['sma_12'], label='12h SMA', color='#4ECDC4', alpha=0.8)
    if not df['sma_24'].isna().all():
        ax1.plot(df['timestamp'], df['sma_24'], label='24h SMA', color='#45B7D1', alpha=0.8)
    
    ax1.set_title('Price Movement with Moving Averages', fontweight='bold')
    ax1.set_ylabel('Price (USDT)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Fill area between high and low
    ax1.fill_between(df['timestamp'], df['low'], df['high'], alpha=0.1, color='gray', label='High-Low Range')
    
    # 2. Volume chart
    ax2 = axes[1]
    colors = ['red' if df['close'].iloc[i] < df['open'].iloc[i] else 'green' for i in range(len(df))]
    ax2.bar(df['timestamp'], df['volume'], color=colors, alpha=0.7, width=0.02)
    
    if not df['volume_sma'].isna().all():
        ax2.plot(df['timestamp'], df['volume_sma'], label='Volume SMA', color='purple', linewidth=2)
    
    ax2.set_title('Trading Volume', fontweight='bold')
    ax2.set_ylabel('Volume')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Volatility and spreads
    ax3 = axes[2]
    if not df['volatility'].isna().all():
        ax3.plot(df['timestamp'], df['volatility'], label='24h Volatility', color='red', linewidth=2)
    
    ax3_twin = ax3.twinx()
    ax3_twin.plot(df['timestamp'], df['hl_spread'], label='High-Low Spread %', color='orange', alpha=0.7)
    
    ax3.set_title('Volatility and Price Spreads', fontweight='bold')
    ax3.set_ylabel('Volatility ($)')
    ax3.set_xlabel('Date & Time')
    ax3_twin.set_ylabel('Spread (%)')
    ax3.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    print(f"📈 Comprehensive chart created for {symbol}")

def compare_all_cryptos():
    """Compare performance of all cryptocurrencies"""
    print(f"\n🔍 CRYPTOCURRENCY COMPARISON")
    print("=" * 60)
    
    symbols = ['BTC_USDT', 'ETH_USDT', 'BNB_USDT']
    comparison_data = []
    
    for symbol in symbols:
        df = load_crypto_data(symbol)
        if df is not None:
            current_price = df['close'].iloc[-1]
            start_price = df['close'].iloc[0]
            change_7d = ((current_price - start_price) / start_price) * 100
            
            # Calculate 24h change if possible
            change_24h = 0
            if len(df) >= 24:
                price_24h_ago = df['close'].iloc[-24]
                change_24h = ((current_price - price_24h_ago) / price_24h_ago) * 100
            
            volatility = df['close'].std()
            avg_volume = df['volume'].mean()
            max_price = df['high'].max()
            min_price = df['low'].min()
            
            comparison_data.append({
                'Crypto': symbol.replace('_USDT', ''),
                'Current Price': f"${current_price:,.2f}",
                '24h Change': f"{change_24h:+.2f}%",
                '7d Change': f"{change_7d:+.2f}%",
                'Volatility': f"${volatility:.2f}",
                'Avg Volume': f"{avg_volume:,.0f}",
                '7d High': f"${max_price:,.2f}",
                '7d Low': f"${min_price:,.2f}"
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Find best and worst performers
        print(f"\n🏆 PERFORMANCE HIGHLIGHTS:")
        
        # Extract numeric values for comparison
        crypto_performance = []
        for data in comparison_data:
            change_7d = float(data['7d Change'].replace('%', '').replace('+', ''))
            crypto_performance.append((data['Crypto'], change_7d))
        
        best_performer = max(crypto_performance, key=lambda x: x[1])
        worst_performer = min(crypto_performance, key=lambda x: x[1])
        
        print(f"   🥇 Best performer (7d):  {best_performer[0]} ({best_performer[1]:+.2f}%)")
        print(f"   🥉 Worst performer (7d): {worst_performer[0]} ({worst_performer[1]:+.2f}%)")
    
    return comparison_data

def generate_market_insights():
    """Generate insights about the overall market"""
    print(f"\n🧠 MARKET INSIGHTS & ANALYSIS")
    print("=" * 50)
    
    symbols = ['BTC_USDT', 'ETH_USDT', 'BNB_USDT']
    market_data = {}
    
    # Collect data for all symbols
    for symbol in symbols:
        df = load_crypto_data(symbol)
        if df is not None:
            market_data[symbol] = df
    
    if not market_data:
        print("❌ No data available for analysis")
        return
    
    # Market correlation analysis
    print("📊 CORRELATION ANALYSIS:")
    if len(market_data) >= 2:
        price_data = {}
        for symbol, df in market_data.items():
            price_data[symbol.replace('_USDT', '')] = df.set_index('timestamp')['close']
        
        correlation_df = pd.DataFrame(price_data).corr()
        print(correlation_df.round(3))
        
        # Interpretation
        if len(market_data) == 3:
            btc_eth_corr = correlation_df.loc['BTC', 'ETH']
            print(f"\n💡 BTC-ETH correlation: {btc_eth_corr:.3f}")
            if btc_eth_corr > 0.8:
                print("   🔗 Strong positive correlation - markets moving together")
            elif btc_eth_corr < 0.3:
                print("   🔀 Weak correlation - independent price movements")
            else:
                print("   ↔️ Moderate correlation - some relationship")
    
    # Market timing analysis
    print(f"\n⏰ TIMING ANALYSIS:")
    all_volumes = []
    hour_volumes = {i: [] for i in range(24)}
    
    for symbol, df in market_data.items():
        df['hour'] = df['timestamp'].dt.hour
        for hour in range(24):
            hour_data = df[df['hour'] == hour]['volume']
            if not hour_data.empty:
                hour_volumes[hour].extend(hour_data.tolist())
    
    # Find peak trading hours
    avg_hourly_volume = {hour: np.mean(volumes) if volumes else 0 
                        for hour, volumes in hour_volumes.items()}
    
    peak_hour = max(avg_hourly_volume, key=avg_hourly_volume.get)
    low_hour = min(avg_hourly_volume, key=avg_hourly_volume.get)
    
    print(f"   🔥 Peak trading hour: {peak_hour}:00 UTC")
    print(f"   😴 Lowest trading hour: {low_hour}:00 UTC")
    
    # Market trend analysis
    print(f"\n📈 TREND ANALYSIS:")
    bullish_count = 0
    bearish_count = 0
    
    for symbol, df in market_data.items():
        if len(df) >= 24:
            recent_trend = df['close'].iloc[-12:].mean() - df['close'].iloc[-24:-12].mean()
            if recent_trend > 0:
                bullish_count += 1
                trend_desc = "📈 Bullish"
            else:
                bearish_count += 1
                trend_desc = "📉 Bearish"
            
            print(f"   {symbol.replace('_USDT', ''):3}: {trend_desc}")
    
    # Overall market sentiment
    if bullish_count > bearish_count:
        print(f"\n🟢 OVERALL MARKET SENTIMENT: Bullish ({bullish_count}/{len(market_data)} coins up)")
    elif bearish_count > bullish_count:
        print(f"\n🔴 OVERALL MARKET SENTIMENT: Bearish ({bearish_count}/{len(market_data)} coins down)")
    else:
        print(f"\n🟡 OVERALL MARKET SENTIMENT: Mixed/Neutral")

def main():
    """Main analysis function"""
    print("📊 CRYPTO TRADING AI - COMPREHENSIVE DATA ANALYSIS")
    print("=" * 65)
    
    # Check if data exists
    if not os.path.exists('data'):
        print("❌ No data folder found!")
        print("💡 Run crypto_data_collector.py first!")
        return
    
    # Count available data files
    data_files = [f for f in os.listdir('data') if f.endswith('.csv')]
    if not data_files:
        print("❌ No data files found!")
        print("💡 Run crypto_data_collector.py first!")
        return
    
    print(f"✅ Found {len(data_files)} data files")
    
    # Detailed analysis for each cryptocurrency
    symbols = ['BTC_USDT', 'ETH_USDT', 'BNB_USDT']
    
    for symbol in symbols:
        analyze_crypto_detailed(symbol)
    
    # Comparison analysis
    compare_all_cryptos()
    
    # Market insights
    generate_market_insights()
    
    # Create charts
    print(f"\n📈 CREATING CHARTS...")
    try:
        create_comprehensive_chart('BTC_USDT')
        print("✅ Charts created successfully!")
    except Exception as e:
        print(f"⚠️ Chart creation failed: {e}")
        print("💡 Charts require matplotlib. Install with: pip install matplotlib")
    
    # Summary and next steps
    print(f"\n🎯 ANALYSIS COMPLETE!")
    print("=" * 30)
    print("✅ Market data analyzed")
    print("✅ Technical indicators calculated")
    print("✅ Correlations identified")
    print("✅ Trends analyzed")
    
    print(f"\n🚀 NEXT STEPS:")
    print("1. 🎯 Add more technical indicators (RSI, MACD)")
    print("2. 🎯 Build prediction models")
    print("3. 🎯 Create trading signals")
    print("4. 🎯 Implement risk management")
    
    print(f"\n🔥 Your crypto AI is getting smarter!")

if __name__ == "__main__":
    main()