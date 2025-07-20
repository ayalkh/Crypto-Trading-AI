"""
Crypto Trading Performance Tracker
Analyzes signal accuracy, win rates, and theoretical P&L
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import os
import warnings
warnings.filterwarnings('ignore')

class PerformanceTracker:
    def __init__(self, initial_capital=10000):
        """Initialize performance tracker"""
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trades = []
        self.signals_history = []
        self.performance_metrics = {}
        
        # Trading parameters
        self.position_size_pct = 0.1  # 10% of capital per trade
        self.max_positions = 3        # Maximum concurrent positions
        self.commission_rate = 0.001  # 0.1% commission per trade
        
        # Performance tracking
        self.active_positions = []
        self.closed_positions = []
        
        print("📊 Performance Tracker Initialized")
        print(f"💰 Starting Capital: ${self.initial_capital:,.2f}")
    
    def analyze_signal_performance(self, df, symbol='BTC_USDT'):
        """Analyze historical signal performance"""
        print(f"\n🔍 Analyzing Signal Performance for {symbol}")
        print("=" * 50)
        
        if 'ultimate_signal' not in df.columns:
            print("❌ No ultimate signals found. Run ultimate_signals.py first!")
            return None
        
        # Extract signals and outcomes
        signal_analysis = []
        
        for i in range(len(df) - 4):  # Leave room for future lookback
            current_signal = df['ultimate_signal'].iloc[i]
            current_price = df['close'].iloc[i]
            current_confidence = df.get('confidence', pd.Series([50]*len(df))).iloc[i]
            timestamp = df['timestamp'].iloc[i]
            
            if current_signal in ['BUY', 'STRONG_BUY', 'SELL', 'STRONG_SELL']:
                # Look ahead to see what happened
                outcomes = self.calculate_signal_outcomes(df, i, current_signal, current_price)
                
                signal_record = {
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'signal': current_signal,
                    'price': current_price,
                    'confidence': current_confidence,
                    **outcomes
                }
                
                signal_analysis.append(signal_record)
        
        # Convert to DataFrame for analysis
        signals_df = pd.DataFrame(signal_analysis)
        
        if signals_df.empty:
            print("⚠️ No signals found for analysis")
            return None
        
        # Calculate performance metrics
        performance = self.calculate_performance_metrics(signals_df)
        
        # Display results
        self.display_signal_performance(signals_df, performance, symbol)
        
        return signals_df, performance
    
    def calculate_signal_outcomes(self, df, signal_index, signal_type, entry_price):
        """Calculate what happened after each signal"""
        outcomes = {}
        
        # Define lookhead periods (hours)
        periods = [1, 4, 8, 24]
        
        for period in periods:
            end_index = min(signal_index + period, len(df) - 1)
            
            if end_index > signal_index:
                # Get price range in the period
                period_data = df.iloc[signal_index:end_index + 1]
                max_price = period_data['high'].max()
                min_price = period_data['low'].min()
                end_price = df['close'].iloc[end_index]
                
                # Calculate returns
                max_return = (max_price - entry_price) / entry_price * 100
                min_return = (min_price - entry_price) / entry_price * 100
                end_return = (end_price - entry_price) / entry_price * 100
                
                # Determine if signal was correct
                if signal_type in ['BUY', 'STRONG_BUY']:
                    # For buy signals, we want price to go up
                    success = end_return > 0
                    best_outcome = max_return
                    worst_outcome = min_return
                else:  # SELL, STRONG_SELL
                    # For sell signals, we want price to go down (short)
                    success = end_return < 0
                    best_outcome = -min_return  # Profit from shorting
                    worst_outcome = -max_return
                
                outcomes.update({
                    f'success_{period}h': success,
                    f'return_{period}h': end_return if signal_type in ['BUY', 'STRONG_BUY'] else -end_return,
                    f'max_return_{period}h': best_outcome,
                    f'min_return_{period}h': worst_outcome,
                    f'end_price_{period}h': end_price
                })
            else:
                # Not enough data for this period
                outcomes.update({
                    f'success_{period}h': None,
                    f'return_{period}h': None,
                    f'max_return_{period}h': None,
                    f'min_return_{period}h': None,
                    f'end_price_{period}h': None
                })
        
        return outcomes
    
    def calculate_performance_metrics(self, signals_df):
        """Calculate comprehensive performance metrics"""
        metrics = {}
        
        # Overall statistics
        total_signals = len(signals_df)
        
        # Win rates by time period
        periods = [1, 4, 8, 24]
        
        for period in periods:
            success_col = f'success_{period}h'
            return_col = f'return_{period}h'
            
            if success_col in signals_df.columns:
                valid_signals = signals_df[signals_df[success_col].notna()]
                
                if not valid_signals.empty:
                    win_rate = valid_signals[success_col].mean() * 100
                    avg_return = valid_signals[return_col].mean()
                    winning_trades = valid_signals[valid_signals[success_col] == True]
                    losing_trades = valid_signals[valid_signals[success_col] == False]
                    
                    avg_win = winning_trades[return_col].mean() if not winning_trades.empty else 0
                    avg_loss = losing_trades[return_col].mean() if not losing_trades.empty else 0
                    
                    metrics[f'{period}h'] = {
                        'total_signals': len(valid_signals),
                        'win_rate': win_rate,
                        'avg_return': avg_return,
                        'avg_win': avg_win,
                        'avg_loss': avg_loss,
                        'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
                    }
        
        # Performance by signal type
        for signal_type in ['BUY', 'STRONG_BUY', 'SELL', 'STRONG_SELL']:
            signal_subset = signals_df[signals_df['signal'] == signal_type]
            
            if not signal_subset.empty:
                # Use 4-hour period as standard
                success_col = 'success_4h'
                return_col = 'return_4h'
                
                if success_col in signal_subset.columns:
                    valid_subset = signal_subset[signal_subset[success_col].notna()]
                    
                    if not valid_subset.empty:
                        metrics[signal_type] = {
                            'count': len(valid_subset),
                            'win_rate': valid_subset[success_col].mean() * 100,
                            'avg_return': valid_subset[return_col].mean(),
                            'best_return': valid_subset[return_col].max(),
                            'worst_return': valid_subset[return_col].min()
                        }
        
        # Performance by confidence level
        confidence_ranges = [
            (90, 100, '90%+'),
            (80, 90, '80-90%'),
            (70, 80, '70-80%'),
            (60, 70, '60-70%'),
            (0, 60, '<60%')
        ]
        
        for min_conf, max_conf, label in confidence_ranges:
            conf_subset = signals_df[
                (signals_df['confidence'] >= min_conf) & 
                (signals_df['confidence'] < max_conf)
            ]
            
            if not conf_subset.empty:
                success_col = 'success_4h'
                return_col = 'return_4h'
                
                if success_col in conf_subset.columns:
                    valid_subset = conf_subset[conf_subset[success_col].notna()]
                    
                    if not valid_subset.empty:
                        metrics[f'confidence_{label}'] = {
                            'count': len(valid_subset),
                            'win_rate': valid_subset[success_col].mean() * 100,
                            'avg_return': valid_subset[return_col].mean()
                        }
        
        return metrics
    
    def display_signal_performance(self, signals_df, metrics, symbol):
        """Display comprehensive performance analysis"""
        print(f"\n📊 SIGNAL PERFORMANCE ANALYSIS - {symbol}")
        print("=" * 60)
        
        total_signals = len(signals_df)
        print(f"📈 Total Signals Analyzed: {total_signals}")
        
        if total_signals == 0:
            print("⚠️ No signals to analyze")
            return
        
        # Time-based performance
        print(f"\n⏰ PERFORMANCE BY TIME HORIZON:")
        print("-" * 40)
        
        periods = [1, 4, 8, 24]
        for period in periods:
            if f'{period}h' in metrics:
                m = metrics[f'{period}h']
                print(f"📅 {period:2d} Hour(s): {m['win_rate']:5.1f}% win rate | "
                      f"Avg: {m['avg_return']:+6.2f}% | "
                      f"Signals: {m['total_signals']:3d}")
        
        # Signal type performance
        print(f"\n🎯 PERFORMANCE BY SIGNAL TYPE:")
        print("-" * 40)
        
        signal_types = ['STRONG_BUY', 'BUY', 'SELL', 'STRONG_SELL']
        for signal_type in signal_types:
            if signal_type in metrics:
                m = metrics[signal_type]
                print(f"🔥 {signal_type:11}: {m['win_rate']:5.1f}% win rate | "
                      f"Avg: {m['avg_return']:+6.2f}% | "
                      f"Count: {m['count']:3d}")
        
        # Confidence level performance
        print(f"\n🔥 PERFORMANCE BY CONFIDENCE LEVEL:")
        print("-" * 40)
        
        conf_levels = ['90%+', '80-90%', '70-80%', '60-70%', '<60%']
        for level in conf_levels:
            key = f'confidence_{level}'
            if key in metrics:
                m = metrics[key]
                print(f"📊 {level:8}: {m['win_rate']:5.1f}% win rate | "
                      f"Avg: {m['avg_return']:+6.2f}% | "
                      f"Count: {m['count']:3d}")
        
        # Best and worst trades
        if 'return_4h' in signals_df.columns:
            best_trade = signals_df.loc[signals_df['return_4h'].idxmax()]
            worst_trade = signals_df.loc[signals_df['return_4h'].idxmin()]
            
            print(f"\n🏆 BEST & WORST TRADES (4-hour horizon):")
            print("-" * 40)
            print(f"🥇 Best:  {best_trade['signal']} at ${best_trade['price']:.2f} → "
                  f"{best_trade['return_4h']:+.2f}% ({best_trade['timestamp'].strftime('%Y-%m-%d %H:%M')})")
            print(f"🥉 Worst: {worst_trade['signal']} at ${worst_trade['price']:.2f} → "
                  f"{worst_trade['return_4h']:+.2f}% ({worst_trade['timestamp'].strftime('%Y-%m-%d %H:%M')})")
    
    def simulate_trading_performance(self, signals_df, symbol='BTC_USDT'):
        """Simulate actual trading with your signals"""
        print(f"\n💰 TRADING SIMULATION - {symbol}")
        print("=" * 50)
        
        if signals_df is None or signals_df.empty:
            print("❌ No signals data available for simulation")
            return None
        
        # Reset simulation
        self.current_capital = self.initial_capital
        self.active_positions = []
        self.closed_positions = []
        
        # Sort signals by timestamp
        signals_df = signals_df.sort_values('timestamp').reset_index(drop=True)
        
        # Simulate each signal
        for _, signal_row in signals_df.iterrows():
            self.process_trading_signal(signal_row)
        
        # Close any remaining positions
        self.close_all_positions()
        
        # Calculate final performance
        final_performance = self.calculate_trading_performance()
        
        # Display results
        self.display_trading_results(final_performance, symbol)
        
        return final_performance
    
    def process_trading_signal(self, signal_row):
        """Process a single trading signal"""
        signal_type = signal_row['signal']
        price = signal_row['price']
        timestamp = signal_row['timestamp']
        confidence = signal_row['confidence']
        
        # Position sizing based on confidence
        base_size = self.current_capital * self.position_size_pct
        confidence_multiplier = confidence / 100  # Scale by confidence
        position_size = base_size * confidence_multiplier
        
        # Check if we can open new position
        if len(self.active_positions) < self.max_positions and position_size > 100:
            
            if signal_type in ['BUY', 'STRONG_BUY']:
                # Open long position
                shares = position_size / price
                commission = position_size * self.commission_rate
                
                position = {
                    'type': 'LONG',
                    'entry_time': timestamp,
                    'entry_price': price,
                    'shares': shares,
                    'position_value': position_size,
                    'commission_paid': commission,
                    'signal_type': signal_type,
                    'confidence': confidence
                }
                
                self.active_positions.append(position)
                self.current_capital -= (position_size + commission)
                
            elif signal_type in ['SELL', 'STRONG_SELL']:
                # Close any long positions or open short
                self.close_long_positions(price, timestamp)
        
        # Check for position exits based on time or profit targets
        self.check_position_exits(signal_row)
    
    def close_long_positions(self, current_price, timestamp):
        """Close all long positions"""
        positions_to_close = [pos for pos in self.active_positions if pos['type'] == 'LONG']
        
        for position in positions_to_close:
            self.close_position(position, current_price, timestamp, 'SIGNAL_EXIT')
    
    def check_position_exits(self, signal_row):
        """Check if any positions should be exited"""
        current_price = signal_row['price']
        timestamp = signal_row['timestamp']
        
        positions_to_close = []
        
        for position in self.active_positions:
            # Time-based exit (8 hours max)
            time_diff = (timestamp - position['entry_time']).total_seconds() / 3600
            
            if time_diff >= 8:  # 8-hour max hold
                positions_to_close.append((position, 'TIME_EXIT'))
                continue
            
            # Profit/Loss exit
            if position['type'] == 'LONG':
                pnl_pct = (current_price - position['entry_price']) / position['entry_price'] * 100
                
                # Take profit at +5% or stop loss at -3%
                if pnl_pct >= 5.0:
                    positions_to_close.append((position, 'TAKE_PROFIT'))
                elif pnl_pct <= -3.0:
                    positions_to_close.append((position, 'STOP_LOSS'))
        
        # Close positions
        for position, exit_reason in positions_to_close:
            self.close_position(position, current_price, timestamp, exit_reason)
    
    def close_position(self, position, exit_price, exit_time, exit_reason):
        """Close a trading position"""
        exit_value = position['shares'] * exit_price
        commission = exit_value * self.commission_rate
        
        total_return = exit_value - position['position_value'] - position['commission_paid'] - commission
        return_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100
        
        closed_position = {
            **position,
            'exit_time': exit_time,
            'exit_price': exit_price,
            'exit_value': exit_value,
            'exit_commission': commission,
            'total_return': total_return,
            'return_pct': return_pct,
            'exit_reason': exit_reason,
            'hold_time_hours': (exit_time - position['entry_time']).total_seconds() / 3600
        }
        
        self.closed_positions.append(closed_position)
        self.active_positions.remove(position)
        self.current_capital += (exit_value - commission)
    
    def close_all_positions(self):
        """Close all remaining active positions"""
        if self.active_positions:
            # Use last known price (approximate)
            last_position = self.closed_positions[-1] if self.closed_positions else None
            if last_position:
                exit_price = last_position['exit_price']
                exit_time = last_position['exit_time']
                
                for position in self.active_positions.copy():
                    self.close_position(position, exit_price, exit_time, 'FINAL_EXIT')
    
    def calculate_trading_performance(self):
        """Calculate comprehensive trading performance metrics"""
        if not self.closed_positions:
            return None
        
        trades_df = pd.DataFrame(self.closed_positions)
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['return_pct'] > 0])
        losing_trades = len(trades_df[trades_df['return_pct'] < 0])
        
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        
        # Return metrics
        total_return = self.current_capital - self.initial_capital
        total_return_pct = total_return / self.initial_capital * 100
        
        avg_return_per_trade = trades_df['return_pct'].mean()
        avg_winning_trade = trades_df[trades_df['return_pct'] > 0]['return_pct'].mean()
        avg_losing_trade = trades_df[trades_df['return_pct'] < 0]['return_pct'].mean()
        
        # Risk metrics
        max_return = trades_df['return_pct'].max()
        max_loss = trades_df['return_pct'].min()
        
        profit_factor = (
            abs(avg_winning_trade * winning_trades) / abs(avg_losing_trade * losing_trades)
            if losing_trades > 0 and not pd.isna(avg_losing_trade) else float('inf')
        )
        
        # Time metrics
        avg_hold_time = trades_df['hold_time_hours'].mean()
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'avg_return_per_trade': avg_return_per_trade,
            'avg_winning_trade': avg_winning_trade,
            'avg_losing_trade': avg_losing_trade,
            'max_return': max_return,
            'max_loss': max_loss,
            'profit_factor': profit_factor,
            'avg_hold_time': avg_hold_time,
            'final_capital': self.current_capital
        }
    
    def display_trading_results(self, performance, symbol):
        """Display comprehensive trading simulation results"""
        if performance is None:
            print("❌ No trading performance data available")
            return
        
        print(f"💼 TRADING SIMULATION RESULTS - {symbol}")
        print("=" * 50)
        
        print(f"💰 CAPITAL & RETURNS:")
        print(f"   Initial Capital:    ${self.initial_capital:>10,.2f}")
        print(f"   Final Capital:      ${performance['final_capital']:>10,.2f}")
        print(f"   Total Return:       ${performance['total_return']:>10,.2f}")
        print(f"   Return %:           {performance['total_return_pct']:>10.2f}%")
        
        print(f"\n📊 TRADE STATISTICS:")
        print(f"   Total Trades:       {performance['total_trades']:>10d}")
        print(f"   Winning Trades:     {performance['winning_trades']:>10d}")
        print(f"   Losing Trades:      {performance['losing_trades']:>10d}")
        print(f"   Win Rate:           {performance['win_rate']:>10.1f}%")
        
        print(f"\n📈 PERFORMANCE METRICS:")
        print(f"   Avg Return/Trade:   {performance['avg_return_per_trade']:>10.2f}%")
        print(f"   Avg Winning Trade:  {performance['avg_winning_trade']:>10.2f}%")
        print(f"   Avg Losing Trade:   {performance['avg_losing_trade']:>10.2f}%")
        print(f"   Best Trade:         {performance['max_return']:>10.2f}%")
        print(f"   Worst Trade:        {performance['max_loss']:>10.2f}%")
        print(f"   Profit Factor:      {performance['profit_factor']:>10.2f}")
        
        print(f"\n⏰ TIMING METRICS:")
        print(f"   Avg Hold Time:      {performance['avg_hold_time']:>10.1f} hours")
        
        # Performance rating
        print(f"\n🏆 SYSTEM RATING:")
        
        rating_score = 0
        if performance['win_rate'] >= 60:
            rating_score += 25
        elif performance['win_rate'] >= 50:
            rating_score += 15
        
        if performance['total_return_pct'] >= 10:
            rating_score += 25
        elif performance['total_return_pct'] >= 5:
            rating_score += 15
        elif performance['total_return_pct'] >= 0:
            rating_score += 10
        
        if performance['profit_factor'] >= 2:
            rating_score += 25
        elif performance['profit_factor'] >= 1.5:
            rating_score += 15
        elif performance['profit_factor'] >= 1:
            rating_score += 10
        
        if performance['avg_hold_time'] <= 8:
            rating_score += 25
        elif performance['avg_hold_time'] <= 12:
            rating_score += 15
        
        if rating_score >= 85:
            rating = "🌟 EXCELLENT - Ready for live trading!"
        elif rating_score >= 70:
            rating = "🔥 VERY GOOD - Minor tweaks needed"
        elif rating_score >= 55:
            rating = "👍 GOOD - Some improvements needed"
        elif rating_score >= 40:
            rating = "⚠️ FAIR - Significant improvements needed"
        else:
            rating = "❌ POOR - Major strategy revision needed"
        
        print(f"   Overall Rating:     {rating}")
        print(f"   Score:              {rating_score}/100")

def run_performance_analysis():
    """Main function to run complete performance analysis"""
    print("🚀 CRYPTO TRADING PERFORMANCE TRACKER")
    print("=" * 55)
    print("📊 Analyzing your trading signals for accuracy and profitability")
    print("=" * 55)
    
    symbols = ['BTC_USDT', 'ETH_USDT', 'BNB_USDT']
    tracker = PerformanceTracker(initial_capital=10000)  # $10k starting capital
    
    all_results = {}
    
    for symbol in symbols:
        print(f"\n{'='*20} {symbol} {'='*20}")
        
        # Load data with signals
        filename = f"data/{symbol}_1h_7days.csv"
        
        try:
            df = pd.read_csv(filename)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Try to get signals (run ultimate_signals first if needed)
            if 'ultimate_signal' not in df.columns:
                print("⚠️ Running ultimate signals analysis first...")
                try:
                    from ultimate_signals_no_charts import load_and_analyze_ultimate_signals
                    df, _, _ = load_and_analyze_ultimate_signals(symbol)
                except:
                    print("❌ Could not load ultimate signals. Run ultimate_signals.py first!")
                    continue
            
            if df is not None and 'ultimate_signal' in df.columns:
                # Analyze signal performance
                signals_df, performance_metrics = tracker.analyze_signal_performance(df, symbol)
                
                if signals_df is not None:
                    # Simulate trading
                    trading_performance = tracker.simulate_trading_performance(signals_df, symbol)
                    
                    all_results[symbol] = {
                        'signals': signals_df,
                        'signal_metrics': performance_metrics,
                        'trading_performance': trading_performance
                    }
                else:
                    print(f"⚠️ No valid signals found for {symbol}")
            else:
                print(f"❌ Could not load data for {symbol}")
                
        except FileNotFoundError:
            print(f"❌ Data file not found for {symbol}")
            print("💡 Run crypto_data_collector.py first!")
    
    # Summary across all cryptos
    if all_results:
        print(f"\n🌟 OVERALL PERFORMANCE SUMMARY")
        print("=" * 60)
        
        total_signals = sum(len(result['signals']) for result in all_results.values() if result['signals'] is not None)
        
        overall_win_rates = []
        overall_returns = []
        
        for symbol, result in all_results.items():
            if result['trading_performance']:
                perf = result['trading_performance']
                overall_win_rates.append(perf['win_rate'])
                overall_returns.append(perf['total_return_pct'])
                
                print(f"📊 {symbol:8}: {perf['win_rate']:5.1f}% win rate | "
                      f"{perf['total_return_pct']:+6.2f}% return | "
                      f"{perf['total_trades']:3d} trades")
        
        if overall_win_rates:
            avg_win_rate = np.mean(overall_win_rates)
            avg_return = np.mean(overall_returns)
            
            print(f"\n🎯 AVERAGE PERFORMANCE:")
            print(f"   Average Win Rate:   {avg_win_rate:6.1f}%")
            print(f"   Average Return:     {avg_return:+6.2f}%")
            print(f"   Total Signals:      {total_signals:6d}")
            
            # Overall recommendation
            if avg_win_rate >= 60 and avg_return >= 5:
                recommendation = "🌟 EXCELLENT - Your system is ready for live trading!"
            elif avg_win_rate >= 55 and avg_return >= 3:
                recommendation = "🔥 VERY GOOD - Consider small live positions"
            elif avg_win_rate >= 50 and avg_return >= 1:
                recommendation = "👍 GOOD - Continue paper trading and optimization"
            else:
                recommendation = "⚠️ NEEDS WORK - Focus on improving signal accuracy"
            
            print(f"\n💡 RECOMMENDATION: {recommendation}")
    
    print(f"\n🎉 Performance Analysis Complete!")
    print(f"💡 Use these insights to optimize your trading strategy!")

if __name__ == "__main__":
    run_performance_analysis()