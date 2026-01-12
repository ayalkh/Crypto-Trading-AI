"""
Profitability Analysis: ML Model vs Random Strategy
Compares the profitability of the ML model against a completely random trading strategy
over the course of a month with realistic trading simulation.
"""

import os
import sys
import warnings
import json
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import lightgbm as lgb

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from crypto_ai.features import FeatureEngineer


class TradingSimulator:
    """Simulates trading with realistic parameters"""
    
    def __init__(self, initial_capital=10000, position_size_pct=0.1, 
                 commission_rate=0.001, max_positions=3):
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.commission_rate = commission_rate
        self.max_positions = max_positions
        self.reset()
    
    def reset(self):
        """Reset simulator state"""
        self.capital = self.initial_capital
        self.positions = []
        self.closed_trades = []
        self.equity_curve = []
    
    def can_open_position(self):
        """Check if we can open a new position"""
        return len(self.positions) < self.max_positions and self.capital > 100
    
    def open_position(self, timestamp, price, direction, confidence=1.0):
        """Open a trading position (LONG or SHORT)"""
        if not self.can_open_position():
            return False
        
        position_size = self.capital * self.position_size_pct * confidence
        commission = position_size * self.commission_rate
        
        position = {
            'entry_time': timestamp,
            'entry_price': price,
            'direction': direction,  # 'LONG' or 'SHORT'
            'size': position_size,
            'commission_entry': commission,
            'confidence': confidence
        }
        
        self.positions.append(position)
        self.capital -= commission
        return True
    
    def close_position(self, position, timestamp, price, reason='TIME'):
        """Close a trading position"""
        exit_commission = position['size'] * self.commission_rate
        
        if position['direction'] == 'LONG':
            # Long position: profit when price goes up
            pnl = position['size'] * ((price - position['entry_price']) / position['entry_price'])
        else:
            # Short position: profit when price goes down
            pnl = position['size'] * ((position['entry_price'] - price) / position['entry_price'])
        
        # Subtract commissions
        net_pnl = pnl - position['commission_entry'] - exit_commission
        
        trade = {
            'entry_time': position['entry_time'],
            'exit_time': timestamp,
            'entry_price': position['entry_price'],
            'exit_price': price,
            'direction': position['direction'],
            'pnl': net_pnl,
            'pnl_pct': (net_pnl / position['size']) * 100,
            'hold_hours': (timestamp - position['entry_time']).total_seconds() / 3600,
            'exit_reason': reason
        }
        
        self.capital += position['size'] + net_pnl
        self.closed_trades.append(trade)
        self.positions.remove(position)
        
        return trade
    
    def update(self, timestamp, price):
        """Update positions and check for exits"""
        positions_to_close = []
        
        for position in self.positions:
            # Calculate current P&L
            if position['direction'] == 'LONG':
                pnl_pct = ((price - position['entry_price']) / position['entry_price']) * 100
            else:
                pnl_pct = ((position['entry_price'] - price) / position['entry_price']) * 100
            
            # Time-based exit (max 24 hours)
            hold_hours = (timestamp - position['entry_time']).total_seconds() / 3600
            if hold_hours >= 24:
                positions_to_close.append((position, 'TIME_24H'))
                continue
            
            # Take profit at +5%
            if pnl_pct >= 5.0:
                positions_to_close.append((position, 'TAKE_PROFIT'))
                continue
            
            # Stop loss at -3%
            if pnl_pct <= -3.0:
                positions_to_close.append((position, 'STOP_LOSS'))
                continue
        
        # Close positions
        for position, reason in positions_to_close:
            self.close_position(position, timestamp, price, reason)
        
        # Track equity
        total_equity = self.capital
        for position in self.positions:
            if position['direction'] == 'LONG':
                unrealized = position['size'] * ((price - position['entry_price']) / position['entry_price'])
            else:
                unrealized = position['size'] * ((position['entry_price'] - price) / position['entry_price'])
            total_equity += unrealized
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': total_equity,
            'cash': self.capital,
            'positions': len(self.positions)
        })
    
    def close_all_positions(self, timestamp, price):
        """Close all remaining positions"""
        for position in self.positions.copy():
            self.close_position(position, timestamp, price, 'FINAL_EXIT')
    
    def get_metrics(self):
        """Calculate performance metrics"""
        if not self.closed_trades:
            return None
        
        trades_df = pd.DataFrame(self.closed_trades)
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        
        total_pnl = trades_df['pnl'].sum()
        total_return_pct = (total_pnl / self.initial_capital) * 100
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else float('inf')
        
        max_win = trades_df['pnl'].max()
        max_loss = trades_df['pnl'].min()
        
        avg_hold_hours = trades_df['hold_hours'].mean()
        
        # Calculate max drawdown
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
        max_drawdown = equity_df['drawdown'].min()
        
        # Sharpe ratio (simplified, assuming daily returns)
        returns = trades_df['pnl_pct'].values
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if len(returns) > 1 and returns.std() > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return_pct': total_return_pct,
            'final_capital': self.capital,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_win': max_win,
            'max_loss': max_loss,
            'avg_hold_hours': avg_hold_hours,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }


def run_ml_strategy(df, symbol, timeframe, n_features=50):
    """Run ML-based trading strategy"""
    print(f"   🤖 Training ML model for {symbol} {timeframe}...")
    
    # Prepare dataframe with timestamp as index
    df_copy = df.copy()
    df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
    df_copy.set_index('timestamp', inplace=True)
    
    # Create features
    fe = FeatureEngineer()
    df_features = fe.create_features(df_copy)
    df_features.dropna(inplace=True)
    
    # Reset index to have timestamp as column again
    df_features.reset_index(inplace=True)
    
    # Create target (direction)
    df_features['target'] = (df_features['close'].shift(-1) > df_features['close']).astype(int)
    df_features.dropna(inplace=True)
    
    # Prepare features
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'target', 'timestamp']
    all_feature_cols = [col for col in df_features.columns if col not in exclude_cols]
    
    X = df_features[all_feature_cols].values
    y = df_features['target'].values
    
    # Train/test split (70% train, 30% test for last month)
    train_size = int(len(df_features) * 0.70)
    
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Feature selection
    selector = SelectKBest(score_func=mutual_info_classif, k=min(n_features, len(all_feature_cols)))
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_test_scaled = scaler.transform(X_test_sel)
    
    # Train model
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=7,
        random_state=42,
        verbose=-1,
        force_col_wise=True
    )
    
    model.fit(X_train_scaled, y_train, callbacks=[lgb.log_evaluation(0)])
    
    # Get predictions with probabilities
    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)
    
    # Create test dataframe with predictions
    test_df = df_features.iloc[train_size:].copy()
    test_df['prediction'] = predictions
    test_df['confidence'] = np.max(probabilities, axis=1)
    
    return test_df


def run_random_strategy(df):
    """Run completely random trading strategy"""
    print(f"   🎲 Generating random predictions...")
    
    test_df = df.copy()
    
    # Generate completely random predictions (50/50 chance)
    np.random.seed(42)  # For reproducibility
    test_df['prediction'] = np.random.randint(0, 2, size=len(test_df))
    test_df['confidence'] = np.random.uniform(0.5, 1.0, size=len(test_df))
    
    return test_df


def simulate_trading(df, strategy_name, simulator):
    """Simulate trading based on predictions"""
    print(f"   💰 Simulating {strategy_name} trading...")
    
    simulator.reset()
    
    for idx, row in df.iterrows():
        timestamp = row['timestamp']
        price = row['close']
        prediction = row['prediction']
        confidence = row.get('confidence', 0.7)
        
        # Update existing positions
        simulator.update(timestamp, price)
        
        # Open new position based on prediction
        if prediction == 1:  # Predicted UP
            simulator.open_position(timestamp, price, 'LONG', confidence)
        elif prediction == 0:  # Predicted DOWN
            simulator.open_position(timestamp, price, 'SHORT', confidence)
    
    # Close all positions at the end
    if len(df) > 0:
        last_row = df.iloc[-1]
        simulator.close_all_positions(last_row['timestamp'], last_row['close'])
    
    return simulator.get_metrics()


def compare_strategies(symbol='BTC/USDT', timeframe='1h', lookback_days=30):
    """Compare ML strategy vs Random strategy"""
    print(f"\n{'='*70}")
    print(f"📊 PROFITABILITY ANALYSIS: {symbol} {timeframe}")
    print(f"{'='*70}")
    
    # Load data
    db_path = 'data/ml_crypto_data.db'
    conn = sqlite3.connect(db_path)
    
    query = f"""
    SELECT timestamp, open, high, low, close, volume
    FROM price_data 
    WHERE symbol = ? AND timeframe = ? 
    AND timestamp >= datetime('now', '-{lookback_days} days')
    ORDER BY timestamp
    """
    
    df = pd.read_sql_query(query, conn, params=(symbol, timeframe))
    conn.close()
    
    if df.empty or len(df) < 100:
        print(f"❌ Insufficient data for {symbol} {timeframe}")
        return None
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"   📅 Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   📊 Total candles: {len(df)}")
    print(f"   💵 Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    # Run ML strategy
    print(f"\n   🤖 ML STRATEGY")
    print(f"   {'-'*60}")
    ml_df = run_ml_strategy(df, symbol, timeframe)
    ml_simulator = TradingSimulator(initial_capital=10000)
    ml_metrics = simulate_trading(ml_df, "ML", ml_simulator)
    
    # Run Random strategy (on same test period)
    print(f"\n   🎲 RANDOM STRATEGY")
    print(f"   {'-'*60}")
    # Use same test period as ML
    train_size = int(len(df) * 0.70)
    random_df = df.iloc[train_size:].copy()
    random_df['timestamp'] = pd.to_datetime(random_df['timestamp'])
    random_df = run_random_strategy(random_df)
    random_simulator = TradingSimulator(initial_capital=10000)
    random_metrics = simulate_trading(random_df, "Random", random_simulator)
    
    return {
        'symbol': symbol,
        'timeframe': timeframe,
        'ml_metrics': ml_metrics,
        'random_metrics': random_metrics,
        'ml_equity': ml_simulator.equity_curve,
        'random_equity': random_simulator.equity_curve
    }


def display_comparison(results):
    """Display detailed comparison between strategies"""
    if results is None:
        return
    
    ml = results['ml_metrics']
    random = results['random_metrics']
    
    print(f"\n{'='*70}")
    print(f"📈 PERFORMANCE COMPARISON")
    print(f"{'='*70}\n")
    
    # Create comparison table
    metrics = [
        ('Total Trades', 'total_trades', ''),
        ('Win Rate', 'win_rate', '%'),
        ('Total Return', 'total_return_pct', '%'),
        ('Final Capital', 'final_capital', '$'),
        ('Profit Factor', 'profit_factor', 'x'),
        ('Avg Win', 'avg_win', '$'),
        ('Avg Loss', 'avg_loss', '$'),
        ('Max Win', 'max_win', '$'),
        ('Max Loss', 'max_loss', '$'),
        ('Max Drawdown', 'max_drawdown', '%'),
        ('Sharpe Ratio', 'sharpe_ratio', ''),
        ('Avg Hold Time', 'avg_hold_hours', 'h'),
    ]
    
    print(f"{'Metric':<20} {'ML Strategy':<20} {'Random Strategy':<20} {'Difference':<15}")
    print("-" * 75)
    
    for metric_name, metric_key, unit in metrics:
        ml_val = ml[metric_key]
        random_val = random[metric_key]
        
        if unit == '$':
            ml_str = f"${ml_val:,.2f}"
            random_str = f"${random_val:,.2f}"
            diff = ml_val - random_val
            diff_str = f"${diff:+,.2f}"
        elif unit == '%':
            ml_str = f"{ml_val:.2f}%"
            random_str = f"{random_val:.2f}%"
            diff = ml_val - random_val
            diff_str = f"{diff:+.2f}%"
        elif unit == 'x':
            ml_str = f"{ml_val:.2f}x" if ml_val != float('inf') else "∞"
            random_str = f"{random_val:.2f}x" if random_val != float('inf') else "∞"
            diff_str = "N/A"
        elif unit == 'h':
            ml_str = f"{ml_val:.1f}h"
            random_str = f"{random_val:.1f}h"
            diff = ml_val - random_val
            diff_str = f"{diff:+.1f}h"
        else:
            ml_str = f"{ml_val:.2f}"
            random_str = f"{random_val:.2f}"
            diff = ml_val - random_val
            diff_str = f"{diff:+.2f}"
        
        print(f"{metric_name:<20} {ml_str:<20} {random_str:<20} {diff_str:<15}")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"🎯 SUMMARY")
    print(f"{'='*70}\n")
    
    ml_return = ml['total_return_pct']
    random_return = random['total_return_pct']
    improvement = ml_return - random_return
    
    print(f"💰 ML Strategy Return:     {ml_return:+.2f}%")
    print(f"🎲 Random Strategy Return: {random_return:+.2f}%")
    print(f"📊 Improvement:            {improvement:+.2f}%")
    
    if improvement > 5:
        verdict = "🌟 EXCELLENT - ML model significantly outperforms random!"
        color = "green"
    elif improvement > 2:
        verdict = "✅ GOOD - ML model shows meaningful improvement"
        color = "green"
    elif improvement > 0:
        verdict = "👍 POSITIVE - ML model slightly better than random"
        color = "yellow"
    elif improvement > -2:
        verdict = "⚠️ NEUTRAL - ML model performs similarly to random"
        color = "yellow"
    else:
        verdict = "❌ POOR - ML model underperforms random strategy"
        color = "red"
    
    print(f"\n🏆 VERDICT: {verdict}")
    
    # Additional insights
    print(f"\n📌 KEY INSIGHTS:")
    
    if ml['win_rate'] > random['win_rate']:
        print(f"   ✓ ML has {ml['win_rate'] - random['win_rate']:.1f}% higher win rate")
    else:
        print(f"   ✗ ML has {random['win_rate'] - ml['win_rate']:.1f}% lower win rate")
    
    if ml['profit_factor'] > random['profit_factor']:
        print(f"   ✓ ML has better profit factor (risk/reward)")
    else:
        print(f"   ✗ Random has better profit factor")
    
    if ml['max_drawdown'] > random['max_drawdown']:
        print(f"   ✗ ML has {abs(ml['max_drawdown'] - random['max_drawdown']):.1f}% worse drawdown")
    else:
        print(f"   ✓ ML has {abs(ml['max_drawdown'] - random['max_drawdown']):.1f}% better drawdown")
    
    if ml['sharpe_ratio'] > random['sharpe_ratio']:
        print(f"   ✓ ML has better risk-adjusted returns (Sharpe: {ml['sharpe_ratio']:.2f} vs {random['sharpe_ratio']:.2f})")
    else:
        print(f"   ✗ Random has better risk-adjusted returns")


def run_full_analysis():
    """Run complete profitability analysis across multiple assets"""
    print("\n" + "="*70)
    print("🚀 PROFITABILITY ANALYSIS: ML vs RANDOM STRATEGY")
    print("="*70)
    print("\n📊 Comparing ML model predictions against completely random trading")
    print("💰 Initial Capital: $10,000 per strategy")
    print("📅 Period: Last 30 days of available data")
    print("⚙️  Settings: 10% position size, 0.1% commission, max 3 positions")
    print("🎯 Exit Rules: 24h max hold, +5% take profit, -3% stop loss")
    
    symbols = ['BTC/USDT', 'ETH/USDT']
    timeframes = ['1h', '4h']
    
    all_results = []
    
    for symbol in symbols:
        for timeframe in timeframes:
            try:
                results = compare_strategies(symbol, timeframe, lookback_days=30)
                if results:
                    display_comparison(results)
                    all_results.append(results)
            except Exception as e:
                print(f"\n❌ Error analyzing {symbol} {timeframe}: {str(e)}")
                continue
    
    # Overall summary
    if all_results:
        print(f"\n{'='*70}")
        print(f"📊 OVERALL SUMMARY ACROSS ALL TESTS")
        print(f"{'='*70}\n")
        
        ml_returns = [r['ml_metrics']['total_return_pct'] for r in all_results]
        random_returns = [r['random_metrics']['total_return_pct'] for r in all_results]
        
        ml_avg = np.mean(ml_returns)
        random_avg = np.mean(random_returns)
        improvement = ml_avg - random_avg
        
        print(f"📈 Average ML Return:     {ml_avg:+.2f}%")
        print(f"🎲 Average Random Return: {random_avg:+.2f}%")
        print(f"📊 Average Improvement:   {improvement:+.2f}%")
        
        ml_wins = sum(1 for r in all_results if r['ml_metrics']['total_return_pct'] > r['random_metrics']['total_return_pct'])
        total_tests = len(all_results)
        
        print(f"\n🏆 ML won in {ml_wins}/{total_tests} tests ({ml_wins/total_tests*100:.0f}%)")
        
        if improvement > 3:
            final_verdict = "🌟 EXCELLENT - Your ML model is significantly profitable!"
        elif improvement > 1:
            final_verdict = "✅ GOOD - Your ML model shows promise"
        elif improvement > -1:
            final_verdict = "⚠️ NEUTRAL - Model needs improvement"
        else:
            final_verdict = "❌ NEEDS WORK - Model underperforms random"
        
        print(f"\n🎯 FINAL VERDICT: {final_verdict}")
        
        # Save results
        output = {
            'timestamp': datetime.now().isoformat(),
            'analysis_type': 'ml_vs_random_profitability',
            'results': []
        }
        
        for r in all_results:
            output['results'].append({
                'symbol': r['symbol'],
                'timeframe': r['timeframe'],
                'ml_metrics': {k: float(v) if v != float('inf') else None for k, v in r['ml_metrics'].items()},
                'random_metrics': {k: float(v) if v != float('inf') else None for k, v in r['random_metrics'].items()}
            })
        
        output_path = 'ml_reports/profitability_analysis.json'
        os.makedirs('ml_reports', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_path}")
    
    print(f"\n✅ Analysis complete!")


if __name__ == "__main__":
    run_full_analysis()
