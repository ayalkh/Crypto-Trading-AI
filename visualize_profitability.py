"""
Create visualization for profitability analysis results
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10

# Load results
with open('ml_reports/profitability_analysis.json', 'r') as f:
    data = json.load(f)

results = data['results']

# Prepare data
symbols_tf = [f"{r['symbol']}\n{r['timeframe']}" for r in results]
ml_returns = [r['ml_metrics']['total_return_pct'] for r in results]
random_returns = [r['random_metrics']['total_return_pct'] for r in results]
ml_win_rates = [r['ml_metrics']['win_rate'] for r in results]
random_win_rates = [r['random_metrics']['win_rate'] for r in results]
ml_sharpe = [r['ml_metrics']['sharpe_ratio'] for r in results]
random_sharpe = [r['random_metrics']['sharpe_ratio'] for r in results]
ml_trades = [r['ml_metrics']['total_trades'] for r in results]
random_trades = [r['random_metrics']['total_trades'] for r in results]

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('ML Model vs Random Strategy - Profitability Analysis\n30-Day Period (Dec 2025 - Jan 2026)', 
             fontsize=16, fontweight='bold', y=0.995)

# 1. Returns Comparison
ax1 = axes[0, 0]
x = np.arange(len(symbols_tf))
width = 0.35
bars1 = ax1.bar(x - width/2, ml_returns, width, label='ML Strategy', color='#3498db', alpha=0.8)
bars2 = ax1.bar(x + width/2, random_returns, width, label='Random Strategy', color='#e74c3c', alpha=0.8)
ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax1.set_xlabel('Asset & Timeframe')
ax1.set_ylabel('Return (%)')
ax1.set_title('Total Returns Comparison', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(symbols_tf, fontsize=9)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom' if height > 0 else 'top',
                fontsize=8)

# 2. Win Rate Comparison
ax2 = axes[0, 1]
bars1 = ax2.bar(x - width/2, ml_win_rates, width, label='ML Strategy', color='#3498db', alpha=0.8)
bars2 = ax2.bar(x + width/2, random_win_rates, width, label='Random Strategy', color='#e74c3c', alpha=0.8)
ax2.axhline(y=50, color='orange', linestyle='--', linewidth=0.8, alpha=0.7, label='50% (Break-even)')
ax2.set_xlabel('Asset & Timeframe')
ax2.set_ylabel('Win Rate (%)')
ax2.set_title('Win Rate Comparison', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(symbols_tf, fontsize=9)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom',
                fontsize=8)

# 3. Sharpe Ratio Comparison
ax3 = axes[0, 2]
bars1 = ax3.bar(x - width/2, ml_sharpe, width, label='ML Strategy', color='#3498db', alpha=0.8)
bars2 = ax3.bar(x + width/2, random_sharpe, width, label='Random Strategy', color='#e74c3c', alpha=0.8)
ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax3.axhline(y=1, color='green', linestyle='--', linewidth=0.8, alpha=0.5, label='Good (>1)')
ax3.set_xlabel('Asset & Timeframe')
ax3.set_ylabel('Sharpe Ratio')
ax3.set_title('Risk-Adjusted Returns (Sharpe Ratio)', fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(symbols_tf, fontsize=9)
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# 4. Number of Trades
ax4 = axes[1, 0]
bars1 = ax4.bar(x - width/2, ml_trades, width, label='ML Strategy', color='#3498db', alpha=0.8)
bars2 = ax4.bar(x + width/2, random_trades, width, label='Random Strategy', color='#e74c3c', alpha=0.8)
ax4.set_xlabel('Asset & Timeframe')
ax4.set_ylabel('Number of Trades')
ax4.set_title('Trading Activity', fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(symbols_tf, fontsize=9)
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom',
                fontsize=8)

# 5. Performance Difference (ML - Random)
ax5 = axes[1, 1]
differences = [ml - rand for ml, rand in zip(ml_returns, random_returns)]
colors = ['green' if d > 0 else 'red' for d in differences]
bars = ax5.bar(x, differences, color=colors, alpha=0.7)
ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax5.set_xlabel('Asset & Timeframe')
ax5.set_ylabel('Return Difference (%)')
ax5.set_title('ML Advantage over Random\n(Positive = ML Better)', fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(symbols_tf, fontsize=9)
ax5.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:+.2f}%', ha='center', va='bottom' if height > 0 else 'top',
            fontsize=8, fontweight='bold')

# 6. Summary Statistics
ax6 = axes[1, 2]
ax6.axis('off')

# Calculate summary stats
avg_ml_return = np.mean(ml_returns)
avg_random_return = np.mean(random_returns)
avg_improvement = avg_ml_return - avg_random_return
ml_wins = sum(1 for d in differences if d > 0)
total_tests = len(differences)

summary_text = f"""
OVERALL SUMMARY
{'='*40}

Average Returns:
  • ML Strategy:      {avg_ml_return:+.2f}%
  • Random Strategy:  {avg_random_return:+.2f}%
  • Difference:       {avg_improvement:+.2f}%

Win Rate:
  • ML won {ml_wins}/{total_tests} tests ({ml_wins/total_tests*100:.0f}%)

Average Win Rates:
  • ML Strategy:      {np.mean(ml_win_rates):.1f}%
  • Random Strategy:  {np.mean(random_win_rates):.1f}%

Average Sharpe Ratio:
  • ML Strategy:      {np.mean(ml_sharpe):.2f}
  • Random Strategy:  {np.mean(random_sharpe):.2f}

VERDICT:
"""

if avg_improvement > 3:
    verdict = "🌟 EXCELLENT\nML significantly outperforms!"
    color = 'green'
elif avg_improvement > 1:
    verdict = "✅ GOOD\nML shows improvement"
    color = 'darkgreen'
elif avg_improvement > -1:
    verdict = "⚠️ NEUTRAL\nSimilar performance"
    color = 'orange'
else:
    verdict = "❌ NEEDS WORK\nML underperforms random"
    color = 'red'

summary_text += f"  {verdict}"

ax6.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', 
         facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('ml_reports/profitability_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Visualization saved to: ml_reports/profitability_analysis.png")
plt.close()

# Create a second figure for detailed metrics
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle('Detailed Performance Metrics', fontsize=14, fontweight='bold')

# Extract more metrics
ml_profit_factors = [r['ml_metrics']['profit_factor'] for r in results]
random_profit_factors = [r['random_metrics']['profit_factor'] for r in results]
ml_max_dd = [r['ml_metrics']['max_drawdown'] for r in results]
random_max_dd = [r['random_metrics']['max_drawdown'] for r in results]
ml_avg_win = [r['ml_metrics']['avg_win'] for r in results]
random_avg_win = [r['random_metrics']['avg_win'] for r in results]
ml_avg_loss = [r['ml_metrics']['avg_loss'] for r in results]
random_avg_loss = [r['random_metrics']['avg_loss'] for r in results]

# Profit Factor
ax1 = axes2[0, 0]
bars1 = ax1.bar(x - width/2, ml_profit_factors, width, label='ML', color='#3498db', alpha=0.8)
bars2 = ax1.bar(x + width/2, random_profit_factors, width, label='Random', color='#e74c3c', alpha=0.8)
ax1.axhline(y=1, color='green', linestyle='--', linewidth=0.8, alpha=0.5, label='Break-even')
ax1.set_ylabel('Profit Factor')
ax1.set_title('Profit Factor (Higher is Better)', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(symbols_tf, fontsize=8)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Max Drawdown
ax2 = axes2[0, 1]
bars1 = ax2.bar(x - width/2, ml_max_dd, width, label='ML', color='#3498db', alpha=0.8)
bars2 = ax2.bar(x + width/2, random_max_dd, width, label='Random', color='#e74c3c', alpha=0.8)
ax2.set_ylabel('Max Drawdown (%)')
ax2.set_title('Maximum Drawdown (Lower is Better)', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(symbols_tf, fontsize=8)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Average Win/Loss
ax3 = axes2[1, 0]
x_pos = np.arange(len(symbols_tf) * 2)
combined_wins = []
combined_losses = []
labels = []
for i, label in enumerate(symbols_tf):
    combined_wins.extend([ml_avg_win[i], random_avg_win[i]])
    combined_losses.extend([ml_avg_loss[i], random_avg_loss[i]])
    labels.extend([f'{label}\nML', f'{label}\nRandom'])

bars1 = ax3.bar(x_pos, combined_wins, color='green', alpha=0.6, label='Avg Win')
bars2 = ax3.bar(x_pos, combined_losses, color='red', alpha=0.6, label='Avg Loss')
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax3.set_ylabel('Amount ($)')
ax3.set_title('Average Win vs Loss per Trade', fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(labels, fontsize=7, rotation=45, ha='right')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# Win Rate vs Return Scatter
ax4 = axes2[1, 1]
ax4.scatter(ml_win_rates, ml_returns, s=100, alpha=0.6, label='ML Strategy', color='#3498db')
ax4.scatter(random_win_rates, random_returns, s=100, alpha=0.6, label='Random Strategy', color='#e74c3c')
ax4.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax4.axvline(x=50, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax4.set_xlabel('Win Rate (%)')
ax4.set_ylabel('Total Return (%)')
ax4.set_title('Win Rate vs Return', fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3)

# Add quadrant labels
ax4.text(60, 0.3, 'High WR\nPositive Return', fontsize=8, alpha=0.5, ha='center')
ax4.text(60, -0.3, 'High WR\nNegative Return', fontsize=8, alpha=0.5, ha='center')
ax4.text(40, 0.3, 'Low WR\nPositive Return', fontsize=8, alpha=0.5, ha='center')
ax4.text(40, -0.3, 'Low WR\nNegative Return', fontsize=8, alpha=0.5, ha='center')

plt.tight_layout()
plt.savefig('ml_reports/detailed_metrics.png', dpi=300, bbox_inches='tight')
print("✅ Detailed metrics saved to: ml_reports/detailed_metrics.png")
plt.close()

print("\n🎉 All visualizations created successfully!")
