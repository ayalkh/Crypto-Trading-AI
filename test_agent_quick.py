"""
Quick test to verify agent sees actual signals
"""

from crypto_agent import CryptoTradingAgent

def main():
    print("\n" + "="*70)
    print("🧪 QUICK AGENT TEST")
    print("="*70)
    
    agent = CryptoTradingAgent()
    
    # Test BTC on 4h
    print("\n📊 Testing BTC/USDT 4h...")
    analysis = agent.analyze_trading_opportunity('BTC/USDT', '4h', save_recommendation=False)
    
    print(f"\nResults:")
    print(f"  Recommendation: {analysis['recommendation']}")
    print(f"  Quality Score: {analysis['quality_score']}/100")
    print(f"  Confidence: {analysis['confidence']:.1%}")
    print(f"  Should Trade: {analysis['should_trade']}")
    
    if analysis['should_trade']:
        print(f"\n✅ TRADE SIGNAL DETECTED!")
        print(f"  Position Size: {analysis['position_sizing'][0]:.1f}%-{analysis['position_sizing'][1]:.1f}%")
    else:
        print(f"\n⚠️  No trade - {', '.join(analysis['risk_factors'][:2])}")
    
    # Get market overview
    print("\n" + "="*70)
    print("🌍 Market Overview:")
    print("="*70)
    
    overview = agent.get_market_overview()
    print(f"\nRegime: {overview['market_regime']}")
    print(f"Risk: {overview['risk_level']}")
    print(f"\nTop Opportunities: {len(overview['top_opportunities'])}")
    
    for i, opp in enumerate(overview['top_opportunities'][:3], 1):
        print(f"  {i}. {opp['symbol']} {opp['timeframe']} - {opp['recommendation']} (Q:{opp['quality_score']})")
    
    print("\n" + "="*70)
    print("✅ Test complete!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()