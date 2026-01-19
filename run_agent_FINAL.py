"""
Crypto Trading Agent Runner
Works with your existing agent implementation
"""

from crypto_agent import CryptoTradingAgent
from datetime import datetime
import sys

def main():
    """Run market analysis using the agent's actual API"""
    
    print("\n" + "="*70)
    print("🤖 CRYPTO TRADING AGENT - MARKET ANALYSIS")
    print("="*70)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    # Initialize agent
    print("🚀 Initializing agent...")
    agent = CryptoTradingAgent()
    print("✅ Agent ready!\n")
    
    # Get market overview
    print("="*70)
    print("🌍 GETTING MARKET OVERVIEW")
    print("="*70 + "\n")
    
    try:
        overview = agent.get_market_overview()
        
        # Display market context
        market_context = overview.get('market_regime', 'Unknown')
        risk_level = overview.get('risk_level', 'UNKNOWN')
        
        print(f"📊 Market Regime: {market_context}")
        print(f"⚠️  Risk Level: {risk_level}")
        print(f"🎯 Symbols Analyzed: {overview.get('symbols_analyzed', 0)}")
        
        # Get opportunities
        top_opps = overview.get('top_opportunities', [])
        
        print(f"\n{'='*70}")
        print("📊 TRADING OPPORTUNITIES")
        print('='*70)
        
        if top_opps:
            print(f"\n✅ Found {len(top_opps)} opportunities:\n")
            
            for i, opp in enumerate(top_opps, 1):
                symbol = opp['symbol']
                timeframe = opp['timeframe']
                recommendation = opp['recommendation']
                quality_score = opp.get('quality_score', 0)  # FIXED: was 'quality'
                confidence = opp.get('confidence', 0)
                
                # Determine if tradeable based on quality threshold
                should_trade = quality_score >= 50 and confidence >= 0.50
                
                # Emoji for recommendation
                emoji = {
                    'STRONG_BUY': '🟢🟢',
                    'BUY': '🟢',
                    'HOLD': '🟡',
                    'SELL': '🔴',
                    'STRONG_SELL': '🔴🔴'
                }.get(recommendation, '⚪')
                
                print(f"{i}. {emoji} {symbol} {timeframe}")
                print(f"   Recommendation: {recommendation}")
                print(f"   Quality: {quality_score}/100")
                print(f"   Confidence: {confidence:.1%}")
                
                if should_trade and recommendation not in ['HOLD']:
                    print(f"   ✅ TRADEABLE SIGNAL")
                else:
                    reasons = []
                    if quality_score < 50:
                        reasons.append(f"Quality too low ({quality_score}/100, need 50+)")
                    if confidence < 0.50:
                        reasons.append(f"Confidence too low ({confidence:.0%})")
                    if recommendation == 'HOLD':
                        reasons.append("Recommendation is HOLD")
                    
                    print(f"   ⚠️  SKIP: {', '.join(reasons) if reasons else 'Unknown reason'}")
                print()
            
            # Show detailed analysis for top tradeable opportunity
            tradeable = [o for o in top_opps 
                        if o.get('quality_score', 0) >= 50 
                        and o.get('confidence', 0) >= 0.50
                        and o['recommendation'] not in ['HOLD']]
            
            if tradeable:
                print(f"\n{'='*70}")
                print("🎯 DETAILED ANALYSIS - TOP OPPORTUNITY")
                print('='*70 + "\n")
                
                top = tradeable[0]
                
                # Get full analysis for this opportunity
                try:
                    full_analysis = agent.analyze_trading_opportunity(
                        top['symbol'], 
                        top['timeframe'],
                        save_recommendation=False
                    )
                    
                    # Display formatted recommendation
                    print(agent.format_recommendation(full_analysis))
                    
                except Exception as e:
                    print(f"⚠️  Could not get detailed analysis: {e}")
        
        else:
            print("\n⚠️  No high-quality trading opportunities right now\n")
            print("   Current Market Status:")
            print(f"   • Regime: {market_context}")
            print(f"   • Risk: {risk_level}")
            print("\n   Why no opportunities?")
            print("   • Market is ranging (no clear trend)")
            print("   • Models predict NEUTRAL")
            print("   • Quality scores below threshold")
            print("\n   ✅ This is GOOD - agent protects you from uncertain trades!")
            print("   📅 Try again later when conditions improve.")
        
    except Exception as e:
        print(f"❌ Error getting market overview: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("💡 TIPS")
    print("="*70)
    print("• Run this daily to catch opportunities")
    print("• Quality 70+ = Strong signals")
    print("• Quality 60-69 = Acceptable")  
    print("• Quality 50-59 = Marginal (use small positions)")
    print("• Below 50 = Don't trade")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)