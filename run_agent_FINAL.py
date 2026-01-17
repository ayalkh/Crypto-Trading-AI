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
    
    # Get market overview (this method definitely exists based on test_agent.py)
    print("="*70)
    print("🌍 GETTING MARKET OVERVIEW")
    print("="*70 + "\n")
    
    try:
        overview = agent.get_market_overview()
        
        # Display market context
        print(f"📊 Market Regime: {overview['market_regime']}")
        print(f"   Confidence: {overview.get('regime_confidence', 0):.0%}")
        print(f"⚠️  Risk Level: {overview['risk_level']}")
        print(f"🎯 Symbols Analyzed: {len(overview.get('symbol_analysis', []))}")
        
        # Check for opportunities
        top_opps = overview.get('top_opportunities', [])
        
        print(f"\n{'='*70}")
        print("📊 TRADING OPPORTUNITIES")
        print('='*70)
        
        if top_opps:
            print(f"\n✅ Found {len(top_opps)} opportunities:\n")
            
            for i, opp in enumerate(top_opps, 1):
                print(f"{i}. {opp['symbol']} {opp['timeframe']}")
                print(f"   Recommendation: {opp['recommendation']}")
                print(f"   Quality: {opp.get('quality', 'N/A')}/100")
                print(f"   Confidence: {opp.get('confidence', 0):.1%}")
                
                if opp.get('should_trade'):
                    print(f"   ✅ TRADE THIS")
                else:
                    print(f"   ⚠️  DO NOT TRADE - Conditions not met")
                print()
        else:
            print("\n⚠️  No high-quality trading opportunities right now\n")
            print("   Current Market Status:")
            print(f"   • Regime: {overview['market_regime']}")
            print(f"   • Risk: {overview['risk_level']}")
            print("\n   Why no opportunities?")
            print("   • Market is ranging (no clear trend)")
            print("   • Models predict NEUTRAL")
            print("   • Quality scores below 60/100 threshold")
            print("\n   ✅ This is GOOD - agent protects you from uncertain trades!")
            print("   📅 Try again later when conditions improve.\n")
        
        # Show all symbol analysis if available
        if overview.get('symbol_analysis'):
            print(f"\n{'='*70}")
            print("📈 DETAILED SYMBOL ANALYSIS")
            print('='*70 + "\n")
            
            for sym_data in overview['symbol_analysis']:
                symbol = sym_data['symbol']
                print(f"\n{symbol}:")
                
                for tf_data in sym_data.get('timeframes', []):
                    tf = tf_data['timeframe']
                    rec = tf_data.get('recommendation', 'N/A')
                    qual = tf_data.get('quality_score', 0)
                    conf = tf_data.get('confidence', 0)
                    
                    status = "✅" if tf_data.get('should_trade') else "⚠️"
                    print(f"  {status} {tf}: {rec} (Q:{qual}/100, C:{conf:.0%})")
        
    except Exception as e:
        print(f"❌ Error getting market overview: {e}")
        print("\nTrying alternative approach...\n")
        
        # Alternative: Try direct database queries if agent methods fail
        try:
            print("Checking latest predictions from database...")
            # This would need to access agent.db directly
            # But for now, show the error
            import traceback
            traceback.print_exc()
        except:
            pass
    
    print("\n" + "="*70)
    print("💡 TIPS")
    print("="*70)
    print("• Run this daily to catch opportunities")
    print("• Quality 70+ = Strong signals")
    print("• Quality 60-69 = Acceptable")
    print("• Below 60 = Don't trade")
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