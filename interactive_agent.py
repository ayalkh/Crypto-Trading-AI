"""
Interactive Agent - Better Single Pair Analysis
Directly accesses the database for individual pair analysis
"""

from crypto_agent import CryptoTradingAgent
from datetime import datetime
import os

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print header"""
    print("\n" + "="*70)
    print("🤖 CRYPTO TRADING AGENT - INTERACTIVE MODE")
    print("="*70)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")

def analyze_single_pair(agent):
    """Analyze a single pair by getting predictions directly"""
    print("\n📊 Analyze Single Pair")
    print("─"*70)
    
    symbol = input("Enter symbol (e.g., BTC/USDT): ").strip().upper()
    timeframe = input("Enter timeframe (5m/15m/1h/4h/1d): ").strip()
    
    if not symbol or not timeframe:
        print("❌ Invalid input")
        return
    
    print(f"\n🔍 Analyzing {symbol} {timeframe}...\n")
    
    try:
        # Access the database directly to get predictions
        predictions = agent.db.get_ml_predictions(symbol, timeframe)
        
        if predictions is None or len(predictions) == 0:
            print(f"❌ No predictions found for {symbol} {timeframe}")
            print("   Run your model training first to generate predictions")
            return
        
        # Get market overview for context
        overview = agent.get_market_overview()
        
        # Display analysis
        print("="*70)
        print(f"🎯 {symbol} {timeframe} ANALYSIS")
        print("="*70)
        
        # Show model predictions
        print(f"\n🤖 Model Predictions ({len(predictions)} models):")
        for _, pred in predictions.iterrows():
            model = pred['model_type']
            direction = pred['predicted_direction']
            price = pred['predicted_price']
            confidence = pred['confidence_score']
            
            print(f"   {model}: {direction} → ${price:,.2f} (confidence: {confidence:.1%})")
        
        # Consensus
        directions = predictions['predicted_direction'].value_counts()
        most_common = directions.index[0] if len(directions) > 0 else "UNKNOWN"
        agreement = (directions.iloc[0] / len(predictions) * 100) if len(directions) > 0 else 0
        
        print(f"\n📊 Consensus: {most_common}")
        print(f"   Agreement: {agreement:.0f}% of models agree")
        
        # Average confidence
        avg_confidence = predictions['confidence_score'].mean()
        print(f"   Average Confidence: {avg_confidence:.1%}")
        
        # Quality assessment
        print(f"\n⭐ Quality Assessment:")
        
        quality_score = 0
        factors = []
        
        # Factor 1: Model agreement
        if agreement >= 80:
            quality_score += 25
            factors.append("✅ Strong model agreement")
        elif agreement >= 60:
            quality_score += 15
            factors.append("⚠️  Moderate model agreement")
        else:
            factors.append("❌ Weak model agreement")
        
        # Factor 2: Confidence
        if avg_confidence >= 0.65:
            quality_score += 25
            factors.append("✅ High confidence")
        elif avg_confidence >= 0.55:
            quality_score += 15
            factors.append("⚠️  Moderate confidence")
        else:
            factors.append("❌ Low confidence")
        
        # Factor 3: Direction (not NEUTRAL)
        if most_common != "NEUTRAL":
            quality_score += 20
            factors.append("✅ Clear directional signal")
        else:
            factors.append("❌ Neutral/unclear direction")
        
        print(f"\n   Quality Score: {quality_score}/100")
        for factor in factors:
            print(f"   {factor}")
        
        # Trading recommendation
        print(f"\n🎯 Recommendation:")
        
        should_trade = (quality_score >= 60 and 
                       most_common != "NEUTRAL" and 
                       avg_confidence >= 0.55)
        
        if should_trade:
            print(f"   ✅ TRADE THIS - {most_common}")
            print(f"   Quality threshold met ({quality_score}/100 ≥ 60)")
            print(f"   Suggested position: 2-4% of portfolio")
        else:
            print(f"   ⚠️  DO NOT TRADE - HOLD")
            reasons = []
            if quality_score < 60:
                reasons.append(f"Quality too low ({quality_score}/100)")
            if most_common == "NEUTRAL":
                reasons.append("No clear direction")
            if avg_confidence < 0.55:
                reasons.append(f"Confidence too low ({avg_confidence:.1%})")
            print(f"   Reasons: {', '.join(reasons)}")
        
        # Market context
        print(f"\n🌍 Market Context:")
        print(f"   Regime: {overview.get('market_regime', 'Unknown')}")
        print(f"   Risk Level: {overview.get('risk_level', 'Unknown')}")
        
        # Prediction age
        latest_time = predictions['timestamp'].max()
        age_hours = (datetime.now() - latest_time).total_seconds() / 3600
        print(f"\n⏰ Prediction Age: {age_hours:.1f} hours old")
        if age_hours > 4:
            print(f"   ⚠️  Predictions may be stale - consider retraining")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def market_overview(agent):
    """Get market overview"""
    print("\n🌍 Market Overview")
    print("─"*70)
    
    try:
        overview = agent.get_market_overview()
        
        print(f"\n📊 Market Regime: {overview['market_regime']}")
        print(f"   Confidence: {overview.get('regime_confidence', 0):.0%}")
        print(f"⚠️  Risk Level: {overview['risk_level']}")
        print(f"🎯 Symbols Analyzed: {len(overview.get('symbol_analysis', []))}")
        
        top_opps = overview.get('top_opportunities', [])
        
        if top_opps:
            print(f"\n✅ Top {len(top_opps)} Opportunities:")
            for i, opp in enumerate(top_opps, 1):
                print(f"\n{i}. {opp['symbol']} {opp['timeframe']}")
                print(f"   Recommendation: {opp['recommendation']}")
                print(f"   Quality: {opp.get('quality', 0)}/100")
                print(f"   Confidence: {opp.get('confidence', 0):.1%}")
        else:
            print("\n⚠️  No high-quality opportunities right now")
            print(f"\n   Current market is {overview['market_regime']}")
            print(f"   Wait for clearer signals")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def scan_all_symbols(agent):
    """Scan all symbols"""
    print("\n🔍 Scanning All Symbols...")
    print("─"*70)
    
    try:
        overview = agent.get_market_overview()
        
        print(f"\n📊 Market: {overview['market_regime']} | Risk: {overview['risk_level']}")
        print(f"\n📈 All Analysis:\n")
        
        for sym_data in overview.get('symbol_analysis', []):
            symbol = sym_data['symbol']
            print(f"\n{symbol}:")
            
            for tf_data in sym_data.get('timeframes', []):
                tf = tf_data['timeframe']
                rec = tf_data.get('recommendation', 'N/A')
                qual = tf_data.get('quality_score', 0)
                conf = tf_data.get('confidence', 0)
                
                status = "✅" if tf_data.get('should_trade') else "⚠️"
                print(f"  {status} {tf:>3s}: {rec:>4s} Q:{qual:>2}/100 C:{conf:>3.0%}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def performance_report(agent):
    """Performance report"""
    print("\n📈 Performance Report")
    print("─"*70)
    
    try:
        # Call without parameters - the method handles defaults internally
        report = agent.get_performance_report()
        
        print(f"\n📊 Performance Summary:")
        print(f"   Total Recommendations: {report.get('total_recommendations', 0)}")
        
        if report.get('total_recommendations', 0) > 0:
            print(f"   Win Rate: {report.get('win_rate', 0):.1%}")
            print(f"   Best Symbol: {report.get('best_symbol', 'N/A')}")
            print(f"   Best Timeframe: {report.get('best_timeframe', 'N/A')}")
        else:
            print("\n   ℹ️  No historical recommendations yet")
            print("   The agent will track performance as you use it")
            print("   Make recommendations and check back later!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def main_menu():
    """Main menu"""
    
    print("🚀 Initializing agent...")
    agent = CryptoTradingAgent()
    print("✅ Agent ready!\n")
    
    while True:
        print_header()
        
        print("📋 Main Menu:")
        print()
        print("1. 📊 Analyze Single Pair")
        print("2. 🌍 Market Overview")
        print("3. 🔍 Scan All Symbols")
        print("4. 📈 Performance Report")
        print("5. 🚪 Exit")
        print()
        
        choice = input("Select (1-5): ").strip()
        
        if choice == "1":
            analyze_single_pair(agent)
        elif choice == "2":
            market_overview(agent)
        elif choice == "3":
            scan_all_symbols(agent)
        elif choice == "4":
            performance_report(agent)
        elif choice == "5":
            print("\n👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice")
        
        input("\nPress Enter...")
        clear_screen()

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n👋 Stopped")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()