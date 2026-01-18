"""
24-Hour Automated Agent Runner
Continuously monitors markets and logs opportunities
"""

import time
import schedule
from datetime import datetime
from crypto_agent import CryptoTradingAgent
import logging
import os

# Setup logging
log_dir = 'logs/automated'
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{log_dir}/automated_{datetime.now():%Y%m%d}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AutomatedTradingAgent:
    """Automated agent that runs every hour"""
    
    def __init__(self):
        """Initialize agent"""
        self.agent = CryptoTradingAgent(log_to_file=True)
        self.symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "DOT/USDT"]
        self.priority_timeframes = ["4h", "1h"]  # Focus on these
        self.opportunity_log = f'{log_dir}/opportunities.txt'
        self.run_count = 0
        
    def analyze_all_markets(self):
        """Run complete market analysis"""
        
        self.run_count += 1
        
        logger.info("\n" + "="*70)
        logger.info(f"🤖 AUTOMATED RUN #{self.run_count} - {datetime.now():%Y-%m-%d %H:%M:%S}")
        logger.info("="*70)
        
        try:
            # Get market overview
            overview = self.agent.get_market_overview()
            
            logger.info(f"\n🌍 Market Status:")
            logger.info(f"   Regime: {overview['market_regime']}")
            logger.info(f"   Risk Level: {overview['risk_level']}")
            
            # Get top opportunities
            opportunities = overview.get('top_opportunities', [])
            
            if opportunities:
                logger.info(f"\n🎯 Found {len(opportunities)} opportunities!")
                
                # Log each opportunity
                for i, opp in enumerate(opportunities[:5], 1):
                    logger.info(f"\n   {i}. {opp['symbol']} {opp['timeframe']}")
                    logger.info(f"      Recommendation: {opp['recommendation']}")
                    logger.info(f"      Quality: {opp['quality_score']}/100")
                    logger.info(f"      Confidence: {opp['confidence']:.1%}")
                    
                    # Get detailed analysis for top 3
                    if i <= 3:
                        try:
                            analysis = self.agent.analyze_trading_opportunity(
                                opp['symbol'],
                                opp['timeframe'],
                                save_recommendation=True
                            )
                            
                            logger.info(f"      Should Trade: {analysis['should_trade']}")
                            if analysis['should_trade']:
                                logger.info(f"      Position Size: {analysis['position_sizing'][0]:.1f}%-{analysis['position_sizing'][1]:.1f}%")
                                logger.info(f"      Stop Loss: ${analysis['stop_loss']:,.2f}" if analysis['stop_loss'] else "")
                        
                        except Exception as e:
                            logger.error(f"      Error analyzing: {e}")
                
                # Save to opportunities log
                self._log_opportunities(opportunities[:5])
            
            else:
                logger.info("\n⚠️  No quality opportunities at this time")
                logger.info("   Market conditions don't meet trading criteria")
            
            logger.info("\n" + "="*70)
            logger.info(f"✅ Run #{self.run_count} complete")
            logger.info("="*70 + "\n")
        
        except Exception as e:
            logger.error(f"\n❌ Error in automated run: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _log_opportunities(self, opportunities):
        """Log opportunities to separate file for easy review"""
        
        try:
            with open(self.opportunity_log, 'a') as f:
                f.write(f"\n{'='*70}\n")
                f.write(f"RUN #{self.run_count} - {datetime.now():%Y-%m-%d %H:%M:%S}\n")
                f.write(f"{'='*70}\n\n")
                
                for opp in opportunities:
                    f.write(f"{opp['symbol']:10s} {opp['timeframe']:4s} | ")
                    f.write(f"{opp['recommendation']:12s} | ")
                    f.write(f"Q:{opp['quality_score']:3d}/100 | ")
                    f.write(f"C:{opp['confidence']:5.1%}\n")
                
                f.write("\n")
        
        except Exception as e:
            logger.error(f"Error logging opportunities: {e}")
    
    def test_single_run(self):
        """Test with a single run"""
        logger.info("🧪 Running test analysis...")
        self.analyze_all_markets()


def main():
    """Main automation function"""
    
    print("\n" + "="*70)
    print("🚀 24-HOUR AUTOMATED TRADING AGENT")
    print("="*70)
    print(f"\n📅 Start Time: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"📊 Monitoring: BTC, ETH, BNB, ADA, DOT")
    print(f"⏰ Schedule: Every 1 hour")
    print(f"📝 Logs: logs/automated/")
    print("\n" + "="*70 + "\n")
    
    agent = AutomatedTradingAgent()
    
    # Run immediately
    print("▶️  Running initial analysis...\n")
    agent.analyze_all_markets()
    
    # Schedule hourly runs
    schedule.every(1).hours.do(agent.analyze_all_markets)
    
    print(f"\n⏰ Next run in 1 hour at {datetime.now():%H:%M}")
    print("⏸️  Press Ctrl+C to stop\n")
    
    # Keep running
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print(f"🛑 Agent stopped after {agent.run_count} runs")
        print(f"⏰ End Time: {datetime.now():%Y-%m-%d %H:%M:%S}")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()