"""
Scheduled Agent Runner
Runs the agent on a schedule and optionally sends alerts
"""

import schedule
import time
from datetime import datetime
from crypto_agent import CryptoTradingAgent
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scheduled_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ScheduledAgent:
    """Run agent analysis on schedule"""
    
    def __init__(self):
        self.agent = CryptoTradingAgent()
        self.symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "DOT/USDT"]
        self.timeframes = ["4h", "1h"]
        
    def analyze_markets(self):
        """Run full market analysis"""
        logger.info("="*70)
        logger.info("🤖 Starting scheduled analysis...")
        logger.info("="*70)
        
        opportunities = []
        
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                try:
                    rec = self.agent.analyze_opportunity(symbol, timeframe)
                    
                    if rec['should_trade']:
                        logger.info(f"✅ OPPORTUNITY: {symbol} {timeframe} - {rec['recommendation']}")
                        logger.info(f"   Quality: {rec['quality_score']}/100 | Confidence: {rec['confidence']:.1%}")
                        
                        opportunities.append({
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'recommendation': rec['recommendation'],
                            'quality': rec['quality_score'],
                            'confidence': rec['confidence'],
                            'price': rec['current_price']
                        })
                        
                except Exception as e:
                    logger.error(f"Error analyzing {symbol} {timeframe}: {e}")
        
        if opportunities:
            logger.info(f"\n🎯 Found {len(opportunities)} opportunities!")
            self.send_alert(opportunities)
        else:
            logger.info("\n⚠️  No opportunities found this run")
        
        logger.info("="*70 + "\n")
    
    def send_alert(self, opportunities):
        """
        Send alert about opportunities
        Placeholder - implement with your preferred notification method
        """
        # TODO: Implement actual notification
        # Options:
        # - Email (using smtplib)
        # - Telegram bot
        # - Discord webhook
        # - SMS (Twilio)
        # - Desktop notification
        
        logger.info("\n📢 ALERT: Trading opportunities detected!")
        for opp in opportunities:
            logger.info(f"   {opp['symbol']} {opp['timeframe']}: {opp['recommendation']}")
        
        # Example: Write to a file that you can monitor
        with open('logs/alerts.txt', 'a') as f:
            f.write(f"\n{datetime.now()} - {len(opportunities)} opportunities\n")
            for opp in opportunities:
                f.write(f"  {opp['symbol']} {opp['timeframe']}: {opp['recommendation']} (Q: {opp['quality']})\n")

def main():
    """
    Main scheduling function
    """
    logger.info("🚀 Scheduled Agent Starting...")
    
    scheduler = ScheduledAgent()
    
    # Schedule runs
    # Run every 4 hours (aligned with 4h timeframe)
    schedule.every(4).hours.do(scheduler.analyze_markets)
    
    # Or run every hour
    # schedule.every(1).hours.do(scheduler.analyze_markets)
    
    # Or run at specific times
    # schedule.every().day.at("09:00").do(scheduler.analyze_markets)
    # schedule.every().day.at("15:00").do(scheduler.analyze_markets)
    # schedule.every().day.at("21:00").do(scheduler.analyze_markets)
    
    logger.info("📅 Schedule set:")
    logger.info("   - Every 4 hours")
    logger.info("\n⏰ Waiting for next run... (Press Ctrl+C to stop)\n")
    
    # Run once immediately
    scheduler.analyze_markets()
    
    # Then run on schedule
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n👋 Scheduled agent stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")