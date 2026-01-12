import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
import time
import os
from dotenv import load_dotenv
from crypto_ai.sentiment import SentimentCollector
from crypto_ai.database.db import DatabaseManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def collect_and_store(symbols=['BTC/USDT', 'ETH/USDT', 'SOL/USDT'], backfill=False, days_back=7):
    """Collect sentiment for symbols and store in DB"""
    load_dotenv()
    
    collector = SentimentCollector()
    db = DatabaseManager()
    
    if not collector.twitter_enabled and not collector.reddit_enabled:
        logging.warning("⚠️ No social media APIs enabled. Check .env file.")
        return

    mode = "historically" if backfill else "live"
    logging.info(f"🚀 Starting {mode} sentiment collection for {len(symbols)} symbols")
    
    for symbol in symbols:
        try:
            logging.info(f"🔍 Collecting {mode} sentiment for {symbol}...")
            
            if backfill:
                # Historical Collection
                history_data = collector.collect_historical(symbol, days_back=days_back)
                if history_data:
                    count = 0
                    for record in history_data:
                        # Add a quick check to avoid duplicates if needed, or rely on DB
                        db.save_sentiment(record)
                        count += 1
                    logging.info(f"✅ Backfilled {count} records for {symbol}")
                else:
                    logging.warning(f"⚠️ No historical data found for {symbol}")
            else:
                # Live Collection
                data = collector.get_sentiment(symbol, hours_back=6)
                if data['sources_available'] > 0:
                    db.save_sentiment(data)
                    logging.info(f"✅ Saved sentiment for {symbol}: Score={data.get('composite_score', 0):.2f}, Vol={data.get('twitter_volume',0)+data.get('reddit_volume',0)}")
                else:
                    logging.warning(f"⚠️ No data sources available for {symbol}")
                
            # Rate limiting
            time.sleep(2)
            
        except Exception as e:
            logging.error(f"❌ Error processing {symbol}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--backfill', action='store_true', help='Backfill historical data')
    parser.add_argument('--days', type=int, default=7, help='Days to backfill (max 7 for Twitter)')
    args = parser.parse_args()
    
    collect_and_store(backfill=args.backfill, days_back=args.days)
