import os
import logging
import pandas as pd
from datetime import datetime, timedelta
import tweepy
import praw
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time

class SentimentCollector:
    """Collect and analyze crypto sentiment from social media (Twitter & Reddit)"""
    
    def __init__(self):
        """Initialize connections to social media APIs"""
        self.vader = SentimentIntensityAnalyzer()
        self.setup_twitter()
        self.setup_reddit()
        logging.info("🧠 Sentiment Collector initialized")

    def setup_twitter(self):
        """Setup Twitter API connection"""
        bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        if bearer_token:
            try:
                self.twitter_client = tweepy.Client(bearer_token=bearer_token)
                self.twitter_enabled = True
                logging.info("✅ Twitter API connected")
            except Exception as e:
                logging.warning(f"⚠️ Twitter API connection failed: {e}")
                self.twitter_enabled = False
        else:
            logging.warning("⚠️ Twitter Bearer Token not found in env")
            self.twitter_enabled = False

    def setup_reddit(self):
        """Setup Reddit API connection"""
        client_id = os.getenv('REDDIT_CLIENT_ID')
        client_secret = os.getenv('REDDIT_SECRET')
        if client_id and client_secret:
            try:
                self.reddit = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent='crypto_sentiment_bot_v1'
                )
                self.reddit_enabled = True
                logging.info("✅ Reddit API connected")
            except Exception as e:
                logging.warning(f"⚠️ Reddit API connection failed: {e}")
                self.reddit_enabled = False
        else:
            logging.warning("⚠️ Reddit credentials not found in env")
            self.reddit_enabled = False

    def get_sentiment(self, symbol: str, hours_back: int = 24) -> dict:
        """
        Get combined sentiment from all available sources
        
        Returns:
            dict with sentiment scores and volume metrics
        """
        sentiment_data = {
            'timestamp': datetime.utcnow(),
            'symbol': symbol,
            'twitter_score': 0,
            'twitter_volume': 0,
            'reddit_score': 0,
            'reddit_volume': 0,
            'composite_score': 0,
            'sources_available': 0
        }
        
        # Strip symbol to base coin (BTC/USDT -> BTC)
        coin = symbol.split('/')[0]
        
        if self.twitter_enabled:
            tw_data = self._fetch_twitter_sentiment(coin, hours_back)
            sentiment_data.update(tw_data)
        
        if self.reddit_enabled:
            rd_data = self._fetch_reddit_sentiment(coin, hours_back)
            sentiment_data.update(rd_data)
            
        # Calculate composite score (weighted average)
        total_vol = sentiment_data['twitter_volume'] + sentiment_data['reddit_volume']
        if total_vol > 0:
            tw_weight = sentiment_data['twitter_volume'] / total_vol
            rd_weight = sentiment_data['reddit_volume'] / total_vol
            
            sentiment_data['composite_score'] = (
                sentiment_data['twitter_score'] * tw_weight +
                sentiment_data['reddit_score'] * rd_weight
            )
            sentiment_data['sources_available'] = int(self.twitter_enabled) + int(self.reddit_enabled)
            
        return sentiment_data

    def _fetch_twitter_sentiment(self, coin: str, hours_back: int) -> dict:
        """Fetch and analyze recent tweets"""
        try:
            query = f"${coin} -is:retweet lang:en"
            start_time = datetime.utcnow() - timedelta(hours=hours_back)
            
            # Note: Free tier has strict limits, this is a basic implementation
            # In production, you'd likely use a paid tier or simpler scraping
            tweets = self.twitter_client.search_recent_tweets(
                query=query,
                start_time=start_time,
                max_results=20, # Low limit for testing/free tier
                tweet_fields=['created_at', 'public_metrics']
            )
            
            if not tweets.data:
                return {'twitter_score': 0, 'twitter_volume': 0}
                
            scores = []
            metrics = {'likes': 0, 'retweets': 0}
            
            for tweet in tweets.data:
                # VADER Analysis
                vs = self.vader.polarity_scores(tweet.text)
                scores.append(vs['compound'])
                
                # Metrics
                if tweet.public_metrics:
                    metrics['likes'] += tweet.public_metrics.get('like_count', 0)
                    metrics['retweets'] += tweet.public_metrics.get('retweet_count', 0)
            
            avg_score = sum(scores) / len(scores) if scores else 0
            
            return {
                'twitter_score': avg_score,
                'twitter_volume': len(scores),
                'twitter_likes': metrics['likes']
            }
            
        except Exception as e:
            logging.error(f"Error fetching tweets for {coin}: {e}")
            return {'twitter_score': 0, 'twitter_volume': 0}

    def collect_historical(self, symbol: str, days_back: int = 7) -> list:
        """
        Backfill historical sentiment data
        
        Args:
            symbol: Crypto symbol (e.g. BTC/USDT)
            days_back: Number of days to fetch (max 7 for Twitter free tier)
            
        Returns:
            list of dicts containing historical sentiment
        """
        results = []
        coin = symbol.split('/')[0]
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days_back)
        
        logging.info(f"🕰️ Starting historical collection for {coin} ({days_back} days)")
        
        # 1. Historical Twitter (Max 7 days on free tier)
        if self.twitter_enabled:
            try:
                query = f"${coin} -is:retweet lang:en"
                # Iterate day by day to manage pagination/limits
                current_start = start_time
                while current_start < end_time:
                    current_end = min(current_start + timedelta(days=1), end_time)
                    
                    try:
                        tweets = self.twitter_client.search_recent_tweets(
                            query=query,
                            start_time=current_start,
                            end_time=current_end,
                            max_results=50, 
                            tweet_fields=['created_at', 'public_metrics']
                        )
                        
                        if tweets.data:
                            # Aggregate by hour for this day
                            df = pd.DataFrame([{
                                'created_at': t.created_at,
                                'text': t.text,
                                'likes': t.public_metrics['like_count']
                            } for t in tweets.data])
                            
                            df['timestamp'] = df['created_at'].dt.floor('h')
                            
                            for ts, group in df.groupby('timestamp'):
                                scores = [self.vader.polarity_scores(t)['compound'] for t in group['text']]
                                avg_score = sum(scores) / len(scores)
                                
                                results.append({
                                    'timestamp': ts.replace(tzinfo=None), # Ensure naive datetime for DB
                                    'symbol': symbol,
                                    'twitter_score': avg_score,
                                    'twitter_volume': len(scores),
                                    'source': 'twitter'
                                })
                                
                        time.sleep(1.1) # Rate limit safety
                        
                    except Exception as e:
                        logging.warning(f"⚠️ Twitter historical fetch error for {current_start}: {e}")
                        
                    current_start += timedelta(days=1)
                    
            except Exception as e:
                logging.error(f"❌ Critical Twitter historical error: {e}")

        # 2. Historical Reddit (Using search, less precise timestamps but goes back further)
        if self.reddit_enabled:
            try:
                subreddits = ['cryptocurrency', 'bitcoin', 'ethereum']
                if coin.lower() not in ['btc', 'eth']:
                     subreddits.append(coin.lower())
                
                all_posts = []
                for sub_name in subreddits:
                    try:
                        subreddit = self.reddit.subreddit(sub_name)
                        # Searching allows 'relevance', 'hot', 'top', 'new'
                        # 'new' is best for historical timeline reconstruction
                        for post in subreddit.search(coin, sort='new', time_filter='month', limit=200):
                            post_time = datetime.fromtimestamp(post.created_utc)
                            if post_time < start_time:
                                continue
                                
                            all_posts.append({
                                'timestamp': post_time,
                                'title': post.title,
                                'text': post.selftext,
                                'score': post.score
                            })
                    except Exception as e:
                        logging.warning(f"⚠️ Reddit fetch error for r/{sub_name}: {e}")
                
                # Aggregate Reddit posts
                if all_posts:
                    df_reddit = pd.DataFrame(all_posts)
                    df_reddit['timestamp'] = pd.to_datetime(df_reddit['timestamp']).dt.floor('h')
                    
                    for ts, group in df_reddit.groupby('timestamp'):
                        scores = []
                        total_karma = 0
                        weighted_score_sum = 0
                        
                        for _, row in group.iterrows():
                            text = f"{row['title']} {row['text'][:500]}"
                            vs = self.vader.polarity_scores(text)['compound']
                            weight = max(1, row['score'])
                            
                            weighted_score_sum += vs * weight
                            total_karma += weight
                            scores.append(vs)
                            
                        avg_score = weighted_score_sum / total_karma if total_karma > 0 else 0
                        
                        # Check if we already have an entry for this hour (from Twitter loop)
                        # If so, update it. If not, create new.
                        existing_idx = next((i for i, r in enumerate(results) if r['timestamp'] == ts.replace(tzinfo=None)), -1)
                        
                        if existing_idx != -1:
                            results[existing_idx]['reddit_score'] = avg_score
                            results[existing_idx]['reddit_volume'] = len(scores)
                        else:
                            results.append({
                                'timestamp': ts.replace(tzinfo=None),
                                'symbol': symbol,
                                'twitter_score': 0, # Default if no twitter data
                                'twitter_volume': 0,
                                'reddit_score': avg_score,
                                'reddit_volume': len(scores),
                                'source': 'reddit'
                            })
                            
            except Exception as e:
                logging.error(f"❌ Critical Reddit historical error: {e}")
        
        # Final cleanup and composite score calculation
        final_results = []
        for res in results:
            tv = res.get('twitter_volume', 0)
            rv = res.get('reddit_volume', 0)
            ts = res.get('twitter_score', 0)
            rs = res.get('reddit_score', 0)
            
            total_vol = tv + rv
            if total_vol > 0:
                comp_score = (ts * tv + rs * rv) / total_vol
            else:
                comp_score = 0
                
            res['composite_score'] = comp_score
            final_results.append(res)
            
        logging.info(f"✅ Collected {len(final_results)} historical sentiment points for {coin}")
        return final_results

if __name__ == "__main__":
    # Test run
    logging.basicConfig(level=logging.INFO)
    collector = SentimentCollector()
    print(collector.get_sentiment("BTC/USDT", hours_back=1))
