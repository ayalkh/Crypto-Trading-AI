import pandas as pd
import numpy as np
import sqlite3
import logging
import os
from datetime import datetime, timedelta
from optimized_ml_system import OptimizedCryptoMLSystem

# Setup logging
logging.basicConfig(level=logging.INFO)

def test_sentiment_integration():
    """
    Test the entire sentiment pipeline using mock data:
    1. Insert mock sentiment data directly into DB
    2. Train models (which should pick up this data)
    3. Verify sentiment features are used
    """
    print("\n🧪 STARTING SENTIMENT PIPELINE INTEGRATION TEST")
    print("=" * 60)
    
    db_path = 'data/ml_crypto_data.db'
    symbol = 'ETH/USDT'
    timeframe = '1h'
    
    # 1. Insert Mock Data
    print("\n1️⃣  Injecting Mock Sentiment Data...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if table exists (should be created by OptimizedCryptoMLSystem if not)
    # We instantiate system first to ensure table creation
    system = OptimizedCryptoMLSystem() 
    
    # Generate 30 days of mock sentiment
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    timestamps = pd.date_range(start=start_date, end=end_date, freq='1h')
    
    mock_data = []
    for ts in timestamps:
        # Simulate correlation: price up = high sentiment
        mock_data.append((
            ts.strftime('%Y-%m-%d %H:%M:%S'),
            symbol,
            np.random.uniform(0.1, 0.9), # Twitter score
            np.random.randint(10, 1000),  # Twitter vol
            np.random.uniform(0.1, 0.9), # Reddit score
            np.random.randint(5, 500),    # Reddit vol
            np.random.uniform(0.1, 0.9)   # Composite
        ))
    
    try:
        cursor.executemany("""
        INSERT INTO sentiment_data (timestamp, symbol, twitter_score, twitter_volume, reddit_score, reddit_volume, composite_score)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, mock_data)
        conn.commit()
        print(f"✅ Injected {len(mock_data)} mock sentiment records")
    except Exception as e:
        print(f"❌ data injection failed (might already exist): {e}")
    finally:
        conn.close()

    # 2. Train Model
    print("\n2️⃣  Training Model with Sentiment...")
    # Verify load_sentiment works
    sent_df = system.load_sentiment(symbol, months_back=1)
    print(f"   Debug: Loaded {len(sent_df)} sentiment rows from DB")
    
    if len(sent_df) == 0:
        print("❌ Failed to load sentiment data. Test aborted.")
        return
        
    # Train
    success = system.train_ensemble(symbol, timeframe)
    
    if success:
        print("✅ Training completed successfully")
        
        # 3. Verify Features
        print("\n3️⃣  Verifying Feature Usage...")
        # Check saved feature file
        import joblib
        feature_file = f"ml_models/{symbol.replace('/','_')}_{timeframe}_price_features.joblib"
        if os.path.exists(feature_file):
            features = joblib.load(feature_file)
            sentiment_features = [f for f in features if 'sentiment' in f or 'social' in f]
            print(f"   Found {len(sentiment_features)} sentiment features selected:")
            print(f"   {sentiment_features[:5]} ...")
            
            if len(sentiment_features) > 0:
                print("✅ SUCCESS: Sentiment features were generated, selected, and used!")
            else:
                print("⚠️ WARNING: No sentiment features selected (Check feature selection logic or correlation)")
        else:
            print("❌ Feature file not found")
    else:
        print("❌ Training failed")

if __name__ == "__main__":
    test_sentiment_integration()
