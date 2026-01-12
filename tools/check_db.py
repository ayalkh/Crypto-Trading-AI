import sqlite3
import pandas as pd

conn = sqlite3.connect('data/ml_crypto_data.db')

print("📊 Predictions for BTC/USDT 4h:")
df = pd.read_sql("""
    SELECT model_type, predicted_direction, direction_probability, 
           confidence_score, predicted_change_pct, timestamp
    FROM ml_predictions 
    WHERE symbol = 'BTC/USDT' AND timeframe = '4h'
    ORDER BY timestamp DESC
    LIMIT 10
""", conn)

print(df.to_string())

print("\n📊 Total predictions by model:")
counts = pd.read_sql("""
    SELECT model_type, COUNT(*) as count
    FROM ml_predictions
    GROUP BY model_type
""", conn)
print(counts)

conn.close()