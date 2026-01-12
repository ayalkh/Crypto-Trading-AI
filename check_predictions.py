import sqlite3
import pandas as pd

conn = sqlite3.connect('data/ml_crypto_data.db')

# Check what columns exist
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(ml_predictions)")
columns = cursor.fetchall()

print("📋 Columns in ml_predictions table:")
for col in columns:
    print(f"   {col[1]:30} {col[2]}")

print("\n📊 Sample predictions:")
df = pd.read_sql("""
    SELECT symbol, timeframe, model_type, predicted_direction, 
           direction_probability, confidence_score, predicted_change_pct
    FROM ml_predictions 
    WHERE symbol = 'BTC/USDT' AND timeframe = '4h'
    LIMIT 5
""", conn)

print(df)

conn.close()