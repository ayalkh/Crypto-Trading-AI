"""Test new feature engineering additions"""
from crypto_ai.features import FeatureEngineer
import pandas as pd
import sqlite3

print("Testing new feature engineering...")
conn = sqlite3.connect('data/ml_crypto_data.db')
df = pd.read_sql('SELECT * FROM price_data WHERE symbol="BTC/USDT" AND timeframe="1h" LIMIT 500', conn)
conn.close()

fe = FeatureEngineer()
df_fe = fe.create_features(df)

new_features = [c for c in df_fe.columns if any(x in c for x in ['ob_', 'funding_', 'oi_', 'arb_', 'corr_', 'rel_'])]
print(f'✅ Total columns: {len(df_fe.columns)}')
print(f'✅ New features ({len(new_features)}):')
for f in new_features:
    non_null = df_fe[f].notna().sum()
    print(f'   - {f}: {non_null}/{len(df_fe)} non-null values')

print(f'\n📊 Sample of new features (last 3 rows):')
print(df_fe[new_features].tail(3).to_string())
