#!/usr/bin/env python3
import sqlite3
import os

db_path = 'data/ml_crypto_data.db'
print(f'DB: {db_path}')
print(f'Size: {os.path.getsize(db_path) / (1024*1024):.1f} MB')

conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cursor.fetchall()]
print(f'Tables ({len(tables)}):')

for t in tables[:10]:
    try:
        cursor.execute(f'SELECT COUNT(*) FROM "{t}"')
        print(f'  {t}: {cursor.fetchone()[0]:,} rows')
    except Exception as e:
        print(f'  {t}: ERROR {e}')

# Check latest timestamp
try:
    cursor.execute('SELECT MAX(timestamp) FROM ohlcv_data')
    latest = cursor.fetchone()[0]
    print(f'\nLatest data: {latest}')
except:
    pass

conn.close()
