#!/usr/bin/env python3
"""
Quick test script to verify baseline configuration works correctly.
Tests that the system initializes in baseline mode and can count features.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.insert(0, '/home/ofri/DS/Crypto-Trading-AI')

from optimized_ml_system import OptimizedCryptoMLSystem
import pandas as pd
import numpy as np

print("=" * 60)
print("BASELINE CONFIGURATION TEST")
print("=" * 60)

# Test 1: Initialize in baseline mode
print("\n1. Testing baseline mode initialization...")
system = OptimizedCryptoMLSystem(
    db_path='data/ml_crypto_data.db',
    n_features=None,  # BASELINE: No feature selection
    enable_tuning=False  # BASELINE: No hyperparameter tuning
)

print("\n✅ System initialized successfully in BASELINE mode")
print(f"   n_features: {system.n_features}")
print(f"   enable_tuning: {system.enable_tuning}")

# Test 2: Create sample data and test feature generation
print("\n2. Testing feature generation...")
dates = pd.date_range('2024-01-01', periods=200, freq='1H')
sample_data = pd.DataFrame({
    'open': np.random.randn(200).cumsum() + 100,
    'high': np.random.randn(200).cumsum() + 101,
    'low': np.random.randn(200).cumsum() + 99,
    'close': np.random.randn(200).cumsum() + 100,
    'volume': np.random.randint(1000, 10000, 200)
}, index=dates)

# Generate features
df_features = system.create_features(sample_data)

print(f"\n✅ Features generated successfully")
print(f"   Total features: {len(df_features.columns)}")
print(f"   Samples: {len(df_features)}")

# Expected: ~66 features (excluding OHLCV)
ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
feature_cols = [col for col in df_features.columns if col not in ohlcv_cols]
print(f"   Feature columns (excluding OHLCV): {len(feature_cols)}")

if 60 <= len(feature_cols) <= 70:
    print(f"\n✅ BASELINE FEATURE COUNT VERIFIED: {len(feature_cols)} features")
    print("   Expected: ~66 features")
    print("   Status: PASS ✓")
else:
    print(f"\n⚠️  WARNING: Feature count is {len(feature_cols)}")
    print("   Expected: ~66 features")
    print("   Status: NEEDS REVIEW")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
