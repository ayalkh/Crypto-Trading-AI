import sys
import os
import numpy as np

print(f"Python: {sys.version}")

print("\n--- XGBoost ---")
try:
    import xgboost as xgb
    print(f"XGBoost Version: {xgb.__version__}")
    try:
        model = xgb.XGBRegressor(device='cuda', verbosity=0)
        X = np.random.rand(100, 10)
        y = np.random.rand(100)
        model.fit(X, y)
        print("✅ XGBoost GPU training test passed.")
    except Exception as e:
        print(f"❌ XGBoost GPU training failed: {e}")
except ImportError:
    print("XGBoost not installed.")
