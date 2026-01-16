
import tensorflow as tf
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import numpy as np

print("--- GPU Availability Check ---")

# TensorFlow
print(f"TensorFlow GPU visible: {tf.config.list_physical_devices('GPU')}")

# XGBoost
try:
    X = np.random.rand(10, 5)
    y = np.random.rand(10)
    dtrain = xgb.DMatrix(X, label=y)
    # Check for both old and new tree_method names
    params = {'tree_method': 'gpu_hist'} 
    xgb.train(params, dtrain, num_boost_round=1)
    print("XGBoost GPU (gpu_hist): Ready")
except Exception as e:
    try:
        params = {'tree_method': 'hist', 'device': 'cuda'}
        xgb.train(params, dtrain, num_boost_round=1)
        print("XGBoost GPU (device=cuda): Ready")
    except Exception as e2:
        print(f"XGBoost GPU: Not Ready (Errors: {e}, {e2})")

# LightGBM
try:
    X = np.random.rand(10, 5)
    y = np.random.rand(10)
    train_data = lgb.Dataset(X, label=y)
    params = {'device': 'gpu', 'verbose': -1}
    lgb.train(params, train_data, num_boost_round=1)
    print("LightGBM GPU: Ready")
except Exception as e:
    print(f"LightGBM GPU: Not Ready ({e})")

# CatBoost
try:
    X = np.random.rand(100, 5) # Needs enough data
    y = np.random.rand(100)
    model = cb.CatBoostRegressor(task_type="GPU", iterations=1, verbose=False)
    model.fit(X, y)
    print("CatBoost GPU: Ready")
except Exception as e:
    print(f"CatBoost GPU: Not Ready ({e})")
