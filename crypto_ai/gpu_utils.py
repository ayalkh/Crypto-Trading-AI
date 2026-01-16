"""
GPU/CUDA Utility Module for Crypto ML System

Provides centralized GPU detection and configuration for:
- XGBoost (CUDA)
- CatBoost (CUDA)
- LightGBM (OpenCL/GPU)
- TensorFlow (CUDA)
"""
import logging
import os

# Suppress TensorFlow warnings during import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# GPU availability flags
XGBOOST_GPU_AVAILABLE = False
CATBOOST_GPU_AVAILABLE = False
LIGHTGBM_GPU_AVAILABLE = False
TENSORFLOW_GPU_AVAILABLE = False


def _check_xgboost_gpu() -> bool:
    """Check if XGBoost can use GPU"""
    try:
        import xgboost as xgb
        import numpy as np
        
        # Try to create a GPU-based DMatrix and train
        X = np.random.rand(10, 5)
        y = np.random.rand(10)
        dtrain = xgb.DMatrix(X, label=y)
        params = {'tree_method': 'hist', 'device': 'cuda'}
        xgb.train(params, dtrain, num_boost_round=1)
        return True
    except Exception:
        return False


def _check_catboost_gpu() -> bool:
    """Check if CatBoost can use GPU"""
    try:
        import catboost as cb
        import numpy as np
        
        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        model = cb.CatBoostRegressor(task_type="GPU", iterations=1, verbose=False)
        model.fit(X, y)
        return True
    except Exception:
        return False


def _check_lightgbm_gpu() -> bool:
    """Check if LightGBM can use GPU (requires GPU build)"""
    try:
        import lightgbm as lgb
        import numpy as np
        
        X = np.random.rand(10, 5)
        y = np.random.rand(10)
        train_data = lgb.Dataset(X, label=y)
        params = {'device': 'gpu', 'verbose': -1}
        lgb.train(params, train_data, num_boost_round=1)
        return True
    except Exception:
        return False


def _check_tensorflow_gpu() -> bool:
    """Check if TensorFlow can use GPU"""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        return len(gpus) > 0
    except Exception:
        return False


def detect_gpu_availability(log_results: bool = True) -> dict:
    """
    Detect GPU availability for all ML libraries.
    
    Returns:
        dict with keys: 'xgboost', 'catboost', 'lightgbm', 'tensorflow'
        Values are boolean indicating GPU availability.
    """
    global XGBOOST_GPU_AVAILABLE, CATBOOST_GPU_AVAILABLE
    global LIGHTGBM_GPU_AVAILABLE, TENSORFLOW_GPU_AVAILABLE
    
    XGBOOST_GPU_AVAILABLE = _check_xgboost_gpu()
    CATBOOST_GPU_AVAILABLE = _check_catboost_gpu()
    LIGHTGBM_GPU_AVAILABLE = _check_lightgbm_gpu()
    TENSORFLOW_GPU_AVAILABLE = _check_tensorflow_gpu()
    
    results = {
        'xgboost': XGBOOST_GPU_AVAILABLE,
        'catboost': CATBOOST_GPU_AVAILABLE,
        'lightgbm': LIGHTGBM_GPU_AVAILABLE,
        'tensorflow': TENSORFLOW_GPU_AVAILABLE
    }
    
    if log_results:
        logging.info("🖥️  GPU Availability Check:")
        logging.info(f"   XGBoost GPU:    {'✅ Ready' if results['xgboost'] else '❌ Not Available'}")
        logging.info(f"   CatBoost GPU:   {'✅ Ready' if results['catboost'] else '❌ Not Available'}")
        logging.info(f"   LightGBM GPU:   {'✅ Ready' if results['lightgbm'] else '❌ Not Available (needs GPU build)'}")
        logging.info(f"   TensorFlow GPU: {'✅ Ready' if results['tensorflow'] else '❌ Not Available (needs CUDA libs)'}")
    
    return results


def get_xgboost_gpu_params() -> dict:
    """Get XGBoost parameters for GPU training"""
    if XGBOOST_GPU_AVAILABLE:
        return {'tree_method': 'hist', 'device': 'cuda'}
    return {}


def get_catboost_gpu_params() -> dict:
    """Get CatBoost parameters for GPU training"""
    if CATBOOST_GPU_AVAILABLE:
        return {'task_type': 'GPU'}
    return {}


def get_lightgbm_gpu_params() -> dict:
    """Get LightGBM parameters for GPU training"""
    if LIGHTGBM_GPU_AVAILABLE:
        return {'device': 'gpu'}
    return {}


def configure_tensorflow_gpu():
    """Configure TensorFlow for GPU with memory growth"""
    if not TENSORFLOW_GPU_AVAILABLE:
        return False
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.info(f"✅ TensorFlow configured with {len(gpus)} GPU(s)")
        return True
    except Exception as e:
        logging.warning(f"⚠️ Failed to configure TensorFlow GPU: {e}")
        return False


# Run detection on import
detect_gpu_availability(log_results=False)
