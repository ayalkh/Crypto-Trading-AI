"""
CUDA vs CPU Training Time Benchmark
Compares training time for XGBoost and CatBoost with/without GPU
"""
import time
import numpy as np
import xgboost as xgb
import catboost as cb
import sqlite3
import pandas as pd


def load_sample_data():
    """Load training data from database"""
    db_path = 'data/ml_crypto_data.db'
    conn = sqlite3.connect(db_path)
    
    query = """
    SELECT timestamp, open, high, low, close, volume
    FROM price_data 
    WHERE symbol = 'BTC/USDT' AND timeframe = '1h'
    ORDER BY timestamp DESC
    LIMIT 5000
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print("❌ No data found, generating synthetic data...")
        n_samples = 5000
        n_features = 50
        X = np.random.rand(n_samples, n_features).astype(np.float32)
        y = np.random.rand(n_samples).astype(np.float32)
        return X, y
    
    # Create simple features
    df['returns'] = df['close'].pct_change()
    for i in range(1, 51):
        df[f'lag_{i}'] = df['close'].shift(i)
    
    df.dropna(inplace=True)
    
    feature_cols = [c for c in df.columns if c.startswith('lag_')]
    X = df[feature_cols].values.astype(np.float32)
    y = df['returns'].values.astype(np.float32)
    
    return X, y


def benchmark_xgboost(X, y, n_rounds=500):
    """Benchmark XGBoost CPU vs GPU"""
    print("\n" + "="*60)
    print("🔥 XGBOOST BENCHMARK")
    print("="*60)
    print(f"   Data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   Iterations: {n_rounds}")
    
    dtrain = xgb.DMatrix(X, label=y)
    
    # CPU Training
    print("\n📊 Training on CPU...")
    cpu_params = {
        'tree_method': 'hist',
        'max_depth': 6,
        'learning_rate': 0.05,
        'verbosity': 0
    }
    
    start = time.time()
    xgb.train(cpu_params, dtrain, num_boost_round=n_rounds)
    cpu_time = time.time() - start
    print(f"   ⏱️  CPU Time: {cpu_time:.2f} seconds")
    
    # GPU Training
    print("\n📊 Training on GPU (CUDA)...")
    gpu_params = {
        'tree_method': 'hist',
        'device': 'cuda',
        'max_depth': 6,
        'learning_rate': 0.05,
        'verbosity': 0
    }
    
    try:
        start = time.time()
        xgb.train(gpu_params, dtrain, num_boost_round=n_rounds)
        gpu_time = time.time() - start
        print(f"   ⏱️  GPU Time: {gpu_time:.2f} seconds")
        
        speedup = cpu_time / gpu_time
        print(f"\n   🚀 SPEEDUP: {speedup:.2f}x faster with GPU")
        return cpu_time, gpu_time, speedup
    except Exception as e:
        print(f"   ❌ GPU training failed: {e}")
        return cpu_time, None, None


def benchmark_catboost(X, y, n_iterations=500):
    """Benchmark CatBoost CPU vs GPU"""
    print("\n" + "="*60)
    print("🐱 CATBOOST BENCHMARK")
    print("="*60)
    print(f"   Data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   Iterations: {n_iterations}")
    
    # CPU Training
    print("\n📊 Training on CPU...")
    cpu_model = cb.CatBoostRegressor(
        iterations=n_iterations,
        depth=6,
        learning_rate=0.05,
        verbose=False,
        task_type='CPU'
    )
    
    start = time.time()
    cpu_model.fit(X, y)
    cpu_time = time.time() - start
    print(f"   ⏱️  CPU Time: {cpu_time:.2f} seconds")
    
    # GPU Training
    print("\n📊 Training on GPU (CUDA)...")
    gpu_model = cb.CatBoostRegressor(
        iterations=n_iterations,
        depth=6,
        learning_rate=0.05,
        verbose=False,
        task_type='GPU'
    )
    
    try:
        start = time.time()
        gpu_model.fit(X, y)
        gpu_time = time.time() - start
        print(f"   ⏱️  GPU Time: {gpu_time:.2f} seconds")
        
        speedup = cpu_time / gpu_time
        print(f"\n   🚀 SPEEDUP: {speedup:.2f}x faster with GPU")
        return cpu_time, gpu_time, speedup
    except Exception as e:
        print(f"   ❌ GPU training failed: {e}")
        return cpu_time, None, None


def main():
    print("\n" + "="*60)
    print("⚡ CUDA vs CPU TRAINING BENCHMARK")
    print("="*60)
    
    # Load data
    print("\n📂 Loading data...")
    X, y = load_sample_data()
    print(f"   ✅ Loaded {len(X)} samples with {X.shape[1]} features")
    
    # Run benchmarks
    xgb_cpu, xgb_gpu, xgb_speedup = benchmark_xgboost(X, y, n_rounds=500)
    cb_cpu, cb_gpu, cb_speedup = benchmark_catboost(X, y, n_iterations=500)
    
    # Summary
    print("\n" + "="*60)
    print("📊 BENCHMARK SUMMARY")
    print("="*60)
    print(f"\n{'Model':<15} {'CPU Time':<12} {'GPU Time':<12} {'Speedup':<10}")
    print("-"*50)
    
    if xgb_gpu:
        print(f"{'XGBoost':<15} {xgb_cpu:.2f}s{'':<6} {xgb_gpu:.2f}s{'':<6} {xgb_speedup:.2f}x")
    else:
        print(f"{'XGBoost':<15} {xgb_cpu:.2f}s{'':<6} {'N/A':<12} {'N/A':<10}")
    
    if cb_gpu:
        print(f"{'CatBoost':<15} {cb_cpu:.2f}s{'':<6} {cb_gpu:.2f}s{'':<6} {cb_speedup:.2f}x")
    else:
        print(f"{'CatBoost':<15} {cb_cpu:.2f}s{'':<6} {'N/A':<12} {'N/A':<10}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
