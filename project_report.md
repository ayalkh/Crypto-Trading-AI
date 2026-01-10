# Crypto-Trading-AI Project Report
*Status: Post-Merge Assessment (Jan 10, 2026)*

## 1. System Integration & Architecture
*   **Orchestrator**: `crypto_control_center.py` (CLI for management).
*   **ML Engine**: `optimized_ml_system.py` (Manages LightGBM/XGBoost/CatBoost).
*   **Data Pipeline**: `comprehensive_ml_collector_v2.py` -> SQLite (`data/ml_crypto_data.db`).

---

## 2. Feature Engineering Logic
*   **Current State**: **Hybrid / Conflict**.
    *   The `FeatureEngineer` code generates **90 features** (Enhanced set).
    *   The simplified/baseline models expect **66 features**.
    *   *Impact*: You cannot strictly load old models with new code without handling this mismatch (as seen in `honest_benchmark.py` failure).

---

## 3. Model Performance Assessment
### **Verdict: Random / Useless (~52%)**
The "85% Accuracy" seen previously was confirmed to be an **artifact of overfitting** or data leakage.

**Evidence (Fair Comparison Benchmark):**
*   **Baseline Features (66)**: 52.58% Accuracy (Coin Flip).
*   **Enhanced Features (90)**: 51.15% Accuracy (Worse than Baseline).
*   **Enhanced + Selection**: 52.35% Accuracy (No significant improvement).

**Interpretation:**
The models are failing to find signal in the current features, whether utilizing the old (66) or new (90) set. The "Enhanced" set appears to add more noise than signal, slightly degrading performance unless feature selection is applied.

---

## 4. Risks & Recommendations

### 1. Hardcoded Hyperparameters
*   **Risk**: `optimized_ml_system.py` uses static parameters (e.g., `n_estimators=500`, `depth=7`).
*   **Fix**: These are likely too complex for the noisy data, leading to the rapid overfitting seen in earlier "85%" runs. We absolutely need **Dynamic Hyperparameter Tuning** (Optuna).

### 2. Feature Selection
*   **Risk**: Feeding 90 noisy features directly to the model hurts performance (51% acc).
*   **Fix**: The benchmark showed `SelectKBest` recovered some performance. This must be integrated into the main pipeline.

### 3. Immediate Action Plan
1.  **Stop Trading**: Do not run this model with real money.
2.  **Implement Strategy**:
    *   **Regularization**: Reduced depth, increased penalties.
    *   **Feature Selection**: Auto-drop bottom 50% features.
    *   **Stationarity**: Ensure all features are trend-agnostic (e.g., Log Returns).
