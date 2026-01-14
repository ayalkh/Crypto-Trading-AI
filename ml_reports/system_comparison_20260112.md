# ML System Performance Assessment

## Overview
We conducted a comparative assessment between the currently active production system (`optimized_ml_system.py`) and the experimental enhanced version (`enhanced_ml_system.py`).

## Results Summary

| Metric | Optimized System (Baseline) | Enhanced System (Experimental) | Change | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Direction Accuracy** | **50.77%** | 50.31% | -0.47% | ❌ Worse |
| **R² Score** | **-0.0050** | -0.0643 | -0.0593 | ❌ Worse |
| **MSE (Error)** | **0.000068** | 0.000085 | +26% | ❌ Worse |

## Detailed Findings

### 1. Optimized System (Winner)
*   **File:** `optimized_ml_system.py`
*   **Strengths:** Better generalization, lower error rates, and slightly higher directional accuracy.
*   **Configuration:** Uses an ensemble of LightGBM, XGBoost, CatBoost, and GRU with dynamic weighting.

### 2. Enhanced System
*   **File:** `enhanced_ml_system.py`
*   **Features:** Attempts to improve performance via stricter `SelectKBest` feature selection and `Optuna` hyperparameter tuning.
*   **Outcome:** The aggressive feature selection or tuning appears to be counter-productive, likely discarding useful signals or overfitting to the validation set which doesn't translate to the test set.

## Recommendation
**Stick with `optimized_ml_system.py`.** The experimental changes in the enhanced version are currently degrading performance. Future improvements should likely focus on feature engineering rather than aggressive feature selection.
