# ML Model Performance Comparison

**Assessment Date:** January 15, 2026

## Summary

This report compares the **baseline** ML system against the **enhanced** version with:
- ✅ 13 new features (order book, funding rate, OI, arbitrage, correlation proxies)  
- ✅ Optuna hyperparameter tuning (15 trials per model)
- ✅ Feature selection (top 50 features)

---

## Results Comparison

### BTC/USDT 15m

| Model | Baseline Acc | Enhanced Acc | Change |
|-------|-------------|--------------|--------|
| LightGBM | 53.63% | 51.00% | **-2.63%** |
| XGBoost | 50.70% | 52.40% | +1.70% |
| Ensemble | 50.59% | 50.76% | +0.17% |

### ETH/USDT 15m

| Model | Baseline Acc | Enhanced Acc | Change |
|-------|-------------|--------------|--------|
| LightGBM | 53.86% | 52.64% | -1.22% |
| XGBoost | 53.04% | 52.29% | -0.75% |
| Ensemble | 52.81% | 52.99% | +0.18% |

### BTC/USDT 4h ⭐

| Model | Baseline Acc | Enhanced Acc | Change |
|-------|-------------|--------------|--------|
| LightGBM | 50.47% | **56.29%** | **+5.82%** |
| XGBoost | 51.41% | 50.63% | -0.78% |
| Ensemble | 51.72% | 50.31% | -1.41% |

### ETH/USDT 4h

| Model | Baseline Acc | Enhanced Acc | Change |
|-------|-------------|--------------|--------|
| LightGBM | 51.10% | 49.06% | -2.04% |
| XGBoost | 48.59% | 50.00% | +1.41% |
| Ensemble | 48.90% | 53.14% | **+4.24%** |

---

## Key Findings

### 🟢 Improvements
1. **BTC/USDT 4h LightGBM**: +5.82% accuracy (50.47% → 56.29%) - Best improvement
2. **ETH/USDT 4h Ensemble**: +4.24% accuracy (48.90% → 53.14%)
3. **BTC/USDT 15m XGBoost**: +1.70% accuracy

### 🔴 Regressions  
1. **BTC/USDT 15m LightGBM**: -2.63% accuracy
2. **ETH/USDT 4h LightGBM**: -2.04% accuracy

### Key Observations
- **4h timeframe benefited most** from the new features and tuning
- **15m timeframe showed mixed results** - potentially overfitting on high-frequency noise
- **LightGBM on 4h BTC** showed the strongest improvement (+5.82%)

---

## Technical Details

### New Features Added (13 total)
| Category | Features |
|----------|----------|
| Order Book Proxies | `ob_imbalance_ma12`, `ob_imbalance_delta`, `ob_spread_zscore` |
| Funding Rate | `funding_rate_zscore`, `funding_cum_3d` |
| Open Interest | `oi_pct_change`, `oi_sentiment` |
| Arbitrage | `arb_exch_delta_pct`, `arb_delta_zscore`, `arb_roc_divergence` |
| Correlation | `corr_btc_eth`, `rel_strength`, `corr_divergence` |

### Hyperparameter Tuning (Optuna)
- **Trials per model:** 15
- **Search space:** learning_rate, max_depth, n_estimators, regularization params

### Feature Selection
- **Method:** SelectKBest with mutual information
- **Selected:** Top 50 features from 109 total

---

## Recommendation

> **The enhanced version performs better on 4h timeframes but shows mixed results on 15m.**

For production use:
- ✅ Use **enhanced model for 4h predictions** (especially BTC/USDT LightGBM)
- ⚠️ Consider **baseline model for 15m predictions** or further tune

---

## Files
- Baseline results: `ml_reports/baseline_assessment_v1.json`
- Enhanced results: `ml_reports/enhanced_assessment_v2.json`
- Feature engineer: `crypto_ai/features/engineer.py`
- Enhanced assessment script: `enhanced_model_assessment.py`
