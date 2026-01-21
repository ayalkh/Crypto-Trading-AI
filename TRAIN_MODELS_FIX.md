# train_models.py - Auto-Run Fix

## Problem
The `train_models.py` script was hanging when called from `setup_and_run.py` because it was waiting for user input:
- Line 878: Asking if user wants hyperparameter optimization
- Line 884: Waiting for ENTER key to start training

## Solution
Added command-line arguments to support automation:

### New Arguments:
```bash
python train_models.py --auto-run              # Skip all prompts
python train_models.py --auto-run --optimize   # Skip prompts + run optimization
python train_models.py --symbols BTC/USDT ETH/USDT  # Train specific symbols
python train_models.py --timeframes 1h 4h      # Train specific timeframes
```

### Usage:

**Interactive mode (default):**
```bash
python train_models.py
# Will ask for optimization (y/N)
# Will wait for ENTER to start
```

**Automated mode (for scripts):**
```bash
python train_models.py --auto-run
# Starts immediately with default parameters
# No user input required
```

**Automated with optimization:**
```bash
python train_models.py --auto-run --optimize
# Starts immediately with Optuna optimization
# Takes 2-3 hours
```

## Updated Files:
1. ✅ `train_models.py` - Added argparse support
2. ✅ `setup_and_run.py` - Now calls with `--auto-run` flag

## GPU Utils Status:
✅ **GPU utils are still present** in `crypto_ai/gpu_utils.py`

The CUDA warnings you saw are normal TensorFlow warnings, not errors. They don't affect training.

## Test It:
```bash
# Quick test (won't hang):
python train_models.py --auto-run --symbols BTC/USDT --timeframes 1h

# Full automated run:
python setup_and_run.py
```

---
*Fixed: 2026-01-21*
