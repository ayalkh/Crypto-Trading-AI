# 🚀 Strategic Plan: Improving Crypto ML Accuracy

Current Status: ~50.4% Accuracy (Random Guess Range)
Goal: Break the 55% threshold for consistent profitability.

## 1. 🧠 Feature Engineering (The Biggest Lever)
The current feature set (RSI, MACD, etc.) is standard. To beat the market, we need **unique info**:
*   **Micro-Structure Features:** Order book imbalance (if data available), bid-ask spread changes.
*   **Correlation Features:** BTC vs ETH movements, BTC vs S&P500 (if available), Dominance index.
*   **Time-Based Features:** Hour of day volatility patterns (crypto is 24/7 but volume peaks during US/EU/Asia overlaps).
*   **On-Chain Data:** (Advanced) Whale alerts, transaction volume spikes.

## 2. 🎯 Label Engineering (Crucial)
Currently, we predict `Close(t+1) > Close(t)`. This is "noisy" because a +0.01% move is treated the same as +5%.
*   **Fix:** Implementation of **"Triple Barrier Method"** or simply adding a **Threshold**.
    *   *Target:* 1 (Buy) if price > +0.5%, -1 (Sell) if price < -0.5%, 0 (Hold) otherwise.
    *   *Benefit:* The model learns to ignore small noise and only predict strong moves.

## 3. 🧹 Data Quality
*   **Outlier Removal:** Flash crashes or API glitches can confuse models.
*   **Stationarity:** Financial data is non-stationary. Ensure all features are percentage changes or log-returns, not raw prices. (We already do this mostly, but need to verify).

## 4. 🤖 Model Architecture Tweaks
*   **Sequence Length:** For the GRU/LSTM, increase lookback from standard candles to longer sequences (e.g., look at past 60 candles instead of 20).
*   **Custom Loss Function:** Penalize "wrong direction" errors more than "missed opportunity" errors.

## Recommended Immediate Next Steps
1.  **Implement Threshold Labeling:** Change the target to only be "1" if price moves > X%.
2.  **Add Correlation Features:** Simply adding ETH price changes as a feature for BTC prediction (and vice versa) often boosts accuracy by 1-2%.
