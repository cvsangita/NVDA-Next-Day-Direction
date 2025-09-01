# NVDA Next-Day Direction — Random Forest vs XGBoost (Walk-Forward Backtest)

Predict whether **tomorrow’s close** for NVIDIA (NVDA) will be higher than **today’s** using daily OHLCV data, simple **rolling features**, and a realistic **expanding walk-forward** evaluation. We compare **Random Forest** and **XGBoost**, convert probabilities to trade signals with a **threshold**, and report classification metrics plus a simple (gross) PnL proxy.

---

## In short:

- **Target:** `1` if `Close(t+1) > Close(t)`, else `0`  
- **Features:** Raw OHLCV + rolling **Close/MA ratios** and **Trend (past up-day counts)** over multiple horizons  
- **Eval:** **Walk-forward** (expanding window) backtest; no look-ahead  
- **Models:** RandomForestClassifier, XGBClassifier  
- **Outputs:** Precision/Recall/F1, Coverage (% days traded), confusion matrix, simple PnL summary, threshold sweeps

---

## Why this project?

1. Time-aware modeling: Markets are not i.i.d.; you need methods that respect order, drift, and autocorrelation.
2. Leakage-free features: Show how to build causal (past-only) features so today’s decision doesn’t peek at tomorrow.
3. Realistic evaluation: Use walk-forward backtesting (train past → test future) instead of random splits.
4. Actionable outputs: Turn probabilities into trade signals with a tunable threshold (hit-rate vs. activity).
5. Model comparison: Provide a clean, reproducible benchmark of Random Forest vs. XGBoost on daily NVDA data.
6. Starter template: A concise, end-to-end pipeline you can adapt to other assets and features.

---

## Data & Label

- Source: [`yfinance`](https://github.com/ranaroussi/yfinance), **adjusted** OHLCV to avoid split/dividend jumps.
- Columns used: `Open, High, Low, Close, Volume`
- Label:
  ```python
  nvda["Tomorrow"] = nvda["Close"].shift(-1)
  nvda["Target"]   = (nvda["Tomorrow"] > nvda["Close"]).astype(int)
