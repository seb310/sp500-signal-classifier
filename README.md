# SP500 Signal Classifier (Work in Progress)
Long/flat next-day classifier for SPY with walk-forward CV, costs, and
reproducible backtests.

# Project Structure
sp500-signal-classifier/
├─ README.md
├─ LICENSE
├─ .gitignore
├─ pyproject.toml
├─ .pre-commit-config.yaml
├─ Makefile
├─ params.yaml               # central config (paths, horizons, costs, CV, etc.)
├─ data/
│  ├─ raw/                   # untouched downloads (e.g., SPY, VIX, yields)
│  ├─ interim/               # cleaned & aligned bars
│  └─ processed/             # feature matrices + labels (train/valid/test splits)
├─ notebooks/
├─ reports/
│  ├─ figures/               # plots exported by scripts/notebooks
│  └─ metrics/               # Cacktests & CV metrics
├─ models/                   # saved models
├─ backtests/
├─ src/
│  ├─ __init__.py
│  ├─ config.py              # loads params.yaml; small helper for paths
│  ├─ data/
│  │  ├─ fetch.py            # yfinance pulls; caching; trading calendar
│  │  └─ clean.py            # OHLCV alignment, missing days, splits/holidays
│  ├─ features/
│  │  ├─ returns.py          # 1D/5D/10D, z-scores
│  │  ├─ trend.py            # SMAs, slopes, distance-to-SMA
│  │  ├─ volatility.py       # std(10), ATR(14)
│  │  ├─ rsi.py
│  │  └─ volume.py           # volume z-score
│  ├─ labels/
│  │  └─ binary.py           # threshold τ; make y_{t+1}; leakage-safe
alignment
│  ├─ modeling/
│  │  ├─ datasets.py         # build X,y; train/val/test splits (walk-forward)
│  │  ├─ baselines.py        # MA5>MA20, logistic-reg baseline
│  │  ├─ models.py           # sklearn models, calibration, pipelines
│  │  └─ cv.py               # expanding/walk-forward CV utilities
│  ├─ trading/
│  │  ├─ costs.py            # per-side/round-trip bps
│  │  ├─ rules.py            # prob→position, thresholds, position sizing
│  │  └─ backtest.py         # long/flat engine; vectorized & deterministic
│  └─ evaluation/
│     ├─ metrics.py          # accuracy on non-neutral set, Sharpe, MDD,
turnover
│     └─ plots.py            # equity curves, drawdowns, confusion matrices
├─ scripts/
└─ tests/
   ├─ test_clean.py          # chronological order; 1-day returns computation
   ├─ test_fetch.py          # dataframe consistency
   ├─ test_returns.py
