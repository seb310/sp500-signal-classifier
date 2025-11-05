# SP500 Signal Classifier
Long/flat next-day classifier for SPY with walk-forward CV, costs, and
reproducible backtests.

# Project Structure
sp500-signal-classifier/
├─ README.md
├─ LICENSE
├─ .gitignore
├─ pyproject.toml            # or requirements.txt + setup.cfg
├─ .pre-commit-config.yaml
├─ Makefile
├─ params.yaml               # central config (paths, horizons, costs, CV,
etc.)
├─ data/
│  ├─ raw/                   # untouched downloads (e.g., SPY, VIX, yields)
│  ├─ interim/               # cleaned & aligned bars
│  └─ processed/             # feature matrices + labels (train/valid/test
splits)
├─ notebooks/
│  ├─ 00_exploration.ipynb
│  ├─ 10_feature_checks.ipynb
│  └─ 20_model_diagnostics.ipynb
├─ reports/
│  ├─ figures/               # plots exported by scripts/notebooks
│  └─ metrics/               # JSON/CSV of backtest & CV metrics
├─ models/                   # saved models (joblib) + calibration objects
├─ backtests/                # serialized equity curves, trades, tearsheets
├─ src/
│  ├─ __init__.py
│  ├─ config.py              # loads params.yaml; small helper for paths
│  ├─ data/
│  │  ├─ fetch.py            # yfinance/FRED pulls; caching; trading calendar
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
│  ├─ 01_fetch_data.py
│  ├─ 02_build_features.py
│  ├─ 03_train_baselines.py
│  ├─ 04_train_model.py
│  ├─ 05_backtest.py
│  └─ 06_report.py           # dump metrics JSON + figures
└─ tests/
   ├─ test_alignment.py      # no leakage; shapes; NaNs only at start
      ├─ test_backtest.py       # cost application, compounding, turnovers
         └─ test_metrics.py        # Sharpe/MDD correctness
