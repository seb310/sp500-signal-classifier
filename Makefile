PY=python
PARAMS=--params params.yaml

setup:
		conda env create -f environment.yml || true
		conda activate sp500-signal-classifier

fetch:
		$(PY) scripts/01_fetch_data.py $(PARAMS)

features:
		$(PY) scripts/02_build_features.py $(PARAMS)

baselines:
		$(PY) scripts/03_train_baselines.py $(PARAMS)

train:
		$(PY) scripts/04_train_model.py $(PARAMS)

backtest:
		$(PY) scripts/05_backtest.py $(PARAMS)

report:
		$(PY) scripts/06_report.py $(PARAMS)
