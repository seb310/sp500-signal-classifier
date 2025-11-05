from src.data.fetch import fetch_ohlcv

spy = fetch_ohlcv("SPY", start="2010-01-01")
print(spy.head())
