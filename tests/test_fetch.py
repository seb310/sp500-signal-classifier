from sp500_signal_classifier.data.fetch import fetch_ohlcv


def test_fetch_ohlcv_runs():
    """Basic test that fetch_ohlcv downloads and returns a valid DataFrame"""
    df = fetch_ohlcv("SPY", start="2005-01-01", end="2010-01-10")

    # Basic sanity tests
    assert not df.empty, "DataFrame is empty"
    expected_columns = {"open", "high", "low", "close", "adj close", "volume"}
    assert expected_columns.issubset(
        df.columns
    ), f"Missing columns: {expected_columns - set(df.columns)}"
    assert df.index.name == "date", f"Index name is not 'date', got {df.index.name}"
    assert df.index.is_monotonic_increasing, "Index is not sorted in ascending order"
