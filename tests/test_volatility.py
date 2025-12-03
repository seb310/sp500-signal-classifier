import numpy as np
import pandas as pd
import pytest

from sp500_signal_classifier.features.volatility import add_volatility


def make_dummy_df():
    # Simple increasing series with small intraday ranges
    idx = pd.date_range("2020-01-01", periods=30, freq="B")
    close = pd.Series(np.linspace(100, 110, len(idx)), index=idx)
    high = close + 1.0
    low = close - 1.0
    df = pd.DataFrame({"close": close, "high": high, "low": low})

    return df


def test_add_volatility_basic_properties():
    """
    Basic tests for add_volatility function.

    Tests:
    1. ``vol10``, ``tr``, ``atr_14``, and ``atr_14_norm`` columns are added.
    2. No infinite values in the resulting DataFrame.
    3. Checks for correct number of NaNs in volatility columns (``vol10``, ``atr_14``, ``atr_14_norm``).
    """
    df = make_dummy_df()
    out_df = add_volatility(df.copy(), vol_window=10)

    # Check that new columns are added
    expected_columns = ["volatility_10", "tr", "atr_14", "atr_14_norm"]
    for col in expected_columns:
        assert col in out_df.columns, f"Missing expected column: {col}"

    # Check for no infinite values
    assert (
        np.isfinite(
            out_df.loc[
                ~out_df[["volatility_10", "tr", "atr_14", "atr_14_norm"]]
                .isna()
                .any(axis=1),
                ["volatility_10", "tr", "atr_14", "atr_14_norm"],
            ]
        )
        .all()
        .all()
    )

    # Check for correct number of NaNs
    assert (
        out_df["volatility_10"].isna().sum() == 9
    ), "Incorrect number of NaNs in volatility_10"
    assert out_df["atr_14"].isna().sum() == 13, "Incorrect number of NaNs in atr_14"
    assert (
        out_df["atr_14_norm"].isna().sum() == 13
    ), "Incorrect number of NaNs in atr_14_norm"


def test_add_volatility_tr_calculation():
    """
    Test that True Range (TR) is calculated correctly.

    Verifies that the TR column matches the maximum of the three components:
    high-low, high-prev_close, and prev_close-low.
    """
    df = make_dummy_df()
    out_df = add_volatility(df.copy(), vol_window=10)

    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr_expected = pd.DataFrame(
        {
            "hl": high - low,
            "hc": (high - prev_close).abs(),
            "lc": (low - prev_close).abs(),
        }
    ).max(axis=1)

    pd.testing.assert_series_equal(
        out_df["tr"],
        tr_expected,
        check_names=False,
        atol=1e-10,
    )


def test_add_volatility_atr_calculation():
    """
    Test that Average True Range (ATR) is calculated correctly.

    Verifies that the atr_14 column matches the rolling mean of the TR column
    over a 14-period window.
    """
    df = make_dummy_df()
    out_df = add_volatility(df.copy(), vol_window=10)

    tr = out_df["tr"]
    atr_expected = tr.rolling(window=14).mean()

    pd.testing.assert_series_equal(
        out_df["atr_14"],
        atr_expected,
        check_names=False,
        atol=1e-10,
    )


def test_add_volatility_atr_normalization():
    """
    Test that normalized ATR (atr_14_norm) is calculated correctly.

    Verifies that the atr_14_norm column matches atr_14 divided by the close price.
    """
    df = make_dummy_df()
    out_df = add_volatility(df.copy(), vol_window=10)

    atr_14 = out_df["atr_14"]
    close = df["close"]
    atr_norm_expected = atr_14 / close

    pd.testing.assert_series_equal(
        out_df["atr_14_norm"],
        atr_norm_expected,
        check_names=False,
        atol=1e-10,
    )


def test_add_volatility_no_lookahead():
    """
    Test that add_volatility does not introduce lookahead bias.

    Ensures that the calculated volatility features at each time point
    depend only on current and past data.
    """
    df = make_dummy_df()
    out_df = add_volatility(df.copy(), vol_window=10)

    # Check that for each row, the volatility features depend only on current and past data
    for i in range(len(df)):
        if i >= 9:  # vol10 requires at least 10 data points
            vol10 = out_df["volatility_10"].iloc[i]
            expected_vol10 = df["close"].iloc[i - 9 : i + 1].std()
            assert (
                abs(vol10 - expected_vol10) < 1e-10
            ), f"volatility_10 lookahead at index {i}"

        if i >= 13:  # atr_14 requires at least 14 data points
            atr_14 = out_df["atr_14"].iloc[i]
            tr_values = out_df["tr"].iloc[i - 13 : i + 1]
            expected_atr_14 = tr_values.mean()
            assert (
                abs(atr_14 - expected_atr_14) < 1e-10
            ), f"atr_14 lookahead at index {i}"
