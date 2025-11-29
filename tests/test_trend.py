import numpy as np
import pandas as pd
import pytest

from sp500_signal_classifier.features.trend import add_trend


def test_add_trend_computes_sma_slopes_and_distance_correctly():
    """
    Test that add_trend correctly computes SMAs and their slopes.

    Verifies that the function calculates simple moving averages over
    specified windows and their day-to-day slopes accurately.
    """
    # Create a simple DataFrame with known close prices
    df = pd.DataFrame(
        {"close": [10, 20, 30, 40, 50, 60, 70]},
        index=pd.date_range("2020-01-01", periods=7, freq="D"),
    )

    result = add_trend(df, sma_fast=3, sma_mid=5, sma_slow=7)

    # Expected SMA values
    expected_sma_3 = pd.Series(
        [np.nan, np.nan, 20.0, 30.0, 40.0, 50.0, 60.0],
        index=result.index,
        name="sma_3",
    )
    expected_sma_5 = pd.Series(
        [np.nan, np.nan, np.nan, np.nan, 30.0, 40.0, 50.0],
        index=result.index,
        name="sma_5",
    )
    expected_sma_7 = pd.Series(
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 40.0],
        index=result.index,
        name="sma_7",
    )

    pd.testing.assert_series_equal(result["sma_3"], expected_sma_3, check_names=True)
    pd.testing.assert_series_equal(result["sma_5"], expected_sma_5, check_names=True)
    pd.testing.assert_series_equal(result["sma_7"], expected_sma_7, check_names=True)

    # Expected slope values (first difference of SMAs)
    expected_sma_3_slope = expected_sma_3.diff()
    expected_sma_3_slope.name = "sma_3_slope"
    expected_sma_5_slope = expected_sma_5.diff()
    expected_sma_5_slope.name = "sma_5_slope"
    expected_sma_7_slope = expected_sma_7.diff()
    expected_sma_7_slope.name = "sma_7_slope"

    pd.testing.assert_series_equal(
        result["sma_3_slope"], expected_sma_3_slope, check_names=True
    )
    pd.testing.assert_series_equal(
        result["sma_5_slope"], expected_sma_5_slope, check_names=True
    )
    pd.testing.assert_series_equal(
        result["sma_7_slope"], expected_sma_7_slope, check_names=True
    )

    # Expected distance from sma_mid
    expected_distance = (df["close"] - expected_sma_5) / expected_sma_5
    expected_distance.name = "sma_5_distance"
    pd.testing.assert_series_equal(
        result["sma_5_distance"], expected_distance, check_names=True
    )
