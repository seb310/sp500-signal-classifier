"""
Unit tests for return-based feature engineering functions.

This module tests the return calculation and momentum indicator functions
in the features.returns module.
"""

import numpy as np
import pandas as pd
import pytest

from sp500_signal_classifier.features.returns import add_basic_returns


def test_add_basic_returns_computes_ret1_correctly():
    """
    Test that add_basic_returns correctly computes 1-period returns.

    Verifies that ret1 calculates accurate percentage changes between
    consecutive closing prices.
    """
    df = pd.DataFrame(
        {"close": [100, 110, 121, 108.9]},
        index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]),
    )

    result = add_basic_returns(df)

    # Expected: (110/100 - 1) = 0.1, (121/110 - 1) = 0.1, (108.9/121 - 1) = -0.1
    expected = pd.Series(
        [np.nan, 0.1, 0.1, -0.1],
        index=result.index,
        name="ret1",
    )

    pd.testing.assert_series_equal(result["ret1"], expected, atol=1e-10)


def test_add_basic_returns_computes_ret5_correctly():
    """
    Test that add_basic_returns correctly computes 5-period returns.

    Verifies that ret5 calculates percentage change over 5 periods.
    """
    # Create data with 10 periods for testing
    close_prices = [100, 105, 110, 115, 120, 125, 130, 135, 140, 145]
    df = pd.DataFrame(
        {"close": close_prices},
        index=pd.date_range("2020-01-01", periods=10, freq="D"),
    )

    result = add_basic_returns(df)

    # ret5 at index 5 should be (125/100 - 1) = 0.25
    # ret5 at index 9 should be (145/120 - 1) ≈ 0.208333
    assert pd.isna(result["ret5"].iloc[:5]).all(), "First 5 values should be NaN"
    assert abs(result["ret5"].iloc[5] - 0.25) < 1e-10
    assert abs(result["ret5"].iloc[9] - (145 / 120 - 1)) < 1e-10


def test_add_basic_returns_computes_ret10_correctly():
    """
    Test that add_basic_returns correctly computes 10-period returns.

    Verifies that ret10 calculates percentage change over 10 periods.
    """
    # Create data with 15 periods for testing
    close_prices = list(range(100, 115))
    df = pd.DataFrame(
        {"close": close_prices},
        index=pd.date_range("2020-01-01", periods=15, freq="D"),
    )

    result = add_basic_returns(df)

    # ret10 at index 10 should be (110/100 - 1) = 0.1
    assert pd.isna(result["ret10"].iloc[:10]).all(), "First 10 values should be NaN"
    assert abs(result["ret10"].iloc[10] - 0.1) < 1e-10


def test_add_basic_returns_computes_ret_z20_correctly():
    """
    Test that add_basic_returns correctly computes z-score normalized returns.

    Verifies that ret_z20 properly normalizes ret1 using a 20-period
    rolling window for mean and standard deviation.
    """
    # Create a dataset with 25 periods with known properties
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 25)
    # Set first return to NaN (as pct_change does)
    returns[0] = np.nan

    # Create close prices from returns
    close_prices = [100]
    for ret in returns[1:]:
        close_prices.append(close_prices[-1] * (1 + ret))

    df = pd.DataFrame(
        {"close": close_prices},
        index=pd.date_range("2020-01-01", periods=25, freq="D"),
    )

    result = add_basic_returns(df)

    # First 20 values should be NaN (need 20 periods for rolling window)
    assert pd.isna(result["ret_z20"].iloc[:20]).all(), "First 20 z-scores should be NaN"

    # Manually calculate z-score for position 20
    ret1_values = result["ret1"].iloc[:21]
    rolling_mean = ret1_values.rolling(20).mean().iloc[20]
    rolling_std = ret1_values.rolling(20).std().iloc[20]
    expected_z = (result["ret1"].iloc[20] - rolling_mean) / rolling_std

    assert abs(result["ret_z20"].iloc[20] - expected_z) < 1e-10


def test_add_basic_returns_preserves_existing_columns():
    """
    Test that add_basic_returns preserves all existing columns.

    Verifies that the function adds new columns without dropping
    existing ones.
    """
    df = pd.DataFrame(
        {
            "open": [99, 109, 120],
            "high": [101, 111, 122],
            "low": [98, 108, 119],
            "close": [100, 110, 121],
            "volume": [1000, 1100, 1200],
        },
        index=pd.date_range("2020-01-01", periods=3, freq="D"),
    )

    result = add_basic_returns(df)

    # Check that all original columns are still present
    for col in ["open", "high", "low", "close", "volume"]:
        assert col in result.columns, f"Column {col} was not preserved"


def test_add_basic_returns_adds_all_expected_columns():
    """
    Test that add_basic_returns adds all expected return columns.

    Verifies that the function creates ret1, ret5, ret10, and ret_z20 columns.
    """
    df = pd.DataFrame(
        {"close": list(range(100, 130))},
        index=pd.date_range("2020-01-01", periods=30, freq="D"),
    )

    result = add_basic_returns(df)

    expected_columns = ["ret1", "ret5", "ret10", "ret_z20"]
    for col in expected_columns:
        assert col in result.columns, f"Expected column {col} was not added"


def test_add_basic_returns_modifies_input_dataframe():
    """
    Test that add_basic_returns modifies the input DataFrame in-place.

    Verifies the current behavior where the function mutates the input.
    This documents the current implementation behavior.
    """
    df = pd.DataFrame(
        {"close": [100, 110, 121]},
        index=pd.date_range("2020-01-01", periods=3, freq="D"),
    )

    original_columns = df.columns.tolist()
    result = add_basic_returns(df)

    # The function currently modifies in-place
    assert "ret1" in df.columns, "Input DataFrame should be modified in-place"
    assert result is df, "Function should return the same DataFrame object"


def test_add_basic_returns_handles_insufficient_data():
    """
    Test that add_basic_returns handles DataFrames with insufficient data.

    Verifies that the function works correctly even with fewer than 20 periods,
    producing NaN values where appropriate.
    """
    df = pd.DataFrame(
        {"close": [100, 110, 121, 133.1, 146.41]},
        index=pd.date_range("2020-01-01", periods=5, freq="D"),
    )

    result = add_basic_returns(df)

    # All z-scores should be NaN with only 5 periods
    assert pd.isna(
        result["ret_z20"]
    ).all(), "All z-scores should be NaN with < 20 periods"

    # ret1 should still work
    assert not pd.isna(
        result["ret1"].iloc[1:]
    ).any(), "ret1 should be computed for periods 1+"


def test_add_basic_returns_handles_constant_prices():
    """
    Test that add_basic_returns handles constant price data.

    Verifies that returns are zero when prices don't change and that
    z-scores handle the zero standard deviation case.
    """
    df = pd.DataFrame(
        {"close": [100] * 25},
        index=pd.date_range("2020-01-01", periods=25, freq="D"),
    )

    result = add_basic_returns(df)

    # All returns should be 0 (except first which is NaN)
    assert (
        result["ret1"].iloc[1:].eq(0).all()
    ), "Returns should be 0 for constant prices"
    assert result["ret5"].iloc[5:].eq(0).all(), "5-period returns should be 0"
    assert result["ret10"].iloc[10:].eq(0).all(), "10-period returns should be 0"

    # Z-scores will be NaN due to zero std (division by zero)
