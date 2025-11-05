"""
Unit tests for data cleaning functions.

This module tests the data preprocessing and transformation functions
in the clean module.
"""

import numpy as np
import pandas as pd

from sp500_signal_classifier.data.clean import add_returns, to_daily


def test_to_daily_sorts_index():
    """
    Test that to_daily enforces a chronologically ordered daily index.

    Ensures that a DataFrame with dates out of order is sorted in ascending
    order, preparing the data for continuous daily (business-day) alignment
    in downstream features and joins.
    """
    # unsorted example data
    df = pd.DataFrame(
        {
            "close": [102, 100, 101],
        },
        index=pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-02"]),
    )

    sorted_df = to_daily(df)

    # Index should now be sorted ascending
    assert all(
        sorted_df.index == sorted(sorted_df.index)
    ), "Index is not sorted ascending"


def test_add_returns_computes_correct_values():
    """
    Test that add_returns correctly computes 1-day percentage returns.

    Verifies that the function calculates accurate percentage changes
    between consecutive closing prices, with NaN for the first value.
    """
    # simple 3-day close prices
    df = pd.DataFrame(
        {"close": [100, 110, 121]},
        index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
    )

    result = add_returns(df)

    # expected simple returns: (110/100 - 1) = 0.1, (121/110 - 1) = 0.1
    expected = pd.Series([np.nan, 0.1, 0.1], index=result.index, name="ret1")

    pd.testing.assert_series_equal(result["ret1"], expected, check_names=True)


def test_add_returns_does_not_modify_original():
    """
    Test that add_returns does not mutate the original DataFrame.

    Verifies that the function returns a copy and leaves the input
    DataFrame unchanged.
    """
    df = pd.DataFrame({"close": [1, 2, 3]})
    df_copy = df.copy()
    _ = add_returns(df)
    # ensure original not modified
    pd.testing.assert_frame_equal(df, df_copy)
