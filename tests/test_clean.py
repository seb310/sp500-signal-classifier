"""
Unit tests for data cleaning functions.

This module tests the data preprocessing and transformation functions
in the clean module.
"""

import numpy as np
import pandas as pd

from sp500_signal_classifier.data.clean import to_daily


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
