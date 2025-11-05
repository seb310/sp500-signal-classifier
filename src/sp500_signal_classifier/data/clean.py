"""
Data cleaning utilities for SP500 signal classifier.

This module provides functions for preprocessing and transforming
raw financial time series data.
"""

import pandas as pd


def to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the DataFrame is sorted by date index for daily frequency.

    This function sorts the DataFrame by its index to ensure proper
    chronological ordering. It prepares data for business day frequency
    alignment without forward-filling price data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex containing OHLCV data.

    Returns
    -------
    pd.DataFrame
        Sorted DataFrame with the same columns as input.
    """
    # ensure business day frequency; forward-fill only for index alignment (not prices)
    df = df.sort_index()
    return df


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 1-day returns to the DataFrame.

    Calculates the percentage change in closing price over a 1-day period
    and adds it as a new column 'ret1'.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least a 'close' column with price data.

    Returns
    -------
    pd.DataFrame
        Copy of input DataFrame with additional 'ret1' column containing
        1-day percentage returns.
    """
    df = df.copy()
    df["ret1"] = df["close"].pct_change(1)
    return df
