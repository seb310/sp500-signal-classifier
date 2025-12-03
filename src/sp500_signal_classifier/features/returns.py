"""
Feature engineering module for return-based features.

This module provides functions for calculating various return metrics
and momentum indicators from price data. It includes percentage changes
over multiple periods and normalized returns.
"""

import pandas as pd


def add_basic_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic return features to the DataFrame.

    Calculates return metrics and a normalized return signal. The function
    adds the following columns to the input DataFrame:

    - ``ret1``: 1-period percentage change in close price.
    - ``ret5``: 5-period percentage change in close price.
    - ``ret10``: 10-period percentage change in close price.
    - ``ret_z20``: Z-score of ``ret1`` normalized over a 20-period rolling
      window.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least a ``close`` column with price data.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with the new return columns added.

    Notes
    -----
    The function modifies the input DataFrame in-place and also returns it.
    The ``ret_z20`` column will be NaN for the first 20 periods because the
    rolling window requires sufficient history.
    """
    df["ret1"] = df["close"].pct_change(1)
    df["ret5"] = df["close"].pct_change(5)
    df["ret10"] = df["close"].pct_change(10)
    df["ret_z20"] = (df["ret1"] - df["ret1"].rolling(20).mean()) / df["ret1"].rolling(
        20
    ).std()
    return df
