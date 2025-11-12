"""
Feature engineering module for return-based features.

This module provides functions for calculating various return metrics
and momentum indicators from price data, including percentage changes
over multiple periods and normalized returns.
"""

import pandas as pd


def add_basic_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic return features to the dataframe.

    Calculates various return metrics including percentage changes over different
    periods and a z-score normalized return.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing at least a 'close' column with price data.

    Returns
    -------
    pd.DataFrame
        The input dataframe with the following columns added:
        - ret1: 1-period percentage change in close price -> Captures most recent momentum
        - ret5: 5-period percentage change in close price -> Captures medium-term trends
        - ret10: 10-period percentage change in close price -> Slower momentum signal
        - ret_z20: Z-score of ret1 normalized over a 20-period rolling window -> Captures how extreme today's move is relative to recent history
                                                                                 > 2: unusually strong up day, < -2: unusually strong down day.
                                                                                Useful for identifying shock days.

    Notes
    -----
    The function modifies the input dataframe in-place and also returns it.
    The ret_z20 calculation may produce NaN values for the first 20 periods
    due to insufficient data for the rolling window calculation.
    """
    df["ret1"] = df["close"].pct_change(1)
    df["ret5"] = df["close"].pct_change(5)
    df["ret10"] = df["close"].pct_change(10)
    df["ret_z20"] = (df["ret1"] - df["ret1"].rolling(20).mean()) / df["ret1"].rolling(
        20
    ).std()
    return df
