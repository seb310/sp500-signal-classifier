"""Feature engineering module for trend-based features.

This module provides functions for calculating trend indicators such as
moving averages and their slopes to capture directional momentum in price data.
"""

import pandas as pd


def add_trend(
    df: pd.DataFrame, sma_fast: int = 5, sma_mid: int = 20, sma_slow: int = 50
) -> pd.DataFrame:
    """
    Add simple moving average (SMA) trend features to the DataFrame.

    Calculates simple moving averages over different periods and their slopes
    (rate of change) to capture trend direction and momentum.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least a 'close' column with price data.
    sma_fast : int, optional
        Window size for fast-moving average. Defaults to 5.
    sma_mid : int, optional
        Window size for medium-moving average. Defaults to 20.
    sma_slow : int, optional
        Window size for slow-moving average. Defaults to 50.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with the following columns added:
        - sma_{sma_fast}: Fast simple moving average
        - sma_{sma_mid}: Medium simple moving average
        - sma_{sma_slow}: Slow simple moving average
        - sma_{sma_fast}_slope: Day-to-day change in fast SMA
        - sma_{sma_mid}_slope: Day-to-day change in medium SMA
        - sma_{sma_slow}_slope: Day-to-day change in slow SMA
        - sma_{sma_mid}_distance: Relative distance of close price from medium SMA

    Notes
    -----
    The function modifies the input DataFrame in-place and also returns it.
    NaN values will appear for the first N periods where N is the window size
    for each respective moving average.
    """
    # SMA calculations
    df[f"sma_{sma_fast}"] = df["close"].rolling(window=sma_fast).mean()
    df[f"sma_{sma_mid}"] = df["close"].rolling(window=sma_mid).mean()
    df[f"sma_{sma_slow}"] = df["close"].rolling(window=sma_slow).mean()

    # Slope calculations
    df[f"sma_{sma_fast}_slope"] = df[f"sma_{sma_fast}"].diff()
    df[f"sma_{sma_mid}_slope"] = df[f"sma_{sma_mid}"].diff()
    df[f"sma_{sma_slow}_slope"] = df[f"sma_{sma_slow}"].diff()

    # Compute distance from SMA(mid)
    sma_mid_col = f"sma_{sma_mid}"
    df[f"{sma_mid_col}_distance"] = (df["close"] - df[sma_mid_col]) / df[sma_mid_col]

    return df
