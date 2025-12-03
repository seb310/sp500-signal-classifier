"""Feature engineering module for volatility-based features.

This module provides functions for calculating volatility indicators including
rolling standard deviation and Average True Range (ATR) to measure price
variability and market turbulence.
"""

import numpy as np
import pandas as pd


def add_volatility(df: pd.DataFrame, vol_window: int = 10) -> pd.DataFrame:
    """
    Add volatility features to the DataFrame.

    Calculates rolling standard deviation and Average True Range (ATR)
    to capture market volatility and price variability.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least ``close``, ``high``, and ``low``
        columns with price data.
    vol_window : int, optional
        Window size for rolling standard deviation calculation. Defaults to
        10.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with the following columns added:

        - ``volatility_{vol_window}``: Rolling standard deviation of close
          prices.
        - ``tr``: True Range (max of high-low, high-prev_close,
          prev_close-low).
        - ``atr_14``: 14-period Average True Range.
        - ``atr_14_norm``: ATR normalized by close price (ATR as fraction of
          price).

    Notes
    -----
    The function modifies the input DataFrame in-place and also returns it.
    Any infinite values resulting from calculations are replaced with NaN.
    The ATR window is fixed at 14 periods, following traditional technical
    analysis.

    True Range measures the largest of:

    - Current high minus current low
    - Absolute value of current high minus previous close
    - Absolute value of current low minus previous close
    """
    # Compute rolling standard deviation as a volatility measure
    df[f"volatility_{vol_window}"] = df["close"].rolling(window=vol_window).std()

    # Compute ATR(14) volatility
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = (high - low).to_frame(name="hl")
    tr["hc"] = (high - prev_close).abs()
    tr["lc"] = (low - prev_close).abs()

    df["tr"] = tr.max(axis=1)
    df["atr_14"] = df["tr"].rolling(window=14).mean()
    df["atr_14_norm"] = df["atr_14"] / close

    return df.replace([np.inf, -np.inf], np.nan)
