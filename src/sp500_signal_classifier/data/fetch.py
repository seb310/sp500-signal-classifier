import pandas as pd
import yfinance as yf


def fetch_ohlcv(symbol: str, start: str = "2005-01-01", end=None) -> pd.DataFrame:
    """
    Fetch OHLCV (Open, High, Low, Close, Volume) data for a given symbol.

    Downloads historical price and volume data from Yahoo Finance and returns
    a formatted DataFrame with lowercase column names.

    Parameters
    ----------
    symbol : str
        Ticker symbol for the asset (e.g., 'AAPL', '^GSPC').
    start : str, optional
        Start date in 'YYYY-MM-DD' format. Defaults to '2005-01-01'.
    end : str or None, optional
        End date in 'YYYY-MM-DD' format. If None, fetches up to the most
        recent available data. Defaults to None.

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex named 'date' and columns:
        ['open', 'high', 'low', 'close', 'adj close', 'volume'].
    """
    df = yf.download(symbol, start=start, end=end, auto_adjust=False)
    df = df.rename(columns=str.lower)[
        ["open", "high", "low", "close", "adj close", "volume"]
    ]
    df.index.name = "date"
    return df
