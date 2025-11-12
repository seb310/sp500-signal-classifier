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
    df = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )

    # --- flatten MultiIndex columns if present (yfinance can return ('Price','Ticker')) --- #
    if isinstance(df.columns, pd.MultiIndex):
        # Try to select the given ticker from the last level (common yfinance shape)
        # yfinance often lowercases the ticker in that level (e.g., 'spy')
        sym_lower = symbol.lower()
        try:
            df = df.xs(sym_lower, axis=1, level=-1)  # select the 'Ticker' level
        except (KeyError, TypeError):
            # Fallback: just drop the last level and keep the price names
            df.columns = df.columns.get_level_values(0)

    # normalize columns, keep only the standard OHLCV set if present
    df.columns = [str(c).strip().lower() for c in df.columns]
    wanted = ["open", "high", "low", "close", "adj close", "volume"]
    cols = [c for c in wanted if c in df.columns]
    df = df[cols]

    # tidy index
    df = df.sort_index()
    df.index = pd.to_datetime(df.index, utc=False)
    df.index.name = "date"

    return df


def cache_data(df: pd.DataFrame, symbol: str, cache_dir: str = "data/raw") -> None:
    """
    Placeholder for caching fetched data locally.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing OHLCV data to cache.
    symbol : str
        Ticker symbol corresponding to the data.
    cache_dir : str, optional
        Directory path to store cached data.
    """
    pass
