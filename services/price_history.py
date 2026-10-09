"""Price history for charting. Called directly by the UI — never by the agent."""

from datetime import datetime, timedelta, timezone
import pandas as pd
import yfinance as yf

from integrations.coinbase.client import CoinbaseClient
from integrations.coinbase.models import Granularity


def _is_crypto_format(symbol: str) -> bool:
    parts = symbol.split("-")
    return len(parts) == 2 and parts[0] and parts[1] == "USD"


def get_price_history(symbol: str, days: int = 90) -> pd.DataFrame:
    """Returns a DataFrame with columns ['date', 'close'], oldest first.
    Empty DataFrame on failure — caller decides how to show that.
    """
    try:
        if _is_crypto_format(symbol):
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=days)
            with CoinbaseClient() as client:
                candles = client.get_candles(symbol, Granularity.ONE_DAY, start, end)
            return pd.DataFrame(
                {"date": [c.timestamp for c in candles], "close": [c.close for c in candles]}
            )
        else:
            hist = yf.Ticker(symbol).history(period=f"{days}d")
            if hist.empty:
                return pd.DataFrame(columns=["date", "close"])
            return pd.DataFrame({"date": hist.index, "close": hist["Close"].values})
    except Exception:
        return pd.DataFrame(columns=["date", "close"])