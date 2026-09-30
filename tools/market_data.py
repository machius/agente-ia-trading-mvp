# tools/market_data.py
import yfinance as yf
from langchain.tools import tool


@tool
def get_market_data(ticker: str, period: str = "6mo") -> str:
    """Get historical OHLCV summary for a ticker.

    Args:
        ticker: Yahoo Finance symbol (e.g. "^GSPC" for S&P 500, "GC=F" for gold)
        period: History window, e.g. "1mo", "6mo", "1y"
    """
    data = yf.Ticker(ticker).history(period=period)
    if data.empty:
        return f"No data found for {ticker}"

    last = data.iloc[-1]
    change_pct = (last["Close"] / data["Close"].iloc[0] - 1) * 100

    return (
        f"{ticker} last close: {last['Close']:.2f}, "
        f"period high: {data['High'].max():.2f}, "
        f"period low: {data['Low'].min():.2f}, "
        f"period change: {change_pct:.2f}%"
    )