import yfinance as yf
# tools/news.py
from ddgs import DDGS
from langchain.tools import tool


@tool
def get_news(query: str, max_items: int = 5) -> str:
    """Get recent news headlines about a financial asset.

    Args:
        query: Search terms for the asset, e.g. "S&P 500", "gold price", "Bitcoin"
        max_items: Max number of headlines to return
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(query=query, max_results=max_items, timelimit="w"))
    except Exception as e:
        return f"News search failed: {e}"

    if not results:
        return f"No recent news found for {query}"

    lines = []
    for r in results:
        title = r.get("title", "Untitled")
        date = r.get("date", "unknown date")
        source = r.get("source", "unknown source")
        lines.append(f"- [{date}] {title} ({source})")

    return "\n".join(lines)

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