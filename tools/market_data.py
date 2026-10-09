import json
import yfinance as yf
from ddgs import DDGS
from langchain.tools import tool
from datetime import date, datetime, timedelta, timezone

from integrations.coinbase.client import CoinbaseClient
from services.market_data_service import MarketDataService, AssetMarketData


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


def _is_valid_crypto_format(symbol: str) -> bool:
    parts = symbol.split("-")
    if len(parts) == 2 and parts[0] and parts[1] == "USD":
        return True
    return False

MIN_ASSETS = 1
MAX_ASSETS = 4
MAX_HISTORY_DAYS = 365

def _validate_as_of_date(as_of_date: str | None) -> tuple[datetime | None, dict | None]:
    """Returns (exclusive upper bound = midnight UTC of the day AFTER as_of_date,
    or None), and (error dict, or None).
    """
    if as_of_date is None:
        return None, None
    try:
        parsed = date.fromisoformat(as_of_date)
    except ValueError:
        return None, {"error": f"Invalid as_of_date format: {as_of_date!r}. Use YYYY-MM-DD."}

    today = datetime.now(timezone.utc).date()
    if parsed > today:
        return None, {"error": f"as_of_date cannot be in the future: {as_of_date}"}
    if (today - parsed).days > MAX_HISTORY_DAYS:
        return None, {"error": f"as_of_date cannot be more than {MAX_HISTORY_DAYS} days in the past"}

    # Límite exclusivo: medianoche UTC del día SIGUIENTE, para que el candle
    # cerrado de as_of_date quede incluido como "el más reciente".
    next_day = datetime(parsed.year, parsed.month, parsed.day, tzinfo=timezone.utc) + timedelta(days=1)
    return next_day, None

@tool
def get_market_data(assets: list[str], as_of_date: str | None = None) -> str:
    """Get market data and indicators for 1 to 4 assets, for today or a past date.

    Args:
        assets: List of 1 to 4 asset symbols. Crypto assets must use BASE-USD format
                (e.g., "BTC-USD"). Non-crypto assets use standard ticker (e.g., "AAPL").
        as_of_date: Optional historical date in YYYY-MM-DD format. Omit for today's
                    data. Must be within the last 365 days and not in the future.
    """
    if not isinstance(assets, list):
        return json.dumps({"error": "assets must be a list of strings"})
    # if len(assets) != 4:
    #     return json.dumps({"error": "Exactly 4 assets are required"})

    if not (MIN_ASSETS <= len(assets) <= MAX_ASSETS):
        return json.dumps({"error": f"Between {MIN_ASSETS} and {MAX_ASSETS} assets are required"})

    as_of_dt, date_error = _validate_as_of_date(as_of_date)
    if date_error:
        return json.dumps(date_error)

    normalized_assets = []
    for a in assets:
        if not isinstance(a, str) or not a.strip():
            return json.dumps({"error": "All assets must be non-empty strings"})
        normalized = a.strip().upper()
        if normalized in normalized_assets:
            return json.dumps({"error": f"Duplicate asset found: {normalized}"})
        normalized_assets.append(normalized)

    results = []
    
    with CoinbaseClient() as coinbase_client:
        service = MarketDataService(coinbase_client)
        
        for asset in normalized_assets:
            # Decisión arquitectónica: Consideramos como "crypto" a cualquier símbolo que contenga un guion "-".
            # Esto nos permite capturar y rechazar casos inválidos (como 'BTC-EUR', '-USD', 'BTC-') antes 
            # de enviarlos a yfinance o Coinbase. Tickers non-crypto soportados (ej. '^GSPC', 'GC=F', 'AAPL')
            # no contienen guiones. Si en el futuro se requiere soportar acciones como 'BRK-B', esta lógica 
            # deberá evolucionar.
            if "-" in asset:
                if not _is_valid_crypto_format(asset):
                    results.append({
                        "symbol": asset,
                        "error": {
                            "code": "INVALID_FORMAT",
                            "message": f"Invalid crypto format: {asset}. Must be BASE-USD."
                        }
                    })
                    continue

                try:
                    snapshot = service.get_snapshot(asset, as_of=as_of_dt)
                except Exception as e:
                    results.append({
                        "symbol": asset,
                        "error": {
                            "code": "MARKET_DATA_ERROR",
                            "message": f"Unexpected error: {str(e)}"
                        }
                    })
                    continue

                if isinstance(snapshot, AssetMarketData):
                    results.append({
                        "symbol": snapshot.symbol,
                        "timestamp": snapshot.timestamp.isoformat(),
                        "close": snapshot.close,
                        "sma_50": snapshot.sma_50,
                        "sma_200": snapshot.sma_200,
                        "rci": snapshot.rci,
                        "candles_used": snapshot.candles_used
                    })
                else:
                    results.append({
                        "symbol": snapshot.symbol,
                        "error": {
                            "code": snapshot.code,
                            "message": snapshot.message
                        }
                    })
            else:
                try:
                    if as_of_dt:
                        start_window = as_of_dt - timedelta(days=400)
                        data = yf.Ticker(asset).history(start=start_window, end=as_of_dt)
                    else:
                        data = yf.Ticker(asset).history(period="1y")
                    if data.empty:
                        results.append({
                            "symbol": asset,
                            "error": {
                                "code": "NO_DATA",
                                "message": f"No data found for {asset}"
                            }
                        })
                    else:
                        last = data.iloc[-1]
                        closes = data["Close"].tolist()
                        sma_50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else None
                        sma_200 = sum(closes[-200:]) / 200 if len(closes) >= 200 else None
                        
                        timestamp_iso = data.index[-1].isoformat() if not data.empty else None
                        
                        results.append({
                            "symbol": asset,
                            "timestamp": timestamp_iso,
                            "close": float(last["Close"]),
                            "sma_50": float(sma_50) if sma_50 is not None else None,
                            "sma_200": float(sma_200) if sma_200 is not None else None,
                            "rci": None,  # Documented logic: RCI not calculated for non-crypto
                            "candles_used": len(closes)
                        })
                except Exception as e:
                    results.append({
                        "symbol": asset,
                        "error": {
                            "code": "YFINANCE_ERROR",
                            "message": str(e)
                        }
                    })

    return json.dumps({
        "timeframe": "ONE_DAY",
        "as_of_date": as_of_date or date.today().isoformat(),
        "indicators": {"sma_periods": [50, 200], "rci_period": 9},
        "assets": results,
    })