import json
import yfinance as yf
# tools/news.py
from ddgs import DDGS
from langchain.tools import tool

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

@tool
def get_market_data(assets: list[str]) -> str:
    """Get historical market data and indicators for EXACTLY 4 assets.

    Args:
        assets: List of exactly 4 asset symbols. Crypto assets must use BASE-USD format
                (e.g., "BTC-USD"). Non-crypto assets use standard ticker (e.g., "AAPL").
    """
    if not isinstance(assets, list):
        return json.dumps({"error": "assets must be a list of strings"})
    
    if len(assets) != 4:
        return json.dumps({"error": "Exactly 4 assets are required"})
    
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
                    snapshot = service.get_snapshot(asset)
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
        "indicators": {
            "sma_periods": [50, 200],
            "rci_period": 9
        },
        "assets": results
    })