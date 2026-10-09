import json
from unittest.mock import MagicMock, patch
import pytest
from datetime import datetime, timezone

from tools.market_data import get_market_data, _is_valid_crypto_format
from services.market_data_service import AssetMarketData, AssetError

def test_crypto_format_validation():
    assert _is_valid_crypto_format("BTC-USD") is True
    assert _is_valid_crypto_format("ETH-USD") is True
    assert _is_valid_crypto_format("BTC-EUR") is False
    assert _is_valid_crypto_format("-USD") is False
    assert _is_valid_crypto_format("BTC-") is False
    assert _is_valid_crypto_format("BTC-USD-XXX") is False
    assert _is_valid_crypto_format("USD") is False

def test_invalid_input_count_and_types():
    res = json.loads(get_market_data.invoke({"assets": ["BTC-USD", "ETH-USD"]}))
    assert "Exactly 4 assets are required" in res["error"]

    res = json.loads(get_market_data.invoke({"assets": ["BTC-USD", "ETH-USD", "LTC-USD", "SOL-USD", "XRP-USD"]}))
    assert "Exactly 4 assets are required" in res["error"]

    res = json.loads(get_market_data.invoke({"assets": ["BTC-USD", "ETH-USD", "BTC-usd", "SOL-USD"]}))
    assert "Duplicate asset found: BTC-USD" in res["error"]

    res = json.loads(get_market_data.invoke({"assets": ["BTC-USD", "ETH-USD", "", "SOL-USD"]}))
    assert "All assets must be non-empty strings" in res["error"]
    
    # Space should be stripped
    res2 = json.loads(get_market_data.invoke({"assets": [" BTC-USD ", "BTC-USD", "LTC-USD", "SOL-USD"]}))
    assert "Duplicate asset found: BTC-USD" in res2["error"]

@patch("tools.market_data.CoinbaseClient")
@patch("tools.market_data.MarketDataService")
@patch("tools.market_data.yf.Ticker")
def test_get_market_data_routing_and_json_contract(mock_ticker, mock_service_class, mock_client_class):
    mock_service = mock_service_class.return_value
    
    def mock_get_snapshot(symbol):
        if symbol == "BTC-USD":
            return AssetMarketData(
                symbol=symbol,
                timestamp=datetime(2026, 10, 7, tzinfo=timezone.utc),
                close=100000.0,
                sma_50=99000.0,
                sma_200=95000.0,
                rci=72.5,
                candles_used=250
            )
        else:
            return AssetError(symbol=symbol, code="TEST_ERROR", message="crypto error")
    
    mock_service.get_snapshot.side_effect = mock_get_snapshot

    mock_hist = MagicMock()
    mock_hist.empty = False
    mock_hist.iloc = [MagicMock()]
    mock_hist.iloc[-1] = {"Close": 150.0}
    mock_hist["Close"].tolist.return_value = [100.0] * 50 + [150.0]
    
    mock_ts = MagicMock()
    mock_ts.isoformat.return_value = "2026-10-07T00:00:00+00:00"
    mock_hist.index = [MagicMock(), mock_ts]

    mock_ticker_instance = MagicMock()
    mock_ticker_instance.history.return_value = mock_hist
    mock_ticker.return_value = mock_ticker_instance

    res = get_market_data.invoke({"assets": ["BTC-USD", "ETH-USD", "AAPL", "BTC-EUR"]})
    data = json.loads(res)
    
    assert data["timeframe"] == "ONE_DAY"
    assert data["indicators"]["sma_periods"] == [50, 200]
    assert data["indicators"]["rci_period"] == 9
    assert len(data["assets"]) == 4
    
    btc = next(a for a in data["assets"] if a["symbol"] == "BTC-USD")
    assert "error" not in btc
    assert btc["close"] == 100000.0
    assert btc["sma_50"] == 99000.0
    assert btc["sma_200"] == 95000.0
    assert btc["rci"] >= -100 and btc["rci"] <= 100
    assert btc["rci"] == 72.5
    assert btc["candles_used"] == 250
    assert btc["timestamp"] == "2026-10-07T00:00:00+00:00"
    
    eth = next(a for a in data["assets"] if a["symbol"] == "ETH-USD")
    assert "error" in eth
    assert eth["error"]["code"] == "TEST_ERROR"
    assert eth["error"]["message"] == "crypto error"
    
    aapl = next(a for a in data["assets"] if a["symbol"] == "AAPL")
    assert "error" not in aapl
    assert aapl["close"] == 150.0
    assert aapl["sma_50"] == 101.0
    assert aapl["sma_200"] is None
    assert aapl["rci"] is None
    assert aapl["candles_used"] == 51
    assert aapl["timestamp"] == "2026-10-07T00:00:00+00:00"

    eur = next(a for a in data["assets"] if a["symbol"] == "BTC-EUR")
    assert "error" in eur
    assert eur["error"]["code"] == "INVALID_FORMAT"

@patch("tools.market_data.yf.Ticker")
def test_yfinance_no_data(mock_ticker):
    mock_hist = MagicMock()
    mock_hist.empty = True
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.history.return_value = mock_hist
    mock_ticker.return_value = mock_ticker_instance

    res = get_market_data.invoke({"assets": ["AAPL", "MSFT", "GOOG", "TSLA"]})
    data = json.loads(res)
    
    for a in data["assets"]:
        assert "error" in a
        assert a["error"]["code"] == "NO_DATA"

@patch("tools.market_data.yf.Ticker")
def test_yfinance_exception(mock_ticker):
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.history.side_effect = Exception("network error")
    mock_ticker.return_value = mock_ticker_instance

    res = get_market_data.invoke({"assets": ["AAPL", "MSFT", "GOOG", "TSLA"]})
    data = json.loads(res)
    
    for a in data["assets"]:
        assert "error" in a
        assert a["error"]["code"] == "YFINANCE_ERROR"

@patch("tools.market_data.CoinbaseClient")
@patch("tools.market_data.MarketDataService")
@patch("tools.market_data.yf.Ticker")
def test_market_data_unexpected_error(mock_ticker, mock_service_class, mock_client_class):
    mock_service = mock_service_class.return_value
    
    def mock_get_snapshot(symbol):
        if symbol == "BTC-USD":
            raise Exception("simulated crash")
        elif symbol == "ETH-USD":
            return AssetMarketData(
                symbol=symbol,
                timestamp=datetime(2026, 10, 7, tzinfo=timezone.utc),
                close=2000.0, sma_50=1900.0, sma_200=1800.0, rci=50.0, candles_used=250
            )
        return AssetError(symbol=symbol, code="ERR", message="err")
    
    mock_service.get_snapshot.side_effect = mock_get_snapshot
    
    mock_hist = MagicMock()
    mock_hist.empty = True
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.history.return_value = mock_hist
    mock_ticker.return_value = mock_ticker_instance

    res = get_market_data.invoke({"assets": ["BTC-USD", "ETH-USD", "AAPL", "MSFT"]})
    data = json.loads(res)
    
    btc = next(a for a in data["assets"] if a["symbol"] == "BTC-USD")
    assert "error" in btc
    assert btc["error"]["code"] == "MARKET_DATA_ERROR"
    assert "simulated crash" in btc["error"]["message"]
    
    eth = next(a for a in data["assets"] if a["symbol"] == "ETH-USD")
    assert "error" not in eth
    assert eth["close"] == 2000.0

