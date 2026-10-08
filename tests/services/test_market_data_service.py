"""Tests for MarketDataService using a fake client. No network access."""

import math
from datetime import datetime, timezone

import pytest

from integrations.coinbase.exceptions import (
    CoinbaseBadRequestError,
    CoinbaseConnectionError,
    CoinbaseInvalidResponseError,
    CoinbaseProductNotFoundError,
    CoinbaseRateLimitError,
    CoinbaseServerError,
    CoinbaseTimeoutError,
)
from integrations.coinbase.models import Candle, Granularity
from services.market_data_service import (
    AssetError,
    AssetMarketData,
    MarketDataService,
)

BASE = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp())
DAY = 86400


def make_candles(closes):
    """One daily candle per close, oldest first, starting 2026-01-01 UTC."""
    return [
        Candle(
            timestamp=datetime.fromtimestamp(BASE + i * DAY, tz=timezone.utc),
            open=c,
            high=c,
            low=c,
            close=float(c),
            volume=1.0,
        )
        for i, c in enumerate(closes)
    ]


def rising(n):
    """Closes 1, 2, ..., n."""
    return make_candles(range(1, n + 1))


class FakeClient:
    """Stands in for CoinbaseClient. Values are candle lists or exceptions."""

    def __init__(self, data):
        self.data = data
        self.calls = []

    def get_recent_candles(self, product_id, granularity, count):
        self.calls.append((product_id, granularity, count))
        result = self.data[product_id]
        if isinstance(result, Exception):
            raise result
        return result[-count:]  # like the real client: at most `count`, latest ones


def service_for(data, **kwargs):
    client = FakeClient(data)
    return MarketDataService(client, **kwargs), client


# ---------------------------------------------------------------- success


def test_four_assets_succeed_in_input_order():
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"]
    service, _ = service_for({s: rising(250) for s in symbols})

    results = service.get_snapshots(symbols)

    assert [r.symbol for r in results] == symbols
    assert all(isinstance(r, AssetMarketData) for r in results)


def test_close_timestamp_and_candles_used_come_from_the_last_closed_candle():
    service, _ = service_for({"BTC-USD": rising(250)})

    result = service.get_snapshot("BTC-USD")

    assert result.close == 250.0
    assert result.timestamp == datetime.fromtimestamp(
        BASE + 249 * DAY, tz=timezone.utc
    )
    assert result.timestamp.tzinfo is not None
    assert result.candles_used == 250


def test_sma50_sma200_and_rci_values_with_250_candles():
    service, _ = service_for({"BTC-USD": rising(250)})

    result = service.get_snapshot("BTC-USD")

    assert result.sma_50 == (201 + 250) / 2  # mean of the last 50 closes
    assert result.sma_200 == (51 + 250) / 2  # mean of the last 200 closes
    assert result.rci == 100.0  # strictly increasing


def test_the_250_candle_margin_does_not_change_the_indicator_definitions():
    # Same last 200 closes, different amount of earlier history.
    last_200 = list(range(1000, 1200))
    short, _ = service_for({"A": make_candles(last_200)})
    long, _ = service_for({"A": make_candles(list(range(50)) + last_200)})

    a, b = short.get_snapshot("A"), long.get_snapshot("A")

    assert a.sma_50 == b.sma_50
    assert a.sma_200 == b.sma_200
    assert a.rci == b.rci


def test_exactly_200_candles_enable_sma200():
    service, _ = service_for({"A": rising(200)})

    result = service.get_snapshot("A")

    assert result.sma_200 == 100.5
    assert result.sma_50 == 175.5


def test_between_50_and_199_candles_only_sma200_is_none():
    service, _ = service_for({"A": rising(120)})

    result = service.get_snapshot("A")

    assert isinstance(result, AssetMarketData)
    assert result.close == 120.0
    assert result.sma_50 == (71 + 120) / 2
    assert result.sma_200 is None
    assert result.rci == 100.0
    assert result.candles_used == 120


def test_exactly_50_candles_enable_sma50():
    service, _ = service_for({"A": rising(50)})

    result = service.get_snapshot("A")

    assert result.sma_50 == 25.5
    assert result.sma_200 is None


def test_fewer_than_50_candles_keep_close_and_rci():
    service, _ = service_for({"A": rising(30)})

    result = service.get_snapshot("A")

    assert isinstance(result, AssetMarketData)
    assert result.close == 30.0
    assert result.sma_50 is None
    assert result.sma_200 is None
    assert result.rci == 100.0


def test_fewer_candles_than_rci_period_leave_only_rci_none():
    service, _ = service_for({"A": rising(5)})

    result = service.get_snapshot("A")

    assert result.close == 5.0
    assert result.rci is None
    assert result.sma_50 is None


def test_single_candle_is_enough_for_close():
    service, _ = service_for({"A": rising(1)})

    result = service.get_snapshot("A")

    assert isinstance(result, AssetMarketData)
    assert result.close == 1.0
    assert result.candles_used == 1
    assert result.rci is None


def test_rci_value_on_non_monotonic_closes():
    closes = [10, 11, 12, 13, 3, 1, 2, 5, 4]  # last 9; ends with the [3,1,2,5,4] shape
    service, _ = service_for({"A": make_candles(closes)}, rci_period=5)

    result = service.get_snapshot("A")

    assert result.rci == 60.0


# ---------------------------------------------------------------- ordering


def test_candles_are_sorted_chronologically_before_calculating():
    candles = rising(250)
    shuffled = candles[::-1]  # newest first, as a misbehaving client could return
    service, _ = service_for({"A": shuffled})

    result = service.get_snapshot("A")

    assert result.close == 250.0
    assert result.timestamp == candles[-1].timestamp
    assert result.sma_50 == 225.5
    assert result.rci == 100.0  # would be -100 if the order were not fixed


# ---------------------------------------------------------------- configuration


def test_default_configuration_is_one_day_250_candles_rci_9():
    service, client = service_for({"A": rising(250)})

    result = service.get_snapshot("A")

    assert client.calls == [("A", Granularity.ONE_DAY, 250)]
    assert result.rci == 100.0


def test_custom_candle_count_is_requested_and_limits_history():
    service, client = service_for({"A": rising(250)}, candle_count=60)

    result = service.get_snapshot("A")

    assert client.calls == [("A", Granularity.ONE_DAY, 60)]
    assert result.candles_used == 60
    assert result.sma_50 == (201 + 250) / 2  # last 50 closes, unaffected by the margin
    assert result.sma_200 is None


def test_custom_rci_period_changes_the_window():
    # last 5 closes are [3, 1, 2, 5, 4] -> RCI(5) = 60; RCI(9) over the last 9 differs
    closes = [10, 20, 30, 40, 3, 1, 2, 5, 4]
    five, _ = service_for({"A": make_candles(closes)}, rci_period=5)
    nine, _ = service_for({"A": make_candles(closes)}, rci_period=9)

    assert five.get_snapshot("A").rci == 60.0
    assert nine.get_snapshot("A").rci != 60.0


def test_custom_granularity_is_passed_to_the_client():
    service, client = service_for({"A": rising(10)}, granularity=Granularity.ONE_HOUR)

    service.get_snapshot("A")

    assert client.calls[0][1] is Granularity.ONE_HOUR


@pytest.mark.parametrize("rci_period", [0, 1, -1, 2.5, True, None])
def test_invalid_rci_period_is_rejected_at_construction(rci_period):
    with pytest.raises(ValueError):
        MarketDataService(FakeClient({}), rci_period=rci_period)


@pytest.mark.parametrize("candle_count", [0, -5, 1.5, True, None])
def test_invalid_candle_count_is_rejected_at_construction(candle_count):
    with pytest.raises(ValueError):
        MarketDataService(FakeClient({}), candle_count=candle_count)


# ---------------------------------------------------------------- errors


def test_one_failing_asset_does_not_stop_the_others():
    service, client = service_for(
        {
            "BTC-USD": rising(250),
            "ETH-USD": rising(250),
            "SOL-USD": CoinbaseProductNotFoundError("Product SOL-USD was not found"),
            "XRP-USD": rising(250),
        }
    )

    results = service.get_snapshots(["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"])

    assert [type(r) for r in results] == [
        AssetMarketData,
        AssetMarketData,
        AssetError,
        AssetMarketData,
    ]
    assert results[2].symbol == "SOL-USD"
    assert results[2].code == "INVALID_PRODUCT"
    assert [c[0] for c in client.calls] == ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"]
    assert results[3].close == 250.0


@pytest.mark.parametrize(
    "error, code",
    [
        (CoinbaseProductNotFoundError("nf", status_code=404), "INVALID_PRODUCT"),
        (CoinbaseBadRequestError("bad", status_code=400), "BAD_REQUEST"),
        (CoinbaseRateLimitError("rl", status_code=429), "RATE_LIMITED"),
        (CoinbaseTimeoutError("t"), "TIMEOUT"),
        (CoinbaseConnectionError("c"), "UNAVAILABLE"),
        (CoinbaseServerError("s", status_code=503), "UNAVAILABLE"),
        (CoinbaseInvalidResponseError("i"), "INVALID_RESPONSE"),
    ],
)
def test_client_errors_are_translated_to_asset_errors_with_their_code(error, code):
    service, _ = service_for({"A": error})

    result = service.get_snapshot("A")

    assert isinstance(result, AssetError)
    assert result.symbol == "A"
    assert result.code == code
    assert result.message == str(error)


def test_zero_candles_is_insufficient_data_not_an_exception():
    service, _ = service_for({"A": []})

    result = service.get_snapshot("A")

    assert isinstance(result, AssetError)
    assert result.code == "INSUFFICIENT_DATA"
    assert "A" in result.message


def test_calculation_error_is_isolated_to_the_affected_asset():
    bad = make_candles(range(1, 251))
    bad[-1] = Candle(bad[-1].timestamp, 1, 1, 1, math.nan, 1)  # NaN close
    service, _ = service_for({"BAD": bad, "GOOD": rising(250)})

    bad_result, good_result = service.get_snapshots(["BAD", "GOOD"])

    assert isinstance(bad_result, AssetError)
    assert bad_result.code == "CALCULATION_ERROR"
    assert isinstance(good_result, AssetMarketData)
    assert good_result.close == 250.0


def test_invalid_argument_rejected_by_the_client_becomes_invalid_input():
    service, _ = service_for({"": ValueError("product_id must be a non-empty string")})

    result = service.get_snapshot("")

    assert isinstance(result, AssetError)
    assert result.code == "INVALID_INPUT"


def test_empty_asset_list_returns_empty_list_without_calling_the_client():
    service, client = service_for({})

    assert service.get_snapshots([]) == []
    assert client.calls == []


def test_results_are_immutable():
    service, _ = service_for({"A": rising(10)})

    result = service.get_snapshot("A")

    with pytest.raises(Exception):
        result.close = 0.0
