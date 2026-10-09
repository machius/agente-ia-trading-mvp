"""Unit tests for CoinbaseClient. No network access: every request goes through
httpx.MockTransport, backoff sleeps are recorded instead of executed, and the clock
is fixed."""

from datetime import datetime, timezone

import httpx
import pytest

from integrations.coinbase.client import CoinbaseClient
from integrations.coinbase.config import CoinbaseConfig
from integrations.coinbase.exceptions import (
    CoinbaseBadRequestError,
    CoinbaseConnectionError,
    CoinbaseError,
    CoinbaseInvalidResponseError,
    CoinbaseProductNotFoundError,
    CoinbaseRateLimitError,
    CoinbaseServerError,
    CoinbaseTimeoutError,
)
from integrations.coinbase.models import Candle, Granularity

DAY = 86400
CANDLES_PATH = "/api/v3/brokerage/market/products/BTC-USD/candles"

# 2026-10-07 15:30 UTC -> the daily candle in formation starts at 2026-10-07 00:00 UTC.
FIXED_NOW = datetime(2026, 10, 7, 15, 30, tzinfo=timezone.utc)
CURRENT_DAY_START = int(datetime(2026, 10, 7, tzinfo=timezone.utc).timestamp())


def raw_candle(ts, close="100.5", **overrides):
    candle = {
        "start": str(ts),
        "low": "90.0",
        "high": "110.0",
        "open": "95.0",
        "close": close,
        "volume": "12.5",
    }
    candle.update(overrides)
    return candle


def json_response(candles, status=200):
    return httpx.Response(status, json={"candles": candles})


class Harness:
    """Builds a client on a MockTransport and records requests and sleeps."""

    def __init__(self, handler, **config):
        self.requests: list[httpx.Request] = []
        self.sleeps: list[float] = []

        def recording_handler(request):
            self.requests.append(request)
            return handler(request)

        self.http = httpx.Client(transport=httpx.MockTransport(recording_handler))
        self.client = CoinbaseClient(
            http_client=self.http,
            config=CoinbaseConfig(**config),
            sleep=self.sleeps.append,
            clock=lambda: FIXED_NOW,
        )


def day(n):
    """Start (UTC) of the n-th day counted from 2026-01-01."""
    base = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp())
    return base + n * DAY


def utc(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def get_range(harness, first_day=0, last_day=4):
    return harness.client.get_candles(
        "BTC-USD", Granularity.ONE_DAY, utc(day(first_day)), utc(day(last_day))
    )


# ---------------------------------------------------------------- parsing


def test_response_is_converted_to_internal_candles():
    h = Harness(lambda r: json_response([raw_candle(day(0), close="101.25")]))

    candles = get_range(h, 0, 0)

    assert candles == [
        Candle(
            timestamp=utc(day(0)),
            open=95.0,
            high=110.0,
            low=90.0,
            close=101.25,
            volume=12.5,
        )
    ]
    assert isinstance(candles[0].close, float)


def test_unix_timestamp_becomes_timezone_aware_utc():
    h = Harness(lambda r: json_response([raw_candle(1791417600)]))

    candle = h.client.get_candles(
        "BTC-USD", Granularity.ONE_DAY, utc(1791417600), utc(1791417600)
    )[0]

    assert candle.timestamp == datetime(2026, 10, 8, 0, 0, tzinfo=timezone.utc)
    assert candle.timestamp.utcoffset().total_seconds() == 0


def test_candles_are_sorted_oldest_first_even_if_api_returns_newest_first():
    h = Harness(
        lambda r: json_response([raw_candle(day(i)) for i in (4, 2, 0, 3, 1)])
    )

    candles = get_range(h, 0, 4)

    assert [c.timestamp for c in candles] == [utc(day(i)) for i in range(5)]


def test_duplicate_timestamps_are_removed():
    h = Harness(
        lambda r: json_response(
            [raw_candle(day(0), close="1"), raw_candle(day(0), close="2"),
             raw_candle(day(1))]
        )
    )

    candles = get_range(h, 0, 1)

    assert [c.timestamp for c in candles] == [utc(day(0)), utc(day(1))]


def test_empty_response_returns_empty_list():
    h = Harness(lambda r: json_response([]))

    assert get_range(h, 0, 4) == []


def test_candles_outside_requested_range_are_discarded():
    h = Harness(lambda r: json_response([raw_candle(day(i)) for i in range(10)]))

    candles = get_range(h, 2, 4)

    assert [c.timestamp for c in candles] == [utc(day(i)) for i in (2, 3, 4)]


# ---------------------------------------------------------------- request


def test_request_targets_the_public_candles_endpoint():
    h = Harness(lambda r: json_response([]))

    get_range(h)

    request = h.requests[0]
    assert request.method == "GET"
    assert request.url.scheme == "https"
    assert request.url.host == "api.coinbase.com"
    assert request.url.path == CANDLES_PATH
    assert request.headers["Accept"] == "application/json"


def test_request_sends_start_end_granularity_and_limit():
    h = Harness(lambda r: json_response([]))

    get_range(h, 0, 4)

    params = dict(h.requests[0].url.params)
    assert params == {
        "start": str(day(0)),
        "end": str(day(4)),
        "granularity": "ONE_DAY",
        "limit": "350",
    }


def test_request_has_no_authorization_header():
    h = Harness(lambda r: json_response([]))

    get_range(h)

    assert "authorization" not in {k.lower() for k in h.requests[0].headers}


def test_product_id_is_url_encoded_in_the_path():
    h = Harness(lambda r: json_response([]))

    h.client.get_candles(
        "BTC/../x", Granularity.ONE_DAY, utc(day(0)), utc(day(1))
    )

    assert "/products/BTC%2F..%2Fx/candles" in str(h.requests[0].url)


# ---------------------------------------------------------------- HTTP errors


def test_404_raises_product_error_without_retry():
    h = Harness(lambda r: httpx.Response(404))

    with pytest.raises(CoinbaseProductNotFoundError) as info:
        get_range(h)

    assert info.value.code == "INVALID_PRODUCT"
    assert len(h.requests) == 1
    assert h.sleeps == []


def test_400_raises_bad_request_not_product_error_and_is_not_retried():
    # Coinbase answers 400 with an empty body for unknown products AND for any
    # other bad request, so 400 must not be reported as INVALID_PRODUCT.
    h = Harness(lambda r: httpx.Response(400))

    with pytest.raises(CoinbaseBadRequestError) as info:
        get_range(h)

    assert not isinstance(info.value, CoinbaseProductNotFoundError)
    assert info.value.code == "BAD_REQUEST"
    assert info.value.status_code == 400
    assert len(h.requests) == 1
    assert h.sleeps == []


def test_429_is_retried_and_then_raises_rate_limit_error():
    h = Harness(lambda r: httpx.Response(429), max_retries=2, backoff_base_seconds=0.5)

    with pytest.raises(CoinbaseRateLimitError) as info:
        get_range(h)

    assert info.value.code == "RATE_LIMITED"
    assert len(h.requests) == 3  # 1 try + 2 retries
    assert h.sleeps == [0.5, 1.0]  # exponential backoff, recorded not slept


def test_429_honours_retry_after_header_with_cap():
    h = Harness(
        lambda r: httpx.Response(429, headers={"Retry-After": "7"}),
        max_retries=1,
        max_delay_seconds=5.0,
    )

    with pytest.raises(CoinbaseRateLimitError) as info:
        get_range(h)

    assert info.value.retry_after == 7.0
    assert h.sleeps == [5.0]  # capped by max_delay_seconds


def test_429_then_success_recovers():
    responses = iter([httpx.Response(429), json_response([raw_candle(day(0))])])
    h = Harness(lambda r: next(responses))

    candles = get_range(h, 0, 0)

    assert len(candles) == 1
    assert len(h.requests) == 2
    assert len(h.sleeps) == 1


@pytest.mark.parametrize("status", [500, 502, 503])
def test_5xx_is_retried_and_then_raises_server_error(status):
    h = Harness(lambda r: httpx.Response(status), max_retries=2)

    with pytest.raises(CoinbaseServerError) as info:
        get_range(h)

    assert info.value.code == "UNAVAILABLE"
    assert len(h.requests) == 3
    assert len(h.sleeps) == 2


def test_other_4xx_is_not_retried():
    h = Harness(lambda r: httpx.Response(418))

    with pytest.raises(CoinbaseError) as info:
        get_range(h)

    assert info.value.code == "BAD_REQUEST"
    assert len(h.requests) == 1


def test_timeout_is_retried_and_then_raises_timeout_error():
    def handler(request):
        raise httpx.ReadTimeout("timed out", request=request)

    h = Harness(handler, max_retries=2)

    with pytest.raises(CoinbaseTimeoutError) as info:
        get_range(h)

    assert info.value.code == "TIMEOUT"
    assert len(h.requests) == 3
    assert len(h.sleeps) == 2


def test_connection_error_is_retried_and_then_raises_connection_error():
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    h = Harness(handler, max_retries=2)

    with pytest.raises(CoinbaseConnectionError) as info:
        get_range(h)

    assert info.value.code == "UNAVAILABLE"
    assert len(h.requests) == 3


def test_max_retries_zero_disables_retries():
    h = Harness(lambda r: httpx.Response(503), max_retries=0)

    with pytest.raises(CoinbaseServerError):
        get_range(h)

    assert len(h.requests) == 1
    assert h.sleeps == []


# ---------------------------------------------------------------- invalid responses


def test_invalid_json_raises_invalid_response_error():
    h = Harness(lambda r: httpx.Response(200, content=b"<html>not json</html>"))

    with pytest.raises(CoinbaseInvalidResponseError) as info:
        get_range(h)

    assert info.value.code == "INVALID_RESPONSE"


@pytest.mark.parametrize(
    "payload",
    [
        {"unexpected": []},
        {"candles": "nope"},
        ["not", "an", "object"],
        {"candles": ["not-an-object"]},
    ],
)
def test_unexpected_payload_shape_raises_invalid_response_error(payload):
    h = Harness(lambda r: httpx.Response(200, json=payload))

    with pytest.raises(CoinbaseInvalidResponseError):
        get_range(h)


@pytest.mark.parametrize("field", ["start", "open", "high", "low", "close", "volume"])
def test_missing_required_candle_field_raises_invalid_response_error(field):
    candle = raw_candle(day(0))
    del candle[field]
    h = Harness(lambda r: json_response([candle]))

    with pytest.raises(CoinbaseInvalidResponseError):
        get_range(h, 0, 0)


@pytest.mark.parametrize("bad", ["abc", None, "nan", "inf", ""])
def test_non_numeric_candle_value_raises_invalid_response_error(bad):
    h = Harness(lambda r: json_response([raw_candle(day(0), close=bad)]))

    with pytest.raises(CoinbaseInvalidResponseError):
        get_range(h, 0, 0)


# ---------------------------------------------------------------- pagination


def window_handler(request):
    """Fake Coinbase: one daily candle for every day inside the requested window."""
    start = int(request.url.params["start"])
    end = int(request.url.params["end"])
    first_aligned = -(-start // DAY) * DAY  # first day boundary >= start
    return json_response([raw_candle(ts) for ts in range(first_aligned, end + 1, DAY)])


def test_ranges_wider_than_350_candles_are_split_into_windows():
    h = Harness(window_handler)
    first, last = 0, 399  # 400 daily candles

    candles = get_range(h, first, last)

    windows = [
        (int(r.url.params["start"]), int(r.url.params["end"])) for r in h.requests
    ]
    assert windows == [
        (day(0), day(350) - 1),  # exactly 350 buckets, ends 1s before the next window
        (day(350), day(399)),
    ]
    assert len(candles) == 400
    assert [c.timestamp for c in candles] == [utc(day(i)) for i in range(400)]


def test_no_window_exceeds_350_buckets():
    h = Harness(window_handler)

    get_range(h, 0, 1000)

    for request in h.requests:
        start, end = int(request.url.params["start"]), int(request.url.params["end"])
        assert (end - start + 1) // DAY <= 350
    assert len(h.requests) == 3


def test_range_within_one_window_makes_a_single_request():
    h = Harness(window_handler)

    get_range(h, 0, 349)

    assert len(h.requests) == 1


def test_pagination_deduplicates_candles_repeated_across_windows():
    def handler(request):
        # Misbehaving API: always includes the boundary candle in every window.
        start = int(request.url.params["start"])
        return json_response([raw_candle(start), raw_candle(day(350))])

    h = Harness(handler)

    candles = get_range(h, 0, 400)

    timestamps = [c.timestamp for c in candles]
    assert len(timestamps) == len(set(timestamps))
    assert timestamps == sorted(timestamps)


def test_error_in_a_later_window_propagates():
    responses = iter([json_response([raw_candle(day(0))]), httpx.Response(500)])
    h = Harness(lambda r: next(responses), max_retries=0)

    with pytest.raises(CoinbaseServerError):
        get_range(h, 0, 400)


# ---------------------------------------------------------------- recent / closed only


def test_recent_candles_window_ends_before_the_candle_in_formation():
    h = Harness(lambda r: json_response([]))

    h.client.get_recent_candles("BTC-USD", Granularity.ONE_DAY, count=250)

    params = h.requests[0].url.params
    assert int(params["end"]) == CURRENT_DAY_START - 1
    assert int(params["start"]) == CURRENT_DAY_START - 250 * DAY
    assert params["granularity"] == "ONE_DAY"


def test_candle_in_formation_is_excluded_even_if_api_returns_it():
    h = Harness(
        lambda r: json_response(
            [raw_candle(CURRENT_DAY_START - 2 * DAY),
             raw_candle(CURRENT_DAY_START - DAY),
             raw_candle(CURRENT_DAY_START)]  # today, still forming
        )
    )

    candles = h.client.get_recent_candles("BTC-USD", Granularity.ONE_DAY, count=10)

    assert [int(c.timestamp.timestamp()) for c in candles] == [
        CURRENT_DAY_START - 2 * DAY,
        CURRENT_DAY_START - DAY,
    ]


def test_recent_candles_returns_only_the_latest_count_oldest_first():
    h = Harness(
        lambda r: json_response(
            [raw_candle(CURRENT_DAY_START - i * DAY) for i in range(1, 11)]
        )
    )

    candles = h.client.get_recent_candles("BTC-USD", Granularity.ONE_DAY, count=3)

    assert [int(c.timestamp.timestamp()) for c in candles] == [
        CURRENT_DAY_START - 3 * DAY,
        CURRENT_DAY_START - 2 * DAY,
        CURRENT_DAY_START - DAY,
    ]


def test_250_recent_candles_need_a_single_request():
    h = Harness(window_handler)

    candles = h.client.get_recent_candles("BTC-USD", Granularity.ONE_DAY, count=250)

    assert len(h.requests) == 1
    assert len(candles) == 250
    assert candles[-1].timestamp == utc(CURRENT_DAY_START - DAY)


def test_recent_candles_on_empty_response_returns_empty_list():
    h = Harness(lambda r: json_response([]))

    assert h.client.get_recent_candles("BTC-USD", Granularity.ONE_DAY, 250) == []


def test_recent_candles_uses_the_granularity_bucket_size():
    h = Harness(lambda r: json_response([]))

    h.client.get_recent_candles("BTC-USD", Granularity.ONE_HOUR, count=10)

    hour_start = int(FIXED_NOW.timestamp()) - int(FIXED_NOW.timestamp()) % 3600
    params = h.requests[0].url.params
    assert int(params["end"]) == hour_start - 1
    assert int(params["start"]) == hour_start - 10 * 3600


# ---------------------------------------------------------------- argument validation / lifecycle


def test_invalid_arguments_raise_value_error():
    h = Harness(lambda r: json_response([]))

    with pytest.raises(ValueError):
        h.client.get_candles("", Granularity.ONE_DAY, utc(day(0)), utc(day(1)))
    with pytest.raises(ValueError):
        h.client.get_candles(
            "BTC-USD", Granularity.ONE_DAY, datetime(2026, 1, 1), utc(day(1))
        )
    with pytest.raises(ValueError):
        h.client.get_candles("BTC-USD", Granularity.ONE_DAY, utc(day(2)), utc(day(1)))
    with pytest.raises(ValueError):
        h.client.get_recent_candles("BTC-USD", Granularity.ONE_DAY, count=0)
    assert h.requests == []


def test_injected_http_client_is_not_closed_by_the_client():
    h = Harness(lambda r: json_response([]))

    h.client.close()

    assert not h.http.is_closed


def test_every_granularity_has_a_positive_bucket_size():
    for granularity in Granularity:
        assert granularity.seconds > 0
    assert Granularity.ONE_DAY.seconds == DAY
