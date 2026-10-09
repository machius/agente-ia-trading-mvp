import math
import time
from collections.abc import Callable
from datetime import datetime, timezone
from urllib.parse import quote

import httpx

from integrations.coinbase.config import CoinbaseConfig
from integrations.coinbase.exceptions import (
    CoinbaseAuthError,
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

_REQUIRED_CANDLE_FIELDS = ("start", "open", "high", "low", "close", "volume")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CoinbaseClient:
    """HTTP client for Coinbase's public market-data API.

    It only talks to Coinbase and returns internal models (`Candle`). It knows
    nothing about indicators, LangChain or the agent.
    """

    def __init__(
        self,
        *,
        http_client: httpx.Client | None = None,
        config: CoinbaseConfig | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], datetime] = _utc_now,
    ):
        """
        Args:
            http_client: Reusable client. Inject one built on `httpx.MockTransport`
                in tests. If omitted, the client creates (and owns) its own.
            config: Non-secret settings; defaults to `CoinbaseConfig()`.
            sleep: Used for retry backoff. Inject a fake in tests.
            clock: Returns the current UTC time. Inject a fixed one in tests.
        """
        self._config = config or CoinbaseConfig()
        self._owns_http_client = http_client is None
        self._http = http_client or httpx.Client()
        self._sleep = sleep
        self._clock = clock

    def close(self) -> None:
        """Close the HTTP client only if this instance created it."""
        if self._owns_http_client:
            self._http.close()

    def __enter__(self) -> "CoinbaseClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def get_candles(
        self,
        product_id: str,
        granularity: Granularity,
        start: datetime,
        end: datetime,
    ) -> list[Candle]:
        """Return candles whose bucket start lies in [start, end], oldest first.

        Ranges wider than Coinbase's per-request limit are split transparently.
        Duplicates (by timestamp) are removed. An empty list is a valid result;
        deciding whether it is enough data is the caller's job.
        """
        self._validate_product_id(product_id)
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("start and end must be timezone-aware datetimes")
        start_ts, end_ts = int(start.timestamp()), int(end.timestamp())
        if start_ts > end_ts:
            raise ValueError("start must not be after end")

        # `end` is inclusive on Coinbase's side, so each window stops 1s before the
        # next one starts; otherwise a full window would hold 351 buckets and the
        # oldest one would be silently dropped.
        span = self._config.max_candles_per_request * granularity.seconds
        by_timestamp: dict[int, Candle] = {}
        window_start = start_ts
        while window_start <= end_ts:
            window_end = min(window_start + span - 1, end_ts)
            for candle in self._fetch_window(
                product_id, granularity, window_start, window_end
            ):
                ts = int(candle.timestamp.timestamp())
                if start_ts <= ts <= end_ts:
                    by_timestamp[ts] = candle
            window_start += span

        return [by_timestamp[ts] for ts in sorted(by_timestamp)]

    def get_recent_candles(
        self,
        product_id: str,
        granularity: Granularity,
        count: int,
    ) -> list[Candle]:
        """Return up to `count` of the most recent CLOSED candles, oldest first.

        The candle currently forming is excluded: the window ends right before the
        start of the current bucket. Buckets are aligned to the Unix epoch (for
        ONE_DAY this is 00:00 UTC, verified against the live API).
        """
        if count < 1:
            raise ValueError("count must be at least 1")

        bucket = granularity.seconds
        now_ts = int(self._clock().timestamp())
        current_bucket_start = now_ts - now_ts % bucket

        candles = self.get_candles(
            product_id,
            granularity,
            start=datetime.fromtimestamp(
                current_bucket_start - count * bucket, tz=timezone.utc
            ),
            end=datetime.fromtimestamp(current_bucket_start - 1, tz=timezone.utc),
        )
        return candles[-count:]

    # ------------------------------------------------------------------ internals

    @staticmethod
    def _validate_product_id(product_id: str) -> None:
        if not isinstance(product_id, str) or not product_id.strip():
            raise ValueError("product_id must be a non-empty string")

    def _fetch_window(
        self,
        product_id: str,
        granularity: Granularity,
        start_ts: int,
        end_ts: int,
    ) -> list[Candle]:
        path = self._config.candles_path.format(
            product_id=quote(product_id, safe="")
        )
        payload = self._request_json(
            f"{self._config.base_url}{path}",
            params={
                "start": str(start_ts),
                "end": str(end_ts),
                "granularity": granularity.value,
                "limit": str(self._config.max_candles_per_request),
            },
            product_id=product_id,
        )
        return self._parse_candles(payload)

    def _request_json(self, url: str, *, params: dict, product_id: str) -> object:
        """GET with retries on 429, 5xx, timeout and connection errors."""
        attempt = 0
        while True:
            retry_after: float | None = None
            try:
                response = self._http.get(
                    url,
                    params=params,
                    headers={"Accept": "application/json"},
                    timeout=self._config.timeout_seconds,
                )
            except httpx.TimeoutException:
                error: CoinbaseError = CoinbaseTimeoutError(
                    f"Request for {product_id} timed out"
                )
            except httpx.HTTPError as exc:
                error = CoinbaseConnectionError(
                    f"Could not reach Coinbase for {product_id}: "
                    f"{type(exc).__name__}"
                )
            else:
                status = response.status_code
                if 200 <= status < 300:
                    try:
                        return response.json()
                    except ValueError:
                        raise CoinbaseInvalidResponseError(
                            f"Coinbase returned invalid JSON for {product_id}",
                            status_code=status,
                        ) from None
                if status == 429:
                    retry_after = self._parse_retry_after(response)
                error = self._error_for_status(status, product_id, retry_after)

            if not error.retryable or attempt >= self._config.max_retries:
                raise error
            self._sleep(self._retry_delay(attempt, retry_after))
            attempt += 1

    @staticmethod
    def _error_for_status(
        status: int, product_id: str, retry_after: float | None
    ) -> CoinbaseError:
        if status == 404:
            # LIMITATION: only 404 is mapped to "product not found". Observed against
            # the live API, an unknown product answers 400 with an empty body, the
            # same status (and body) as any other malformed request, so Coinbase gives
            # no way to tell them apart. Such cases surface as CoinbaseBadRequestError
            # (BAD_REQUEST). We deliberately do not spend an extra request on a
            # product lookup just to disambiguate.
            return CoinbaseProductNotFoundError(
                f"Product {product_id} was not found on Coinbase",
                status_code=status,
            )
        if status in (401, 403):
            return CoinbaseAuthError(
                f"Coinbase refused the request for {product_id}", status_code=status
            )
        if status == 429:
            return CoinbaseRateLimitError(
                "Coinbase rate limit exceeded",
                status_code=status,
                retry_after=retry_after,
            )
        if status >= 500:
            return CoinbaseServerError(
                f"Coinbase server error ({status})", status_code=status
            )
        if 400 <= status < 500:
            return CoinbaseBadRequestError(
                f"Coinbase rejected the request for {product_id} ({status})",
                status_code=status,
            )
        return CoinbaseInvalidResponseError(
            f"Unexpected HTTP status {status} from Coinbase", status_code=status
        )

    @staticmethod
    def _parse_retry_after(response: httpx.Response) -> float | None:
        raw = response.headers.get("Retry-After")
        if raw is None:
            return None
        try:
            value = float(raw)
        except ValueError:
            return None
        return value if math.isfinite(value) and value >= 0 else None

    def _retry_delay(self, attempt: int, retry_after: float | None) -> float:
        delay = (
            retry_after
            if retry_after is not None
            else self._config.backoff_base_seconds * (2**attempt)
        )
        return min(delay, self._config.max_delay_seconds)

    @staticmethod
    def _parse_candles(payload: object) -> list[Candle]:
        if not isinstance(payload, dict) or not isinstance(
            payload.get("candles"), list
        ):
            raise CoinbaseInvalidResponseError(
                "Coinbase response does not contain a 'candles' list"
            )

        candles = []
        for raw in payload["candles"]:
            try:
                if not isinstance(raw, dict):
                    raise TypeError("candle is not an object")
                missing = [f for f in _REQUIRED_CANDLE_FIELDS if f not in raw]
                if missing:
                    raise KeyError(", ".join(missing))
                values = [float(raw[f]) for f in _REQUIRED_CANDLE_FIELDS[1:]]
                if not all(math.isfinite(v) for v in values):
                    raise ValueError("non-finite number")
                timestamp = datetime.fromtimestamp(int(raw["start"]), tz=timezone.utc)
            except (KeyError, TypeError, ValueError, OverflowError, OSError) as exc:
                raise CoinbaseInvalidResponseError(
                    f"Malformed candle in Coinbase response ({type(exc).__name__})"
                ) from None
            open_, high, low, close, volume = values
            candles.append(Candle(timestamp, open_, high, low, close, volume))
        return candles

