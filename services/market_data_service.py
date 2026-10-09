from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from integrations.coinbase.client import CoinbaseClient
from integrations.coinbase.exceptions import CoinbaseError
from integrations.coinbase.models import Granularity
from services.indicators import rci, sma

SMA_SHORT_PERIOD = 50
SMA_LONG_PERIOD = 200

# Error codes produced by this layer (Coinbase errors keep their own `code`).
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
CALCULATION_ERROR = "CALCULATION_ERROR"
INVALID_INPUT = "INVALID_INPUT"


@dataclass(frozen=True)
class AssetMarketData:
    """Successful result for one asset.

    `timestamp` is the start (UTC) of the last closed candle and `close` its close.
    An indicator is None when there is not enough history to compute it.
    `candles_used` is how many closed candles were available for the calculation.
    """

    symbol: str
    timestamp: datetime
    close: float
    sma_50: float | None
    sma_200: float | None
    rci: float | None
    candles_used: int


@dataclass(frozen=True)
class AssetError:
    """Failure for one asset. `code` is a stable, HTTP-agnostic identifier."""

    symbol: str
    code: str
    message: str


AssetResult = AssetMarketData | AssetError


class MarketDataService:
    """Turns Coinbase candles into per-asset market data and indicators.

    It knows nothing about LangChain, JSON or text for the LLM. Assets are processed
    sequentially and failures are isolated: one failing asset never affects the rest.

    The semantics are fixed regardless of `candle_count`, which is only the amount of
    history requested:

        close  = close of the last CLOSED candle
        sma_50 = mean of the last 50 closes
        sma_200 = mean of the last 200 closes
        rci    = RCI(rci_period) of the last `rci_period` closes
    """

    def __init__(
        self,
        client: CoinbaseClient,
        *,
        granularity: Granularity = Granularity.ONE_DAY,
        rci_period: int = 9,
        candle_count: int = 250,
    ):
        if isinstance(rci_period, bool) or not isinstance(rci_period, int) or rci_period < 2:
            raise ValueError("rci_period must be an integer >= 2")
        if isinstance(candle_count, bool) or not isinstance(candle_count, int) or candle_count < 1:
            raise ValueError("candle_count must be an integer >= 1")
        self._client = client
        self._granularity = granularity
        self._rci_period = rci_period
        self._candle_count = candle_count

    def get_snapshots(self, product_ids: Sequence[str], as_of: datetime | None = None) -> list[AssetResult]:
        """Return one result per product, in the same order as `product_ids`."""
        return [self.get_snapshot(product_id, as_of) for product_id in product_ids]

    def get_snapshot(self, product_id: str,  as_of: datetime | None = None) -> AssetResult:
        """Fetch candles and compute the indicators for a single product."""
        try:
            candles = self._client.get_recent_candles(
                product_id, self._granularity, self._candle_count, reference_time=as_of
            )
        except CoinbaseError as exc:
            return AssetError(product_id, exc.code, str(exc))
        except ValueError as exc:  # invalid argument rejected by the client
            return AssetError(product_id, INVALID_INPUT, str(exc))

        if not candles:
            return AssetError(
                product_id,
                INSUFFICIENT_DATA,
                f"No closed candles available for {product_id}",
            )

        # The client already returns oldest-first; sort anyway because the indicators
        # depend entirely on chronological order.
        candles = sorted(candles, key=lambda c: c.timestamp)
        closes = [c.close for c in candles]
        last = candles[-1]

        try:
            return AssetMarketData(
                symbol=product_id,
                timestamp=last.timestamp,
                close=last.close,
                sma_50=sma(closes, SMA_SHORT_PERIOD),
                sma_200=sma(closes, SMA_LONG_PERIOD),
                rci=rci(closes, self._rci_period),
                candles_used=len(candles),
            )
        except Exception as exc:  # isolate calculation failures to this asset
            return AssetError(
                product_id,
                CALCULATION_ERROR,
                f"Indicator calculation failed for {product_id} ({type(exc).__name__})",
            )

