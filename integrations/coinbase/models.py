from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Granularity(Enum):
    """Candle size. Values are the identifiers Coinbase expects in the request."""

    ONE_MINUTE = "ONE_MINUTE"
    FIVE_MINUTE = "FIVE_MINUTE"
    FIFTEEN_MINUTE = "FIFTEEN_MINUTE"
    THIRTY_MINUTE = "THIRTY_MINUTE"
    ONE_HOUR = "ONE_HOUR"
    TWO_HOUR = "TWO_HOUR"
    FOUR_HOUR = "FOUR_HOUR"
    SIX_HOUR = "SIX_HOUR"
    ONE_DAY = "ONE_DAY"

    @property
    def seconds(self) -> int:
        """Length of one candle bucket in seconds."""
        return _GRANULARITY_SECONDS[self]


_GRANULARITY_SECONDS = {
    Granularity.ONE_MINUTE: 60,
    Granularity.FIVE_MINUTE: 300,
    Granularity.FIFTEEN_MINUTE: 900,
    Granularity.THIRTY_MINUTE: 1800,
    Granularity.ONE_HOUR: 3600,
    Granularity.TWO_HOUR: 7200,
    Granularity.FOUR_HOUR: 14400,
    Granularity.SIX_HOUR: 21600,
    Granularity.ONE_DAY: 86400,
}


@dataclass(frozen=True)
class Candle:
    """One OHLCV bucket. `timestamp` is the bucket start, timezone-aware UTC."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

