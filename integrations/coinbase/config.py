from dataclasses import dataclass


@dataclass(frozen=True)
class CoinbaseConfig:
    """Non-secret settings for the Coinbase public market-data API.

    The public candles endpoint needs no credentials, so nothing here is sensitive
    and no environment variables are required.
    """

    base_url: str = "https://api.coinbase.com"
    candles_path: str = "/api/v3/brokerage/market/products/{product_id}/candles"

    # Coinbase rejects (HTTP 400) any request spanning more than 350 buckets.
    max_candles_per_request: int = 350

    timeout_seconds: float = 10.0

    # Retries apply to 429, 5xx, timeouts and connection errors only.
    max_retries: int = 2
    backoff_base_seconds: float = 0.5
    max_delay_seconds: float = 30.0

