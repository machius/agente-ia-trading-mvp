class CoinbaseError(Exception):
    """Base class for every error raised by the Coinbase integration.

    `code` is a stable, HTTP-agnostic identifier that upper layers can use
    without knowing anything about Coinbase status codes.
    `retryable` tells the client whether the failure is worth retrying.
    """

    code = "COINBASE_ERROR"
    retryable = False

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class CoinbaseProductNotFoundError(CoinbaseError):
    """Coinbase answered 404: the product does not exist.

    Limitation: Coinbase answers an unknown product with HTTP 400 and an empty body,
    indistinguishable from other bad requests, so that case is reported as
    CoinbaseBadRequestError instead. Only a 404 produces this error.
    """

    code = "INVALID_PRODUCT"


class CoinbaseAuthError(CoinbaseError):
    """Coinbase answered 401/403. Not expected on public endpoints."""

    code = "AUTH_ERROR"


class CoinbaseBadRequestError(CoinbaseError):
    """Coinbase answered 400 (or another 4xx that is not 401/403/404/429).

    This includes unknown products (see CoinbaseProductNotFoundError).
    """

    code = "BAD_REQUEST"


class CoinbaseRateLimitError(CoinbaseError):
    """Rate limit (HTTP 429) still present after all retries."""

    code = "RATE_LIMITED"
    retryable = True

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after: float | None = None,
    ):
        super().__init__(message, status_code=status_code)
        self.retry_after = retry_after


class CoinbaseTimeoutError(CoinbaseError):
    """The request timed out after all retries."""

    code = "TIMEOUT"
    retryable = True


class CoinbaseConnectionError(CoinbaseError):
    """Network-level failure after all retries."""

    code = "UNAVAILABLE"
    retryable = True


class CoinbaseServerError(CoinbaseError):
    """Coinbase answered 5xx after all retries."""

    code = "UNAVAILABLE"
    retryable = True


class CoinbaseInvalidResponseError(CoinbaseError):
    """The response was not valid JSON or did not have the expected shape."""

    code = "INVALID_RESPONSE"

