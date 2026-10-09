"""Pure, dependency-free technical indicators.

Every function takes closing prices in CHRONOLOGICAL order, `[oldest, ..., newest]`,
and works on the last `period` values. They return `None` when there is not enough
data (or the result is mathematically undefined) and raise `ValueError` for invalid
arguments or non-finite prices.
"""

import math
from collections.abc import Sequence


def _validate_period(period: int, minimum: int) -> None:
    if isinstance(period, bool) or not isinstance(period, int) or period < minimum:
        raise ValueError(f"period must be an integer >= {minimum}, got {period!r}")


def _last_window(closes: Sequence[float], period: int) -> list[float] | None:
    """Last `period` closes, or None if there are fewer than `period`."""
    if len(closes) < period:
        return None
    window = [float(c) for c in closes[-period:]]
    if not all(math.isfinite(v) for v in window):
        raise ValueError("closes must be finite numbers")
    return window


def sma(closes: Sequence[float], period: int) -> float | None:
    """Simple moving average: arithmetic mean of the last `period` closes.

    Args:
        closes: Closing prices, oldest first.
        period: Number of closes to average (>= 1).

    Returns:
        The mean, or None if fewer than `period` closes are available.
    """
    _validate_period(period, 1)
    window = _last_window(closes, period)
    if window is None:
        return None
    return math.fsum(window) / period


def _average_ranks(values: Sequence[float]) -> list[float]:
    """1-based ascending ranks; tied values share the mean of their positions."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2 + 1  # mean of 1-based positions i+1 .. j+1
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def rci(closes: Sequence[float], period: int) -> float | None:
    """Rank Correlation Index (RCI) of the last `period` closes. NOT the RSI.

    RCI is Spearman's rank correlation between time and price, scaled to
    [-100, +100]. Over a window of n = `period` closes:

        time rank   t_i: 1 (oldest) .. n (newest)
        price rank  p_i: 1 (lowest close) .. n (highest close)
        d_i = t_i - p_i

        RCI = (1 - 6 * sum(d_i^2) / (n * (n^2 - 1))) * 100        (no tied prices)

    +100 means every close is higher than the previous one, -100 that every close is
    lower, values near 0 mean no relation between time and price.

    With tied closes, tied prices get the average of their ranks and the correlation
    is the Pearson correlation of the two rank series (Spearman with ties).

    Args:
        closes: Closing prices, oldest first (`[oldest, ..., newest]`). Only the
            last `period` values are used.
        period: Window size n (>= 2; the formula is undefined for n = 1).

    Returns:
        The RCI in [-100, 100]; None if fewer than `period` closes are available, or
        if all closes in the window are equal (correlation undefined).
    """
    _validate_period(period, 2)
    window = _last_window(closes, period)
    if window is None:
        return None

    n = period
    time_ranks = [float(i) for i in range(1, n + 1)]
    price_ranks = _average_ranks(window)

    if len(set(window)) == n:  # no ties: classic d^2 formula, exact on integers
        sum_d2 = sum((t - p) ** 2 for t, p in zip(time_ranks, price_ranks))
        value = (1 - 6 * sum_d2 / (n * (n * n - 1))) * 100
    else:  # ties: Pearson correlation of the rank series
        mean = (n + 1) / 2
        cov = sum((t - mean) * (p - mean) for t, p in zip(time_ranks, price_ranks))
        var_t = sum((t - mean) ** 2 for t in time_ranks)
        var_p = sum((p - mean) ** 2 for p in price_ranks)
        if var_p == 0:  # every close equal
            return None
        value = cov / math.sqrt(var_t * var_p) * 100

    return max(-100.0, min(100.0, value))

