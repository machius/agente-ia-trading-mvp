"""Deterministic tests for the pure indicators. Known inputs, no I/O."""

import math
import random

import pytest

from services.indicators import rci, sma


# ---------------------------------------------------------------- SMA


def test_sma50_of_known_data():
    closes = list(range(1, 101))  # last 50 are 51..100

    assert sma(closes, 50) == 75.5


def test_sma200_of_known_data():
    closes = list(range(1, 301))  # last 200 are 101..300

    assert sma(closes, 200) == 200.5


def test_sma_exactly_period_values():
    assert sma(list(range(1, 51)), 50) == 25.5


def test_sma_uses_only_the_last_period_closes_in_chronological_order():
    assert sma([1000.0, 1.0, 2.0, 3.0], 3) == 2.0
    assert sma([1.0, 2.0, 3.0, 1000.0], 3) == pytest.approx(1005.0 / 3)


def test_sma_insufficient_data_returns_none():
    assert sma(list(range(49)), 50) is None
    assert sma([1.0, 2.0], 3) is None


def test_sma_empty_list_returns_none():
    assert sma([], 50) is None


def test_sma_with_floats():
    assert sma([0.1, 0.2, 0.3], 3) == pytest.approx(0.2)
    assert sma([100.5, 101.25, 99.75, 100.0], 4) == pytest.approx(100.375)


def test_sma_period_one_is_the_last_close():
    assert sma([1.0, 2.0, 7.5], 1) == 7.5


def test_sma_accepts_tuples_and_does_not_mutate_input():
    data = [3.0, 1.0, 2.0]

    assert sma(tuple(data), 3) == 2.0
    assert sma(data, 3) == 2.0
    assert data == [3.0, 1.0, 2.0]


@pytest.mark.parametrize("period", [0, -1, 1.5, True, "5", None])
def test_sma_invalid_period_raises(period):
    with pytest.raises(ValueError):
        sma([1.0, 2.0, 3.0], period)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_sma_non_finite_close_in_window_raises(bad):
    with pytest.raises(ValueError):
        sma([1.0, bad, 3.0], 3)


# ---------------------------------------------------------------- RCI


def test_rci_strictly_increasing_is_plus_100():
    assert rci(list(range(1, 10)), 9) == 100.0
    assert rci([1.5, 2.5, 10.0, 11.0, 50.0], 5) == 100.0


def test_rci_strictly_decreasing_is_minus_100():
    assert rci(list(range(9, 0, -1)), 9) == -100.0
    assert rci([50.0, 11.0, 10.0, 2.5, 1.5], 5) == -100.0


def test_rci_known_vector_is_60():
    # time ranks 1..5, price ranks [3,1,2,5,4] -> d = [-2,1,1,-1,1] -> sum d^2 = 8
    # RCI = (1 - 6*8 / (5*24)) * 100 = 60
    assert rci([3, 1, 2, 5, 4], 5) == 60.0


def test_rci_insufficient_data_returns_none():
    assert rci([1.0, 2.0, 3.0], 9) is None
    assert rci(list(range(8)), 9) is None


def test_rci_empty_list_returns_none():
    assert rci([], 9) is None


@pytest.mark.parametrize("period", [0, 1, -3, 2.5, True, "9", None])
def test_rci_invalid_period_raises(period):
    with pytest.raises(ValueError):
        rci([1.0, 2.0, 3.0, 4.0], period)


def test_rci_constant_series_is_undefined_and_returns_none():
    assert rci([5.0] * 9, 9) is None
    assert rci([5.0, 5.0], 2) is None


def test_rci_with_ties_uses_average_ranks_and_spearman():
    # price ranks [1, 2.5, 2.5, 4, 5], time ranks 1..5 (mean 3):
    # cov = 9.5, var_t = 10, var_p = 9.5 -> r = 9.5 / sqrt(95)
    assert rci([1, 2, 2, 3, 4], 5) == pytest.approx(100 * 9.5 / math.sqrt(95))


def test_rci_with_two_tied_pairs():
    # price ranks [1.5, 1.5, 3.5, 3.5], time ranks 1..4 (mean 2.5):
    # cov = 4, var_t = 5, var_p = 4 -> r = 4 / sqrt(20)
    assert rci([1, 1, 2, 2], 4) == pytest.approx(100 * 4 / math.sqrt(20))


def test_rci_ties_do_not_reach_the_extremes_by_accident():
    value = rci([1, 2, 2, 3, 4], 5)

    assert 0 < value < 100


def test_rci_uses_only_the_last_period_closes_in_chronological_order():
    # the leading 100 and 50 must be ignored
    assert rci([100, 50, 1, 2, 3, 4, 5], 5) == 100.0
    # same values reversed (newest first) would give -100: order matters
    assert rci([5, 4, 3, 2, 1, 50, 100], 5) == rci([3, 2, 1, 50, 100], 5)
    assert rci([5, 4, 3, 2, 1], 5) == -100.0


def test_rci_period_two():
    assert rci([1.0, 2.0], 2) == 100.0
    assert rci([2.0, 1.0], 2) == -100.0


def test_rci_is_always_within_bounds():
    rng = random.Random(12345)
    for _ in range(500):
        n = rng.randint(2, 30)
        # small integer range on purpose, to produce many ties
        series = [rng.randint(0, 6) for _ in range(n + rng.randint(0, 10))]
        value = rci(series, n)
        assert value is None or -100.0 <= value <= 100.0

    for _ in range(200):
        floats = [rng.uniform(-1e6, 1e6) for _ in range(40)]
        value = rci(floats, 9)
        assert -100.0 <= value <= 100.0


def test_rci_is_antisymmetric_when_the_series_is_negated():
    series = [3.0, 1.0, 4.0, 1.5, 9.0, 2.6, 5.3, 5.8, 9.7]

    assert rci(series, 9) == pytest.approx(-rci([-x for x in series], 9))


def test_rci_accepts_tuples_and_does_not_mutate_input():
    data = [3, 1, 2, 5, 4]

    assert rci(tuple(data), 5) == 60.0
    assert rci(data, 5) == 60.0
    assert data == [3, 1, 2, 5, 4]


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_rci_non_finite_close_in_window_raises(bad):
    with pytest.raises(ValueError):
        rci([1.0, bad, 3.0], 3)
