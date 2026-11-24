import math

import pytest

from models import Penalty, Solve, format_solve, format_time
from stats import average_of, best_single, compute_stats, mean_of, session_mean


def S(t, p=Penalty.OK):
    return Solve(time=t, penalty=p)


DNF = Penalty.DNF
P2 = Penalty.PLUS2


def test_ao5_basic():
    # drop 8 (best) and 14 (worst): mean(10, 11, 12) = 11
    assert average_of([S(10), S(8), S(12), S(14), S(11)], 5) == pytest.approx(11.0)


def test_ao5_uses_last_five():
    solves = [S(100), S(10), S(8), S(12), S(14), S(11)]
    assert average_of(solves, 5) == pytest.approx(11.0)


def test_ao5_one_dnf_counts_as_worst():
    # DNF dropped as worst, 8 dropped as best -> mean(10, 11, 12)
    assert average_of([S(10), S(8), S(12, DNF), S(11), S(12)], 5) == pytest.approx(11.0)


def test_ao5_two_dnfs_is_dnf():
    assert math.isinf(average_of([S(10), S(8, DNF), S(12, DNF), S(11), S(12)], 5))


def test_ao5_plus_two_applied():
    # effective: 10, 9+2=11, 12, 13, 14 -> drop 10 and 14 -> mean(11, 12, 13) = 12
    assert average_of([S(10), S(9, P2), S(12), S(13), S(14)], 5) == pytest.approx(12.0)


def test_ao5_not_enough():
    assert average_of([S(1)] * 4, 5) is None


def test_ao12_with_one_dnf():
    times = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 9]
    solves = [S(t) for t in times] + [S(5, DNF)]
    # drop 9 (best) and DNF (worst): mean of 10..19
    assert average_of(solves, 12) == pytest.approx(sum(range(10, 20)) / 10)


def test_ao12_two_dnfs_is_dnf():
    solves = [S(t) for t in range(10, 20)] + [S(5, DNF), S(6, DNF)]
    assert math.isinf(average_of(solves, 12))


def test_ao12_dnf_only_in_older_solves_is_ignored():
    solves = [S(1, DNF), S(1, DNF)] + [S(t) for t in range(10, 22)]
    assert average_of(solves, 12) == pytest.approx(sum(range(11, 21)) / 10)


def test_mo3():
    assert mean_of([S(10), S(11), S(12)], 3) == pytest.approx(11.0)
    assert math.isinf(mean_of([S(10), S(11, DNF), S(12)], 3))
    assert mean_of([S(10), S(11)], 3) is None


def test_best_and_mean():
    solves = [S(10), S(8, DNF), S(9, P2), S(12)]
    assert best_single(solves) == pytest.approx(10.0)
    assert session_mean(solves) == pytest.approx((10 + 11 + 12) / 3)
    assert best_single([]) is None
    assert session_mean([]) is None
    assert math.isinf(session_mean([S(1, DNF)]))


def test_compute_stats():
    solves = [S(t) for t in (12, 11, 13, 10, 14)]
    st = compute_stats(solves)
    assert st.count == 5 and st.dnf_count == 0
    assert st.current == 14
    assert st.best == 10
    assert st.mo3 == pytest.approx((13 + 10 + 14) / 3)
    assert st.ao5 == pytest.approx(12.0)
    assert st.ao12 is None


def test_format_time():
    assert format_time(None) == "-"
    assert format_time(math.inf) == "DNF"
    assert format_time(9.876) == "9.87"  # truncated, like WCA
    assert format_time(62.345) == "1:02.34"
    assert format_time(0) == "0.00"
    assert format_solve(S(9.5, P2)) == "11.50+"
    assert format_solve(S(9.5, DNF)) == "DNF(9.50)"
