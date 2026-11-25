"""Draw mode: a traced mouse path becomes a clean, simple, closed shape."""
import math

import numpy as np
import pytest

from freehand import MAX_POINTS, clean_shape, polygon_area, resample_closed, smooth_closed


def _segments_cross(a, b, c, d) -> bool:
    def orient(p, q, r):
        return np.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))
    return orient(a, b, c) != orient(a, b, d) and orient(c, d, a) != orient(c, d, b)


def is_simple(poly) -> bool:
    n = len(poly)
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue                     # neighbours through the closing edge
            if _segments_cross(poly[i], poly[(i + 1) % n], poly[j], poly[(j + 1) % n]):
                return False
    return True


def signed_area(p):
    x, y = p[:, 0], p[:, 1]
    return (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2


def noisy_ellipse(rx=30, ry=15, overshoot=1.08, noise=0.6, n=300, seed=1):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * math.pi * overshoot, n)
    return np.column_stack([rx * np.cos(t), ry * np.sin(t)]) + rng.normal(0, noise, (n, 2))


def test_noisy_loop_with_overshoot_becomes_a_clean_shape():
    s = clean_shape(noisy_ellipse())
    assert s is not None and 8 <= len(s) <= MAX_POINTS
    assert is_simple(s)                                     # the overshoot doesn't make a self-crossing
    assert signed_area(s) > 0                               # clockwise on screen (y down)
    assert polygon_area(s) == pytest.approx(math.pi * 30 * 15, rel=0.06)
    r = np.hypot(s[:, 0] / 30, s[:, 1] / 15)
    assert np.all(np.abs(r - 1) < 0.08)                     # jitter smoothed away, size kept


def test_units_do_not_matter():
    a = clean_shape(noisy_ellipse())
    b = clean_shape(noisy_ellipse() * 25 + (640, 400))     # the same path in screen px
    assert len(a) == len(b)
    assert np.allclose(a * 25 + (640, 400), b, atol=25 * 0.05)


def test_unclosed_gap_and_spur():
    # a rectangle traced with a gap at the end, plus a scribble poking out of one side
    rect = resample_closed(np.array([(0, 0), (40, 0), (40, 20), (0, 20)], float), 400)[:370]
    spur = np.array([(40 + 6 * u, 10 + 0.3 * u) for u in np.linspace(0, 1, 15)] +
                    [(46 - 6 * u, 10.3) for u in np.linspace(0, 1, 15)])
    path = np.vstack([rect[:150], spur, rect[150:]])
    s = clean_shape(path)
    assert s is not None and is_simple(s)
    assert s[:, 0].max() < 41.5                             # the thin spur is gone
    lo, hi = s.min(axis=0), s.max(axis=0)
    assert np.allclose(lo, (0, 0), atol=1.2) and np.allclose(hi, (40, 20), atol=1.2)


def test_lines_and_dots_are_rejected():
    assert clean_shape(np.column_stack([np.linspace(0, 50, 80), np.linspace(0, 3, 80)])) is None
    assert clean_shape([(0, 0), (1, 1), (2, 0)]) is None
    assert clean_shape(np.zeros((20, 2))) is None
    assert clean_shape([(0, 0), (float("nan"), 1), (3, 3)]) is None


def test_helpers():
    sq = np.array([(0, 0), (10, 0), (10, 10), (0, 10)], float)
    r = resample_closed(sq, 40)
    assert len(r) == 40 and np.allclose(np.linalg.norm(np.diff(r, axis=0), axis=1), 1.0)
    sm = smooth_closed(r, 2.0)
    assert polygon_area(sm) < 100 and polygon_area(sm) > 90     # corners rounded a little
    assert np.allclose(sm.mean(axis=0), (5, 5), atol=0.01)
