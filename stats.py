"""Session statistics using WCA / csTimer averaging rules.

Values are seconds (float), math.inf for DNF, or None when there are too few solves.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from models import Solve


def _trim_count(n: int) -> int:
    """Solves dropped from each end of an average of n (csTimer: ceil(5%), min 1)."""
    return max(1, math.ceil(n * 0.05))


def mean_of(solves: Sequence[Solve], n: int) -> float | None:
    """Mean of the last n solves (mo3). Any DNF makes it DNF."""
    if n <= 0 or len(solves) < n:
        return None
    times = [s.effective for s in solves[-n:]]
    if any(math.isinf(x) for x in times):
        return math.inf
    return sum(times) / n


def average_of(solves: Sequence[Solve], n: int) -> float | None:
    """WCA average of the last n solves: drop best and worst, mean the rest.

    DNFs count as the worst result; more DNFs than trimmed solves -> DNF.
    """
    if n < 3 or len(solves) < n:
        return None
    times = sorted(s.effective for s in solves[-n:])
    k = _trim_count(n)
    kept = times[k:-k]
    if any(math.isinf(x) for x in kept):
        return math.inf
    return sum(kept) / len(kept)


def best_single(solves: Sequence[Solve]) -> float | None:
    if not solves:
        return None
    return min(s.effective for s in solves)  # inf if all DNF


def session_mean(solves: Sequence[Solve]) -> float | None:
    """Mean of all non-DNF solves (csTimer-style session mean)."""
    ok = [s.effective for s in solves if not math.isinf(s.effective)]
    if not ok:
        return None if not solves else math.inf
    return sum(ok) / len(ok)


@dataclass(frozen=True)
class SessionStats:
    count: int
    dnf_count: int
    current: float | None
    best: float | None
    mo3: float | None
    ao5: float | None
    ao12: float | None
    mean: float | None


def compute_stats(solves: Sequence[Solve]) -> SessionStats:
    return SessionStats(
        count=len(solves),
        dnf_count=sum(1 for s in solves if math.isinf(s.effective)),
        current=solves[-1].effective if solves else None,
        best=best_single(solves),
        mo3=mean_of(solves, 3),
        ao5=average_of(solves, 5),
        ao12=average_of(solves, 12),
        mean=session_mean(solves),
    )
