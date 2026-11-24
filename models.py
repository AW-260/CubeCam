"""Shared data types used across CubeCam (no OpenCV / MediaPipe here)."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum


class Penalty(str, Enum):
    OK = "OK"
    PLUS2 = "+2"
    DNF = "DNF"


@dataclass
class Solve:
    """One recorded solve. `time` is the raw measured time in seconds (no penalty)."""

    time: float
    penalty: Penalty = Penalty.OK
    scramble: str = ""
    timestamp: str = ""          # ISO-8601 local time the solve finished
    fps: float | None = None     # effective sampling rate used to time this solve
    inspection: float | None = None  # inspection time used, if inspection was on

    @property
    def effective(self) -> float:
        """Time with penalty applied; DNF is +inf so it sorts as the worst."""
        if self.penalty == Penalty.DNF:
            return math.inf
        if self.penalty == Penalty.PLUS2:
            return self.time + 2.0
        return self.time

    def to_dict(self) -> dict:
        d = asdict(self)
        d["penalty"] = self.penalty.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Solve":
        return cls(
            time=float(d["time"]),
            penalty=Penalty(d.get("penalty", "OK")),
            scramble=str(d.get("scramble", "")),
            timestamp=str(d.get("timestamp", "")),
            fps=None if d.get("fps") is None else float(d["fps"]),
            inspection=None if d.get("inspection") is None else float(d["inspection"]),
        )


def format_time(seconds: float | None, decimals: int = 2) -> str:
    """Format seconds csTimer-style: '9.87', '1:02.34', 'DNF', '-' for None."""
    if seconds is None:
        return "-"
    if math.isinf(seconds):
        return "DNF"
    # Truncate (not round) like WCA / csTimer displays.
    scale = 10 ** decimals
    total = math.floor(seconds * scale + 1e-9) / scale
    minutes, secs = divmod(total, 60)
    if minutes >= 1:
        return f"{int(minutes)}:{secs:0{decimals + 3}.{decimals}f}"
    return f"{secs:.{decimals}f}"


def format_solve(solve: Solve) -> str:
    if solve.penalty == Penalty.DNF:
        return f"DNF({format_time(solve.time)})"
    if solve.penalty == Penalty.PLUS2:
        return f"{format_time(solve.effective)}+"
    return format_time(solve.time)
