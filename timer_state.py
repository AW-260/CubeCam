"""Pure timer state machine (no OpenCV, no clocks): feed it timestamped samples.

    IDLE --(trigger held >= hold_time)--> READY
    READY --(trigger released)--> INSPECTION   (if inspection on and not yet inspected)
    INSPECTION --(trigger held >= hold_time)--> READY   (inspection keeps running)
    READY --(trigger released)--> SOLVING
    SOLVING --(trigger engaged)--> STOPPED
    STOPPED --(trigger released)--> IDLE

"Trigger" is a single boolean per sample: both hands in the start zone, or the
spacebar held down. It is debounced: a change only counts once the new value
has persisted for `debounce` seconds, but the edge is timestamped at the
*first* sample that showed the new value, so debouncing adds latency, not error.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from models import Penalty

_EPS = 1e-9


class State(str, Enum):
    IDLE = "IDLE"
    READY = "READY"
    INSPECTION = "INSPECTION"
    SOLVING = "SOLVING"
    STOPPED = "STOPPED"


class EventType(str, Enum):
    READY = "ready"
    INSPECTION_START = "inspection_start"
    INSPECTION_WARNING = "inspection_warning"  # data = warning second (8 or 12)
    SOLVE_START = "solve_start"
    SOLVE_STOP = "solve_stop"                  # data = SolveResult
    IDLE = "idle"
    ABORT = "abort"


@dataclass(frozen=True)
class Event:
    type: EventType
    t: float
    data: object = None


@dataclass
class TimerConfig:
    hold_time: float = 0.3
    debounce: float = 0.05
    inspection_enabled: bool = False
    inspection_limit: float = 15.0
    dnf_limit: float = 17.0
    warnings: tuple[float, ...] = (8.0, 12.0)


@dataclass(frozen=True)
class SolveResult:
    time: float
    penalty: Penalty
    inspection_time: float | None
    start: float
    end: float


def inspection_penalty(elapsed: float, limit: float = 15.0, dnf_limit: float = 17.0) -> Penalty:
    """WCA A3/A7: start after 15 s of inspection -> +2, after 17 s -> DNF."""
    if elapsed > dnf_limit + _EPS:
        return Penalty.DNF
    if elapsed > limit + _EPS:
        return Penalty.PLUS2
    return Penalty.OK


class TimerStateMachine:
    def __init__(self, config: TimerConfig | None = None):
        self.config = config or TimerConfig()
        self.state = State.IDLE
        self.state_since: float | None = None
        self.inspection_start: float | None = None
        self.solve_start: float | None = None
        self.last_result: SolveResult | None = None
        self.raw_active = False
        self._last_t: float | None = None
        self._zone = False                 # debounced trigger
        self._zone_since: float | None = None
        self._cand_since: float | None = None
        self._hold_ok_after = float("-inf")  # holds must start at/after this time
        self._warned: set[float] = set()

    # ----------------------------------------------------------------- queries
    @property
    def zone_active(self) -> bool:
        """Debounced trigger value."""
        return self._zone

    def hold_progress(self, t: float) -> float:
        """0..1 progress towards READY while the trigger is held (for UI)."""
        if self.state not in (State.IDLE, State.INSPECTION) or not self._zone:
            return 0.0
        if self._zone_since is None or self._zone_since < self._hold_ok_after:
            return 0.0
        return min(1.0, (t - self._zone_since) / max(self.config.hold_time, _EPS))

    def solve_elapsed(self, t: float) -> float | None:
        if self.state == State.SOLVING and self.solve_start is not None:
            return max(0.0, t - self.solve_start)
        if self.last_result is not None:
            return self.last_result.time
        return None

    def inspection_elapsed(self, t: float) -> float | None:
        if self.inspection_start is None:
            return None
        return max(0.0, t - self.inspection_start)

    @property
    def in_inspection(self) -> bool:
        return self.inspection_start is not None and self.state in (State.INSPECTION, State.READY)

    # ---------------------------------------------------------------- commands
    def update(self, t: float, active: bool) -> list[Event]:
        """Feed one sample. `t` must be non-decreasing; older samples are ignored."""
        if self._last_t is not None and t < self._last_t:
            return []
        self._last_t = t
        if self.state_since is None:
            self.state_since = t
        self.raw_active = bool(active)
        events: list[Event] = []
        edge = self._debounce(t, bool(active))
        cfg = self.config

        if self.in_inspection:
            elapsed = t - self.inspection_start
            for w in cfg.warnings:
                if elapsed + _EPS >= w and w not in self._warned:
                    self._warned.add(w)
                    events.append(Event(EventType.INSPECTION_WARNING, t, w))

        st = self.state
        if st in (State.IDLE, State.INSPECTION):
            if (
                self._zone
                and self._zone_since is not None
                and self._zone_since >= self._hold_ok_after
                and t - self._zone_since + _EPS >= cfg.hold_time
            ):
                self._set(State.READY, t)
                events.append(Event(EventType.READY, t))
        elif st == State.READY:
            if edge is not None and not self._zone:
                lift = edge
                if cfg.inspection_enabled and self.inspection_start is None:
                    self.inspection_start = lift
                    self._warned.clear()
                    self._set(State.INSPECTION, lift)
                    events.append(Event(EventType.INSPECTION_START, lift))
                else:
                    self.solve_start = lift
                    self._set(State.SOLVING, lift)
                    events.append(Event(EventType.SOLVE_START, lift))
        elif st == State.SOLVING:
            if edge is not None and self._zone:
                result = self._finish(edge)
                events.append(Event(EventType.SOLVE_STOP, edge, result))
        elif st == State.STOPPED:
            if edge is not None and not self._zone:
                self._set(State.IDLE, edge)
                events.append(Event(EventType.IDLE, edge))
        return events

    def abort(self, t: float) -> list[Event]:
        """Cancel inspection / a running solve without recording anything."""
        if self.state in (State.IDLE, State.STOPPED):
            return []
        self.inspection_start = None
        self.solve_start = None
        self._warned.clear()
        self._hold_ok_after = t  # hands must lift and be placed again
        self._set(State.IDLE, t)
        return [Event(EventType.ABORT, t)]

    # ---------------------------------------------------------------- internals
    def _finish(self, end: float) -> SolveResult:
        insp = None
        penalty = Penalty.OK
        if self.inspection_start is not None:
            insp = self.solve_start - self.inspection_start
            penalty = inspection_penalty(insp, self.config.inspection_limit, self.config.dnf_limit)
        result = SolveResult(
            time=end - self.solve_start,
            penalty=penalty,
            inspection_time=insp,
            start=self.solve_start,
            end=end,
        )
        self.last_result = result
        self.inspection_start = None
        self.solve_start = None
        self._set(State.STOPPED, end)
        return result

    def _set(self, state: State, t: float) -> None:
        self.state = state
        self.state_since = t

    def _debounce(self, t: float, raw: bool) -> float | None:
        """Update the debounced trigger. Returns the edge timestamp when it flips."""
        if raw == self._zone:
            self._cand_since = None
            return None
        if self._cand_since is None:
            self._cand_since = t
        if t - self._cand_since + _EPS >= self.config.debounce:
            self._zone = raw
            self._zone_since = self._cand_since
            self._cand_since = None
            return self._zone_since
        return None
