import pytest

from models import Penalty
from timer_state import (
    EventType,
    State,
    TimerConfig,
    TimerStateMachine,
    inspection_penalty,
)

FPS = 60
DT = 1.0 / FPS


class Driver:
    """Feeds per-frame trigger samples at a fixed frame rate."""

    def __init__(self, **cfg):
        self.m = TimerStateMachine(TimerConfig(**cfg))
        self.t = 0.0
        self.events = []

    def feed(self, active: bool, seconds: float = 0.0, frames: int | None = None):
        n = frames if frames is not None else max(1, round(seconds * FPS))
        for _ in range(n):
            self.events += self.m.update(self.t, active)
            self.t += DT
        return self

    def types(self):
        return [e.type for e in self.events]

    def last(self, etype):
        return [e for e in self.events if e.type == etype][-1]


def do_solve(d: Driver, solve_seconds: float):
    d.feed(False, 0.2).feed(True, 0.5)          # arm
    assert d.m.state == State.READY
    d.feed(False, solve_seconds).feed(True, 0.2)  # solve, stop
    return d.last(EventType.SOLVE_STOP).data


def test_basic_solve_no_inspection():
    d = Driver()
    d.feed(False, 0.5)
    assert d.m.state == State.IDLE
    d.feed(True, 0.5)
    assert d.m.state == State.READY
    lift_t = d.t
    d.feed(False, 10.0)
    assert d.m.state == State.SOLVING
    stop_t = d.t
    d.feed(True, 0.2)
    assert d.m.state == State.STOPPED
    r = d.last(EventType.SOLVE_STOP).data
    # Start = first frame hands were out, stop = first frame hands were back.
    assert r.start == pytest.approx(lift_t)
    assert r.end == pytest.approx(stop_t)
    assert r.time == pytest.approx(10.0, abs=1e-6)
    assert r.penalty == Penalty.OK
    assert r.inspection_time is None
    d.feed(False, 0.2)
    assert d.m.state == State.IDLE
    assert d.types() == [EventType.READY, EventType.SOLVE_START, EventType.SOLVE_STOP, EventType.IDLE]


def test_short_hold_does_not_arm():
    d = Driver(hold_time=0.3)
    d.feed(True, 0.2).feed(False, 0.5)
    assert d.m.state == State.IDLE
    assert EventType.READY not in d.types()


def test_hold_exactly_threshold_arms():
    d = Driver(hold_time=0.3, debounce=0.0)
    d.feed(True, frames=19)  # t = 0 .. 0.3
    assert d.m.state == State.READY


def test_flicker_out_during_ready_does_not_start():
    d = Driver(debounce=0.05)
    d.feed(True, 0.5)
    assert d.m.state == State.READY
    for _ in range(5):  # single-frame tracking dropouts
        d.feed(False, frames=1).feed(True, frames=5)
    assert d.m.state == State.READY
    assert EventType.SOLVE_START not in d.types()


def test_flicker_in_during_solve_does_not_stop():
    d = Driver(debounce=0.05)
    d.feed(True, 0.5).feed(False, 1.0)
    assert d.m.state == State.SOLVING
    for _ in range(5):
        d.feed(True, frames=2).feed(False, frames=10)  # 2 frames = 33 ms < debounce
    assert d.m.state == State.SOLVING
    assert EventType.SOLVE_STOP not in d.types()


def test_debounce_does_not_bias_time():
    """Edge timestamps are the first changed frame, not when debounce confirms."""
    for debounce in (0.0, 0.05, 0.1):
        d = Driver(debounce=debounce)
        r = do_solve(d, 7.5)
        assert r.time == pytest.approx(7.5, abs=DT / 2)


def test_flicker_while_idle_holding_restarts_hold():
    d = Driver(hold_time=0.3, debounce=0.0)
    d.feed(True, 0.2).feed(False, frames=1).feed(True, 0.2)
    assert d.m.state == State.IDLE  # neither hold reached 0.3 s
    d.feed(True, 0.15)
    assert d.m.state == State.READY


def test_stopped_requires_release_before_rearming():
    d = Driver()
    do_solve(d, 3.0)
    d.feed(True, 2.0)  # keep hands down after stopping
    assert d.m.state == State.STOPPED
    d.feed(False, 0.2)
    assert d.m.state == State.IDLE
    d.feed(True, 0.5)
    assert d.m.state == State.READY


def test_consecutive_solves():
    d = Driver()
    r1 = do_solve(d, 5.0)
    d.feed(False, 1.0)
    r2 = do_solve(d, 8.0)
    assert r1.time == pytest.approx(5.0, abs=1e-6)
    assert r2.time == pytest.approx(8.0, abs=1e-6)


def _inspect_then_solve(inspection_s: float, solve_s: float = 5.0):
    d = Driver(inspection_enabled=True, debounce=0.05, hold_time=0.3)
    d.feed(True, 0.5)                       # arm (hands placed after scrambling)
    assert d.m.state == State.READY
    d.feed(False, 1.0)                      # lift -> inspection starts
    assert d.m.state == State.INSPECTION
    insp_start = d.last(EventType.INSPECTION_START).t
    # Remaining inspection time: put hands down so they lift exactly at inspection_s.
    d.feed(False, max(0.0, inspection_s - 1.0 - 0.5))
    d.feed(True, 0.5)
    assert d.m.state == State.READY
    d.feed(False, solve_s).feed(True, 0.2)
    r = d.last(EventType.SOLVE_STOP).data
    return d, r, insp_start


@pytest.mark.parametrize(
    "inspection_s, expected",
    [(5.0, Penalty.OK), (14.9, Penalty.OK), (15.5, Penalty.PLUS2), (16.9, Penalty.PLUS2),
     (17.5, Penalty.DNF), (25.0, Penalty.DNF)],
)
def test_inspection_penalties(inspection_s, expected):
    d, r, _ = _inspect_then_solve(inspection_s)
    assert r.inspection_time == pytest.approx(inspection_s, abs=DT)
    assert r.penalty == expected
    assert r.time == pytest.approx(5.0, abs=1e-6)  # raw time excludes the penalty


def test_inspection_penalty_function_boundaries():
    assert inspection_penalty(15.0) == Penalty.OK
    assert inspection_penalty(15.01) == Penalty.PLUS2
    assert inspection_penalty(17.0) == Penalty.PLUS2
    assert inspection_penalty(17.01) == Penalty.DNF


def test_inspection_warnings_fire_once_each():
    d, r, insp_start = _inspect_then_solve(13.0)
    warns = [e for e in d.events if e.type == EventType.INSPECTION_WARNING]
    assert [w.data for w in warns] == [8.0, 12.0]
    assert warns[0].t - insp_start == pytest.approx(8.0, abs=DT)
    assert warns[1].t - insp_start == pytest.approx(12.0, abs=DT)


def test_no_warnings_when_started_early():
    d, _, _ = _inspect_then_solve(5.0)
    assert EventType.INSPECTION_WARNING not in d.types()


def test_inspection_starts_on_lift_and_flicker_in_inspection_ignored():
    d = Driver(inspection_enabled=True)
    d.feed(True, 0.5).feed(False, 2.0)
    assert d.m.state == State.INSPECTION
    d.feed(True, frames=1).feed(False, 1.0)  # brief flicker: not armed, no solve
    assert d.m.state == State.INSPECTION
    assert EventType.SOLVE_START not in d.types()


def test_full_state_sequence_with_inspection():
    d, _, _ = _inspect_then_solve(6.0)
    d.feed(False, 0.2)
    seq = [e.type for e in d.events if e.type != EventType.INSPECTION_WARNING]
    assert seq == [EventType.READY, EventType.INSPECTION_START, EventType.READY,
                   EventType.SOLVE_START, EventType.SOLVE_STOP, EventType.IDLE]


def test_abort_resets_and_requires_new_hold():
    d = Driver()
    d.feed(True, 0.5).feed(False, 2.0)
    assert d.m.state == State.SOLVING
    d.feed(True, frames=1)  # hands touching zone (not yet a debounced stop)
    evs = d.m.abort(d.t)
    assert [e.type for e in evs] == [EventType.ABORT]
    assert d.m.state == State.IDLE
    d.feed(True, 1.0)  # these hands were already down at abort time -> no auto-arm
    assert d.m.state == State.IDLE
    d.feed(False, 0.2).feed(True, 0.5)
    assert d.m.state == State.READY


def test_abort_when_idle_is_noop():
    d = Driver()
    assert d.m.abort(1.0) == []


def test_out_of_order_timestamps_ignored():
    m = TimerStateMachine(TimerConfig(debounce=0.0))
    m.update(1.0, True)
    assert m.update(0.5, False) == []
    assert m.raw_active is True


def test_solve_elapsed_and_hold_progress():
    d = Driver(hold_time=0.3, debounce=0.0)
    d.feed(True, frames=10)
    assert 0.0 < d.m.hold_progress(d.t - DT) < 1.0
    d.feed(True, 0.3).feed(False, 2.0)
    assert d.m.solve_elapsed(d.t) == pytest.approx(2.0, abs=DT * 1.5)
    assert d.m.hold_progress(d.t) == 0.0


def test_toggle_inspection_between_solves():
    d = Driver()
    do_solve(d, 2.0)
    d.feed(False, 0.2)
    d.m.config.inspection_enabled = True
    d.feed(True, 0.5).feed(False, 0.5)
    assert d.m.state == State.INSPECTION
