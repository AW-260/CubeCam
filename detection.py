"""Trigger-detector interface.

A detector looks at one camera frame and answers a single question: "is the
timer trigger engaged right now?" (stage 1: both hands in the start zone).
Future detectors (cube motion, colour reading) implement the same interface and
can be combined with `CombinedDetector`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

from zone import Zone  # noqa: F401  (re-exported: the start zone is a polygon, see zone.py)


class DetectorUnavailable(RuntimeError):
    """Raised when a detector cannot be constructed (missing lib, model, DLL...)."""


@dataclass
class TriggerSample:
    t: float                      # capture timestamp of the frame (perf_counter seconds)
    active: bool                  # trigger engaged?
    debug: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class TriggerDetector(Protocol):
    name: str

    def process(self, frame_bgr, t: float) -> TriggerSample: ...

    def close(self) -> None: ...


class NullDetector:
    """Never active. Used when no real detector is available (keyboard mode)."""

    name = "none"

    def process(self, frame_bgr, t: float) -> TriggerSample:
        return TriggerSample(t, False)

    def close(self) -> None:
        pass


class CombinedDetector:
    """Combine detectors with 'all' (AND) or 'any' (OR)."""

    def __init__(self, detectors: Sequence[TriggerDetector], mode: str = "all"):
        if mode not in ("all", "any"):
            raise ValueError("mode must be 'all' or 'any'")
        self.detectors = list(detectors)
        self.mode = mode
        self.name = f"{mode}(" + ",".join(d.name for d in self.detectors) + ")"

    def process(self, frame_bgr, t: float) -> TriggerSample:
        samples = [d.process(frame_bgr, t) for d in self.detectors]
        values = [s.active for s in samples]
        active = all(values) if self.mode == "all" else any(values)
        debug: dict[str, Any] = {}
        for d, s in zip(self.detectors, samples):
            debug.update(s.debug)
            debug[f"{d.name}.active"] = s.active
        return TriggerSample(t, active, debug)

    def close(self) -> None:
        for d in self.detectors:
            d.close()
