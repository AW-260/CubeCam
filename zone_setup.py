"""Interactive start-zone setup (key M). Pure logic: main.py feeds it, ui.py draws it.

Modes (M cycles drag -> hands -> markers -> closed; Esc cancels, Enter accepts):
  * drag:    click and drag a rectangle on the camera view;
  * hands:   rest both hands where the mat should be and hold still ~2 s;
             a rectangle around both hands (plus a margin) is proposed;
  * markers: 4 printed ArUco markers at the mat corners; the zone follows them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np

from zone import HandsZoneBuilder, Point, Zone, palm_centre, zone_warnings

# OpenCV mouse event codes (kept here so this module doesn't need cv2)
EVENT_MOUSEMOVE, EVENT_LBUTTONDOWN, EVENT_LBUTTONUP = 0, 1, 4
MIN_DRAG = 0.02  # a drag shorter than this (fraction of the frame) is a click, not a zone


class SetupMode(str, Enum):
    DRAG = "drag"
    HANDS = "hands"
    MARKERS = "markers"


TITLES = {
    SetupMode.DRAG: "ZONE SETUP 1/3: drag",
    SetupMode.HANDS: "ZONE SETUP 2/3: hands",
    SetupMode.MARKERS: "ZONE SETUP 3/3: markers",
}


@dataclass
class SetupView:
    """What ui.py needs to draw the setup overlay."""

    title: str
    lines: list[str]
    candidate: Zone | None = None
    drag: tuple[Point, Point] | None = None
    progress: float = 0.0
    warnings: list[str] = field(default_factory=list)
    palms: list[Point] = field(default_factory=list)
    markers: list[Point] = field(default_factory=list)


class ZoneSetup:
    def __init__(self, hands_available: bool = True, markers_available: bool = True, hold: float = 2.0):
        self.hands_available = hands_available
        self.markers_available = markers_available
        self.builder = HandsZoneBuilder(hold=hold)
        self.active = False
        self.mode = SetupMode.DRAG
        self.candidate: Zone | None = None
        self._drag_start: Point | None = None
        self._drag_now: Point | None = None
        self._palms: list[Point] = []
        self._markers: list[Point] = []
        self._marker_status = ""

    # ---------------------------------------------------------------- control
    def start(self, mode: SetupMode = SetupMode.DRAG) -> None:
        self.active = True
        self._enter(mode)

    def _enter(self, mode: SetupMode) -> None:
        self.mode = mode
        self.candidate = None
        self._drag_start = self._drag_now = None
        self._palms = []
        self._markers = []
        self._marker_status = ""
        self.builder.reset()

    def cycle(self) -> SetupMode | None:
        """M pressed during setup: next mode, or close (returns None) after the last one."""
        order = [SetupMode.DRAG, SetupMode.HANDS, SetupMode.MARKERS]
        for mode in order[order.index(self.mode) + 1:]:
            if (mode is SetupMode.HANDS and not self.hands_available) or \
               (mode is SetupMode.MARKERS and not self.markers_available):
                continue
            self._enter(mode)
            return mode
        self.cancel()
        return None

    def cancel(self) -> None:
        self.active = False
        self._enter(SetupMode.DRAG)

    def redo(self) -> None:
        self._enter(self.mode)

    def accept(self) -> tuple[str, Zone] | None:
        """Enter pressed: ('fixed' | 'markers', zone), or None if there is nothing to accept yet."""
        if self.candidate is None:
            return None
        result = ("markers" if self.mode is SetupMode.MARKERS else "fixed", self.candidate)
        self.cancel()
        return result

    # ----------------------------------------------------------------- inputs
    def on_mouse(self, event: int, x: int, y: int, width: int, height: int) -> None:
        if not self.active or self.mode is not SetupMode.DRAG or width <= 0 or height <= 0:
            return
        p = (min(max(x / width, 0.0), 1.0), min(max(y / height, 0.0), 1.0))
        if event == EVENT_LBUTTONDOWN:
            self._drag_start = self._drag_now = p
        elif event == EVENT_MOUSEMOVE and self._drag_start is not None:
            self._drag_now = p
        elif event == EVENT_LBUTTONUP and self._drag_start is not None:
            a, b = self._drag_start, p
            self._drag_start = self._drag_now = None
            if abs(a[0] - b[0]) >= MIN_DRAG and abs(a[1] - b[1]) >= MIN_DRAG:
                self.candidate = Zone.from_corners(a, b)

    def update(self, t: float, hands: Sequence[np.ndarray] | None = None,
               marker_zone: Zone | None = None, marker_status: str = "",
               marker_points: Sequence[Point] = ()) -> None:
        """Call once per frame while active."""
        if not self.active:
            return
        hands = list(hands or [])
        if self.mode is SetupMode.HANDS:
            self._palms = [palm_centre(h) for h in hands]
            if self.candidate is None:
                z = self.builder.update(t, hands)
                if z is not None:
                    self.candidate = z
        elif self.mode is SetupMode.MARKERS:
            self._markers = list(marker_points)
            self._marker_status = marker_status
            self.candidate = marker_zone

    # ------------------------------------------------------------------- view
    def view(self) -> SetupView | None:
        if not self.active:
            return None
        nxt = {SetupMode.DRAG: "M: hands mode", SetupMode.HANDS: "M: marker mode",
               SetupMode.MARKERS: "M: close"}[self.mode]
        if self.mode is SetupMode.MARKERS or (self.mode is SetupMode.HANDS and not self.markers_available):
            nxt = "M: close"
        keys = f"ENTER accept   R redo   {nxt}   ESC cancel"
        progress = 0.0
        if self.mode is SetupMode.DRAG:
            lines = ["Click and drag a rectangle over the desk area where you rest your hands."]
            if self.candidate is not None:
                lines = ["Press ENTER to use this zone, or drag again."]
        elif self.mode is SetupMode.HANDS:
            if self.candidate is None:
                lines = [f"Rest both hands on the mat and hold still for {self.builder.hold:g} s.",
                         self.builder.hint]
                progress = self.builder.progress
            else:
                lines = ["Zone built around both hands. ENTER to use it, R to try again."]
        else:
            lines = ["Place the 4 printed markers at the corners of your mat (tools/make_markers.py).",
                     f"Markers: {self._marker_status or 'looking...'}"]
            if self.candidate is not None:
                lines.append("ENTER: use the markers (the zone follows them from now on).")
        drag = (self._drag_start, self._drag_now) if self._drag_start and self._drag_now else None
        return SetupView(
            title=TITLES[self.mode],
            lines=lines + [keys],
            candidate=self.candidate,
            drag=drag,
            progress=progress,
            warnings=zone_warnings(self.candidate) if self.candidate is not None else [],
            palms=list(self._palms),
            markers=list(self._markers),
        )
