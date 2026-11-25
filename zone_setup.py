"""Interactive start-zone setup (key M). Pure logic: main.py feeds it, ui.py draws it.

Modes (M cycles through them, then closes; Esc cancels, Enter accepts):
  * mat (only with a table plane, see key T): the AR mat lies on the table.
    Drag it to slide it across the table in perspective; [ and ] rotate it.
  * drag (without a table plane): click and drag a rectangle on the view.
  * hands: rest both hands where the mat should be and hold still ~2 s. With a
    table plane the mat is built on the table round your hands; without one
    it's a rectangle round both hands plus a margin.
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
ROTATE_STEP = 5.0


class SetupMode(str, Enum):
    MAT = "mat"
    DRAG = "drag"
    HANDS = "hands"
    MARKERS = "markers"


NAMES = {SetupMode.MAT: "mat", SetupMode.DRAG: "drag", SetupMode.HANDS: "hands", SetupMode.MARKERS: "markers"}


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
        self.plane = None                  # TablePlane while setting up on the table
        self.mat_size = (30.0, 22.0)
        self.candidate_mat = None          # ar_mat.Mat on the plane
        self._start_mat = None
        self._grab = None                  # table offset from the mouse to the mat centre while dragging
        self._drag_start: Point | None = None
        self._drag_now: Point | None = None
        self._palms: list[Point] = []
        self._markers: list[Point] = []
        self._marker_status = ""
        self.note = ""

    # ---------------------------------------------------------------- control
    @property
    def modes(self) -> list[SetupMode]:
        first = SetupMode.MAT if self.plane is not None else SetupMode.DRAG
        out = [first]
        if self.hands_available:
            out.append(SetupMode.HANDS)
        if self.markers_available:
            out.append(SetupMode.MARKERS)
        return out

    def start(self, mode: SetupMode | None = None, plane=None, mat_size=None, current_mat=None) -> None:
        """Open the setup. With a table plane the first mode places the AR mat."""
        self.plane = plane
        if mat_size:
            self.mat_size = (float(mat_size[0]), float(mat_size[1]))
        self._start_mat = current_mat
        self.active = True
        self._enter(mode if mode is not None else self.modes[0])

    def _enter(self, mode: SetupMode) -> None:
        if mode is SetupMode.MAT and self.plane is None:
            mode = SetupMode.DRAG
        self.mode = mode
        self.candidate = None
        self.candidate_mat = None
        self._grab = None
        self._drag_start = self._drag_now = None
        self._palms = []
        self._markers = []
        self._marker_status = ""
        self.note = ""
        self.builder.reset()
        if mode is SetupMode.MAT:
            self._set_mat(self._start_mat if self._start_mat is not None else self._default_mat())
            if self.candidate_mat is None:
                self.note = "The table isn't visible where the mat would go - drag it onto the table"

    def _default_mat(self):
        from ar_mat import mat_at_image_point

        for y in (0.75, 0.7, 0.8, 0.65, 0.85, 0.6, 0.9):
            m = mat_at_image_point(self.plane, (0.5, y), *self.mat_size)
            if m is not None:
                return m
        return None

    def _set_mat(self, mat) -> bool:
        from ar_mat import mat_fits

        if mat is None or not mat_fits(mat, self.plane):
            return False
        self.candidate_mat = mat
        self.candidate = mat.zone(self.plane)
        return True

    def cycle(self) -> SetupMode | None:
        """M pressed during setup: next mode, or close (returns None) after the last one."""
        modes = self.modes
        i = modes.index(self.mode) if self.mode in modes else -1
        if i + 1 < len(modes):
            self._enter(modes[i + 1])
            return self.mode
        self.cancel()
        return None

    def cancel(self) -> None:
        self.active = False
        self._enter(SetupMode.DRAG)
        self.plane = None
        self._start_mat = None

    def redo(self) -> None:
        if self.mode is SetupMode.MAT:
            self._start_mat = None          # R in mat mode: back to the default spot
        self._enter(self.mode)

    def rotate(self, delta_deg: float) -> bool:
        if self.candidate_mat is None or self.plane is None:
            return False
        return self._set_mat(self.candidate_mat.rotated(delta_deg))

    def accept(self):
        """Enter pressed: (mode, zone, mat) with mode 'fixed' | 'markers' | 'mat', or None if nothing to accept."""
        if self.candidate is None:
            return None
        if self.candidate_mat is not None and self.plane is not None:
            result = ("mat", self.candidate, self.candidate_mat)
        else:
            result = ("markers" if self.mode is SetupMode.MARKERS else "fixed", self.candidate, None)
        self.cancel()
        return result

    # ----------------------------------------------------------------- inputs
    def on_mouse(self, event: int, x: int, y: int, width: int, height: int) -> None:
        if not self.active or width <= 0 or height <= 0:
            return
        p = (min(max(x / width, 0.0), 1.0), min(max(y / height, 0.0), 1.0))
        if self.mode is SetupMode.MAT:
            self._mat_mouse(event, p)
            return
        if self.mode is not SetupMode.DRAG:
            return
        if event == EVENT_LBUTTONDOWN:
            self._drag_start = self._drag_now = p
        elif event == EVENT_MOUSEMOVE and self._drag_start is not None:
            self._drag_now = p
        elif event == EVENT_LBUTTONUP and self._drag_start is not None:
            a, b = self._drag_start, p
            self._drag_start = self._drag_now = None
            if abs(a[0] - b[0]) >= MIN_DRAG and abs(a[1] - b[1]) >= MIN_DRAG:
                self.candidate = Zone.from_corners(a, b)

    def _mat_mouse(self, event: int, p: Point) -> None:
        from ar_mat import Mat

        tp, ok = self.plane.to_table_norm([p])
        if event == EVENT_LBUTTONDOWN:
            if not ok[0]:
                self.note = "That's above the horizon - click on the table"
                return
            if self.candidate_mat is None:
                w, h = self.mat_size
                if self._set_mat(Mat(float(tp[0, 0]), float(tp[0, 1]), 0.0, w, h)):
                    self.note = ""
                self._grab = np.zeros(2) if self.candidate_mat is not None else None
            else:
                self._grab = np.array([self.candidate_mat.cx, self.candidate_mat.cy]) - tp[0]
        elif event == EVENT_MOUSEMOVE and self._grab is not None and ok[0]:
            c = tp[0] + self._grab
            if self._set_mat(self.candidate_mat.moved_to(c[0], c[1])):
                self.note = ""
            else:
                self.note = "The mat would leave the visible table there"
        elif event == EVENT_LBUTTONUP:
            self._grab = None

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
                    if self.plane is not None:
                        from ar_mat import mat_from_hands

                        mat = mat_from_hands(self.plane, self.builder.mean_hands, *self.mat_size)
                        if mat is not None and self._set_mat(mat):
                            if mat.width_cm > self.mat_size[0] + 0.01:
                                self.note = f"Mat widened to {mat.width_cm:.0f} cm so both hands fit"
                        else:
                            self.note = "Couldn't put the mat on the table there - using a flat rectangle"
        elif self.mode is SetupMode.MARKERS:
            self._markers = list(marker_points)
            self._marker_status = marker_status
            self.candidate = marker_zone

    # ------------------------------------------------------------------- view
    def view(self) -> SetupView | None:
        if not self.active:
            return None
        modes = self.modes
        i = modes.index(self.mode) if self.mode in modes else 0
        title = f"ZONE SETUP {i + 1}/{len(modes)}: {NAMES[self.mode]}"
        nxt = f"M: {NAMES[modes[i + 1]]} mode" if i + 1 < len(modes) else "M: close"
        keys = f"ENTER accept   R redo   {nxt}   ESC cancel"
        progress = 0.0
        if self.mode is SetupMode.MAT:
            lines = ["Drag the mat to slide it across the table.  [ and ] rotate it.  ENTER saves."]
            m = self.candidate_mat
            if m is not None:
                lines.append(f"Mat {m.width_cm:g} x {m.height_cm:g} cm, turned {m.angle_deg:+.0f} deg "
                             "(size: mat.width_cm / height_cm in config.yaml)")
            keys = f"ENTER accept   [ ] rotate   R reset   {nxt}   ESC cancel"
        elif self.mode is SetupMode.DRAG:
            lines = ["Click and drag a rectangle over the desk area where you rest your hands."]
            if self.candidate is not None:
                lines = ["Press ENTER to use this zone, or drag again."]
        elif self.mode is SetupMode.HANDS:
            on_table = " on the table" if self.plane is not None else ""
            if self.candidate is None:
                lines = [f"Rest both hands on the mat and hold still for {self.builder.hold:g} s.",
                         self.builder.hint]
                progress = self.builder.progress
            else:
                lines = [f"Zone built{on_table} round both hands. ENTER to use it, R to try again."]
                if self.candidate_mat is not None:
                    keys = f"ENTER accept   [ ] rotate   R redo   {nxt}   ESC cancel"
        else:
            lines = ["Place the 4 printed markers at the corners of your mat (tools/make_markers.py).",
                     f"Markers: {self._marker_status or 'looking...'}"]
            if self.candidate is not None:
                lines.append("ENTER: use the markers (the zone follows them from now on).")
        if self.note:
            lines.append(self.note)
        drag = (self._drag_start, self._drag_now) if self._drag_start and self._drag_now else None
        return SetupView(
            title=title,
            lines=lines + [keys],
            candidate=self.candidate,
            drag=drag,
            progress=progress,
            warnings=zone_warnings(self.candidate) if self.candidate is not None else [],
            palms=list(self._palms),
            markers=list(self._markers),
        )
