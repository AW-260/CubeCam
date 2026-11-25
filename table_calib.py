"""Table-plane calibration (key T). Pure logic: main.py feeds it, ui.py draws it.

Modes (T cycles phone -> click -> closed; Enter accepts, Esc cancels):
  * phone: a phone lying flat on the table shows one large ArUco marker
    (tools/make_phone_marker.py). It is found automatically; the corner
    positions are averaged over the last few detections while the phone lies
    still. The marker's real width comes from config (table.phone_marker_mm).
  * click: click the 4 corners of any rectangle lying on the table, in order
    round its edge (a zoom loupe follows the mouse), then give its size:
    A = A4, L = US Letter (press again to turn it 90 degrees), or type it,
    e.g. 29.7x21 (cm). The first two clicks lie along the first number.
"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from plane import TablePlane, plane_from_rectangle, polygon_area

EVENT_MOUSEMOVE, EVENT_LBUTTONDOWN, EVENT_RBUTTONDOWN = 0, 1, 2
DEFAULT_PHONE_MARKER_MM = 56.0      # black square of the default PNG on a typical ~68 mm wide screen
PRESETS = {"a": ("A4", 29.7, 21.0), "l": ("US Letter", 27.94, 21.59)}
_SIZE = re.compile(r"^\s*(\d+(?:\.\d*)?)\s*[x*]\s*(\d+(?:\.\d*)?)\s*$")


class CalibMode(str, Enum):
    PHONE = "phone"
    CLICK = "click"


@dataclass
class CalibView:
    title: str
    lines: list[str]
    warnings: list[str] = field(default_factory=list)
    marker_px: np.ndarray | None = None            # detected phone marker corners (4, 2)
    clicks_px: list[tuple[float, float]] = field(default_factory=list)
    mouse_px: tuple[float, float] | None = None    # for the zoom loupe (click mode)
    grid_px: list[np.ndarray] = field(default_factory=list)  # 10 cm grid of the candidate plane
    progress: float = 0.0


def parse_size(text: str) -> tuple[float, float] | None:
    m = _SIZE.match(text.replace(",", ".").replace("×", "x").lower())
    if not m:
        return None
    w, h = float(m.group(1)), float(m.group(2))
    return (w, h) if 1.0 <= w <= 500 and 1.0 <= h <= 500 else None


def is_simple_quad(p: np.ndarray) -> bool:
    """True for a convex quadrilateral given in order round its edge (not a bow-tie)."""
    p = np.asarray(p, dtype=np.float64).reshape(4, 2)
    signs = []
    for i in range(4):
        a, b, c = p[i], p[(i + 1) % 4], p[(i + 2) % 4]
        signs.append(np.sign((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])))
    return all(s > 0 for s in signs) or all(s < 0 for s in signs)


def grid_lines(plane: TablePlane, half_x: float = 40, near: float = 30, far: float = 60,
               step: float = 10, samples: int = 12) -> list[np.ndarray]:
    """Visible pieces of a table grid (every `step` cm round the origin) as image polylines."""
    lines = []
    xs = np.arange(-half_x, half_x + 1e-6, step)
    ys = np.arange(-far, near + 1e-6, step)
    for x in xs:
        lines.append(np.column_stack([np.full(samples, x), np.linspace(ys[0], ys[-1], samples)]))
    for y in ys:
        lines.append(np.column_stack([np.linspace(xs[0], xs[-1], samples), np.full(samples, y)]))
    out = []
    for pts in lines:
        img, ok = plane.to_image(pts)
        run: list = []
        for p, good in zip(img, ok):
            if good and np.all(np.abs(p) < 1e5):
                run.append(p)
            elif len(run) > 1:
                out.append(np.array(run))
                run = []
            else:
                run = []
        if len(run) > 1:
            out.append(np.array(run))
    return out


class TableCalibration:
    def __init__(self, phone_marker_mm: float | None = None, phone_id: int = 10, average: int = 10,
                 steady_tol: float = 0.012, hfov_deg: float | None = None):
        self.phone_marker_mm = phone_marker_mm
        self.hfov_deg = hfov_deg             # camera field of view if known (else estimated)
        self.phone_id = int(phone_id)
        self.average = int(average)
        self.steady_tol = steady_tol           # fraction of the frame width
        self.active = False
        self.mode = CalibMode.PHONE
        self.candidate: TablePlane | None = None
        self.image_size: tuple[int, int] = (1280, 720)
        self._samples: deque[np.ndarray] = deque(maxlen=self.average)
        self._last_seen: float | None = None
        self._now = 0.0
        self.clicks: list[tuple[float, float]] = []
        self.mouse: tuple[float, float] | None = None
        self.size_text = ""
        self._preset: tuple[str, float, float] | None = None
        self.message = ""

    @property
    def marker_cm(self) -> float:
        return (self.phone_marker_mm or DEFAULT_PHONE_MARKER_MM) / 10.0

    @property
    def scale_known(self) -> bool:
        return bool(self.phone_marker_mm)

    # ---------------------------------------------------------------- control
    def start(self, mode: CalibMode = CalibMode.PHONE, image_size=None) -> None:
        self.active = True
        if image_size:
            self.image_size = (int(image_size[0]), int(image_size[1]))
        self._enter(mode)

    def _enter(self, mode: CalibMode) -> None:
        self.mode = mode
        self.candidate = None
        self._samples.clear()
        self._last_seen = None
        self.clicks = []
        self.mouse = None
        self.size_text = ""
        self._preset = None
        self.message = ""

    def cycle(self) -> CalibMode | None:
        if self.mode is CalibMode.PHONE:
            self._enter(CalibMode.CLICK)
            return CalibMode.CLICK
        self.cancel()
        return None

    def cancel(self) -> None:
        self.active = False
        self._enter(CalibMode.PHONE)

    def accept(self) -> TablePlane | None:
        if self.candidate is None:
            return None
        plane = self.candidate
        self.cancel()
        return plane

    # ------------------------------------------------------------------ phone
    def update_phone(self, t: float, corners: dict[int, np.ndarray] | None, image_size=None) -> None:
        """Feed the latest ArUco detection (normalised corners per id), e.g. a few times a second."""
        if not self.active or self.mode is not CalibMode.PHONE:
            return
        self._now = t
        if image_size:
            self.image_size = (int(image_size[0]), int(image_size[1]))
        c = (corners or {}).get(self.phone_id)
        if c is None:
            return
        px = np.asarray(c, dtype=np.float64).reshape(4, 2) * self.image_size
        if self._samples:
            mean = np.mean(self._samples, axis=0)
            if np.max(np.linalg.norm(px - mean, axis=1)) > self.steady_tol * self.image_size[0]:
                self._samples.clear()          # the phone moved: start averaging again
        self._samples.append(px)
        self._last_seen = t
        if len(self._samples) >= 3:
            mean = np.mean(self._samples, axis=0)
            s = self.marker_cm
            try:
                self.candidate = plane_from_rectangle([(0, 0), (s, 0), (s, s), (0, s)], mean,
                                                      self.image_size, "phone", self.hfov_deg)
            except ValueError:
                self.candidate = None

    # ------------------------------------------------------------------ click
    def on_mouse(self, event: int, x: float, y: float, width: int, height: int) -> None:
        if not self.active or self.mode is not CalibMode.CLICK or width <= 0 or height <= 0:
            return
        # clicks are kept in calibration-image pixels (normally the same as the window canvas)
        p = (x * self.image_size[0] / width, y * self.image_size[1] / height)
        self.mouse = p
        if event == EVENT_LBUTTONDOWN and len(self.clicks) < 4:
            self.clicks.append(p)
            self._recompute_click()
        elif event == EVENT_RBUTTONDOWN:
            self.undo()

    def undo(self) -> None:
        if self.clicks:
            self.clicks.pop()
            self._recompute_click()

    def on_key(self, key: int) -> bool:
        """Typing in click mode. Returns True when the key was used here."""
        if not self.active or self.mode is not CalibMode.CLICK:
            return False
        code = key & 0xFF
        ch = chr(code).lower() if 32 <= code < 127 else ""
        if code in (8, 127):
            if self.size_text:
                self.size_text = self.size_text[:-1]
                self._preset = None
            else:
                self.undo()
            self._recompute_click()
            return True
        if len(self.clicks) < 4:
            return False
        if ch in PRESETS:
            name, a, b = PRESETS[ch]
            if self._preset and self._preset[0] == name and self.size_text == f"{self._preset[1]:g}x{self._preset[2]:g}":
                a, b = self._preset[2], self._preset[1]    # pressed again: turned 90 degrees
            self._preset = (name, a, b)
            self.size_text = f"{a:g}x{b:g}"
            self._recompute_click()
            return True
        if ch and (ch.isdigit() or ch in ".x*,"):
            self.size_text = (self.size_text + ch)[:16]
            self._preset = None
            self._recompute_click()
            return True
        return False

    def _recompute_click(self) -> None:
        self.candidate = None
        self.message = ""
        if len(self.clicks) < 4:
            return
        quad = np.array(self.clicks)
        if not is_simple_quad(quad) or polygon_area(quad) < 100:
            self.message = "Those 4 clicks don't make a rectangle - click the corners in order round its edge"
            return
        size = parse_size(self.size_text)
        if size is None:
            return
        w, h = size
        try:
            self.candidate = plane_from_rectangle([(0, 0), (w, 0), (w, h), (0, h)], quad, self.image_size, "click",
                                                  self.hfov_deg)
        except ValueError as exc:
            self.message = str(exc)

    # ------------------------------------------------------------------- view
    def view(self) -> CalibView | None:
        if not self.active:
            return None
        warnings = list(self.candidate.warnings()) if self.candidate is not None else []
        grid = grid_lines(self.candidate) if self.candidate is not None else []
        if self.mode is CalibMode.PHONE:
            lines = ["Open phone_marker.png on your phone (tools/make_phone_marker.py), brightness up,",
                     "and lay the phone flat on the table where the camera sees it well."]
            seen = self._last_seen is not None and self._now - self._last_seen < 1.0
            if not seen:
                lines.append("Looking for the phone marker... (avoid glare on the screen)")
            else:
                lines.append(f"Phone marker found - hold still ({len(self._samples)}/{self.average} steady)."
                             + (" ENTER to use it." if self.candidate is not None else ""))
            if not self.scale_known:
                lines.append(f"Scale approximate: assuming a {self.marker_cm * 10:.0f} mm marker. Measure the black "
                             "square and set table.phone_marker_mm in config.yaml.")
            lines.append("ENTER accept   T: click 4 corners instead   X forget calibration   ESC cancel")
            marker = np.mean(self._samples, axis=0) if (self._samples and seen) else None
            return CalibView("TABLE CALIBRATION 1/2: phone marker", lines, warnings, marker_px=marker,
                             grid_px=grid, progress=len(self._samples) / self.average if seen else 0.0)
        n = len(self.clicks)
        if n < 4:
            lines = ["Click the 4 corners of a rectangle lying flat on the table (phone, A4 sheet...),",
                     f"in order round its edge. Corner {n + 1} of 4.  Right-click / BACKSPACE: undo."]
        else:
            size = parse_size(self.size_text)
            name = f" ({self._preset[0]})" if self._preset else ""
            lines = ["Size of the rectangle in cm, first number = edge from click 1 to click 2:",
                     f"  > {self.size_text or '_'}{name}" + ("" if size or not self.size_text else "   (e.g. 29.7x21)"),
                     "A = A4, L = US Letter (press again to turn 90 deg), or type e.g. 29.7x21"]
            if self.candidate is not None:
                lines.append("Check the 10 cm grid lies flat on the table, then ENTER.")
        if self.message:
            warnings = [self.message] + warnings
        lines.append("ENTER accept   T close   X forget calibration   ESC cancel")
        return CalibView("TABLE CALIBRATION 2/2: click 4 corners", lines, warnings,
                         clicks_px=list(self.clicks), mouse_px=self.mouse if n < 4 else None, grid_px=grid)
