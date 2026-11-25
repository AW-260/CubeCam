"""Marker mode: 4 printed ArUco markers at the corners of a physical mat.

The zone is the polygon through the centres of the 4 markers (ids 0-3 of
DICT_4X4_50 by default; print them with `python tools/make_markers.py`). Which
marker sits at which corner doesn't matter: the points are put in order around
their centroid.

Detection runs in a background thread a few times per second, so it never
slows the timer loop, and keeps following the mat if the camera is bumped.
Hands resting on the mat often cover a marker:
  * 3 markers seen: the 4th is estimated from where all 4 were last time
    (an affine fit of the 3 visible ones), so the zone still follows the mat;
  * fewer seen: the last known zone is held.

ArUco codes are not mirror-symmetric, so markers are detected in the raw
camera image and the positions are mirrored afterwards when the view is.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from zone import Point, Zone, order_convex

DEFAULT_DICT = "DICT_4X4_50"
DEFAULT_IDS = (0, 1, 2, 3)


def aruco_available() -> bool:
    try:
        import cv2

        return hasattr(cv2, "aruco") and hasattr(cv2.aruco, "ArucoDetector")
    except ImportError:  # pragma: no cover
        return False


def get_dictionary(name: str = DEFAULT_DICT):
    import cv2

    code = getattr(cv2.aruco, name, None)
    if code is None:
        raise ValueError(f"unknown ArUco dictionary {name!r}")
    return cv2.aruco.getPredefinedDictionary(code)


def affine_from_3(src: Sequence[Point], dst: Sequence[Point]) -> np.ndarray | None:
    """2x3 affine matrix mapping 3 src points onto 3 dst points (None if degenerate)."""
    a = np.array([[x, y, 1.0] for x, y in src])
    if abs(np.linalg.det(a)) < 1e-9:
        return None
    b = np.array(dst, dtype=np.float64)
    return np.linalg.solve(a, b).T  # (2, 3)


@dataclass
class MarkerZoneTracker:
    """Turns per-frame marker sightings into a stable zone (pure logic, no OpenCV)."""

    ids: tuple[int, ...] = DEFAULT_IDS
    deadband: float = 0.004       # ignore movements smaller than this (detection jitter)
    zone: Zone | None = None
    last_full: dict[int, Point] = field(default_factory=dict)
    seen: dict[int, Point] = field(default_factory=dict)
    last_update: float | None = None   # when the zone last came from markers (4 or 3 seen)
    status: str = "no markers seen yet"

    def update(self, t: float, found: dict[int, Point]) -> Zone | None:
        found = {i: p for i, p in found.items() if i in self.ids}
        self.seen = found
        n = len(found)
        if n == len(self.ids):
            pts = {i: found[i] for i in self.ids}
            self.last_full = pts
            self._set(t, pts, f"{n}/{n} markers")
        elif n == len(self.ids) - 1 and len(self.last_full) == len(self.ids):
            vis = [i for i in self.ids if i in found]
            m = affine_from_3([self.last_full[i] for i in vis], [found[i] for i in vis])
            if m is not None:
                pts = dict(found)
                for i in self.ids:
                    if i not in found:
                        x, y = self.last_full[i]
                        pts[i] = tuple(m @ np.array([x, y, 1.0]))
                self._set(t, pts, f"{n}/{len(self.ids)} markers (4th estimated)")
            else:
                self.status = self._holding(t, n)
        else:
            self.status = self._holding(t, n)
        return self.zone

    def _holding(self, t: float, n: int) -> str:
        if self.zone is None:
            return f"{n}/{len(self.ids)} markers - show all {len(self.ids)}"
        age = t - self.last_update if self.last_update is not None else 0.0
        return f"{n}/{len(self.ids)} markers - holding last zone ({age:.0f} s)"

    def _set(self, t: float, pts: dict[int, Point], status: str) -> None:
        new = Zone(tuple(order_convex(pts.values())))
        if self.zone is None or max(math.dist(a, b) for a, b in zip(self.zone.points, new.points)) > self.deadband:
            self.zone = new
        self.last_update = t
        self.status = status


def make_detector(dict_name: str = DEFAULT_DICT, refine: bool = False):
    import cv2

    params = cv2.aruco.DetectorParameters()
    if refine:
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(get_dictionary(dict_name), params)


def detect_marker_corners(image_bgr: np.ndarray, detector, mirror: bool,
                          process_width: int = 960) -> dict[int, np.ndarray]:
    """{marker id: (4, 2) corners} normalised to the (possibly mirrored) view.

    Corners keep ArUco's order (the marker's own top-left, top-right,
    bottom-right, bottom-left); mirroring only flips the x coordinates.
    """
    import cv2

    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    if process_width and w > process_width:
        gray = cv2.resize(gray, (process_width, round(h * process_width / w)), interpolation=cv2.INTER_AREA)
    gh, gw = gray.shape[:2]
    corners, ids, _ = detector.detectMarkers(gray)
    out: dict[int, np.ndarray] = {}
    if ids is None:
        return out
    sx, sy = w / gw, h / gh
    for c, i in zip(corners, ids.ravel()):
        # full-resolution pixel-centre coordinates (exact when the detection ran on a smaller copy)
        p = (c.reshape(-1, 2).astype(np.float64) + 0.5) * (sx, sy) - 0.5
        if mirror:
            p[:, 0] = (w - 1) - p[:, 0]      # same as cv2.flip(image, 1)
        out[int(i)] = p / (w, h)
    return out


def detect_markers(image_bgr: np.ndarray, detector, mirror: bool, process_width: int = 960) -> dict[int, Point]:
    """{marker id: centre} in normalised coordinates of the (possibly mirrored) view."""
    return {i: (float(c[:, 0].mean()), float(c[:, 1].mean()))
            for i, c in detect_marker_corners(image_bgr, detector, mirror, process_width).items()}


def plane_from_marker(corners_norm: np.ndarray, size_cm: float, image_size, source: str = "marker",
                      hfov_deg: float | None = None):
    """Table plane from one square marker of known side lying on the table."""
    from plane import plane_from_rectangle

    s = float(size_cm)
    square = [(0.0, 0.0), (s, 0.0), (s, s), (0.0, s)]
    return plane_from_rectangle(square, np.asarray(corners_norm) * image_size, image_size, source, hfov_deg)


class ArucoWorker:
    """Background ArUco detection. `submit` is O(1) and never blocks the caller.

    Subclasses (or the default) turn each detection into state in `_handle`,
    which runs under the lock; `snapshot`-style readers take the same lock.
    """

    def __init__(self, dict_name: str = DEFAULT_DICT, mirror: bool = True, interval: float = 0.25,
                 process_width: int = 960, refine: bool = False):
        self.mirror = mirror
        self.interval = float(interval)
        self.process_width = int(process_width)
        self._detector = make_detector(dict_name, refine)
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._pending: tuple[np.ndarray, float] | None = None
        self._last_submit = -math.inf
        self._running = True
        self.detect_ms = 0.0
        self.error: str | None = None
        self.image_size: tuple[int, int] | None = None
        self.corners: dict[int, np.ndarray] = {}
        self.corners_t: float | None = None
        self._thread = threading.Thread(target=self._loop, name="aruco", daemon=True)
        self._thread.start()

    def submit(self, image: np.ndarray, t: float) -> None:
        """Offer a raw (un-mirrored) camera frame; only taken every `interval` s. Never copies."""
        if t - self._last_submit < self.interval:
            return
        self._last_submit = t
        with self._cond:
            self._pending = (image, t)
            self._cond.notify()

    def latest(self) -> tuple[dict[int, np.ndarray], float | None]:
        with self._lock:
            return dict(self.corners), self.corners_t

    def _handle(self, t: float, corners: dict[int, np.ndarray]) -> None:
        pass

    def _loop(self) -> None:
        while True:
            with self._cond:
                while self._running and self._pending is None:
                    self._cond.wait(0.5)
                if not self._running:
                    return
                image, t = self._pending
                self._pending = None
            try:
                t0 = time.perf_counter()
                corners = detect_marker_corners(image, self._detector, self.mirror, self.process_width)
                self.detect_ms = (time.perf_counter() - t0) * 1000
                with self._lock:
                    self.image_size = (image.shape[1], image.shape[0])
                    self.corners, self.corners_t = corners, t
                    self._handle(t, corners)
            except Exception as exc:  # never let marker detection kill anything
                self.error = f"{type(exc).__name__}: {exc}"

    def close(self) -> None:
        with self._cond:
            self._running = False
            self._cond.notify()
        self._thread.join(timeout=1.0)


class MarkerWorker(ArucoWorker):
    """The 4 mat markers: a zone that follows them, plus the table plane they give."""

    def __init__(self, cfg: dict | None = None, mirror: bool = True):
        cfg = cfg or {}
        super().__init__(str(cfg.get("dictionary", DEFAULT_DICT)), mirror, float(cfg.get("interval", 0.25)),
                         int(cfg.get("process_width", 960)))
        ids = tuple(int(i) for i in cfg.get("ids", DEFAULT_IDS))
        self.size_cm = float(cfg.get("size_mm", 60)) / 10.0
        self.tracker = MarkerZoneTracker(ids=ids)
        self.plane = None

    def _handle(self, t: float, corners: dict[int, np.ndarray]) -> None:
        self.tracker.update(t, {i: (float(c[:, 0].mean()), float(c[:, 1].mean())) for i, c in corners.items()})
        mine = {i: c for i, c in corners.items() if i in self.tracker.ids}
        if mine and self.image_size:
            # The biggest marker in the picture gives the most precise plane.
            best = max(mine.values(), key=lambda c: _quad_area(c * self.image_size))
            try:
                self.plane = plane_from_marker(best, self.size_cm, self.image_size, "markers")
            except ValueError:
                pass

    def snapshot(self) -> tuple[Zone | None, str, dict[int, Point]]:
        with self._lock:
            tr = self.tracker
            return tr.zone, tr.status, dict(tr.seen)

    def plane_snapshot(self):
        with self._lock:
            return self.plane


def _quad_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)
