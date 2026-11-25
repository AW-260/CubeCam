"""Start-zone ("mat") geometry and the hands-mode zone builder.

All coordinates are normalised to the displayed frame: (0, 0) top-left,
(1, 1) bottom-right. When the camera view is mirrored, that is the mirrored
image, the same one the hand landmarks and mouse clicks refer to.

The zone is a polygon (a rectangle is just the 4-point case). A hand counts as
"in the zone" when its palm centre is inside the polygon or on its edge.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

Point = tuple[float, float]

# Hand-landmark indices (MediaPipe): wrist + the four finger MCP joints.
PALM_IDS = (0, 5, 9, 13, 17)

# Zone sanity thresholds (fractions of the frame).
MIN_INSIDE_FRACTION = 0.5   # less than this much of the zone in the frame -> "mostly outside"
MIN_AREA = 0.02             # visible area below 2 % of the frame -> "too small"
MIN_WIDTH = 0.12            # visible width below 12 % of the frame width -> two hands won't fit
MIN_HEIGHT = 0.05


def _area(points: Sequence[Point]) -> float:
    """Absolute polygon area (shoelace)."""
    n = len(points)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _on_segment(px: float, py: float, a: Point, b: Point, eps: float = 1e-9) -> bool:
    (ax, ay), (bx, by) = a, b
    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    if abs(cross) > eps * max(1.0, math.hypot(bx - ax, by - ay)):
        return False
    return min(ax, bx) - eps <= px <= max(ax, bx) + eps and min(ay, by) - eps <= py <= max(ay, by) + eps


def clip_to_frame(points: Sequence[Point]) -> list[Point]:
    """Sutherland-Hodgman clip of a polygon to the unit square [0, 1] x [0, 1]."""
    out = [tuple(map(float, p)) for p in points]
    # (inside test, intersection) for each of the 4 frame edges
    edges = [
        (lambda p: p[0] >= 0.0, 0, 0.0),
        (lambda p: p[0] <= 1.0, 0, 1.0),
        (lambda p: p[1] >= 0.0, 1, 0.0),
        (lambda p: p[1] <= 1.0, 1, 1.0),
    ]
    for inside, axis, value in edges:
        if not out:
            break
        src, out = out, []
        for i, cur in enumerate(src):
            prev = src[i - 1]
            cin, pin = inside(cur), inside(prev)
            if cin != pin:
                # intersection of prev->cur with the line axis == value
                d = cur[axis] - prev[axis]
                r = (value - prev[axis]) / d if d else 0.0
                q = (prev[0] + r * (cur[0] - prev[0]), prev[1] + r * (cur[1] - prev[1]))
                out.append(q)
            if cin:
                out.append(cur)
    return out


def order_convex(points: Iterable[Point]) -> list[Point]:
    """Sort points by angle around their centroid (clockwise on screen, starting top-left-ish).

    Used for marker corners, so it doesn't matter which marker went where.
    """
    pts = [tuple(map(float, p)) for p in points]
    if len(pts) < 3:
        return pts
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    # Image y points down, so increasing atan2 angle is clockwise on screen.
    pts.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    # Start at the point closest to the top-left corner for a stable order.
    k = min(range(len(pts)), key=lambda i: pts[i][0] + pts[i][1])
    return pts[k:] + pts[:k]


@dataclass(frozen=True)
class Zone:
    """Start zone as a polygon in normalised frame coordinates."""

    points: tuple[Point, ...]

    def __post_init__(self):
        pts = tuple((float(x), float(y)) for x, y in self.points)
        if len(pts) < 3:
            raise ValueError("a zone needs at least 3 points")
        if not all(math.isfinite(v) for p in pts for v in p):
            raise ValueError("zone points must be finite numbers")
        object.__setattr__(self, "points", pts)

    # ----------------------------------------------------------- construction
    @classmethod
    def rect(cls, x: float, y: float, w: float, h: float) -> "Zone":
        """Axis-aligned rectangle from its top-left corner and size."""
        if w <= 0 or h <= 0:
            raise ValueError("zone width and height must be positive")
        return cls(((x, y), (x + w, y), (x + w, y + h), (x, y + h)))

    @classmethod
    def from_corners(cls, a: Point, b: Point) -> "Zone":
        """Rectangle spanned by two opposite corners (any order), e.g. a mouse drag."""
        x1, x2 = sorted((a[0], b[0]))
        y1, y2 = sorted((a[1], b[1]))
        return cls.rect(x1, y1, max(x2 - x1, 1e-6), max(y2 - y1, 1e-6))

    @classmethod
    def from_config(cls, cfg: dict) -> "Zone":
        """`points: [[x, y], ...]` if present, else the old `x, y, w, h` rectangle."""
        pts = cfg.get("points")
        if pts:
            return cls(tuple((float(p[0]), float(p[1])) for p in pts))
        return cls.rect(float(cfg["x"]), float(cfg["y"]), float(cfg["w"]), float(cfg["h"]))

    def to_config(self, decimals: int = 4) -> list[list[float]]:
        return [[round(x, decimals), round(y, decimals)] for x, y in self.points]

    # --------------------------------------------------------------- geometry
    def contains(self, px: float, py: float) -> bool:
        """Point-in-polygon (even-odd ray casting); points on an edge count as inside."""
        pts = self.points
        n = len(pts)
        for i in range(n):
            if _on_segment(px, py, pts[i - 1], pts[i]):
                return True
        inside = False
        for i in range(n):
            (x1, y1), (x2, y2) = pts[i - 1], pts[i]
            if (y1 > py) != (y2 > py):
                x_cross = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
                if px < x_cross:
                    inside = not inside
        return inside

    @property
    def area(self) -> float:
        return _area(self.points)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return min(xs), min(ys), max(xs), max(ys)

    @property
    def is_rect(self) -> bool:
        if len(self.points) != 4:
            return False
        x1, y1, x2, y2 = self.bbox
        return all((abs(x - x1) < 1e-9 or abs(x - x2) < 1e-9) and (abs(y - y1) < 1e-9 or abs(y - y2) < 1e-9)
                   for x, y in self.points)

    def inside_fraction(self) -> float:
        """Share of the zone's area that lies inside the camera frame."""
        a = self.area
        return _area(clip_to_frame(self.points)) / a if a > 0 else 0.0

    def to_pixels(self, width: int, height: int) -> np.ndarray:
        """int32 (N, 1, 2) array for cv2.polylines / fillPoly."""
        return np.array([[(round(x * width), round(y * height))] for x, y in self.points], np.int32)

    def mirrored(self) -> "Zone":
        """The same zone in the horizontally flipped image."""
        return Zone(tuple((1.0 - x, y) for x, y in reversed(self.points)))


def zone_warnings(zone: Zone) -> list[str]:
    """Human-readable problems with a zone (empty list = looks fine)."""
    warnings = []
    frac = zone.inside_fraction()
    visible = clip_to_frame(zone.points)
    if frac < MIN_INSIDE_FRACTION:
        warnings.append(f"Zone is mostly outside the camera view ({frac:.0%} visible)")
    if len(visible) < 3:
        return warnings or ["Zone is outside the camera view"]
    vis_area = _area(visible)
    xs = [p[0] for p in visible]
    ys = [p[1] for p in visible]
    if vis_area < MIN_AREA or max(xs) - min(xs) < MIN_WIDTH or max(ys) - min(ys) < MIN_HEIGHT:
        warnings.append(f"Zone is too small ({vis_area:.1%} of the frame) - both hands may not fit")
    return warnings


# --------------------------------------------------------------- hands mode
def palm_centre(landmarks: np.ndarray) -> Point:
    lm = np.asarray(landmarks, dtype=np.float64)
    c = lm[list(PALM_IDS)].mean(axis=0)
    return float(c[0]), float(c[1])


def palm_size(landmarks: np.ndarray) -> float:
    """Wrist -> middle-finger knuckle distance, a scale for 'one hand' on screen."""
    lm = np.asarray(landmarks, dtype=np.float64)
    return float(np.hypot(*(lm[9] - lm[0])))


def zone_from_hands(hands: Sequence[np.ndarray], margin: float = 0.6, min_pad: float = 0.03) -> Zone:
    """Rectangle around both resting hands with a margin.

    The box covers every landmark of both hands (whole hands, not only the
    palms), then grows on each side by `margin` x the palm size (at least
    `min_pad` of the frame), so palms stay inside when you put your hands down
    a little differently next time. The result is clipped to the frame.
    """
    if len(hands) < 2:
        raise ValueError("need two hands")
    pts = np.vstack([np.asarray(h, dtype=np.float64)[:, :2] for h in hands])
    size = float(np.mean([palm_size(h) for h in hands]))
    pad = max(margin * size, min_pad)
    x1, y1 = pts.min(axis=0) - pad
    x2, y2 = pts.max(axis=0) + pad
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(1.0, x2), min(1.0, y2)
    return Zone.from_corners((x1, y1), (x2, y2))


@dataclass
class HandsZoneBuilder:
    """Waits for two hands held still for `hold` seconds, then proposes a zone.

    Feed it every frame with the detected hand landmarks. A hand "holds still"
    while its palm centre stays within `still_tol` (fraction of the frame
    width) of where it was when the hold started. Short tracking dropouts
    (< `grace` s) are forgiven; moving or losing a hand for longer restarts the
    countdown. Landmarks are averaged over the hold for a steadier zone.
    """

    hold: float = 2.0
    still_tol: float = 0.03
    grace: float = 0.25
    margin: float = 0.6
    min_pad: float = 0.03
    _start: float | None = None
    _anchor: list[Point] = field(default_factory=list)
    _sum: list[np.ndarray] = field(default_factory=list)
    _count: int = 0
    _last_ok: float | None = None
    progress: float = 0.0
    result: Zone | None = None
    mean_hands: list[np.ndarray] = field(default_factory=list)   # averaged landmarks of the hold
    hint: str = "Rest both hands where you want the mat and hold still"

    def reset(self) -> None:
        self._start = None
        self._anchor = []
        self._sum = []
        self._count = 0
        self._last_ok = None
        self.progress = 0.0
        self.result = None
        self.mean_hands = []
        self.hint = "Rest both hands where you want the mat and hold still"

    @staticmethod
    def _pick_two(hands: Sequence[np.ndarray]) -> list[np.ndarray]:
        """The two largest hands, left one first (stable order between frames)."""
        two = sorted(hands, key=palm_size, reverse=True)[:2]
        return sorted(two, key=lambda h: palm_centre(h)[0])

    def update(self, t: float, hands: Sequence[np.ndarray]) -> Zone | None:
        """Returns the proposed zone once (when the hold completes), else None."""
        if self.result is not None:
            return None
        if len(hands) < 2:
            if self._start is not None and self._last_ok is not None and t - self._last_ok <= self.grace:
                return None  # brief dropout: keep counting from where we were
            self._restart()
            self.hint = "Show both hands" if not hands else "Only one hand seen - show both"
            return None
        two = self._pick_two(hands)
        palms = [palm_centre(h) for h in two]
        if self._start is None or any(math.dist(p, a) > self.still_tol for p, a in zip(palms, self._anchor)):
            self._restart()
            self._start = t
            self._anchor = palms
            self.hint = "Hold still..."
        self._last_ok = t
        if not self._sum:
            self._sum = [np.zeros_like(np.asarray(h, dtype=np.float64)[:, :2]) for h in two]
        for s, h in zip(self._sum, two):
            s += np.asarray(h, dtype=np.float64)[:, :2]
        self._count += 1
        self.progress = min(1.0, (t - self._start) / self.hold) if self.hold > 0 else 1.0
        if self.progress >= 1.0:
            self.mean_hands = [s / self._count for s in self._sum]
            self.result = zone_from_hands(self.mean_hands, self.margin, self.min_pad)
            self.hint = "Zone ready"
            return self.result
        return None

    def _restart(self) -> None:
        self._start = None
        self._anchor = []
        self._sum = []
        self._count = 0
        self._last_ok = None
        self.progress = 0.0
