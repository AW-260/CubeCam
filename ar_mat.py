"""The AR start area: a holographic mat lying on the table plane, drawn in perspective.

* `Mat`: centre / angle / size in table centimetres (see plane.py for the frame),
  optionally with a drawn outline (`shape`). Its projected outline is the
  timer's start zone.
* `mat_texture`: the holographic look, one variant per timer state: a glowing
  outline, a faint grid lying on the table, light along the inside of the
  edge, small corner marks on the rectangle. It is mostly transparent, so the
  table shows through. Colours come from theme.py, shared with the on-screen
  panels. Original artwork, no brand names or logos.
* `ArMatRenderer`: warps the texture onto the projected area with
  cv2.warpPerspective and caches the result, so it only re-warps when the
  plane, the area or the state changes. Each frame it only composites:
  states cross-fade, light runs round the edge while holding, the ready state
  pulses softly, a ripple crosses the area when a solve starts or stops, and
  the area is cut away where your hands are so they appear on top of it.
* `hand_mask`: landmark-based hand shape plus skin-coloured pixels near the
  hands, with feathered edges.
"""
from __future__ import annotations

import math
import threading
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Sequence

import cv2
import numpy as np

import theme as T
from plane import TablePlane
from zone import PALM_IDS, Zone

PX_PER_CM = 16          # texture resolution
MARGIN_CM = 2.5         # transparent margin round the mat in the texture (room for the glow)
STATES = ("idle", "holding", "ready", "solving")

BONES = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
         (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17))


# ======================================================================= model
Shape = tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class Mat:
    """The start area on the table: a rectangle, or a drawn outline (`shape`).

    shape: the outline in cm relative to the centre, before rotation; the
    centre is the middle of its bounding box, which is width_cm x height_cm.
    """

    cx: float                 # centre on the table (cm)
    cy: float
    angle_deg: float = 0.0    # + = clockwise on screen
    width_cm: float = 30.0
    height_cm: float = 22.0
    shape: Shape | None = None

    @classmethod
    def from_outline(cls, pts_cm, angle_deg: float = 0.0) -> "Mat":
        """A drawn area from its outline in table cm (angle 0: the outline is used as it is)."""
        p = np.asarray(pts_cm, dtype=np.float64).reshape(-1, 2)
        lo, hi = p.min(axis=0), p.max(axis=0)
        c = (lo + hi) / 2
        local = tuple((round(float(x), 2), round(float(y), 2)) for x, y in p - c)
        return cls(round(float(c[0]), 3), round(float(c[1]), 3), angle_deg, round(float(hi[0] - lo[0]), 2),
                   round(float(hi[1] - lo[1]), 2), local)

    def _to_table(self, local: np.ndarray) -> np.ndarray:
        a = math.radians(self.angle_deg)
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        return np.asarray(local, dtype=np.float64) @ R.T + (self.cx, self.cy)

    def corners(self) -> np.ndarray:
        """Table cm of the bounding-box corners: far-left, far-right, near-right, near-left (as seen at angle 0)."""
        w, h = self.width_cm / 2, self.height_cm / 2
        return self._to_table(np.array([(-w, -h), (w, -h), (w, h), (-w, h)]))

    def outline(self) -> np.ndarray:
        """Table cm of the start area's outline (the 4 corners for a rectangle)."""
        return self._to_table(np.array(self.shape)) if self.shape else self.corners()

    def moved_to(self, cx: float, cy: float) -> "Mat":
        return replace(self, cx=float(cx), cy=float(cy))

    def rotated(self, delta_deg: float) -> "Mat":
        a = (self.angle_deg + delta_deg + 180.0) % 360.0 - 180.0
        return replace(self, angle_deg=round(a, 6))

    def image_quad(self, plane: TablePlane) -> np.ndarray | None:
        """Projected bounding-box corners in image px, or None if part of it is beyond the horizon."""
        quad, ok = plane.to_image(self.corners())
        if not ok.all():
            return None
        return quad

    def zone(self, plane: TablePlane) -> Zone | None:
        if self.image_quad(plane) is None:
            return None
        pts, ok = plane.to_image(self.outline())
        if not ok.all():
            return None
        return Zone(tuple(map(tuple, pts / np.array(plane.image_size, dtype=np.float64))))

    def to_config(self) -> dict:
        d = {"centre_cm": [round(self.cx, 3), round(self.cy, 3)], "angle_deg": round(self.angle_deg, 3),
             "width_cm": round(self.width_cm, 3), "height_cm": round(self.height_cm, 3)}
        if self.shape:
            d["shape_cm"] = [[round(x, 2), round(y, 2)] for x, y in self.shape]
        return d

    @classmethod
    def from_config(cls, d: dict | None) -> "Mat | None":
        if not d or d.get("centre_cm") is None:
            return None
        cx, cy = d["centre_cm"]
        shape = d.get("shape_cm")
        if shape:
            shape = tuple((float(x), float(y)) for x, y in shape)
            if len(shape) < 3:
                raise ValueError("shape_cm needs at least 3 points")
        return cls(float(cx), float(cy), float(d.get("angle_deg", 0.0)),
                   float(d.get("width_cm", 30.0)), float(d.get("height_cm", 22.0)), shape or None)


def mat_fits(mat: Mat, plane: TablePlane, max_extent: float = 3.0) -> bool:
    """All corners visible (in front of the horizon) and not absurdly far outside the frame."""
    quad = mat.image_quad(plane)
    if quad is None:
        return False
    n = quad / np.array(plane.image_size, dtype=np.float64)
    return bool(np.all(n > -max_extent) and np.all(n < 1 + max_extent))


def mat_at_image_point(plane: TablePlane, xy_norm, width_cm: float, height_cm: float,
                       angle_deg: float = 0.0) -> Mat | None:
    """A mat centred on the table point under an image point (normalised)."""
    p, ok = plane.to_table_norm([xy_norm])
    if not ok[0]:
        return None
    mat = Mat(float(p[0, 0]), float(p[0, 1]), angle_deg, width_cm, height_cm)
    return mat if mat_fits(mat, plane) else None


def mat_from_hands(plane: TablePlane, hands_norm: Sequence[np.ndarray], width_cm: float, height_cm: float,
                   margin_cm: float = 6.0, max_angle: float = 40.0) -> Mat | None:
    """A mat on the table centred between both resting palms, turned to follow the line between them.

    It keeps the configured size unless the hands are further apart than that
    allows, in which case it gets wider (palm distance + 2 x margin_cm).
    """
    if len(hands_norm) < 2:
        return None
    palms = [np.asarray(h, dtype=np.float64)[list(PALM_IDS)].mean(axis=0) for h in hands_norm[:2]]
    tp, ok = plane.to_table_norm(palms)
    if not ok.all():
        return None
    left, right = sorted(tp, key=lambda p: p[0])
    d = right - left
    angle = math.degrees(math.atan2(d[1], d[0]))
    angle = max(-max_angle, min(max_angle, angle))
    width = max(width_cm, float(np.hypot(*d)) + 2 * margin_cm)
    centre = (left + right) / 2
    mat = Mat(float(centre[0]), float(centre[1]), round(angle, 2), round(width, 2), float(height_cm))
    return mat if mat_fits(mat, plane) else None


def reanchor(mat: Mat, old: TablePlane, new: TablePlane) -> Mat | None:
    """After re-calibrating: keep the mat where it was on screen (centre and direction)."""
    c_img, ok = old.to_image([[mat.cx, mat.cy]])
    a = math.radians(mat.angle_deg)
    tip_img, ok2 = old.to_image([[mat.cx + 5 * math.cos(a), mat.cy + 5 * math.sin(a)]])
    if not (ok[0] and ok2[0]):
        return None
    pts, ok3 = new.to_table(np.vstack([c_img, tip_img]).astype(np.float64) * (
        np.array(new.image_size, dtype=np.float64) / np.array(old.image_size, dtype=np.float64)))
    if not ok3.all():
        return None
    d = pts[1] - pts[0]
    out = replace(mat, cx=float(pts[0, 0]), cy=float(pts[0, 1]), angle_deg=round(math.degrees(math.atan2(d[1], d[0])), 3))
    return out if mat_fits(out, new) else None


# ===================================================================== texture
CORNER_CM = 2.0         # corner radius of the rectangular mat
CORE_CM = 0.22          # outline width
GRID_CM = 2.5           # grid spacing
MAX_TEX_PX = 1600       # big drawn areas use fewer texture px per cm


def rounded_rect_points(x0, y0, x1, y1, r, per_corner: int = 6) -> np.ndarray:
    """Outline of a rounded rectangle as a polygon (clockwise on screen)."""
    r = min(r, (x1 - x0) / 2, (y1 - y0) / 2)
    pts = []
    for cx, cy, a0 in ((x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)):
        for a in np.linspace(a0, a0 + 90, per_corner):
            pts.append((cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a))))
    return np.array(pts)


def _poly_mask(hw, poly: np.ndarray, thickness: float = 0.0, closed: bool = True, ss: int = 4) -> np.ndarray:
    """Anti-aliased filled polygon (thickness 0) or polyline, float32 0..1, via supersampling."""
    h, w = hw
    big = np.zeros((h * ss, w * ss), np.uint8)
    pts = np.int32(np.round(np.asarray(poly, dtype=np.float64) * ss * 16))
    if thickness > 0:
        cv2.polylines(big, [pts], closed, 255, max(1, int(round(thickness * ss))), cv2.LINE_AA, shift=4)
    else:
        cv2.fillPoly(big, [pts], 255, cv2.LINE_AA, shift=4)
    return cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def _shape_key(shape) -> tuple | None:
    return None if not shape else tuple((round(float(x), 2), round(float(y), 2)) for x, y in shape)


_TEX_BASE: "OrderedDict[tuple, dict]" = OrderedDict()
_TEX: "OrderedDict[tuple, tuple[np.ndarray, tuple]]" = OrderedDict()
_TEX_LOCK = threading.RLock()   # textures are also built by the renderer's helper thread


@dataclass(frozen=True)
class MatLayout:
    """Where the area is in its texture (texture pixels)."""

    rect: tuple[float, float, float, float]          # the bounding box (x0, y0, x1, y1)
    size: tuple[int, int]                              # texture (w, h)
    ppc: int                                           # texture px per cm
    outline: np.ndarray = field(compare=False, repr=False)   # (N, 2) the area's outline
    rounded: bool = True                               # the default rectangle (rounded corners, corner marks)


def texture_ppc(w_cm: float, h_cm: float, ppc: int = PX_PER_CM) -> int:
    return int(max(4, min(ppc, MAX_TEX_PX // (max(w_cm, h_cm) + 2 * MARGIN_CM))))


def mat_layout(w_cm: float, h_cm: float, ppc: int = PX_PER_CM, shape: Shape | None = None) -> MatLayout:
    ppc = texture_ppc(w_cm, h_cm, ppc)
    m = MARGIN_CM * ppc
    W, H = int(round((w_cm + 2 * MARGIN_CM) * ppc)), int(round((h_cm + 2 * MARGIN_CM) * ppc))
    x0, y0, x1, y1 = m, m, W - m, H - m
    if shape:
        outline = (chaikin(np.asarray(shape, dtype=np.float64)) + (w_cm / 2, h_cm / 2)) * ppc + m
    else:
        outline = rounded_rect_points(x0, y0, x1, y1, CORNER_CM * ppc, per_corner=12)
    return MatLayout((x0, y0, x1, y1), (W, H), ppc, outline, not shape)


def chaikin(poly: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Corner cutting: a closed polygon drawn as a smooth curve (for display; the zone keeps the polygon)."""
    p = np.asarray(poly, dtype=np.float64)
    for _ in range(iterations):
        q = np.roll(p, -1, axis=0)
        p = np.stack([0.75 * p + 0.25 * q, 0.25 * p + 0.75 * q], axis=1).reshape(-1, 2)
    return p


def _arc_map(hw, outline: np.ndarray) -> np.ndarray:
    """For every texel: how far round the outline its nearest outline point is, 0..1.

    0 at the middle of the near edge (closest to you), 1 at the opposite
    point, counted both ways round, so a fill up to `progress` grows round
    both sides and meets at the far side.
    """
    h, w = hw
    closed = np.vstack([outline, outline[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    perim = float(cum[-1])
    n = max(16, int(perim / 0.7))
    t = np.linspace(0.0, perim, n, endpoint=False)
    pts = np.column_stack([np.interp(t, cum, closed[:, 0]), np.interp(t, cum, closed[:, 1])])
    span = pts[:, 1].max() - pts[:, 1].min()
    near = pts[:, 1] >= pts[:, 1].max() - 0.01 * span
    cx = pts[:, 0].mean()
    start = int(np.flatnonzero(near)[np.argmin(np.abs(pts[near, 0] - cx))])
    arc = (t - t[start]) % perim
    frac = (np.minimum(arc, perim - arc) / (perim / 2)).astype(np.float32)
    xs = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
    seeds = np.ones((h, w), np.uint8)
    seeds[ys, xs] = 0
    _, labels = cv2.distanceTransformWithLabels(seeds, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    lut = np.zeros(int(labels.max()) + 1, np.float32)
    lut[labels[ys, xs]] = frac
    return lut[labels]


def _base_layers(w_cm: float, h_cm: float, ppc: int, shape: Shape | None = None) -> dict:
    """State-independent layers of the area (cached per size and shape)."""
    key = (round(w_cm, 2), round(h_cm, 2), ppc, _shape_key(shape))
    with _TEX_LOCK:
        if key in _TEX_BASE:
            _TEX_BASE.move_to_end(key)
            return _TEX_BASE[key]
        out = _make_base_layers(w_cm, h_cm, ppc, shape)
        _TEX_BASE[key] = out
        while len(_TEX_BASE) > 6:
            _TEX_BASE.popitem(last=False)
        return out


def _make_base_layers(w_cm, h_cm, ppc, shape) -> dict:
    L = mat_layout(w_cm, h_cm, ppc, shape)
    p = L.ppc
    W, H = L.size
    hw = (H, W)
    inside = _poly_mask(hw, L.outline)
    core = _poly_mask(hw, L.outline, thickness=CORE_CM * p)
    glow = np.clip(cv2.GaussianBlur(core, (0, 0), 0.45 * p) * 2.2, 0, 1)
    haze = np.clip(cv2.GaussianBlur(core, (0, 0), 1.3 * p) * 1.8, 0, 1)
    d_in = cv2.distanceTransform((inside > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    edge_light = np.exp(-d_in / (1.2 * p)) * inside                  # light just inside the outline
    # the faint grid, aligned with the area, a little stronger towards the outline
    x0, y0, x1, y1 = L.rect
    gx, gy = (x0 + x1) / 2, (y0 + y1) / 2
    step = GRID_CM * p
    xs = gx + step * np.arange(-int(W / step) - 1, int(W / step) + 2)
    ys = gy + step * np.arange(-int(H / step) - 1, int(H / step) + 2)
    lines = np.zeros(hw, np.uint8)
    for x in xs:
        cv2.line(lines, (int(round(x * 16)), 0), (int(round(x * 16)), H * 16), 255, 1, cv2.LINE_AA, shift=4)
    for y in ys:
        cv2.line(lines, (0, int(round(y * 16))), (W * 16, int(round(y * 16))), 255, 1, cv2.LINE_AA, shift=4)
    dots = np.zeros(hw, np.uint8)
    r_dot = max(1, int(0.09 * p * 16))
    for x in xs:
        for y in ys:
            cv2.circle(dots, (int(round(x * 16)), int(round(y * 16))), r_dot, 255, -1, cv2.LINE_AA, shift=4)
    grid = np.maximum(lines.astype(np.float32) * (0.75 / 255.0), dots.astype(np.float32) / 255.0)
    fade = 0.45 + 0.55 * np.exp(-d_in / (5.0 * p))
    grid *= inside * fade * np.clip(d_in / (0.6 * p), 0, 1)           # keep clear of the outline itself
    marks = None
    if L.rounded:
        # small corner marks just outside the rectangle, like the frame in a viewfinder
        o, arm = 0.55 * p, 2.2 * p
        segs = []
        for cx, cy, sx, sy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x1, y1, -1, -1), (x0, y1, 1, -1)):
            c = np.array([cx - sx * o, cy - sy * o])
            segs.append(np.array([c + (sx * arm, 0), c, c + (0, sy * arm)]))
        marks = np.maximum.reduce([_poly_mask(hw, sg, thickness=0.12 * p, closed=False) for sg in segs])
    # maps for the per-frame light effects
    edge = np.clip(core + 0.9 * glow + 0.45 * haze, 0, 1)
    arc = _arc_map(hw, L.outline)
    m = cv2.moments((inside > 0.5).astype(np.uint8), binaryImage=True)
    c = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]]) if m["m00"] else np.array([gx, gy])
    rmax = float(np.linalg.norm(L.outline - c, axis=1).max())
    Y, X = np.mgrid[0:H, 0:W].astype(np.float32)
    radial = (np.sqrt((X - c[0]) ** 2 + (Y - c[1]) ** 2) / max(rmax, 1.0)).astype(np.float32)
    return dict(layout=L, inside=inside, core=core, glow=glow, haze=haze, edge_light=edge_light, grid=grid,
                marks=marks, edge=edge, arc=arc, radial=radial)


def _mix(a, b, t: float) -> np.ndarray:
    return np.array(a, np.float32) * (1 - t) + np.array(b, np.float32) * t


def mat_texture(w_cm: float, h_cm: float, state: str = "idle", progress: float = 0.0,
                ppc: int = PX_PER_CM, shape: Shape | None = None) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """(BGRA uint8 texture, bounding box (x0, y0, x1, y1) of the area inside it, in texture px).

    The holding light is normally drawn per frame by ArMatRenderer (smooth);
    a progress > 0 here bakes it in (for still pictures).
    """
    with _TEX_LOCK:
        return _mat_texture(w_cm, h_cm, state, progress, ppc, shape)


def _mat_texture(w_cm, h_cm, state, progress, ppc, shape):
    if state not in STATES:
        state = "idle"
    level = round(min(max(progress, 0.0), 1.0), 3) if state == "holding" else 0.0
    key = (round(w_cm, 2), round(h_cm, 2), ppc, _shape_key(shape), state, level)
    if key in _TEX:
        _TEX.move_to_end(key)
        return _TEX[key]
    b = _base_layers(w_cm, h_cm, ppc, shape)
    lv = T.HOLO[state]
    col = np.array(T.HOLO_COLOR[state], np.float32)
    H, W = b["inside"].shape
    P = np.zeros((H, W, 3), np.float32)          # premultiplied colour
    A = np.zeros((H, W), np.float32)

    def over(color, alpha):
        nonlocal P, A
        a = np.clip(alpha, 0, 1).astype(np.float32)
        P = P * (1 - a)[..., None] + np.asarray(color, np.float32) * a[..., None]
        A = a + A * (1 - a)

    over(T.PANEL, b["inside"] * lv["fill"])
    over(col, b["edge_light"] * lv["edge"])
    over(col, b["haze"] * lv["haze"])
    over(col, b["grid"] * lv["grid"])
    over(col, b["glow"] * lv["glow"])
    if b["marks"] is not None:
        over(col, b["marks"] * lv["core"] * 0.75)
    over(_mix(col, T.TEXT, T.HOT_MIX), b["core"] * lv["core"])
    if level > 0:
        over(hold_color(), b["edge"] * np.clip((level * 1.04 - b["arc"]) / 0.04, 0, 1) * T.HOLD_LIGHT)
    color = P / np.maximum(A, 1e-4)[..., None]
    tex = np.dstack([np.clip(color, 0, 255), np.clip(A * 255, 0, 255)]).astype(np.uint8)
    out = (tex, b["layout"].rect)
    _TEX[key] = out
    while len(_TEX) > 48:
        _TEX.popitem(last=False)
    return out


def hold_color() -> tuple[int, int, int]:
    """The light that runs round the outline while holding: amber, a little brighter."""
    return tuple(int(v) for v in np.round(_mix(T.STATE["holding"], T.TEXT, 0.35)))


# =================================================================== hand mask
SKIN_LO = np.array([0, 135, 85], np.uint8)      # YCrCb
SKIN_HI = np.array([255, 178, 135], np.uint8)


def hand_mask(roi_bgr: np.ndarray, hands_px: Sequence[np.ndarray], skin: bool = True,
              feather: float | None = None, work_palm_px: float = 32.0,
              background: np.ndarray | None = None) -> np.ndarray | None:
    """Where the hands are inside an image region: float32 0..1 (feathered), or None if no hand overlaps.

    hands_px are (21, 2) landmark arrays in the region's pixel coordinates.
    The shape comes from the landmarks (palm polygon, finger bones of a width
    proportional to the hand, a stub of the forearm), plus skin-coloured pixels
    close to them, so the real outline of fingers and the wrist is followed.

    A skin-coloured pixel only counts if it also differs from the table
    there, because a wooden desk is often "skin-coloured" too: compared with
    `background` (a clean picture of the region without hands) when given,
    otherwise with the table's typical colour away from the hands.
    Big hands are processed at reduced resolution (palm ~work_palm_px pixels)
    and scaled back up; the edges are feathered anyway.
    """
    h, w = roi_bgr.shape[:2]
    hands = []
    for lm in hands_px:
        lm = np.asarray(lm, dtype=np.float64)
        s = float(np.linalg.norm(lm[9] - lm[0]))
        lo, hi = lm.min(axis=0) - 2 * s, lm.max(axis=0) + 2 * s
        if s >= 2 and not (hi[0] < 0 or hi[1] < 0 or lo[0] > w or lo[1] > h):
            hands.append((lm, s))
    if not hands:
        return None
    s_full = float(np.mean([s for _, s in hands]))
    k = min(1.0, work_palm_px / s_full) if work_palm_px else 1.0
    bg = background if background is not None and background.shape == roi_bgr.shape else None
    if k < 0.95:
        sw, sh = max(1, int(round(w * k))), max(1, int(round(h * k)))
        kx, ky = sw / w, sh / h
        small = cv2.resize(roi_bgr, (sw, sh), interpolation=cv2.INTER_AREA) if skin else None
        if skin and bg is not None:
            bg = cv2.resize(bg, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        sw, sh, kx, ky, small = w, h, 1.0, 1.0, roi_bgr
    mask = np.zeros((sh, sw), np.uint8)
    for lm, _ in hands:
        p = lm * (kx, ky)
        s = float(np.linalg.norm(p[9] - p[0]))
        pts = np.int32(np.round(p * 4))  # 2 bits of sub-pixel precision
        cv2.fillConvexPoly(mask, cv2.convexHull(pts[[0, 1, 2, 5, 9, 13, 17]]), 255, cv2.LINE_AA, shift=2)
        thick = max(2, int(round(0.24 * s)))   # real fingers are ~0.2 x palm length wide
        for a, b in BONES:
            cv2.line(mask, tuple(pts[a]), tuple(pts[b]), 255, thick, cv2.LINE_AA, shift=2)
        for i in (4, 8, 12, 16, 20):
            cv2.circle(mask, tuple(pts[i]), int(round(0.12 * s * 4)), 255, -1, cv2.LINE_AA, shift=2)
        # forearm stub: from the wrist away from the fingers
        away = p[0] - p[9]
        away /= max(np.linalg.norm(away), 1e-6)
        cv2.line(mask, tuple(pts[0]), tuple(np.int32(np.round((p[0] + away * 1.3 * s) * 4))), 255,
                 max(2, int(round(0.8 * s))), cv2.LINE_AA, shift=2)
    s = s_full * kx
    if skin:
        # "near the hands": within ~half a palm of the landmark shape (a distance
        # transform is much cheaper than dilating with a big kernel)
        dist = cv2.distanceTransform(cv2.bitwise_not(mask), cv2.DIST_L2, 3)
        near = (dist < 0.45 * s).astype(np.uint8) * 255
        ys, xs = np.nonzero(near)
        if len(xs):
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            ycc = cv2.cvtColor(small[y0:y1, x0:x1], cv2.COLOR_BGR2YCrCb)
            sk = cv2.inRange(ycc, SKIN_LO, SKIN_HI)
            if bg is not None:
                # must differ from the clean picture of the table (any channel, > ~25 levels)
                diff = cv2.absdiff(small[y0:y1, x0:x1], bg[y0:y1, x0:x1]).max(axis=2)
                sk = cv2.bitwise_and(sk, (diff > 25).astype(np.uint8) * 255)
            else:
                # no clean picture yet: must differ from the table's colour away from the hands
                far = dist > 1.2 * s
                if far.sum() > 50:
                    full_ycc = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
                    cr, cb = np.median(full_ycc[..., 1][far]), np.median(full_ycc[..., 2][far])
                    d = np.abs(ycc[..., 1].astype(np.int16) - cr) + np.abs(ycc[..., 2].astype(np.int16) - cb)
                    sk = cv2.bitwise_and(sk, (d > 12).astype(np.uint8) * 255)
            sk = cv2.morphologyEx(sk, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            sub = mask[y0:y1, x0:x1]
            np.maximum(sub, cv2.bitwise_and(sk, near[y0:y1, x0:x1]), out=sub)
    sigma = (feather if feather is not None else max(1.0, 0.03 * s_full)) * kx
    soft = cv2.GaussianBlur(mask, (0, 0), max(0.5, sigma))
    if (sw, sh) != (w, h):
        soft = cv2.resize(soft, (w, h), interpolation=cv2.INTER_LINEAR)
    return soft.astype(np.float32) / 255.0


# ==================================================================== renderer
@dataclass
class _Warp:
    x0: int
    y0: int
    color: np.ndarray      # uint8 (h, w, 3), not premultiplied
    alpha: np.ndarray      # float32 (h, w), 0..1
    inv_alpha: np.ndarray  # float32 (h, w), 1 - alpha


RADIAL_STEPS = 200      # radial map: 8-bit levels per area radius (0..255 covers 0..1.27)


@dataclass
class _Lights:
    """State-independent maps for the per-frame light effects, warped and cropped like the _Warp.

    arc and radial are 8-bit, so each frame turns them into light with one
    cv2.LUT pass (a 256-entry table for the current progress / ripple radius)
    instead of several float operations over the whole area.
    """

    x0: int
    y0: int
    edge: np.ndarray       # float32: where the outline's light lives (0..1)
    arc: np.ndarray        # uint8: how far round the outline, 0 (near edge middle) .. 255 (far side)
    radial: np.ndarray     # uint8: distance from the area's centre x RADIAL_STEPS (1 = furthest outline point)
    ripple: np.ndarray     # float32: what a ripple lights up (the area, a little more on the outline)


def _warp_geometry(quad, rect, tex_wh, frame_wh):
    """(shrink factor, shrunk texture size, texture->ROI matrix, ROI box) or None if off-screen."""
    x0, y0, x1, y1 = rect
    src = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    H_full = cv2.getPerspectiveTransform(src, np.float32(quad)).astype(np.float64)
    # pre-shrink the texture when the area is small on screen (avoids aliasing)
    on_screen = max(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3]))
    f = 1.0
    while on_screen / ((x1 - x0) * f) < 0.6 and f > 0.2:
        f *= 0.5
    tw, th = tex_wh
    if f < 1.0:
        tw, th = max(2, int(tw * f)), max(2, int(th * f))
    Hq = H_full @ np.diag([1 / f, 1 / f, 1.0])
    outer = cv2.perspectiveTransform(np.float32([[[0, 0]], [[tw, 0]], [[tw, th]], [[0, th]]]), Hq).reshape(-1, 2)
    W, H = frame_wh
    bx0, by0 = int(max(0, np.floor(outer[:, 0].min()))), int(max(0, np.floor(outer[:, 1].min())))
    bx1, by1 = int(min(W, np.ceil(outer[:, 0].max()) + 1)), int(min(H, np.ceil(outer[:, 1].max()) + 1))
    if bx1 <= bx0 or by1 <= by0:
        return None
    Tr = np.array([[1, 0, -bx0], [0, 1, -by0], [0, 0, 1]], np.float64)
    return f, (tw, th), Tr @ Hq, (bx0, by0, bx1, by1)


def warp_mat(quad: np.ndarray, size_cm: tuple[float, float], state: str, level: float,
             frame_wh: tuple[int, int], ppc: int = PX_PER_CM, shape: Shape | None = None) -> _Warp | None:
    """Warp the area's texture onto the image quad (its bounding box: far-left, far-right, near-right, near-left)."""
    tex, rect = mat_texture(size_cm[0], size_cm[1], state, level, ppc, shape)
    g = _warp_geometry(np.asarray(quad, dtype=np.float64), rect, (tex.shape[1], tex.shape[0]), frame_wh)
    if g is None:
        return None
    f, (tw, th), M, (bx0, by0, bx1, by1) = g
    if f < 1.0:
        tex = cv2.resize(tex, (tw, th), interpolation=cv2.INTER_AREA)
    # premultiply first so the edges blend correctly after interpolation
    a = tex[..., 3:4].astype(np.float32) / 255.0
    pm = np.dstack([tex[..., :3].astype(np.float32) * a, a[..., 0] * 255.0])
    warped = cv2.warpPerspective(pm, M, (bx1 - bx0, by1 - by0), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    alpha = np.ascontiguousarray(warped[..., 3] / 255.0, dtype=np.float32)
    color = np.clip(warped[..., :3] / np.maximum(alpha, 1e-4)[..., None], 0, 255).astype(np.uint8)
    return _Warp(bx0, by0, color, alpha, np.ascontiguousarray(1.0 - alpha))


def warp_lights(quad: np.ndarray, size_cm: tuple[float, float], frame_wh: tuple[int, int],
                ppc: int = PX_PER_CM, shape: Shape | None = None) -> _Lights | None:
    """Warp the light-effect maps exactly like warp_mat warps the texture."""
    b = _base_layers(size_cm[0], size_cm[1], ppc, shape)
    stack = np.dstack([b["edge"], b["arc"], b["radial"], b["inside"]]).astype(np.float32)
    g = _warp_geometry(np.asarray(quad, dtype=np.float64), b["layout"].rect, (stack.shape[1], stack.shape[0]),
                       frame_wh)
    if g is None:
        return None
    f, (tw, th), M, (bx0, by0, bx1, by1) = g
    if f < 1.0:
        stack = cv2.resize(stack, (tw, th), interpolation=cv2.INTER_AREA)
    warped = cv2.warpPerspective(stack, M, (bx1 - bx0, by1 - by0), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    edge = np.ascontiguousarray(warped[..., 0])
    arc = np.clip(np.round(warped[..., 1] * 255), 0, 255).astype(np.uint8)
    radial = np.clip(np.round(warped[..., 2] * RADIAL_STEPS), 0, 255).astype(np.uint8)
    ripple = np.ascontiguousarray(warped[..., 3] * 0.6 + edge * 0.4, dtype=np.float32)
    return _Lights(bx0, by0, edge, arc, radial, ripple)


_LEVELS = np.arange(256, dtype=np.float32) / 255.0     # the 8-bit map levels as 0..1


def _approach(cur: float, target: float, step: float) -> float:
    return min(target, cur + step) if cur < target else max(target, cur - step)


class ArMatRenderer:
    """Draws the holographic start area into frames.

    Warps are cached per (quad, size, shape, state). When an area position is
    first seen, the current state is warped right away (a few ms) and the
    other states, plus the light-effect maps, are prepared in a background
    thread, so a later state change never stalls the frame loop. Each frame
    then only composites:
      * state changes cross-fade over theme.FADE_S (no hard switch);
      * holding: light runs round the outline as the hold progresses;
      * ready: a soft pulse of the outline's glow;
      * solve start / stop: a ripple spreads across the area;
      * the area is cut away under the hands.
    With animate=False (still pictures) there are no fades, pulses or ripples.
    """

    def __init__(self, ppc: int = PX_PER_CM, cache_size: int = 40, background: bool = True, animate: bool = True):
        self.ppc = ppc
        self.cache_size = cache_size
        self.background = background
        self.animate = animate
        self._cache: "OrderedDict[tuple, object]" = OrderedDict()
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._todo: list[tuple] = []
        self._running = True
        self._thread = None
        self.warps = 0          # how many warps were made (tests / profiling)
        self.last_ms = 0.0
        self._bg: np.ndarray | None = None     # clean picture of the area's region (no hands)
        self._bg_box: tuple | None = None
        self._shown: tuple | None = None       # (quad key, state, warp) drawn last frame
        self._fade: tuple | None = None        # (previous warp, start time) while cross-fading
        self._progress = 0.0
        self._state: str | None = None         # the state asked for last frame
        self._last_now: float | None = None
        self._hold_vis = 0.0                   # light effects fade in / out with their state
        self._ready_vis = 0.0
        self._ready_t0 = 0.0
        self._ripple: tuple | None = None      # (start time, colour)
        self._solid: dict = {}

    # ----------------------------------------------------------------- cache
    @staticmethod
    def _qkey(quad, size_cm, frame_wh, shape=None):
        return (tuple(np.round(np.asarray(quad).ravel() * 4).astype(int)), round(size_cm[0], 2),
                round(size_cm[1], 2), tuple(frame_wh), _shape_key(shape))

    def _store(self, key, value) -> None:
        with self._lock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
            self.warps += 1

    def _get(self, key):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return True, self._cache[key]
            return False, None

    def _make(self, job):
        quad, size_cm, what, frame_wh, shape = job
        if what == "lights":
            return warp_lights(quad, size_cm, frame_wh, self.ppc, shape)
        return warp_mat(quad, size_cm, what, 0.0, frame_wh, self.ppc, shape)

    def _queue_rest(self, quad, size_cm, frame_wh, shape) -> None:
        jobs = [(quad.copy(), size_cm, what, frame_wh, shape) for what in STATES + ("lights",)]
        if not self.background:
            for job in jobs:
                k = self._qkey(job[0], size_cm, frame_wh, shape) + (job[2],)
                if not self._get(k)[0]:
                    self._store(k, self._make(job))
            return
        with self._cond:
            self._todo = jobs          # newest area position first; drop work for older positions
            if self._thread is None:
                self._thread = threading.Thread(target=self._loop, name="ar-mat", daemon=True)
                self._thread.start()
            self._cond.notify()

    def _loop(self) -> None:
        while True:
            with self._cond:
                while self._running and not self._todo:
                    self._cond.wait(0.5)
                if not self._running:
                    return
                job = self._todo.pop(0)
            k = self._qkey(job[0], job[1], job[3], job[4]) + (job[2],)
            if self._get(k)[0]:
                continue
            try:
                self._store(k, self._make(job))
            except Exception:  # pragma: no cover - never let the helper thread die on a bad quad
                pass

    def warp_for(self, quad, size_cm, state, frame_wh, shape=None) -> tuple[_Warp | None, str]:
        """The warp to show for this state (or the idle one while it's being prepared)."""
        if state not in STATES:
            state = "idle"
        qk = self._qkey(quad, size_cm, frame_wh, shape)
        found, wp = self._get(qk + (state,))
        if found:
            return wp, state
        found_idle, idle = self._get(qk + ("idle",))
        if found_idle and self.background:
            self._queue_rest(quad, size_cm, frame_wh, shape)
            return idle, "idle"
        wp = warp_mat(quad, size_cm, state, 0.0, frame_wh, self.ppc, shape)
        self._store(qk + (state,), wp)
        self._queue_rest(quad, size_cm, frame_wh, shape)
        return wp, state

    def lights_for(self, quad, size_cm, frame_wh, shape=None) -> _Lights | None:
        """The light-effect maps, or None while the background thread prepares them."""
        k = self._qkey(quad, size_cm, frame_wh, shape) + ("lights",)
        found, lt = self._get(k)
        if found:
            return lt
        job = (np.asarray(quad, dtype=np.float64).copy(), size_cm, "lights", frame_wh, shape)
        if not self.background:
            lt = self._make(job)
            self._store(k, lt)
            return lt
        with self._cond:
            if not any(j[2] == "lights" and self._qkey(j[0], j[1], j[3], j[4]) == k[:-1] for j in self._todo):
                self._todo.append(job)
                self._cond.notify()
        return None

    def wait_idle(self, timeout: float = 10.0) -> None:
        """Block until the background thread has nothing left to do (tests, screenshots)."""
        import time

        end = time.perf_counter() + timeout
        while time.perf_counter() < end:
            with self._lock:
                if not self._todo:
                    break
            time.sleep(0.01)
        time.sleep(0.05)

    def close(self) -> None:
        with self._cond:
            self._running = False
            self._cond.notify()

    # ------------------------------------------------------------------ draw
    def _solid_like(self, color, shape) -> np.ndarray:
        key = (tuple(int(v) for v in color), shape)
        img = self._solid.get(key)
        if img is None:
            if len(self._solid) > 8:
                self._solid.clear()
            img = np.empty(shape, np.uint8)
            img[:] = key[0]
            self._solid[key] = img
        return img

    def _light_layers(self, lt: _Lights | None, now: float) -> list[tuple[np.ndarray, tuple]]:
        """This frame's light effects as (alpha map, colour), in the ROI."""
        out = []
        if lt is None:
            return out
        if self._hold_vis > 0.001 and self._progress > 0.002:
            # light runs round the outline from the near edge, both ways, meeting at the far side:
            # the lit part plus a brighter moving head, as a table over "how far round"
            d = np.float32(self._progress * 1.04) - _LEVELS
            head = 0.35 * min(1.0, (1.0 - self._progress) * 8.0)
            table = np.minimum(np.clip(d * 25.0, 0, 1) * 0.8 + np.clip(1.0 - np.abs(d) * 14.0, 0, 1) * head, 1.0)
            light = cv2.LUT(lt.arc, table.astype(np.float32))
            out.append((cv2.multiply(lt.edge, light, scale=T.HOLD_LIGHT * self._hold_vis), hold_color()))
        if self._ready_vis > 0.001:
            ph = (now - self._ready_t0) / T.PULSE_S
            pulse = 0.5 - 0.5 * math.cos(2 * math.pi * ph) if self.animate else 0.5
            level = (T.PULSE_LOW + (T.PULSE_HIGH - T.PULSE_LOW) * pulse) * self._ready_vis
            out.append((lt.edge * np.float32(level), T.STATE["ready"]))
        if self._ripple is not None:
            u = (now - self._ripple[0]) / T.RIPPLE_S
            if 0.0 <= u < 1.0:
                radius = 1.2 * (1.0 - (1.0 - u) ** 2)               # fast at first, then easing out
                env = min(1.0, u / 0.08) * (1.0 - u) ** 1.3          # fades in quickly, out slowly
                d = _LEVELS * (255.0 / RADIAL_STEPS) - radius           # radial level -> distance to the ring
                ring = np.minimum(np.clip(1.0 - d / T.RIPPLE_FRONT, 0, 1), np.clip(1.0 + d / T.RIPPLE_TAIL, 0, 1))
                ring = cv2.LUT(lt.radial, (ring * ring).astype(np.float32))
                out.append((cv2.multiply(lt.ripple, ring, scale=T.RIPPLE_ALPHA * env), self._ripple[1]))
        return out

    def _advance(self, state: str, progress: float, now: float, same_area: bool) -> None:
        """Track state changes: start ripples, ease the light effects in and out."""
        dt = 0.0 if self._last_now is None else min(0.2, max(0.0, now - self._last_now))
        self._last_now = now
        prev = self._state
        if not same_area:
            self._ripple = None
        elif self.animate and prev is not None and state != prev:
            if state == "solving":
                self._ripple = (now, T.STATE["solving"])        # solve start
            elif prev == "solving":
                self._ripple = (now, T.TEXT)                    # solve stop
        if self._ripple is not None and now - self._ripple[0] >= T.RIPPLE_S:
            self._ripple = None
        if state == "ready" and prev != "ready":
            self._ready_t0 = now
        self._state = state
        step = dt / T.FADE_S if self.animate else 1.0
        self._hold_vis = _approach(self._hold_vis, 1.0 if state == "holding" else 0.0, step)
        self._ready_vis = _approach(self._ready_vis, 1.0 if state == "ready" else 0.0, step)
        if state == "holding":
            # the hold progress eases a little so it never jumps (it can drop back at once)
            target = min(max(progress, 0.0), 1.0)
            self._progress = target if (not self.animate or target < self._progress) else \
                self._progress + (target - self._progress) * 0.6
        elif self._hold_vis <= 0.0:
            self._progress = 0.0

    def draw(self, img: np.ndarray, quad_px: np.ndarray, size_cm: tuple[float, float], state: str = "idle",
             progress: float = 0.0, hands_px: Sequence[np.ndarray] = (), occlude: bool = True,
             shape: Shape | None = None, now: float | None = None) -> None:
        """Composite the area into img (in place).

        quad_px: the image corners of the area's bounding box (far-left,
        far-right, near-right, near-left); shape: a drawn outline (cm, see
        Mat.shape), None for the rectangle; hands_px: (21, 2) landmark arrays in
        image px; now: the time (perf_counter seconds) for the animations.
        """
        import time

        t0 = time.perf_counter()
        now = t0 if now is None else float(now)
        if state not in STATES:
            state = "idle"
        quad = np.asarray(quad_px, dtype=np.float64).reshape(4, 2)
        h, w = img.shape[:2]
        size = (float(size_cm[0]), float(size_cm[1]))
        shape = _shape_key(shape)
        wp, shown_state = self.warp_for(quad, size, state, (w, h), shape)
        if wp is None:
            return
        qk = self._qkey(quad, size, (w, h), shape)
        same_area = self._shown is not None and self._shown[0] == qk
        # start a cross-fade when the state changes on the same area
        if same_area and self._shown[1] != shown_state and self.animate:
            prev = self._shown[2]
            if prev is not None and prev.alpha.shape == wp.alpha.shape and (prev.x0, prev.y0) == (wp.x0, wp.y0):
                self._fade = (prev, now)
        elif not same_area:
            self._fade = None
        self._shown = (qk, shown_state, wp)
        self._advance(state, progress, now, same_area)

        rh, rw = wp.alpha.shape
        roi = img[wp.y0:wp.y0 + rh, wp.x0:wp.x0 + rw]
        keep = None
        if occlude and len(hands_px):
            local = [np.asarray(lm, dtype=np.float64) - (wp.x0, wp.y0) for lm in hands_px]
            m = hand_mask(roi, local, background=self._bg if self._bg_box == (wp.x0, wp.y0, rw, rh) else None)
            if m is not None:
                keep = cv2.subtract(1.0, m)
        if occlude and keep is None:
            self._learn_background(roi, (wp.x0, wp.y0, rw, rh))

        def composite(warp: _Warp, base: np.ndarray) -> np.ndarray:
            if keep is None:
                return cv2.blendLinear(warp.color, base, warp.alpha, warp.inv_alpha)
            a = cv2.multiply(warp.alpha, keep)          # cut the area away under the hands
            return cv2.blendLinear(warp.color, base, a, cv2.subtract(1.0, a))

        out = composite(wp, roi)
        if self._fade is not None:
            prev, start = self._fade
            f = (now - start) / T.FADE_S
            if f >= 1.0 or f < 0.0:
                self._fade = None
            else:
                f = f * f * (3 - 2 * f)                   # smoothstep
                out = cv2.addWeighted(out, f, composite(prev, roi), 1.0 - f, 0)
        wants_lights = (self._hold_vis > 0.001 and self._progress > 0.002) or self._ready_vis > 0.001 \
            or self._ripple is not None
        if wants_lights:
            lt = self.lights_for(quad, size, (w, h), shape)
            if lt is not None and (lt.x0, lt.y0) == (wp.x0, wp.y0) and lt.edge.shape == wp.alpha.shape:
                for a, color in self._light_layers(lt, now):
                    if keep is not None:
                        a = cv2.multiply(a, keep)
                    out = cv2.blendLinear(self._solid_like(color, out.shape), out, a, cv2.subtract(1.0, a))
        roi[:] = out
        self.last_ms = (time.perf_counter() - t0) * 1000

    def _learn_background(self, roi: np.ndarray, box: tuple) -> None:
        """Slowly track what the table under the area looks like while no hand is over it."""
        if self._bg_box != box or self._bg is None:
            self._bg, self._bg_box = roi.copy(), box
        else:
            cv2.addWeighted(self._bg, 0.9, roi, 0.1, 0, dst=self._bg)
