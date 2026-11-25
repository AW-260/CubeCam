"""The AR start mat: a virtual mat lying on the table plane, drawn in perspective.

* `Mat`: centre / angle / size in table centimetres (see plane.py for the frame).
  Its projected outline is the timer's start zone.
* `mat_texture`: an original mat graphic (graphite body with a soft rim and
  sheen, two raised hand pads with hand outlines, an inset light strip in the
  state colour, a status window with TrueType text) with one variant per timer
  state. Colours come from theme.py, shared with the on-screen panels. No real
  brand names or logos.
* `ArMatRenderer`: warps the graphic onto the projected mat with
  cv2.warpPerspective and caches the result, so it only re-warps when the
  plane, the mat or the state changes. Each frame it only composites:
  states cross-fade, the holding fill grows smoothly, and the mat is cut away
  where your hands are so they appear on top of it.
* `hand_mask`: landmark-based hand shape plus skin-coloured pixels near the
  hands, with feathered edges.
"""
from __future__ import annotations

import math
import threading
from collections import OrderedDict
from dataclasses import dataclass, replace
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
@dataclass(frozen=True)
class Mat:
    cx: float                 # centre on the table (cm)
    cy: float
    angle_deg: float = 0.0    # + = clockwise on screen
    width_cm: float = 30.0
    height_cm: float = 22.0

    def corners(self) -> np.ndarray:
        """Table cm of the corners: far-left, far-right, near-right, near-left (as seen at angle 0)."""
        a = math.radians(self.angle_deg)
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        w, h = self.width_cm / 2, self.height_cm / 2
        local = np.array([(-w, -h), (w, -h), (w, h), (-w, h)])
        return local @ R.T + (self.cx, self.cy)

    def moved_to(self, cx: float, cy: float) -> "Mat":
        return replace(self, cx=float(cx), cy=float(cy))

    def rotated(self, delta_deg: float) -> "Mat":
        a = (self.angle_deg + delta_deg + 180.0) % 360.0 - 180.0
        return replace(self, angle_deg=round(a, 6))

    def image_quad(self, plane: TablePlane) -> np.ndarray | None:
        """Projected corners in image px, or None if part of the mat is beyond the horizon."""
        quad, ok = plane.to_image(self.corners())
        if not ok.all():
            return None
        return quad

    def zone(self, plane: TablePlane) -> Zone | None:
        quad = self.image_quad(plane)
        if quad is None:
            return None
        return Zone(tuple(map(tuple, quad / np.array(plane.image_size, dtype=np.float64))))

    def to_config(self) -> dict:
        return {"centre_cm": [round(self.cx, 3), round(self.cy, 3)], "angle_deg": round(self.angle_deg, 3),
                "width_cm": round(self.width_cm, 3), "height_cm": round(self.height_cm, 3)}

    @classmethod
    def from_config(cls, d: dict | None) -> "Mat | None":
        if not d or d.get("centre_cm") is None:
            return None
        cx, cy = d["centre_cm"]
        return cls(float(cx), float(cy), float(d.get("angle_deg", 0.0)),
                   float(d.get("width_cm", 30.0)), float(d.get("height_cm", 22.0)))


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
    out = Mat(float(pts[0, 0]), float(pts[0, 1]), round(math.degrees(math.atan2(d[1], d[0])), 3),
              mat.width_cm, mat.height_cm)
    return out if mat_fits(out, new) else None


# ===================================================================== texture
def _rounded_rect_mask(shape, x0, y0, x1, y1, r, ss: int = 4) -> np.ndarray:
    """Anti-aliased rounded rectangle (float 0..1) via supersampling."""
    h, w = shape
    big = np.zeros((h * ss, w * ss), np.uint8)
    X0, Y0, X1, Y1, R = (int(round(v * ss)) for v in (x0, y0, x1, y1, r))
    R = max(1, min(R, (X1 - X0) // 2, (Y1 - Y0) // 2))
    cv2.rectangle(big, (X0 + R, Y0), (X1 - R, Y1), 255, -1)
    cv2.rectangle(big, (X0, Y0 + R), (X1, Y1 - R), 255, -1)
    for cx, cy in ((X0 + R, Y0 + R), (X1 - R, Y0 + R), (X0 + R, Y1 - R), (X1 - R, Y1 - R)):
        cv2.circle(big, (cx, cy), R, 255, -1)
    return cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def _ring(shape, x0, y0, x1, y1, r, width) -> np.ndarray:
    outer = _rounded_rect_mask(shape, x0, y0, x1, y1, r)
    inner = _rounded_rect_mask(shape, x0 + width, y0 + width, x1 - width, y1 - width, max(0.5, r - width))
    return np.clip(outer - inner, 0, 1)


def _blend(img: np.ndarray, color, alpha: np.ndarray) -> None:
    a = alpha[..., None]
    img[:] = img * (1 - a) + np.array(color, np.float32) * a


def rounded_rect_points(x0, y0, x1, y1, r, per_corner: int = 6) -> np.ndarray:
    """Outline of a rounded rectangle as a polygon (clockwise on screen)."""
    r = min(r, (x1 - x0) / 2, (y1 - y0) / 2)
    pts = []
    for cx, cy, a0 in ((x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)):
        for a in np.linspace(a0, a0 + 90, per_corner):
            pts.append((cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a))))
    return np.array(pts)


def clip_below(poly: np.ndarray, y_top: float) -> np.ndarray:
    """The part of a polygon with y >= y_top (Sutherland-Hodgman against one line)."""
    out = []
    n = len(poly)
    for i in range(n):
        cur, prev = poly[i], poly[i - 1]
        cin, pin = cur[1] >= y_top, prev[1] >= y_top
        if cin != pin:
            t = (y_top - prev[1]) / (cur[1] - prev[1])
            out.append(prev + t * (cur - prev))
        if cin:
            out.append(cur)
    return np.array(out) if out else np.zeros((0, 2))


def _hand_icon(shape, cx, cy, size, mirror: bool) -> np.ndarray:
    """Outline of a simple open hand (palm + four fingers + thumb) as a soft line mask."""
    h, w = shape
    ss = 4
    big = np.zeros((h * ss, w * ss), np.uint8)
    s = size * ss
    sx = -1 if mirror else 1
    X, Y = cx * ss, cy * ss

    def P(x, y):
        return (int(round(X + sx * x * s)), int(round(Y + y * s)))

    # palm
    cv2.ellipse(big, P(0.0, 0.18), (int(0.30 * s), int(0.30 * s)), 0, 0, 360, 255, -1)
    cv2.rectangle(big, P(-0.30, 0.02), P(0.30, 0.30), 255, -1)
    # fingers: (x, length, width)
    for fx, ln, fw in ((-0.225, 0.42, 0.115), (-0.075, 0.50, 0.12), (0.075, 0.47, 0.115), (0.215, 0.37, 0.105)):
        top = 0.02 - ln
        tip = fx * 1.35                       # fingers fan out a little
        cv2.line(big, P(fx, 0.04), P(tip, top), 255, int(fw * s))
        cv2.circle(big, P(tip, top), int(fw * s / 2), 255, -1)
    # thumb, angled out to the side
    cv2.line(big, P(-0.26, 0.30), P(-0.50, 0.02), 255, int(0.14 * s))
    cv2.circle(big, P(-0.50, 0.02), int(0.07 * s), 255, -1)
    filled = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    edge = filled - cv2.erode(filled, np.ones((3, 3), np.uint8), iterations=max(1, int(round(size * 0.035))))
    return np.clip(edge * 1.4, 0, 1)


_TEX_BASE: dict = {}
_TEX: "OrderedDict[tuple, tuple[np.ndarray, tuple]]" = OrderedDict()
_TEX_LOCK = threading.RLock()   # textures are also built by the renderer's helper thread


@dataclass(frozen=True)
class MatLayout:
    """Where things are in the texture (texture pixels)."""

    rect: tuple[float, float, float, float]          # the mat itself (x0, y0, x1, y1)
    pads: tuple[tuple[float, float, float, float], ...]
    pad_radius: float
    window: tuple[float, float, float, float]
    strip: tuple[float, float, float, float]          # the inset light strip's outer rectangle
    size: tuple[int, int]                              # texture (w, h)


def mat_layout(w_cm: float, h_cm: float, ppc: int = PX_PER_CM) -> MatLayout:
    m = MARGIN_CM * ppc
    W, H = int(round((w_cm + 2 * MARGIN_CM) * ppc)), int(round((h_cm + 2 * MARGIN_CM) * ppc))
    x0, y0, x1, y1 = m, m, W - m, H - m
    mw, mh = x1 - x0, y1 - y0
    pw, ph = 0.38 * mw, 0.60 * mh
    pads = tuple((x0 + fx * mw - pw / 2, y0 + 0.61 * mh - ph / 2, x0 + fx * mw + pw / 2, y0 + 0.61 * mh + ph / 2)
                 for fx in (0.255, 0.745))
    sw, sh = 0.30 * mw, 0.15 * mh
    window = (x0 + mw / 2 - sw / 2, y0 + 0.075 * mh, x0 + mw / 2 + sw / 2, y0 + 0.075 * mh + sh)
    inset = 0.9 * ppc
    strip = (x0 + inset, y0 + inset, x1 - inset, y1 - inset)
    return MatLayout((x0, y0, x1, y1), pads, 2.4 * ppc, window, strip, (W, H))


def _base_layers(w_cm: float, h_cm: float, ppc: int):
    """State-independent parts of the texture (cached per size)."""
    key = (round(w_cm, 2), round(h_cm, 2), ppc)
    if key in _TEX_BASE:
        return _TEX_BASE[key]
    L = mat_layout(w_cm, h_cm, ppc)
    W, H = L.size
    x0, y0, x1, y1 = L.rect
    shape = (H, W)
    rng = np.random.default_rng(7)
    radius = 3.0 * ppc

    body = _rounded_rect_mask(shape, x0, y0, x1, y1, radius)
    color = np.empty((H, W, 3), np.float32)
    # graphite with a soft light fall-off from the far edge, plus a faint diagonal sheen
    v = np.clip((np.arange(H, dtype=np.float32)[:, None] - y0) / max(1.0, y1 - y0), 0, 1)[..., None]
    u = np.clip((np.arange(W, dtype=np.float32)[None, :] - x0) / max(1.0, x1 - x0), 0, 1)[..., None]
    top, bot = np.array(T.MAT_BODY_TOP, np.float32), np.array(T.MAT_BODY_BOTTOM, np.float32)
    color[:] = top * (1 - v) + bot * v
    color += (6.0 * np.exp(-((u - 0.3) * 1.6 + (v - 0.2) * 1.2) ** 2 * 3.0))
    grain = cv2.GaussianBlur(rng.normal(0, 1, shape).astype(np.float32), (0, 0), 0.7)
    color += grain[..., None] * 2.2
    # rim: a darker edge band and a fine highlight on its inside, brightest along the far edge
    _blend(color, T.MAT_RIM, _ring(shape, x0, y0, x1, y1, radius, 0.42 * ppc) * 0.9)
    hl = _ring(shape, x0 + 0.42 * ppc, y0 + 0.42 * ppc, x1 - 0.42 * ppc, y1 - 0.42 * ppc,
               radius - 0.42 * ppc, 0.07 * ppc)
    _blend(color, T.MAT_HIGHLIGHT, hl * (0.25 + 0.45 * (1 - v[..., 0])))

    # hand pads: slightly raised, with a soft inner shadow at the far side and a hand outline
    pad_masks = []
    for i, (px0, py0, px1, py1) in enumerate(L.pads):
        pm = _rounded_rect_mask(shape, px0, py0, px1, py1, L.pad_radius)
        pad_masks.append(pm)
        pv = np.clip((np.arange(H, dtype=np.float32)[:, None] - py0) / max(1.0, py1 - py0), 0, 1)
        shade = np.array(T.MAT_PAD, np.float32) * (0.94 + 0.10 * pv[..., None])
        color = color * (1 - pm[..., None]) + shade * pm[..., None]
        inner = np.clip(pm - np.roll(pm, int(0.22 * ppc), axis=0), 0, 1)       # far-side inner shadow
        _blend(color, T.MAT_RIM, inner * 0.55)
        _blend(color, T.MAT_PAD_EDGE, _ring(shape, px0, py0, px1, py1, L.pad_radius, 0.06 * ppc) * 0.8)
    icons = []
    for i, (px0, py0, px1, py1) in enumerate(L.pads):
        icon = _hand_icon(shape, (px0 + px1) / 2, (py0 + py1) / 2 + 0.05 * (py1 - py0), 0.62 * (py1 - py0),
                          mirror=(i == 0))
        icons.append(icon)
        _blend(color, T.MAT_ICON, icon * 0.75)

    # status window
    wx0, wy0, wx1, wy1 = L.window
    win = _rounded_rect_mask(shape, *L.window, (wy1 - wy0) / 2)
    _blend(color, T.MAT_WINDOW, win)
    _blend(color, T.MAT_HIGHLIGHT, _ring(shape, *L.window, (wy1 - wy0) / 2, 0.05 * ppc) * 0.35)

    out = (body, color, pad_masks, icons, L)
    _TEX_BASE[key] = out
    return out


STATE_TEXT = {"idle": "PLACE HANDS", "holding": "HOLD STILL", "ready": "READY", "solving": "SOLVING"}


def mat_texture(w_cm: float, h_cm: float, state: str = "idle", progress: float = 0.0,
                ppc: int = PX_PER_CM) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """(BGRA uint8 texture, mat rectangle (x0, y0, x1, y1) inside it in texture px).

    The holding fill is normally drawn per frame by ArMatRenderer (smooth);
    a progress > 0 here bakes it in (for still pictures).
    """
    with _TEX_LOCK:
        return _mat_texture(w_cm, h_cm, state, progress, ppc)


def _mat_texture(w_cm, h_cm, state, progress, ppc):
    if state not in STATES:
        state = "idle"
    level = round(min(max(progress, 0.0), 1.0), 3) if state == "holding" else 0.0
    key = (round(w_cm, 2), round(h_cm, 2), ppc, state, level)
    if key in _TEX:
        _TEX.move_to_end(key)
        return _TEX[key]
    body, base_color, pad_masks, icons, L = _base_layers(w_cm, h_cm, ppc)
    color = base_color.copy()
    alpha = body.copy()
    shape = body.shape
    accent = T.STATE[state]
    pad_union = np.maximum.reduce(pad_masks)
    icon_union = np.maximum.reduce(icons)
    sx0, sy0, sx1, sy1 = L.strip
    strip = _ring(shape, sx0, sy0, sx1, sy1, 2.2 * ppc, 0.16 * ppc)
    strength = {"idle": 0.30, "holding": 0.85, "ready": 1.0, "solving": 0.75}[state]
    if state != "idle":
        # the strip glows a little into the rubber round it
        _blend(color, accent, np.clip(cv2.GaussianBlur(strip, (0, 0), 0.35 * ppc) * 1.2, 0, 1) * 0.35 * strength)
    _blend(color, accent, strip * strength)
    if state == "holding":
        _blend(color, accent, icon_union * 0.6)
        if level > 0:
            fill = np.zeros(shape, np.float32)
            for (px0, py0, px1, py1), pm in zip(L.pads, pad_masks):
                top = py1 - level * (py1 - py0)
                fill[int(top):int(math.ceil(py1)) + 1, int(px0):int(math.ceil(px1)) + 1] = 1.0
            _blend(color, accent, fill * pad_union * 0.42)
    elif state == "ready":
        _blend(color, accent, pad_union * 0.30)
        _blend(color, accent, icon_union * 0.9)
        glow = cv2.GaussianBlur(body, (0, 0), 1.0 * ppc)
        outside = np.clip(glow * 1.5 - body, 0, 1) * 0.6
        color = color * body[..., None] + np.array(accent, np.float32) * (1 - body[..., None])
        alpha = np.maximum(body, outside)
    elif state == "solving":
        _blend(color, accent, pad_union * 0.08)
    # status text (TrueType) in the window, with a small state dot
    from gfx import text_mask

    wx0, wy0, wx1, wy1 = L.window
    text = STATE_TEXT[state]
    px = int((wy1 - wy0) * 0.46)
    m = text_mask(text, "semibold", px).astype(np.float32) / 255.0
    dot_r = (wy1 - wy0) * 0.12
    gap = dot_r * 2.2
    tw = m.shape[1] + gap + 2 * dot_r
    scale = min(1.0, (wx1 - wx0) * 0.84 / tw)
    if scale < 1.0:
        m = cv2.resize(m, (max(1, int(m.shape[1] * scale)), max(1, int(m.shape[0] * scale))),
                       interpolation=cv2.INTER_AREA)
        tw = m.shape[1] + gap + 2 * dot_r
    tx = int((wx0 + wx1 - tw) / 2 + 2 * dot_r + gap)
    ty = int((wy0 + wy1 - m.shape[0]) / 2)
    canvas = np.zeros(shape, np.float32)
    canvas[ty:ty + m.shape[0], tx:tx + m.shape[1]] = m[:max(0, min(m.shape[0], shape[0] - ty)),
                                                       :max(0, min(m.shape[1], shape[1] - tx))]
    text_col = T.TEXT_MUTED if state == "idle" else accent
    _blend(color, text_col, canvas)
    dot = np.zeros(shape, np.uint8)
    cv2.circle(dot, (int(round((wx0 + wx1 - tw) / 2 + dot_r) * 4), int(round((wy0 + wy1) / 2 * 4))),
               int(round(dot_r * 4)), 255, -1, cv2.LINE_AA, shift=2)
    _blend(color, accent if state != "idle" else T.TEXT_FAINT, dot.astype(np.float32) / 255.0)

    tex = np.dstack([np.clip(color, 0, 255), np.clip(alpha * 255, 0, 255)]).astype(np.uint8)
    out = (tex, L.rect)
    _TEX[key] = out
    while len(_TEX) > 48:
        _TEX.popitem(last=False)
    return out


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
    tex_to_roi: np.ndarray  # 3x3: full-resolution texture px -> ROI px
    layout: MatLayout


def warp_mat(quad: np.ndarray, size_cm: tuple[float, float], state: str, level: float,
             frame_wh: tuple[int, int], ppc: int = PX_PER_CM) -> _Warp | None:
    """Warp the mat texture onto the image quad (far-left, far-right, near-right, near-left), cropped to its box."""
    tex, (x0, y0, x1, y1) = mat_texture(size_cm[0], size_cm[1], state, level, ppc)
    layout = mat_layout(size_cm[0], size_cm[1], ppc)
    src = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    H_full = cv2.getPerspectiveTransform(src, np.float32(quad)).astype(np.float64)
    Hq = H_full
    # pre-shrink the texture when the mat is small on screen (avoids aliasing)
    on_screen = max(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3]))
    f = 1.0
    while on_screen / ((x1 - x0) * f) < 0.6 and f > 0.2:
        f *= 0.5
    if f < 1.0:
        tex = cv2.resize(tex, (max(2, int(tex.shape[1] * f)), max(2, int(tex.shape[0] * f))),
                         interpolation=cv2.INTER_AREA)
        Hq = H_full @ np.diag([1 / f, 1 / f, 1.0])
    th, tw = tex.shape[:2]
    outer = cv2.perspectiveTransform(np.float32([[[0, 0]], [[tw, 0]], [[tw, th]], [[0, th]]]), Hq).reshape(-1, 2)
    W, H = frame_wh
    bx0, by0 = int(max(0, np.floor(outer[:, 0].min()))), int(max(0, np.floor(outer[:, 1].min())))
    bx1, by1 = int(min(W, np.ceil(outer[:, 0].max()) + 1)), int(min(H, np.ceil(outer[:, 1].max()) + 1))
    if bx1 <= bx0 or by1 <= by0:
        return None
    Tr = np.array([[1, 0, -bx0], [0, 1, -by0], [0, 0, 1]], np.float64)
    # premultiply first so the edges blend correctly after interpolation
    a = tex[..., 3:4].astype(np.float32) / 255.0
    pm = np.dstack([tex[..., :3].astype(np.float32) * a, a[..., 0] * 255.0])
    warped = cv2.warpPerspective(pm, Tr @ Hq, (bx1 - bx0, by1 - by0), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    alpha = np.ascontiguousarray(warped[..., 3] / 255.0, dtype=np.float32)
    color = np.clip(warped[..., :3] / np.maximum(alpha, 1e-4)[..., None], 0, 255).astype(np.uint8)
    return _Warp(bx0, by0, color, alpha, np.ascontiguousarray(1.0 - alpha), Tr @ H_full, layout)


def _variants() -> list[tuple[str, float]]:
    return [(s, 0.0) for s in STATES]


class ArMatRenderer:
    """Draws the mat into frames.

    Warps are cached per (quad, size, state). When a mat position is first
    seen, the current state is warped right away (a few ms) and the other
    states are prepared in a background thread, so a later state change never
    stalls the frame loop. Each frame then only composites:
      * state changes cross-fade over theme.FADE_S (no hard switch);
      * the holding progress fills the hand pads continuously (drawn per frame
        in perspective, not as stepped textures);
      * the mat is cut away under the hands.
    """

    def __init__(self, ppc: int = PX_PER_CM, cache_size: int = 32, background: bool = True, animate: bool = True):
        self.ppc = ppc
        self.cache_size = cache_size
        self.background = background
        self.animate = animate
        self._cache: "OrderedDict[tuple, _Warp | None]" = OrderedDict()
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._todo: list[tuple] = []
        self._running = True
        self._thread = None
        self.warps = 0          # how many warps were made (tests / profiling)
        self.last_ms = 0.0
        self._bg: np.ndarray | None = None     # clean picture of the mat region (no hands)
        self._bg_box: tuple | None = None
        self._shown: tuple | None = None       # (quad key, state, warp) drawn last frame
        self._fade: tuple | None = None        # (previous warp, start time) while cross-fading
        self._progress = 0.0

    # ----------------------------------------------------------------- cache
    @staticmethod
    def _qkey(quad, size_cm, frame_wh):
        return (tuple(np.round(np.asarray(quad).ravel() * 4).astype(int)), round(size_cm[0], 2),
                round(size_cm[1], 2), tuple(frame_wh))

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

    def _queue_rest(self, quad, size_cm, frame_wh) -> None:
        jobs = [(quad.copy(), size_cm, st, lv, frame_wh) for st, lv in _variants()]
        if not self.background:
            for job in jobs:
                k = self._qkey(job[0], size_cm, frame_wh) + (job[2],)
                if not self._get(k)[0]:
                    self._store(k, warp_mat(*job, ppc=self.ppc))
            return
        with self._cond:
            self._todo = jobs          # newest mat position first; drop work for older positions
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
                quad, size_cm, st, lv, frame_wh = self._todo.pop(0)
            k = self._qkey(quad, size_cm, frame_wh) + (st,)
            if self._get(k)[0]:
                continue
            try:
                self._store(k, warp_mat(quad, size_cm, st, lv, frame_wh, self.ppc))
            except Exception:  # pragma: no cover - never let the helper thread die on a bad quad
                pass

    def warp_for(self, quad, size_cm, state, frame_wh) -> tuple[_Warp | None, str]:
        """The warp to show for this state (or the idle one while it's being prepared)."""
        if state not in STATES:
            state = "idle"
        qk = self._qkey(quad, size_cm, frame_wh)
        found, wp = self._get(qk + (state,))
        if found:
            return wp, state
        found_idle, idle = self._get(qk + ("idle",))
        if found_idle and self.background:
            self._queue_rest(quad, size_cm, frame_wh)
            return idle, "idle"
        wp = warp_mat(quad, size_cm, state, 0.0, frame_wh, self.ppc)
        self._store(qk + (state,), wp)
        self._queue_rest(quad, size_cm, frame_wh)
        return wp, state

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
    def _hold_fill(self, wp: _Warp, progress: float) -> np.ndarray:
        """The warp's colour with the pads filled up to `progress`, drawn in perspective."""
        color = wp.color.copy()
        L = wp.layout
        for px0, py0, px1, py1 in L.pads:
            poly = clip_below(rounded_rect_points(px0, py0, px1, py1, L.pad_radius), py1 - progress * (py1 - py0))
            if len(poly) < 3:
                continue
            q = cv2.perspectiveTransform(poly.reshape(-1, 1, 2).astype(np.float64), wp.tex_to_roi).reshape(-1, 2)
            h, w = color.shape[:2]
            x0, y0 = max(0, int(np.floor(q[:, 0].min()))), max(0, int(np.floor(q[:, 1].min())))
            x1, y1 = min(w, int(np.ceil(q[:, 0].max())) + 1), min(h, int(np.ceil(q[:, 1].max())) + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            m = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillPoly(m, [np.int32(np.round((q - (x0, y0)) * 16))], 255, cv2.LINE_AA, shift=4)
            a = m.astype(np.float32) * (0.45 / 255.0)
            roi = color[y0:y1, x0:x1]
            col = np.empty_like(roi)
            col[:] = T.STATE["holding"]
            roi[:] = cv2.blendLinear(col, roi, a, 1.0 - a)
        return color

    def draw(self, img: np.ndarray, quad_px: np.ndarray, size_cm: tuple[float, float], state: str = "idle",
             progress: float = 0.0, hands_px: Sequence[np.ndarray] = (), occlude: bool = True) -> None:
        """Composite the mat into img (in place). hands_px: (21, 2) landmark arrays in image px."""
        import time

        t0 = time.perf_counter()
        quad = np.asarray(quad_px, dtype=np.float64).reshape(4, 2)
        h, w = img.shape[:2]
        size = (float(size_cm[0]), float(size_cm[1]))
        wp, shown_state = self.warp_for(quad, size, state, (w, h))
        if wp is None:
            return
        qk = self._qkey(quad, size, (w, h))
        # start a cross-fade when the state changes on the same mat position
        if self._shown is not None and self._shown[0] == qk and self._shown[1] != shown_state and self.animate:
            prev = self._shown[2]
            if prev is not None and prev.alpha.shape == wp.alpha.shape and (prev.x0, prev.y0) == (wp.x0, wp.y0):
                self._fade = (prev, t0)
        elif self._shown is None or self._shown[0] != qk:
            self._fade = None
        self._shown = (qk, shown_state, wp)
        # holding progress eases a little so it never jumps
        target = min(max(progress, 0.0), 1.0) if state == "holding" else 0.0
        self._progress = target if (not self.animate or target < self._progress) else \
            self._progress + (target - self._progress) * 0.6

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

        def composite(warp: _Warp, color: np.ndarray) -> np.ndarray:
            if keep is None:
                return cv2.blendLinear(color, roi, warp.alpha, warp.inv_alpha)
            a = cv2.multiply(warp.alpha, keep)          # cut the mat away under the hands
            return cv2.blendLinear(color, roi, a, cv2.subtract(1.0, a))

        color = self._hold_fill(wp, self._progress) if (shown_state == "holding" and self._progress > 0.002) else wp.color
        out = composite(wp, color)
        if self._fade is not None:
            prev, start = self._fade
            f = (t0 - start) / T.FADE_S
            if f >= 1.0:
                self._fade = None
            else:
                f = f * f * (3 - 2 * f)                   # smoothstep
                out = cv2.addWeighted(out, f, composite(prev, prev.color), 1.0 - f, 0)
        roi[:] = out
        self.last_ms = (time.perf_counter() - t0) * 1000

    def _learn_background(self, roi: np.ndarray, box: tuple) -> None:
        """Slowly track what the table under the mat looks like while no hand is over it."""
        if self._bg_box != box or self._bg is None:
            self._bg, self._bg_box = roi.copy(), box
        else:
            cv2.addWeighted(self._bg, 0.9, roi, 0.1, 0, dst=self._bg)
