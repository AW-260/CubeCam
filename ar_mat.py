"""The AR start mat: a virtual mat lying on the table plane, drawn in perspective.

* `Mat`: centre / angle / size in table centimetres (see plane.py for the frame).
  Its projected outline is the timer's start zone.
* `mat_texture`: an original mat graphic (dark rubber, rounded corners, two
  hand pads with grip dots, a small status window, a subtle border) with one
  variant per timer state. No real brand names or logos.
* `ArMatRenderer`: warps the graphic onto the projected mat with
  cv2.warpPerspective and caches the result, so it only re-warps when the
  plane, the mat or the state changes. Each frame it only composites, and cuts
  the mat away where your hands are so they appear on top of it.
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

from plane import TablePlane
from zone import PALM_IDS, Zone

PX_PER_CM = 16          # texture resolution
MARGIN_CM = 2.5         # transparent margin round the mat in the texture (room for the glow)
PROGRESS_LEVELS = 12    # holding progress is drawn in this many steps (each step cached)
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


_TEX_BASE: dict = {}
_TEX: "OrderedDict[tuple, tuple[np.ndarray, tuple]]" = OrderedDict()
_TEX_LOCK = threading.RLock()   # textures are also built by the renderer's helper thread


def _base_layers(w_cm: float, h_cm: float, ppc: int):
    """State-independent parts of the texture (cached per size)."""
    key = (round(w_cm, 2), round(h_cm, 2), ppc)
    if key in _TEX_BASE:
        return _TEX_BASE[key]
    m = MARGIN_CM * ppc
    W, H = int(round((w_cm + 2 * MARGIN_CM) * ppc)), int(round((h_cm + 2 * MARGIN_CM) * ppc))
    x0, y0, x1, y1 = m, m, W - m, H - m
    mw, mh = x1 - x0, y1 - y0
    shape = (H, W)
    rng = np.random.default_rng(7)

    body = _rounded_rect_mask(shape, x0, y0, x1, y1, 2.2 * ppc)
    color = np.empty((H, W, 3), np.float32)
    # dark rubber: soft light falloff towards the far edge + fine grain
    yy = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None, None]
    color[:] = np.array([42, 40, 38], np.float32) * (0.92 + 0.12 * yy)
    grain = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 0.8)
    color += grain[..., None] * 3.0
    # bevel: a darker rim and a faint highlight line just inside it
    _blend(color, (24, 23, 22), _ring(shape, x0, y0, x1, y1, 2.2 * ppc, 0.35 * ppc) * 0.8)
    _blend(color, (78, 76, 74), _ring(shape, x0 + 0.35 * ppc, y0 + 0.35 * ppc, x1 - 0.35 * ppc,
                                      y1 - 0.35 * ppc, 1.9 * ppc, 0.08 * ppc) * 0.6)
    # subtle inset border
    bi = 1.0 * ppc
    _blend(color, (66, 64, 62), _ring(shape, x0 + bi, y0 + bi, x1 - bi, y1 - bi, 1.4 * ppc, 0.12 * ppc) * 0.8)

    # two hand pads with grip dots
    pads = []
    pw, ph = 0.36 * mw, 0.58 * mh
    for fx in (0.25, 0.75):
        cx, cy = x0 + fx * mw, y0 + 0.6 * mh
        pads.append((cx - pw / 2, cy - ph / 2, cx + pw / 2, cy + ph / 2))
    pad_masks = []
    dots = np.zeros(shape, np.uint8)
    step = 0.7 * ppc
    for px0, py0, px1, py1 in pads:
        pm = _rounded_rect_mask(shape, px0, py0, px1, py1, 2.4 * ppc)
        pad_masks.append(pm)
        _blend(color, (60, 58, 56), pm)
        for gy in np.arange(py0 + step, py1 - step / 2, step):
            for gx in np.arange(px0 + step, px1 - step / 2, step):
                cv2.circle(dots, (int(gx * 4), int(gy * 4)), int(0.13 * ppc * 4), 255, -1, shift=2)
        _blend(color, (88, 86, 84), _ring(shape, px0, py0, px1, py1, 2.4 * ppc, 0.07 * ppc) * 0.7)
    dot_mask = dots.astype(np.float32) / 255.0 * np.maximum.reduce(pad_masks)
    _blend(color, (84, 82, 80), dot_mask * 0.9)

    # status window at the far edge, centred between the pads
    sw, sh = 0.26 * mw, 0.15 * mh
    win = (x0 + mw / 2 - sw / 2, y0 + 0.07 * mh, x0 + mw / 2 + sw / 2, y0 + 0.07 * mh + sh)
    _blend(color, (18, 18, 18), _rounded_rect_mask(shape, *win, 0.6 * ppc))
    _blend(color, (60, 60, 60), _ring(shape, *win, 0.6 * ppc, 0.06 * ppc))
    # a thin groove between the pads
    groove = np.zeros(shape, np.float32)
    gx = int(x0 + mw / 2)
    cv2.line(groove, (gx, int(win[3] + 1.2 * ppc)), (gx, int(y1 - 1.6 * ppc)), 1.0, max(1, int(0.08 * ppc)), cv2.LINE_AA)
    _blend(color, (30, 29, 28), groove * 0.8)

    out = (body, color, pad_masks, pads, win, (x0, y0, x1, y1))
    _TEX_BASE[key] = out
    return out


def mat_texture(w_cm: float, h_cm: float, state: str = "idle", progress: float = 0.0,
                ppc: int = PX_PER_CM) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """(BGRA uint8 texture, mat rectangle (x0, y0, x1, y1) inside it in texture px)."""
    with _TEX_LOCK:
        return _mat_texture(w_cm, h_cm, state, progress, ppc)


def _mat_texture(w_cm, h_cm, state, progress, ppc):
    if state not in STATES:
        state = "idle"
    level = int(round(min(max(progress, 0.0), 1.0) * PROGRESS_LEVELS)) if state == "holding" else 0
    key = (round(w_cm, 2), round(h_cm, 2), ppc, state, level)
    if key in _TEX:
        _TEX.move_to_end(key)
        return _TEX[key]
    body, base_color, pad_masks, pads, win, rect = _base_layers(w_cm, h_cm, ppc)
    color = base_color.copy()
    alpha = body.copy()
    shape = body.shape
    x0, y0, x1, y1 = rect
    accent = {"idle": (150, 150, 150), "holding": (40, 210, 255), "ready": (80, 230, 80), "solving": (70, 70, 235)}[state]
    pad_union = np.maximum.reduce(pad_masks)
    if state == "holding":
        # pads fill up from the near edge
        frac = level / PROGRESS_LEVELS
        fill = np.zeros(shape, np.float32)
        for px0, py0, px1, py1 in pads:
            top = py1 - frac * (py1 - py0)
            fill[int(top):int(math.ceil(py1)) + 1, int(px0):int(math.ceil(px1)) + 1] = 1.0
        _blend(color, accent, fill * pad_union * 0.5)
        _blend(color, accent, _ring(shape, x0 + 1.0 * ppc, y0 + 1.0 * ppc, x1 - 1.0 * ppc, y1 - 1.0 * ppc,
                                    1.4 * ppc, 0.12 * ppc) * 0.7)
    elif state == "ready":
        _blend(color, accent, pad_union * 0.45)
        _blend(color, accent, _ring(shape, x0, y0, x1, y1, 2.2 * ppc, 0.3 * ppc))
        # soft green glow round the outside of the mat
        glow = cv2.GaussianBlur(body, (0, 0), 0.9 * ppc)
        outside = np.clip(glow * 1.6 - body, 0, 1) * 0.85
        color = color * body[..., None] + np.array(accent, np.float32) * (1 - body[..., None])
        alpha = np.maximum(body, outside)
    elif state == "solving":
        _blend(color, accent, pad_union * 0.12)
        _blend(color, accent, _ring(shape, x0 + 1.0 * ppc, y0 + 1.0 * ppc, x1 - 1.0 * ppc, y1 - 1.0 * ppc,
                                    1.4 * ppc, 0.12 * ppc) * 0.8)
    # status text in the window
    text = {"idle": "REST HANDS", "holding": "HOLD", "ready": "READY", "solving": "SOLVING"}[state]
    wx0, wy0, wx1, wy1 = win
    scale = 1.0
    th = max(1, int(round(0.09 * ppc)))
    (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, th)
    scale = min((wx1 - wx0) * 0.8 / tw, (wy1 - wy0) * 0.55 / tht)
    (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, th)
    canvas = np.zeros(shape, np.uint8)
    cv2.putText(canvas, text, (int((wx0 + wx1 - tw) / 2), int((wy0 + wy1 + tht) / 2)), cv2.FONT_HERSHEY_DUPLEX,
                scale, 255, th, cv2.LINE_AA)
    _blend(color, accent, canvas.astype(np.float32) / 255.0)

    tex = np.dstack([np.clip(color, 0, 255), np.clip(alpha * 255, 0, 255)]).astype(np.uint8)
    out = (tex, rect)
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
    color: np.ndarray     # uint8 (h, w, 3), not premultiplied
    alpha: np.ndarray     # float32 (h, w), 0..1
    inv_alpha: np.ndarray  # float32 (h, w), 1 - alpha


def warp_mat(quad: np.ndarray, size_cm: tuple[float, float], state: str, level: int,
             frame_wh: tuple[int, int], ppc: int = PX_PER_CM) -> _Warp | None:
    """Warp the mat texture onto the image quad (far-left, far-right, near-right, near-left), cropped to its box."""
    tex, (x0, y0, x1, y1) = mat_texture(size_cm[0], size_cm[1], state, level / PROGRESS_LEVELS, ppc)
    src = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    Hq = cv2.getPerspectiveTransform(src, np.float32(quad))
    # pre-shrink the texture when the mat is small on screen (avoids aliasing)
    on_screen = max(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3]))
    f = 1.0
    while on_screen / ((x1 - x0) * f) < 0.6 and f > 0.2:
        f *= 0.5
    if f < 1.0:
        tex = cv2.resize(tex, (max(2, int(tex.shape[1] * f)), max(2, int(tex.shape[0] * f))),
                         interpolation=cv2.INTER_AREA)
        Hq = Hq @ np.diag([1 / f, 1 / f, 1.0])
    th, tw = tex.shape[:2]
    outer = cv2.perspectiveTransform(np.float32([[[0, 0]], [[tw, 0]], [[tw, th]], [[0, th]]]), Hq).reshape(-1, 2)
    W, H = frame_wh
    bx0, by0 = int(max(0, np.floor(outer[:, 0].min()))), int(max(0, np.floor(outer[:, 1].min())))
    bx1, by1 = int(min(W, np.ceil(outer[:, 0].max()) + 1)), int(min(H, np.ceil(outer[:, 1].max()) + 1))
    if bx1 <= bx0 or by1 <= by0:
        return None
    T = np.array([[1, 0, -bx0], [0, 1, -by0], [0, 0, 1]], np.float64)
    # premultiply first so the edges blend correctly after interpolation
    a = tex[..., 3:4].astype(np.float32) / 255.0
    pm = np.dstack([tex[..., :3].astype(np.float32) * a, a[..., 0] * 255.0])
    warped = cv2.warpPerspective(pm, T @ Hq, (bx1 - bx0, by1 - by0), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    alpha = np.ascontiguousarray(warped[..., 3] / 255.0, dtype=np.float32)
    color = np.clip(warped[..., :3] / np.maximum(alpha, 1e-4)[..., None], 0, 255).astype(np.uint8)
    return _Warp(bx0, by0, color, alpha, np.ascontiguousarray(1.0 - alpha))


def _variants() -> list[tuple[str, int]]:
    return [("idle", 0), ("ready", 0), ("solving", 0)] + [("holding", i) for i in range(PROGRESS_LEVELS + 1)]


class ArMatRenderer:
    """Draws the mat into frames.

    Warps are cached per (quad, size, state). When a mat position is first
    seen, the current state is warped right away (a few ms) and every other
    state is prepared in a background thread, so a later state change never
    stalls the frame loop; until a variant is ready the idle one is shown (a
    frame or two). A steady mat therefore costs only the composite per frame.
    """

    def __init__(self, ppc: int = PX_PER_CM, cache_size: int = 64, background: bool = True):
        import threading

        self.ppc = ppc
        self.cache_size = cache_size
        self.background = background
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

    # ----------------------------------------------------------------- cache
    @staticmethod
    def _key(quad, size_cm, state, level, frame_wh):
        return (tuple(np.round(np.asarray(quad).ravel() * 4).astype(int)), round(size_cm[0], 2),
                round(size_cm[1], 2), state, level, tuple(frame_wh))

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
                k = self._key(job[0], size_cm, job[2], job[3], frame_wh)
                if not self._get(k)[0]:
                    self._store(k, warp_mat(*job, ppc=self.ppc))
            return
        with self._cond:
            # newest mat position first; drop queued work for older positions
            self._todo = jobs
            if self._thread is None:
                import threading

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
            k = self._key(quad, size_cm, st, lv, frame_wh)
            if self._get(k)[0]:
                continue
            try:
                self._store(k, warp_mat(quad, size_cm, st, lv, frame_wh, self.ppc))
            except Exception:  # pragma: no cover - never let the helper thread die on a bad quad
                pass

    def warp_for(self, quad, size_cm, state, progress, frame_wh) -> _Warp | None:
        level = int(round(min(max(progress, 0.0), 1.0) * PROGRESS_LEVELS)) if state == "holding" else 0
        if state not in STATES:
            state, level = "idle", 0
        key = self._key(quad, size_cm, state, level, frame_wh)
        found, wp = self._get(key)
        if found:
            return wp
        found_idle, idle = self._get(self._key(quad, size_cm, "idle", 0, frame_wh))
        if found_idle and self.background:
            self._queue_rest(quad, size_cm, frame_wh)
            return idle            # shown until the right variant is ready
        wp = warp_mat(quad, size_cm, state, level, frame_wh, self.ppc)
        self._store(key, wp)
        self._queue_rest(quad, size_cm, frame_wh)
        return wp

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
    def draw(self, img: np.ndarray, quad_px: np.ndarray, size_cm: tuple[float, float], state: str = "idle",
             progress: float = 0.0, hands_px: Sequence[np.ndarray] = (), occlude: bool = True) -> None:
        """Composite the mat into img (in place). hands_px: (21, 2) landmark arrays in image px."""
        import time

        t0 = time.perf_counter()
        quad = np.asarray(quad_px, dtype=np.float64).reshape(4, 2)
        h, w = img.shape[:2]
        wp = self.warp_for(quad, (float(size_cm[0]), float(size_cm[1])), state, progress, (w, h))
        if wp is None:
            return
        rh, rw = wp.alpha.shape
        roi = img[wp.y0:wp.y0 + rh, wp.x0:wp.x0 + rw]
        alpha, inv_alpha = wp.alpha, wp.inv_alpha
        m = None
        if occlude and len(hands_px):
            local = [np.asarray(lm, dtype=np.float64) - (wp.x0, wp.y0) for lm in hands_px]
            box = (wp.x0, wp.y0, rw, rh)
            m = hand_mask(roi, local, background=self._bg if self._bg_box == box else None)
            if m is not None:
                alpha = cv2.multiply(alpha, cv2.subtract(1.0, m))   # cut the mat away under the hands
                inv_alpha = cv2.subtract(1.0, alpha)
        if occlude and m is None:
            self._learn_background(roi, (wp.x0, wp.y0, rw, rh))
        # roi = colour * alpha + roi * (1 - alpha), on uint8 in one SIMD pass
        roi[:] = cv2.blendLinear(wp.color, roi, alpha, inv_alpha)
        self.last_ms = (time.perf_counter() - t0) * 1000

    def _learn_background(self, roi: np.ndarray, box: tuple) -> None:
        """Slowly track what the table under the mat looks like while no hand is over it."""
        if self._bg_box != box or self._bg is None:
            self._bg, self._bg_box = roi.copy(), box
        else:
            cv2.addWeighted(self._bg, 0.9, roi, 0.1, 0, dst=self._bg)
