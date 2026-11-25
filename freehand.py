"""Turn a traced mouse path into a clean closed shape (zone setup, "draw" mode).

The path is in any metric units: table centimetres when the table is
calibrated, screen pixels otherwise. Steps:

1. Rasterise the path as a closed, filled polygon (the end joins the start
   with a straight line) at a fixed working resolution relative to its size.
2. Morphological close then open: small gaps and notches are filled, and
   thin spurs vanish (for example where you overshoot the start point).
   Self-intersections and small loops disappear because only the outer
   contour of the largest blob is kept.
3. Resample the contour evenly and smooth it with a circular Gaussian, which
   removes hand jitter.
4. Simplify it with Douglas-Peucker into a few dozen points. The curves stay
   smooth, and the polygon stays cheap for the per-frame inside test.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

WORK_PX = 512          # the shape's longest side in the working raster
MAX_POINTS = 64


def path_length(pts) -> float:
    p = np.asarray(pts, dtype=np.float64)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


def polygon_area(pts) -> float:
    p = np.asarray(pts, dtype=np.float64)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def resample_closed(pts: np.ndarray, n: int) -> np.ndarray:
    """n points evenly spaced along the closed polygon."""
    p = np.vstack([pts, pts[:1]]).astype(np.float64)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] <= 0:
        return np.repeat(p[:1], n, axis=0)
    s = np.linspace(0.0, cum[-1], n, endpoint=False)
    return np.column_stack([np.interp(s, cum, p[:, 0]), np.interp(s, cum, p[:, 1])])


def smooth_closed(pts: np.ndarray, sigma: float) -> np.ndarray:
    """Circular Gaussian smoothing of a closed point sequence (sigma in samples)."""
    if sigma <= 0:
        return pts
    r = int(math.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    n = len(pts)
    idx = (np.arange(n)[:, None] + np.arange(-r, r + 1)[None, :]) % n
    return np.column_stack([pts[idx, 0] @ k, pts[idx, 1] @ k])


def _signed_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2


def clean_shape(path, smooth: float = 0.012, simplify: float = 0.001, min_fill: float = 0.04) -> np.ndarray | None:
    """A clean, simple, closed polygon (N, 2) from a traced path, or None if the path doesn't enclose an area.

    smooth: Gaussian sigma as a fraction of the perimeter; simplify:
    Douglas-Peucker tolerance as a fraction of the perimeter; min_fill: the
    shape must cover at least this share of the path's bounding box (a
    straight stroke or a scribbled line encloses nothing).
    The result runs clockwise on screen (y down) and doesn't repeat its first point.
    """
    p = np.asarray(path, dtype=np.float64).reshape(-1, 2)
    p = p[np.all(np.isfinite(p), axis=1)]
    if len(p) < 4:
        return None
    lo, hi = p.min(axis=0), p.max(axis=0)
    ext = float((hi - lo).max())
    if ext <= 0:
        return None
    S = WORK_PX / ext
    pad = 24
    size = np.int32(np.ceil((hi - lo) * S)) + 2 * pad
    q = (p - lo) * S + pad
    mask = np.zeros((int(size[1]), int(size[0])), np.uint8)
    pts16 = np.int32(np.round(q * 16))
    cv2.fillPoly(mask, [pts16], 255, cv2.LINE_8, shift=4)
    cv2.polylines(mask, [pts16], True, 255, 3, cv2.LINE_8, shift=4)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    box_area = float(np.prod(np.maximum(hi - lo, 1e-9) * S))
    if cv2.contourArea(c) < min_fill * box_area or len(c) < 8:
        return None
    c = c.reshape(-1, 2).astype(np.float64)
    n = 256
    c = resample_closed(c, n)
    c = smooth_closed(c, smooth * n)
    perim = path_length(np.vstack([c, c[:1]]))
    eps = simplify * perim
    out = cv2.approxPolyDP(np.float32(c).reshape(-1, 1, 2), eps, True).reshape(-1, 2).astype(np.float64)
    while len(out) > MAX_POINTS:
        eps *= 1.4
        out = cv2.approxPolyDP(np.float32(c).reshape(-1, 1, 2), eps, True).reshape(-1, 2).astype(np.float64)
    if len(out) < 3:
        return None
    if _signed_area(out) < 0:          # clockwise on screen = positive shoelace area with y down
        out = out[::-1]
    return (out - pad) / S + lo
