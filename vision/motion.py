"""Stage 3: per-frame motion features of the cube during a solve (no images kept).

Pipeline per frame (in a background thread, see MotionWorker):

  1. Cube region. The cube is held between the two hands, so the region is a
     square centred between the two palm centres (from the hand tracker the
     main loop already runs), its side `roi_scale` x the palm distance. It is
     smoothed over time and held when a hand is lost; before any hands are
     seen a default region is used.
  2. The same square is cropped from the previous and the current raw frame and
     resized to `crop` x `crop` px. Skin (hands/fingers) is masked out, so a
     finger moving over the cube is not "cube motion"; anything outside the
     square (arms, background) is never looked at.
  3. Dense optical flow (DIS) between the two crops, in cube-region widths per
     second. From it a compact row of numbers is kept:
        g      median flow of the visible cube = how the cube *body* moves
        v      how the hands (region) move = the cube being carried
        Eres   mean |flow - g|: one part moving relative to the rest (layer turn)
        Erel   the same, but a pixel also counts as still when it moves with the
               hands (carried) - the layer-turn signal used by the analysis
        |g-v|  the whole cube moving relative to the hands (rotation)
        a 5x5 grid of mean flow vectors + visible fraction (the analysis works
        out from it, per solve, which cells are cube and where its centre is),
        curl, frame difference.
  4. The frames are dropped right away. Only the previous raw frame is kept
     (as a reference) to difference against the next one.

The analysis that turns these rows into turns/rotations/pauses lives in
vision/motion_analysis.py and is pure numpy, so it can be re-run offline on
stored rows (practice runs) after tuning.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

PALM_IDS = (0, 5, 9, 13, 17)
# MediaPipe hand skeleton (same as hands.HAND_CONNECTIONS; repeated so this module has no mediapipe import)
HAND_BONES = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)

GRID = 5          # the region is split into GRID x GRID cells
NCELL = GRID * GRID
COLUMNS = (
    ["dt", "roi_x", "roi_y", "roi_s", "roi_src", "skin", "valid", "gx", "gy", "vx", "vy", "hx", "hy",
     "E", "Eres", "Erel", "mov_g", "mov_v", "curl", "diff"]
    # GRID x GRID cells of the region (row-major, raw camera orientation): mean flow vector
    # (camera frame, region widths / s) and the visible (not hand, inside the frame) fraction
    + [f"fx{i}" for i in range(NCELL)] + [f"fy{i}" for i in range(NCELL)] + [f"cv{i}" for i in range(NCELL)]
)
COL = {name: i for i, name in enumerate(COLUMNS)}

# Storage: the grid flows as int8 in steps of FLOW_STEP region widths / s (+-6.35), the visible
# fractions as uint8 percent, everything else float16 - about 4x smaller than float32 rows.
FLOW_STEP = 0.05
_FLOW_COLS = np.array([COL[f"fx{i}"] for i in range(NCELL)] + [COL[f"fy{i}"] for i in range(NCELL)])
_VIS_COLS = np.array([COL[f"cv{i}"] for i in range(NCELL)])
_OTHER_COLS = np.array([i for i in range(len(COLUMNS)) if i not in set(_FLOW_COLS) | set(_VIS_COLS)])


def quantize(f: np.ndarray) -> dict:
    f = np.asarray(f, np.float32).reshape(-1, len(COLUMNS))
    return {
        "q_flow": np.clip(np.round(f[:, _FLOW_COLS] / FLOW_STEP), -127, 127).astype(np.int8),
        "q_vis": np.clip(np.round(f[:, _VIS_COLS] * 100), 0, 255).astype(np.uint8),
        "q_other": f[:, _OTHER_COLS].astype(np.float16),
    }


def dequantize(q: dict) -> np.ndarray:
    n = len(q["q_other"])
    f = np.zeros((n, len(COLUMNS)), np.float32)
    f[:, _FLOW_COLS] = q["q_flow"].astype(np.float32) * FLOW_STEP
    f[:, _VIS_COLS] = q["q_vis"].astype(np.float32) / 100.0
    f[:, _OTHER_COLS] = q["q_other"].astype(np.float32)
    return f


def roundtrip(f: np.ndarray) -> np.ndarray:
    """Rows exactly as they will be after saving and loading (so stored results can be reproduced)."""
    return dequantize(quantize(f))

ROI_DEFAULT, ROI_HELD, ROI_ONE_HAND, ROI_HANDS = 0, 1, 2, 3


@dataclass
class MotionConfig:
    crop: int = 128                # flow is computed on a crop x crop image of the cube region
    roi_scale: float = 1.0         # region side = roi_scale x distance between the palm centres
    roi_dy: float = 0.0            # shift the region down (+) / up (-) by this fraction of its side
    roi_smooth: float = 0.5        # EMA factor for region centre/size (1 = no smoothing)
    default_roi: tuple = (0.5, 0.45, 0.40)   # centre x, y (fractions of the frame), side (fraction of height)
    hold_frames: int = 45          # keep the last region this long without hands, then fall back
    move_thr: float = 0.8          # |flow| above this (region widths / s) counts as "moving" for mov_*
    flow_preset: int = 1           # DIS preset: 0 ultrafast, 1 fast, 2 medium
    hand_mask: bool = True         # also mask the hands drawn from the landmarks (not only skin colour)
    finger_width: float = 0.35     # finger thickness for that mask, as a fraction of the palm width
    hand_hold: int = 3             # reuse the last landmarks this many frames when tracking drops out
    skin_near_hands: float = 0.06  # also mask skin-coloured pixels this close to the landmark hands (crop widths)

    @classmethod
    def from_dict(cls, d: dict | None) -> "MotionConfig":
        d = d or {}
        kw = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        if "default_roi" in kw:
            kw["default_roi"] = tuple(float(v) for v in kw["default_roi"])
        return cls(**kw)


def skin_mask(bgr: np.ndarray) -> np.ndarray:
    """Broad skin detector (YCrCb box + moderate saturation, so saturated red/orange stickers are not skin)."""
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    cr, cb = ycc[..., 1], ycc[..., 2]
    s, v = hsv[..., 1], hsv[..., 2]
    return (cr >= 135) & (cr <= 180) & (cb >= 80) & (cb <= 135) & (s >= 35) & (s <= 175) & (v >= 50)


class HandROI:
    """Square cube region (pixels in the raw frame) from the palm centres of both hands."""

    def __init__(self, cfg: MotionConfig):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.c: np.ndarray | None = None      # centre (px)
        self.s: float | None = None           # side (px)
        self.v = np.zeros(2)                  # region velocity (px/s), smoothed
        self.src = ROI_DEFAULT
        self._missing = 0
        self._last_mid: np.ndarray | None = None
        self._last_t: float | None = None

    def update(self, hands, w: int, h: int, t: float, mirrored: bool) -> tuple[np.ndarray, float, int]:
        cfg = self.cfg
        palms = []
        for lm in hands or []:
            lm = np.asarray(lm, np.float64)
            if lm.shape != (21, 2):
                continue
            p = lm[list(PALM_IDS)].mean(axis=0)
            if mirrored:                          # landmarks come from the selfie (flipped) image
                p[0] = 1.0 - p[0]
            palms.append(p * [w, h])
        dt = None if self._last_t is None else max(1e-3, t - self._last_t)
        self._last_t = t
        a = cfg.roi_smooth
        if len(palms) >= 2:
            palms.sort(key=lambda q: q[0])
            pl, pr = palms[0], palms[-1]
            mid = (pl + pr) / 2
            side = float(np.clip(np.linalg.norm(pr - pl) * cfg.roi_scale, 0.08 * h, 0.95 * h))
            mid = mid + [0.0, cfg.roi_dy * side]
            if self._last_mid is not None and dt is not None:
                vel = (mid - self._last_mid) / dt
                self.v = 0.5 * self.v + 0.5 * vel
            else:
                self.v = np.zeros(2)
            self._last_mid = mid
            if self.c is None:
                self.c, self.s = mid, side
            else:
                self.c = (1 - a) * self.c + a * mid
                self.s = (1 - a) * self.s + a * side
            self.src = ROI_HANDS
            self._missing = 0
        else:
            self._missing += 1
            self._last_mid = None
            self.v = 0.5 * self.v                 # assume the cube settles when the hands are lost
            if self.c is not None and self._missing <= cfg.hold_frames:
                self.src = ROI_ONE_HAND if palms else ROI_HELD
            else:
                dx, dy, ds = cfg.default_roi
                self.c, self.s = np.array([dx * w, dy * h]), ds * h
                self.src = ROI_DEFAULT
        return self.c.copy(), float(self.s), self.src


def _crop(img: np.ndarray, c: np.ndarray, s: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Square crop centred at c with side s, resized to n x n; pixels outside the frame are black.

    Returns (crop, inside-mask)."""
    H, W = img.shape[:2]
    x0, y0 = int(round(c[0] - s / 2)), int(round(c[1] - s / 2))
    si = max(2, int(round(s)))
    x1, y1 = x0 + si, y0 + si
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    inside = np.ones((n, n), bool)
    if cx1 <= cx0 or cy1 <= cy0:
        return np.zeros((n, n, 3), np.uint8), ~inside
    sub = img[cy0:cy1, cx0:cx1]
    if (cx0, cy0, cx1, cy1) != (x0, y0, x1, y1):
        sub = cv2.copyMakeBorder(sub, cy0 - y0, y1 - cy1, cx0 - x0, x1 - cx1, cv2.BORDER_CONSTANT, value=0)
        k = n / si
        inside[:, :] = False
        inside[int((cy0 - y0) * k):int(math.ceil((cy1 - y0) * k)), int((cx0 - x0) * k):int(math.ceil((cx1 - x0) * k))] = True
    interp = cv2.INTER_AREA if si > n else cv2.INTER_LINEAR
    return cv2.resize(sub, (n, n), interpolation=interp), inside


class MotionFeatureExtractor:
    """Turns consecutive frames (+ hand landmarks) into feature rows. Not thread-safe; one per worker."""

    def __init__(self, cfg: MotionConfig | None = None):
        self.cfg = cfg or MotionConfig()
        self.roi = HandROI(self.cfg)
        self._flow = cv2.DISOpticalFlow_create(int(self.cfg.flow_preset))
        n = self.cfg.crop
        yy, xx = np.mgrid[0:n, 0:n]
        self._xr = (xx + 0.5) / n - 0.5          # crop coordinates in region widths, centred
        self._yr = (yy + 0.5) / n - 0.5
        self._r2 = self._xr ** 2 + self._yr ** 2
        self._cell = (np.minimum(yy * GRID // n, GRID - 1) * GRID + np.minimum(xx * GRID // n, GRID - 1)).ravel()
        self.reset()

    def reset(self) -> None:
        self.roi.reset()
        self._prev: np.ndarray | None = None
        self._prev_t: float | None = None
        self._prev_hands: list[np.ndarray] = []
        self._hands_age = 0

    def _hand_pixels(self, hands, W: int, H: int, mirrored: bool) -> list[np.ndarray]:
        out = []
        for lm in hands or []:
            lm = np.asarray(lm, np.float64)
            if lm.shape != (21, 2):
                continue
            p = lm * [W, H]
            if mirrored:
                p[:, 0] = W - p[:, 0]
            out.append(p)
        return out

    def _landmark_mask(self, hands_px: list[np.ndarray], c: np.ndarray, s: float, n: int) -> np.ndarray:
        """Hands drawn from the landmarks (palm polygon + finger bones) in crop coordinates."""
        mask = np.zeros((n, n), np.uint8)
        k = n / s
        origin = c - s / 2
        for p in hands_px:
            q = (p - origin) * k
            palm_w = float(np.linalg.norm(q[5] - q[17])) or float(np.linalg.norm(q[0] - q[9])) * 0.8
            thick = int(np.clip(palm_w * self.cfg.finger_width, 2, n / 4))
            poly = np.round(q[list(PALM_IDS)] * 4).astype(np.int32)
            cv2.fillPoly(mask, [poly], 255, cv2.LINE_8, shift=2)
            for i, j in HAND_BONES:
                cv2.line(mask, (int(q[i, 0]), int(q[i, 1])), (int(q[j, 0]), int(q[j, 1])), 255, thick)
        return mask > 0

    def process(self, image: np.ndarray, t: float, hands=None, mirrored: bool = False) -> np.ndarray | None:
        """Feature row for the motion between the previous frame and this one (None for the first frame)."""
        cfg = self.cfg
        H, W = image.shape[:2]
        c, s, src = self.roi.update(hands, W, H, t, mirrored)
        hands_px = self._hand_pixels(hands, W, H, mirrored) if cfg.hand_mask else []
        if hands_px:
            self._hands_age = 0
        elif self._hands_age < cfg.hand_hold:
            self._hands_age += 1
            hands_px = self._prev_hands
        prev_hands, self._prev_hands = self._prev_hands, hands_px
        prev, prev_t = self._prev, self._prev_t
        self._prev, self._prev_t = image, t
        if prev is None or prev.shape != image.shape or t <= prev_t:
            return None
        dt = t - prev_t
        n = cfg.crop
        a, inside = _crop(prev, c, s, n)
        b, _ = _crop(image, c, s, n)
        ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        if hands_px or prev_hands:
            # Where the hands were in either frame, from the landmarks, plus skin-coloured pixels
            # NEAR them (finger edges the skeleton misses). Colour alone is not used: in warm or
            # dim light red/orange stickers look like skin and would be masked away.
            skin = self._landmark_mask(list(hands_px) + list(prev_hands), c, s, n)
            if cfg.skin_near_hands > 0:
                r = max(1, int(cfg.skin_near_hands * n))
                near = cv2.dilate(skin.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
                skin |= near & (skin_mask(a) | skin_mask(b))
        else:
            skin = skin_mask(a) | skin_mask(b)
        skin_frac = float(skin[inside].mean()) if inside.any() else 1.0
        skin = cv2.dilate(skin.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        valid = inside & ~skin
        flow = self._flow.calc(ga, gb, None)                  # px per frame in the crop
        fx = flow[..., 0] / (n * dt)                          # region widths per second
        fy = flow[..., 1] / (n * dt)
        vx, vy = self.roi.v / max(s, 1.0)
        row = np.zeros(len(COLUMNS), np.float32)
        row[COL["dt"]] = dt
        row[COL["roi_x"]], row[COL["roi_y"]], row[COL["roi_s"]] = c[0] / W, c[1] / H, s / H
        row[COL["roi_src"]] = src
        row[COL["skin"]] = skin_frac
        nv = int(valid.sum())
        row[COL["valid"]] = nv / valid.size
        row[COL["vx"]], row[COL["vy"]] = vx, vy
        hand = skin & inside
        hx = hy = None
        if hand.sum() > 0.02 * hand.size:                     # how the visible hand/finger pixels move
            hx, hy = float(np.median(fx[hand])), float(np.median(fy[hand]))
            row[COL["hx"]], row[COL["hy"]] = hx, hy
        if nv < 0.05 * valid.size:
            return row
        vfx, vfy = fx[valid], fy[valid]
        gx, gy = float(np.median(vfx)), float(np.median(vfy))
        row[COL["gx"]], row[COL["gy"]] = gx, gy
        rx, ry = vfx - vx, vfy - vy                           # relative to the carrying hands
        mag_v = np.hypot(rx, ry)
        res_x, res_y = vfx - gx, vfy - gy                     # relative to the cube body
        mag_g = np.hypot(res_x, res_y)
        row[COL["E"]] = mag_v.mean()
        row[COL["Eres"]] = mag_g.mean()
        still = np.minimum(mag_g, mag_v)                      # not moving with the body, nor with the palms
        if hx is not None:
            still = np.minimum(still, np.hypot(vfx - hx, vfy - hy))   # ... nor with the hand pixels
        row[COL["Erel"]] = still.mean()
        row[COL["mov_g"]] = (mag_g > cfg.move_thr).mean()
        row[COL["mov_v"]] = (mag_v > cfg.move_thr).mean()
        xr, yr = self._xr[valid], self._yr[valid]
        row[COL["curl"]] = float((xr * res_y - yr * res_x).sum() / max(1e-6, self._r2[valid].sum()))
        row[COL["diff"]] = float(np.abs(ga.astype(np.int16) - gb.astype(np.int16))[valid].mean() / 255.0)
        cells = self._cell[valid.ravel()]
        cnt = np.bincount(cells, minlength=NCELL).astype(np.float64)
        safe = np.maximum(cnt, 1)
        row[COL["fx0"]:COL["fx0"] + NCELL] = np.bincount(cells, vfx, NCELL) / safe
        row[COL["fy0"]:COL["fy0"] + NCELL] = np.bincount(cells, vfy, NCELL) / safe
        row[COL["cv0"]:COL["cv0"] + NCELL] = cnt / (valid.size / NCELL)
        return row


class MotionWorker:
    """Background thread: collects feature rows between begin() and finish(). Never blocks the caller.

    submit() only swaps a reference under a lock; if the worker is still busy with
    the previous frame, the waiting one is replaced (and counted as dropped).
    """

    def __init__(self, cfg: MotionConfig | None = None):
        self.cfg = cfg or MotionConfig()
        self.extractor = MotionFeatureExtractor(self.cfg)
        self._cond = threading.Condition()
        self._pending = None
        self._collecting = False
        self._epoch = 0
        self._t: list[float] = []
        self._rows: list[np.ndarray] = []
        self.submitted = 0
        self.dropped = 0
        self.processed = 0
        self.errors = 0
        self.ms: list[float] = []
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="motion", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ main thread
    def begin(self) -> None:
        with self._cond:
            self._epoch += 1
            self._collecting = True
            self._pending = None
            self._t, self._rows = [], []
            self.submitted = self.dropped = self.processed = self.errors = 0
            self.ms = []
            self._reset = True

    def submit(self, image: np.ndarray, t: float, hands=None, mirrored: bool = False) -> None:
        """Hand over a frame (must not be modified afterwards). O(1), never waits for the worker."""
        with self._cond:
            if not self._collecting:
                return
            if self._pending is not None:
                self.dropped += 1
            self._pending = (image, t, hands, mirrored, self._epoch)
            self.submitted += 1
            self._cond.notify()

    def finish(self, t_end: float | None = None) -> dict:
        """Stop collecting and return {'t', 'f', 'columns', 'meta'} for rows with t <= t_end."""
        with self._cond:
            self._collecting = False
            self._pending = None
            t = np.array(self._t, np.float64)
            f = np.array(self._rows, np.float32).reshape(-1, len(COLUMNS))
            meta = {"submitted": self.submitted, "processed": self.processed, "dropped": self.dropped,
                    "errors": self.errors, "ms_median": float(np.median(self.ms)) if self.ms else None,
                    "ms_p95": float(np.percentile(self.ms, 95)) if self.ms else None}
            self._epoch += 1
        if t_end is not None and len(t):
            keep = t <= t_end + 1e-9
            t, f = t[keep], f[keep]
        return {"t": t, "f": f, "columns": list(COLUMNS), "meta": meta}

    @property
    def collecting(self) -> bool:
        return self._collecting

    def close(self) -> None:
        with self._cond:
            self._running = False
            self._cond.notify()
        self._thread.join(timeout=1.0)

    # ------------------------------------------------------------------ worker thread
    def _loop(self) -> None:
        while True:
            with self._cond:
                while self._running and self._pending is None:
                    self._cond.wait(0.5)
                if not self._running:
                    return
                image, t, hands, mirrored, epoch = self._pending
                self._pending = None
                if getattr(self, "_reset", False):
                    self.extractor.reset()
                    self._reset = False
            t0 = time.perf_counter()
            try:
                row = self.extractor.process(image, t, hands, mirrored)
                err = False
            except Exception:  # never let motion analysis kill anything
                row, err = None, True
            ms = (time.perf_counter() - t0) * 1000
            del image
            with self._cond:
                if epoch != self._epoch:
                    continue
                self.processed += 1
                self.errors += err
                self.ms.append(ms)
                if row is not None:
                    self._t.append(t)
                    self._rows.append(row)
