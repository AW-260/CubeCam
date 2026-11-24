"""Only read colours when the cube is held still.

A face counts as still when, for `still_time` seconds, it was detected in
(nearly) every processed frame, its sticker centres moved less than
`max_motion` sticker pitches between frames and overall, and the flattened
face image barely changed (that catches a layer being turned while the
outline stays put). Colours are averaged over the still window, which also
averages out sensor noise.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from vision.face_detect import FaceDetection


@dataclass
class StableFace:
    lab: np.ndarray            # (9,3) averaged over the still window
    confidence: np.ndarray     # (9,)
    seen: np.ndarray           # (9,) seen in most frames
    occluded: np.ndarray       # (9,) occluded in some frames
    frames: int
    duration: float
    det: FaceDetection         # latest detection (for drawing)


class StabilityTracker:
    def __init__(self, still_time: float = 0.3, max_motion: float = 0.12, max_drift: float = 0.25,
                 max_warp_diff: float = 9.0, max_gap: float = 0.12):
        self.still_time = still_time
        self.max_motion = max_motion
        self.max_drift = max_drift
        self.max_warp_diff = max_warp_diff
        self.max_gap = max_gap
        self._hist: deque[tuple[float, FaceDetection, np.ndarray]] = deque()
        self._last_t: float | None = None

    def reset(self) -> None:
        self._hist.clear()
        self._last_t = None

    def progress(self, t: float) -> float:
        if not self._hist:
            return 0.0
        return min(1.0, (t - self._hist[0][0]) / max(self.still_time, 1e-6))

    def update(self, t: float, det: FaceDetection | None) -> StableFace | None:
        if det is None:
            if self._last_t is None or t - self._last_t > self.max_gap:
                self.reset()
            return None
        gray = cv2.cvtColor(det.warp, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if self._hist:
            _, prev, prev_gray = self._hist[-1]
            pitch = max(det.pitch, 1.0)
            step = np.linalg.norm(det.centers - prev.centers, axis=1).max() / pitch
            drift = np.linalg.norm(det.centers - self._hist[0][1].centers, axis=1).max() / pitch
            diff = float(np.mean(np.abs(gray - prev_gray)))
            if step > self.max_motion or drift > self.max_drift or diff > self.max_warp_diff:
                self._hist.clear()
        self._hist.append((t, det, gray))
        self._last_t = t
        # keep a window a bit longer than still_time
        while len(self._hist) > 2 and t - self._hist[1][0] >= self.still_time:
            self._hist.popleft()
        duration = t - self._hist[0][0]
        if duration + 1e-9 < self.still_time or len(self._hist) < 3:
            return None
        dets = [d for _, d, _ in self._hist]
        conf = np.stack([d.confidence for d in dets])          # (F,9)
        labs = np.stack([d.lab for d in dets])                 # (F,9,3)
        w = conf + 1e-6
        lab = (labs * w[..., None]).sum(axis=0) / w.sum(axis=0)[:, None]
        # colours that wobble between frames are less trustworthy
        wobble = np.hypot(labs[..., 1].std(axis=0), labs[..., 2].std(axis=0))
        c = conf.mean(axis=0) * np.clip(1.0 - (wobble - 3.0) / 8.0, 0.2, 1.0)
        occluded = np.stack([d.occluded for d in dets]).mean(axis=0) > 0.3
        c[occluded] = 0.0
        seen = np.stack([d.seen for d in dets]).mean(axis=0) >= 0.5
        return StableFace(lab, c, seen, occluded, len(dets), duration, det)
