"""Background thread running face detection + colour reading.

The main loop hands over the newest (downscaled) frame with `submit()`, which
never blocks: if the worker is still busy, the waiting frame is simply
replaced. Results are picked up with `latest`. main.py only submits frames
while the timer is IDLE with nothing in the start zone, so this work can never
delay a timing-relevant frame. OpenCV releases the GIL, so the detection runs
truly in parallel with hand tracking.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from vision.colors import ColorModel, ColorResult, plausible_face, read_stickers
from vision.face_detect import FaceDetection, FaceDetector
from vision.scan import FaceRead
from vision.stability import StabilityTracker

# reading-order index of the same sticker after a horizontal mirror
MIRROR = [r * 3 + (2 - c) for r in range(3) for c in range(3)]


@dataclass
class VisionResult:
    t: float                              # capture time of the analysed frame
    seq: int
    ms: float                             # detection + colour time
    det: FaceDetection | None
    points: np.ndarray | None = None      # (9,2) normalised display coords, TRUE sticker order
    corners: np.ndarray | None = None     # (4,2) normalised display coords
    live: list[ColorResult] | None = None # this frame's reads, true order
    face: FaceRead | None = None          # set when the cube has been still for still_time
    stable_progress: float = 0.0
    error: str | None = None


class VisionWorker:
    def __init__(self, detector: FaceDetector | None = None, model: ColorModel | None = None,
                 still_time: float = 0.3):
        self.detector = detector or FaceDetector()
        self.model = model or ColorModel.default()
        self.tracker = StabilityTracker(still_time=still_time)
        self._cond = threading.Condition()
        self._pending: tuple[np.ndarray, float, int, bool] | None = None
        self._latest: VisionResult | None = None
        self._reset = False
        self._running = True
        self.busy = False
        self.processed = 0
        self.ms_history: list[float] = []
        self._thread = threading.Thread(target=self._loop, name="vision", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ main-thread API
    def submit(self, image: np.ndarray, t: float, seq: int, mirrored: bool) -> None:
        """Queue a frame (the worker keeps only the newest). `image` must not be modified afterwards."""
        with self._cond:
            self._pending = (image, t, seq, mirrored)
            self._cond.notify()

    def reset(self) -> None:
        """Forget stillness/latest result (e.g. when vision is paused)."""
        with self._cond:
            self._pending = None
            self._latest = None
            self._reset = True

    @property
    def latest(self) -> VisionResult | None:
        with self._cond:
            return self._latest

    def set_model(self, model: ColorModel) -> None:
        self.model = model

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
                image, t, seq, mirrored = self._pending
                self._pending = None
                if self._reset:
                    self.tracker.reset()
                    self._reset = False
                self.busy = True
            try:
                res = self.analyse(image, t, seq, mirrored)
            except Exception as exc:  # never let vision kill anything
                res = VisionResult(t, seq, 0.0, None, error=f"{type(exc).__name__}: {exc}")
            with self._cond:
                self.busy = False
                if not self._reset:
                    self._latest = res
                self.processed += 1
                self.ms_history = (self.ms_history + [res.ms])[-120:]

    def analyse(self, image: np.ndarray, t: float, seq: int = 0, mirrored: bool = False) -> VisionResult:
        t0 = time.perf_counter()
        model = self.model
        det = self.detector.detect(image)
        if det is not None and not plausible_face(det.lab, det.seen, model):
            det = None
        stable = self.tracker.update(t, det)
        if det is None:
            return VisionResult(t, seq, (time.perf_counter() - t0) * 1000, None)
        order = MIRROR if mirrored else list(range(9))
        h, w = image.shape[:2]
        norm = np.array([w, h], np.float64)
        live = read_stickers(det.lab, det.confidence, model, det.occluded)
        res = VisionResult(
            t, seq, 0.0, det,
            points=det.centers[order] / norm,
            corners=det.corners / norm,
            live=[live[i] for i in order],
            stable_progress=self.tracker.progress(t),
        )
        if stable is not None:
            reads = read_stickers(stable.lab, stable.confidence, model, stable.occluded)
            res.face = FaceRead(
                colors=[reads[i].label for i in order],
                confidence=[reads[i].confidence for i in order],
                ambiguous=[reads[i].ambiguous for i in order],
                lab=[stable.lab[i].tolist() for i in order],
            )
        res.ms = (time.perf_counter() - t0) * 1000
        return res
