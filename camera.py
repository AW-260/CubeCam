"""Threaded webcam capture with per-frame timestamps taken right after grab().

The capture thread does nothing but grab/retrieve/timestamp, so slow hand
detection in the main thread never delays a timestamp. The main thread always
gets the newest frame (frames it was too slow for are skipped, and the
*effective* sampling rate is reported separately by main.py).

Why perf_counter instead of CAP_PROP_POS_MSEC: for live webcams on Windows,
POS_MSEC is unreliable (often 0 or driver-dependent). grab() blocks until the
driver hands over a frame, so perf_counter() taken right after it is the
delivery time. The exposure/transfer latency is nearly constant, so it cancels
out between the start and stop frames of a solve.
"""
from __future__ import annotations

import sys
import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

_BACKENDS = {
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "v4l2": cv2.CAP_V4L2,
    "avfoundation": cv2.CAP_AVFOUNDATION,
    "any": cv2.CAP_ANY,
}


@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    t: float      # time.perf_counter() seconds, taken immediately after grab()
    seq: int


class Camera:
    def __init__(
        self,
        index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 120,
        backend: str = "auto",
        fourcc: str | None = "MJPG",
    ):
        self.index = index
        self.req_width, self.req_height, self.req_fps = width, height, fps
        self.backend = backend
        self.fourcc = fourcc
        self.error: str | None = None
        self.backend_name = ""
        self.reported_fps = 0.0
        self.width = self.height = 0
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._cond = threading.Condition()
        self._frame: Frame | None = None
        self._times: deque[float] = deque(maxlen=90)

    # ------------------------------------------------------------------ open
    def _candidate_backends(self) -> list[tuple[str, int]]:
        if self.backend != "auto":
            return [(self.backend, _BACKENDS.get(self.backend, cv2.CAP_ANY))]
        if sys.platform == "win32":
            # DirectShow + MJPG usually unlocks the highest frame rates on Windows.
            return [("dshow", cv2.CAP_DSHOW), ("msmf", cv2.CAP_MSMF)]
        return [("any", cv2.CAP_ANY)]

    def open(self) -> bool:
        """Open the camera and start the capture thread. Returns False on failure."""
        errors = []
        for name, api in self._candidate_backends():
            try:
                cap = cv2.VideoCapture(self.index, api)
            except Exception as exc:  # pragma: no cover - driver specific
                errors.append(f"{name}: {exc}")
                continue
            if not cap.isOpened():
                cap.release()
                errors.append(f"{name}: could not open camera {self.index}")
                continue
            # Order matters for DirectShow: codec, then size, then fps.
            if self.fourcc:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.req_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.req_height)
            cap.set(cv2.CAP_PROP_FPS, self.req_fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep latency low where supported
            ok, img = cap.read()
            if not ok or img is None:
                cap.release()
                errors.append(f"{name}: opened but returned no frames")
                continue
            self._cap = cap
            self.backend_name = name
            self.height, self.width = img.shape[:2]
            self.reported_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            self._running = True
            self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
            self._thread.start()
            return True
        self.error = "; ".join(errors) or "no camera backend available"
        return False

    # ------------------------------------------------------------------ thread
    def _loop(self) -> None:
        cap = self._cap
        seq = 0
        failures = 0
        while self._running:
            ok = cap.grab()
            t = time.perf_counter()
            if not ok:
                failures += 1
                if failures > 50:
                    self.error = "camera stopped delivering frames (disconnected?)"
                    self._running = False
                    with self._cond:
                        self._cond.notify_all()
                    break
                time.sleep(0.01)
                continue
            failures = 0
            ok, img = cap.retrieve()
            if not ok or img is None:
                continue
            seq += 1
            with self._cond:
                self._frame = Frame(img, t, seq)
                self._times.append(t)
                self._cond.notify_all()

    # ------------------------------------------------------------------ access
    @property
    def alive(self) -> bool:
        return self._running

    def wait_new(self, last_seq: int, timeout: float = 0.1) -> Frame | None:
        """Return the newest frame with seq > last_seq, waiting up to `timeout`."""
        with self._cond:
            if self._frame is None or self._frame.seq <= last_seq:
                self._cond.wait(timeout)
            f = self._frame
            return f if f is not None and f.seq > last_seq else None

    @property
    def measured_fps(self) -> float:
        with self._cond:
            if len(self._times) < 2:
                return 0.0
            span = self._times[-1] - self._times[0]
            return (len(self._times) - 1) / span if span > 0 else 0.0

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
