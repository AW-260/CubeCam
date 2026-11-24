"""Record solves (frames + timestamps + scramble + result) for later training data.

Layout (datasets/ is git-ignored):

    datasets/<YYYYmmdd-HHMMSS>_<n>/
        frames/000001.jpg ...        raw camera frames (NOT mirrored)
        frames.csv                   index, seq, t (perf_counter s), t_rel (s from solve start)
        meta.json                    scramble, result, penalty, event times, camera info, drops

How it stays out of the timer's way:
  * The camera thread calls `on_frame()` for every captured frame; that only
    appends a reference to a deque or a queue (microseconds, never blocks).
  * JPEG encoding and file writes happen in a separate writer thread.
  * If the writer falls behind and the queue is full, frames are DROPPED (and
    counted in meta.json) rather than blocking anything.
  * A short pre-roll ring buffer keeps the frames just before the start.
"""
from __future__ import annotations

import csv
import json
import os
import queue
import shutil
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2


@dataclass
class _Session:
    path: Path
    t_start: float
    meta: dict
    t_end: float | None = None
    frames: int = 0
    dropped: int = 0
    rows: list = field(default_factory=list)


_FINISH = object()


class DatasetRecorder:
    def __init__(self, root: str | os.PathLike = "datasets", jpeg_quality: int = 90, max_queue: int = 120,
                 preroll: float = 0.5, tail: float = 0.5, max_width: int = 0):
        self.root = Path(root)
        self.jpeg_quality = int(jpeg_quality)
        self.preroll = float(preroll)
        self.tail = float(tail)
        self.max_width = int(max_width)
        self.enabled = False
        self._lock = threading.Lock()
        self._ring: deque = deque()
        self._session: _Session | None = None
        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self._bytes = 0
        self._free: int | None = None
        self.saved_solves = 0
        self.last_error: str | None = None
        self._running = True
        self._thread = threading.Thread(target=self._writer, name="dataset-writer", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ control (main thread)
    def toggle(self) -> bool:
        with self._lock:
            self.enabled = not self.enabled
            if not self.enabled:
                self._ring.clear()
                if self._session is not None:
                    self._close_locked(aborted=self._session.t_end is None)
        return self.enabled

    def begin(self, t_start: float, meta: dict) -> None:
        """Start recording a solve at capture time t_start (inspection or solve start)."""
        with self._lock:
            if not self.enabled:
                return
            if self._session is not None:      # unfinished previous one: close it
                self._close_locked(aborted=True)
            name = datetime.now().strftime("%Y%m%d-%H%M%S")
            path = self.root / name
            n = 1
            while path.exists():
                n += 1
                path = self.root / f"{name}_{n}"
            self._session = _Session(path, t_start, dict(meta))
            sess = self._session
            for frame in self._ring:
                if frame.t >= t_start - self.preroll:
                    self._enqueue_locked(sess, frame)
            self._ring.clear()

    def end(self, t_end: float, result: dict) -> None:
        with self._lock:
            if self._session is None:
                return
            self._session.meta.update(result)
            self._session.t_end = t_end

    def abort(self, t: float) -> None:
        with self._lock:
            if self._session is not None:
                self._session.meta["aborted"] = True
                self._session.t_end = t

    # ------------------------------------------------------------------ camera thread
    def on_frame(self, frame) -> None:
        """Called by the camera thread for every frame. Must stay cheap."""
        if not self.enabled:
            return
        with self._lock:
            sess = self._session
            if sess is None:
                self._ring.append(frame)
                while self._ring and frame.t - self._ring[0].t > self.preroll:
                    self._ring.popleft()
                return
            if sess.t_end is not None and frame.t > sess.t_end + self.tail:
                self._close_locked(aborted=bool(sess.meta.get("aborted")))
                self._ring.append(frame)
                return
            self._enqueue_locked(sess, frame)

    def _enqueue_locked(self, sess: _Session, frame) -> None:
        try:
            self._q.put_nowait((sess, frame))
            sess.frames += 1
        except queue.Full:
            sess.dropped += 1

    def _close_locked(self, aborted: bool) -> None:
        sess = self._session
        self._session = None
        if sess is None:
            return
        sess.meta["aborted"] = aborted
        try:
            self._q.put_nowait((sess, _FINISH))
        except queue.Full:
            # the writer will still finish it when it sees the session closed at shutdown
            threading.Thread(target=self._q.put, args=((sess, _FINISH),), daemon=True).start()

    # ------------------------------------------------------------------ writer thread
    def _writer(self) -> None:
        self._bytes = _dir_size(self.root)
        last_free = 0.0
        while self._running or not self._q.empty():
            try:
                sess, item = self._q.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                if item is _FINISH:
                    self._write_meta(sess)
                    self.saved_solves += 1
                else:
                    self._write_frame(sess, item)
            except OSError as exc:
                self.last_error = str(exc)
            if time.monotonic() - last_free > 5:
                last_free = time.monotonic()
                try:
                    self._free = shutil.disk_usage(self.root if self.root.exists() else Path(".")).free
                except OSError:
                    self._free = None

    def _write_frame(self, sess: _Session, frame) -> None:
        d = sess.path / "frames"
        d.mkdir(parents=True, exist_ok=True)
        img = frame.image
        if self.max_width and img.shape[1] > self.max_width:
            s = self.max_width / img.shape[1]
            img = cv2.resize(img, (self.max_width, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return
        idx = len(sess.rows) + 1
        name = f"{idx:06d}.jpg"
        (d / name).write_bytes(buf.tobytes())
        self._bytes += len(buf)
        sess.rows.append((idx, frame.seq, f"{frame.t:.6f}", f"{frame.t - sess.t_start:.6f}", name))

    def _write_meta(self, sess: _Session) -> None:
        sess.path.mkdir(parents=True, exist_ok=True)
        with open(sess.path / "frames.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["index", "seq", "t", "t_rel", "file"])
            w.writerows(sess.rows)
        meta = dict(sess.meta)
        meta.update({"frames_written": len(sess.rows), "frames_dropped": sess.dropped,
                     "t_start": sess.t_start, "t_end": sess.t_end, "mirrored": False})
        (sess.path / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        self._bytes += 2048

    # ------------------------------------------------------------------ status
    @property
    def recording(self) -> bool:
        return self._session is not None

    @property
    def queue_size(self) -> int:
        return self._q.qsize()

    def status_text(self) -> str:
        state = "REC" if self.recording else ("armed" if self.enabled else "off")
        free = f", {self._free / 1e9:.0f} GB free" if self._free is not None else ""
        sess = self._session
        drops = f", dropped {sess.dropped}" if sess is not None and sess.dropped else ""
        return (f"dataset {state}: {self.saved_solves} solves this session, {self._bytes / 1e9:.2f} GB on disk"
                f"{free}, queue {self.queue_size}{drops}")

    def close(self, timeout: float = 10.0) -> None:
        with self._lock:
            if self._session is not None:
                self._close_locked(aborted=self._session.t_end is None)
        self._running = False
        self._thread.join(timeout=timeout)


def _dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total
