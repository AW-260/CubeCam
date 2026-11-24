import csv
import json
import time
from pathlib import Path

import numpy as np

from camera import Frame
from config import DEFAULTS
from vision.controller import VisionController
from vision.dataset import DatasetRecorder


def _frames(n, t0=100.0, fps=60.0, shape=(72, 128, 3)):
    img = np.zeros(shape, np.uint8)
    return [Frame(img, t0 + i / fps, i + 1) for i in range(n)]


def test_recorder_writes_solve_with_preroll_and_tail(tmp_path):
    rec = DatasetRecorder(tmp_path / "datasets", preroll=0.2, tail=0.1)
    rec.toggle()
    frames = _frames(120)                       # 2 s at 60 fps, t = 100.000 .. 101.983
    for f in frames[:30]:                       # before the solve: only the pre-roll is kept
        rec.on_frame(f)
    rec.begin(frames[30].t, {"scramble": "R U R'"})
    for f in frames[30:90]:
        rec.on_frame(f)
    rec.end(frames[89].t, {"time": 1.0, "penalty": "OK"})
    for f in frames[90:]:
        rec.on_frame(f)
    rec.close()
    dirs = list((tmp_path / "datasets").iterdir())
    assert len(dirs) == 1
    meta = json.loads((dirs[0] / "meta.json").read_text())
    assert meta["scramble"] == "R U R'" and meta["time"] == 1.0 and meta["aborted"] is False
    rows = list(csv.DictReader(open(dirs[0] / "frames.csv")))
    ts = [float(r["t"]) for r in rows]
    assert ts == sorted(ts)
    assert ts[0] < frames[30].t and ts[0] >= frames[30].t - 0.2 - 1e-9       # pre-roll
    assert ts[-1] > frames[89].t and ts[-1] <= frames[89].t + 0.1 + 1 / 60   # tail
    assert len(list((dirs[0] / "frames").glob("*.jpg"))) == len(rows) == meta["frames_written"]
    assert meta["frames_dropped"] == 0


def test_recorder_never_blocks_and_counts_drops(tmp_path):
    rec = DatasetRecorder(tmp_path / "d", max_queue=3, preroll=0.0)
    rec.toggle()
    big = _frames(300, shape=(720, 1280, 3))
    rec.begin(big[0].t, {})
    worst = 0.0
    for f in big:
        t0 = time.perf_counter()
        rec.on_frame(f)
        worst = max(worst, time.perf_counter() - t0)
    rec.end(big[-1].t, {"time": 5.0})
    rec.on_frame(_frames(1, t0=big[-1].t + 1.0)[0])   # closes the session
    rec.close()
    meta = json.loads(next((tmp_path / "d").iterdir()).joinpath("meta.json").read_text())
    assert meta["frames_dropped"] > 0                  # writer could not keep up with a 3-frame queue
    assert meta["frames_written"] + meta["frames_dropped"] == 300
    assert worst < 0.005, f"on_frame blocked for {worst * 1000:.1f} ms"


def test_recorder_disabled_records_nothing(tmp_path):
    rec = DatasetRecorder(tmp_path / "d")
    rec.begin(0.0, {})
    for f in _frames(10):
        rec.on_frame(f)
    rec.close()
    assert not (tmp_path / "d").exists() or not any((tmp_path / "d").iterdir())


def test_aborted_solve_is_marked(tmp_path):
    rec = DatasetRecorder(tmp_path / "d", preroll=0.0, tail=0.0)
    rec.toggle()
    fr = _frames(20)
    rec.begin(fr[0].t, {})
    for f in fr[:10]:
        rec.on_frame(f)
    rec.abort(fr[9].t)
    for f in fr[10:]:
        rec.on_frame(f)
    rec.close()
    meta = json.loads(next((tmp_path / "d").iterdir()).joinpath("meta.json").read_text())
    assert meta["aborted"] is True


class _FakeWorker:
    def __init__(self):
        self.submitted = 0
        self.resets = 0
        self.latest = None
        self.ms_history = []

    def submit(self, *a):
        self.submitted += 1

    def reset(self):
        self.resets += 1

    def set_model(self, m):
        pass

    def close(self):
        pass


def _controller(tmp_path, msgs):
    cfg = {k: dict(v) if isinstance(v, dict) else v for k, v in DEFAULTS.items()}
    cfg["vision"] = dict(DEFAULTS["vision"], calibration_path=str(tmp_path / "cal.json"))
    cfg["dataset"] = dict(DEFAULTS["dataset"], path=str(tmp_path / "ds"))
    def resolve(p: str) -> Path:
        return Path(p) if Path(p).is_absolute() else tmp_path / p

    c = VisionController(cfg, resolve, lambda m, s=3.0: msgs.append(m), have_camera=True)
    c.worker.close()
    c.worker = _FakeWorker()
    return c


def test_vision_never_runs_unless_allowed(tmp_path):
    msgs = []
    c = _controller(tmp_path, msgs)
    img = np.zeros((360, 640, 3), np.uint8)
    for i in range(10):
        c.feed(img, i / 30, i, True, allowed=False, debug=True)
    assert c.worker.submitted == 0
    for i in range(10, 15):
        c.feed(img, i / 30, i, True, allowed=True, debug=False)
    assert c.worker.submitted == 5
    c.feed(img, 1.0, 99, True, allowed=False, debug=False)   # e.g. hands entered the zone
    assert c.worker.submitted == 5 and c.worker.resets == 1
    c.close()


def test_modes_only_start_when_idle_and_timer_cancels_them(tmp_path):
    msgs = []
    c = _controller(tmp_path, msgs)
    assert c.on_key(ord("s"), "s", idle=False) and c.mode is None
    assert c.on_key(ord("s"), "s", idle=True) and c.mode == "scan" and c.modal
    # timer keys are swallowed while scanning, quit is not
    assert c.on_key(ord("d"), "d", idle=True)
    assert not c.on_key(ord("q"), "q", idle=True)
    # if the timer leaves IDLE (e.g. spacebar), the mode closes
    c.feed(np.zeros((36, 64, 3), np.uint8), 0.0, 1, True, allowed=False, debug=False)
    assert c.mode is None
    assert c.on_key(ord("c"), "c", idle=True) and c.mode == "calibrate"
    assert c.on_key(27, "\x1b", idle=True) and c.mode is None
    c.close()
