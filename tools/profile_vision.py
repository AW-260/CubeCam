"""Profile cube-face detection on LIVE camera frames.

    python tools/profile_vision.py [--seconds 8] [--config config.yaml]

Measures, using the same settings as the app:
  1. face detection + colour reading alone (ms/frame), on mirrored frames downscaled
     to vision.process_width, exactly as the vision worker sees them;
  2. the main-loop rate with hand tracking only (what the timer gets);
  3. the main-loop rate with hand tracking AND the vision worker running in parallel
     (what happens in IDLE with the scramble check on).
Hold a cube up to the camera during the run to profile realistic content.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from camera import Camera  # noqa: E402
from config import load_config  # noqa: E402
from detection import Zone  # noqa: E402
from vision.colors import ColorModel  # noqa: E402
from vision.face_detect import FaceDetector  # noqa: E402
from vision.worker import VisionWorker  # noqa: E402


def stats(xs):
    a = np.array(xs) if xs else np.zeros(1)
    return f"median {np.median(a):.1f} ms, p95 {np.percentile(a, 95):.1f} ms, max {a.max():.1f} ms (n={len(xs)})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.yaml"))
    args = ap.parse_args()
    cfg, _ = load_config(args.config)
    cc, vc = cfg["camera"], cfg["vision"]
    cam = Camera(int(cc["index"]), int(cc["width"]), int(cc["height"]), int(cc["fps"]), str(cc["backend"]),
                 cc.get("fourcc") or None)
    if not cam.open():
        print("camera error:", cam.error)
        return 1
    print(f"camera {cam.width}x{cam.height} via {cam.backend_name}, driver says {cam.reported_fps:.0f} fps")
    time.sleep(1.0)
    print(f"measured capture rate: {cam.measured_fps:.1f} fps -> frame budget {1000 / max(cam.measured_fps, 1):.1f} ms")
    pw = int(vc["process_width"])
    model, _ = ColorModel.load(Path(args.config).parent / vc["calibration_path"])
    det = FaceDetector(process_width=pw)
    hands = None
    try:
        from hands import HandZoneDetector

        dcfg = dict(cfg["detection"])
        dcfg["model_path"] = str(Path(args.config).parent / dcfg["model_path"])
        hands = HandZoneDetector(Zone(**cfg["zone"]), dcfg)
    except Exception as exc:  # noqa: BLE001
        print("hand tracking unavailable:", exc)

    def frames(seconds):
        last, end = 0, time.perf_counter() + seconds
        while time.perf_counter() < end:
            f = cam.wait_new(last, 0.2)
            if f is None:
                continue
            last = f.seq
            yield f

    # 1. detection alone
    det_ms, found, resize_ms = [], 0, []
    for f in frames(args.seconds):
        img = cv2.flip(f.image, 1)
        t0 = time.perf_counter()
        small = cv2.resize(img, (pw, int(img.shape[0] * pw / img.shape[1])), interpolation=cv2.INTER_AREA)
        resize_ms.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        d = det.detect(small)
        det_ms.append((time.perf_counter() - t0) * 1000)
        found += d is not None
    print(f"\n[1] face detection alone: {stats(det_ms)}; face found in {found}/{len(det_ms)} frames")
    print(f"    main-thread cost to hand a frame to the worker (resize): {stats(resize_ms)}")

    # 2/3. main loop rate with hands only vs hands + vision worker
    def loop(seconds, with_vision):
        worker = VisionWorker(det, model, float(vc["still_time"])) if with_vision else None
        ts, hand_ms = [], []
        for f in frames(seconds):
            img = cv2.flip(f.image, 1)
            if hands is not None:
                s = hands.process(img, f.t)
                hand_ms.append(s.debug["detect_ms"])
            if worker is not None:
                small = cv2.resize(img, (pw, int(img.shape[0] * pw / img.shape[1])), interpolation=cv2.INTER_AREA)
                worker.submit(small, f.t, f.seq, True)
            ts.append(f.t)
        wms = list(worker.ms_history) if worker else []
        done = worker.processed if worker else 0
        if worker:
            worker.close()
        rate = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 else 0
        return rate, hand_ms, wms, done

    r2, h2, _, _ = loop(args.seconds, False)
    print(f"\n[2] main loop, hand tracking only: {r2:.1f} fps processed; hand detect {stats(h2)}")
    r3, h3, w3, done = loop(args.seconds, True)
    print(f"[3] main loop, hands + vision worker: {r3:.1f} fps processed; hand detect {stats(h3)}")
    print(f"    vision worker: {done} frames analysed ({done / args.seconds:.1f}/s), {stats(w3)}")
    print(f"\ncapture rate during test: {cam.measured_fps:.1f} fps")
    cam.close()
    if hands is not None:
        hands.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
