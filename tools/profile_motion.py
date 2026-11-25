"""Does stage 3 slow the timer down? Live camera, same settings as the app.

    python tools/profile_motion.py [--seconds 10] [--config config.yaml]

Runs the SOLVING main loop (grab -> mirror -> hand tracking) for --seconds:
  A. hand tracking only (stage 3 off),
  B. hand tracking + the stage-3 motion worker fed every frame (stage 3 on),
  C. A again, to check the two A/B runs were not affected by something else.
Prints the processed hand-tracking FPS (what the timer gets), hand-detection ms,
the main-thread cost of handing a frame to the worker, and the worker's own
speed and dropped frames. Hold a cube with both hands and turn it during the run
for realistic numbers.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from camera import Camera  # noqa: E402
from config import load_config  # noqa: E402
from detection import Zone  # noqa: E402
from vision.motion import MotionConfig, MotionWorker  # noqa: E402


def stats(xs):
    a = np.array(xs) if len(xs) else np.zeros(1)
    return f"median {np.median(a):.2f} ms, p95 {np.percentile(a, 95):.2f} ms, max {a.max():.2f} ms (n={len(xs)})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args()
    cfg, _ = load_config(args.config)
    cc = cfg["camera"]
    cam = Camera(int(cc["index"]), int(cc["width"]), int(cc["height"]), int(cc["fps"]), str(cc["backend"]),
                 cc.get("fourcc") or None)
    if not cam.open():
        print("camera error:", cam.error)
        return 1
    time.sleep(1.0)
    print(f"camera {cam.width}x{cam.height} via {cam.backend_name}, capture {cam.measured_fps:.1f} fps")
    hands = None
    try:
        from hands import HandZoneDetector

        dcfg = dict(cfg["detection"])
        dcfg["model_path"] = str(ROOT / dcfg["model_path"])
        hands = HandZoneDetector(Zone(**cfg["zone"]), dcfg)
    except Exception as exc:  # noqa: BLE001
        print("hand tracking unavailable:", exc)
    mirror = bool(cc["mirror"])

    def loop(seconds: float, worker: MotionWorker | None):
        ts, hand_ms, feed_ms, seen = [], [], [], 0
        last, end = 0, time.perf_counter() + seconds
        if worker is not None:
            worker.begin()
        while time.perf_counter() < end:
            f = cam.wait_new(last, 0.2)
            if f is None:
                continue
            last = f.seq
            img = cv2.flip(f.image, 1) if mirror else f.image.copy()
            lms = None
            if hands is not None:
                s = hands.process(img, f.t)
                hand_ms.append(s.debug["detect_ms"])
                lms = s.debug.get("hands")
                seen += len(lms) >= 2
            if worker is not None:
                t0 = time.perf_counter()
                worker.submit(f.image, f.t, lms, mirror)
                feed_ms.append((time.perf_counter() - t0) * 1000)
            ts.append(f.t)
        out = None
        if worker is not None:
            out = worker.finish()
        rate = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 else 0.0
        return rate, hand_ms, feed_ms, out, seen / max(1, len(ts))

    worker = MotionWorker(MotionConfig.from_dict(cfg["motion"]))
    try:
        ra, ha, _, _, sa = loop(args.seconds, None)
        print(f"\n[A] stage 3 OFF: processed {ra:.1f} fps | hand detect {stats(ha)} | both hands in {sa:.0%} of frames")
        rb, hb, fb, out, sb = loop(args.seconds, worker)
        m = out["meta"]
        print(f"[B] stage 3 ON : processed {rb:.1f} fps | hand detect {stats(hb)} | both hands in {sb:.0%} of frames")
        print(f"    main-thread cost of submit(): {stats(fb)}")
        print(f"    worker: {m['processed']} of {m['submitted']} frames analysed, {m['dropped']} dropped, "
              f"median {m['ms_median'] or 0:.1f} ms/frame (p95 {m['ms_p95'] or 0:.1f})")
        rc, hc, _, _, _ = loop(args.seconds, None)
        print(f"[C] stage 3 OFF: processed {rc:.1f} fps | hand detect {stats(hc)}")
        base = (ra + rc) / 2
        print(f"\nprocessed FPS with stage 3 vs without: {rb:.1f} vs {base:.1f} ({(rb - base) / max(base, 1e-6) * 100:+.1f}%), "
              f"capture {cam.measured_fps:.1f} fps")
    finally:
        worker.close()
        cam.close()
        if hands is not None:
            hands.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
