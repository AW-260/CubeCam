"""Measure the UI's cost in the real loop: processed FPS and render time, AR mat off / on.

    python tools/profile_ui.py --seconds 6

Runs the app's real per-frame work on the live camera (capture -> hand
tracking -> state machine -> build_view -> render), without showing a window,
in four phases: AR mat off, AR mat on with two (simulated) hands over the mat,
off again, on without hands. The table plane is calibrated from the synthetic
phone picture used by tools/ar_screenshots.py, so the mat is drawn even if your
camera doesn't see a table; the hands are fake landmarks over the mat so the
hand cut-out runs too. Nothing is saved (a temporary config is used).
"""
from __future__ import annotations

import argparse
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=6.0, help="length of each phase")
    args = ap.parse_args(argv)

    import main as M
    from ar_mat import mat_at_image_point
    from ar_screenshots import build_scene
    from detection import TriggerSample
    from markers import detect_marker_corners, make_detector
    from synth_table import hand_landmarks
    from table_calib import CalibMode, TableCalibration

    tmp = Path(tempfile.mkdtemp())
    text = (ROOT / "config.yaml").read_text(encoding="utf-8")
    text = text.replace("path: solves.json", f"path: {(tmp / 'solves.json').as_posix()}")
    (tmp / "config.yaml").write_text(text, encoding="utf-8")
    app = M.App(tmp / "config.yaml")
    if app.camera is None:
        print("no camera")
        return 1
    W, H = app.camera.width, app.camera.height
    app._canvas = (W, H)

    # calibrate from the synthetic phone picture, place a mat
    view, _, marker_mm = build_scene()
    view = cv2.resize(view, (W, H))
    corners = detect_marker_corners(cv2.flip(view, 1), make_detector(refine=True), mirror=True, process_width=0)
    calib = TableCalibration(phone_marker_mm=marker_mm)
    calib.start(CalibMode.PHONE, (W, H))
    for i in range(6):
        calib.update_phone(i * 0.1, corners, (W, H))
    app.plane = calib.accept()
    app.mat = mat_at_image_point(app.plane, (0.5, 0.75), *app.mat_size)
    app.zone_mode = "mat"
    app.set_zone(app.mat.zone(app.plane))
    quad = app.mat.image_quad(app.plane)
    cx, cy = quad.mean(axis=0)
    hands = [hand_landmarks((cx - 150, cy + 10), 90, right=False) / (W, H), hand_landmarks((cx + 150, cy + 10), 90) / (W, H)]

    def phase(seconds, ar_on, fake_hands):
        app.ar_enabled = ar_on
        last, ts, render_ms, detect_ms = 0, [], [], []
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            f = app.camera.wait_new(last, 0.2)
            if f is None:
                continue
            last = f.seq
            t = f.t
            img = cv2.flip(f.image, 1) if app.mirror else f.image.copy()
            t0 = time.perf_counter()
            sample = app.detector.process(img, t)
            detect_ms.append((time.perf_counter() - t0) * 1000)
            app.loop_rate.tick(t)
            app.handle_events(app.machine.update(t, sample.active))
            if fake_hands is not None:
                sample = TriggerSample(t, sample.active, dict(sample.debug, hands=fake_hands))
            t0 = time.perf_counter()
            vm = app.build_view(t, sample, False, img.shape[1])
            app.renderer.render(img, vm)
            render_ms.append((time.perf_counter() - t0) * 1000)
            ts.append(t)
        fps = (len(ts) - 1) / (ts[-1] - ts[0])
        return fps, render_ms, detect_ms

    print(f"camera {W}x{H}, capture {app.camera.measured_fps:.1f} fps")
    rows = []
    for label, ar_on, fh in (("AR off", False, None), ("AR on, 2 hands over the mat", True, hands),
                             ("AR off again", False, None), ("AR on, no hands", True, None)):
        phase(0.8, ar_on, fh)                 # settle, let background warps finish
        fps, rms, dms = phase(args.seconds, ar_on, fh)
        rows.append((label, fps, rms, dms))
        print(f"{label:28s} processed {fps:5.1f} fps | render median {statistics.median(rms):5.2f} ms, "
              f"p95 {np.percentile(rms, 95):5.2f} ms | hand tracking median {statistics.median(dms):5.1f} ms")
    print(f"camera capture {app.camera.measured_fps:.1f} fps")
    app.close()
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
