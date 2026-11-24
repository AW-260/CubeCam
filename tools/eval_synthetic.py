"""Accuracy report for face detection + colour reading on synthetic scenes.

    python tools/eval_synthetic.py [--n 300] [--seed 0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision.evaluate import calibrate_synthetic, evaluate, evaluate_false_positives  # noqa: E402
from vision.face_detect import FaceDetector  # noqa: E402
from vision.synth import Lighting  # noqa: E402

HARD_LIGHT = {
    "warm (tungsten)": Lighting(gain=1.0, tint=(0.78, 0.95, 1.12), gamma=1.0, blur=0.6, noise=3, jpeg=85, vignette=0.15),
    "cool (daylight)": Lighting(gain=1.0, tint=(1.15, 1.02, 0.85), gamma=1.0, blur=0.6, noise=3, jpeg=85, vignette=0.15),
    "dim": Lighting(gain=0.62, tint=(1.0, 1.0, 1.0), gamma=1.1, blur=0.8, noise=5, jpeg=80, vignette=0.25),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    det = FaceDetector()
    print(evaluate(a.n, a.seed, detector=det).summary("random scenes (pose +-30, 30% stickerless, 0-3 fingers)"))
    print(evaluate(a.n // 3, a.seed + 1, detector=det, stickerless=False, fingers=0).summary("black cube, no fingers"))
    print(evaluate(a.n // 3, a.seed + 2, detector=det, stickerless=True, fingers=0).summary("stickerless, no fingers"))
    print(evaluate(a.n // 3, a.seed + 3, detector=det, fingers=3).summary("3 fingers"))
    hits, n = evaluate_false_positives(a.n, a.seed + 4, detector=det)
    print(f"false positives: {hits}/{n} cube-free cluttered frames")
    for name, light in HARD_LIGHT.items():
        d = evaluate(a.n // 3, a.seed + 5, detector=det, fixed_lighting=light)
        m = calibrate_synthetic(np.random.default_rng(a.seed + 6), light, det)
        c = evaluate(a.n // 3, a.seed + 5, detector=det, fixed_lighting=light, model=m)
        print(d.summary(f"{name}, default colours"))
        print(c.summary(f"{name}, calibrated"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
