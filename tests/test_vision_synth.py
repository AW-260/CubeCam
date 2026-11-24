"""Detection + colour accuracy on synthetic cube images (vision/synth.py).

Numbers are printed (run `pytest -s tests/test_vision_synth.py` to see them);
`python tools/eval_synthetic.py` runs the same evaluation on more scenes.
"""
import numpy as np
import pytest

import cube
from cube import SOLVED, scrambled_state, to_colors
from vision.colors import ColorModel
from vision.evaluate import calibrate_synthetic, evaluate, evaluate_false_positives
from vision.face_detect import FaceDetector
from vision.scan import check_face
from vision.stability import StabilityTracker
from vision.synth import Lighting, render_scene
from vision.worker import VisionWorker

DET = FaceDetector()


def test_detection_and_colour_accuracy_random_scenes():
    """Random pose (+-30 deg yaw/pitch/roll), 30% stickerless, 0-3 fingers, random light/blur/noise/JPEG."""
    r = evaluate(100, seed=101, detector=DET)
    print("\n" + r.summary("random scenes"))
    assert r.detection_rate >= 0.90
    assert r.wrong_face <= 2
    assert r.confident_accuracy >= 0.99
    assert r.coverage >= 0.85
    assert r.ro_confident_wrong <= 1


def test_clean_black_cube_no_fingers():
    r = evaluate(40, seed=7, detector=DET, stickerless=False, fingers=0)
    print("\n" + r.summary("black cube, no fingers"))
    assert r.detection_rate >= 0.95
    assert r.confident_accuracy >= 0.995
    assert r.coverage >= 0.9


def test_stickerless_with_fingers():
    r = evaluate(40, seed=8, detector=DET, stickerless=True, fingers=2)
    print("\n" + r.summary("stickerless, 2 fingers"))
    assert r.detection_rate >= 0.8
    assert r.confident_accuracy >= 0.98
    assert r.occluded_leak <= 0.3


def test_no_false_faces_in_cluttered_scenes():
    hits, n = evaluate_false_positives(100, seed=33, detector=DET)
    print(f"\nfalse positives: {hits}/{n} cube-free cluttered frames")
    assert hits <= 5


def test_calibration_beats_defaults_in_warm_dim_light():
    light = Lighting(gain=0.7, tint=(0.8, 0.95, 1.12), gamma=1.05, blur=0.7, noise=4, jpeg=80, vignette=0.2)
    d = evaluate(40, seed=21, detector=DET, fixed_lighting=light)
    model = calibrate_synthetic(np.random.default_rng(9), light, DET)
    c = evaluate(40, seed=21, detector=DET, fixed_lighting=light, model=model)
    print("\n" + d.summary("warm+dim, default colours"))
    print(c.summary("warm+dim, calibrated"))
    assert model.calibrated
    assert c.confident_accuracy >= 0.99
    assert c.coverage >= d.coverage
    assert c.raw_accuracy >= d.raw_accuracy


@pytest.mark.parametrize("roll", [0, 90, 180, -90])
def test_scramble_check_end_to_end_with_rotated_face(roll):
    """Render the scrambled cube with the face rotated in the image; the check must still pass."""
    scramble = "R U2 F' L D B2 R' U"
    colors = to_colors(scrambled_state(scramble))
    rng = np.random.default_rng(abs(roll) + 1)
    worker = VisionWorker(DET, ColorModel.default(), still_time=0.1)
    try:
        sc = render_scene(rng, colors54=colors, fingers=0, stickerless=False, pose=(8, -6, roll),
                          lighting=Lighting(blur=0.5, noise=2, jpeg=90))
        res = None
        for i in range(5):
            res = worker.analyse(sc.image, t=i * 0.033)
        assert res is not None and res.face is not None
        chk = check_face(res.face, colors)
        assert chk.face == "F" and chk.status == "ok", (chk, res.face.colors)
        # and a cube that is one move off is caught
        wrong = to_colors(scrambled_state(scramble + " R"))
        assert check_face(res.face, wrong).status in ("mismatch", "no_centre")
    finally:
        worker.close()


def test_mirrored_frame_gives_true_face_order():
    import cv2

    rng = np.random.default_rng(5)
    colors = to_colors(scrambled_state("R U F' D2"))
    sc = render_scene(rng, colors54=colors, fingers=0, stickerless=False, pose=(5, 5, 10),
                      lighting=Lighting(blur=0.4, noise=2, jpeg=92))
    worker = VisionWorker(DET, ColorModel.default(), still_time=0.05)
    try:
        img = cv2.flip(sc.image, 1)       # what the selfie display shows
        res = None
        for i in range(4):
            res = worker.analyse(img, t=i * 0.03, mirrored=True)
        assert res.face is not None
        assert res.face.colors == list(cube.face(colors, "F"))
    finally:
        worker.close()


def test_stability_requires_stillness():
    rng = np.random.default_rng(2)
    sc = render_scene(rng, colors54=to_colors(SOLVED), fingers=0, stickerless=False, pose=(0, 0, 5),
                      lighting=Lighting(blur=0.4, noise=2, jpeg=92))
    det = DET.detect(sc.image)
    assert det is not None
    tr = StabilityTracker(still_time=0.3)
    out = [tr.update(i / 30, det) for i in range(12)]
    assert all(o is None for o in out[:9]) and out[-1] is not None       # still for >= 0.3 s
    # moving cube: shift the detection a little every frame -> never stable
    tr.reset()
    import copy

    for i in range(20):
        d = copy.copy(det)
        d.centers = det.centers + np.array([i * det.pitch * 0.2, 0])
        assert tr.update(i / 30, d) is None
    # losing the face for longer than the gap resets
    tr.reset()
    for i in range(6):
        tr.update(i / 30, det)
    tr.update(0.5, None)
    assert tr.update(0.52, det) is None


def test_detection_speed_on_synthetic_frames():
    import time

    rng = np.random.default_rng(12)
    imgs = [render_scene(rng).image for _ in range(15)]
    for im in imgs[:2]:
        DET.detect(im)
    t0 = time.perf_counter()
    for im in imgs:
        DET.detect(im)
    ms = (time.perf_counter() - t0) / len(imgs) * 1000
    print(f"\nface detection: {ms:.1f} ms/frame at 640x360")
    assert ms < 40
