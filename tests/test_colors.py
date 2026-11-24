import itertools

import numpy as np

from vision.calibration import CalibrationSession
from vision.colors import (COLORS, DEFAULT_BGR, ColorModel, balanced_assign, bgr_to_lab, hungarian,
                           read_stickers, sample_patch_lab)


def test_default_model_classifies_reference_colours():
    m = ColorModel.default()
    for c in COLORS:
        r = m.classify(bgr_to_lab(DEFAULT_BGR[c]))
        assert r.label == c and r.confidence > 0.8 and r.ambiguous is None


def test_red_orange_midpoint_is_flagged_not_guessed():
    m = ColorModel.default()
    mid = (m.means["R"] + m.means["O"]) / 2
    r = m.classify(mid)
    assert r.label in "RO"
    assert r.ambiguous == ("R", "O")
    assert r.confidence < 0.5
    assert r.text == "R/O?"


def test_far_from_every_colour_gets_low_confidence():
    m = ColorModel.default()
    magenta = bgr_to_lab((200, 0, 200))
    assert m.classify(magenta).confidence < 0.5


def test_skin_is_reported_as_unknown():
    m = ColorModel.default()
    skin = bgr_to_lab((115, 150, 205))
    r = read_stickers([skin], [1.0], m)[0]
    assert r.label == "?" and r.confidence == 0.0


def test_patch_sampling_ignores_glare():
    patch = np.full((14, 14, 3), DEFAULT_BGR["R"], np.uint8)
    patch[2:5, 2:6] = 255   # specular highlight
    lab, spread = sample_patch_lab(patch)
    assert np.linalg.norm(lab - bgr_to_lab(DEFAULT_BGR["R"])) < 3


def test_calibration_roundtrip(tmp_path):
    sess = CalibrationSession(frames_needed=2)
    rng = np.random.default_rng(1)
    tint = np.array([0.8, 0.95, 1.15])   # warm light
    for c in sess.order:
        base = bgr_to_lab(np.clip(np.array(DEFAULT_BGR[c]) * tint, 0, 255))
        while sess.current == c:
            labs = np.array([base + rng.normal(0, 1.5, 3) for _ in range(9)])
            sess.add_current(labs, [0.9] * 9)
    assert sess.done
    model = sess.build()
    assert model.calibrated and all(model.info["samples"][c] >= 18 for c in COLORS)
    path = tmp_path / "calibration.json"
    model.save(path)
    loaded, warn = ColorModel.load(path)
    assert warn is None and loaded.calibrated
    for c in COLORS:
        np.testing.assert_allclose(loaded.means[c], model.means[c], atol=1e-2)
        lab = bgr_to_lab(np.clip(np.array(DEFAULT_BGR[c]) * tint, 0, 255))
        assert loaded.classify(lab).label == c


def test_calibration_scrambled_face_uses_only_matching_stickers():
    sess = CalibrationSession()
    labs = np.array([bgr_to_lab(DEFAULT_BGR[c]) for c in "RGBOWYRGB"])   # centre is W
    added = sess.add("W", labs, [1.0] * 9)
    assert added == 1


def test_broken_calibration_file_falls_back(tmp_path):
    p = tmp_path / "calibration.json"
    p.write_text("{not json")
    model, warn = ColorModel.load(p)
    assert not model.calibrated and warn


def test_hungarian_matches_brute_force():
    rng = np.random.default_rng(3)
    for n in (2, 4, 6):
        for _ in range(20):
            cost = rng.random((n, n))
            sol = hungarian(cost)
            best = min(itertools.permutations(range(n)), key=lambda p: sum(cost[i, p[i]] for i in range(n)))
            assert abs(sum(cost[i, sol[i]] for i in range(n)) - sum(cost[i, best[i]] for i in range(n))) < 1e-9


def test_balanced_assign_gives_nine_of_each():
    rng = np.random.default_rng(0)
    labs = [bgr_to_lab(DEFAULT_BGR[c]) + rng.normal(0, 4, 3) for c in COLORS for _ in range(9)]
    out = balanced_assign(labs, ColorModel.default())
    assert all(out.count(c) == 9 for c in COLORS)
