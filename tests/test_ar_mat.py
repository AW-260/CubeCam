"""AR mat: table plane, phone / 4-click calibration, mat placement, zone, rendering, hand occlusion."""
import importlib.util
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from ar_mat import (
    ArMatRenderer,
    Mat,
    hand_mask,
    mat_at_image_point,
    mat_fits,
    mat_from_hands,
    mat_texture,
    reanchor,
)
from config import load_config, replace_block, save_blocks
from markers import detect_marker_corners, make_detector, plane_from_marker
from plane import TablePlane, _h_apply, estimate_focal, focal_from_hfov, plane_from_rectangle
from synth_table import SynthCamera, TableScene, draw_hand, hand_landmarks, phone_image
from table_calib import EVENT_LBUTTONDOWN, EVENT_RBUTTONDOWN, CalibMode, TableCalibration, is_simple_quad, parse_size
from zone_setup import EVENT_LBUTTONUP, EVENT_MOUSEMOVE, SetupMode, ZoneSetup

ROOT = Path(__file__).resolve().parents[1]
W, H = 1280, 720


def _tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


make_phone_marker = _tool("make_phone_marker").make_phone_marker


def mirror_px(p):
    p = np.array(p, dtype=np.float64).reshape(-1, 2)
    p[:, 0] = (W - 1) - p[:, 0]
    return p


def table_dist_error(plane, H_true, pairs, mirror=False):
    """Relative error of table distances measured through `plane` (from image points of known table points)."""
    errs = []
    for a, b in pairs:
        img, _ = _h_apply(H_true, np.array([a, b], dtype=np.float64))
        if mirror:
            img = mirror_px(img)
        t, ok = plane.to_table(img)
        assert ok.all()
        errs.append(np.linalg.norm(t[0] - t[1]) / np.linalg.norm(np.subtract(a, b)) - 1)
    return np.array(errs)


PAIRS = [((-20, 45), (20, 75)), ((-25, 40), (25, 40)), ((0, 35), (0, 80))]


def synthetic_plane(tilt=35, height=40, hfov=70):
    """A plane straight from the synthetic camera (exact), expressed in the canonical frame.

    Canonical table coordinates: origin at the reference centre (synthetic (0, 55)),
    +X = screen right, +Y = towards the viewer, i.e. canonical (x, y) = synthetic (x, 55 - y).
    """
    cam = SynthCamera(tilt_deg=tilt, height_cm=height, hfov_deg=hfov)
    Ht = cam.H_table_to_image()
    sq = np.array([(-5, 50), (5, 50), (5, 60), (-5, 60)], float)
    img, _ = _h_apply(Ht, sq)
    return TablePlane.from_reference(sq, img, (W, H), "synthetic"), Ht


# ======================================================================= plane
def test_homography_round_trip():
    plane, _ = synthetic_plane()
    rng = np.random.default_rng(0)
    pts = rng.uniform([-30, -20], [30, 20], (50, 2))
    img, ok = plane.to_image(pts)
    assert ok.all()
    back, ok2 = plane.to_table(img)
    assert ok2.all() and np.allclose(back, pts, atol=1e-7)
    img2, ok3 = plane.to_image_norm(pts)
    back2, _ = plane.to_table_norm(img2)
    assert np.allclose(back2, pts, atol=1e-7)


def test_plane_measures_true_distances_and_canonical_axes():
    plane, Ht = synthetic_plane()
    assert np.allclose(table_dist_error(plane, Ht, PAIRS), 0, atol=1e-5)
    o, x, y = plane.to_image(np.array([[0, 0], [10, 0], [0, 10]]))[0]
    assert x[0] - o[0] > 20 and abs(x[1] - o[1]) < 1e-6    # +X = screen right
    assert y[1] - o[1] > 10                                  # +Y = towards the bottom of the screen
    # mirrored view: still right / down, same handedness
    sq = np.array([(-5, 50), (5, 50), (5, 60), (-5, 60)], float)
    img = mirror_px(_h_apply(Ht, sq)[0])
    m = TablePlane.from_reference(sq, img, (W, H))
    o, x, y = m.to_image(np.array([[0, 0], [10, 0], [0, 10]]))[0]
    assert x[0] > o[0] and y[1] > o[1]
    assert np.allclose(table_dist_error(m, Ht, PAIRS, mirror=True), 0, atol=1e-5)


def test_horizon_and_behind_camera_are_invalid():
    plane, _ = synthetic_plane(tilt=20)
    _, ok = plane.to_table(np.array([[640, 5]]))          # above the horizon at 20 deg tilt
    assert not ok[0]
    _, ok = plane.to_table(np.array([[640, 700]]))
    assert ok[0]
    _, ok = plane.to_image(np.array([[0, 500]]))           # 5 m towards the viewer: behind the camera
    assert not ok[0]


def test_scaled_to_other_resolution_and_config_round_trip():
    plane, _ = synthetic_plane()
    small = plane.scaled_to((640, 360))
    pts = np.array([[0, 0], [12, -7]])
    assert np.allclose(small.to_image_norm(pts)[0], plane.to_image_norm(pts)[0])
    d = plane.to_config()
    yaml_text = yaml.safe_dump({"table_calibration": d})
    again = TablePlane.from_config(yaml.safe_load(yaml_text)["table_calibration"])
    assert np.allclose(again.to_image(pts)[0], plane.to_image(pts)[0], atol=1e-4)
    d["homography"] = (-np.array(d["homography"])).tolist()  # stored with the other sign: still works
    flipped = TablePlane.from_config(d)
    assert flipped.to_image(pts)[1].all()
    assert TablePlane.from_config(None) is None
    with pytest.raises(ValueError):
        TablePlane.from_config({"homography": [[1, 0], [0, 1]], "image_size": [W, H]})


def test_elevation_estimate_and_warnings():
    plane, _ = synthetic_plane(tilt=35, height=40)
    # reference centre is at (0, 55): elevation atan(40 / 55) ~ 36 deg
    assert abs(plane.elevation_deg() - math.degrees(math.atan(40 / 55))) < 2.0
    assert plane.warnings() == []
    shallow, _ = synthetic_plane(tilt=6, height=8)
    assert any("edge-on" in w for w in shallow.warnings())
    cam = SynthCamera(tilt_deg=35, height_cm=40)
    tiny = np.array([(0, 90), (1.5, 90), (1.5, 91.5), (0, 91.5)])
    p = TablePlane.from_reference(tiny, _h_apply(cam.H_table_to_image(), tiny)[0], (W, H))
    assert any("very small" in w for w in p.warnings())


def test_focal_estimate_from_a_view():
    for hfov in (55, 70, 85):
        cam = SynthCamera(tilt_deg=35, hfov_deg=hfov)
        f = estimate_focal(cam.H_table_to_image(), (W, H))
        assert f == pytest.approx(focal_from_hfov(hfov, W), rel=1e-6)


# ============================================================ phone marker
def phone_scene(tilt=32, pos=(22, 55), angle=25, hfov=70, ppc=16, glare=False):
    cam = SynthCamera(tilt_deg=tilt, height_cm=38, hfov_deg=hfov)
    scene = TableScene(cam, px_per_cm=ppc)
    png = make_phone_marker()
    body, mask = phone_image(png, 600)
    phone_w = 7.2
    scene.paste(body, pos, phone_w, angle, mask)
    raw = scene.render()
    if glare:
        from synth_table import add_glare

        c = _h_apply(scene.H_true(), np.array([pos], float))[0][0]
        raw = add_glare(raw, c + (25, -10), 30, 0.35)
    side_px = 6 * (int(1080 * 0.82) // 6)
    marker_cm = phone_w * (600 / (600 + 2 * int(600 * 0.06))) * side_px / 1080
    return raw, scene.H_true(), marker_cm


def test_phone_marker_image():
    img = make_phone_marker()
    assert img.shape == (2340, 1080)
    found = detect_marker_corners(img, make_detector(), mirror=False, process_width=0)
    assert set(found) == {10}
    c = found[10] * (1080, 2340)
    side = np.linalg.norm(c[1] - c[0])
    assert 0.75 * 1080 < side < 0.85 * 1080   # big, with a white quiet zone round it


@pytest.mark.parametrize("mirror", [False, True])
def test_phone_on_table_at_an_angle_calibrates_the_plane(mirror):
    raw, Ht, marker_cm = phone_scene()
    det = make_detector(refine=True)
    corners = detect_marker_corners(raw, det, mirror=mirror, process_width=0)
    assert 10 in corners
    calib = TableCalibration(phone_marker_mm=marker_cm * 10)
    calib.start(CalibMode.PHONE, (W, H))
    for i in range(5):                      # a few steady detections
        calib.update_phone(i * 0.12, corners, (W, H))
    v = calib.view()
    assert v.marker_px is not None and v.grid_px and "found" in " ".join(v.lines)
    plane = calib.accept()
    assert plane is not None and plane.source == "phone" and not calib.active
    errs = table_dist_error(plane, Ht, PAIRS, mirror=mirror)
    assert np.max(np.abs(errs)) < 0.05, errs              # within 5 % over ~50 cm from a 5 cm marker
    assert abs(plane.elevation_deg() - 31) < 4
    assert plane.warnings() == []


def test_phone_calibration_accuracy_over_several_poses():
    det = make_detector(refine=True)
    worst = []
    for tilt, pos, ang in ((45, (-10, 50), -60), (30, (-25, 45), 10), (40, (15, 40), -30), (25, (5, 60), 90)):
        raw, Ht, marker_cm = phone_scene(tilt, pos, ang)
        c = detect_marker_corners(raw, det, mirror=False, process_width=0)
        assert 10 in c, (tilt, pos)
        plane = plane_from_marker(c[10], marker_cm, (W, H), "phone")
        worst.append(np.max(np.abs(table_dist_error(plane, Ht, PAIRS))))
    assert np.median(worst) < 0.04 and max(worst) < 0.12, worst


def test_phone_with_glare_still_found():
    raw, Ht, marker_cm = phone_scene(glare=True)
    c = detect_marker_corners(raw, make_detector(refine=True), mirror=False, process_width=0)
    assert 10 in c


def test_phone_moving_restarts_averaging_and_default_scale_is_flagged():
    calib = TableCalibration(phone_marker_mm=None)
    calib.start(CalibMode.PHONE, (W, H))
    sq = np.array([[0.4, 0.6], [0.45, 0.6], [0.45, 0.68], [0.4, 0.68]])
    for i in range(4):
        calib.update_phone(i * 0.1, {10: sq}, (W, H))
    assert len(calib._samples) == 4
    calib.update_phone(0.5, {10: sq + 0.1}, (W, H))         # phone moved a lot
    assert len(calib._samples) == 1
    calib.update_phone(0.6, {3: sq}, (W, H))                 # another marker id: ignored
    assert len(calib._samples) == 1
    assert any("approximate" in line for line in calib.view().lines)


# ============================================================ 4-click mode
def test_parse_size_and_quad_order():
    assert parse_size("29.7x21") == (29.7, 21.0)
    assert parse_size(" 21 * 29,7 ") == (21.0, 29.7)
    assert parse_size("29.7") is None and parse_size("0x5") is None
    assert is_simple_quad(np.array([[0, 0], [10, 0], [10, 5], [0, 5]]))
    assert not is_simple_quad(np.array([[0, 0], [10, 5], [10, 0], [0, 5]]))   # bow-tie


def _click(calib, pts):
    for x, y in pts:
        calib.on_mouse(EVENT_LBUTTONDOWN, x, y, W, H)


def test_four_click_calibration_with_a4_preset_and_typed_size():
    cam = SynthCamera(tilt_deg=35, height_cm=40)
    Ht = cam.H_table_to_image()
    # an A4 sheet in landscape on the table, corners clicked round its edge (with ~1 px of hand jitter)
    a4 = np.array([(-14.85, 60.5), (14.85, 60.5), (14.85, 39.5), (-14.85, 39.5)])  # far edge first
    clicks = _h_apply(Ht, a4)[0] + np.random.default_rng(3).normal(0, 0.7, (4, 2))
    calib = TableCalibration()
    calib.start(CalibMode.CLICK, (W, H))
    _click(calib, clicks[:3])
    calib.on_mouse(EVENT_RBUTTONDOWN, 0, 0, W, H)            # undo the third click
    assert len(calib.clicks) == 2
    _click(calib, clicks[2:])
    assert len(calib.clicks) == 4 and calib.candidate is None
    assert calib.view().mouse_px is None                       # loupe only while clicking corners
    calib.on_key(ord("a"))                                    # A4: 29.7 x 21
    assert calib.size_text == "29.7x21" and calib.candidate is not None
    errs = table_dist_error(calib.candidate, Ht, PAIRS)
    assert np.max(np.abs(errs)) < 0.03, errs
    calib.on_key(ord("a"))                                    # pressed again: turned 90 degrees
    assert calib.size_text == "21x29.7"
    wrong = table_dist_error(calib.candidate, Ht, PAIRS)
    assert np.max(np.abs(wrong)) > 0.1                       # the wrong way round is clearly wrong
    for _ in range(7):
        calib.on_key(8)                                       # backspace the text
    for ch in "29.7x21":
        calib.on_key(ord(ch))
    assert calib.candidate is not None
    plane = calib.accept()
    assert plane.source == "click" and np.max(np.abs(table_dist_error(plane, Ht, PAIRS))) < 0.03


def test_four_click_bow_tie_is_rejected():
    calib = TableCalibration()
    calib.start(CalibMode.CLICK, (W, H))
    _click(calib, [(400, 400), (800, 500), (800, 400), (400, 500)])
    calib.on_key(ord("a"))
    assert calib.candidate is None
    assert any("in order" in w for w in calib.view().warnings)


def test_click_loupe_follows_mouse():
    calib = TableCalibration()
    calib.start(CalibMode.CLICK, (W, H))
    calib.on_mouse(0, 300, 200, W, H)
    assert calib.view().mouse_px == (300, 200)
    calib.on_mouse(0, 150, 100, 640, 360)      # window at half size: still calibration pixels
    assert calib.view().mouse_px == (300, 200)


def test_calibration_cycle():
    calib = TableCalibration()
    calib.start()
    assert calib.mode is CalibMode.PHONE
    assert calib.cycle() is CalibMode.CLICK
    assert calib.cycle() is None and not calib.active


# ================================================================ the mat
def test_mat_corners_rotation_and_config():
    m = Mat(0, 0, 0, 30, 22)
    c = m.corners()
    assert np.allclose(c, [(-15, -11), (15, -11), (15, 11), (-15, 11)])
    r = m.rotated(90)
    assert np.allclose(r.corners(), [(11, -15), (11, 15), (-11, 15), (-11, -15)])
    assert m.rotated(185).angle_deg == pytest.approx(-175)
    again = Mat.from_config(yaml.safe_load(yaml.safe_dump(m.moved_to(3, -4).rotated(12).to_config())))
    assert again == Mat(3, -4, 12, 30, 22)
    assert Mat.from_config(None) is None


def test_mat_on_plane_zone_is_projected_outline():
    plane, Ht = synthetic_plane()
    mat = mat_at_image_point(plane, (0.5, 0.75), 30, 22)
    assert mat is not None and mat_fits(mat, plane)
    zone = mat.zone(plane)
    quad = mat.image_quad(plane)
    assert np.allclose(np.array(zone.points) * (W, H), quad)
    # trapezoid in perspective: the far edge looks shorter than the near edge
    assert np.linalg.norm(quad[1] - quad[0]) < np.linalg.norm(quad[2] - quad[3])
    # palms resting on the mat are inside the zone, a palm beside it is not
    centre = plane.to_image_norm([[mat.cx, mat.cy]])[0][0]
    assert zone.contains(*centre)
    beside = plane.to_image_norm([[mat.cx + 20, mat.cy]])[0][0]
    assert not zone.contains(*beside)


def test_mat_beyond_horizon_rejected():
    plane, _ = synthetic_plane(tilt=20)
    assert mat_at_image_point(plane, (0.5, 0.01), 30, 22) is None
    behind = Mat(0, 400, 0, 30, 22)          # 4 m towards the viewer: behind the camera
    assert not mat_fits(behind, plane)


def test_drag_mat_slides_in_perspective_and_rotates():
    plane, _ = synthetic_plane()
    s = ZoneSetup()
    s.start(plane=plane, mat_size=(30, 22))
    assert s.mode is SetupMode.MAT and s.candidate_mat is not None
    m0 = s.candidate_mat
    c0 = plane.to_image([[m0.cx, m0.cy]])[0][0]
    s.on_mouse(EVENT_LBUTTONDOWN, c0[0], c0[1], W, H)        # grab the mat at its centre
    target = (c0[0] + 200, c0[1] - 60)
    s.on_mouse(EVENT_MOUSEMOVE, *target, W, H)
    s.on_mouse(EVENT_LBUTTONUP, *target, W, H)
    m1 = s.candidate_mat
    c1 = plane.to_image([[m1.cx, m1.cy]])[0][0]
    assert np.allclose(c1, target, atol=1.0)                  # the centre followed the mouse
    assert (m1.width_cm, m1.height_cm, m1.angle_deg) == (30, 22, 0)
    # it moved on the table: further away (smaller on screen) since we dragged up
    q0, q1 = m0.image_quad(plane), m1.image_quad(plane)
    assert np.linalg.norm(q1[1] - q1[0]) < np.linalg.norm(q0[1] - q0[0])
    assert s.rotate(5) and s.rotate(5)
    assert s.candidate_mat.angle_deg == 10
    assert np.allclose(np.array(s.candidate.points) * (W, H), s.candidate_mat.image_quad(plane))
    mode, zone, mat = s.accept()
    assert mode == "mat" and mat == s_mat(mat) and zone is not None


def s_mat(m):
    return m


def test_drag_above_horizon_keeps_mat_on_table():
    plane, _ = synthetic_plane(tilt=12, height=30)        # horizon inside the picture
    s = ZoneSetup()
    s.start(plane=plane, mat_size=(30, 22))
    before = s.candidate_mat
    assert before is not None
    c = plane.to_image([[before.cx, before.cy]])[0][0]
    s.on_mouse(EVENT_LBUTTONDOWN, c[0], c[1], W, H)
    s.on_mouse(EVENT_MOUSEMOVE, c[0], 2, W, H)               # far above the horizon
    assert s.candidate_mat == before
    s.on_mouse(EVENT_LBUTTONUP, c[0], 2, W, H)
    s.on_mouse(EVENT_LBUTTONDOWN, c[0], 2, W, H)             # clicking the sky
    assert "horizon" in s.note


def test_setup_without_plane_is_unchanged():
    s = ZoneSetup()
    s.start()
    assert s.mode is SetupMode.DRAG and s.candidate_mat is None


def test_hands_mode_builds_mat_on_plane_round_resting_hands():
    plane, _ = synthetic_plane()
    s = ZoneSetup(hold=0.5)
    s.start(SetupMode.HANDS, plane=plane, mat_size=(30, 22))
    # two hands resting 26 cm apart, the line between them turned by ~10 degrees on the table
    a = math.radians(10)
    left_t = np.array([-13 * math.cos(a), -13 * math.sin(a)]) + (0, 4)
    right_t = np.array([13 * math.cos(a), 13 * math.sin(a)]) + (0, 4)
    lp, rp = plane.to_image([left_t, right_t])[0]
    size = 70
    hands = [hand_landmarks(lp, size, right=False) / (W, H), hand_landmarks(rp, size) / (W, H)]
    for i in range(20):
        s.update(i / 30, hands)
    m = s.candidate_mat
    assert m is not None
    assert np.allclose((m.cx, m.cy), (left_t + right_t) / 2, atol=1.5)
    assert abs(m.angle_deg - 10) < 3
    assert m.width_cm == pytest.approx(26 + 2 * 6, abs=1.5)    # widened so both hands fit
    assert "widened" in s.view().lines[-2]
    mode, zone, mat = s.accept()
    assert mode == "mat"
    for p in (lp, rp):
        assert zone.contains(p[0] / W, p[1] / H)


def test_mat_from_hands_keeps_size_when_hands_fit():
    plane, _ = synthetic_plane()
    lp, rp = plane.to_image([[-9, 3], [9, 3]])[0]           # palms 18 cm apart
    hands = [hand_landmarks(lp, 60) / (W, H), hand_landmarks(rp, 60) / (W, H)]
    m = mat_from_hands(plane, hands, 30, 22)
    assert m.width_cm == 30 and m.height_cm == 22 and abs(m.angle_deg) < 1


def test_reanchor_keeps_mat_on_screen():
    plane, Ht = synthetic_plane()
    mat = Mat(4, 2, 15, 30, 22)
    # re-calibrate with a reference somewhere else: a different table frame
    sq = np.array([(20, 70), (26, 70), (26, 76), (20, 76)], float)
    other = TablePlane.from_reference(sq, _h_apply(Ht, sq)[0], (W, H))
    moved = reanchor(mat, plane, other)
    assert np.allclose(moved.image_quad(other), mat.image_quad(plane), atol=0.5)


# ================================================================ rendering
def test_texture_states():
    for state in ("idle", "holding", "ready", "solving"):
        tex, (x0, y0, x1, y1) = mat_texture(30, 22, state, 0.5)
        assert tex.shape[2] == 4
        assert tex[0, 0, 3] == 0                              # transparent margin
        assert tex[int(y0) + 2, int(x0) + 2, 3] < 255         # rounded corner
        assert tex[int((y0 + y1) / 2), int((x0 + x1) / 2), 3] == 255
    ready, (x0, y0, x1, y1) = mat_texture(30, 22, "ready")
    idle, _ = mat_texture(30, 22, "idle")
    gx, gy = int((x0 + x1) / 2), int(y0) - 6                   # just outside the far edge
    assert ready[gy, gx, 3] > 40 and idle[gy, gx, 3] == 0        # green glow only when ready


def test_renderer_caches_warps():
    plane, _ = synthetic_plane()
    quad = Mat(0, 0, 0, 30, 22).image_quad(plane)
    r = ArMatRenderer(background=False)
    img = np.full((H, W, 3), 120, np.uint8)
    r.draw(img, quad, (30, 22))
    n = r.warps
    assert n >= 4                                    # every state prepared once
    for state in ("idle", "holding", "ready", "solving"):
        r.draw(img.copy(), quad, (30, 22), state, 0.4)
    assert r.warps == n                              # nothing re-warped
    r.draw(img.copy(), quad + (3, 0), (30, 22))       # the mat moved: warped again
    assert r.warps > n
    mid = np.int32(quad.mean(axis=0))
    assert img[mid[1], mid[0]].max() < 90             # dark mat drawn in the middle


def test_renderer_background_thread_prepares_states():
    plane, _ = synthetic_plane()
    quad = Mat(0, 0, 0, 30, 22).image_quad(plane)
    r = ArMatRenderer()
    img = np.full((H, W, 3), 120, np.uint8)
    r.draw(img, quad, (30, 22), "idle")
    r.wait_idle()
    n = r.warps
    r.draw(img.copy(), quad, (30, 22), "ready")
    assert r.warps == n
    r.close()


def _scene_with_hands():
    raw, Ht, _ = phone_scene()
    plane = TablePlane.from_reference(*[np.array(v, float) for v in (
        [(-5, 50), (5, 50), (5, 60), (-5, 60)], _h_apply(Ht, np.array([(-5, 50), (5, 50), (5, 60), (-5, 60)], float))[0])],
        (W, H))
    mat = mat_at_image_point(plane, (0.5, 0.78), 30, 22)
    quad = mat.image_quad(plane)
    c = quad.mean(axis=0)
    hands = [hand_landmarks(c + (-140, 10), 90, -8, right=False), hand_landmarks(c + (140, 10), 90, 8)]
    with_hands = raw.copy()
    for h in hands:
        draw_hand(with_hands, h)
    return raw, with_hands, quad, hands


def test_hand_mask_covers_hands_not_table():
    raw, img, quad, hands = _scene_with_hands()
    m = hand_mask(img, hands)
    assert m is not None and m.shape == img.shape[:2]
    for h in hands:
        for i in (0, 9, 8, 12, 20):                    # wrist, palm, finger tips
            x, y = np.int32(np.round(h[i]))
            assert m[y, x] > 0.8, i
        palm = np.int32(np.round(h[[0, 5, 9, 13, 17]].mean(axis=0)))
        assert m[palm[1], palm[0]] > 0.95
    far = np.int32(np.round(quad.mean(axis=0) + (0, -60)))   # middle of the mat, between the hands
    assert m[far[1], far[0]] < 0.05
    assert ((m > 0.05) & (m < 0.95)).sum() > 200       # feathered edges
    # light-wood table round the hands is skin-coloured, but must not be cut out
    near_table = np.int32(np.round(hands[0][8] + (0, -35)))  # just beyond the index finger tip
    assert m[near_table[1], near_table[0]] < 0.3
    assert hand_mask(img, [h + (5000, 0) for h in hands]) is None   # hands outside the region


def test_skin_pixels_extend_the_landmark_shape():
    img = np.full((300, 400, 3), (60, 60, 60), np.uint8)    # grey table: clearly not skin
    lm = hand_landmarks((200, 170), 90)
    draw_hand(img, lm)
    # make the drawn fingers fatter than the landmark model expects
    for a, b in ((5, 8), (9, 12)):
        cv2.line(img, tuple(np.int32(lm[a])), tuple(np.int32(lm[b])), (120, 150, 205), 30)
    with_skin = hand_mask(img, [lm])
    without = hand_mask(img, [lm], skin=False)
    edge = np.int32(np.round((lm[6] + lm[7]) / 2 + (-14, 0)))   # outer side of the index finger
    assert with_skin[edge[1], edge[0]] > without[edge[1], edge[0]] + 0.3


def test_hands_are_drawn_on_top_of_the_mat():
    raw, img, quad, hands = _scene_with_hands()
    r = ArMatRenderer(background=False)
    r.draw(raw.copy(), quad, (30, 22))                  # one frame without hands: learns the table
    out = img.copy()
    r.draw(out, quad, (30, 22), "ready", 0, hands)
    for h in hands:
        palm = np.int32(np.round(h[[0, 5, 9, 13, 17]].mean(axis=0)))
        assert np.abs(out[palm[1], palm[0]].astype(int) - img[palm[1], palm[0]].astype(int)).max() <= 3
    mid = np.int32(np.round(quad.mean(axis=0) + (0, -60)))
    assert np.abs(out[mid[1], mid[0]].astype(int) - img[mid[1], mid[0]].astype(int)).max() > 40
    # timing: a steady mat with two hands costs a few ms per frame
    import time

    t0 = time.perf_counter()
    for _ in range(10):
        r.draw(img.copy(), quad, (30, 22), "ready", 0, hands)
    assert (time.perf_counter() - t0) / 10 < 0.015


# ================================================================== config
def test_save_blocks_round_trip(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text((ROOT / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    plane, _ = synthetic_plane()
    mat = Mat(1.5, -2, 7, 30, 22)
    z = mat.zone(plane)
    save_blocks(cfg, zone=("mat", z.to_config()), table_calibration=plane.to_config(),
                mat_placement=mat.to_config())
    c, warn = load_config(cfg)
    assert warn is None and c["zone"]["mode"] == "mat"
    p2 = TablePlane.from_config(c["table_calibration"])
    assert np.allclose(p2.to_image([[3, 4]])[0], plane.to_image([[3, 4]])[0], atol=1e-3)
    assert Mat.from_config(c["mat_placement"]) == mat
    assert c["table"]["phone_marker_id"] == 10 and c["timer"]["hold_time"] == 0.3   # rest kept
    text = cfg.read_text(encoding="utf-8")
    save_blocks(cfg, table_calibration=None)            # forget: a single null line
    c, _ = load_config(cfg)
    assert c["table_calibration"] is None and c["mat_placement"] is not None
    assert cfg.read_text(encoding="utf-8").count("table_calibration") == 1
    assert len(text) > 100


def test_replace_block_handles_null_and_flow_lines():
    text = "a: 1\nk: null  # comment\nb:\n  x: 2\n"
    out = replace_block(text, "k", ["k:", "  v: 3"])
    assert yaml.safe_load(out) == {"a": 1, "k": {"v": 3}, "b": {"x": 2}}
    out2 = replace_block(out, "k", ["k: null"])
    assert yaml.safe_load(out2) == {"a": 1, "k": None, "b": {"x": 2}}
