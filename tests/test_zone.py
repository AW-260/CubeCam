"""Start zone: polygon geometry, warnings, hands-mode builder, setup controller, markers, config."""
import math

import cv2
import numpy as np
import pytest
import yaml

from config import load_config, replace_zone_block, save_zone
from markers import MarkerZoneTracker, affine_from_3, detect_markers, get_dictionary
from zone import (
    HandsZoneBuilder,
    Zone,
    clip_to_frame,
    order_convex,
    palm_centre,
    zone_from_hands,
    zone_warnings,
)
from zone_setup import EVENT_LBUTTONDOWN, EVENT_LBUTTONUP, EVENT_MOUSEMOVE, SetupMode, ZoneSetup


# ------------------------------------------------------------------ helpers
def fake_hand(cx: float, cy: float, s: float = 0.08) -> np.ndarray:
    """21 MediaPipe-style landmarks of a flat hand, palm centre near (cx, cy), fingers up."""
    pts = np.zeros((21, 2))
    pts[0] = (0, 0.55)                                   # wrist
    for f, dx in enumerate((-0.45, -0.15, 0.1, 0.32)):   # index, middle, ring, pinky (MCP at y=-0.1)
        base = 5 + 4 * f
        for j in range(4):
            pts[base + j] = (dx * (1 + 0.1 * j), -0.1 - 0.3 * j)
    for j in range(1, 5):                                # thumb
        pts[j] = (-0.4 - 0.15 * j, 0.35 - 0.12 * j)
    pts = pts * s
    return (pts - pts[[0, 5, 9, 13, 17]].mean(axis=0) + (cx, cy)).astype(np.float32)


# ------------------------------------------------------------------ geometry
def test_rect_contains_including_edges():
    z = Zone.rect(0.1, 0.6, 0.8, 0.3)
    assert z.contains(0.5, 0.75)
    assert z.contains(0.1, 0.6) and z.contains(0.9, 0.9) and z.contains(0.5, 0.9)  # corners / edges
    assert not z.contains(0.05, 0.75) and not z.contains(0.5, 0.95) and not z.contains(0.95, 0.5)
    assert z.is_rect and math.isclose(z.area, 0.24)


def test_polygon_contains_concave_and_slanted():
    # L-shape: the notch at the top right is outside
    z = Zone(((0.1, 0.1), (0.5, 0.1), (0.5, 0.5), (0.9, 0.5), (0.9, 0.9), (0.1, 0.9)))
    assert z.contains(0.3, 0.3) and z.contains(0.7, 0.7)
    assert not z.contains(0.7, 0.3)
    assert not z.is_rect
    assert math.isclose(z.area, 0.8 * 0.8 - 0.4 * 0.4)
    # a trapezoid like a desk seen in perspective (narrower at the far edge)
    t = Zone(((0.3, 0.6), (0.7, 0.6), (0.95, 0.95), (0.05, 0.95)))
    assert t.contains(0.1, 0.93) and not t.contains(0.1, 0.62)
    assert t.contains(0.5, 0.6)  # on the top edge


def test_point_order_does_not_matter_for_contains():
    a = Zone(((0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)))
    b = Zone(((0.2, 0.8), (0.8, 0.8), (0.8, 0.2), (0.2, 0.2)))  # counter-clockwise
    rng = np.random.default_rng(0)
    for x, y in rng.random((200, 2)):
        assert a.contains(x, y) == b.contains(x, y)


def test_invalid_zones_rejected():
    with pytest.raises(ValueError):
        Zone(((0.1, 0.1), (0.2, 0.2)))
    with pytest.raises(ValueError):
        Zone.rect(0.1, 0.1, 0.0, 0.2)
    with pytest.raises(ValueError):
        Zone(((0.1, 0.1), (0.2, float("nan")), (0.3, 0.3)))


def test_from_corners_any_drag_direction():
    a = Zone.from_corners((0.8, 0.9), (0.2, 0.5))
    assert a == Zone.rect(0.2, 0.5, 0.6, 0.4) or np.allclose(a.points, Zone.rect(0.2, 0.5, 0.6, 0.4).points)


def test_config_formats_round_trip():
    old = Zone.from_config({"x": 0.1, "y": 0.68, "w": 0.8, "h": 0.3})
    assert np.allclose(old.points, [(0.1, 0.68), (0.9, 0.68), (0.9, 0.98), (0.1, 0.98)])
    poly = Zone(((0.31, 0.6), (0.7, 0.62), (0.9, 0.97), (0.12, 0.95)))
    again = Zone.from_config({"mode": "fixed", "points": poly.to_config(), "x": 0, "y": 0, "w": 1, "h": 1})
    assert np.allclose(again.points, poly.points)  # points win over the old keys


def test_clip_and_inside_fraction():
    assert math.isclose(Zone.rect(0.2, 0.2, 0.5, 0.5).inside_fraction(), 1.0)
    half = Zone.rect(0.5, 0.2, 1.0, 0.4)           # right half sticks out of the frame
    assert math.isclose(half.inside_fraction(), 0.5)
    assert clip_to_frame(Zone.rect(1.2, 0.2, 0.3, 0.3).points) == []
    tri = Zone(((0.5, -0.5), (1.5, 0.5), (0.5, 0.5)))
    assert 0 < tri.inside_fraction() < 1


def test_zone_warnings():
    assert zone_warnings(Zone.rect(0.1, 0.6, 0.8, 0.35)) == []
    w = zone_warnings(Zone.rect(0.6, 0.6, 0.9, 0.9))
    assert any("mostly outside" in s for s in w)
    w = zone_warnings(Zone.rect(0.4, 0.4, 0.08, 0.08))
    assert any("too small" in s for s in w)
    w = zone_warnings(Zone.rect(0.3, 0.5, 0.5, 0.02))  # thin strip
    assert any("too small" in s for s in w)
    assert zone_warnings(Zone.rect(1.5, 1.5, 0.3, 0.3))  # completely off-screen


def test_order_convex_any_input_order():
    corners = [(0.1, 0.6), (0.9, 0.62), (0.95, 0.97), (0.05, 0.95)]
    for perm in ([2, 0, 3, 1], [3, 2, 1, 0], [1, 3, 0, 2]):
        out = order_convex([corners[i] for i in perm])
        assert out == corners  # starts top-left, clockwise on screen


def test_mirrored_zone():
    z = Zone.rect(0.1, 0.6, 0.3, 0.3)
    m = z.mirrored()
    assert m.contains(0.75, 0.7) and not m.contains(0.25, 0.7)
    assert math.isclose(m.area, z.area)


def test_to_pixels_shape():
    px = Zone.rect(0.1, 0.5, 0.5, 0.25).to_pixels(1280, 720)
    assert px.shape == (4, 1, 2) and px.dtype == np.int32
    assert px[0, 0].tolist() == [128, 360] and px[2, 0].tolist() == [768, 540]


# --------------------------------------------------------------- hands mode
def test_fake_hand_palm_centre():
    h = fake_hand(0.3, 0.7)
    assert np.allclose(palm_centre(h), (0.3, 0.7), atol=1e-6)


def test_zone_from_hands_covers_both_hands_with_margin():
    left, right = fake_hand(0.3, 0.75), fake_hand(0.7, 0.75)
    z = zone_from_hands([left, right])
    assert z.is_rect
    for h in (left, right):
        assert all(z.contains(float(x), float(y)) for x, y in h)   # every landmark inside
    x1, y1, x2, y2 = z.bbox
    pts = np.vstack([left, right])
    assert x1 < pts[:, 0].min() - 0.02 and x2 > pts[:, 0].max() + 0.02   # a real margin
    # palms can move a bit next time and still be inside
    for dx, dy in ((-0.05, 0), (0.05, 0), (0, 0.04), (0, -0.04)):
        assert z.contains(0.3 + dx, 0.75 + dy) and z.contains(0.7 + dx, 0.75 + dy)
    assert zone_warnings(z) == []


def test_zone_from_hands_clipped_to_frame():
    z = zone_from_hands([fake_hand(0.08, 0.93), fake_hand(0.6, 0.93)])
    x1, y1, x2, y2 = z.bbox
    assert x1 == 0.0 and y2 == 1.0 and 0 < y1 < 0.93


def test_zone_from_hands_scales_with_hand_size():
    small = zone_from_hands([fake_hand(0.3, 0.6, 0.05), fake_hand(0.7, 0.6, 0.05)])
    big = zone_from_hands([fake_hand(0.3, 0.6, 0.10), fake_hand(0.7, 0.6, 0.10)])
    assert big.area > small.area


def test_zone_from_hands_needs_two():
    with pytest.raises(ValueError):
        zone_from_hands([fake_hand(0.5, 0.5)])


def _feed(builder, t0, seconds, hands_fn, fps=30):
    out = None
    n = int(seconds * fps)
    for i in range(n):
        t = t0 + i / fps
        z = builder.update(t, hands_fn(t))
        out = out or z
    return out, t0 + n / fps


def test_builder_needs_two_seconds_of_stillness():
    b = HandsZoneBuilder(hold=2.0)
    rng = np.random.default_rng(1)
    jitter = lambda t: [fake_hand(0.3 + rng.normal(0, 0.003), 0.75), fake_hand(0.7, 0.75 + rng.normal(0, 0.003))]
    z, t = _feed(b, 0.0, 1.8, jitter)
    assert z is None and 0.8 < b.progress < 1.0
    z, t = _feed(b, t, 0.4, jitter)
    assert z is not None and b.result is z and b.progress == 1.0
    assert z.contains(0.3, 0.75) and z.contains(0.7, 0.75)
    assert b.update(t + 1, jitter(t)) is None  # proposed only once


def test_builder_restarts_when_hands_move():
    b = HandsZoneBuilder(hold=2.0)
    still = lambda t: [fake_hand(0.3, 0.75), fake_hand(0.7, 0.75)]
    _, t = _feed(b, 0.0, 1.5, still)
    assert b.progress > 0.7
    b.update(t, [fake_hand(0.3, 0.6), fake_hand(0.7, 0.75)])  # left hand jumps
    assert b.progress == 0.0
    z, t2 = _feed(b, t + 1 / 30, 1.5, lambda t: [fake_hand(0.3, 0.6), fake_hand(0.7, 0.75)])
    assert z is None
    z, _ = _feed(b, t2, 0.6, lambda t: [fake_hand(0.3, 0.6), fake_hand(0.7, 0.75)])
    assert z is not None and z.contains(0.3, 0.6)


def test_builder_slow_drift_counts_as_moving():
    b = HandsZoneBuilder(hold=2.0, still_tol=0.03)
    drift = lambda t: [fake_hand(0.3 + 0.05 * t, 0.75), fake_hand(0.7 + 0.05 * t, 0.75)]
    z, _ = _feed(b, 0.0, 3.0, drift)
    assert z is None  # never 2 s within 3 % of the start point


def test_builder_one_hand_or_brief_dropout():
    b = HandsZoneBuilder(hold=2.0, grace=0.25)
    both = [fake_hand(0.3, 0.75), fake_hand(0.7, 0.75)]
    _, t = _feed(b, 0.0, 1.0, lambda t: both)
    p = b.progress
    # 0.1 s with one hand lost (tracking flicker): progress kept
    _, t = _feed(b, t, 0.1, lambda t: both[:1])
    assert b.progress == pytest.approx(p)
    z, t = _feed(b, t, 1.1, lambda t: both)
    assert z is not None
    # a hand gone for longer restarts
    b.reset()
    _, t = _feed(b, 0.0, 1.0, lambda t: both)
    _, t = _feed(b, t, 0.5, lambda t: both[:1])
    assert b.progress == 0.0 and "one hand" in b.hint.lower()


def test_builder_ignores_third_small_hand():
    b = HandsZoneBuilder(hold=0.5)
    hands = [fake_hand(0.3, 0.75), fake_hand(0.9, 0.2, 0.03), fake_hand(0.7, 0.75)]
    z, _ = _feed(b, 0.0, 1.0, lambda t: hands)
    assert z is not None and not z.contains(0.9, 0.2)


# ------------------------------------------------------------- setup control
def test_setup_drag_mode():
    s = ZoneSetup()
    s.start()
    assert s.active and s.mode is SetupMode.DRAG
    s.on_mouse(EVENT_LBUTTONDOWN, 128, 400, 1280, 720)
    s.on_mouse(EVENT_MOUSEMOVE, 600, 600, 1280, 720)
    assert s.view().drag is not None and s.candidate is None
    s.on_mouse(EVENT_LBUTTONUP, 1152, 700, 1280, 720)
    assert np.allclose(s.candidate.bbox, (0.1, 400 / 720, 0.9, 700 / 720))
    mode, zone, mat = s.accept()
    assert mode == "fixed" and zone.is_rect and mat is None and not s.active


def test_setup_drag_click_is_ignored_and_escape_cancels():
    s = ZoneSetup()
    s.start()
    s.on_mouse(EVENT_LBUTTONDOWN, 500, 500, 1280, 720)
    s.on_mouse(EVENT_LBUTTONUP, 505, 503, 1280, 720)
    assert s.candidate is None and s.accept() is None and s.active
    s.cancel()
    assert not s.active and s.view() is None


def test_setup_drag_outside_window_is_clamped():
    s = ZoneSetup()
    s.start()
    s.on_mouse(EVENT_LBUTTONDOWN, -50, 300, 1280, 720)
    s.on_mouse(EVENT_LBUTTONUP, 2000, 900, 1280, 720)
    assert s.candidate.bbox == (0.0, 300 / 720, 1.0, 1.0)


def test_setup_cycle_order_and_availability():
    s = ZoneSetup()
    s.start()
    assert s.cycle() is SetupMode.HANDS
    assert s.cycle() is SetupMode.MARKERS
    assert s.cycle() is None and not s.active
    s = ZoneSetup(hands_available=True, markers_available=False)
    s.start()
    assert s.cycle() is SetupMode.HANDS and s.cycle() is None


def test_setup_hands_mode_proposes_zone_and_mouse_is_ignored():
    s = ZoneSetup(hold=1.0)
    s.start()
    s.cycle()
    s.on_mouse(EVENT_LBUTTONDOWN, 0, 0, 1280, 720)
    s.on_mouse(EVENT_LBUTTONUP, 600, 600, 1280, 720)
    assert s.candidate is None
    hands = [fake_hand(0.3, 0.75), fake_hand(0.7, 0.75)]
    for i in range(40):
        s.update(i / 30, hands)
    v = s.view()
    assert s.candidate is not None and len(v.palms) == 2 and v.warnings == []
    # once proposed, it stays put even if the hands move away (so you can press ENTER)
    s.update(2.0, [fake_hand(0.5, 0.2), fake_hand(0.6, 0.2)])
    assert s.candidate.contains(0.3, 0.75)
    s.redo()
    assert s.candidate is None and s.mode is SetupMode.HANDS
    for i in range(40):
        s.update(3 + i / 30, hands)
    assert s.accept()[0] == "fixed"


def test_setup_markers_mode():
    s = ZoneSetup()
    s.start(SetupMode.MARKERS)
    s.update(0.0, [], None, "1/4 markers - show all 4", [(0.1, 0.6)])
    assert s.accept() is None
    z = Zone(((0.1, 0.6), (0.9, 0.6), (0.9, 0.95), (0.1, 0.95)))
    s.update(0.3, [], z, "4/4 markers", [p for p in z.points])
    assert "4/4" in " ".join(s.view().lines)
    assert s.accept() == ("markers", z, None)


def test_setup_view_warns_about_bad_candidate():
    s = ZoneSetup()
    s.start()
    s.on_mouse(EVENT_LBUTTONDOWN, 600, 300, 1280, 720)
    s.on_mouse(EVENT_LBUTTONUP, 650, 340, 1280, 720)
    assert any("too small" in w for w in s.view().warnings)


# --------------------------------------------------------- hands-in-zone test
class _FakeBackend:
    name = "fake"

    def __init__(self):
        self.hands = []

    def detect(self, rgb, t):
        return self.hands

    def close(self):
        pass


def test_detector_uses_palm_centre_inside_polygon():
    from hands import HandZoneDetector

    det = HandZoneDetector.__new__(HandZoneDetector)
    det.zone = Zone(((0.3, 0.55), (0.7, 0.55), (0.95, 0.95), (0.05, 0.95)))  # trapezoid
    det.hands_required = 2
    det.process_width = 0
    det._backend = _FakeBackend()
    det.backend_name = "fake"
    frame = np.zeros((72, 128, 3), np.uint8)
    det._backend.hands = [fake_hand(0.15, 0.9), fake_hand(0.85, 0.9)]
    assert det.process(frame, 0.0).active
    # same x, but higher up where the trapezoid is narrower: outside
    det._backend.hands = [fake_hand(0.15, 0.6), fake_hand(0.85, 0.9)]
    s = det.process(frame, 0.1)
    assert not s.active and s.debug["palms_in_zone"] == [False, True]
    # swapping the zone takes effect on the next frame
    det.zone = Zone.rect(0.0, 0.5, 1.0, 0.5)
    assert det.process(frame, 0.2).active


# -------------------------------------------------------------------- markers
def test_affine_from_3():
    src = [(0, 0), (1, 0), (0, 1)]
    dst = [(0.1, 0.2), (1.1, 0.2), (0.1, 1.2)]
    m = affine_from_3(src, dst)
    assert np.allclose(m @ [1, 1, 1], (1.1, 1.2))
    assert affine_from_3([(0, 0), (1, 1), (2, 2)], dst) is None


CORNERS = {0: (0.1, 0.6), 1: (0.9, 0.62), 2: (0.95, 0.95), 3: (0.05, 0.93)}


def test_tracker_all_four_then_three_then_none():
    tr = MarkerZoneTracker()
    assert tr.update(0.0, {0: CORNERS[0]}) is None and "show all" in tr.status
    z = tr.update(0.1, CORNERS)
    assert z is not None and tr.status.startswith("4/4")
    assert z.contains(0.5, 0.8)
    # camera bumped: everything shifts right by 0.05, marker 2 is under a hand
    shifted = {i: (x + 0.05, y) for i, (x, y) in CORNERS.items()}
    z2 = tr.update(0.2, {i: p for i, p in shifted.items() if i != 2})
    assert "estimated" in tr.status
    assert np.allclose(sorted(z2.points), sorted(shifted.values()), atol=1e-9)
    # only 2 visible: hold the last zone
    z3 = tr.update(5.0, {0: (0.5, 0.5), 1: (0.6, 0.5)})
    assert z3 is z2 and "holding" in tr.status
    # unknown ids are ignored
    assert tr.update(5.1, {7: (0.1, 0.1), **{i: p for i, p in shifted.items() if i != 2}}) is z2


def test_tracker_deadband_ignores_jitter():
    tr = MarkerZoneTracker(deadband=0.004)
    z = tr.update(0.0, CORNERS)
    z2 = tr.update(0.1, {i: (x + 0.001, y - 0.001) for i, (x, y) in CORNERS.items()})
    assert z2 is z
    z3 = tr.update(0.2, {i: (x + 0.02, y) for i, (x, y) in CORNERS.items()})
    assert z3 is not z


def _scene_with_markers(mirror_view: bool):
    """A 1280x720 grey 'desk' with the 4 markers placed in perspective, plus their true centres."""
    dictionary = get_dictionary()
    img = np.full((720, 1280, 3), 120, np.uint8)
    centres = {0: (250, 430), 1: (1030, 440), 2: (1150, 670), 3: (130, 660)}
    for i, (cx, cy) in centres.items():
        m = cv2.aruco.generateImageMarker(dictionary, i, 60)
        m = cv2.copyMakeBorder(m, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
        h, w = m.shape
        # squash vertically like a marker lying on a desk seen at an angle
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[cx - 42, cy - 26], [cx + 42, cy - 28], [cx + 44, cy + 26], [cx - 44, cy + 28]])
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(cv2.cvtColor(m, cv2.COLOR_GRAY2BGR), M, (1280, 720),
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (1280, 720)) > 0
        img[mask] = warped[mask]
    truth = {i: ((1 - x / 1280) if mirror_view else x / 1280, y / 720) for i, (x, y) in centres.items()}
    return img, truth


@pytest.mark.parametrize("mirror", [False, True])
def test_detect_markers_in_raw_frame_and_mirror(mirror):
    det = cv2.aruco.ArucoDetector(get_dictionary(), cv2.aruco.DetectorParameters())
    img, truth = _scene_with_markers(mirror)
    found = detect_markers(img, det, mirror=mirror, process_width=960)
    assert set(found) == {0, 1, 2, 3}
    for i, (x, y) in truth.items():
        assert abs(found[i][0] - x) < 0.01 and abs(found[i][1] - y) < 0.01
    zone = MarkerZoneTracker().update(0.0, found)
    assert zone_warnings(zone) == []
    # a hand over one marker: 3 found
    img[400:480, 200:300] = (90, 140, 200)
    assert set(detect_markers(img, det, mirror=mirror)) == {1, 2, 3}


def test_mirrored_markers_are_not_detected():
    """Why detection must run on the raw (un-flipped) frame."""
    det = cv2.aruco.ArucoDetector(get_dictionary(), cv2.aruco.DetectorParameters())
    img, _ = _scene_with_markers(False)
    assert detect_markers(cv2.flip(img, 1), det, mirror=False) == {}


def test_marker_sheet_is_detectable_and_pdf_well_formed(tmp_path):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("make_markers", Path(__file__).parents[1] / "tools" / "make_markers.py")
    mm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mm)
    page = mm.make_page(60)
    assert page.shape == (3508, 2480)
    det = cv2.aruco.ArucoDetector(get_dictionary(), cv2.aruco.DetectorParameters())
    small = cv2.resize(page, (620, 877), interpolation=cv2.INTER_AREA)   # like a photo of the sheet
    assert set(detect_markers(small, det, mirror=False, process_width=0)) == {0, 1, 2, 3}
    assert mm.main(["--out", str(tmp_path / "m"), "--size", "50"]) == 0
    pdf = (tmp_path / "m.pdf").read_bytes()
    assert pdf.startswith(b"%PDF-1.4") and pdf.rstrip().endswith(b"%%EOF")
    xref = int(pdf.rsplit(b"startxref", 1)[1].split()[0])
    assert pdf[xref:xref + 4] == b"xref"
    table = pdf[xref:].split(b"\n")[3:8]  # objects 1..5
    for k, row in enumerate(table, 1):
        off = int(row.split()[0])
        assert pdf[off:off + len(f"{k} 0 obj")] == f"{k} 0 obj".encode()
    assert (tmp_path / "m.png").stat().st_size > 10_000


# --------------------------------------------------------------------- config
SAMPLE = """# comment at top
camera:
  index: 0   # keep me

# Start zone comment
zone:
  x: 0.10
  y: 0.68
  w: 0.80
  h: 0.30

timer:
  hold_time: 0.3
"""


def test_replace_zone_block_keeps_everything_else():
    out = replace_zone_block(SAMPLE, "fixed", [[0.1, 0.5], [0.9, 0.5], [0.9, 0.95], [0.1, 0.95]])
    assert "# comment at top" in out and "index: 0   # keep me" in out and "# Start zone comment" in out
    assert "x: 0.10" not in out and "\n\ntimer:" in out
    data = yaml.safe_load(out)
    assert data["zone"] == {"mode": "fixed", "points": [[0.1, 0.5], [0.9, 0.5], [0.9, 0.95], [0.1, 0.95]]}
    assert data["timer"] == {"hold_time": 0.3} and data["camera"] == {"index": 0}


def test_replace_zone_block_appends_when_missing():
    out = replace_zone_block("timer:\n  hold_time: 0.3\n", "markers", [[0.1, 0.1], [0.5, 0.1], [0.3, 0.4]])
    data = yaml.safe_load(out)
    assert data["timer"]["hold_time"] == 0.3 and data["zone"]["mode"] == "markers"
    assert yaml.safe_load(replace_zone_block("", "fixed", [[0, 0], [1, 0], [1, 1]]))["zone"]["mode"] == "fixed"


def test_save_zone_round_trip_and_old_configs_still_load(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(SAMPLE, encoding="utf-8")
    c, warn = load_config(cfg)
    assert warn is None and Zone.from_config(c["zone"]).is_rect  # old x/y/w/h config
    zone = Zone(((0.123456, 0.6), (0.9, 0.61), (0.95, 0.97), (0.05, 0.96)))
    save_zone(cfg, "markers", zone.to_config())
    c, warn = load_config(cfg)
    assert warn is None and c["zone"]["mode"] == "markers"
    assert np.allclose(Zone.from_config(c["zone"]).points, zone.points, atol=1e-4)
    assert c["camera"]["index"] == 0 and c["timer"]["hold_time"] == 0.3
    save_zone(cfg, "fixed", Zone.rect(0.1, 0.6, 0.8, 0.3).to_config())  # saving twice keeps one block
    text = cfg.read_text(encoding="utf-8")
    assert text.count("zone:") == 1 and "# Start zone comment" in text
    with pytest.raises(ValueError):
        save_zone(cfg, "weird", zone.to_config())


def test_shipped_config_zone_is_valid():
    from pathlib import Path

    c, warn = load_config(Path(__file__).parents[1] / "config.yaml")
    assert warn is None
    assert zone_warnings(Zone.from_config(c["zone"])) == []
