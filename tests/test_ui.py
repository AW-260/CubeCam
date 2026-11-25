"""GUI: TrueType text, panels, the full render, smooth state changes, the help card."""
import math
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import ImageFont

import gfx
import theme as T
from ar_mat import ArMatRenderer, Mat, mat_layout, texture_ppc
from plane import TablePlane, _h_apply
from stats import compute_stats
from synth_table import SynthCamera, hand_landmarks
from table_calib import CalibView
from timer_state import State
from ui import ArView, Renderer, ViewModel
from zone import Zone
from zone_setup import SetupView

ROOT = Path(__file__).resolve().parents[1]
W, H = 1280, 720


def test_fonts_are_truetype():
    for role in ("regular", "semibold", "digits"):
        f = gfx.font(role, 20)
        assert isinstance(f, ImageFont.FreeTypeFont), role        # never the old bitmap font
    m = gfx.text_mask("59.88", "digits", 60)
    assert m.dtype == np.uint8 and m.max() == 255
    assert abs(m.shape[1] - gfx.text_width("59.88", "digits", 60)) <= 3
    assert m.shape[0] == gfx.line_height("digits", 60)


def test_timer_digits_have_equal_width():
    # tabular figures: the time doesn't jitter sideways while it runs
    widths = {gfx.text_width(d, "digits", 100) for d in "0123456789"}
    assert max(widths) - min(widths) <= 2


def test_draw_text_anchors_and_clipping():
    img = np.zeros((200, 400, 3), np.uint8)
    x0, y0, x1, y1 = gfx.draw_text(img, "Hello", 200, 100, 30, (255, 255, 255), anchor="mm")
    assert abs((x0 + x1) / 2 - 200) <= 1 and abs((y0 + y1) / 2 - 100) <= 1
    assert img[y0:y1, x0:x1].max() > 200
    gfx.draw_text(img, "partly outside", 380, -10, 30, (255, 255, 255))   # must not raise


def test_rounded_panel_blends_and_keeps_corners():
    img = np.full((100, 200, 3), 200, np.uint8)
    gfx.panel(img, 10, 10, 180, 80, 20, T.PANEL, 0.7)
    assert img[50, 100].max() < 120                 # darkened inside
    assert (img[11, 11] == 200).all()                # rounded corner left alone
    assert (img[0, 0] == 200).all()


def _plane():
    cam = SynthCamera(tilt_deg=35, height_cm=40)
    sq = np.array([(-5, 50), (5, 50), (5, 60), (-5, 60)], float)
    return TablePlane.from_reference(sq, _h_apply(cam.H_table_to_image(), sq)[0], (W, H))


def _full_vm(**kw):
    from models import Penalty, Solve

    solves = [Solve(t, Penalty.OK, "", "", 30.0, None) for t in (61.4, 58.1, 64.9, 55.3, 59.9, 62.2)]
    plane = _plane()
    mat = Mat(0, 0, 0, 30, 22)
    quad = mat.image_quad(plane)
    base = dict(
        state=State.READY, time_text="59.88", time_color=T.STATE["ready"], sub_text="Ready — lift to start",
        scramble="R U2 F' L D2 B R' U F2 L2 D' R2 B2 U' F L' D R2 U2 B'", stats=compute_stats(solves),
        recent=["1:02.20", "59.88"], zone=mat.zone(plane), zone_mode="ready", hold_progress=0.0,
        status_lines=["Camera 1280x720", "Zone: AR mat"], message="Zone set", flash=T.WARN, flash_level=0.5,
        show_debug=True, debug={"state": "READY", "hands seen": 2}, show_help=True,
        warnings=["Zone is too small - press M"],
        ar=ArView(quad, (30, 22), "ready", 0.0, [hand_landmarks(quad.mean(axis=0) + (-120, 0), 80)]),
    )
    base.update(kw)
    return ViewModel(**base)


@pytest.mark.parametrize("size", [(720, 1280), (1080, 1920), (480, 640)])
def test_full_render_every_overlay(size):
    h, w = size
    img = np.full((h, w, 3), 90, np.uint8)
    r = Renderer(animate=False)
    vm = _full_vm()
    if (w, h) != (W, H):
        vm.ar = None
    out = r.render(img, vm)
    assert out.shape == (h, w, 3)
    # setup and calibration overlays too
    setup = SetupView("ZONE SETUP 1/3: mat", ["Drag the mat", "ENTER accept   R redo   ESC cancel"],
                      candidate=Zone.rect(0.2, 0.6, 0.5, 0.3), warnings=["too small"], progress=0.4,
                      palms=[(0.3, 0.7)], markers=[(0.2, 0.6)], drag=((0.1, 0.1), (0.3, 0.3)),
                      stroke=[(0.3 + 0.2 * math.cos(a), 0.75 + 0.1 * math.sin(a)) for a in np.linspace(0, 5, 40)])
    r.render(np.full((h, w, 3), 90, np.uint8), _full_vm(ar=None, setup=setup, show_help=False))
    calib = CalibView("TABLE CALIBRATION 2/2", ["Click 4 corners"], ["warn"], clicks_px=[(100, 100), (300, 120)],
                      mouse_px=(w * 0.5, h * 0.5), grid_px=[np.array([[10.0, 10.0], [200.0, 300.0]])],
                      marker_px=np.array([[50, 50], [90, 50], [90, 90], [50, 90]], float), progress=0.5)
    r.render(np.full((h, w, 3), 90, np.uint8), _full_vm(ar=None, calib=calib, show_help=False))
    r.close()


def test_bottom_of_the_picture_stays_clear():
    """Nothing but the zone / mat is drawn over the lower part of the picture (help off)."""
    img = np.random.default_rng(0).integers(0, 255, (H, W, 3), dtype=np.uint8)
    vm = _full_vm(zone=None, ar=None, show_help=False, show_debug=False, flash=None, message="hi")
    out = Renderer(animate=False).render(img.copy(), vm)
    bottom = int(H * 0.6)
    assert np.array_equal(out[bottom:], img[bottom:])


def test_help_is_hidden_by_default_and_toggles():
    from timer_state import State as S

    vm = ViewModel(state=S.IDLE, time_text="0.00")
    assert vm.show_help is False
    import main as M

    cfg = ROOT / "config.yaml"
    app = M.App.__new__(M.App)
    app.show_help = False
    app.space = type("S", (), {"on_key": staticmethod(lambda k: False)})()
    assert app.handle_key(ord("?"), 0.0) and app.show_help
    assert app.handle_key(0x700000, 0.0) and not app.show_help     # F1 via waitKeyEx on Windows
    assert cfg.exists()


def test_colours_ease_instead_of_jumping():
    r = Renderer(animate=True)
    r._dt = 0.016
    a = r._smooth("timer", T.TEXT)
    r._dt = 0.016
    b = r._smooth("timer", T.STATE["ready"])
    assert b != tuple(T.STATE["ready"]) and b != a                # part of the way there
    for _ in range(60):
        r._dt = 0.016
        c = r._smooth("timer", T.STATE["ready"])
    assert c == tuple(T.STATE["ready"])
    still = Renderer(animate=False)
    assert still._smooth("timer", T.STATE["ready"]) == tuple(T.STATE["ready"])


def _mat_frame():
    plane = _plane()
    quad = Mat(0, 0, 0, 30, 22).image_quad(plane)
    return np.full((H, W, 3), 120, np.uint8), quad


def test_mat_state_change_cross_fades():
    # idle -> holding (no hold progress yet, so only the static look changes)
    img, quad = _mat_frame()
    r = ArMatRenderer(background=False, animate=True)
    r.draw(img.copy(), quad, (30, 22), "idle")
    idle = img.copy()
    r2 = ArMatRenderer(background=False, animate=False)
    r2.draw(idle, quad, (30, 22), "idle")
    ready = img.copy()
    r2.draw(ready, quad, (30, 22), "holding")
    first = img.copy()
    r.draw(first, quad, (30, 22), "holding")                 # the fade starts from the old state
    assert r._fade is not None
    assert np.abs(first.astype(int) - idle.astype(int)).max() <= 1
    time.sleep(T.FADE_S / 2)
    mid = img.copy()
    r.draw(mid, quad, (30, 22), "holding")                   # half-way: in between
    diff_idle = np.abs(mid.astype(int) - idle.astype(int)).sum()
    diff_ready = np.abs(mid.astype(int) - ready.astype(int)).sum()
    total = np.abs(ready.astype(int) - idle.astype(int)).sum()
    assert diff_idle < total and diff_ready < total and diff_ready > 0
    time.sleep(T.FADE_S + 0.02)
    end = img.copy()
    r.draw(end, quad, (30, 22), "holding")
    assert r._fade is None and np.abs(end.astype(int) - ready.astype(int)).max() <= 1


def test_hold_fill_grows_smoothly():
    img, quad = _mat_frame()
    r = ArMatRenderer(background=False, animate=False)
    counts = []
    for p in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        out = img.copy()
        r.draw(out, quad, (30, 22), "holding", p)
        counts.append(int((out.astype(int).sum(axis=2) > 3 * 120 + 150).sum()))   # brightly lit pixels
    assert all(b > a for a, b in zip(counts, counts[1:])), counts   # more of the edge lit at every step
    # drawn per frame, not as stepped textures: one warp per state plus the light-effect maps
    assert r.warps <= 5


def test_mat_layout_inside_texture():
    L = mat_layout(30, 22)
    x0, y0, x1, y1 = L.rect
    W_, H_ = L.size
    assert 0 < x0 < x1 < W_ and 0 < y0 < y1 < H_ and L.rounded
    o = L.outline
    assert o[:, 0].min() >= x0 - 0.5 and o[:, 0].max() <= x1 + 0.5
    assert o[:, 1].min() >= y0 - 0.5 and o[:, 1].max() <= y1 + 0.5
    # a drawn shape fills its own bounding box; big ones use fewer texture px per cm
    tri = ((-20.0, 10.0), (0.0, -10.0), (20.0, 10.0))
    Ls = mat_layout(40, 20, shape=tri)
    assert not Ls.rounded and len(Ls.outline) > 3                    # drawn smooth (corner cutting)
    assert Ls.outline[:, 0].min() >= Ls.rect[0] - 0.5 and Ls.outline[:, 1].max() <= Ls.rect[3] + 0.5
    assert texture_ppc(120, 60) < texture_ppc(30, 22)
    assert max(mat_layout(120, 60).size) <= 1600
