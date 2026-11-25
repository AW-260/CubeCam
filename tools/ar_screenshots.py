"""Render screenshots of the holographic start area on a synthetic table (no camera needed).

    python tools/ar_screenshots.py                 # -> docs/screenshots/after/*.jpg

It builds a synthetic camera view of a wooden table with a phone showing the
calibration marker, calibrates the table plane from it exactly like key T
does, places the mat, and renders the real app UI (ui.Renderer) for each
state: idle, holding, ready (at the top of its pulse), solving, the ripple
when a solve starts, AR off, the calibration screen, the mat placement screen
and the help card. Then the draw mode: a hand-drawn path is fed through the
real zone setup as mouse events (while tracing, the cleaned shape, the drawn
area in use) and once more without a table calibration (a shape on the
screen). Hands are drawn into the picture so you can see them covering the
area.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import theme as T  # noqa: E402
from ar_mat import mat_at_image_point  # noqa: E402
from make_phone_marker import make_phone_marker  # noqa: E402
from markers import detect_marker_corners, make_detector  # noqa: E402
from models import Penalty, Solve, format_solve  # noqa: E402
from stats import compute_stats  # noqa: E402
from synth_table import SynthCamera, TableScene, draw_hand, hand_landmarks, phone_image  # noqa: E402
from table_calib import CalibMode, TableCalibration  # noqa: E402
from timer_state import State  # noqa: E402
from ui import GREEN, WHITE, YELLOW, ArView, Renderer, ViewModel  # noqa: E402
from zone_setup import EVENT_LBUTTONDOWN, EVENT_LBUTTONUP, EVENT_MOUSEMOVE, SetupMode, ZoneSetup  # noqa: E402

W, H = 1280, 720
READY = "Ready — lift to start"
SCRAMBLE = "R U2 F' L D2 B R' U F2 L2 D' R2 B2 U' F L' D R2 U2 B'"


def build_scene():
    cam = SynthCamera(tilt_deg=33, height_cm=40, hfov_deg=70)
    scene = TableScene(cam, px_per_cm=16, seed=3)
    body, mask = phone_image(make_phone_marker(), 600)
    phone_w = 7.2
    scene.paste(body, (24, 62), phone_w, 28, mask)
    raw = scene.render()
    side_px = 6 * (int(1080 * 0.82) // 6)
    marker_mm = 10 * phone_w * (600 / (600 + 2 * int(600 * 0.06))) * side_px / 1080
    return cv2.flip(raw, 1), cv2.flip(raw, 1), marker_mm   # the app shows the mirrored view


def traced_path(centre, rx: float, ry: float, n: int = 140, seed: int = 5) -> np.ndarray:
    """A hand-drawn-looking closed loop in screen px: a bean shape, wobbly, overshooting its start."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * math.pi * 1.06, n)
    r = 1 + 0.10 * np.cos(2 * t) - 0.16 * np.cos(t) ** 8 * (np.sin(t) < 0)     # a dent in the far side
    x = centre[0] + rx * r * np.cos(t) * (1 + 0.06 * np.sin(3 * t))
    y = centre[1] + ry * r * np.sin(t)
    wobble = np.cumsum(rng.normal(0, 0.9, (n, 2)), axis=0)
    wobble -= np.linspace(0, 1, n)[:, None] * wobble[-1]
    return np.column_stack([x, y]) + wobble + rng.normal(0, 0.8, (n, 2))


def trace(setup: ZoneSetup, path: np.ndarray, upto: float = 1.0) -> None:
    """Feed a path to the zone setup as mouse events; stop early (button still down) if upto < 1."""
    n = max(2, int(len(path) * upto))
    setup.on_mouse(EVENT_LBUTTONDOWN, *map(int, path[0]), W, H)
    for p in path[1:n]:
        setup.on_mouse(EVENT_MOUSEMOVE, *map(int, p), W, H)
    if upto >= 1.0:
        setup.on_mouse(EVENT_LBUTTONUP, *map(int, path[-1]), W, H)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "docs" / "screenshots" / "after"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    view, _, marker_mm = build_scene()
    # --- calibrate from the phone exactly like the app (detect on the raw frame, mirror the corners)
    raw = cv2.flip(view, 1)
    corners = detect_marker_corners(raw, make_detector(refine=True), mirror=True, process_width=0)
    calib = TableCalibration(phone_marker_mm=round(marker_mm, 1))
    calib.start(CalibMode.PHONE, (W, H))
    for i in range(6):
        calib.update_phone(i * 0.12, corners, (W, H))
    calib_view = calib.view()
    plane = calib.accept()
    print(f"plane: {plane.source}, camera ~{plane.elevation_deg():.0f} deg above the table, "
          f"field of view ~{plane.hfov_deg:.0f} deg, warnings: {plane.warnings() or 'none'}")

    mat = mat_at_image_point(plane, (0.47, 0.75), 30, 22).rotated(-4)
    c = mat.image_quad(plane).mean(axis=0)
    hands = [hand_landmarks(c + (-150, 12), 92, -12, right=False), hand_landmarks(c + (150, 12), 92, 12)]
    with_hands = view.copy()
    for h in hands:
        draw_hand(with_hands, h)

    solves = [Solve(t, Penalty.OK, "", "", 30.0, None) for t in (61.42, 58.07, 64.9, 55.31, 59.88, 62.2)]
    stats = compute_stats(solves)
    recent = [format_solve(s) for s in solves[-5:]]
    still = Renderer(animate=False)      # stills: no cross-fades, pulses or ripples
    live = Renderer(animate=True)        # for the animated moments, driven by an explicit clock
    for r in (still, live):
        r.ar.draw(view.copy(), mat.image_quad(plane), (mat.width_cm, mat.height_cm))  # learn the table
        r.ar.wait_idle()

    def vm_for(state, time_text, color, sub, zone_mode, progress=0.0, ar_mat=mat, ar_plane=plane, hands_px=(),
               zone=None, calib_v=None, setup_v=None, zone_txt="AR mat", **extra):
        ar = None
        if ar_mat is not None:
            ar = ArView(ar_mat.image_quad(ar_plane), (ar_mat.width_cm, ar_mat.height_cm),
                        zone_mode if zone_mode != "setup" else "idle", progress, list(hands_px), ar_mat.shape)
        return ViewModel(**extra,
            state=state, time_text=time_text, time_color=color, sub_text=sub, scramble=SCRAMBLE, stats=stats,
            recent=recent, zone=zone if zone is not None else (ar_mat or mat).zone(ar_plane), zone_mode=zone_mode,
            hold_progress=progress,
            status_lines=["Camera 1280x720 (synthetic table) | processed 30 fps | timing resolution +/-33 ms",
                          f"Zone: {zone_txt} | State {state.value} | inspection OFF"],
            ar=ar, calib=calib_v, setup=setup_v,
        )

    def save(name, frame):
        cv2.imwrite(str(out / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print("wrote", out / name)

    def shot(name, img, *a, **kw):
        vm = vm_for(*a, **kw)
        still.render(img.copy(), vm)
        still.ar.wait_idle()             # a new area position: let the helper thread prepare every state
        save(name, still.render(img.copy(), vm))

    def live_frames(steps, quad_mat=mat):
        """Render a sequence (img, now, vm args) with the animated renderer; returns the last frame."""
        frame = None
        for i, (img, now, a, kw) in enumerate(steps):
            frame = live.render(img.copy(), vm_for(*a, ar_mat=quad_mat, **kw), now=now)
            if i == 0:
                live.ar.wait_idle()
        return frame

    shot("1_idle.jpg", view, State.IDLE, "59.88", WHITE, "", "idle")
    shot("2_holding.jpg", with_hands, State.IDLE, "59.88", YELLOW, "", "holding", 0.55, hands_px=hands)
    # ready, at the top of its soft pulse
    t0 = 100.0
    ready_args = ((State.READY, "59.88", GREEN, READY, "ready"), dict(hands_px=hands))
    steps = [(with_hands, t0 + i / 30, *ready_args) for i in range(0, int(T.PULSE_S / 2 * 30) + 1)]
    save("3_ready.jpg", live_frames(steps))
    shot("4_solving.jpg", view, State.SOLVING, "23.41", WHITE, "", "solving")
    # the ripple a moment after the solve starts (hands lifted)
    t1 = 200.0
    steps = [(with_hands, t1 + i / 30, *ready_args) for i in range(12)]
    steps += [(view, t1 + 0.4 + i / 30, (State.SOLVING, "0.%02d" % (i * 3), WHITE, "", "solving"), {})
              for i in range(int(0.36 * 30))]
    save("4b_solve_start_ripple.jpg", live_frames(steps))
    shot("5_ar_off.jpg", with_hands, State.READY, "59.88", GREEN, READY, "ready", ar_mat=None,
         zone=mat.zone(plane), zone_txt="mat (AR off)")
    # calibration screen: show it on the plain view, with the phone outlined and the 10 cm grid
    shot("6_calibration.jpg", view, State.IDLE, "59.88", WHITE, "", "setup", ar_mat=None, zone=mat.zone(plane),
         calib_v=calib_view)
    # mat placement screen
    setup = ZoneSetup()
    setup.start(plane=plane, mat_size=(30, 22), current_mat=mat)
    shot("7_place_mat.jpg", view, State.IDLE, "59.88", WHITE, "", "setup", setup_v=setup.view())
    shot("8_help.jpg", view, State.IDLE, "59.88", WHITE, "", "idle", show_help=True)
    shot("9_plain_zone_holding.jpg", with_hands, State.IDLE, "59.88", YELLOW, "", "holding", 0.55, ar_mat=None,
         zone=mat.zone(plane), zone_txt="mat (AR off)", message="Zone set (fixed), saved to config.yaml")

    # --- draw mode on the calibrated table
    path = traced_path(c + (0, 4), 300, 92)
    setup = ZoneSetup()
    setup.start(SetupMode.DRAW, plane=plane, mat_size=(30, 22))
    trace(setup, path, upto=0.8)
    shot("10_draw_tracing.jpg", view, State.IDLE, "59.88", WHITE, "", "setup", ar_mat=None, zone=mat.zone(plane),
         setup_v=setup.view())
    setup.on_mouse(EVENT_LBUTTONUP, *map(int, path[int(len(path) * 0.8)]), W, H)   # let go early...
    setup.redo()
    trace(setup, path)                                                              # ...and trace it all
    drawn = setup.candidate_mat
    print(f"drawn area: {len(drawn.shape)} points, {drawn.width_cm:.0f} x {drawn.height_cm:.0f} cm")
    shot("11_draw_shape.jpg", view, State.IDLE, "59.88", WHITE, "", "setup", ar_mat=drawn, setup_v=setup.view())
    dc = drawn.image_quad(plane).mean(axis=0)
    dhands = [hand_landmarks(dc + (-170, 8), 92, -10, right=False), hand_landmarks(dc + (170, 8), 92, 10)]
    d_with_hands = view.copy()
    for h in dhands:
        draw_hand(d_with_hands, h)
    shot("12_drawn_holding.jpg", d_with_hands, State.IDLE, "59.88", YELLOW, "", "holding", 0.7, ar_mat=drawn,
         hands_px=dhands, zone_txt="AR mat (drawn)")
    t2 = 300.0
    d_ready = ((State.READY, "59.88", GREEN, READY, "ready"), dict(hands_px=dhands))
    steps = [(d_with_hands, t2 + i / 30, *d_ready) for i in range(0, int(T.PULSE_S / 2 * 30) + 1)]
    save("13_drawn_ready.jpg", live_frames(steps, drawn))
    # --- draw mode without a table calibration: a shape on the screen
    flat = ZoneSetup()
    flat.start(SetupMode.DRAW)
    trace(flat, path)
    mode, zone, _ = flat.accept()
    shot("14_drawn_screen_shape.jpg", d_with_hands, State.IDLE, "59.88", YELLOW, "", "holding", 0.55, ar_mat=None,
         zone=zone, zone_txt="polygon", message="Zone set (drawn shape), saved to config.yaml")
    still.close()
    live.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
