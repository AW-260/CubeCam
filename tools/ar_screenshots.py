"""Render screenshots of the AR start mat on a synthetic table (no camera needed).

    python tools/ar_screenshots.py                 # -> docs/screenshots/ar_mat/*.jpg

It builds a synthetic camera view of a wooden table with a phone showing the
calibration marker, calibrates the table plane from it exactly like key T
does, places the mat, and renders the real app UI (ui.Renderer) for each
state: idle, holding, ready, solving, AR off, the calibration screen and the
mat placement screen. Hands are drawn into the picture so you can see them
covering the mat.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from ar_mat import mat_at_image_point  # noqa: E402
from make_phone_marker import make_phone_marker  # noqa: E402
from markers import detect_marker_corners, make_detector  # noqa: E402
from models import Penalty, Solve, format_solve  # noqa: E402
from stats import compute_stats  # noqa: E402
from synth_table import SynthCamera, TableScene, draw_hand, hand_landmarks, phone_image  # noqa: E402
from table_calib import CalibMode, TableCalibration  # noqa: E402
from timer_state import State  # noqa: E402
from ui import GREEN, WHITE, YELLOW, ArView, Renderer, ViewModel  # noqa: E402
from zone_setup import ZoneSetup  # noqa: E402

W, H = 1280, 720


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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "docs" / "screenshots" / "ar_mat"))
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
    quad = mat.image_quad(plane)
    zone = mat.zone(plane)
    c = quad.mean(axis=0)
    hands = [hand_landmarks(c + (-150, 12), 92, -12, right=False), hand_landmarks(c + (150, 12), 92, 12)]
    with_hands = view.copy()
    for h in hands:
        draw_hand(with_hands, h)

    solves = [Solve(t, Penalty.OK, "", "", 30.0, None) for t in (61.42, 58.07, 64.9, 55.31, 59.88, 62.2)]
    stats = compute_stats(solves)
    recent = [format_solve(s) for s in solves[-5:]]
    renderer = Renderer()
    renderer.ar.draw(view.copy(), quad, (mat.width_cm, mat.height_cm))  # learn the table, prepare states
    renderer.ar.wait_idle()

    def shot(name, img, state, time_text, color, sub, zone_mode, progress=0.0, ar=True, hands_px=(),
             calib_v=None, setup_v=None, status=""):
        vm = ViewModel(
            state=state, time_text=time_text, time_color=color, sub_text=sub,
            scramble="R U2 F' L D2 B R' U F2 L2 D' R2 B2 U' F L' D R2 U2 B'", stats=stats, recent=recent,
            zone=zone, zone_mode=zone_mode, hold_progress=progress,
            status_lines=[status or "Camera 1280x720 (synthetic table) | processed 30 fps | timing resolution +/-33 ms",
                          f"Zone: {'AR mat' if ar else 'mat (AR off)'} | State {state.value} | inspection OFF"],
            ar=ArView(quad, (mat.width_cm, mat.height_cm), zone_mode if zone_mode != "setup" else "idle",
                      progress, list(hands_px)) if ar else None,
            calib=calib_v, setup=setup_v,
        )
        frame = renderer.render(img.copy(), vm)
        cv2.imwrite(str(out / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print("wrote", out / name)

    shot("1_idle.jpg", view, State.IDLE, "59.88", WHITE, "", "idle")
    shot("2_holding.jpg", with_hands, State.IDLE, "59.88", YELLOW, "", "holding", 0.55, hands_px=hands)
    shot("3_ready.jpg", with_hands, State.READY, "59.88", GREEN, "READY - lift to start", "ready", hands_px=hands)
    shot("4_solving.jpg", view, State.SOLVING, "23.41", WHITE, "", "solving")
    shot("5_ar_off.jpg", with_hands, State.READY, "59.88", GREEN, "READY - lift to start", "ready", ar=False)
    # calibration screen: show it on the plain view, with the phone outlined and the 10 cm grid
    shot("6_calibration.jpg", view, State.IDLE, "59.88", WHITE, "", "setup", ar=False, calib_v=calib_view)
    # mat placement screen
    setup = ZoneSetup()
    setup.start(plane=plane, mat_size=(30, 22), current_mat=mat)
    shot("7_place_mat.jpg", view, State.IDLE, "59.88", WHITE, "", "setup", setup_v=setup.view())
    renderer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
