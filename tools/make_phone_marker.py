"""Make the phone-screen calibration marker for the table plane (key T in CubeCam).

    python tools/make_phone_marker.py                       # phone_marker.png, 1080 x 2340 (portrait)
    python tools/make_phone_marker.py --width 1170 --height 2532 --out my_phone.png

Open the PNG on your phone (full screen, no zoom; most photo viewers do this
by default), turn the brightness up, and lay the phone flat on the table.

Measure the black square ONCE with a ruler (outer edge to outer edge, in mm)
and put that number in config.yaml as table.phone_marker_mm. It sets the
real-world scale. Without it a typical phone is assumed and sizes are
approximate.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from markers import DEFAULT_DICT, get_dictionary  # noqa: E402

DEFAULT_ID = 10


def make_phone_marker(width: int = 1080, height: int = 2340, marker_id: int = DEFAULT_ID,
                      dict_name: str = DEFAULT_DICT, fill: float = 0.82) -> np.ndarray:
    """Portrait phone-screen image: white background, one large marker in the middle.

    The marker side is a whole number of pixels per module (crisp edges at
    100 % zoom), about `fill` of the screen width; the white around it is the
    quiet zone ArUco needs.
    """
    dictionary = get_dictionary(dict_name)
    modules = dictionary.markerSize + 2          # data bits + the black border
    per = max(4, int(width * fill) // modules)
    side = per * modules
    img = np.full((height, width), 255, np.uint8)
    marker = cv2.aruco.generateImageMarker(dictionary, int(marker_id), side)
    x0, y0 = (width - side) // 2, (height - side) // 2
    img[y0:y0 + side, x0:x0 + side] = marker
    s = width / 1080
    lines = ["CubeCam table marker", "Lay the phone flat, brightness up.",
             "Measure the black square once (mm)", "-> config.yaml table.phone_marker_mm"]
    y = y0 + side + int(110 * s)
    for i, text in enumerate(lines):
        scale = (1.5 if i == 0 else 1.05) * s
        thick = max(1, int((3 if i == 0 else 2) * s))
        tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0]
        cv2.putText(img, text, ((width - tw) // 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, 90, thick, cv2.LINE_AA)
        y += int((80 if i == 0 else 62) * s)
    return img


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--width", type=int, default=1080, help="phone screen width in pixels (default 1080)")
    ap.add_argument("--height", type=int, default=2340, help="phone screen height in pixels (default 2340)")
    ap.add_argument("--id", type=int, default=DEFAULT_ID, help=f"marker id (default {DEFAULT_ID}, = table.phone_marker_id)")
    ap.add_argument("--dict", default=DEFAULT_DICT, help=f"ArUco dictionary (default {DEFAULT_DICT})")
    ap.add_argument("--out", default="phone_marker.png")
    args = ap.parse_args(argv)
    if args.width < 300 or args.height < args.width:
        ap.error("expects a portrait screen at least 300 px wide")
    img = make_phone_marker(args.width, args.height, args.id, args.dict)
    if not cv2.imwrite(args.out, img):
        print(f"could not write {args.out}", file=sys.stderr)
        return 1
    print(f"wrote {args.out} ({args.width}x{args.height}, marker id {args.id}, {args.dict}).")
    print("Open it full screen on your phone, brightness up, lay the phone flat on the table, press T in CubeCam.")
    print("Measure the black square once with a ruler and set table.phone_marker_mm in config.yaml.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
