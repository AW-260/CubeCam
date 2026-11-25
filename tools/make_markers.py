"""Print the 4 ArUco markers for CubeCam's marker mode (zone setup, key M).

    python tools/make_markers.py                    # markers.png + markers.pdf (A4, 60 mm markers)
    python tools/make_markers.py --size 50 --out my_markers

Print at 100 % / "actual size" (not "fit to page"), cut the 4 markers out along
the dashed lines and tape them at the 4 corners of your mat. The page itself
repeats these instructions.
"""
from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from markers import DEFAULT_DICT, DEFAULT_IDS, get_dictionary  # noqa: E402

DPI = 300
A4_MM = (210.0, 297.0)
MM = DPI / 25.4  # pixels per millimetre

INSTRUCTIONS = [
    "CubeCam start-zone markers",
    "",
    "1. Print at 100% / 'actual size' (not 'fit to page'). Check: the scale bar is 50 mm.",
    "2. Cut out the 4 squares along the dashed lines. Keep the white border round each marker.",
    "3. Tape them flat at the 4 corners of the area where you rest your hands.",
    "   The zone is the shape through the marker CENTRES. Any marker can go at any corner,",
    "   either way up. Leave a hand's width of room between your hands and the edges.",
    "4. Camera: it must see the desk and all 4 markers at once. Tilt it down so the",
    "   mat is in the lower part of the picture, and avoid glare on the paper.",
    "5. In CubeCam press M three times (marker mode), wait for '4/4 markers', press ENTER.",
    "   The zone then follows the markers if the camera moves. A marker hidden under",
    "   a hand is fine: the last known zone is kept.",
]


def _dashed_rect(img, x1, y1, x2, y2, dash=int(4 * MM), color=150, thick=3):
    for (ax, ay), (bx, by) in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        length = int(np.hypot(bx - ax, by - ay))
        for s in range(0, length, 2 * dash):
            e = min(s + dash, length)
            p = (int(ax + (bx - ax) * s / length), int(ay + (by - ay) * s / length))
            q = (int(ax + (bx - ax) * e / length), int(ay + (by - ay) * e / length))
            cv2.line(img, p, q, color, thick)


def _text(img, s, org, scale=1.25, thick=2):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, 0, thick, cv2.LINE_AA)


def make_page(size_mm: float = 60.0, ids=DEFAULT_IDS, dict_name: str = DEFAULT_DICT) -> np.ndarray:
    """A4 page (grayscale, 300 dpi) with 4 markers in a 2x2 grid and instructions."""
    W, H = round(A4_MM[0] * MM), round(A4_MM[1] * MM)
    page = np.full((H, W), 255, np.uint8)
    dictionary = get_dictionary(dict_name)
    side = round(size_mm * MM)
    border = round(8 * MM)              # white quiet zone kept when cutting
    cell = side + 2 * border
    gap = round(12 * MM)
    x0 = (W - (2 * cell + gap)) // 2
    y0 = round(18 * MM)
    for k, marker_id in enumerate(ids[:4]):
        r, c = divmod(k, 2)
        cx, cy = x0 + c * (cell + gap), y0 + r * (cell + gap)
        img = cv2.aruco.generateImageMarker(dictionary, int(marker_id), side)
        page[cy + border:cy + border + side, cx + border:cx + border + side] = img
        _dashed_rect(page, cx, cy, cx + cell, cy + cell)
        _text(page, f"id {marker_id}", (cx + border, cy + cell - round(2.2 * MM)), 1.0, 2)
    # instructions
    y = y0 + 2 * cell + gap + round(14 * MM)
    for i, line in enumerate(INSTRUCTIONS):
        if line:
            _text(page, line, (round(15 * MM), y), 1.6 if i == 0 else 1.15, 3 if i == 0 else 2)
        y += round(6.2 * MM) if i else round(9 * MM)
    # 50 mm scale bar to check the print scale
    y += round(4 * MM)
    bx = round(15 * MM)
    cv2.rectangle(page, (bx, y), (bx + round(50 * MM), y + round(3 * MM)), 0, -1)
    _text(page, f"50 mm   (markers: {size_mm:g} mm, {dict_name})", (bx + round(55 * MM), y + round(3 * MM)), 1.0, 2)
    return page


def write_pdf(path: Path, gray: np.ndarray, page_mm=A4_MM) -> None:
    """Single-page PDF holding the image at full page size (no extra dependencies)."""
    h, w = gray.shape
    pw, ph = page_mm[0] / 25.4 * 72, page_mm[1] / 25.4 * 72
    data = zlib.compress(np.ascontiguousarray(gray).tobytes(), 9)
    content = f"q {pw:.2f} 0 0 {ph:.2f} 0 0 cm /Im0 Do Q".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {pw:.2f} {ph:.2f}] "
        f"/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>".encode(),
        f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace /DeviceGray "
        f"/BitsPerComponent 8 /Filter /FlateDecode /Length {len(data)} >>\nstream\n".encode() + data + b"\nendstream",
        f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=float, default=60.0, help="marker side in mm (default 60, max 85)")
    ap.add_argument("--out", default="markers", help="output path without extension (default: markers)")
    ap.add_argument("--dict", default=DEFAULT_DICT, help=f"ArUco dictionary (default {DEFAULT_DICT})")
    ap.add_argument("--ids", type=int, nargs=4, default=list(DEFAULT_IDS), help="the 4 marker ids (default 0 1 2 3)")
    args = ap.parse_args(argv)
    if not 20 <= args.size <= 85:
        ap.error("--size must be between 20 and 85 mm to fit 4 markers and the instructions on A4")
    page = make_page(args.size, tuple(args.ids), args.dict)
    out = Path(args.out)
    png, pdf = out.with_suffix(".png"), out.with_suffix(".pdf")
    if not cv2.imwrite(str(png), page, [cv2.IMWRITE_PNG_COMPRESSION, 9]):
        print(f"could not write {png}", file=sys.stderr)
        return 1
    write_pdf(pdf, page)
    print(f"wrote {png} and {pdf} (A4, {args.size:g} mm markers, ids {args.ids}, {args.dict}).")
    print("Print at 100% / actual size, cut out, tape to the mat corners, then press M x3 in CubeCam.")
    if args.dict != DEFAULT_DICT or list(args.ids) != list(DEFAULT_IDS):
        print("Remember to set markers.dictionary / markers.ids in config.yaml to match.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
