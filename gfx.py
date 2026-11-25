"""Drawing helpers for the UI: TrueType text (Pillow), rounded translucent panels, blending.

Text is rendered by Pillow/FreeType into small alpha masks that are cached, then
blended into the camera frame with OpenCV, so the frame itself never goes
through Pillow (that conversion alone would cost several ms per frame).

Fonts: Segoe UI and Bahnschrift on Windows, the usual sans fonts elsewhere,
any .ttf/.otf put in assets/fonts/, and Pillow's built-in scalable font as
the last resort, so there is always a TrueType font.
"""
from __future__ import annotations

import os
import sys
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent

# role -> candidate files (first one found wins), optional variable-font instance
FONT_CANDIDATES = {
    "regular": (["segoeui.ttf", "SegoeUI.ttf", "Inter-Regular.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf",
                 "Arial.ttf", "arial.ttf", "Helvetica.ttc"], None),
    "semibold": (["seguisb.ttf", "segoeuib.ttf", "Inter-SemiBold.ttf", "DejaVuSans-Bold.ttf",
                  "LiberationSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf"], None),
    # timer digits: equal-width figures so the time doesn't jitter
    "digits": (["bahnschrift.ttf", "Inter-SemiBold.ttf", "seguisb.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf",
                "arialbd.ttf"], b"SemiBold"),
}


def _font_dirs() -> list[Path]:
    dirs = [ROOT / "assets" / "fonts"]
    if sys.platform == "win32":
        windir = os.environ.get("WINDIR", r"C:\Windows")
        dirs.append(Path(windir) / "Fonts")
        local = os.environ.get("LOCALAPPDATA")
        if local:
            dirs.append(Path(local) / "Microsoft" / "Windows" / "Fonts")
    elif sys.platform == "darwin":
        dirs += [Path("/System/Library/Fonts"), Path("/Library/Fonts"), Path.home() / "Library" / "Fonts"]
    else:
        dirs += [Path("/usr/share/fonts"), Path("/usr/local/share/fonts"), Path.home() / ".fonts"]
    return [d for d in dirs if d.is_dir()]


@lru_cache(maxsize=None)
def font_path(role: str) -> str | None:
    """Path of the font used for a role, or None for Pillow's built-in font."""
    names, _ = FONT_CANDIDATES[role]
    dirs = _font_dirs()
    for name in names:
        for d in dirs:
            p = d / name
            if p.is_file():
                return str(p)
        for d in dirs:  # Linux keeps fonts in sub-folders
            if d.name in ("fonts", ".fonts"):
                hits = list(d.rglob(name))
                if hits:
                    return str(hits[0])
    return None


@lru_cache(maxsize=64)
def font(role: str, px: int) -> ImageFont.FreeTypeFont:
    px = max(6, int(px))
    path = font_path(role)
    if path is None:
        return ImageFont.load_default(px)
    f = ImageFont.truetype(path, px)
    variation = FONT_CANDIDATES[role][1]
    if variation:
        try:
            f.set_variation_by_name(variation)
        except (OSError, ValueError):
            pass  # not a variable font
    return f


def text_width(text: str, role: str, px: int) -> int:
    if role == "digits":
        return text_mask(text, role, px).shape[1] - 3
    return int(round(font(role, px).getlength(text)))


def line_height(role: str, px: int) -> int:
    asc, desc = font(role, px).getmetrics()
    return asc + desc


_MASKS: "OrderedDict[tuple, np.ndarray]" = OrderedDict()


def text_mask(text: str, role: str, px: int) -> np.ndarray:
    """uint8 alpha mask of one line of text: width = advance, height = ascent + descent."""
    key = (text, role, int(px))
    m = _MASKS.get(key)
    if m is not None:
        _MASKS.move_to_end(key)
        return m
    f = font(role, px)
    asc, desc = f.getmetrics()
    if role == "digits":
        m = _tabular(text, f, asc + desc)
    else:
        w = max(1, int(np.ceil(f.getlength(text))) + 2)
        im = Image.new("L", (w, asc + desc))
        ImageDraw.Draw(im).text((1, 0), text, font=f, fill=255)
        m = np.asarray(im)
    _MASKS[key] = m
    while len(_MASKS) > 768:
        _MASKS.popitem(last=False)
    return m


@lru_cache(maxsize=512)
def _glyph(ch: str, f: ImageFont.FreeTypeFont, height: int) -> np.ndarray:
    w = max(1, int(np.ceil(f.getlength(ch))) + 2)
    im = Image.new("L", (w, height))
    ImageDraw.Draw(im).text((1, 0), ch, font=f, fill=255)
    return np.asarray(im)


def _tabular(text: str, f: ImageFont.FreeTypeFont, height: int) -> np.ndarray:
    """Text with every digit in a cell of the same width, so a running time doesn't jitter sideways
    (many fonts, Bahnschrift included, have proportional figures)."""
    cell = max(f.getlength(d) for d in "0123456789")
    widths = [cell if ch.isdigit() else f.getlength(ch) for ch in text]
    out = np.zeros((height, int(np.ceil(sum(widths))) + 3), np.uint8)
    x = 1.0
    for ch, adv in zip(text, widths):
        g = _glyph(ch, f, height)
        gx = int(round(x + (adv - f.getlength(ch)) / 2)) - 1     # centre the glyph in its cell
        x0, x1 = max(0, gx), min(out.shape[1], gx + g.shape[1])
        if x1 > x0:
            np.maximum(out[:, x0:x1], g[:, x0 - gx:x1 - gx], out=out[:, x0:x1])
        x += adv
    return out


# ------------------------------------------------------------------ blending
def blend_mask(img: np.ndarray, mask: np.ndarray, x: int, y: int, color, opacity: float = 1.0) -> None:
    """Paint `color` into img through a uint8 / float mask placed with its top-left at (x, y)."""
    h, w = mask.shape[:2]
    H, W = img.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0 or opacity <= 0:
        return
    m = mask[y0 - y:y1 - y, x0 - x:x1 - x]
    a = m.astype(np.float32) * (opacity / 255.0) if m.dtype == np.uint8 else m.astype(np.float32) * opacity
    roi = img[y0:y1, x0:x1]
    col = np.empty_like(roi)
    col[:] = color
    roi[:] = cv2.blendLinear(col, roi, a, 1.0 - a)


def draw_text(img: np.ndarray, text: str, x: float, y: float, px: float, color, role: str = "regular",
              anchor: str = "lt", opacity: float = 1.0, shadow: float = 0.0) -> tuple[int, int, int, int]:
    """Draw one line. anchor: [l|m|r][t|m|b] for the line box. Returns the box (x0, y0, x1, y1)."""
    if not text:
        return (int(x), int(y), int(x), int(y))
    m = text_mask(text, role, int(round(px)))
    h, w = m.shape
    ax, ay = anchor[0], anchor[1]
    x0 = int(round(x - (w / 2 if ax == "m" else w if ax == "r" else 0)))
    y0 = int(round(y - (h / 2 if ay == "m" else h if ay == "b" else 0)))
    if shadow > 0:
        blur = max(1.0, px * 0.05)
        sh = _shadow(text, role, int(round(px)), blur)
        pad = int(blur * 3)
        blend_mask(img, sh, x0 - pad + int(px * 0.02), y0 - pad + int(px * 0.03), (0, 0, 0), shadow * opacity)
    blend_mask(img, m, x0, y0, color, opacity)
    return (x0, y0, x0 + w, y0 + h)


_SHADOWS: "OrderedDict[tuple, np.ndarray]" = OrderedDict()


def _shadow(text: str, role: str, px: int, blur: float) -> np.ndarray:
    key = (text, role, px)
    s = _SHADOWS.get(key)
    if s is None:
        m = text_mask(text, role, px)
        pad = int(blur * 3)
        s = cv2.GaussianBlur(cv2.copyMakeBorder(m, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0), (0, 0), blur)
        _SHADOWS[key] = s
        while len(_SHADOWS) > 128:
            _SHADOWS.popitem(last=False)
    return s


def wrap(text: str, max_w: int, px: int, role: str = "regular") -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if cur and text_width(trial, role, px) > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


# -------------------------------------------------------------------- panels
@lru_cache(maxsize=256)
def rounded_mask(w: int, h: int, r: int, ss: int = 3) -> np.ndarray:
    """Anti-aliased rounded-rectangle alpha mask (float32 0..1), cached per size."""
    r = max(0, min(r, w // 2, h // 2))
    big = np.zeros((h * ss, w * ss), np.uint8)
    R = r * ss
    if R == 0:
        big[:] = 255
    else:
        cv2.rectangle(big, (R, 0), (w * ss - R - 1, h * ss - 1), 255, -1)
        cv2.rectangle(big, (0, R), (w * ss - 1, h * ss - R - 1), 255, -1)
        for cx, cy in ((R, R), (w * ss - R - 1, R), (R, h * ss - R - 1), (w * ss - R - 1, h * ss - R - 1)):
            cv2.circle(big, (cx, cy), R, 255, -1, cv2.LINE_AA)
    m = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    m.setflags(write=False)
    return m


@lru_cache(maxsize=256)
def rounded_ring(w: int, h: int, r: int, width: float = 1.0) -> np.ndarray:
    outer = rounded_mask(w, h, r)
    inner = np.zeros_like(outer)
    d = max(1, int(round(width)))
    if w > 2 * d and h > 2 * d:
        inner[d:h - d, d:w - d] = rounded_mask(w - 2 * d, h - 2 * d, max(0, r - d))
    m = np.clip(outer - inner, 0, 1)
    m.setflags(write=False)
    return m


def panel(img: np.ndarray, x: int, y: int, w: int, h: int, radius: int, color, alpha: float,
          edge_color=None, edge_alpha: float = 0.0) -> None:
    """Translucent rounded panel with an optional hairline border."""
    w, h = int(w), int(h)
    if w <= 0 or h <= 0:
        return
    blend_mask(img, rounded_mask(w, h, int(radius)), int(x), int(y), color, alpha)
    if edge_color is not None and edge_alpha > 0:
        blend_mask(img, rounded_ring(w, h, int(radius)), int(x), int(y), edge_color, edge_alpha)


def fill_poly(img: np.ndarray, pts: np.ndarray, color, alpha: float) -> None:
    """Translucent anti-aliased polygon (only its bounding box is touched)."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    H, W = img.shape[:2]
    x0, y0 = max(0, int(np.floor(p[:, 0].min()))), max(0, int(np.floor(p[:, 1].min())))
    x1, y1 = min(W, int(np.ceil(p[:, 0].max())) + 1), min(H, int(np.ceil(p[:, 1].max())) + 1)
    if x1 <= x0 or y1 <= y0:
        return
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(mask, [np.int32(np.round((p - (x0, y0)) * 16))], 255, cv2.LINE_AA, shift=4)
    blend_mask(img, mask, x0, y0, color, alpha)


def lerp_color(a, b, t: float) -> tuple[int, int, int]:
    t = min(max(t, 0.0), 1.0)
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))
