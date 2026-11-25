"""One small palette and set of sizes, shared by the on-screen panels and the AR mat.

Colours are BGR (OpenCV order). Sizes are pixels at 720p and scaled with the frame.
"""
from __future__ import annotations


def rgb(r: int, g: int, b: int) -> tuple[int, int, int]:
    return (b, g, r)


# ---------------------------------------------------------------- surfaces
PANEL = rgb(17, 20, 26)          # dark slate, used at PANEL_ALPHA over the camera picture
PANEL_ALPHA = 0.66
PANEL_EDGE = rgb(255, 255, 255)  # hairline border at EDGE_ALPHA
EDGE_ALPHA = 0.10
SHADOW = rgb(0, 0, 0)

# -------------------------------------------------------------------- text
TEXT = rgb(241, 244, 248)
TEXT_MUTED = rgb(148, 158, 172)
TEXT_FAINT = rgb(100, 110, 124)

# ------------------------------------------------------------------ accents
ACCENT = rgb(96, 196, 255)       # interactive things: setup, calibration, selection
WARN = rgb(255, 166, 77)
DANGER = rgb(255, 94, 94)

# timer / mat states (the same colours on the timer, the zone and the AR mat)
STATE = {
    "idle": rgb(148, 163, 184),
    "holding": rgb(255, 196, 61),
    "ready": rgb(64, 214, 140),
    "solving": rgb(255, 108, 96),
    "setup": ACCENT,
    "off": rgb(148, 163, 184),
}

# ------------------------------------------------------------------- the mat
MAT_BODY_TOP = rgb(44, 49, 58)       # graphite, lit a little from the far side
MAT_BODY_BOTTOM = rgb(30, 34, 41)
MAT_RIM = rgb(18, 21, 26)
MAT_HIGHLIGHT = rgb(96, 104, 118)
MAT_PAD = rgb(52, 58, 69)
MAT_PAD_EDGE = rgb(78, 86, 100)
MAT_ICON = rgb(98, 108, 124)
MAT_WINDOW = rgb(14, 16, 20)

# ------------------------------------------------------------------- layout
MARGIN = 16          # distance from the frame edge
PAD = 14             # inside panels
GAP = 10             # between panels
RADIUS = 14          # panel corners
RADIUS_PILL = 999    # fully rounded ends

# font sizes (px at 720p)
SIZE_TIMER = 156
SIZE_SUB = 21
SIZE_SCRAMBLE = 25
SIZE_BODY = 18
SIZE_SMALL = 15
SIZE_TITLE = 20

# animation
FADE_S = 0.18        # colour / state cross-fades
TOAST_FADE_S = 0.35  # messages fade out over this long
