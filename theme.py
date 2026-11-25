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

# ------------------------------------------------------- the holographic mat
# Layers of the start area on the table, as opacities (0..1) per state. Each
# layer is drawn in that state's colour (fill: PANEL, a slight darkening so the
# light reads on a bright desk; core: the state colour mixed with TEXT, "hot").
#   fill  - the whole area      edge  - light along the inside of the outline
#   haze  - wide soft glow      grid  - the faint table grid
#   glow  - glow round the line core  - the outline itself
HOLO = {
    "idle":    {"fill": 0.14, "edge": 0.22, "haze": 0.22, "grid": 0.42, "glow": 0.50, "core": 0.85},
    "holding": {"fill": 0.12, "edge": 0.10, "haze": 0.10, "grid": 0.24, "glow": 0.14, "core": 0.30},
    "ready":   {"fill": 0.14, "edge": 0.32, "haze": 0.38, "grid": 0.42, "glow": 0.62, "core": 0.95},
    "solving": {"fill": 0.10, "edge": 0.12, "haze": 0.14, "grid": 0.14, "glow": 0.24, "core": 0.50},
}
# the hologram's colour per state: the state colours, with idle's slate tinted towards ACCENT
HOLO_COLOR = {**STATE, "idle": tuple(round(a * 0.4 + b * 0.6) for a, b in zip(STATE["idle"], ACCENT))}
HOT_MIX = 0.45       # how much TEXT (near white) is mixed into the outline's core
HOLD_LIGHT = 1.0     # holding: the light that runs round the outline
PULSE_S = 1.5        # ready: one soft pulse
PULSE_LOW, PULSE_HIGH = 0.10, 0.50
RIPPLE_S = 0.9       # solve start / stop: the ripple crossing the area
RIPPLE_FRONT, RIPPLE_TAIL = 0.035, 0.16   # ring: sharp leading edge, soft tail (fractions of the radius)
RIPPLE_ALPHA = 0.42

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
