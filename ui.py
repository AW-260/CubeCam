"""Rendering. Pure drawing: main.py builds a ViewModel, ui draws it over the camera picture.

Layout (720p reference, scaled with the frame):
  * top centre: the scramble in a rounded panel
  * left column: a compact session-stats panel
  * centre: the big timer (TrueType digits with a soft shadow), a state line under it
  * top stack under the scramble: setup / calibration instructions, then messages
  * right column: a small "? help" hint, warnings, the H debug panel
  * the bottom of the picture is left free for the mat; keys and status live in
    the help card (? or F1)

Text is TrueType (gfx.py); colours and sizes come from theme.py, shared with the AR mat.
State colours cross-fade instead of switching, and messages fade out.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

import theme as T
from ar_mat import ArMatRenderer
from gfx import (blend_mask, draw_text, fill_poly, line_height, panel, rounded_mask, rounded_ring, text_width,
                 wrap)
from hands import HAND_CONNECTIONS
from models import format_time
from stats import SessionStats
from table_calib import CalibView
from timer_state import State
from zone import Zone
from zone_setup import SetupView

# Timer colours used by main.py (names kept from before; values from the shared palette)
WHITE = T.TEXT
GREY = T.TEXT_MUTED
YELLOW = T.STATE["holding"]
GREEN = T.STATE["ready"]
ORANGE = T.WARN
RED = T.DANGER
CYAN = T.ACCENT

HELP_KEYS = [
    ("Space", "start / stop (keyboard)"), ("Esc", "abort the solve"),
    ("2", "+2 on last solve"), ("D", "DNF on last solve"),
    ("Backspace", "delete last solve"), ("N", "new scramble"),
    ("I", "inspection on / off"), ("M", "zone / mat setup"),
    ("T", "table calibration"), ("A", "AR mat look on / off"),
    ("H", "debug overlay"), ("? / F1", "this help"),
    ("Q", "quit"), ("", ""),
]
HELP = "  ".join(f"{k} {v}" for k, v in HELP_KEYS if k)   # one-line version (kept for reference)


@dataclass
class ArView:
    """The AR mat to draw this frame."""

    quad: np.ndarray                         # (4, 2) image px of the bounding box: far-left, far-right, near-right, near-left
    size_cm: tuple[float, float]
    state: str = "idle"                      # idle | holding | ready | solving
    progress: float = 0.0
    hands_px: list = field(default_factory=list)   # (21, 2) landmark arrays in image px
    shape: tuple | None = None               # a drawn outline (cm, see ar_mat.Mat.shape); None = rectangle


@dataclass
class ViewModel:
    state: State
    time_text: str
    time_color: tuple = WHITE
    sub_text: str = ""                       # e.g. "READY - lift to start"
    scramble: str = ""
    stats: SessionStats | None = None
    recent: list[str] = field(default_factory=list)
    zone: Zone | None = None
    zone_mode: str = "idle"                  # idle | holding | ready | solving | setup | off
    hold_progress: float = 0.0
    status_lines: list[str] = field(default_factory=list)   # shown in the help card
    message: str | None = None
    message_alpha: float = 1.0               # messages fade out
    flash: tuple | None = None               # border colour for inspection warnings
    flash_level: float = 1.0                 # 1 -> 0 while it fades
    show_debug: bool = False
    show_help: bool = False
    debug: dict[str, Any] = field(default_factory=dict)
    setup: SetupView | None = None           # zone setup overlay (key M)
    zone_warning: bool = False               # zone mostly off-screen / too small
    warnings: list[str] = field(default_factory=list)       # small warning pills (right column)
    calib: CalibView | None = None           # table calibration overlay (key T)
    ar: ArView | None = None                 # AR mat (replaces the flat zone outline)


class Renderer:
    def __init__(self, animate: bool = True):
        self.animate = animate
        self.ar = ArMatRenderer(animate=animate)
        self._colors: dict[str, tuple] = {}
        self._last_t: float | None = None
        self._dt = 0.0

    def close(self) -> None:
        self.ar.close()

    # ------------------------------------------------------------- animation
    def _smooth(self, name: str, target) -> tuple:
        """Colour that eases towards `target` over ~theme.FADE_S instead of jumping."""
        target = np.array(target, dtype=np.float64)
        cur = self._colors.get(name)
        if cur is None or not self.animate:
            cur = target
        else:
            cur = cur + (target - cur) * min(1.0, self._dt / T.FADE_S * 2.2)
            if np.abs(target - cur).max() < 0.75:      # close enough: land exactly
                cur = target
        self._colors[name] = cur
        return tuple(int(round(v)) for v in cur)

    # ---------------------------------------------------------------- render
    def render(self, img: np.ndarray, vm: ViewModel, now: float | None = None) -> np.ndarray:
        """Draw vm over img (in place) and return it. now: the animation clock (perf_counter s)."""
        anim_now = now
        now = time.perf_counter() if now is None else now
        self._dt = 0.0 if self._last_t is None else min(0.2, now - self._last_t)
        self._last_t = now
        h, w = img.shape[:2]
        k = h / 720.0
        loupe = None
        if vm.calib is not None and vm.calib.mouse_px is not None:
            loupe = self._loupe_crop(img, vm.calib.mouse_px, k)   # from the clean frame, before overlays
        if vm.ar is not None:
            a = vm.ar
            self.ar.draw(img, a.quad, a.size_cm, a.state, a.progress, a.hands_px, shape=a.shape, now=anim_now)
        self._zone(img, vm, k)
        if vm.setup is not None:
            self._setup_marks(img, vm.setup, k, on_ar=vm.ar is not None)
        if vm.calib is not None:
            self._calib_marks(img, vm.calib, loupe, k)
        if vm.show_debug:
            self._debug_marks(img, vm, k)

        y = self._scramble(img, vm, k)
        side = self._stats(img, vm, k)
        right_y = self._hint(img, vm, k)
        if vm.setup is not None:
            y = self._instructions(img, vm.setup.title, vm.setup.lines, vm.setup.warnings, vm.setup.progress, y, side, k)
        elif vm.calib is not None:
            y = self._instructions(img, vm.calib.title, vm.calib.lines, vm.calib.warnings, vm.calib.progress, y, side, k)
        else:
            self._timer(img, vm, k)
        if vm.message:
            y = self._toast(img, vm.message, vm.message_alpha, y, side, k)
        right_y = self._warnings(img, vm.warnings, max(right_y, y), k)
        if vm.show_debug:
            self._debug_panel(img, vm, right_y, k)
        if vm.flash is not None and vm.flash_level > 0:
            self._flash(img, vm.flash, vm.flash_level, k)
        if vm.show_help:
            self._help(img, vm, k)
        return img

    # ------------------------------------------------------------- top / side
    def _side_width(self, k: float) -> int:
        return int(196 * k)

    def _scramble(self, img, vm: ViewModel, k: float) -> int:
        """Returns the y just below the panel."""
        h, w = img.shape[:2]
        m = int(T.MARGIN * k)
        if not vm.scramble:
            return m
        px = T.SIZE_SCRAMBLE * k
        pad_x, pad_y = int(T.PAD * 1.4 * k), int(T.PAD * 0.6 * k)
        max_w = w - 2 * (m + self._side_width(k) + int(T.GAP * k)) - 2 * pad_x
        lines = wrap(vm.scramble, max_w, int(px), "semibold")
        lh = line_height("semibold", int(px))
        tw = max(text_width(s, "semibold", int(px)) for s in lines)
        pw, ph = tw + 2 * pad_x, lh * len(lines) + 2 * pad_y
        x0 = (w - pw) // 2
        panel(img, x0, m, pw, ph, int(T.RADIUS * k), T.PANEL, T.PANEL_ALPHA, T.PANEL_EDGE, T.EDGE_ALPHA)
        for i, s in enumerate(lines):
            draw_text(img, s, w / 2, m + pad_y + i * lh, px, T.TEXT, "semibold", "mt")
        return m + ph + int(T.GAP * k)

    def _stats(self, img, vm: ViewModel, k: float) -> int:
        """Compact session panel in the left column. Returns the column's right edge."""
        m = int(T.MARGIN * k)
        sw = self._side_width(k)
        if vm.stats is None:
            return m
        st = vm.stats
        rows = [("last", format_time(st.current)), ("best", format_time(st.best)), ("mo3", format_time(st.mo3)),
                ("ao5", format_time(st.ao5)), ("ao12", format_time(st.ao12)), ("mean", format_time(st.mean))]
        recent = vm.recent[-5:][::-1]
        pad = int(T.PAD * k)
        px, small = T.SIZE_BODY * k, T.SIZE_SMALL * k
        lh = int(line_height("regular", int(px)) * 1.02)
        slh = int(line_height("regular", int(small)) * 1.05)
        title_h = int(line_height("semibold", int(small)) * 1.25)
        height = pad + title_h + lh * len(rows) + (int(8 * k) + slh * len(recent) if recent else 0) + pad
        panel(img, m, m, sw, height, int(T.RADIUS * k), T.PANEL, T.PANEL_ALPHA, T.PANEL_EDGE, T.EDGE_ALPHA)
        x0, x1 = m + pad, m + sw - pad
        y = m + pad
        draw_text(img, "SESSION", x0, y, small, T.TEXT_MUTED, "semibold")
        draw_text(img, f"{st.count - st.dnf_count}/{st.count}", x1, y, small, T.TEXT_MUTED, "regular", "rt")
        y += title_h
        for name, val in rows:
            draw_text(img, name, x0, y, px, T.TEXT_MUTED)
            draw_text(img, val, x1, y, px, T.TEXT if val != "-" else T.TEXT_FAINT, "semibold", "rt")
            y += lh
        if recent:
            y += int(4 * k)
            cv2.line(img, (x0, y), (x1, y), T.TEXT_FAINT, 1, cv2.LINE_AA)
            y += int(4 * k)
            for j, s in enumerate(recent):
                draw_text(img, f"{st.count - j}.", x0, y, small, T.TEXT_FAINT)
                draw_text(img, s, x1, y, small, T.TEXT_MUTED, "regular", "rt")
                y += slh
        return m + sw

    def _hint(self, img, vm: ViewModel, k: float) -> int:
        """Small '? help' pill at the top right. Returns the y below it."""
        h, w = img.shape[:2]
        m = int(T.MARGIN * k)
        px = T.SIZE_SMALL * k
        label = "?  help" if not vm.show_help else "?  close help"
        lh = line_height("regular", int(px))
        pad = int(10 * k)
        pw = text_width(label, "regular", int(px)) + 2 * pad
        ph = lh + int(8 * k)
        panel(img, w - m - pw, m, pw, ph, ph // 2, T.PANEL, T.PANEL_ALPHA * 0.8, T.PANEL_EDGE, T.EDGE_ALPHA)
        draw_text(img, label, w - m - pw / 2, m + ph / 2, px, T.TEXT_MUTED, "regular", "mm")
        return m + ph + int(T.GAP * k)

    def _warnings(self, img, warnings: list[str], y: int, k: float) -> int:
        if not warnings:
            return y
        h, w = img.shape[:2]
        m = int(T.MARGIN * k)
        px = T.SIZE_SMALL * k
        pad = int(10 * k)
        max_w = int(380 * k)
        lh = line_height("regular", int(px))
        for msg in warnings:
            lines = wrap(msg, max_w - 2 * pad - int(14 * k), int(px))
            tw = max(text_width(s, "regular", int(px)) for s in lines)
            pw, ph = tw + 2 * pad + int(14 * k), lh * len(lines) + int(10 * k)
            x0 = w - m - pw
            panel(img, x0, y, pw, ph, int(10 * k), T.PANEL, 0.78, T.WARN, 0.55)
            cv2.circle(img, (x0 + pad + int(3 * k), y + ph // 2), max(2, int(4 * k)), T.WARN, -1, cv2.LINE_AA)
            for i, s in enumerate(lines):
                draw_text(img, s, x0 + pad + int(14 * k), y + int(5 * k) + i * lh, px, T.TEXT)
            y += ph + int(6 * k)
        return y + int(T.GAP * k)

    # ----------------------------------------------------------------- timer
    def _timer(self, img, vm: ViewModel, k: float) -> None:
        h, w = img.shape[:2]
        color = self._smooth("timer", vm.time_color)
        px = T.SIZE_TIMER * k
        cy = int(h * 0.40)
        box = draw_text(img, vm.time_text, w / 2, cy, px, color, "digits", "mm", shadow=0.55)
        if vm.sub_text:
            spx = T.SIZE_SUB * k
            lh = line_height("regular", int(spx))
            pad = int(12 * k)
            dot = int(16 * k)
            tw = text_width(vm.sub_text, "regular", int(spx))
            pw, ph = tw + 2 * pad + dot, lh + int(8 * k)
            x0, y0 = int(w / 2 - pw / 2), box[3] - int(4 * k)
            panel(img, x0, y0, pw, ph, ph // 2, T.PANEL, T.PANEL_ALPHA, T.PANEL_EDGE, T.EDGE_ALPHA)
            cv2.circle(img, (x0 + pad + int(4 * k), y0 + ph // 2), max(2, int(5 * k)), color, -1, cv2.LINE_AA)
            draw_text(img, vm.sub_text, x0 + pad + dot, y0 + ph / 2, spx, T.TEXT, "regular", "lm")

    # ------------------------------------------------------------------ zone
    def _zone(self, img, vm: ViewModel, k: float) -> None:
        if vm.zone is None or vm.zone_mode == "off":
            return
        h, w = img.shape[:2]
        pts = vm.zone.to_pixels(w, h)
        if vm.zone_mode == "setup":  # current zone, faint, under the setup overlay
            if vm.ar is None:
                cv2.polylines(img, [pts], True, T.TEXT_FAINT, 1, cv2.LINE_AA)
            return
        if vm.ar is not None:        # the AR mat shows the state itself
            return
        target = T.STATE.get(vm.zone_mode, T.STATE["idle"])
        if vm.zone_warning and vm.zone_mode == "idle":
            target = T.WARN
        color = self._smooth("zone", target)
        poly = pts[:, 0, :].astype(np.float64)
        fill = {"holding": 0.10 + 0.12 * vm.hold_progress, "ready": 0.22}.get(vm.zone_mode, 0.0)
        if fill:
            fill_poly(img, poly, color, fill)
        cv2.polylines(img, [np.int32(np.round(poly * 16))], True, color, max(2, int(2.5 * k)), cv2.LINE_AA, shift=4)
        # label pill just inside the top edge, centred, with the hold progress in it
        label = {"idle": "Rest both hands here", "holding": "Hold still",
                 "ready": "Ready - lift to start", "solving": "Hands back here to stop"}[vm.zone_mode]
        px = T.SIZE_SMALL * k
        lh = line_height("semibold", int(px))
        pad = int(10 * k)
        pw, ph = text_width(label, "semibold", int(px)) + 2 * pad, lh + int(10 * k)
        x0 = int(min(max(poly[:, 0].mean() - pw / 2, 0), w - pw))
        y0 = int(min(max(poly[:, 1].min() + 8 * k, 0), h - ph))
        panel(img, x0, y0, pw, ph, ph // 2, T.PANEL, T.PANEL_ALPHA, color, 0.6)
        if vm.zone_mode == "holding" and vm.hold_progress > 0:
            bw = int((pw - 2 * pad) * min(vm.hold_progress, 1.0))
            if bw > 0:
                panel(img, x0 + pad, y0 + ph - int(5 * k), bw, max(2, int(3 * k)), int(2 * k), color, 0.95)
        draw_text(img, label, x0 + pw / 2, y0 + ph / 2 - int(1 * k), px, T.TEXT, "semibold", "mm")

    # ---------------------------------------------------------------- setup
    def _setup_marks(self, img, sv: SetupView, k: float, on_ar: bool = False) -> None:
        h, w = img.shape[:2]
        thick = max(2, int(2.5 * k))
        if sv.candidate is not None:
            poly = np.array(sv.candidate.points) * (w, h)
            color = T.WARN if sv.warnings else T.ACCENT
            if not on_ar:                    # the AR mat preview already shows the area
                fill_poly(img, poly, color, 0.16)
            if not (on_ar and len(poly) > 8):   # a drawn area on the table: its hologram is the outline
                cv2.polylines(img, [np.int32(np.round(poly * 16))], True, color, thick, cv2.LINE_AA, shift=4)
            if len(poly) <= 8:               # corner handles (not on a drawn outline's many points)
                for p in poly:
                    cv2.circle(img, (int(p[0]), int(p[1])), max(4, int(5 * k)), color, -1, cv2.LINE_AA)
        if len(sv.stroke) > 1:
            self._stroke(img, np.array(sv.stroke) * (w, h), k)
        if sv.drag is not None:
            (ax, ay), (bx, by) = sv.drag
            r = np.array([(ax, ay), (bx, ay), (bx, by), (ax, by)]) * (w, h)
            fill_poly(img, r, T.ACCENT, 0.12)
            cv2.polylines(img, [np.int32(np.round(r))], True, T.ACCENT, thick, cv2.LINE_AA)
        for x, y in sv.markers:
            cv2.drawMarker(img, (int(x * w), int(y * h)), T.STATE["ready"], cv2.MARKER_CROSS, int(26 * k), thick)
        for x, y in sv.palms:
            cv2.circle(img, (int(x * w), int(y * h)), int(10 * k), T.STATE["holding"], -1, cv2.LINE_AA)
            cv2.circle(img, (int(x * w), int(y * h)), int(10 * k), T.PANEL, max(1, int(2 * k)), cv2.LINE_AA)

    def _stroke(self, img, pts: np.ndarray, k: float) -> None:
        """The path being traced in draw mode: a soft glowing line, the start point, and a faint closing line."""
        H, W = img.shape[:2]
        glow_w = max(4, int(9 * k))
        x0, y0 = (np.floor(pts.min(axis=0)) - glow_w).astype(int).clip(0, (W, H))
        x1, y1 = (np.ceil(pts.max(axis=0)) + glow_w + 1).astype(int).clip(0, (W, H))
        if x1 > x0 and y1 > y0:                       # the soft glow, only in the stroke's box
            sub = img[y0:y1, x0:x1]
            layer = sub.copy()
            cv2.polylines(layer, [np.int32(np.round((pts - (x0, y0)) * 16))], False, T.ACCENT, glow_w, cv2.LINE_AA,
                          shift=4)
            cv2.addWeighted(layer, 0.25, sub, 0.75, 0, dst=sub)
        p16 = np.int32(np.round(pts * 16))
        cv2.polylines(img, [p16], False, T.ACCENT, max(2, int(2.5 * k)), cv2.LINE_AA, shift=4)
        a, b = pts[0], pts[-1]
        n = max(2, int(np.hypot(*(b - a)) / (12 * k)))
        for i in range(0, n, 2):                      # dashed: where the shape will close
            u0, u1 = i / n, min(1.0, (i + 1) / n)
            cv2.line(img, tuple(np.int32(a + (b - a) * u0)), tuple(np.int32(a + (b - a) * u1)), T.TEXT_MUTED,
                     max(1, int(1.5 * k)), cv2.LINE_AA)
        cv2.circle(img, tuple(np.int32(np.round(a))), max(4, int(6 * k)), T.ACCENT, -1, cv2.LINE_AA)
        cv2.circle(img, tuple(np.int32(np.round(a))), max(4, int(6 * k)), T.PANEL, max(1, int(1.5 * k)), cv2.LINE_AA)

    def _instructions(self, img, title: str, lines_in: list[str], warnings: list[str], progress: float,
                      y: int, side: int, k: float) -> int:
        """Instruction card in the top stack. Returns the y below it."""
        h, w = img.shape[:2]
        m = int(T.MARGIN * k)
        x0 = side + int(T.GAP * k)
        x1 = w - m - self._side_width(k) // 2
        pad = int(T.PAD * 1.2 * k)
        px, tpx = T.SIZE_BODY * k, T.SIZE_TITLE * k
        max_w = x1 - x0 - 2 * pad
        lines = []
        for s in lines_in:
            if "   " in s:     # a key-hint line ("ENTER accept   R redo   ESC cancel"): muted, dot-separated
                hint = "  ·  ".join(part.strip() for part in s.split("   ") if part.strip())
                lines += [("hint", ln) for ln in wrap(hint, max_w, int(px))]
            else:
                lines += [("text", ln) for ln in wrap(s, max_w, int(px))]
        warn = [ln for s in warnings for ln in wrap(s, max_w - int(16 * k), int(px))]
        lh = int(line_height("regular", int(px)) * 1.02)
        th = int(line_height("semibold", int(tpx)) * 1.15)
        bar = int(18 * k) if progress > 0 else 0
        ph = pad + th + lh * (len(lines) + len(warn)) + bar + pad
        panel(img, x0, y, x1 - x0, ph, int(T.RADIUS * k), T.PANEL, 0.8, T.PANEL_EDGE, T.EDGE_ALPHA)
        cy = y + pad
        draw_text(img, title, x0 + pad, cy, tpx, T.ACCENT, "semibold")
        cy += th
        for kind, ln in lines:
            draw_text(img, ln, x0 + pad, cy, px, T.TEXT_MUTED if kind == "hint" else T.TEXT)
            cy += lh
        for ln in warn:
            cv2.circle(img, (x0 + pad + int(4 * k), cy + lh // 2), max(2, int(4 * k)), T.WARN, -1, cv2.LINE_AA)
            draw_text(img, ln, x0 + pad + int(16 * k), cy, px, T.WARN)
            cy += lh
        if progress > 0:
            bw = x1 - x0 - 2 * pad
            bh = max(4, int(6 * k))
            panel(img, x0 + pad, cy + int(6 * k), bw, bh, bh // 2, T.TEXT_FAINT, 0.5)
            fw = int(bw * min(progress, 1.0))
            if fw > 1:
                panel(img, x0 + pad, cy + int(6 * k), fw, bh, bh // 2, T.STATE["holding"], 1.0)
        return y + ph + int(T.GAP * k)

    def _toast(self, img, msg: str, alpha: float, y: int, side: int, k: float) -> int:
        h, w = img.shape[:2]
        px = T.SIZE_BODY * k
        pad = int(T.PAD * k)
        max_w = min(int(760 * k), w - 2 * (side + int(T.GAP * k)))
        lines = wrap(msg, max_w - 2 * pad, int(px))
        lh = int(line_height("regular", int(px)) * 1.02)
        tw = max(text_width(s, "regular", int(px)) for s in lines)
        pw, ph = tw + 2 * pad, lh * len(lines) + int(T.PAD * 1.2 * k)
        x0 = (w - pw) // 2
        a = max(0.0, min(1.0, alpha))
        panel(img, x0, y, pw, ph, int(T.RADIUS * k), T.PANEL, 0.82 * a, T.ACCENT, 0.45 * a)
        for i, s in enumerate(lines):
            draw_text(img, s, w / 2, y + int(T.PAD * 0.6 * k) + i * lh, px, T.TEXT, "regular", "mt", opacity=a)
        return y + ph + int(T.GAP * k)

    # ------------------------------------------------------- table calibration
    LOUPE_SRC = 60      # half-size of the magnified area (px, at 720p)
    LOUPE_ZOOM = 3

    def _loupe_crop(self, img, mouse, k):
        r = int(self.LOUPE_SRC * k)
        x, y = int(round(mouse[0])), int(round(mouse[1]))
        pad = cv2.copyMakeBorder(img, r, r, r, r, cv2.BORDER_CONSTANT, value=(40, 40, 40))
        return pad[y:y + 2 * r, x:x + 2 * r].copy(), (x - r, y - r)

    def _calib_marks(self, img, cv_: CalibView, loupe, k: float) -> None:
        h, w = img.shape[:2]
        if cv_.grid_px:
            layer = img.copy()
            for line in cv_.grid_px:
                cv2.polylines(layer, [np.int32(np.round(line * 16))], False, T.ACCENT, 1, cv2.LINE_AA, shift=4)
            cv2.addWeighted(layer, 0.75, img, 0.25, 0, dst=img)
        if cv_.marker_px is not None:
            mp = np.array(cv_.marker_px, dtype=np.float64)
            fill_poly(img, mp, T.STATE["ready"], 0.25)
            cv2.polylines(img, [np.int32(np.round(mp * 16))], True, T.STATE["ready"], max(2, int(2.5 * k)),
                          cv2.LINE_AA, shift=4)
        clicks = [tuple(map(int, map(round, c))) for c in cv_.clicks_px]
        if len(clicks) > 1:
            cv2.polylines(img, [np.int32(clicks)], len(clicks) == 4, T.STATE["holding"], max(1, int(2 * k)), cv2.LINE_AA)
        for i, c in enumerate(clicks):
            r = int(11 * k)
            cv2.circle(img, c, r, T.STATE["holding"], -1, cv2.LINE_AA)
            draw_text(img, str(i + 1), c[0], c[1], T.SIZE_SMALL * k, T.PANEL, "semibold", "mm")
        if loupe is not None:
            crop, (ox, oy) = loupe
            z = self.LOUPE_ZOOM
            big = cv2.resize(crop, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
            bh, bw = big.shape[:2]
            for c in clicks:   # earlier clicks, inside the loupe
                q = ((c[0] - ox) * z, (c[1] - oy) * z)
                if 0 <= q[0] < bw and 0 <= q[1] < bh:
                    cv2.circle(big, (int(q[0]), int(q[1])), 6, T.STATE["holding"], 2, cv2.LINE_AA)
            cx, cy = bw // 2, bh // 2
            cv2.line(big, (cx - 16, cy), (cx - 4, cy), T.DANGER, 2)
            cv2.line(big, (cx + 4, cy), (cx + 16, cy), T.DANGER, 2)
            cv2.line(big, (cx, cy - 16), (cx, cy - 4), T.DANGER, 2)
            cv2.line(big, (cx, cy + 4), (cx, cy + 16), T.DANGER, 2)
            mx = cv_.mouse_px[0]
            m = int(T.MARGIN * k)
            lx = w - bw - m if mx < w * 0.6 else m
            ly = h - bh - m
            # rounded loupe: blend the magnified picture through a rounded mask, then a border
            rmask = rounded_mask(bw, bh, int(T.RADIUS * k))
            roi = img[ly:ly + bh, lx:lx + bw]
            roi[:] = cv2.blendLinear(big, roi, rmask, 1.0 - rmask)
            blend_mask(img, rounded_ring(bw, bh, int(T.RADIUS * k), 2), lx, ly, T.TEXT, 0.9)
            draw_text(img, f"{z}x", lx + int(10 * k), ly + int(8 * k), T.SIZE_SMALL * k, T.TEXT, "semibold", shadow=0.8)

    # ----------------------------------------------------------------- debug
    def _debug_marks(self, img, vm: ViewModel, k: float) -> None:
        h, w = img.shape[:2]
        d = vm.debug
        for lm in d.get("hands", []):
            pts = [(int(x * w), int(y * h)) for x, y in lm]
            for a, b in HAND_CONNECTIONS:
                cv2.line(img, pts[a], pts[b], T.ACCENT, max(1, int(2 * k)), cv2.LINE_AA)
            for p in pts:
                cv2.circle(img, p, max(2, int(3 * k)), T.TEXT, -1, cv2.LINE_AA)
        for (px, py), inside in zip(d.get("palms", []), d.get("palms_in_zone", [])):
            c = (int(px * w), int(py * h))
            cv2.circle(img, c, int(11 * k), T.STATE["ready"] if inside else T.DANGER, -1, cv2.LINE_AA)
            cv2.circle(img, c, int(11 * k), T.PANEL, max(1, int(2 * k)), cv2.LINE_AA)

    def _debug_panel(self, img, vm: ViewModel, y: int, k: float) -> None:
        h, w = img.shape[:2]
        m = int(T.MARGIN * k)
        rows = [(str(key), str(val)) for key, val in vm.debug.items() if key not in ("hands", "palms", "palms_in_zone")]
        if not rows:
            return
        px = T.SIZE_SMALL * k
        lh = line_height("regular", int(px))
        pad = int(12 * k)
        pw = int(300 * k)
        ph = pad * 2 + lh * len(rows)
        x0 = w - m - pw
        panel(img, x0, y, pw, ph, int(T.RADIUS * k), T.PANEL, 0.78, T.PANEL_EDGE, T.EDGE_ALPHA)
        for i, (key, val) in enumerate(rows):
            draw_text(img, key, x0 + pad, y + pad + i * lh, px, T.TEXT_MUTED)
            draw_text(img, val[:28], x0 + pw - pad, y + pad + i * lh, px, T.TEXT, "regular", "rt")

    # ------------------------------------------------------------ flash, help
    def _flash(self, img, color, level: float, k: float) -> None:
        """Inspection warning: a coloured edge glow that fades out (no hard on/off flash)."""
        h, w = img.shape[:2]
        t = max(2, int(10 * k))
        ramp = np.linspace(1.0, 0.0, t, dtype=np.float32)
        a = level * 0.9
        top = np.repeat(ramp[:, None], w, axis=1)
        left = np.repeat(ramp[None, :], h, axis=0)
        blend_mask(img, top, 0, 0, color, a)
        blend_mask(img, top[::-1], 0, h - t, color, a)
        blend_mask(img, left, 0, 0, color, a)
        blend_mask(img, left[:, ::-1], w - t, 0, color, a)

    def _help(self, img, vm: ViewModel, k: float) -> None:
        h, w = img.shape[:2]
        px, small, tpx = T.SIZE_BODY * k, T.SIZE_SMALL * k, T.SIZE_TITLE * k
        pad = int(T.PAD * 1.6 * k)
        lh = int(line_height("regular", int(px)) * 1.15)
        slh = line_height("regular", int(small))
        th = int(line_height("semibold", int(tpx)) * 1.4)
        pw = int(720 * k)
        cols = 2
        rows = (len(HELP_KEYS) + 1) // cols
        status = [ln for s in vm.status_lines for ln in wrap(s, pw - 2 * pad, int(small))]
        ph = pad + th + rows * lh + int(12 * k) + slh * len(status) + pad
        x0, y0 = (w - pw) // 2, max(int(T.MARGIN * k), (h - ph) // 2 - int(40 * k))
        panel(img, x0, y0, pw, ph, int(T.RADIUS * 1.3 * k), T.PANEL, 0.9, T.PANEL_EDGE, 0.16)
        draw_text(img, "Keys", x0 + pad, y0 + pad, tpx, T.TEXT, "semibold")
        draw_text(img, "? or F1 to close", x0 + pw - pad, y0 + pad + int(3 * k), small, T.TEXT_MUTED, "regular", "rt")
        col_w = (pw - 2 * pad) // cols
        key_w = int(96 * k)
        for i, (key, desc) in enumerate(HELP_KEYS):
            if not key:
                continue
            c, r = i % cols, i // cols
            cx, cy = x0 + pad + c * col_w, y0 + pad + th + r * lh
            kw = text_width(key, "semibold", int(small)) + int(14 * k)
            kh = line_height("semibold", int(small)) + int(4 * k)
            panel(img, cx, cy + (lh - kh) // 2 - int(2 * k), kw, kh, int(6 * k), T.TEXT, 0.12, T.TEXT, 0.18)
            draw_text(img, key, cx + kw / 2, cy + lh / 2 - int(2 * k), small, T.TEXT, "semibold", "mm")
            draw_text(img, desc, cx + key_w + int(8 * k), cy + lh / 2 - int(2 * k), px, T.TEXT_MUTED, "regular", "lm")
        y = y0 + pad + th + rows * lh + int(12 * k)
        for s in status:
            draw_text(img, s, x0 + pad, y, small, T.TEXT_FAINT)
            y += slh
