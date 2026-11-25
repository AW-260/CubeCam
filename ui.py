"""OpenCV rendering. Pure drawing: main.py builds a ViewModel, ui draws it."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from hands import HAND_CONNECTIONS
from models import format_time
from stats import SessionStats
from timer_state import State
from zone import Zone
from zone_setup import SetupView

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

# BGR colours
WHITE = (255, 255, 255)
GREY = (170, 170, 170)
DARK = (25, 25, 25)
GREEN = (80, 220, 80)
YELLOW = (40, 210, 255)
ORANGE = (0, 150, 255)
RED = (60, 60, 240)
CYAN = (230, 200, 60)

HELP = "SPACE timer  2 +2  D DNF  BKSP delete  I inspection  N scramble  M zone setup  H debug  ESC abort  Q quit"


@dataclass
class ViewModel:
    state: State
    time_text: str
    time_color: tuple = WHITE
    sub_text: str = ""                       # e.g. "+2", "inspection 9.1s"
    scramble: str = ""
    stats: SessionStats | None = None
    recent: list[str] = field(default_factory=list)
    zone: Zone | None = None
    zone_mode: str = "idle"                  # idle | holding | ready | solving | setup | off
    hold_progress: float = 0.0
    status_lines: list[str] = field(default_factory=list)
    message: str | None = None
    flash: tuple | None = None               # border colour for inspection warnings
    show_debug: bool = False
    debug: dict[str, Any] = field(default_factory=dict)
    setup: SetupView | None = None           # zone setup overlay (key M)
    zone_warning: bool = False               # zone mostly off-screen / too small


def _text(img, s, org, scale, color=WHITE, thick=1, font=FONT, shadow=True):
    if shadow:
        cv2.putText(img, s, (org[0] + 2, org[1] + 2), font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, font, scale, color, thick, cv2.LINE_AA)


def _panel(img, x1, y1, x2, y2, alpha=0.55, color=DARK):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    overlay = np.empty_like(roi)
    overlay[:] = color
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)


def _fill_poly(img, pts: np.ndarray, color, alpha: float) -> None:
    """Translucent polygon fill (only the polygon's bounding box is blended)."""
    h, w = img.shape[:2]
    x1, y1 = max(0, int(pts[:, 0, 0].min())), max(0, int(pts[:, 0, 1].min()))
    x2, y2 = min(w, int(pts[:, 0, 0].max()) + 1), min(h, int(pts[:, 0, 1].max()) + 1)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    mask = np.zeros(roi.shape[:2], np.uint8)
    cv2.fillPoly(mask, [pts - np.array([x1, y1], np.int32)], 255)
    overlay = roi.copy()
    overlay[mask > 0] = color
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)


def _wrap(s: str, max_w: int, scale: float, thick: int) -> list[str]:
    lines, cur = [], ""
    for word in s.split():
        trial = f"{cur} {word}".strip()
        if cv2.getTextSize(trial, FONT, scale, thick)[0][0] > max_w and cur:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


class Renderer:
    def render(self, img: np.ndarray, vm: ViewModel) -> np.ndarray:
        h, w = img.shape[:2]
        k = h / 720.0  # scale factor relative to a 720p layout
        self._zone(img, vm, k)
        if vm.setup is not None:
            self._setup(img, vm.setup, k)
        if vm.show_debug:
            self._debug(img, vm, k)
        self._scramble(img, vm, k)
        self._timer(img, vm, k)
        self._stats(img, vm, k)
        self._status(img, vm, k)
        if vm.message:
            self._message(img, vm.message, k)
        if vm.flash is not None:
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), vm.flash, max(6, int(14 * k)))
        return img

    # ------------------------------------------------------------------ parts
    def _zone(self, img, vm: ViewModel, k: float) -> None:
        if vm.zone is None or vm.zone_mode == "off":
            return
        h, w = img.shape[:2]
        pts = vm.zone.to_pixels(w, h)
        thick = max(2, int(3 * k))
        if vm.zone_mode == "setup":  # current zone, faint, under the setup overlay
            cv2.polylines(img, [pts], True, GREY, 1, cv2.LINE_AA)
            return
        color = {"idle": GREY, "holding": YELLOW, "ready": GREEN, "solving": RED}.get(vm.zone_mode, GREY)
        if vm.zone_warning and vm.zone_mode == "idle":
            color = ORANGE
        if vm.zone_mode in ("ready", "holding"):
            _fill_poly(img, pts, color, 0.25)
        cv2.polylines(img, [pts], True, color, thick, cv2.LINE_AA)
        x1, y1 = int(pts[:, 0, 0].min()), int(pts[:, 0, 1].min())
        x2, y2 = int(pts[:, 0, 0].max()), int(pts[:, 0, 1].max())
        # Keep the label and the hold bar on screen even if the zone pokes out of the frame.
        lx1, lx2 = max(0, x1), min(w, x2)
        ly1, ly2 = max(0, y1), min(h - 1, y2)
        if vm.zone_mode == "holding" and vm.hold_progress > 0:
            px = lx1 + int((lx2 - lx1) * vm.hold_progress)
            cv2.rectangle(img, (lx1, ly2 - int(10 * k)), (px, ly2), color, -1)
        label = {"idle": "START ZONE - rest both hands here", "holding": "hold...",
                 "ready": "READY - lift to start", "solving": "return hands to stop"}[vm.zone_mode]
        scale, lthick = 0.7 * k, max(1, int(2 * k))
        if vm.zone.is_rect:
            lx = lx1 + int(10 * k)
        else:  # slanted sides: centre the label so it stays inside the shape
            tw = cv2.getTextSize(label, FONT, scale, lthick)[0][0]
            lx = min(max(0, (lx1 + lx2 - tw) // 2), max(0, w - tw))
        _text(img, label, (lx, ly1 + int(28 * k)), scale, color, lthick)

    def _setup(self, img, sv: SetupView, k: float) -> None:
        h, w = img.shape[:2]
        thick = max(2, int(3 * k))
        if sv.candidate is not None:
            pts = sv.candidate.to_pixels(w, h)
            color = ORANGE if sv.warnings else CYAN
            _fill_poly(img, pts, color, 0.2)
            cv2.polylines(img, [pts], True, color, thick, cv2.LINE_AA)
            for p in pts[:, 0]:
                cv2.circle(img, (int(p[0]), int(p[1])), max(4, int(6 * k)), color, -1, cv2.LINE_AA)
        if sv.drag is not None:
            (ax, ay), (bx, by) = sv.drag
            cv2.rectangle(img, (int(ax * w), int(ay * h)), (int(bx * w), int(by * h)), YELLOW, thick)
        for x, y in sv.markers:
            c = (int(x * w), int(y * h))
            cv2.drawMarker(img, c, GREEN, cv2.MARKER_CROSS, int(28 * k), thick)
        for x, y in sv.palms:
            cv2.circle(img, (int(x * w), int(y * h)), int(12 * k), YELLOW, -1, cv2.LINE_AA)
        # Instruction panel under the scramble line.
        scale, lh = 0.62 * k, int(26 * k)
        lines = [ln for s in sv.lines for ln in _wrap(s, w - int(300 * k), scale, 1)]
        warn = [ln for s in sv.warnings for ln in _wrap("WARNING: " + s, w - int(300 * k), scale, 1)]
        x0, y0 = int(240 * k), int(92 * k)
        bottom = y0 + int(34 * k) + lh * (len(lines) + len(warn)) + (int(18 * k) if sv.progress > 0 else 0)
        _panel(img, x0 - int(14 * k), y0 - int(8 * k), w - int(30 * k), bottom, alpha=0.75)
        _text(img, sv.title, (x0, y0 + int(22 * k)), 0.8 * k, CYAN, max(1, int(2 * k)), shadow=False)
        y = y0 + int(22 * k) + lh
        for ln in lines:
            _text(img, ln, (x0, y), scale, WHITE, 1, shadow=False)
            y += lh
        for ln in warn:
            _text(img, ln, (x0, y), scale, ORANGE, max(1, int(2 * k)), shadow=False)
            y += lh
        if sv.progress > 0:
            bar_w = w - int(30 * k) - x0 - int(14 * k)
            cv2.rectangle(img, (x0, y - int(12 * k)), (x0 + bar_w, y), GREY, 1)
            cv2.rectangle(img, (x0, y - int(12 * k)), (x0 + int(bar_w * sv.progress), y), YELLOW, -1)

    def _scramble(self, img, vm: ViewModel, k: float) -> None:
        if not vm.scramble:
            return
        h, w = img.shape[:2]
        scale, thick = 0.85 * k, max(1, int(2 * k))
        lines = _wrap(vm.scramble, w - int(40 * k), scale, thick)
        line_h = int(36 * k)
        _panel(img, 0, 0, w, int(14 * k) + line_h * len(lines), alpha=0.6)
        for i, line in enumerate(lines):
            tw = cv2.getTextSize(line, FONT, scale, thick)[0][0]
            _text(img, line, ((w - tw) // 2, int(36 * k) + i * line_h), scale, WHITE, thick)

    def _timer(self, img, vm: ViewModel, k: float) -> None:
        h, w = img.shape[:2]
        scale, thick = 3.6 * k, max(2, int(6 * k))
        (tw, th), _ = cv2.getTextSize(vm.time_text, FONT_BOLD, scale, thick)
        y = int(h * 0.45)
        _text(img, vm.time_text, ((w - tw) // 2, y), scale, vm.time_color, thick, FONT_BOLD)
        if vm.sub_text:
            s2, t2 = 1.0 * k, max(1, int(2 * k))
            sw = cv2.getTextSize(vm.sub_text, FONT, s2, t2)[0][0]
            _text(img, vm.sub_text, ((w - sw) // 2, y + int(50 * k)), s2, vm.time_color, t2)

    def _stats(self, img, vm: ViewModel, k: float) -> None:
        if vm.stats is None:
            return
        st = vm.stats
        rows = [
            ("current", format_time(st.current)),
            ("best", format_time(st.best)),
            ("mo3", format_time(st.mo3)),
            ("ao5", format_time(st.ao5)),
            ("ao12", format_time(st.ao12)),
            ("mean", format_time(st.mean)),
            ("solves", f"{st.count - st.dnf_count}/{st.count}"),
        ]
        scale, thick, lh = 0.6 * k, 1, int(26 * k)
        x0, y0 = int(12 * k), int(130 * k)
        n_recent = min(len(vm.recent), 5)
        height = lh * (len(rows) + n_recent) + (int(8 * k) if n_recent else 0) + int(16 * k)
        _panel(img, 0, y0 - int(24 * k), int(215 * k), y0 + height - int(16 * k), alpha=0.55)
        for i, (name, val) in enumerate(rows):
            y = y0 + i * lh
            _text(img, name, (x0, y), scale, GREY, thick, shadow=False)
            _text(img, val, (x0 + int(90 * k), y), scale, WHITE, thick, shadow=False)
        y = y0 + len(rows) * lh + int(8 * k)
        for j, s in enumerate(vm.recent[-5:][::-1]):
            _text(img, f"{vm.stats.count - j}. {s}", (x0, y + j * lh), 0.55 * k, CYAN, thick, shadow=False)

    def _status(self, img, vm: ViewModel, k: float) -> None:
        h, w = img.shape[:2]
        scale, lh = 0.5 * k, int(22 * k)
        lines = vm.status_lines + [HELP]
        _panel(img, 0, h - lh * len(lines) - int(10 * k), w, h, alpha=0.6)
        for i, line in enumerate(lines):
            _text(img, line, (int(10 * k), h - int(10 * k) - (len(lines) - 1 - i) * lh), scale,
                  GREY if line is HELP else WHITE, 1, shadow=False)

    def _message(self, img, msg: str, k: float) -> None:
        h, w = img.shape[:2]
        scale, thick = 0.7 * k, max(1, int(2 * k))
        lines = _wrap(msg, w - int(80 * k), scale, thick)
        lh = int(30 * k)
        y0 = int(h * 0.58)
        _panel(img, int(30 * k), y0 - lh, w - int(30 * k), y0 + lh * (len(lines) - 1) + int(14 * k), alpha=0.75)
        for i, line in enumerate(lines):
            _text(img, line, (int(45 * k), y0 + i * lh), scale, YELLOW, thick, shadow=False)

    def _debug(self, img, vm: ViewModel, k: float) -> None:
        h, w = img.shape[:2]
        d = vm.debug
        for lm in d.get("hands", []):
            pts = [(int(x * w), int(y * h)) for x, y in lm]
            for a, b in HAND_CONNECTIONS:
                cv2.line(img, pts[a], pts[b], CYAN, max(1, int(2 * k)), cv2.LINE_AA)
            for p in pts:
                cv2.circle(img, p, max(2, int(3 * k)), WHITE, -1, cv2.LINE_AA)
        for (px, py), inside in zip(d.get("palms", []), d.get("palms_in_zone", [])):
            cv2.circle(img, (int(px * w), int(py * h)), int(12 * k), GREEN if inside else RED, -1, cv2.LINE_AA)
        lines = [f"{key}: {val}" for key, val in d.items() if key not in ("hands", "palms", "palms_in_zone")]
        x0 = w - int(330 * k)
        y0 = int(130 * k)
        _panel(img, x0 - int(10 * k), y0 - int(24 * k), w, y0 + len(lines) * int(24 * k), alpha=0.6)
        for i, line in enumerate(lines):
            _text(img, line, (x0, y0 + i * int(24 * k)), 0.55 * k, YELLOW, 1, shadow=False)
