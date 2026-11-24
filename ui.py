"""OpenCV rendering. Pure drawing: main.py builds a ViewModel, ui draws it."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from detection import Zone
from hands import HAND_CONNECTIONS
from models import format_time
from stats import SessionStats
from timer_state import State

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

HELP = ("SPACE timer  2 +2  D DNF  BKSP delete  I inspection  N scramble  C calibrate  S scan  V check  "
        "R record  H debug  ESC abort  Q quit")

# Sticker colours for drawing (BGR)
STICKER = {"W": (240, 240, 240), "Y": (0, 215, 255), "R": (40, 40, 215), "O": (0, 140, 255),
           "G": (60, 185, 60), "B": (205, 95, 25), "?": (90, 90, 90)}
STATUS_COLOR = {"ok": (80, 220, 80), "mismatch": (40, 40, 255), "uncertain": (0, 220, 255), "unseen": (120, 120, 120)}
# Net layout: face -> (column, row) in units of faces
NET_POS = {"U": (1, 0), "L": (0, 1), "F": (1, 1), "R": (2, 1), "B": (3, 1), "D": (1, 2)}


@dataclass
class VisionView:
    """Everything stage-2 vision wants drawn on top of the camera image."""

    corners: Any = None                      # (4,2) normalised face outline
    points: Any = None                       # (9,2) normalised sticker centres (true sticker order)
    labels: list[str] | None = None          # 9 texts ("R", "R/O?", "?")
    colors: list[str] | None = None          # 9 letters for fill colour
    confs: list[float] | None = None
    status: list[str] | None = None          # 9 x ok|mismatch|uncertain|unseen (scramble check)
    progress: float = 0.0                    # stillness 0..1
    show_grid: bool = False                  # labels + confidences (debug or a mode)
    panel_title: str = ""
    panel_lines: list[str] = field(default_factory=list)
    net: list[str] | None = None             # 54 colour letters (URFDLB order) for the cube net
    net_target: str | None = None            # face being scanned
    net_marks: set = field(default_factory=set)  # facelet indices to mark (unsure / changed)
    check_text: str = ""
    check_color: tuple = WHITE
    rec_on: bool = False
    rec_active: bool = False


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
    zone_mode: str = "idle"                  # idle | holding | ready | solving | off
    hold_progress: float = 0.0
    status_lines: list[str] = field(default_factory=list)
    message: str | None = None
    flash: tuple | None = None               # border colour for inspection warnings
    show_debug: bool = False
    debug: dict[str, Any] = field(default_factory=dict)
    vision: VisionView | None = None


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
        self._panel_bottom = 0
        if vm.vision is not None:
            self._vision_overlay(img, vm.vision, k)
        self._scramble(img, vm, k)
        if vm.vision is not None:
            self._vision_panels(img, vm, k)
        if vm.show_debug:
            self._debug(img, vm, k)
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
        x1, y1, x2, y2 = vm.zone.to_pixels(w, h)
        color = {"idle": GREY, "holding": YELLOW, "ready": GREEN, "solving": RED}.get(vm.zone_mode, GREY)
        if vm.zone_mode in ("ready", "holding"):
            _panel(img, x1, y1, x2, y2, alpha=0.25, color=color)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, max(2, int(3 * k)))
        if vm.zone_mode == "holding" and vm.hold_progress > 0:
            px = x1 + int((x2 - x1) * vm.hold_progress)
            cv2.rectangle(img, (x1, y2 - int(10 * k)), (px, y2), color, -1)
        label = {"idle": "START ZONE - rest both hands here", "holding": "hold...",
                 "ready": "READY - lift to start", "solving": "return hands to stop"}[vm.zone_mode]
        _text(img, label, (x1 + int(10 * k), y1 + int(28 * k)), 0.7 * k, color, max(1, int(2 * k)))

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
        y0 = max(int(130 * k), getattr(self, "_panel_bottom", 0) + int(34 * k))
        _panel(img, x0 - int(10 * k), y0 - int(24 * k), w, y0 + len(lines) * int(24 * k), alpha=0.6)
        for i, line in enumerate(lines):
            _text(img, line, (x0, y0 + i * int(24 * k)), 0.55 * k, YELLOW, 1, shadow=False)

    # ------------------------------------------------------------------ stage 2: vision
    def _vision_overlay(self, img, v: VisionView, k: float) -> None:
        if v.points is None or v.corners is None:
            return
        h, w = img.shape[:2]
        corners = (np.asarray(v.corners) * [w, h]).astype(np.int32)
        pts = np.asarray(v.points) * [w, h]
        side = float(np.mean(np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1))) / 3
        col = GREEN if v.progress >= 1 else YELLOW
        cv2.polylines(img, [corners], True, col, max(1, int(2 * k)), cv2.LINE_AA)
        if 0 < v.progress < 1:  # stillness bar under the face
            x0, y0 = int(corners[:, 0].min()), int(corners[:, 1].max()) + int(8 * k)
            x1 = int(corners[:, 0].max())
            cv2.rectangle(img, (x0, y0), (x0 + int((x1 - x0) * v.progress), y0 + int(5 * k)), YELLOW, -1)
        r = max(4, int(side * 0.18))
        lw = max(2, int(3 * k))
        for i in range(9):
            p = (int(pts[i, 0]), int(pts[i, 1]))
            letter = (v.colors or ["?"] * 9)[i]
            if v.status is not None:
                st = v.status[i]
                cv2.circle(img, p, r + lw, STATUS_COLOR[st], lw, cv2.LINE_AA)
                if st == "mismatch":
                    d = int(r * 1.4)
                    cv2.line(img, (p[0] - d, p[1] - d), (p[0] + d, p[1] + d), STATUS_COLOR[st], lw)
                    cv2.line(img, (p[0] - d, p[1] + d), (p[0] + d, p[1] - d), STATUS_COLOR[st], lw)
            if v.show_grid:
                cv2.circle(img, p, r, STICKER.get(letter, STICKER["?"]), -1, cv2.LINE_AA)
                cv2.circle(img, p, r, DARK, 1, cv2.LINE_AA)
                label = (v.labels or ["?"] * 9)[i]
                conf = (v.confs or [0.0] * 9)[i]
                _text(img, label, (p[0] - int(8 * k), p[1] - r - int(4 * k)), 0.5 * k, WHITE, 1)
                _text(img, f"{conf:.2f}", (p[0] - int(14 * k), p[1] + r + int(14 * k)), 0.38 * k,
                      GREEN if conf >= 0.5 else ORANGE, 1)

    def _vision_panels(self, img, vm: ViewModel, k: float) -> None:
        v = vm.vision
        h, w = img.shape[:2]
        if v.check_text:
            scale, thick = 0.75 * k, max(1, int(2 * k))
            tw = cv2.getTextSize(v.check_text, FONT, scale, thick)[0][0]
            y = int(98 * k)
            _panel(img, (w - tw) // 2 - int(10 * k), y - int(24 * k), (w + tw) // 2 + int(10 * k),
                   y + int(10 * k), 0.6)
            _text(img, v.check_text, ((w - tw) // 2, y), scale, v.check_color, thick, shadow=False)
        if v.rec_on:
            y = int(98 * k)
            col = RED if v.rec_active else GREY
            cv2.circle(img, (int(24 * k), y - int(7 * k)), int(8 * k), col, -1, cv2.LINE_AA)
            _text(img, "REC" if v.rec_active else "REC armed", (int(38 * k), y), 0.6 * k, col, max(1, int(2 * k)))
        if not (v.panel_title or v.net is not None):
            return
        cell = int(17 * k)
        pw = int(370 * k)
        x0 = w - pw - int(10 * k)
        y0 = int(120 * k)
        lines = []
        for line in v.panel_lines:
            lines.extend(_wrap(line, pw - int(20 * k), 0.55 * k, 1))
        net_h = cell * 9 + int(16 * k) if v.net is not None else 0
        ph = int(40 * k) + len(lines) * int(22 * k) + net_h
        _panel(img, x0, y0, x0 + pw, y0 + ph, 0.75)
        self._panel_bottom = y0 + ph
        _text(img, v.panel_title, (x0 + int(10 * k), y0 + int(28 * k)), 0.75 * k, CYAN, max(1, int(2 * k)),
              shadow=False)
        y = y0 + int(52 * k)
        for line in lines:
            _text(img, line, (x0 + int(10 * k), y), 0.55 * k, WHITE, 1, shadow=False)
            y += int(22 * k)
        if v.net is not None:
            self._net(img, v, x0 + (pw - cell * 12) // 2, y - int(8 * k), cell)

    def _net(self, img, v: VisionView, x0: int, y0: int, cell: int) -> None:
        for fi, f in enumerate("URFDLB"):
            fx, fy = NET_POS[f]
            ox, oy = x0 + fx * 3 * cell, y0 + fy * 3 * cell
            for j in range(9):
                r, c = divmod(j, 3)
                letter = v.net[fi * 9 + j]
                p1 = (ox + c * cell, oy + r * cell)
                p2 = (p1[0] + cell - 1, p1[1] + cell - 1)
                fill = STICKER.get(letter, STICKER["?"]) if letter != "?" else (45, 45, 45)
                cv2.rectangle(img, p1, p2, fill, -1)
                cv2.rectangle(img, p1, p2, DARK, 1)
                if fi * 9 + j in v.net_marks:
                    cv2.line(img, p1, p2, (255, 0, 255), 2)
            if f == v.net_target:
                cv2.rectangle(img, (ox - 2, oy - 2), (ox + 3 * cell + 1, oy + 3 * cell + 1), YELLOW, 2)
