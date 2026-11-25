"""Synthetic cube images with ground truth, for testing detection and colours.

A 3D cube (black body with stickers, or stickerless tiles) is posed with random
yaw/pitch/roll, projected through a pinhole camera onto a cluttered background,
then degraded: lighting gain and colour tint, gamma, vignetting, blur, noise,
JPEG compression and optional finger occlusions in front of the front face.

The front face is the cube's F face; its ground-truth sticker colours and
centres are returned in reading order (row by row, as seen in the image).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from cube import GEOMETRY, FACES, SOLVED, apply_moves, scrambled_state, to_colors
from scramble import generate_scramble

# Base BGR sticker colours with the spread seen across cube brands and webcams.
STICKER_BGR = {
    "W": ((228, 228, 225), (14, 14, 14)),
    "Y": ((40, 215, 240), (12, 15, 10)),
    "R": ((45, 35, 195), (12, 12, 18)),
    "O": ((25, 120, 250), (10, 18, 6)),
    "G": ((80, 175, 30), (15, 18, 15)),
    "B": ((185, 80, 20), (18, 15, 12)),
}
SKIN_BGR = ((115, 150, 205), (20, 22, 22))


@dataclass
class Lighting:
    gain: float = 1.0
    tint: tuple[float, float, float] = (1.0, 1.0, 1.0)   # BGR multipliers
    gamma: float = 1.0
    blur: float = 0.0
    noise: float = 3.0
    jpeg: int = 90
    vignette: float = 0.0

    @classmethod
    def random(cls, rng: np.random.Generator, strength: float = 1.0) -> "Lighting":
        tint = tuple(float(v) for v in 1.0 + rng.uniform(-0.10, 0.10, 3) * strength)
        return cls(
            gain=float(rng.uniform(0.75, 1.15)),
            tint=tint,
            gamma=float(rng.uniform(0.85, 1.15)),
            blur=float(rng.uniform(0.0, 1.4)),
            noise=float(rng.uniform(1.0, 6.0)),
            jpeg=int(rng.integers(60, 96)),
            vignette=float(rng.uniform(0.0, 0.35)),
        )


@dataclass
class SynthScene:
    image: np.ndarray
    colors: list[str]                    # front face, reading order
    centers: np.ndarray                  # (9,2) pixel centres, reading order
    occluded: np.ndarray                 # (9,) bool: finger covers the sampling patch
    face_quad: np.ndarray                # (4,2) outer corners of the front face
    pitch: float                         # mean sticker spacing in px
    params: dict = field(default_factory=dict)


def _rot_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return rz @ rx @ ry


def _jitter(rng, spec) -> np.ndarray:
    mean, sd = spec
    return np.clip(np.array(mean, float) + rng.normal(0, 1, 3) * np.array(sd, float), 0, 255)


def _background(rng: np.random.Generator, w: int, h: int) -> np.ndarray:
    c1, c2 = rng.uniform(30, 220, 3), rng.uniform(30, 220, 3)
    gx = np.linspace(0, 1, w)[None, :, None]
    gy = np.linspace(0, 1, h)[:, None, None]
    a = rng.uniform(0, 1)
    g = a * gx + (1 - a) * gy
    img = (c1 * (1 - g) + c2 * g).astype(np.float32)
    img = np.ascontiguousarray(img)
    for _ in range(int(rng.integers(4, 22))):
        col = tuple(float(v) for v in rng.uniform(0, 255, 3))
        kind = rng.integers(0, 4)
        if kind == 0:
            cx, cy = rng.uniform(0, w), rng.uniform(0, h)
            ww, hh = rng.uniform(15, w * 0.4), rng.uniform(15, h * 0.4)
            box = cv2.boxPoints(((cx, cy), (ww, hh), rng.uniform(0, 180))).astype(np.int32)
            cv2.fillConvexPoly(img, box, col, cv2.LINE_AA)
        elif kind == 1:
            cv2.circle(img, (int(rng.uniform(0, w)), int(rng.uniform(0, h))), int(rng.uniform(8, h * 0.3)),
                       col, -1, cv2.LINE_AA)
        elif kind == 2:
            p1 = (int(rng.uniform(0, w)), int(rng.uniform(0, h)))
            p2 = (int(rng.uniform(0, w)), int(rng.uniform(0, h)))
            cv2.line(img, p1, p2, col, int(rng.integers(1, 8)), cv2.LINE_AA)
        else:  # a small "tiled" distractor (keyboard keys, book spines ...)
            x0, y0 = rng.uniform(0, w * 0.8), rng.uniform(0, h * 0.8)
            s = rng.uniform(8, 25)
            for i in range(int(rng.integers(2, 5))):
                for j in range(int(rng.integers(1, 3))):
                    cv2.rectangle(img, (int(x0 + i * s * 1.3), int(y0 + j * s * 1.3)),
                                  (int(x0 + i * s * 1.3 + s), int(y0 + j * s * 1.3 + s)), col, -1)
    noise = cv2.resize(rng.normal(0, 12, (h // 16 + 1, w // 16 + 1, 3)).astype(np.float32), (w, h))
    return np.clip(img + noise, 0, 255)


def _face_axes(face_letter: str):
    base = FACES.index(face_letter) * 9
    pos = np.array([GEOMETRY[base + i][0] for i in range(9)], float)
    n = np.array(GEOMETRY[base][1], float)
    col = pos[1] - pos[0]
    row = pos[3] - pos[0]
    return pos, n, col, row


def render_scene(
    rng: np.random.Generator,
    width: int = 640,
    height: int = 360,
    colors54: str | None = None,
    stickerless: bool | None = None,
    fingers: int | None = None,
    lighting: Lighting | None = None,
    max_angle: float = 30.0,
    face_px: tuple[float, float] = (110, 250),
    supersample: int = 2,
    sticker_bgr: dict | None = None,
    pose: tuple[float, float, float] | None = None,
) -> SynthScene:
    """Render one cube scene. `colors54` is a 54-char colour string (default: random scramble)."""
    if colors54 is None:
        colors54 = to_colors(scrambled_state(generate_scramble(rng=_py_rng(rng))))
    if stickerless is None:
        stickerless = bool(rng.random() < 0.3)
    if fingers is None:
        fingers = int(rng.choice([0, 0, 1, 2, 3]))
    lighting = lighting or Lighting.random(rng)
    palette = {c: _jitter(rng, sticker_bgr[c] if sticker_bgr else STICKER_BGR[c]) for c in "WYROGB"}

    S = supersample
    W, H = width * S, height * S
    f = 0.9 * W
    deg = math.radians
    if pose is None:
        yaw, pitch, roll = (deg(rng.uniform(-max_angle, max_angle)) for _ in range(3))
    else:
        yaw, pitch, roll = (deg(a) for a in pose)
    R = _rot_matrix(yaw, pitch, roll)
    face_w = rng.uniform(*face_px) * S
    Z = f * 3.0 / face_w
    margin = face_w * 0.9
    cx = rng.uniform(margin, W - margin) if W > 2 * margin else W / 2
    cy = rng.uniform(margin * 0.8, H - margin * 0.8) if H > 1.6 * margin else H / 2
    T = np.array([(cx - W / 2) * Z / f, (cy - H / 2) * Z / f, Z])

    def to_cam(p):  # cube frame (x right, y up, z to viewer) -> camera frame (x right, y down, z forward)
        q = (R @ np.asarray(p, float).T).T
        q = q * np.array([1, -1, -1])
        return q + T

    def project(p):
        c = to_cam(p)
        c = np.atleast_2d(c)
        return np.stack([f * c[:, 0] / c[:, 2] + W / 2, f * c[:, 1] / c[:, 2] + H / 2], axis=1)

    img = _background(rng, W, H)
    body = np.array([18, 18, 20], float) + rng.uniform(0, 25)
    sticker_size = 0.97 if stickerless else rng.uniform(0.84, 0.92)
    light = np.array([rng.uniform(-0.4, 0.4), rng.uniform(-0.6, 0.1), -1.0])
    light /= np.linalg.norm(light)

    faces = []
    for fl in FACES:
        pos, n, col, row = _face_axes(fl)
        centre = n * 1.5
        n_cam = (R @ n) * np.array([1, -1, -1])
        c_cam = to_cam(centre)
        if n_cam @ (-c_cam) <= 0:
            continue
        faces.append((float(c_cam[2]), fl, pos, n, col, row, n_cam))
    faces.sort(key=lambda x: -x[0])  # far first

    front_centers = None
    front_quad = None
    lattice_to_img = None
    for _, fl, pos, n, col, row, n_cam in faces:
        shade = 0.55 + 0.45 * max(0.0, float(n_cam @ light))   # light points from the camera side
        centre = n * 1.5
        quad3 = [centre + 1.5 * (sx * col + sy * row) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        quad = project(np.array(quad3))
        seam = body if not stickerless else np.array([45, 45, 48], float)
        cv2.fillConvexPoly(img, np.round(quad).astype(np.int32), tuple(float(v) for v in seam * shade), cv2.LINE_AA)
        base = FACES.index(fl) * 9
        cents = []
        for k in range(9):
            pc = pos[k] + 0.5 * n
            s = sticker_size / 2
            sq3 = [pc + s * (sx * col + sy * row) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            sq = project(np.array(sq3))
            color = palette[colors54[base + k]] * shade * rng.uniform(0.95, 1.05)
            cv2.fillConvexPoly(img, np.round(sq).astype(np.int32), tuple(float(v) for v in color), cv2.LINE_AA)
            cents.append(project(pc)[0])
        if fl == "F":
            front_centers = np.array(cents)
            front_quad = quad
            src = np.array([[-0.5, -0.5], [2.5, -0.5], [2.5, 2.5], [-0.5, 2.5]], np.float32)
            lattice_to_img = cv2.getPerspectiveTransform(src, quad.astype(np.float32))

    occluded = np.zeros(9, bool)
    if front_centers is not None and fingers:
        finger_mask = np.zeros((H, W), np.uint8)
        skin = _jitter(rng, SKIN_BGR)
        cell_px = face_w / 3
        for _ in range(fingers):
            side = int(rng.integers(0, 4))
            t = rng.uniform(0.0, 2.0)
            depth = rng.uniform(0.25, 1.1)    # how far into the face the fingertip reaches (cells)
            if side == 0:
                a, b = (t, -1.6), (t, -0.5 + depth)
            elif side == 1:
                a, b = (t, 3.6), (t, 2.5 - depth)
            elif side == 2:
                a, b = (-1.6, t), (-0.5 + depth, t)
            else:
                a, b = (3.6, t), (2.5 - depth, t)
            pts = cv2.perspectiveTransform(np.array([[a, b]], np.float32), lattice_to_img)[0]
            thick = int(cell_px * rng.uniform(0.55, 0.85))
            p1, p2 = tuple(int(v) for v in pts[0]), tuple(int(v) for v in pts[1])
            cv2.line(finger_mask, p1, p2, 255, thick, cv2.LINE_AA)
            cv2.line(img, p1, p2, tuple(float(v) for v in skin), thick, cv2.LINE_AA)
            # fingernail-ish highlight
            cv2.circle(img, p2, max(2, thick // 4), tuple(float(min(255, v * 1.15)) for v in skin), -1, cv2.LINE_AA)
        for k in range(9):
            r, c = divmod(k, 3)
            patch = np.array([[c - 0.22, r - 0.22], [c + 0.22, r - 0.22], [c + 0.22, r + 0.22], [c - 0.22, r + 0.22]],
                             np.float32)
            poly = cv2.perspectiveTransform(patch[None], lattice_to_img)[0]
            m = np.zeros((H, W), np.uint8)
            cv2.fillConvexPoly(m, np.round(poly).astype(np.int32), 255)
            area = max(1, int((m > 0).sum()))
            occluded[k] = ((finger_mask > 127) & (m > 0)).sum() / area > 0.10

    img = _apply_lighting(img, lighting, rng)
    if S > 1:
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    img = _post(img, lighting, rng)
    centers = front_centers / S
    pitch_px = float(np.mean([np.linalg.norm(centers[i + 1] - centers[i]) for i in (0, 1, 3, 4, 6, 7)]))
    return SynthScene(
        image=img,
        colors=list(colors54[18:27]),
        centers=centers,
        occluded=occluded,
        face_quad=front_quad / S,
        pitch=pitch_px,
        params={"yaw": math.degrees(yaw), "pitch": math.degrees(pitch), "roll": math.degrees(roll),
                "stickerless": stickerless, "fingers": fingers, "lighting": lighting},
    )


def _apply_lighting(img: np.ndarray, L: Lighting, rng) -> np.ndarray:
    out = img.astype(np.float32) * L.gain * np.array(L.tint, np.float32)[None, None, :]
    if L.vignette > 0:
        h, w = out.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
        out *= (1 - L.vignette * np.clip(r, 0, 1.4) ** 2 / 2)[..., None]
    out = np.clip(out, 0, 255) / 255.0
    out = np.power(out, L.gamma) * 255.0
    return out


def _post(img: np.ndarray, L: Lighting, rng) -> np.ndarray:
    if L.blur > 0.2:
        img = cv2.GaussianBlur(img, (0, 0), L.blur)
    img = img + rng.normal(0, L.noise, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(L.jpeg)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else img


def render_background(rng: np.random.Generator, width: int = 640, height: int = 360) -> np.ndarray:
    """A cube-free cluttered scene (for false-positive tests)."""
    img = _apply_lighting(_background(rng, width, height), Lighting.random(rng), rng)
    return _post(img, Lighting.random(rng), rng)


def calibration_views(colors_to_show: str = "WYROGB") -> dict[str, str]:
    """54-colour strings of a solved cube rotated so each colour faces the camera (F)."""
    rot = {"G": "", "R": "y", "B": "y2", "O": "y'", "W": "x'", "Y": "x"}
    return {c: to_colors(apply_moves(SOLVED, rot[c])) for c in colors_to_show}


def _py_rng(rng: np.random.Generator):
    import random

    return random.Random(int(rng.integers(0, 2**31)))


# =========================================================================== stage 3: animated solves
# A cube built from 26 cubies, so single layers (and the whole cube) can be
# rotated part-way. Rendered at 30 fps with motion blur (sub-frame averaging),
# a slowly drifting / wobbling pose, two skin-coloured hands holding it from
# the sides (fingers on the face that points at the camera), sleeves, and an
# optional moving distractor far from the cube. The default view is the one a
# webcam in front of the solver gets: the back (B) face, a bit of U from above.
import cube as _cube  # noqa: E402

_CUBIES = [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1) if (x, y, z) != (0, 0, 0)]
_DIRS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
SLEEVE_BGR = ((90, 60, 50), (25, 20, 20))


def _axis_rot(axis: int, ang: float) -> np.ndarray:
    c, s = math.cos(ang), math.sin(ang)
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def move_spec(move: str) -> tuple[int, frozenset, float]:
    """(axis, layers, total right-hand angle in radians) for a canonical move."""
    base = move.rstrip("2'")
    axis, layers, direction = _cube._MOVE_DEFS[base]
    q = 2 if move.endswith("2") else -1 if move.endswith("'") else 1
    return axis, layers, direction * q * math.pi / 2


def _build_cubie_faces():
    """Static arrays for all 26*6 cubie faces: centre, normal, 4 corners, facelet index (-1 = body), cubie."""
    cents, norms, corners, fidx, pos = [], [], [], [], []
    for p in _CUBIES:
        for n in _DIRS:
            a = [i for i in range(3) if n[i] == 0]
            u = np.zeros(3)
            v = np.zeros(3)
            u[a[0]] = 0.5
            v[a[1]] = 0.5
            c = np.array(p, float) + 0.5 * np.array(n, float)
            cents.append(c)
            norms.append(n)
            corners.append([c - u - v, c + u - v, c + u + v, c - u + v])
            fidx.append(_cube._INDEX.get((p, n), -1))
            pos.append(p)
    return np.array(cents), np.array(norms, float), np.array(corners), np.array(fidx), np.array(pos)


_CF_CENT, _CF_NORM, _CF_CORN, _CF_FIDX, _CF_POS = _build_cubie_faces()


@dataclass
class TimedMove:
    move: str            # canonical ("R", "U2", "y'")
    kind: str            # "turn" | "rotation"
    t0: float
    t1: float


@dataclass
class MotionScript:
    moves: list[TimedMove]
    duration: float
    pauses: list[tuple[float, float]] = field(default_factory=list)   # deliberate pauses (P tokens)

    @property
    def turns(self) -> int:
        return sum(m.kind == "turn" for m in self.moves)

    @property
    def rotations(self) -> int:
        return sum(m.kind == "rotation" for m in self.moves)

    @property
    def half_turns(self) -> int:
        return sum(m.kind == "turn" and m.move.endswith("2") for m in self.moves)

    def active_at(self, t) -> np.ndarray:
        t = np.asarray(t, float)
        out = np.zeros(t.shape, bool)
        for m in self.moves:
            out |= (t >= m.t0) & (t <= m.t1)
        return out


def make_script(steps, rng: np.random.Generator, quarter=(0.20, 0.34), half_factor=(1.45, 1.85),
                rotation=(0.32, 0.50), gap=(0.10, 0.45), lead=0.8, tail=0.6) -> MotionScript:
    """Timed performance of practice steps (vision.practice.Step list) at roughly 1-2 turns per second."""
    t = lead
    moves, pauses = [], []
    for s in steps:
        if s.kind == "pause":
            pauses.append((t, t + s.seconds))
            t += s.seconds
            continue
        if s.kind == "rotation":
            d = rng.uniform(*rotation) * (1.5 if s.move.endswith("2") else 1.0)
        else:
            d = rng.uniform(*quarter) * (rng.uniform(*half_factor) if s.move.endswith("2") else 1.0)
        moves.append(TimedMove(s.move, s.kind, t, t + d))
        t += d + rng.uniform(*gap)
    return MotionScript(moves, t + tail, pauses)


@dataclass
class MotionFrame:
    t: float
    image: np.ndarray
    hands: list                                     # 21x2 normalised landmarks per detected hand (raw image)
    cube_box: tuple[float, float, float, float]     # normalised x, y, w, h of the cube silhouette


def _smooth(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


class MotionRenderer:
    """Renders a MotionScript frame by frame (a generator, so clips never sit in memory)."""

    def __init__(self, rng: np.random.Generator, width: int = 640, height: int = 360, fps: float = 30.0,
                 face_px: float | None = None, base_pose=(180.0, 20.0, 0.0), stickerless: bool = False,
                 blur_subframes: int = 3, exposure: float = 0.55, wobble: float = 3.0, sway: float = 0.03,
                 lift: bool = True, hands: bool = True, hand_dropout: float = 0.03, distractor: bool = True,
                 landmark_noise: float = 1.5, lighting: Lighting | None = None):
        self.rng = rng
        self.W, self.H, self.fps = width, height, fps
        self.f = 0.9 * width
        self.face_px = face_px if face_px is not None else float(rng.uniform(0.30, 0.42) * height)
        self.base = [math.radians(a) for a in base_pose]
        self.base[0] += math.radians(rng.uniform(-15, 15))
        self.base[1] += math.radians(rng.uniform(-6, 6))
        self.stickerless = stickerless
        self.K = max(1, int(blur_subframes))
        self.exposure = exposure / fps
        self.wobble = math.radians(wobble)
        self.sway = sway
        self.lift = lift
        self.hands_on = hands
        self.hand_dropout = hand_dropout
        self.distractor = distractor
        self.lm_noise = landmark_noise
        self.lighting = lighting or Lighting.random(rng, 0.7)
        self.lighting.blur = min(self.lighting.blur, 0.8)
        self.palette = {c: _jitter(rng, STICKER_BGR[c]) for c in "WYROGB"}
        self.body = np.array([18, 18, 20], float) + rng.uniform(0, 20)
        self.skin = _jitter(rng, SKIN_BGR)
        self.sleeve = _jitter(rng, SLEEVE_BGR)
        self.bg = _background(rng, width, height).astype(np.float32)
        self.sticker = 0.97 if stickerless else float(rng.uniform(0.84, 0.90))
        light = np.array([rng.uniform(-0.4, 0.4), rng.uniform(-0.6, 0.1), -1.0])
        self.light = light / np.linalg.norm(light)
        self.phase = rng.uniform(0, 2 * math.pi, 6)
        self.freq = rng.uniform(0.15, 0.45, 6)
        self.cx0 = width * rng.uniform(0.42, 0.58)
        self.cy0 = height * rng.uniform(0.38, 0.50)
        L = self.lighting
        yy, xx = np.mgrid[0:height, 0:width]
        r = np.sqrt(((xx - width / 2) / (width / 2)) ** 2 + ((yy - height / 2) / (height / 2)) ** 2)
        self.bg = np.clip(self.bg * (1 - L.vignette * np.clip(r, 0, 1.4) ** 2 / 2)[..., None], 0, 255)
        self.bg8 = self.bg.astype(np.uint8)
        # global gain / tint / gamma as one lookup table per channel (fast on uint8)
        x = np.arange(256, dtype=np.float64)
        self.lut = np.stack([np.clip((np.clip(x * L.gain * L.tint[c], 0, 255) / 255.0) ** L.gamma * 255.0, 0, 255)
                             for c in range(3)], axis=-1).astype(np.uint8)[:, None, :]
        self._noise = np.empty((height, width, 3), np.int16)
        self.fingers = {side: [(float(rng.uniform(-0.55, 0.55)), float(rng.uniform(0.18, 0.34)))
                               for _ in range(3)] for side in (0, 1)}
        self._states: list[str] = []

    # ------------------------------------------------------------------ pose / state
    def _pose(self, t: float, script: MotionScript):
        w = self.wobble
        yaw = self.base[0] + w * math.sin(2 * math.pi * self.freq[0] * t + self.phase[0])
        pitch = self.base[1] + w * math.sin(2 * math.pi * self.freq[1] * t + self.phase[1])
        roll = self.base[2] + w * math.sin(2 * math.pi * self.freq[2] * t + self.phase[2])
        cx = self.cx0 + self.sway * self.W * math.sin(2 * math.pi * self.freq[3] * t + self.phase[3])
        cy = self.cy0 + self.sway * self.H * math.sin(2 * math.pi * self.freq[4] * t + self.phase[4])
        if self.lift:   # the cube is lifted into view at the start and lowered at the end (hands carry it)
            up = _smooth(t / 0.5) * _smooth((script.duration - t) / 0.5)
            cy += (1 - up) * 0.25 * self.H
        return _rot_matrix(yaw, pitch, roll), cx, cy

    def _state_at(self, t: float, script: MotionScript):
        """(facelet state before the current move, current move or None, progress 0..1)."""
        if len(self._states) != len(script.moves) + 1:
            self._states = [_cube.SOLVED]
            for m in script.moves:
                self._states.append(_cube.apply_moves(self._states[-1], m.move))
        for i, m in enumerate(script.moves):
            if t < m.t0:
                return self._states[i], None, 0.0
            if t < m.t1:
                return self._states[i], m, _smooth((t - m.t0) / (m.t1 - m.t0))
        return self._states[-1], None, 0.0

    # ------------------------------------------------------------------ drawing
    def _draw_cube(self, img: np.ndarray, t: float, script: MotionScript):
        R, cx, cy = self._pose(t, script)
        Z = self.f * 3.0 / self.face_px
        T = np.array([(cx - self.W / 2) * Z / self.f, (cy - self.H / 2) * Z / self.f, Z])
        state, mv, prog = self._state_at(t, script)
        cents, norms, corns = _CF_CENT, _CF_NORM, _CF_CORN
        if mv is not None:
            axis, layers, ang = move_spec(mv.move)
            Rl = _axis_rot(axis, ang * prog)
            sel = np.isin(_CF_POS[:, axis], list(layers))
            cents, norms, corns = cents.copy(), norms.copy(), corns.copy()
            cents[sel] = cents[sel] @ Rl.T
            norms[sel] = norms[sel] @ Rl.T
            corns[sel] = corns[sel] @ Rl.T
        flip = np.array([1, -1, -1])
        c_cam = (cents @ R.T) * flip + T
        n_cam = (norms @ R.T) * flip
        k_cam = (corns @ R.T) * flip + T
        vis = np.einsum("ij,ij->i", n_cam, -c_cam) > 0
        order = np.argsort(-c_cam[:, 2])
        colors = to_colors(state)
        f, W2, H2 = self.f, self.W / 2, self.H / 2
        pts2 = np.stack([f * k_cam[..., 0] / k_cam[..., 2] + W2, f * k_cam[..., 1] / k_cam[..., 2] + H2], -1)
        seam = np.array([45, 45, 48], float)
        for i in order:
            if not vis[i]:
                continue
            shade = 0.55 + 0.45 * max(0.0, float(n_cam[i] @ self.light))
            quad = pts2[i]
            fi = _CF_FIDX[i]
            body = seam if self.stickerless and fi >= 0 else self.body
            cv2.fillConvexPoly(img, np.round(quad * 4).astype(np.int32), tuple(float(v) for v in body * shade),
                               cv2.LINE_AA, shift=2)
            if fi >= 0:
                c = quad.mean(axis=0)
                sq = c + (quad - c) * self.sticker
                col = self.palette[colors[fi]] * shade
                cv2.fillConvexPoly(img, np.round(sq * 4).astype(np.int32), tuple(float(v) for v in col),
                                   cv2.LINE_AA, shift=2)
        box3 = np.array([[x, y, z] for x in (-1.5, 1.5) for y in (-1.5, 1.5) for z in (-1.5, 1.5)])
        b_cam = (box3 @ R.T) * flip + T
        b2 = np.stack([f * b_cam[:, 0] / b_cam[:, 2] + W2, f * b_cam[:, 1] / b_cam[:, 2] + H2], -1)
        return b2, mv, prog

    def _hand_geometry(self, box2: np.ndarray, mv, prog: float):
        """Palm centres + finger segments in pixels for both hands, animated by the current move."""
        x0, y0 = box2.min(axis=0)
        x1, y1 = box2.max(axis=0)
        cw, ch = x1 - x0, y1 - y0
        ycen = (y0 + y1) / 2 + 0.08 * ch
        hands = []
        for side in (0, 1):                       # 0 = image-left hand, 1 = image-right hand
            sgn = -1 if side == 0 else 1
            palm = np.array([(x0 if side == 0 else x1) + sgn * 0.10 * cw, ycen])
            dy_hand = 0.0
            flick = 0.0
            if mv is not None:
                base = mv.move.rstrip("2'")
                bump = math.sin(math.pi * prog)
                # after the 180-degree yaw the cube's R face is on the image LEFT
                if (base in ("R", "Rw") and side == 0) or (base in ("L", "Lw") and side == 1):
                    # the wrist turns with the layer, then regrips (out and back within the move)
                    _, _, ang = move_spec(mv.move)
                    dy_hand = -0.18 * ch * math.copysign(bump, ang) * sgn
                elif base in ("U", "Uw", "D", "Dw") and side == (0 if mv.move.endswith("'") else 1):
                    flick = bump
                elif mv.kind == "rotation" and side == (1 if base == "y" else 0):
                    palm = palm + np.array([sgn * 0.30 * cw * bump, -0.15 * ch * bump])
            palm = palm + np.array([0.0, dy_hand])
            fingers = []
            for j, (fy, reach) in enumerate(self.fingers[side]):
                b = np.array([palm[0] - sgn * 0.05 * cw, ycen + fy * ch * 0.8 + dy_hand])
                tip = b + np.array([-sgn * reach * cw, -0.04 * ch])
                if j == 0 and flick:
                    tip = tip + np.array([-sgn * 0.35 * cw * flick, -0.25 * ch * flick])
                fingers.append((b, tip))
            hands.append((palm, fingers))
        return hands, (cw, ch)

    def _draw_hands(self, img, hands, size):
        cw, ch = size
        thick = max(3, int(0.11 * cw))
        for side, (palm, fingers) in enumerate(hands):
            sgn = -1 if side == 0 else 1
            elbow = palm + np.array([sgn * 0.9 * cw, 1.6 * ch])
            start = palm + np.array([sgn * 0.15 * cw, 0.3 * ch])
            cv2.line(img, (int(start[0]), int(start[1])), (int(elbow[0]), int(elbow[1])),
                     tuple(float(v) for v in self.sleeve), max(3, int(0.45 * cw)), cv2.LINE_AA)
            cv2.ellipse(img, (int(palm[0]), int(palm[1])), (max(2, int(0.16 * cw)), max(2, int(0.26 * ch))), 0, 0,
                        360, tuple(float(v) for v in self.skin), -1, cv2.LINE_AA)
            for b, tip in fingers:
                cv2.line(img, (int(b[0]), int(b[1])), (int(tip[0]), int(tip[1])),
                         tuple(float(min(255.0, v * 1.03)) for v in self.skin), thick, cv2.LINE_AA)

    def _landmarks(self, hands) -> list[np.ndarray]:
        """MediaPipe-style 21 points: wrist 0, thumb 1-4, index/middle/ring MCP-PIP-DIP-TIP 5-16, pinky 17-20."""
        out = []
        for side, (palm, fingers) in enumerate(hands):
            sgn = -1 if side == 0 else 1
            fp = self.face_px
            pts = np.zeros((21, 2))
            pts[0] = palm + np.array([sgn * 0.10, 0.12]) * fp                  # wrist: outside and below
            for k, (b, tip) in enumerate(fingers):                           # three fingers on the cube
                for j in range(4):
                    pts[5 + 4 * k + j] = b + (tip - b) * j / 3
            pin = palm + np.array([0.0, 0.14]) * fp                          # pinky curled under the palm
            for j in range(4):
                pts[17 + j] = pin + np.array([-sgn * 0.03 * j, 0.02 * j]) * fp
            for j in range(4):                                               # thumb along the palm's top
                pts[1 + j] = palm + np.array([-sgn * 0.03 * j, -0.10 - 0.02 * j]) * fp
            pts += self.rng.normal(0, self.lm_noise, pts.shape)
            out.append((pts / [self.W, self.H]).astype(np.float32))
        return out

    def frames(self, script: MotionScript):
        """Yield a MotionFrame for t = 0, 1/fps, ... up to script.duration."""
        n = int(script.duration * self.fps)
        L = self.lighting
        for k in range(n):
            t = k / self.fps
            acc = np.zeros((self.H, self.W, 3), np.uint16)
            hands_geo, box = None, None
            for s in range(self.K):
                ts = t + self.exposure * (s / (self.K - 1) - 0.5) if self.K > 1 else t
                img = self.bg8.copy()
                if self.distractor:   # something moving far from the cube (a ball, someone's arm ...)
                    dx = 0.08 * self.W * math.sin(2 * math.pi * 0.7 * ts)
                    cv2.circle(img, (int(0.08 * self.W + dx), int(0.88 * self.H)), int(0.05 * self.H),
                               (40, 160, 230), -1, cv2.LINE_AA)
                box2, mv, prog = self._draw_cube(img, ts, script)
                if self.hands_on:
                    hg, size = self._hand_geometry(box2, mv, prog)
                    self._draw_hands(img, hg, size)
                    if s == self.K // 2:
                        hands_geo = hg
                if s == self.K // 2:
                    box = box2
                acc += img
            img = (acc // self.K).astype(np.uint8) if self.K > 1 else img
            img = cv2.LUT(img, self.lut)
            if L.blur > 0.2:
                img = cv2.GaussianBlur(img, (0, 0), L.blur)
            cv2.randn(self._noise, 0, L.noise)
            img = np.clip(img.astype(np.int16) + self._noise, 0, 255).astype(np.uint8)
            ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(L.jpeg)])
            img = cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else img
            lms = []
            if hands_geo is not None:
                lms = [lm for lm in self._landmarks(hands_geo) if self.rng.random() >= self.hand_dropout]
            x0, y0 = box.min(axis=0)
            x1, y1 = box.max(axis=0)
            yield MotionFrame(t, img, lms, (x0 / self.W, y0 / self.H, (x1 - x0) / self.W, (y1 - y0) / self.H))
