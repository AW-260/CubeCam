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
