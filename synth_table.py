"""Synthetic camera views of a table (for tests, tools/ar_screenshots.py and demos).

A pinhole camera looks down at a wooden table. Things lying on the table (a
phone showing the calibration marker, an A4 sheet, printed markers) are drawn
into a top view in table centimetres, which is then warped into the camera
image with the exact table->image homography, so tests know the ground truth.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SynthCamera:
    width: int = 1280
    height: int = 720
    hfov_deg: float = 70.0
    tilt_deg: float = 35.0       # how far the camera looks down from horizontal
    height_cm: float = 40.0      # camera height above the table
    yaw_deg: float = 0.0         # turn left/right
    roll_deg: float = 0.0

    @property
    def K(self) -> np.ndarray:
        f = (self.width / 2) / math.tan(math.radians(self.hfov_deg) / 2)
        return np.array([[f, 0, self.width / 2], [0, f, self.height / 2], [0, 0, 1.0]])

    def H_table_to_image(self) -> np.ndarray:
        """Table (X right, Y away from the camera, cm; origin below the camera) -> image px."""
        # Camera axes in world coords for a camera looking along +Y, tilted down by `tilt`.
        t, yw, rl = (math.radians(a) for a in (self.tilt_deg, self.yaw_deg, self.roll_deg))
        fwd = np.array([math.sin(yw) * math.cos(t), math.cos(yw) * math.cos(t), -math.sin(t)])
        right = np.array([math.cos(yw), -math.sin(yw), 0.0])
        down = np.cross(fwd, right)
        # roll around the viewing axis
        right, down = (math.cos(rl) * right + math.sin(rl) * down, -math.sin(rl) * right + math.cos(rl) * down)
        R = np.vstack([right, down, fwd])            # world -> camera
        C = np.array([0.0, 0.0, self.height_cm])     # camera centre
        t_vec = -R @ C
        M = np.column_stack([R[:, 0], R[:, 1], t_vec])  # plane Z = 0
        H = self.K @ M
        return H / H[2, 2]


def wood_texture(w_px: int, h_px: int, seed: int = 0) -> np.ndarray:
    """A light-wood table top (BGR)."""
    rng = np.random.default_rng(seed)
    y = np.arange(h_px)[:, None].astype(np.float32)
    x = np.arange(w_px)[None, :].astype(np.float32)
    noise = cv2.GaussianBlur(rng.normal(0, 1, (h_px, w_px)).astype(np.float32), (0, 0), 6)
    grain = np.sin(x * 0.045 + 8 * noise + 0.002 * y) * 0.5 + 0.5
    fine = cv2.GaussianBlur(rng.normal(0, 1, (h_px, w_px)).astype(np.float32), (0, 0), 1.2)
    base = np.array([96, 140, 182], np.float32)  # BGR light oak
    img = base[None, None, :] * (0.86 + 0.12 * grain[..., None] + 0.03 * fine[..., None])
    return np.clip(img, 0, 255).astype(np.uint8)


class TableScene:
    """Top view of a table area in cm, rendered into a camera image."""

    def __init__(self, cam: SynthCamera, x_range=(-60.0, 60.0), y_range=(15.0, 110.0), px_per_cm: float = 12.0,
                 seed: int = 0):
        self.cam = cam
        self.x0, self.x1 = x_range
        self.y0, self.y1 = y_range
        self.ppc = px_per_cm
        w = int((self.x1 - self.x0) * px_per_cm)
        h = int((self.y1 - self.y0) * px_per_cm)
        self.top = wood_texture(w, h, seed)

    # top-view pixel <-> table cm. Top-view rows grow towards the camera (smaller Y),
    # so the top view looks like the table seen from above with the camera at the bottom.
    def top_px(self, pts_cm) -> np.ndarray:
        p = np.asarray(pts_cm, dtype=np.float64).reshape(-1, 2)
        return np.column_stack([(p[:, 0] - self.x0) * self.ppc, (self.y1 - p[:, 1]) * self.ppc])

    def top_to_table(self) -> np.ndarray:
        """3x3: top-view px -> table cm."""
        s = 1.0 / self.ppc
        return np.array([[s, 0, self.x0], [0, -s, self.y1], [0, 0, 1.0]])

    def H_true(self) -> np.ndarray:
        """Table cm -> camera image px."""
        return self.cam.H_table_to_image()

    def paste(self, image: np.ndarray, centre_cm, width_cm: float, angle_deg: float = 0.0,
              mask: np.ndarray | None = None) -> None:
        """Lay an image flat on the table: centred at centre_cm, width_cm wide, rotated (deg, ccw seen from above)."""
        h, w = image.shape[:2]
        s = width_cm * self.ppc / w
        c = self.top_px([centre_cm])[0]
        a = math.radians(angle_deg)
        R = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]]) * s
        t = c - R @ np.array([w / 2, h / 2])
        M = np.hstack([R, t[:, None]])
        warped = cv2.warpAffine(image, M, (self.top.shape[1], self.top.shape[0]), flags=cv2.INTER_AREA)
        m = mask if mask is not None else np.full((h, w), 255, np.uint8)
        wm = cv2.warpAffine(m, M, (self.top.shape[1], self.top.shape[0]), flags=cv2.INTER_LINEAR)
        a3 = (wm.astype(np.float32) / 255)[..., None]
        self.top[:] = (self.top * (1 - a3) + warped * a3).astype(np.uint8)

    def render(self, sky=(205, 200, 195)) -> np.ndarray:
        """The camera image (raw, not mirrored)."""
        cam = self.cam
        H = self.H_true() @ self.top_to_table()
        img = np.empty((cam.height, cam.width, 3), np.uint8)
        # wall / background behind the table: a soft vertical gradient
        g = np.linspace(0.8, 1.0, cam.height, dtype=np.float32)[:, None, None]
        img[:] = (np.array(sky, np.float32)[None, None, :] * g).astype(np.uint8)
        table = cv2.warpPerspective(self.top, H, (cam.width, cam.height), flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(np.full(self.top.shape[:2], 255, np.uint8), H, (cam.width, cam.height))
        img[mask > 0] = table[mask > 0]
        return img


def phone_image(marker_png: np.ndarray, screen_w_px: int, body_margin: float = 0.06) -> tuple[np.ndarray, np.ndarray]:
    """A phone seen from above: black body with rounded corners around the screen image. Returns (BGR, mask)."""
    sh, sw = marker_png.shape[:2]
    scale = screen_w_px / sw
    screen = cv2.resize(marker_png, (screen_w_px, int(sh * scale)), interpolation=cv2.INTER_AREA)
    if screen.ndim == 2:
        screen = cv2.cvtColor(screen, cv2.COLOR_GRAY2BGR)
    m = int(screen_w_px * body_margin)
    h, w = screen.shape[0] + 2 * m, screen.shape[1] + 2 * m
    body = np.full((h, w, 3), 18, np.uint8)
    body[m:m + screen.shape[0], m:m + screen.shape[1]] = screen
    mask = np.zeros((h, w), np.uint8)
    r = int(w * 0.12)
    cv2.rectangle(mask, (r, 0), (w - r, h), 255, -1)
    cv2.rectangle(mask, (0, r), (w, h - r), 255, -1)
    for cx, cy in ((r, r), (w - r, r), (r, h - r), (w - r, h - r)):
        cv2.circle(mask, (cx, cy), r, 255, -1, cv2.LINE_AA)
    return body, mask


def add_glare(img: np.ndarray, centre, radius: float, strength: float = 0.5) -> np.ndarray:
    """A soft bright spot (screen reflection)."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d2 = ((xx - centre[0]) ** 2 + (yy - centre[1]) ** 2) / radius ** 2
    g = (strength * np.exp(-d2))[..., None]
    return np.clip(img.astype(np.float32) * (1 - g) + 255 * g, 0, 255).astype(np.uint8)


# --------------------------------------------------------------- fake hands
def hand_landmarks(palm_xy, palm_px: float, angle_deg: float = 0.0, right: bool = True) -> np.ndarray:
    """21 MediaPipe-style landmarks (pixels) of a flat hand resting on a table, fingers pointing up-screen.

    palm_px is the wrist -> middle-knuckle distance in pixels.
    """
    s = palm_px
    pts = np.zeros((21, 2))
    pts[0] = (0, 0.55)
    side = 1 if right else -1
    for f, dx in enumerate((-0.42, -0.14, 0.12, 0.36)):
        base = 5 + 4 * f
        for j in range(4):
            pts[base + j] = (side * dx * (1 + 0.08 * j), -0.45 - 0.28 * j * (0.85 if f == 3 else 1.0))
    for j in range(1, 5):
        pts[j] = (side * (-0.38 - 0.16 * j), 0.3 - 0.17 * j)
    pts = pts * s
    pts -= pts[[0, 5, 9, 13, 17]].mean(axis=0)
    a = math.radians(angle_deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return pts @ R.T + np.asarray(palm_xy, dtype=np.float64)


HAND_BONES = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17))


def draw_hand(img: np.ndarray, lm_px: np.ndarray, skin=(120, 150, 205), sleeve=(90, 60, 40)) -> None:
    """Paint a hand (and a sleeved forearm) at the given landmarks, in place."""
    s = float(np.linalg.norm(lm_px[9] - lm_px[0]))
    wrist = lm_px[0]
    away = wrist - lm_px[9]
    away /= max(np.linalg.norm(away), 1e-6)
    elbow = wrist + away * s * 4.0
    cv2.line(img, tuple(np.int32(wrist + away * s * 0.6)), tuple(np.int32(elbow)), sleeve, int(s * 0.95), cv2.LINE_AA)
    cv2.line(img, tuple(np.int32(wrist)), tuple(np.int32(wrist + away * s * 0.7)), skin, int(s * 0.7), cv2.LINE_AA)
    palm = cv2.convexHull(np.int32(lm_px[[0, 1, 2, 5, 9, 13, 17]]))
    cv2.fillConvexPoly(img, palm, skin, cv2.LINE_AA)
    for a, b in HAND_BONES:
        cv2.line(img, tuple(np.int32(lm_px[a])), tuple(np.int32(lm_px[b])), skin, max(3, int(s * 0.28)), cv2.LINE_AA)
    for p in lm_px[[4, 8, 12, 16, 20]]:
        cv2.circle(img, tuple(np.int32(p)), max(2, int(s * 0.14)), skin, -1, cv2.LINE_AA)
