"""The table plane: a homography between table centimetres and camera pixels.

Coordinates
-----------
* Image: pixels of the *displayed* frame (mirrored if the view is mirrored),
  for the frame size the calibration was made at (`image_size`). Other frame
  sizes are handled by scaling.
* Table: centimetres on the table surface. The frame is made canonical after
  calibration, so it doesn't depend on how the phone or sheet was lying:
  origin at the reference's centre, +X pointing to the right as seen on
  screen, +Y pointing towards the bottom of the screen (towards you), and
  the same handedness as the screen. A mat at angle 0 is therefore square to
  the view and its text reads normally.

The homography is normalised so that the reference centre has a positive
homogeneous scale. A table point whose scale comes out <= 0 lies beyond the
horizon (it can't be seen), and `to_image` / `to_table` report it as invalid.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np

SHALLOW_DEG = 20.0          # table seen at less than this elevation -> warning
MIN_REF_AREA = 0.0012       # reference smaller than 0.12 % of the frame -> warning
MIN_REF_SIDE_PX = 24.0      # or a side shorter than this (pixels)


def _h_apply(H: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply a 3x3 homography to (N, 2) points -> ((N, 2) points, (N,) homogeneous scales)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    hom = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    w = hom[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = hom[:, :2] / w[:, None]
    return out, w


def homography(src: Sequence, dst: Sequence) -> np.ndarray:
    """Homography mapping src -> dst (4 points exact, more: least squares)."""
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    if len(src) != len(dst) or len(src) < 4:
        raise ValueError("need at least 4 point pairs")
    if len(src) == 4:
        H = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32)).astype(np.float64)
    else:
        H, _ = cv2.findHomography(src, dst, 0)
        if H is None:
            raise ValueError("points are degenerate")
    if not np.all(np.isfinite(H)) or abs(np.linalg.det(H)) < 1e-12:
        raise ValueError("points are degenerate (three in a line?)")
    return H


def polygon_area(pts: np.ndarray) -> float:
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


@dataclass
class TablePlane:
    H: np.ndarray                         # table cm -> image px (3x3)
    image_size: tuple[int, int]           # (width, height) the homography is for
    source: str = ""                      # phone | click | markers
    reference_px: np.ndarray | None = None  # the reference's 4 image corners (for warnings)
    hfov_deg: float | None = None         # camera field of view used for the pose fit (info)
    _Hinv: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self):
        self.H = np.asarray(self.H, dtype=np.float64).reshape(3, 3)
        self.image_size = (int(self.image_size[0]), int(self.image_size[1]))
        self._Hinv = np.linalg.inv(self.H)

    # ----------------------------------------------------------- construction
    @classmethod
    def from_reference(cls, table_pts: Sequence, image_pts: Sequence, image_size, source: str = "") -> "TablePlane":
        """Plane from >= 4 table points (cm) and where they appear in the image (px)."""
        table_pts = np.asarray(table_pts, dtype=np.float64).reshape(-1, 2)
        image_pts = np.asarray(image_pts, dtype=np.float64).reshape(-1, 2)
        H0 = homography(table_pts, image_pts)
        return cls.from_homography(H0, table_pts.mean(axis=0), image_size, source, image_pts)

    @classmethod
    def from_homography(cls, H0: np.ndarray, centre_table, image_size, source: str = "",
                        reference_px=None, hfov_deg: float | None = None) -> "TablePlane":
        """Plane from any table->image homography; re-expressed in the canonical frame round centre_table."""
        centre_img, w = _h_apply(H0, np.asarray(centre_table, dtype=np.float64).reshape(1, 2))
        if w[0] < 0:
            H0 = -H0
        H = _canonical(H0, centre_img[0])
        ref = None if reference_px is None else np.asarray(reference_px, dtype=np.float64).reshape(-1, 2).copy()
        return cls(H, image_size, source, ref, hfov_deg)

    def scaled_to(self, image_size) -> "TablePlane":
        """The same plane for a different frame size (camera resolution changed)."""
        sx = image_size[0] / self.image_size[0]
        sy = image_size[1] / self.image_size[1]
        if math.isclose(sx, 1.0) and math.isclose(sy, 1.0):
            return self
        S = np.diag([sx, sy, 1.0])
        ref = None if self.reference_px is None else self.reference_px * [sx, sy]
        return TablePlane(S @ self.H, image_size, self.source, ref, self.hfov_deg)

    def moved(self, M: np.ndarray) -> "TablePlane":
        """Re-express in another table frame: new = M @ old (3x3 affine)."""
        return TablePlane(self.H @ np.linalg.inv(M), self.image_size, self.source, self.reference_px, self.hfov_deg)

    # ---------------------------------------------------------------- mapping
    def to_image(self, table_pts) -> tuple[np.ndarray, np.ndarray]:
        """(N, 2) cm -> ((N, 2) px, (N,) bool visible)."""
        pts, w = _h_apply(self.H, table_pts)
        return pts, (w > 1e-9) & np.all(np.isfinite(pts), axis=1)

    def to_table(self, image_pts) -> tuple[np.ndarray, np.ndarray]:
        """(N, 2) px -> ((N, 2) cm, (N,) bool on the table, i.e. below the horizon)."""
        pts, w = _h_apply(self._Hinv, image_pts)
        return pts, (w > 1e-12) & np.all(np.isfinite(pts), axis=1)

    def to_image_norm(self, table_pts) -> tuple[np.ndarray, np.ndarray]:
        pts, ok = self.to_image(table_pts)
        return pts / np.array(self.image_size, dtype=np.float64), ok

    def to_table_norm(self, norm_pts) -> tuple[np.ndarray, np.ndarray]:
        return self.to_table(np.asarray(norm_pts, dtype=np.float64).reshape(-1, 2) * self.image_size)

    # ---------------------------------------------------------------- quality
    def px_per_cm(self, table_pt=(0.0, 0.0)) -> tuple[float, float]:
        """Largest and smallest image scale (px per cm) at a table point (singular values of the Jacobian)."""
        J = _jacobian(self.H, np.asarray(table_pt, dtype=np.float64))
        s = np.linalg.svd(J, compute_uv=False)
        return float(s[0]), float(s[1])

    def elevation_deg(self, table_pt=(0.0, 0.0)) -> float:
        """Roughly how steeply the camera looks at the table there (90 = straight down, 0 = edge-on).

        A circle on the table shows up as an ellipse whose axis ratio is about
        sin(elevation); the Jacobian's singular-value ratio measures exactly that.
        """
        big, small = self.px_per_cm(table_pt)
        return math.degrees(math.asin(min(1.0, small / big))) if big > 0 else 0.0

    def warnings(self) -> list[str]:
        out = []
        elev = self.elevation_deg()
        if elev < SHALLOW_DEG:
            out.append(f"Camera sees the table almost edge-on ({elev:.0f} deg) - tilt it down or raise it")
        if self.reference_px is not None:
            ref = self.reference_px
            area = polygon_area(ref) / (self.image_size[0] * self.image_size[1])
            side = min(np.linalg.norm(ref[i] - ref[i - 1]) for i in range(len(ref)))
            if area < MIN_REF_AREA or side < MIN_REF_SIDE_PX:
                out.append(f"Reference is very small in the picture ({area:.2%} of the frame) - "
                           "bring it closer or use a bigger one")
        return out

    # ----------------------------------------------------------------- config
    def to_config(self) -> dict:
        d = {
            "source": self.source or "unknown",
            "image_size": [int(v) for v in self.image_size],
            "homography": [[float(f"{v:.10g}") for v in row] for row in (self.H / self.H[2, 2]).tolist()],
        }
        if self.reference_px is not None:
            d["reference_px"] = [[round(float(x), 2), round(float(y), 2)] for x, y in self.reference_px]
        if self.hfov_deg is not None:
            d["hfov_deg"] = round(float(self.hfov_deg), 1)
        return d

    @classmethod
    def from_config(cls, d: dict | None) -> "TablePlane | None":
        if not d or not d.get("homography"):
            return None
        H = np.array(d["homography"], dtype=np.float64)
        if H.shape != (3, 3) or not np.all(np.isfinite(H)):
            raise ValueError("table_calibration.homography must be a 3x3 matrix")
        ref = d.get("reference_px")
        hfov = d.get("hfov_deg")
        plane = cls(H, tuple(d["image_size"]), str(d.get("source", "")),
                    None if ref is None else np.array(ref, dtype=np.float64), None if hfov is None else float(hfov))
        if plane.to_image([[0.0, 0.0]])[1][0]:
            return plane
        return cls(-H, plane.image_size, plane.source, plane.reference_px, plane.hfov_deg)  # stored with the other sign


# ------------------------------------------------------------ pose-based fit
HFOV_RANGE = (50.0, 90.0)    # plausible webcam horizontal field of view (deg)
DEFAULT_HFOV = 70.0


def focal_from_hfov(hfov_deg: float, width: int) -> float:
    return (width / 2) / math.tan(math.radians(hfov_deg) / 2)


def hfov_from_focal(f: float, width: int) -> float:
    return 2 * math.degrees(math.atan((width / 2) / f))


def estimate_focal(H: np.ndarray, image_size) -> float | None:
    """Focal length (px) implied by a plane homography, assuming square pixels and the
    principal point at the image centre: the first two columns of K^-1 H must be
    orthogonal and of equal length (Zhang). None if the view gives no answer
    (e.g. looking straight down)."""
    W, Hh = image_size
    T = np.array([[1, 0, -W / 2], [0, 1, -Hh / 2], [0, 0, 1.0]])
    h = T @ H
    a1 = h[0, 0] * h[0, 1] + h[1, 0] * h[1, 1]
    b1 = h[2, 0] * h[2, 1]
    a2 = h[0, 0] ** 2 + h[1, 0] ** 2 - h[0, 1] ** 2 - h[1, 1] ** 2
    b2 = h[2, 0] ** 2 - h[2, 1] ** 2
    den = a1 * a1 + a2 * a2
    if den <= 0:
        return None
    w = -(a1 * b1 + a2 * b2) / den       # = 1 / f^2
    return math.sqrt(1.0 / w) if w > 0 else None


def plane_from_rectangle(table_pts: Sequence, image_px: Sequence, image_size, source: str = "",
                         hfov_deg: float | None = None, max_reproj: float = 0.03) -> TablePlane:
    """Table plane from a rectangle of known size lying on the table (phone marker, A4 sheet...).

    A plain homography through 4 points has 8 free parameters and, from a
    small reference, extrapolates poorly (perspective errors of 10-20 % half a
    metre away). Instead the camera pose is fitted (6 parameters, cv2.solvePnP)
    with a focal length that is either given (hfov_deg) or estimated from the
    same homography and kept within a plausible webcam range. If the pose fit
    doesn't reproduce the corners (within max_reproj of the reference size),
    the plain homography is used.
    """
    table_pts = np.asarray(table_pts, dtype=np.float64).reshape(-1, 2)
    image_px = np.asarray(image_px, dtype=np.float64).reshape(-1, 2)
    W, Hh = int(image_size[0]), int(image_size[1])
    H0 = homography(table_pts, image_px)
    if hfov_deg:
        f = focal_from_hfov(float(hfov_deg), W)
    else:
        f = estimate_focal(H0, (W, Hh))
        lo, hi = (focal_from_hfov(a, W) for a in (HFOV_RANGE[1], HFOV_RANGE[0]))
        f = focal_from_hfov(DEFAULT_HFOV, W) if f is None else min(max(f, lo), hi)
    K = np.array([[f, 0, W / 2], [0, f, Hh / 2], [0, 0, 1.0]])
    centre = table_pts.mean(axis=0)
    obj = np.column_stack([table_pts - centre, np.zeros(len(table_pts))])
    H = H0
    try:
        ok, rvec, tvec = cv2.solvePnP(obj, image_px, K, None, flags=cv2.SOLVEPNP_IPPE)
        if ok:
            R, _ = cv2.Rodrigues(rvec)
            Hp = K @ np.column_stack([R[:, 0], R[:, 1], tvec.ravel()])
            Hp = Hp @ np.array([[1, 0, -centre[0]], [0, 1, -centre[1]], [0, 0, 1.0]])
            proj, w = _h_apply(Hp, table_pts)
            size = math.sqrt(max(polygon_area(image_px), 1.0))
            if np.all(w > 0) and np.max(np.linalg.norm(proj - image_px, axis=1)) <= max_reproj * size:
                H = Hp
    except cv2.error:
        pass
    return TablePlane.from_homography(H, centre, (W, Hh), source, image_px, hfov_from_focal(f, W))


def _jacobian(H: np.ndarray, p: np.ndarray) -> np.ndarray:
    x, y = p
    u = H @ np.array([x, y, 1.0])
    w = u[2]
    return np.array([
        [(H[0, 0] * w - u[0] * H[2, 0]) / w ** 2, (H[0, 1] * w - u[0] * H[2, 1]) / w ** 2],
        [(H[1, 0] * w - u[1] * H[2, 0]) / w ** 2, (H[1, 1] * w - u[1] * H[2, 1]) / w ** 2],
    ])


def _canonical(H0: np.ndarray, centre_img: np.ndarray) -> np.ndarray:
    """Re-express a table frame so it starts at `centre_img`'s table point with +X = screen right,
    +Y = screen down (towards the viewer) and the screen's handedness."""
    Hinv = np.linalg.inv(H0)
    c = centre_img
    probes = np.array([c, c + (10.0, 0.0), c + (0.0, 10.0)])
    tp, _ = _h_apply(Hinv, probes)
    origin = tp[0]
    ex = tp[1] - origin
    ex /= np.linalg.norm(ex)
    ey = np.array([-ex[1], ex[0]])
    if np.dot(ey, tp[2] - origin) < 0:
        ey = -ey
    A = np.vstack([ex, ey])                          # old -> new rotation (maybe with reflection)
    M = np.eye(3)
    M[:2, :2] = A
    M[:2, 2] = -A @ origin
    H = H0 @ np.linalg.inv(M)
    # H[2, 2] is the homogeneous scale of the new origin (the reference centre), positive
    # because H0 was normalised that way and M keeps w; dividing keeps the sign convention.
    return H / H[2, 2]
