"""Sticker colour classification in CIELAB.

Each of the six colours is modelled by a mean and a per-channel spread in
(L*, a*, b*). A sample is assigned to the colour with the smallest weighted,
spread-normalised distance; L* gets a low weight because it changes with
exposure, shadows and glare far more than the chroma channels do.

Red vs orange is the classic failure (they differ mostly in hue and a little in
lightness). It gets its own 1-D test along the red->orange axis, and a sample
near the boundary is reported as ambiguous ("R/O?") instead of silently guessed.

Without calibration a default model (typical webcam sticker colours) is used.
Calibration (vision.calibration in main.py) learns this cube in this light and
is saved to calibration.json.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

COLORS = "WYROGB"
# Typical webcam BGR values for standard stickers under neutral light.
DEFAULT_BGR = {
    "W": (225, 225, 225),
    "Y": (45, 215, 235),
    "R": (45, 35, 195),
    "O": (25, 120, 250),
    "G": (85, 175, 35),
    "B": (185, 85, 25),
}
DEFAULT_STD = np.array([22.0, 13.0, 13.0])
CHANNEL_WEIGHT = np.array([0.35, 1.0, 1.0])  # L*, a*, b*
STD_FLOOR = np.array([7.0, 3.5, 3.5])
FORMAT_VERSION = 1


def bgr_to_lab(bgr) -> np.ndarray:
    """BGR (0..255, any shape ending in 3) -> float LAB with L* 0..100, a*/b* signed."""
    a = np.asarray(bgr, dtype=np.float32)
    lab = cv2.cvtColor((a.reshape(-1, 1, 3) / 255.0).astype(np.float32), cv2.COLOR_BGR2LAB)
    return lab.reshape(a.shape).astype(np.float64)


def lab_to_bgr(lab) -> tuple[int, int, int]:
    arr = np.asarray(lab, dtype=np.float32).reshape(1, 1, 3)
    bgr = cv2.cvtColor(arr, cv2.COLOR_LAB2BGR).reshape(3)
    return tuple(int(v) for v in np.clip(bgr * 255, 0, 255))


@dataclass
class ColorResult:
    label: str                       # best colour letter
    confidence: float                # 0..1
    second: str                      # runner-up
    distance: float                  # normalised distance to the best colour
    ambiguous: tuple[str, str] | None = None  # e.g. ("R", "O") when too close to call

    @property
    def text(self) -> str:
        return f"{self.ambiguous[0]}/{self.ambiguous[1]}?" if self.ambiguous else self.label


@dataclass
class ColorModel:
    means: dict[str, np.ndarray]
    stds: dict[str, np.ndarray]
    calibrated: bool = False
    info: dict = field(default_factory=dict)
    ambiguity_margin: float = 0.18   # (d2-d1)/(d2+d1) below this -> ambiguous
    outlier_distance: float = 4.5    # normalised distance beyond which a sample fits no colour

    # ---------------------------------------------------------------- creation
    @classmethod
    def default(cls) -> "ColorModel":
        means = {c: bgr_to_lab(DEFAULT_BGR[c]) for c in COLORS}
        stds = {c: DEFAULT_STD.copy() for c in COLORS}
        # White/yellow vary mostly in lightness; red/orange are close, keep them tighter in chroma.
        stds["W"] = np.array([25.0, 9.0, 11.0])
        return cls(means, stds, calibrated=False, info={"source": "defaults"})

    @classmethod
    def from_samples(cls, samples: dict[str, list], base: "ColorModel | None" = None) -> "ColorModel":
        """Build a model from LAB samples per colour; colours without samples keep `base` values."""
        base = base or cls.default()
        means = {c: v.copy() for c, v in base.means.items()}
        stds = {c: v.copy() for c, v in base.stds.items()}
        counts = {}
        for c, vals in samples.items():
            arr = np.asarray(vals, dtype=np.float64).reshape(-1, 3)
            if len(arr) == 0:
                continue
            means[c] = np.median(arr, axis=0)
            spread = arr.std(axis=0) * 1.6 if len(arr) >= 3 else np.zeros(3)
            stds[c] = np.maximum(spread, STD_FLOOR)
            counts[c] = int(len(arr))
        calibrated = all(c in counts for c in COLORS)
        return cls(means, stds, calibrated=calibrated,
                   info={"source": "calibration", "samples": counts, "created": time.strftime("%Y-%m-%d %H:%M:%S")})

    # ---------------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {
            "version": FORMAT_VERSION,
            "calibrated": self.calibrated,
            "info": self.info,
            "colors": {c: {"lab_mean": [round(float(x), 3) for x in self.means[c]],
                           "lab_std": [round(float(x), 3) for x in self.stds[c]]} for c in COLORS},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ColorModel":
        cols = d["colors"]
        means = {c: np.array(cols[c]["lab_mean"], dtype=np.float64) for c in COLORS}
        stds = {c: np.maximum(np.array(cols[c]["lab_std"], dtype=np.float64), STD_FLOOR) for c in COLORS}
        return cls(means, stds, calibrated=bool(d.get("calibrated", True)), info=dict(d.get("info", {})))

    def save(self, path: str | os.PathLike) -> None:
        p = Path(path)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        os.replace(tmp, p)

    @classmethod
    def load(cls, path: str | os.PathLike) -> tuple["ColorModel", str | None]:
        """(model, warning). Missing or broken file -> default model."""
        p = Path(path)
        if not p.exists():
            return cls.default(), None
        try:
            return cls.from_dict(json.loads(p.read_text(encoding="utf-8"))), None
        except (ValueError, KeyError, TypeError) as exc:
            return cls.default(), f"could not read {p.name} ({exc}); using default colours"

    # ---------------------------------------------------------------- classify
    def _matrices(self) -> tuple[np.ndarray, np.ndarray]:
        mu = np.stack([self.means[c] for c in COLORS])
        sd = np.stack([self.stds[c] for c in COLORS])
        return mu, sd

    def distances(self, labs: np.ndarray) -> np.ndarray:
        """(N,3) LAB -> (N,6) weighted normalised distances."""
        labs = np.asarray(labs, dtype=np.float64).reshape(-1, 3)
        mu, sd = self._matrices()
        z = (labs[:, None, :] - mu[None]) / sd[None]
        return np.sqrt((z * z * CHANNEL_WEIGHT).sum(axis=2) / CHANNEL_WEIGHT.sum() * 3)

    def red_orange_t(self, lab: np.ndarray) -> tuple[float, float]:
        """Position of a sample along red(0) -> orange(1) and the decision boundary t0."""
        w = np.sqrt(CHANNEL_WEIGHT)
        r, o = self.means["R"] * w, self.means["O"] * w
        axis = o - r
        n2 = float(axis @ axis) or 1.0
        t = float((np.asarray(lab) * w - r) @ axis / n2)
        unit = axis / math.sqrt(n2)
        sr = float(np.abs(self.stds["R"] * w) @ np.abs(unit))
        so = float(np.abs(self.stds["O"] * w) @ np.abs(unit))
        t0 = sr / (sr + so) if sr + so > 0 else 0.5
        return t, t0

    def red_orange_hue_t(self, lab: np.ndarray) -> tuple[float, float]:
        """Hue-angle position between red (0) and orange (1); boundary 0.5."""
        hr = math.atan2(self.means["R"][2], self.means["R"][1])
        ho = math.atan2(self.means["O"][2], self.means["O"][1])
        h = math.atan2(float(lab[2]), float(lab[1]))
        return (h - hr) / (ho - hr if ho != hr else 1e-6), 0.5

    def classify_many(self, labs) -> list[ColorResult]:
        labs = np.asarray(labs, dtype=np.float64).reshape(-1, 3)
        d = self.distances(labs)
        out = []
        for i in range(len(labs)):
            order = np.argsort(d[i])
            b, s = int(order[0]), int(order[1])
            d1, d2 = float(d[i, b]), float(d[i, s])
            label, second = COLORS[b], COLORS[s]
            margin = (d2 - d1) / (d2 + d1 + 1e-9)
            conf = min(1.0, margin / (2 * self.ambiguity_margin))
            ambiguous = None
            if {label, second} == {"R", "O"}:
                if self.calibrated:
                    # Calibrated: same light as the reference, so lightness helps. Decide along the
                    # red->orange axis in weighted LAB.
                    t, t0 = self.red_orange_t(labs[i])
                    band = 0.22
                else:
                    # Defaults: exposure/gamma make lightness unreliable (dim orange looks as dark as
                    # red), so decide on hue angle alone, with a wide "unsure" band.
                    t, t0 = self.red_orange_hue_t(labs[i])
                    band = 0.32
                label, second = ("O", "R") if t > t0 else ("R", "O")
                conf = min(conf, min(1.0, abs(t - t0) / band))
                if abs(t - t0) < band * 0.5:
                    ambiguous = ("R", "O")
            if margin < self.ambiguity_margin * 0.5 and ambiguous is None:
                ambiguous = tuple(sorted((label, second), key=COLORS.index))
            if d1 > self.outlier_distance:  # far from every colour (skin, glare, background)
                conf *= (self.outlier_distance / d1) ** 2
            out.append(ColorResult(label, float(max(0.0, min(1.0, conf))), second, d1, ambiguous))
        return out

    def classify(self, lab) -> ColorResult:
        return self.classify_many(np.asarray(lab).reshape(1, 3))[0]


# Broad skin model: fingers covering a sticker should be reported, not classified.
SKIN_LAB = np.array([64.0, 15.0, 20.0])
SKIN_STD = np.array([22.0, 6.0, 7.0])


def read_stickers(labs, visual_conf, model: ColorModel, occluded=None) -> list[ColorResult]:
    """Classify sampled stickers and fold in the detector's visual confidence.

    A patch that looks more like skin than like any sticker colour is labelled
    '?' with confidence 0 (a finger is covering it).
    """
    labs = np.asarray(labs, dtype=np.float64).reshape(-1, 3)
    res = model.classify_many(labs)
    zs = (labs - SKIN_LAB) / SKIN_STD
    d_skin = np.sqrt((zs * zs * CHANNEL_WEIGHT).sum(axis=1) / CHANNEL_WEIGHT.sum() * 3)
    out = []
    occ = np.zeros(len(labs), bool) if occluded is None else np.asarray(occluded, dtype=bool)
    for r, v, ds, oc in zip(res, np.asarray(visual_conf, dtype=np.float64), d_skin, occ):
        if oc or (ds < 1.6 and ds < 0.8 * r.distance):
            out.append(ColorResult("?", 0.0, r.label, r.distance, None))
        else:
            out.append(ColorResult(r.label, float(r.confidence * v), r.second, r.distance, r.ambiguous))
    return out


def plausible_face(labs, seen, model: ColorModel, min_fraction: float = 0.75) -> bool:
    """Do the stickers whose outlines were found look like cube colours at all?

    Rejects grids of keyboard keys, tiles, book spines... in colours no cube has.
    """
    labs = np.asarray(labs, dtype=np.float64).reshape(-1, 3)
    seen = np.asarray(seen, dtype=bool)
    if not seen.any():
        return True
    d = model.distances(labs[seen]).min(axis=1)
    return float((d < model.outlier_distance * 0.8).mean()) >= min_fraction


# --------------------------------------------------------------------------- sampling
def sample_patch_lab(patch_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """Robust LAB colour of a sticker patch: median, ignoring the brightest 15% (glare).

    Returns (lab, spread) where spread is a ΔE-like measure of non-uniformity.
    """
    px = patch_bgr.reshape(-1, 3).astype(np.float32)
    if len(px) == 0:
        return np.zeros(3), 99.0
    lab = cv2.cvtColor((px / 255.0).reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    if len(lab) >= 10:
        keep = lab[:, 0] <= np.percentile(lab[:, 0], 85)
        if keep.sum() >= 5:
            lab = lab[keep]
    med = np.median(lab, axis=0)
    dev = np.abs(lab - med)
    spread = float(np.mean(dev[:, 0]) * 0.5 + np.mean(dev[:, 1]) + np.mean(dev[:, 2]))
    return med.astype(np.float64), spread


# --------------------------------------------------------------------------- balanced assignment
def hungarian(cost: np.ndarray) -> np.ndarray:
    """Minimum-cost perfect assignment for a square matrix. Returns col index per row."""
    cost = np.asarray(cost, dtype=np.float64)
    n = cost.shape[0]
    INF = float("inf")
    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=int)     # p[j] = row matched to column j (1-based)
    way = np.zeros(n + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, INF)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            cur = cost[i0 - 1] - u[i0] - v[1:]
            free = ~used[1:]
            better = free & (cur < minv[1:])
            minv[1:][better] = cur[better]
            way[1:][better] = j0
            cand = np.where(free, minv[1:], INF)
            j1 = int(np.argmin(cand)) + 1
            delta = cand[j1 - 1]
            u[p[used]] += delta
            v[used] -= delta
            minv[1:][free] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    out = np.zeros(n, dtype=int)
    for j in range(1, n + 1):
        out[p[j] - 1] = j - 1
    return out


def balanced_assign(labs, model: ColorModel, centre_idx: list[int] | None = None,
                    per_color: int = 9, iterations: int = 3) -> list[str]:
    """Assign exactly `per_color` stickers to each colour (full-cube scan).

    The six centres seed the cluster means (blended with the model), then a
    minimum-cost assignment with 9 slots per colour is solved and the means are
    refined. This fixes most red/orange and white/yellow confusions because the
    whole cube is decided jointly instead of sticker by sticker.
    """
    labs = np.asarray(labs, dtype=np.float64).reshape(-1, 3)
    n = len(labs)
    if n != per_color * len(COLORS):
        raise ValueError(f"need {per_color * len(COLORS)} samples")
    mu = np.stack([model.means[c] for c in COLORS]).copy()
    sd = np.stack([model.stds[c] for c in COLORS])
    if centre_idx:
        cres = model.classify_many(labs[centre_idx])
        cl = [r.label for r in cres]
        if len(set(cl)) == 6:
            for k, i in zip(cl, centre_idx):
                mu[COLORS.index(k)] = 0.5 * mu[COLORS.index(k)] + 0.5 * labs[i]
    labels = None
    for _ in range(iterations):
        z = (labs[:, None, :] - mu[None]) / sd[None]
        d = np.sqrt((z * z * CHANNEL_WEIGHT).sum(axis=2))
        cost = np.repeat(d, per_color, axis=1)          # (54, 54): 9 slots per colour
        cols = hungarian(cost)
        labels = cols // per_color
        for k in range(len(COLORS)):
            mu[k] = 0.5 * mu[k] + 0.5 * labs[labels == k].mean(axis=0)
    return [COLORS[k] for k in labels]
