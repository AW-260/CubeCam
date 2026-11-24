"""Find one 3x3 cube face in a frame and sample its sticker colours.

Pipeline (about 3-8 ms at 640x360):

1. Edges: Canny on the blurred L*, a* and b* channels, OR-ed together. The
   chroma channels matter for stickerless cubes, where white next to yellow
   has almost no brightness edge.
2. Candidates: regions enclosed by edges that look like one sticker
   (convex, roughly square up to ~55 degrees of tilt, plausible size).
3. Lattice: every candidate is tried as a seed. Its own side vectors give
   the local grid directions; neighbours give the sticker pitch; candidates
   that land on integer lattice positions within a 3x3 window are inliers.
   The seed with the best score (inliers, squareness, size) wins.
4. Homography: lattice (col, row) -> image is fitted to the inliers and
   refined, so perspective from tilting the cube is handled. The lattice is
   relabelled so row 0 is at the top and column 0 on the left of the image.
5. Fallback for faces with no visible sticker borders (a solved face on a
   stickerless cube): a large convex quad that has dark seams where the
   inner grid lines should be is subdivided into 3x3.
6. The face is warped to a flat 96x96 image; a patch in the middle of each
   cell is sampled (median, glare-trimmed) in LAB.

Each sticker gets a visual confidence: was its own outline found ("seen"),
is the patch uniform, is it inside the frame. Stickers hidden by fingers are
usually not seen and not uniform, so they come out low.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from vision.colors import sample_patch_lab

CELL = 32                       # warped cell size in px
PATCH = 0.22                    # half-size of the sampled patch, in cells
_LATTICE = np.array([(c, r) for r in range(3) for c in range(3)], np.float64)


@dataclass
class Candidate:
    center: np.ndarray
    quad: np.ndarray
    area: float
    e1: np.ndarray
    e2: np.ndarray
    squareness: float


@dataclass
class FaceDetection:
    centers: np.ndarray          # (9,2) sticker centres in input-image px, reading order as seen in the image
    corners: np.ndarray          # (4,2) face outline TL, TR, BR, BL
    seen: np.ndarray             # (9,) own outline detected
    lab: np.ndarray              # (9,3) sampled colour
    spread: np.ndarray           # (9,) patch non-uniformity
    confidence: np.ndarray       # (9,) visual confidence 0..1
    pitch: float                 # sticker spacing in px
    method: str                  # "grid" | "outline"
    homography: np.ndarray       # lattice (col,row) -> input px
    warp: np.ndarray             # flat 96x96 BGR face
    image_size: tuple[int, int]  # (w, h) of the input image
    ms: float = 0.0
    occluded: np.ndarray = field(default_factory=lambda: np.zeros(9, bool))  # something covers it (finger)
    debug: dict = field(default_factory=dict)

    @property
    def n_seen(self) -> int:
        return int(self.seen.sum())


def _order_quad(q: np.ndarray) -> np.ndarray:
    """Corners in a consistent rotational order starting top-left-ish."""
    c = q.mean(axis=0)
    ang = np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0])
    q = q[np.argsort(ang)]
    start = int(np.argmin(q.sum(axis=1)))
    return np.roll(q, -start, axis=0)


class FaceDetector:
    def __init__(self, process_width: int = 640, min_seen: int = 4,
                 min_sticker_frac: float = 0.0006, max_sticker_frac: float = 0.06,
                 canny_l: tuple[int, int] = (30, 70), canny_ab: tuple[int, int] = (14, 32),
                 chroma_blur: int = 7, adaptive: bool = False, min_solidity: float = 0.75):
        self.process_width = int(process_width)
        self.min_seen = int(min_seen)
        self.min_sticker_frac = min_sticker_frac
        self.max_sticker_frac = max_sticker_frac
        self.canny_l, self.canny_ab = canny_l, canny_ab
        self.chroma_blur = chroma_blur
        self.adaptive = adaptive
        self.min_solidity = min_solidity
        self._kernel = np.ones((3, 3), np.uint8)

    # ------------------------------------------------------------------ public
    def detect(self, image_bgr: np.ndarray) -> FaceDetection | None:
        t0 = time.perf_counter()
        h0, w0 = image_bgr.shape[:2]
        scale = 1.0
        img = image_bgr
        if self.process_width and w0 > self.process_width:
            scale = self.process_width / w0
            img = cv2.resize(image_bgr, (self.process_width, int(round(h0 * scale))), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        lab8 = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2LAB)
        edges = self._edges(lab8)
        masks = [255 - edges]
        if self.adaptive:
            masks.append(self._bright_regions(lab8[:, :, 0], w))
        cands, bigs = self._candidates(masks, w * h)
        det = self._fit_grid(cands, img, lab8, edges)
        if det is None:
            det = self._fit_outline(bigs, img, lab8, edges)
        if det is None or det.occluded.sum() > 4:
            return None
        if scale != 1.0:
            s = 1.0 / scale
            det.centers = det.centers * s
            det.corners = det.corners * s
            det.pitch *= s
            det.homography = np.diag([s, s, 1.0]) @ det.homography
        det.image_size = (w0, h0)
        det.debug["candidates"] = len(cands)
        det.ms = (time.perf_counter() - t0) * 1000
        return det

    # ------------------------------------------------------------------ steps
    def _edges(self, lab8: np.ndarray) -> np.ndarray:
        L, a, b = cv2.split(lab8)
        e = cv2.Canny(L, *self.canny_l)
        if self.chroma_blur > 1:
            k = (self.chroma_blur, self.chroma_blur)
            a, b = cv2.GaussianBlur(a, k, 0), cv2.GaussianBlur(b, k, 0)
        e |= cv2.Canny(a, *self.canny_ab)
        e |= cv2.Canny(b, *self.canny_ab)
        return cv2.dilate(e, self._kernel, iterations=1)

    def _bright_regions(self, L: np.ndarray, width: int) -> np.ndarray:
        """Pixels brighter than their neighbourhood: stickers against a dark cube body."""
        block = max(15, (width // 14) | 1)
        m = cv2.adaptiveThreshold(L, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, -4)
        return cv2.erode(m, self._kernel, iterations=1)

    def _candidates(self, masks: list[np.ndarray], frame_area: float) -> tuple[list[Candidate], list[np.ndarray]]:
        amin, amax = frame_area * self.min_sticker_frac, frame_area * self.max_sticker_frac
        contours = []
        for m in masks:
            contours.extend(cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0])
        cands: list[Candidate] = []
        bigs: list[np.ndarray] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < amin or area > amax * 12:
                continue
            hull = cv2.convexHull(c)
            harea = cv2.contourArea(hull)
            if harea <= 0 or area / harea < self.min_solidity:
                continue
            peri = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, 0.06 * peri, True).reshape(-1, 2).astype(np.float64)
            if len(approx) == 4:
                quad = approx
            else:
                rect = cv2.minAreaRect(hull)
                rw, rh = rect[1]
                if rw * rh <= 0 or harea / (rw * rh) < 0.80:
                    continue
                quad = cv2.boxPoints(rect).astype(np.float64)
            quad = _order_quad(quad)
            sides = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
            if sides.min() <= 1:
                continue
            squareness = float(sides.min() / sides.max())
            if area > amax:
                if squareness > 0.5:
                    bigs.append(quad)
                continue
            if squareness < 0.55:
                continue
            e1 = ((quad[1] - quad[0]) + (quad[2] - quad[3])) / 2
            e2 = ((quad[3] - quad[0]) + (quad[2] - quad[1])) / 2
            cosang = abs(e1 @ e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
            if cosang > 0.55:
                continue
            cands.append(Candidate(quad.mean(axis=0), quad, area, e1, e2, squareness))
        # De-duplicate nested regions (keep the larger)
        cands.sort(key=lambda k: -k.area)
        kept: list[Candidate] = []
        for k in cands:
            r = 0.35 * np.sqrt(k.area)
            if all(np.linalg.norm(k.center - o.center) > r for o in kept):
                kept.append(k)
        return kept, bigs

    def _fit_grid(self, cands: list[Candidate], img, lab8, edges) -> FaceDetection | None:
        n = len(cands)
        if n < 3:
            return None
        C = np.array([k.center for k in cands])
        A = np.array([k.area for k in cands])
        SQ = np.array([k.squareness for k in cands])
        hyps = []
        covered: set[int] = set()      # members of strong hypotheses: seeding from them repeats work
        order = np.argsort(-A)
        for s in order:
            if s in covered:
                continue
            M = np.column_stack([cands[s].e1, cands[s].e2])
            if abs(np.linalg.det(M)) < 1e-6:
                continue
            uv = (C - C[s]) @ np.linalg.inv(M).T
            ratio = A / A[s]
            area_ok = (ratio > 0.4) & (ratio < 2.5)
            au = np.abs(uv)
            mx, mn = au.max(axis=1), au.min(axis=1)
            neigh = area_ok & (mx > 0.9) & (mx < 1.9) & ((mn < 0.3 * mx) | (np.abs(mx - mn) < 0.3 * mx))
            if not neigh.any():
                continue
            m = mx[neigh]
            p = float(np.median(m[m < m.min() * 1.25]))
            g = uv / p
            gi = np.round(g)
            res = np.linalg.norm(g - gi, axis=1)
            ok = area_ok & (res < 0.28) & (np.abs(gi) <= 2).all(axis=1)
            if ok.sum() < 3:
                continue
            occ = np.zeros((5, 5), bool)                 # occ[row+2, col+2]
            gk = gi[ok].astype(int) + 2
            occ[gk[:, 1], gk[:, 0]] = True
            best_win = None
            for oy in (0, 1, 2):
                for ox in (0, 1, 2):
                    win = occ[oy:oy + 3, ox:ox + 3]
                    cnt = int(win.sum())
                    if cnt < 3:
                        continue
                    wscore = cnt + (2 if win.any(axis=0).all() and win.any(axis=1).all() else 0)
                    if best_win is None or wscore > best_win[0]:
                        best_win = (wscore, ox - 2, oy - 2)
            if best_win is None:
                continue
            _, ox, oy = best_win
            inw = ok & (gi[:, 0] >= ox) & (gi[:, 0] <= ox + 2) & (gi[:, 1] >= oy) & (gi[:, 1] <= oy + 2)
            # one candidate per cell (closest to the lattice point)
            cells: dict[tuple[int, int], int] = {}
            for i in np.where(inw)[0]:
                key = (int(gi[i, 0] - ox), int(gi[i, 1] - oy))
                if key not in cells or res[i] < res[cells[key]]:
                    cells[key] = i
            members = list(cells.values())
            score = len(cells) + 3.0 * float(SQ[members].mean()) + 2.0 * float(A[members].mean() / A.max())                 - 2.0 * float(res[members].mean())
            hyps.append((score, s, p, cells))
            if len(cells) >= 7:
                covered.update(members)
        hyps.sort(key=lambda hp: -hp[0])
        tried: list[frozenset] = []
        for score, s, p, cells in hyps[:6]:
            key = frozenset(cells.values())
            if any(len(key & t) >= len(key) - 1 for t in tried):
                continue
            tried.append(key)
            result = self._refine(cands, C, A, s, p, cells)
            if result is not None:
                H, seen = result
                return self._finish(H, seen, img, lab8, "grid")
        return None

    def _refine(self, cands, C, A, s, p, cells) -> tuple[np.ndarray, np.ndarray] | None:
        """Fit H to a hypothesis, re-match candidates to the predicted lattice, check coverage."""
        lat = np.array(list(cells.keys()), np.float64)
        pts = np.array([C[i] for i in cells.values()], np.float64)
        H = self._fit_h(lat, pts, cands[s], p)
        if H is None:
            return None
        ref_area = float(np.median(A[list(cells.values())]))
        matched: dict[int, int] = {}
        for _ in range(2):
            pred = _project(H, _LATTICE)
            pitch = _pitch(pred)
            matched = {}
            for cell, pp in enumerate(pred):
                d = np.linalg.norm(C - pp, axis=1)
                j = int(np.argmin(d))
                if d[j] < 0.3 * pitch and 0.4 < A[j] / ref_area < 2.5:
                    matched[cell] = j
            if len(matched) < 3:
                return None
            lat = _LATTICE[list(matched.keys())]
            pts = C[list(matched.values())]
            H2 = self._fit_h(lat, pts)
            if H2 is None:
                break
            H = H2
        if len(matched) < self.min_seen or len(set(matched.values())) < len(matched) or not _sane(H):
            return None
        seen = np.zeros(9, bool)
        seen[list(matched.keys())] = True
        g = seen.reshape(3, 3)
        # Every row and column must contain at least one sticker whose own outline was found;
        # otherwise the 3x3 window position is a guess (a row hidden by fingers, a side face, tiles...).
        if not (g.any(axis=0).all() and g.any(axis=1).all()):
            return None
        return _normalize_orientation(H, seen)


    def _fit_h(self, lat: np.ndarray, pts: np.ndarray, seed: Candidate | None = None,
               p: float | None = None) -> np.ndarray | None:
        """Homography lattice -> image; affine when points are too few or collinear."""
        if len(lat) >= 4 and np.unique(lat[:, 0]).size >= 2 and np.unique(lat[:, 1]).size >= 2:
            centred = lat - lat.mean(axis=0)
            if np.linalg.matrix_rank(centred, tol=1e-6) == 2:
                H, _ = cv2.findHomography(lat, pts, 0)
                if H is not None and _sane(H):
                    return H
        if len(lat) >= 3:
            centred = lat - lat.mean(axis=0)
            if np.linalg.matrix_rank(centred, tol=1e-6) == 2:
                X = np.column_stack([lat, np.ones(len(lat))])
                sol, *_ = np.linalg.lstsq(X, pts, rcond=None)
                return np.vstack([sol.T, [0, 0, 1]])
        if seed is not None and p is not None and len(lat) >= 1:
            # collinear inliers: use the seed's own side vectors
            E = np.column_stack([seed.e1, seed.e2]) * p
            i0 = 0
            origin = pts[i0] - E @ lat[i0]
            return np.vstack([np.column_stack([E, origin]), [0, 0, 1]])
        return None

    def _fit_outline(self, bigs: list[np.ndarray], img, lab8, edges) -> FaceDetection | None:
        best = None
        src = np.array([[-0.5, -0.5], [2.5, -0.5], [2.5, 2.5], [-0.5, 2.5]], np.float32)
        for quad in bigs:
            H = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
            H, _ = _normalize_orientation(H, np.zeros(9, bool))
            seam = _seam_score(H, lab8[:, :, 0])
            if seam < 0.55:
                continue
            area = cv2.contourArea(quad.astype(np.float32))
            if best is None or (seam, area) > best[:2]:
                best = (seam, area, H)
        if best is None:
            return None
        det = self._finish(best[2], np.zeros(9, bool), img, lab8, "outline")
        det.confidence *= 0.8
        det.debug["seam"] = best[0]
        return det

    def _finish(self, H: np.ndarray, seen: np.ndarray, img, lab8, method: str) -> FaceDetection:
        h, w = img.shape[:2]
        centers = _project(H, _LATTICE)
        corners = _project(H, np.array([[-0.5, -0.5], [2.5, -0.5], [2.5, 2.5], [-0.5, 2.5]]))
        S = np.array([[1.0 / CELL, 0, -0.5], [0, 1.0 / CELL, -0.5], [0, 0, 1]])
        warp = cv2.warpPerspective(img, H @ S, (3 * CELL, 3 * CELL),
                                   flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        lab = np.zeros((9, 3))
        spread = np.zeros(9)
        conf = np.zeros(9)
        half = int(round(PATCH * CELL))
        for k in range(9):
            r, c = divmod(k, 3)
            cx, cy = c * CELL + CELL // 2, r * CELL + CELL // 2
            patch = warp[cy - half:cy + half, cx - half:cx + half]
            lab[k], spread[k] = sample_patch_lab(patch)
            inside = 0 <= centers[k, 0] < w and 0 <= centers[k, 1] < h
            base = 1.0 if seen[k] else 0.6
            uniform = float(np.clip(1.0 - (spread[k] - 6.0) / 10.0, 0.1, 1.0))
            conf[k] = base * uniform if inside else 0.0
        occluded = np.zeros(9, bool)
        for k in range(9):
            if not seen[k] and conf[k] > 0 and _continues_outward(H, k, lab[k], lab8):
                occluded[k] = True
                conf[k] = 0.0
        return FaceDetection(
            centers=centers, corners=corners, seen=seen, lab=lab, spread=spread, confidence=conf,
            pitch=_pitch(centers), method=method, homography=H, warp=warp, image_size=(w, h),
            occluded=occluded,
        )


# --------------------------------------------------------------------------- helpers
def _lab_at(lab8: np.ndarray, pt: np.ndarray, r: int = 2) -> np.ndarray | None:
    h, w = lab8.shape[:2]
    x, y = int(round(pt[0])), int(round(pt[1]))
    if x - r < 0 or y - r < 0 or x + r >= w or y + r >= h:
        return None
    m = lab8[y - r:y + r + 1, x - r:x + r + 1].reshape(-1, 3).astype(np.float64).mean(axis=0)
    return np.array([m[0] * 100.0 / 255.0, m[1] - 128.0, m[2] - 128.0])


def _continues_outward(H: np.ndarray, k: int, cell_lab: np.ndarray, lab8: np.ndarray) -> bool:
    """True if the cell's colour carries on, without a border, past the edge of the face.

    A real sticker is enclosed by the cube body / seams; a finger lying over the
    face continues outwards. Checked towards each face edge the cell touches
    (all four directions for the centre cell, through its neighbour).
    """
    r, c = divmod(k, 3)
    dirs = []
    if c == 0:
        dirs.append((-1, 0))
    if c == 2:
        dirs.append((1, 0))
    if r == 0:
        dirs.append((0, -1))
    if r == 2:
        dirs.append((0, 1))
    if not dirs:
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dx, dy in dirs:
        # distance from the cell centre to the face edge along this direction, in cells
        edge = (c + 0.5 if dx < 0 else 2.5 - c) if dx else (r + 0.5 if dy < 0 else 2.5 - r)
        ts = list(np.arange(0.5, edge, 0.5)) + [edge, edge + 0.35]
        pts = _project(H, np.array([[c + dx * t, r + dy * t] for t in ts], np.float64))
        same = True
        for pt in pts:
            v = _lab_at(lab8, pt)
            if v is None or np.hypot(*(v[1:] - cell_lab[1:])) > 12 or abs(v[0] - cell_lab[0]) > 20:
                same = False
                break
        if same:
            return True
    return False


def _project(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.column_stack([pts, np.ones(len(pts))]) @ H.T
    return p[:, :2] / p[:, 2:3]


def _pitch(centers: np.ndarray) -> float:
    c = centers.reshape(3, 3, 2)
    d = np.concatenate([np.linalg.norm(c[:, 1:] - c[:, :-1], axis=2).ravel(),
                        np.linalg.norm(c[1:] - c[:-1], axis=2).ravel()])
    return float(d.mean())


def _sane(H: np.ndarray) -> bool:
    """Reject homographies that fold the 3x3 face (points behind the horizon, flips, collapse)."""
    corners = np.array([[-0.5, -0.5], [2.5, -0.5], [2.5, 2.5], [-0.5, 2.5]])
    p = np.column_stack([corners, np.ones(4)]) @ H.T
    if not (np.all(p[:, 2] > 1e-9) or np.all(p[:, 2] < -1e-9)):   # face crosses the horizon
        return False
    q = p[:, :2] / p[:, 2:3]
    cross = []
    for i in range(4):
        a, b, c = q[i], q[(i + 1) % 4], q[(i + 2) % 4]
        cross.append(np.cross(b - a, c - b))
    cross = np.array(cross)
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return False
    sides = np.linalg.norm(np.roll(q, -1, axis=0) - q, axis=1)
    if sides.min() <= 3 or sides.min() / sides.max() < 0.35:
        return False
    # Opposite sides of a face tilted up to ~40 degrees stay within ~1.6x of each other.
    return max(sides[0], sides[2]) / min(sides[0], sides[2]) < 1.6 and         max(sides[1], sides[3]) / min(sides[1], sides[3]) < 1.6


_D4 = [np.array(m, np.float64) for m in (
    [[1, 0], [0, 1]], [[0, -1], [1, 0]], [[-1, 0], [0, -1]], [[0, 1], [-1, 0]],
    [[-1, 0], [0, 1]], [[1, 0], [0, -1]], [[0, 1], [1, 0]], [[0, -1], [-1, 0]],
)]


def _normalize_orientation(H: np.ndarray, seen: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Relabel lattice cells so +col points right and +row points down in the image."""
    best = None
    centre = _project(H, np.array([[1.0, 1.0]]))[0]
    for T in _D4:
        Ti = np.linalg.inv(T)
        dc = _project(H, np.array([[1, 1] + Ti @ [1, 0]]))[0] - centre
        dr = _project(H, np.array([[1, 1] + Ti @ [0, 1]]))[0] - centre
        if dc[0] * dr[1] - dc[1] * dr[0] <= 0:      # keep image handedness (no mirror)
            continue
        score = dc[0] / (np.linalg.norm(dc) + 1e-9) + dr[1] / (np.linalg.norm(dr) + 1e-9)
        if best is None or score > best[0]:
            best = (score, Ti)
    if best is None:
        return H, seen
    Ti = best[1]
    G = np.eye(3)
    G[:2, :2] = Ti
    G[:2, 2] = np.array([1.0, 1.0]) - Ti @ np.array([1.0, 1.0])
    H2 = H @ G
    # reorder `seen`: new cell q came from old cell Ti (q - 1) + 1
    new_seen = np.zeros(9, bool)
    for k in range(9):
        q = _LATTICE[k]
        old = np.rint(Ti @ (q - 1) + 1).astype(int)
        new_seen[k] = seen[old[1] * 3 + old[0]]
    return H2, new_seen


def _seam_score(H: np.ndarray, L: np.ndarray) -> float:
    """Fraction of inner grid-line samples that are darker than the cells beside them."""
    h, w = L.shape
    hits, total = 0, 0
    for t in np.linspace(-0.35, 2.35, 10):
        for line in (0.5, 1.5):
            for vertical in (True, False):
                if vertical:
                    on, a, b = (line, t), (line - 0.3, t), (line + 0.3, t)
                else:
                    on, a, b = (t, line), (t, line - 0.3), (t, line + 0.3)
                pts = _project(H, np.array([on, a, b], np.float64))
                if np.any(pts < 0) or np.any(pts[:, 0] >= w) or np.any(pts[:, 1] >= h):
                    continue
                total += 1
                v = [float(L[int(y), int(x)]) for x, y in pts]
                if v[0] < min(v[1], v[2]) - 6:
                    hits += 1
    return hits / total if total >= 20 else 0.0
