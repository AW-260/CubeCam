"""Stage 3 analysis: per-frame motion rows -> turns, rotations, pauses, TPS, timeline.

Pure numpy, no OpenCV and no clock, so it can be re-run offline on stored rows
(practice runs) when the parameters change. Steps:

1. Activity per frame
       turn part      Erel          one part of the cube moves relative to the rest (and
                                    not just with the hands that carry it)
       rotation part  max(0, min(|g|, |g - v|) - rot_tol)
                      the cube body moves (|g|) and it is not just being carried by
                      the hands (|g - v|); a hand regripping on its own has g ~ 0
   A = turn part + rotation part, lightly smoothed.
2. Hysteresis (on/off) gives active frames; short blips are dropped.
3. Each active burst is split into single moves at dips in A (two turns in quick
   succession, R U, show as two humps). Back-to-back humps where the SAME region
   keeps moving the SAME way are merged again: that is one half turn done as a
   double flick (U2), not two moves.
4. Each event is classified by how much of its activity is whole-cube motion:
   rotation if that share is high, turn if it is low, "unclassified" in between
   (never a guess).
5. Half turns: an event made of two same-direction humps, or clearly longer / farther
   than the solve's typical quarter turn (median over its turns), or longer than
   `half_abs_dur`, is a half turn. It still counts as ONE turn (R2 = 1), it is only
   reported separately; it is never split into two turns.
6. Turning vs paused: a pause is a stop (no turning or rotating) of at least
   `min_pause` (0.5 s); the rest of the solve is turning, including the short
   gaps between consecutive moves and the rotations. Paused time = the sum of
   the pauses, so turning time + paused time = solve time.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from vision.motion import COL, GRID, NCELL, ROI_HANDS

RESULT_VERSION = 1


@dataclass
class AnalysisParams:
    smooth: int = 3              # frames, centred moving average of the activity
    rot_tol: float = 0.35        # region widths / s of "whole cube" motion treated as carrying / noise
    on: float = 0.12             # activity to start an event (region widths / s)
    off: float = 0.07            # ... and to end it
    min_event: float = 0.10      # s, shorter active blips are dropped
    min_peak: float = 0.14       # an event's peak activity must reach this
    valley: float = 0.65         # split a burst where activity dips below valley x the smaller neighbour peak
    min_sep: float = 0.15        # s, minimum distance between two humps to split them
    same_layer: float = 0.75     # cosine similarity of the cell motion pattern to merge humps (double flick)
    merge_gap: float = 0.5       # s, stops shorter than this are part of turning (not paused time)
    min_pause: float = 0.5       # s, stops at least this long are counted as pauses
    rot_hi: float = 0.60         # centre-cell share of an event at/above which it is a rotation
    rot_lo: float = 0.45         # ... at/below which it is a layer turn (in between: unclassified)
    half_units: float = 1.6      # event size in "typical quarter turns" from which it is a half turn
    half_abs_dur: float = 0.36   # s, an event at least this long is a half turn whatever the solve's typical turn
    ref_dur: float = 0.27        # s, typical quarter turn when a solve has too few turns to measure it
    ref_disp: float = 0.30       # region widths, same for displacement
    min_ref_events: int = 4
    min_cell_visible: float = 0.25   # a grid cell needs this visible (not hand) fraction to be used
    cube_cell: float = 0.35          # a cell is "cube" if it moves at least this much (vs the busiest cell)
    carry_speed: float = 0.6         # an event during which the cube is carried at least this fast ...
    carry_peak: float = 0.45         # ... and whose activity peak stays below this is a carrying artefact
    centre_mode: int = 0             # rotation cue: 0 middle cell of the region, 1 adaptive, 2 middle + neighbours
    centre_bias: float = 0.04        # adaptive: prefer the middle of the region when choosing the centre cell
    min_centre_frames: int = 15      # active frames needed to locate the centre (else: middle of the region)

    @classmethod
    def from_dict(cls, d: dict | None) -> "AnalysisParams":
        d = d or {}
        return cls(**{k: type(getattr(cls(), k))(d[k]) for k in cls.__dataclass_fields__ if k in d})


@dataclass
class MotionEvent:
    t0: float
    t1: float
    kind: str = "unclassified"   # turn | rotation | unclassified
    half: bool = False
    peak: float = 0.0
    disp: float = 0.0
    rot_share: float = 0.0
    spread: float = 0.0
    units: float = 1.0
    humps: int = 1               # same-direction activity humps merged into this event
    pattern: np.ndarray | None = field(default=None, repr=False)

    @property
    def dur(self) -> float:
        return self.t1 - self.t0


@dataclass
class MotionResult:
    duration: float
    turns: int
    rotations: int
    unclassified: int
    half_turns: int
    turning_time: float
    paused_time: float
    pauses: int                                  # paused periods >= min_pause
    longest_pauses: list[tuple[float, float]]    # (start s after solve start, length s), longest first
    turning: list[tuple[float, float]]           # turning periods (s after solve start)
    events: list[MotionEvent]
    coverage: float                              # fraction of the solve with motion rows
    hands_seen: float                            # fraction of rows with both hands tracked
    quality: str                                 # ok | low (+ reason)
    meta: dict = field(default_factory=dict)

    @property
    def tps(self) -> float | None:
        return self.turns / self.duration if self.duration > 0 else None

    @property
    def tps_turning(self) -> float | None:
        return self.turns / self.turning_time if self.turning_time > 0 else None

    def to_dict(self) -> dict:
        """Compact form stored with each solve in solves.json (a few KB at most)."""
        r = lambda x, n=2: None if x is None else round(float(x), n)  # noqa: E731
        return {
            "v": RESULT_VERSION,
            "turns": self.turns,
            "rotations": self.rotations,
            "unclassified": self.unclassified,
            "half_turns": self.half_turns,
            "tps": r(self.tps),
            "tps_turning": r(self.tps_turning),
            "tps_incl_unclassified": r((self.turns + self.unclassified) / self.duration if self.duration > 0 else None),
            "duration": r(self.duration),
            "turning_time": r(self.turning_time),
            "paused_time": r(self.paused_time),
            "pauses": self.pauses,
            "longest_pauses": [[r(a), r(b)] for a, b in self.longest_pauses],
            "turning": [[r(a), r(b)] for a, b in self.turning],
            "events": [[r(e.t0), r(e.t1), {"turn": "T", "rotation": "R"}.get(e.kind, "U") + ("2" if e.half else "")]
                       for e in self.events],
            "coverage": r(self.coverage),
            "hands_seen": r(self.hands_seen),
            "quality": self.quality,
            "meta": self.meta,
        }


# --------------------------------------------------------------------------- helpers
def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index runs of True."""
    if not len(mask):
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(x) < k:
        return x.copy()
    pad = k // 2
    xp = np.pad(x, (pad, k - 1 - pad), mode="edge")
    return np.convolve(xp, np.ones(k) / k, mode="valid")


def _hysteresis(a: np.ndarray, on: float, off: float) -> np.ndarray:
    out = np.zeros(len(a), bool)
    state = False
    for i, v in enumerate(a):
        state = v >= on if not state else v > off
        out[i] = state
    return out


INNER = [r * GRID + c for r in range(1, GRID - 1) for c in range(1, GRID - 1)]   # candidate centre cells
MIDDLE = (GRID * GRID) // 2


def find_cube_cells(speed: np.ndarray, ok: np.ndarray, active: np.ndarray, p: AnalysisParams) -> tuple[np.ndarray, int]:
    """(cube cell mask, centre cell) for one solve.

    Over a whole solve every part of the cube moves at some point, background inside the
    region never does: cube cells are those that move. The cube's centre is the inner cube
    cell that moves LEAST relative to the busiest cell while the cube is active, because
    R/L/U/D/F turns move the outer rows and columns and leave the middle of the face still;
    only rotations (and slices) move it. This finds the centre even when the region (from
    the palms) is off-centre on the cube."""
    ncell = speed.shape[1]
    if active.sum() < p.min_centre_frames:
        return np.ones(ncell, bool), MIDDLE
    sp = speed[active]
    hi = np.percentile(sp, 90, axis=0)
    cube = hi >= p.cube_cell * max(hi.max(), 1e-6)
    vis = ok[active].mean(axis=0)
    smax = np.maximum(sp.max(axis=1, keepdims=True), 1e-6)
    rel = np.median(sp / smax, axis=0)
    best, best_score = MIDDLE, np.inf
    for i in INNER:
        if not cube[i] or vis[i] < 0.5:
            continue
        r, c = divmod(i, GRID)
        dist = abs(r - GRID // 2) + abs(c - GRID // 2)
        score = rel[i] + p.centre_bias * dist
        if score < best_score:
            best, best_score = i, score
    return cube, best


def signals(t: np.ndarray, F: np.ndarray, p: AnalysisParams) -> dict:
    F = F.astype(np.float64)
    n = len(F)
    g = F[:, [COL["gx"], COL["gy"]]]
    # carrying motion: the palms when both hands are tracked, else the visible hand pixels
    v = np.where((F[:, COL["roi_src"]] >= ROI_HANDS)[:, None], F[:, [COL["vx"], COL["vy"]]], F[:, [COL["hx"], COL["hy"]]])
    h = F[:, [COL["hx"], COL["hy"]]]
    c = np.stack([F[:, COL["fx0"]:COL["fx0"] + NCELL], F[:, COL["fy0"]:COL["fy0"] + NCELL]], axis=-1)  # (N, cells, 2)
    ok = F[:, COL["cv0"]:COL["cv0"] + NCELL] >= p.min_cell_visible
    # per-cell speed that is neither "still in the camera" nor "carried by the hands" (palms or hand pixels)
    speed = np.minimum(np.linalg.norm(c, axis=2), np.linalg.norm(c - v[:, None, :], axis=2))
    speed = np.where(ok, np.minimum(speed, np.linalg.norm(c - h[:, None, :], axis=2)), 0.0)
    smax = speed.max(axis=1)
    # layer motion: not still, not carried by the hands
    turn = F[:, COL["Erel"]]
    # how fast the cube body is being carried (hands AND the middle of the cube moving together);
    # used to drop weak events that are only edge artefacts of lifting / putting down the cube
    carried = np.minimum(np.linalg.norm(v, axis=1), np.where(ok[:, MIDDLE], np.linalg.norm(c[:, MIDDLE], axis=1), 0.0))
    dt = np.diff(t, prepend=t[0] - (np.median(np.diff(t)) if len(t) > 1 else 1 / 30))
    dt = np.clip(dt, 1e-3, 0.2)
    # pass 1: where is the cube and its centre (activity from layer motion alone)
    active1 = _hysteresis(_smooth(turn, p.smooth), p.on, p.off)
    cube, centre_cell = find_cube_cells(speed, ok, active1, p)
    # pass 2: whole-cube motion = the cube's centre moving (relative to the camera and the hands)
    if p.centre_mode == 0:                         # the middle cell of the region
        centre_cell = MIDDLE
        centre = speed[:, MIDDLE]
    elif p.centre_mode == 2:                       # middle cell and its 4 neighbours, the slowest 3 averaged
        plus = [MIDDLE, MIDDLE - 1, MIDDLE + 1, MIDDLE - GRID, MIDDLE + GRID]
        sp = np.where(ok[:, plus], speed[:, plus], np.nan)
        with np.errstate(all="ignore"):
            centre = np.nan_to_num(np.nanmean(np.sort(sp, axis=1)[:, :3], axis=1))
        centre_cell = MIDDLE
    else:                                          # adaptive (find_cube_cells)
        centre = speed[:, centre_cell]
    rot = np.maximum(0.0, centre - p.rot_tol)
    a = _smooth(turn + rot, p.smooth)
    res = np.where(ok, np.linalg.norm(c - g[:, None, :], axis=2), 0.0)
    top3 = np.sort(res, axis=1)[:, -3:].mean(axis=1)
    pattern = np.where(ok[..., None], c - g[:, None, :], 0.0).reshape(n, 2 * NCELL)   # motion relative to the body
    return {"a": a, "turn": turn, "rot": rot, "body": centre, "smax": smax, "speed": speed, "top3": top3,
            "pattern": pattern, "dt": dt, "cube_cells": cube, "centre_cell": centre_cell, "carried": carried}


def _split_burst(a: np.ndarray, t: np.ndarray, i0: int, i1: int, p: AnalysisParams) -> list[tuple[int, int]]:
    """Split [i0, i1) at activity dips between humps."""
    seg = a[i0:i1]
    if len(seg) < 3:
        return [(i0, i1)]
    # local maxima (plateaus count once)
    peaks = [i for i in range(len(seg)) if (i == 0 or seg[i] > seg[i - 1]) and (i == len(seg) - 1 or seg[i] >= seg[i + 1])]
    peaks = [i for i in peaks if seg[i] >= p.min_peak * 0.8] or [int(np.argmax(seg))]
    cuts = []
    cur = peaks[0]
    for nxt in peaks[1:]:
        lo = cur + int(np.argmin(seg[cur:nxt + 1]))
        if seg[lo] < p.valley * min(seg[cur], seg[nxt]) and t[i0 + nxt] - t[i0 + cur] >= p.min_sep:
            cuts.append(lo)
            cur = nxt
        elif seg[nxt] > seg[cur]:
            cur = nxt
    bounds = [0] + cuts + [len(seg)]
    return [(i0 + bounds[k], i0 + bounds[k + 1]) for k in range(len(bounds) - 1) if bounds[k + 1] > bounds[k]]


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-9 and nb > 1e-9 else 0.0


def _event(t, s, i0: int, i1: int) -> MotionEvent:
    sl = slice(i0, i1)
    dt = s["dt"][sl]
    body_i = float((s["body"][sl] * dt).sum())
    all_i = float((s["smax"][sl] * dt).sum())
    e = MotionEvent(float(t[i0] - s["dt"][i0] / 2), float(t[i1 - 1]))
    e.peak = float(s["a"][sl].max())
    e.rot_share = body_i / all_i if all_i > 1e-6 else 0.0
    # spread: share of the cube's cells that moved, relative to the fastest one (cube cells only)
    disp_cells = (s["speed"][sl] * dt[:, None]).sum(axis=0)
    cube = s["cube_cells"]
    if cube.sum() >= 3 and disp_cells[cube].max() > 1e-6:
        e.spread = float((disp_cells[cube] >= 0.45 * disp_cells[cube].max()).mean())
    else:
        e.spread = 0.0
    e.pattern = (s["pattern"][sl] * dt[:, None]).sum(axis=0)
    e._turn_disp = float((s["top3"][sl] * dt).sum())          # type: ignore[attr-defined]
    e._rot_disp = float((s["body"][sl] * dt).sum())           # type: ignore[attr-defined]
    e._i = (i0, i1)                                           # type: ignore[attr-defined]
    return e


# --------------------------------------------------------------------------- main entry
def analyze(t: np.ndarray, F: np.ndarray, t_start: float, t_end: float,
            params: AnalysisParams | None = None, meta: dict | None = None) -> MotionResult:
    """Analyse motion rows (t: capture times, F: rows with vision.motion.COLUMNS) of one solve."""
    p = params or AnalysisParams()
    t = np.asarray(t, np.float64)
    F = np.asarray(F, np.float32)
    keep = (t >= t_start - 1e-9) & (t <= t_end + 1e-9)
    t, F = t[keep], F[keep]
    duration = max(0.0, t_end - t_start)
    if len(t) < 5 or duration <= 0:
        return MotionResult(duration, 0, 0, 0, 0, 0.0, duration, 1 if duration >= p.min_pause else 0,
                            [(0.0, round(duration, 3))] if duration >= p.min_pause else [], [], [],
                            coverage=0.0, hands_seen=0.0, quality="low: no motion data", meta=meta or {})
    s = signals(t, F, p)
    active = _hysteresis(s["a"], p.on, p.off)

    # events
    events: list[MotionEvent] = []
    for i0, i1 in _runs(active):
        if t[i1 - 1] - t[i0] + s["dt"][i0] < p.min_event or s["a"][i0:i1].max() < p.min_peak:
            active[i0:i1] = False
            continue
        parts = [_event(t, s, a0, a1) for a0, a1 in _split_burst(s["a"], t, i0, i1, p)]
        merged = [parts[0]]
        for e in parts[1:]:
            prev = merged[-1]
            if _cos(prev.pattern, e.pattern) >= p.same_layer and (prev.rot_share < p.rot_hi) == (e.rot_share < p.rot_hi):
                # the same region keeps moving the same way: one half turn (or y2), not two moves
                humps = prev.humps + e.humps
                merged[-1] = _event(t, s, prev._i[0], e._i[1])
                merged[-1].humps = humps
            else:
                merged.append(e)
        for e in merged:
            a0, a1 = e._i
            if s["carried"][a0:a1].mean() >= p.carry_speed and e.peak < p.carry_peak and e.rot_share < p.rot_hi:
                active[a0:a1] = False          # lifting / putting the cube down, not a move
            else:
                events.append(e)

    # classification
    for e in events:
        e.kind = "rotation" if e.rot_share >= p.rot_hi else "turn" if e.rot_share <= p.rot_lo else "unclassified"
        e.disp = e._rot_disp if e.kind == "rotation" else e._turn_disp

    # half turns, relative to this solve's typical quarter move
    for kind in ("turn", "rotation", "unclassified"):
        evs = [e for e in events if e.kind == kind]
        same = [e for e in events if e.kind == (kind if kind != "unclassified" else "turn")]
        if len(same) >= p.min_ref_events:
            ref_dur = float(np.median([e.dur for e in same]))
            ref_disp = float(np.median([e.disp for e in same]))
        else:
            ref_dur, ref_disp = p.ref_dur, p.ref_disp
        for e in evs:
            e.units = 0.5 * (e.dur / max(ref_dur, 1e-3) + e.disp / max(ref_disp, 1e-6))
            e.half = e.humps >= 2 or e.units >= p.half_units or e.dur >= p.half_abs_dur

    # turning vs paused: a pause is a stop of at least min_pause; everything else is turning
    spans = [(float(t[i0] - s["dt"][i0] / 2 - t_start), float(t[i1 - 1] - t_start)) for i0, i1 in _runs(active)]
    gaps, prev_end = [], 0.0
    for a, b in spans:
        if a > prev_end:
            gaps.append((prev_end, a - prev_end))
        prev_end = max(prev_end, b)
    if duration > prev_end:
        gaps.append((prev_end, duration - prev_end))
    pause_list = [g for g in gaps if g[1] >= max(p.min_pause, p.merge_gap)]
    turning, cur = [], 0.0
    for a, d in pause_list:
        if a > cur:
            turning.append((cur, a))
        cur = a + d
    if duration > cur:
        turning.append((cur, duration))
    if not spans:
        turning = []
    turning_time = sum(b - a for a, b in turning)
    long_pauses = sorted(pause_list, key=lambda g: -g[1])
    paused_time = max(0.0, duration - turning_time)

    for e in events:
        e.t0 -= t_start
        e.t1 -= t_start
    n_turn = sum(e.kind == "turn" for e in events)
    expected = duration * (1.0 / float(np.median(s["dt"]))) if len(t) > 1 else 1.0
    coverage = min(1.0, len(t) / max(1.0, expected))
    hands_seen = float((F[:, COL["roi_src"]] >= ROI_HANDS).mean())
    reasons = []
    if coverage < 0.6:
        reasons.append(f"only {coverage:.0%} of frames analysed")
    if hands_seen < 0.5:
        reasons.append(f"hands tracked in {hands_seen:.0%} of frames")
    return MotionResult(
        duration=duration,
        turns=n_turn,
        rotations=sum(e.kind == "rotation" for e in events),
        unclassified=sum(e.kind == "unclassified" for e in events),
        half_turns=sum(e.kind == "turn" and e.half for e in events),
        turning_time=turning_time,
        paused_time=paused_time,
        pauses=len(long_pauses),
        longest_pauses=[(round(a, 3), round(d, 3)) for a, d in long_pauses[:3]],
        turning=turning,
        events=events,
        coverage=coverage,
        hands_seen=hands_seen,
        quality="ok" if not reasons else "low: " + ", ".join(reasons),
        meta=dict(meta or {}),
    )
