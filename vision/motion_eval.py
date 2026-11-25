"""Accuracy of stage-3 move counting against known labels.

Two kinds of labels:
  * practice runs (practice_data/practice.json): true turn / rotation / half-turn
    counts, plus, for paced runs, the cue times of every move and pause;
  * synthetic clips (vision/synth.py): exact start/end of every move.

Used by tools/evaluate_counts.py and tests/test_motion.py.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from vision.motion import MotionConfig, MotionFeatureExtractor
from vision.motion_analysis import AnalysisParams, MotionResult, analyze
from vision.practice import expand_spec, load_features, parse_sequence


# --------------------------------------------------------------------------- counts
@dataclass
class CountRow:
    name: str
    turns: int
    true_turns: int
    rotations: int
    true_rotations: int
    unclassified: int
    halves: int
    true_halves: int
    pauses_found: int | None = None      # deliberate pauses detected (paced runs / synthetic)
    pauses_true: int | None = None
    seg_agree: float | None = None       # frame agreement moving-block vs deliberate pause
    lag: float | None = None
    extra: str = ""

    @property
    def turn_err(self) -> int:
        return self.turns - self.true_turns

    @property
    def rot_err(self) -> int:
        return self.rotations - self.true_rotations


@dataclass
class Report:
    rows: list[CountRow] = field(default_factory=list)

    def add(self, row: CountRow) -> None:
        self.rows.append(row)

    @property
    def n(self) -> int:
        return len(self.rows)

    def totals(self) -> dict:
        r = self.rows
        tt = sum(x.true_turns for x in r)
        tr = sum(x.true_rotations for x in r)
        th = sum(x.true_halves for x in r)
        pf = [x for x in r if x.pauses_true]
        sa = [x.seg_agree for x in r if x.seg_agree is not None]
        return {
            "runs": len(r),
            "true_turns": tt,
            "turn_abs_err": sum(abs(x.turn_err) for x in r),
            "turn_err_pct": 100.0 * sum(abs(x.turn_err) for x in r) / max(1, tt),
            "turn_bias_pct": 100.0 * sum(x.turn_err for x in r) / max(1, tt),
            "turns_exact_or_1": sum(abs(x.turn_err) <= 1 for x in r),
            "true_rotations": tr,
            "rot_abs_err": sum(abs(x.rot_err) for x in r),
            "unclassified": sum(x.unclassified for x in r),
            "true_halves": th,
            "half_abs_err": sum(abs(x.halves - x.true_halves) for x in r),
            "pauses_found": sum(x.pauses_found or 0 for x in pf),
            "pauses_true": sum(x.pauses_true or 0 for x in pf),
            "seg_agree": float(np.mean(sa)) if sa else None,
        }

    def format(self, title: str = "") -> str:
        lines = [title] if title else []
        lines.append(f"{'run':28s} {'turns':>9s} {'rot':>7s} {'unc':>4s} {'half':>7s} {'pauses':>7s} {'seg':>5s}  notes")
        for x in self.rows:
            pz = "" if x.pauses_true is None else f"{x.pauses_found}/{x.pauses_true}"
            seg = "" if x.seg_agree is None else f"{x.seg_agree:.0%}"
            lines.append(f"{x.name[:28]:28s} {x.turns:4d}/{x.true_turns:<4d} {x.rotations:3d}/{x.true_rotations:<3d} "
                         f"{x.unclassified:4d} {x.halves:3d}/{x.true_halves:<3d} {pz:>7s} {seg:>5s}  {x.extra}")
        t = self.totals()
        lines.append(
            f"TOTAL {t['runs']} runs: turns |err| {t['turn_abs_err']}/{t['true_turns']} = {t['turn_err_pct']:.1f}% "
            f"(bias {t['turn_bias_pct']:+.1f}%, {t['turns_exact_or_1']}/{t['runs']} runs within +-1), "
            f"rotations |err| {t['rot_abs_err']}/{t['true_rotations']}, unclassified {t['unclassified']}, "
            f"half turns |err| {t['half_abs_err']}/{t['true_halves']}"
            + (f", deliberate pauses found {t['pauses_found']}/{t['pauses_true']}" if t["pauses_true"] else "")
            + (f", moving/paused agreement {t['seg_agree']:.0%}" if t["seg_agree"] is not None else ""))
        return "\n".join(lines)


# --------------------------------------------------------------------------- segmentation vs cues
def _timeline(intervals, grid: np.ndarray) -> np.ndarray:
    out = np.zeros(len(grid), bool)
    for a, b in intervals:
        out |= (grid >= a) & (grid < b)
    return out


def segmentation_vs_cues(result: dict, steps: list[str], cues: list[list[float]], max_lag: float = 1.2,
                         min_pause: float = 1.0) -> tuple[int, int, float | None, float]:
    """Deliberate pauses found, deliberate pauses, frame agreement, estimated lag.

    Labels from the cue schedule: a "moving block" runs from the first cue of a run of
    moves to the end of its last cue; a deliberate pause (P token >= min_pause) is paused.
    The performer lags the cues by a reaction time, estimated as the shift (0..max_lag)
    that best aligns the measured turning periods with the moving blocks."""
    if not cues:
        return 0, 0, None, 0.0
    dur = float(result["duration"])
    blocks, pauses, cur = [], [], None
    for tok, (a, b) in zip(steps, cues):
        if tok.startswith("P"):
            if cur is not None:
                blocks.append(cur)
                cur = None
            if b - a >= min_pause:
                pauses.append((a, b))
        else:
            cur = (cur[0], b) if cur is not None else (a, b)
    if cur is not None:
        blocks.append(cur)
    grid = np.arange(0.0, dur, 0.02)
    measured = _timeline(result.get("turning", []), grid)
    best = (-1.0, 0.0)
    for lag in np.arange(0.0, max_lag + 1e-9, 0.05):
        lab_move = _timeline([(a + lag, b + lag) for a, b in blocks], grid)
        lab_pause = _timeline([(a + lag + 0.25, b + lag) for a, b in pauses], grid)
        care = lab_move | lab_pause
        if not care.any():
            continue
        agree = float(((measured & lab_move) | (~measured & lab_pause))[care].mean())
        if agree > best[0]:
            best = (agree, float(lag))
    agree, lag = best
    found = 0
    for a, b in pauses:
        a, b = a + lag + 0.25, b + lag
        win = (grid >= a) & (grid < b)
        if win.any() and (~measured[win]).mean() >= 0.6:
            found += 1
    return found, len(pauses), (agree if agree >= 0 else None), lag


# --------------------------------------------------------------------------- practice runs
def evaluate_practice(root: str | Path, runs: list[dict], params: AnalysisParams | None = None,
                      reanalyse: bool = True) -> Report:
    """Score stored practice runs; with features saved, re-run the analysis with `params`."""
    rep = Report()
    for run in runs:
        truth = run.get("truth") or {}
        measured = run.get("measured") or {}
        note = ""
        if reanalyse and run.get("features"):
            path = Path(root) / run["features"]
            if path.exists():
                fe = load_features(path)
                t_end = float(fe["meta"].get("t_end", run.get("time", 0.0)))
                measured = analyze(fe["t"], fe["f"], 0.0, t_end, params, measured.get("meta")).to_dict()
            else:
                note = "features missing, stored result used"
        if not measured:
            continue
        found, total, agree, lag = segmentation_vs_cues(measured, run.get("steps", []), run.get("cues", []))
        rep.add(CountRow(
            name=f"{run.get('id', '?')} {run.get('spec', '')}",
            turns=measured["turns"], true_turns=truth.get("turns", 0),
            rotations=measured["rotations"], true_rotations=truth.get("rotations", 0),
            unclassified=measured["unclassified"], halves=measured["half_turns"],
            true_halves=truth.get("half_turns", 0),
            pauses_found=found if total else None, pauses_true=total if total else None,
            seg_agree=agree, lag=lag if total or agree is not None else None,
            extra=(f"lag {lag:.2f}s " if run.get("cues") else "free pace ")
            + (f"tempo {run.get('tempo')} " if run.get("cues") else "") + note))
    return rep


# --------------------------------------------------------------------------- synthetic clips
@dataclass
class SyntheticClip:
    name: str
    script: object             # vision.synth.MotionScript
    t: np.ndarray
    f: np.ndarray
    ms: list[float]


def run_synthetic(seq: str, seed: int, cfg: MotionConfig | None = None, **render_kw) -> SyntheticClip:
    """Render a sequence (vision/synth.py) and extract motion rows, as the worker would."""
    from vision.synth import MotionRenderer, make_script

    import time as _time

    rng = np.random.default_rng(seed)
    seq = expand_spec(seq, random.Random(seed))
    script = make_script(parse_sequence(seq), rng)
    r = MotionRenderer(rng, **render_kw)
    ex = MotionFeatureExtractor(cfg or MotionConfig())
    ts, rows, ms = [], [], []
    for fr in r.frames(script):
        t0 = _time.perf_counter()
        row = ex.process(fr.image, fr.t, fr.hands, False)
        ms.append((_time.perf_counter() - t0) * 1000)
        if row is not None:
            ts.append(fr.t)
            rows.append(row)
    return SyntheticClip(seq, script, np.array(ts), np.array(rows, np.float32).reshape(len(rows), -1), ms)


def match_events(result: MotionResult, script) -> dict:
    """Greedy time-overlap matching of measured events to the true moves."""
    used, pairs = set(), []
    for m in script.moves:
        best, bo = None, 0.0
        for j, e in enumerate(result.events):
            o = min(m.t1, e.t1) - max(m.t0, e.t0)
            if j not in used and o > bo:
                best, bo = j, o
        if best is not None:
            used.add(best)
        pairs.append((m, None if best is None else result.events[best]))
    extra = [e for j, e in enumerate(result.events) if j not in used]
    conf: dict[tuple[str, str], int] = {}
    for m, e in pairs:
        key = (m.kind, "missed" if e is None else e.kind)
        conf[key] = conf.get(key, 0) + 1
    for e in extra:
        conf[("none", e.kind)] = conf.get(("none", e.kind), 0) + 1
    return {"pairs": pairs, "extra": extra, "confusion": conf}


def evaluate_synthetic_clip(clip: SyntheticClip, params: AnalysisParams | None = None) -> tuple[CountRow, MotionResult]:
    res = analyze(clip.t, clip.f, 0.0, clip.script.duration, params)
    sc = clip.script
    grid = np.arange(0.0, sc.duration, 0.02)
    measured = _timeline([(a, b) for a, b in res.turning], grid)
    found = 0
    for a, b in sc.pauses:
        win = (grid >= a + 0.15) & (grid < b - 0.15)
        if win.any() and (~measured[win]).mean() >= 0.8:
            found += 1
    m = match_events(res, sc)
    missed = [p[0].move for p in m["pairs"] if p[1] is None]
    row = CountRow(clip.name[:28], res.turns, sc.turns, res.rotations, sc.rotations, res.unclassified,
                   res.half_turns, sc.half_turns, found if sc.pauses else None, len(sc.pauses) if sc.pauses else None,
                   extra=(f"missed {' '.join(missed[:6])} " if missed else "") + (f"extra {len(m['extra'])}" if m["extra"] else ""))
    return row, res
