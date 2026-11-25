"""Stage 3: move counting, rotations, pauses, practice sequences, storage, worker timing.

The end-to-end tests render animated cubes (vision/synth.py: layer turns and
rotations with motion blur, hands holding the cube, the cube lifted at the start,
a moving object outside the cube region) at 30 fps and run the real extractor +
analysis. Run `pytest -s tests/test_motion.py` to see the reports.
"""
import json
import time
from pathlib import Path

import numpy as np
import pytest

from camera import Frame  # noqa: F401  (documents the frame type the worker receives)
from config import DEFAULTS
from models import Solve
from stats import motion_averages
from storage import SolveStore
from vision.motion import COL, COLUMNS, GRID, NCELL, MotionConfig, MotionWorker
from vision.motion_analysis import AnalysisParams, analyze
from vision.motion_controller import MotionController, summary_lines
from vision.motion_eval import Report, evaluate_practice, evaluate_synthetic_clip, run_synthetic
from vision.practice import (PracticeSession, PracticeStore, SequenceError, cue_index, load_features, parse_sequence,
                             random_sequence, schedule, truth_of)


# --------------------------------------------------------------------------- practice sequences
def test_parse_groups_repeats_pauses_and_counts():
    steps = parse_sequence("(R U R' U')x6")
    assert len(steps) == 24 and truth_of(steps).turns == 24 and truth_of(steps).rotations == 0
    tr = truth_of(parse_sequence("(y R U R' y' P1)*3"))
    assert (tr.turns, tr.rotations, tr.pauses) == (9, 6, [1.0, 1.0, 1.0])
    tr = truth_of(parse_sequence("R2 U2 R' x2 M Rw r P2.5"))
    assert (tr.turns, tr.rotations, tr.half_turns, tr.pauses) == (6, 1, 2, [2.5])
    assert [s.move for s in parse_sequence("r x2 U'")] == ["Rw", "x2", "U'"]
    for bad in ["(R U", "R U)", "R Q", "(R U) *2"]:
        with pytest.raises(SequenceError):
            parse_sequence(bad)


def test_random_sequence_has_requested_counts():
    import random

    for seed in range(20):
        steps = parse_sequence(random_sequence(random.Random(seed), n_turns=15, n_rotations=3))
        tr = truth_of(steps)
        assert tr.turns == 15 and tr.rotations == 3
        turns = [s.move[0] for s in steps if s.kind == "turn"]
        assert all(a != b for a, b in zip(turns, turns[1:]))      # no face twice in a row


def test_paced_schedule_and_cue_index():
    steps = parse_sequence("R U P2 R'")
    cues = schedule(steps, tempo=2.0, lead_in=1.0)
    assert [(c.t0, c.t1) for c in cues] == [(1.0, 1.5), (1.5, 2.0), (2.0, 4.0), (4.0, 4.5)]
    assert cue_index(cues, 0.5) == -1 and cue_index(cues, 1.2) == 0 and cue_index(cues, 3.0) == 2
    assert cue_index(cues, 5.0) == 4
    assert schedule(steps, tempo=0, lead_in=1.0) == []           # free pace: no cues


def test_practice_session_labels():
    ps = PracticeSession(["(R U)x2 P1", "random"], tempo=1.0, lead_in=2.0)
    lab = ps.labels()
    assert lab["truth"]["turns"] == 4 and lab["steps"] == ["R", "U", "R", "U", "P1"]
    assert lab["cues"][0] == [2.0, 3.0] and lab["cues"][-1] == [6.0, 7.0]
    ps.next()
    assert ps.truth.turns == 15 and ps.truth.rotations == 3


# --------------------------------------------------------------------------- storage
def test_old_solves_still_load_and_motion_roundtrips(tmp_path):
    p = tmp_path / "solves.json"
    p.write_text(json.dumps({"version": 1, "solves": [{"time": 12.3, "penalty": "OK", "scramble": "R"}]}))
    store = SolveStore(p)
    solves = store.load()
    assert len(solves) == 1 and solves[0].motion is None and store.warning is None
    m = {"v": 1, "turns": 50, "rotations": 3, "unclassified": 1, "half_turns": 8, "tps": 1.2, "tps_turning": 1.8,
         "paused_time": 14.0, "events": [[0.1, 0.3, "T"]]}
    solves.append(Solve(40.0, motion=m))
    store.save(solves)
    again = SolveStore(p).load()
    assert again[0].motion is None and again[1].motion == m


def test_motion_averages_over_last_n_with_data():
    solves = [Solve(30.0)] + [Solve(30.0, motion={"turns": 40 + i, "tps": 1.0 + i / 10, "tps_turning": 2.0,
                                                  "paused_time": 10.0 + i}) for i in range(12)]
    a5 = motion_averages(solves, 5)
    assert a5.turns == pytest.approx(np.mean([47, 48, 49, 50, 51]))
    assert a5.tps == pytest.approx(np.mean([1.7, 1.8, 1.9, 2.0, 2.1]))
    assert a5.paused == pytest.approx(np.mean([17, 18, 19, 20, 21]))
    assert motion_averages(solves, 12).turns == pytest.approx(np.mean(range(40, 52)))
    assert motion_averages(solves[:4], 5).tps is None


# --------------------------------------------------------------------------- analysis on hand-made rows
def _rows(n, fps=30.0):
    t = np.arange(n) / fps
    f = np.zeros((n, len(COLUMNS)), np.float32)
    f[:, COL["dt"]] = 1 / fps
    f[:, COL["roi_src"]] = 3
    f[:, COL["cv0"]:COL["cv0"] + NCELL] = 1.0
    f[:, COL["Eres"]] = f[:, COL["Erel"]] = 0.03
    return t, f


def _turn(f, i0, i1, col=0, speed=1.5):
    """A layer turn: one outer column of grid cells moves up, the rest of the cube is still."""
    k = np.sin(np.linspace(0, np.pi, i1 - i0))
    for r in range(GRID):
        f[i0:i1, COL[f"fy{r * GRID + col}"]] = -speed * k
    f[i0:i1, COL["Eres"]] = f[i0:i1, COL["Erel"]] = 0.5 * speed * k


def _rotation(f, i0, i1, speed=1.5):
    """A whole-cube rotation: every cell (the centre included) moves sideways, the hands do not."""
    k = np.sin(np.linspace(0, np.pi, i1 - i0))
    f[i0:i1, COL["fx0"]:COL["fx0"] + NCELL] = speed * k[:, None]
    f[i0:i1, COL["gx"]] = speed * k
    f[i0:i1, COL["Eres"]] = f[i0:i1, COL["Erel"]] = 0.2 * speed * k


def test_analysis_counts_turns_rotations_and_pauses():
    t, f = _rows(300)                                   # 10 s
    for i0 in (30, 45, 60):                             # R L R'-ish burst at 1-2 TPS (outer columns)
        _turn(f, i0, i0 + 8, col=0 if i0 != 45 else GRID - 1)
    _rotation(f, 100, 112)                              # y after a 1.1 s stop
    _turn(f, 150, 158, col=GRID - 1)
    _turn(f, 220, 236, col=GRID - 1)                    # a long one: half turn
    res = analyze(t, f, 0.0, 10.0)
    assert (res.turns, res.rotations, res.unclassified) == (5, 1, 0), [(e.kind, e.rot_share) for e in res.events]
    assert res.half_turns == 1
    assert res.turning_time + res.paused_time == pytest.approx(10.0, abs=0.05)
    # pauses: before the first turn (1 s), 67->100, 112->150, 158->220, after 236 (2.1 s)
    assert res.pauses == 5
    # (turn edges below the activity threshold make the measured stops ~0.1 s longer than the gaps)
    assert res.longest_pauses[0][1] == pytest.approx(2.2, abs=0.15) and res.longest_pauses[1][1] == pytest.approx(2.15, abs=0.15)
    assert {round(a) for a, _ in res.longest_pauses[:2]} == {5, 8}      # after the L-turn at 5.2 s and at ~7.9 s
    d = res.to_dict()
    assert d["tps"] == pytest.approx(0.5) and d["tps_turning"] > d["tps"]
    assert len(json.dumps(d)) < 4000


def test_carrying_the_cube_is_not_a_rotation():
    t, f = _rows(120)
    k = np.sin(np.linspace(0, np.pi, 20))
    f[20:40, COL["fy0"]:COL["fy0"] + NCELL] = -1.5 * k[:, None]   # whole cube moves up ...
    f[20:40, COL["gy"]] = -1.5 * k
    f[20:40, COL["vy"]] = -1.5 * k                           # ... together with the hands (lifting it)
    f[20:40, COL["hy"]] = -1.5 * k
    res = analyze(t, f, 0.0, 4.0)
    assert res.turns == res.rotations == res.unclassified == 0
    assert res.paused_time == pytest.approx(4.0)


def test_hands_regripping_alone_is_not_motion():
    t, f = _rows(90)
    f[30:45, COL["vx"]] = 1.5                                  # palms move, cube does not
    f[30:45, COL["hx"]] = 1.5
    assert analyze(t, f, 0.0, 3.0).events == []


def test_no_data_is_reported_as_low_quality():
    res = analyze(np.zeros(0), np.zeros((0, len(COLUMNS))), 0.0, 20.0)
    assert res.turns == 0 and res.quality.startswith("low") and res.paused_time == 20.0


# --------------------------------------------------------------------------- worker timing
def test_worker_never_blocks_and_drops_frames():
    w = MotionWorker(MotionConfig())
    try:
        w.begin()
        big = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), np.uint8)
        worst = 0.0
        for i in range(200):
            t0 = time.perf_counter()
            w.submit(big, i / 30, None, False)
            worst = max(worst, time.perf_counter() - t0)
        time.sleep(0.2)
        out = w.finish(t_end=1e9)
        m = out["meta"]
        assert worst < 0.002, f"submit blocked for {worst * 1000:.2f} ms"
        assert m["dropped"] > 0 and m["processed"] + m["dropped"] == m["submitted"] == 200
        assert out["f"].shape[1] == len(COLUMNS)
        w.submit(big, 99.0, None, False)                           # not collecting: ignored
        assert w.submitted == 200
    finally:
        w.close()


def test_worker_keeps_only_numbers_between_solves():
    w = MotionWorker(MotionConfig())
    try:
        w.begin()
        img = np.zeros((360, 640, 3), np.uint8)
        for i in range(10):
            w.submit(img.copy(), i / 30, None, False)
            time.sleep(0.01)
        out = w.finish(t_end=5 / 30)
        assert all(t <= 5 / 30 + 1e-9 for t in out["t"])
        assert out["f"].dtype == np.float32 and out["f"].nbytes < 10_000
    finally:
        w.close()


# --------------------------------------------------------------------------- synthetic end-to-end
CLIP_KW = dict(width=480, height=270, blur_subframes=2)


@pytest.fixture(scope="module")
def clips():
    specs = {
        "ru": ("(R U R' U')x3", 11),
        "rot": ("(y R U R' y')x2", 12),
        "pause": ("R U R' U' P2 L' U' L U P1.5 R U2 R'", 13),
        "half": ("(R2 U2)x3", 14),
        "still": ("P3", 15),
        "rot_only": ("(x y x' y')x1", 16),
    }
    return {k: run_synthetic(seq, seed, **CLIP_KW) for k, (seq, seed) in specs.items()}


def test_synthetic_turn_counts(clips):
    rep = Report()
    for key in ("ru", "pause", "half"):
        row, _ = evaluate_synthetic_clip(clips[key])
        rep.add(row)
    print("\n" + rep.format("synthetic turn counting"))
    for row in rep.rows:
        assert abs(row.turn_err) <= 1, row
        assert row.true_rotations == 0 and row.rotations == 0
    assert rep.totals()["turn_err_pct"] <= 8.0


def test_synthetic_rotations_are_counted_separately(clips):
    for key in ("rot", "rot_only"):
        row, res = evaluate_synthetic_clip(clips[key])
        print("\n", row, [(round(e.t0, 2), e.kind, round(e.rot_share, 2)) for e in res.events])
        assert row.true_rotations - row.rotations <= 1          # at most one rotation missed / unclassified
        assert abs(row.turn_err) <= 1
        assert row.rotations <= row.true_rotations              # never invents rotations


def test_synthetic_pauses_found_and_segmented(clips):
    row, res = evaluate_synthetic_clip(clips["pause"])
    sc = clips["pause"].script
    print("\n", row, res.longest_pauses)
    assert row.pauses_found == row.pauses_true == 2
    # the two deliberate pauses are among the three longest
    starts = [a for a, _ in res.longest_pauses]
    for a, b in sc.pauses:
        assert any(abs(s - a) < 0.6 for s in starts), (sc.pauses, res.longest_pauses)
    # turning periods cover the moves
    grid = np.arange(0, sc.duration, 0.02)
    moving = sc.active_at(grid)
    measured = np.zeros(len(grid), bool)
    for a, b in res.turning:
        measured |= (grid >= a) & (grid < b)
    assert measured[moving].mean() >= 0.9


def test_synthetic_half_turns_count_once(clips):
    row, res = evaluate_synthetic_clip(clips["half"])
    print("\n", row, [(round(e.t0, 2), round(e.dur, 2), e.humps, e.half) for e in res.events])
    assert abs(row.turn_err) <= 1                               # R2 counts as one turn, not two
    assert row.halves >= row.true_halves // 2                    # at least half of them recognised as half turns


def test_synthetic_still_cube_and_outside_motion_give_nothing(clips):
    """Cube held still for 3 s (lifted in/out, hands jitter, a ball moves outside the region)."""
    _, res = evaluate_synthetic_clip(clips["still"])
    assert res.turns == 0 and res.rotations == 0 and res.unclassified == 0


def test_extractor_is_fast_enough_for_30fps(clips):
    ms = np.concatenate([c.ms for c in clips.values()])
    print(f"\nextractor: median {np.median(ms):.2f} ms, p95 {np.percentile(ms, 95):.2f} ms per frame")
    assert np.median(ms) < 15


# --------------------------------------------------------------------------- controller + practice storage
def _controller(tmp_path, msgs, save="practice"):
    cfg = {k: dict(v) if isinstance(v, dict) else v for k, v in DEFAULTS.items()}
    cfg["motion"] = dict(DEFAULTS["motion"], save_features=save)
    cfg["practice"] = dict(DEFAULTS["practice"], sequences=["(R U R' U')x2 P1.5 (R U R' U')x1"], tempo=2.0,
                           lead_in=0.5)

    def resolve(p: str) -> Path:
        return Path(p) if Path(p).is_absolute() else tmp_path / p

    return MotionController(cfg, resolve, lambda m, s=3.0: msgs.append(m), have_camera=True)


def _feed(ctrl, clip_seq, seed, t0=1000.0):
    from vision.synth import MotionRenderer, make_script

    rng = np.random.default_rng(seed)
    sc = make_script(parse_sequence(clip_seq), rng, lead=0.5)
    r = MotionRenderer(rng, **CLIP_KW)
    ctrl.on_solve_start(t0)
    for fr in r.frames(sc):
        n = ctrl.worker.processed
        ctrl.feed(fr.image, t0 + fr.t, fr.hands, False)
        deadline = time.perf_counter() + 1.0
        while ctrl.worker.processed == n and time.perf_counter() < deadline:   # no drops in the test
            time.sleep(0.001)
    return sc, t0 + sc.duration


def test_practice_run_saved_with_labels_and_small_features(tmp_path):
    msgs = []
    c = _controller(tmp_path, msgs)
    try:
        assert c.on_key("p", idle=True) and c.practicing
        assert not c.on_key("p", idle=False) or c.practicing      # cannot leave practice mid-solve
        sc, t_end = _feed(c, "(R U R' U')x2 P1.5 (R U R' U')x1", seed=21)
        out = c.on_solve_end(1000.0, t_end, {"processed_fps": 30.0})
        assert out is None                                        # practice runs are not solves
        runs = PracticeStore(tmp_path / "practice_data").load()
        assert len(runs) == 1
        run = runs[0]
        assert run["truth"]["turns"] == 12 and run["steps"][8] == "P1.5" and len(run["cues"]) == 13
        assert abs(run["measured"]["turns"] - 12) <= 1
        feat = tmp_path / "practice_data" / run["features"]
        assert feat.exists() and feat.stat().st_size < 100_000
        fe = load_features(feat)
        assert fe["f"].shape[1] == len(COLUMNS) and fe["meta"]["t_end"] == pytest.approx(sc.duration, abs=1e-6)
        # no images anywhere
        assert not [p for p in tmp_path.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".avi", ".mp4")]
        # re-analysis from the stored numbers reproduces the saved result
        rep = evaluate_practice(tmp_path / "practice_data", runs, AnalysisParams())
        assert rep.rows[0].turns == run["measured"]["turns"]
        assert summary_lines(run["measured"], run["truth"])[0].startswith("Moves ")
        v = c.view([], 0.0, solving=False)
        assert v.result_title == "LAST PRACTICE RUN" and v.timeline is not None and v.practice_tokens
    finally:
        c.close()


def test_normal_solve_result_is_small_and_features_not_saved_by_default(tmp_path):
    msgs = []
    c = _controller(tmp_path, msgs)
    try:
        sc, t_end = _feed(c, "R U R' U' R U R'", seed=22)
        out = c.on_solve_end(1000.0, t_end, {"processed_fps": 30.0})
        assert out is not None and abs(out["turns"] - 7) <= 1
        assert len(json.dumps(out)) < 3000
        assert not (tmp_path / "motion_data").exists()
    finally:
        c.close()


def test_abort_discards_motion(tmp_path):
    c = _controller(tmp_path, [])
    try:
        c.on_solve_start(0.0)
        c.feed(np.zeros((270, 480, 3), np.uint8), 0.0, None, False)
        c.on_abort()
        assert not c.worker.collecting and c.last is None
    finally:
        c.close()
