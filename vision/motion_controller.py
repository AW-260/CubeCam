"""Stage-3 app logic: motion worker per solve, analysis, practice mode, on-screen view.

main.py tells it when a solve starts/stops and hands it each frame while
SOLVING (a reference plus the hand landmarks it already has, nothing is copied).
"""
from __future__ import annotations

import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable

from stats import motion_averages
from ui import MotionView
from vision.motion import MotionConfig, MotionWorker, roundtrip
from vision.motion_analysis import AnalysisParams, analyze
from vision.practice import (DEFAULT_SEQUENCES, PracticeSession, PracticeStore, SequenceError, cue_index,
                             save_features)


def summary_lines(m: dict, truth: dict | None = None) -> list[str]:
    """Human-readable lines for one result dict (MotionResult.to_dict())."""
    def tps(x):
        return "-" if x is None else f"{x:.2f}"

    turns = f"{m['turns']}"
    if truth is not None:
        turns += f" (true {truth['turns']})"
    rot = f"{m['rotations']}" + (f" (true {truth['rotations']})" if truth is not None else "")
    half = f"{m['half_turns']}" + (f" (true {truth['half_turns']})" if truth is not None else "")
    lines = [
        f"Moves {turns} | rotations {rot} | unclassified {m['unclassified']}",
        f"TPS {tps(m['tps'])} overall | {tps(m['tps_turning'])} while turning | half turns {half}",
        f"Paused {m['paused_time']:.1f} s of {m['duration']:.1f} s in {m['pauses']} pauses (>= 0.5 s)",
    ]
    lp = m.get("longest_pauses") or []
    if lp:
        lines.append("Longest: " + ", ".join(f"[{i + 1}] {d:.1f} s at {a:.1f} s" for i, (a, d) in enumerate(lp)))
    if m.get("quality", "ok") != "ok":
        lines.append("Motion data " + m["quality"])
    return lines


class MotionController:
    def __init__(self, cfg: dict, resolve: Callable[[str], Path], say: Callable[[str, float], None],
                 have_camera: bool, beep: Callable[[int, int], None] | None = None):
        mc, pc = cfg["motion"], cfg["practice"]
        self.say = say
        self.beep = beep
        self.enabled = bool(mc["enabled"]) and have_camera
        self.mcfg = MotionConfig.from_dict(mc)
        self.params = AnalysisParams.from_dict(mc.get("analysis"))
        self.save_mode = str(mc.get("save_features", "practice"))
        self.features_dir = resolve(str(mc.get("features_path", "motion_data")))
        self.worker: MotionWorker | None = MotionWorker(self.mcfg) if self.enabled else None
        specs = pc.get("sequences") or list(DEFAULT_SEQUENCES)
        self.practice_specs = [str(s) for s in specs]
        self.tempo = float(pc.get("tempo", 1.0))
        self.lead_in = float(pc.get("lead_in", 2.0))
        self.metronome = bool(pc.get("metronome", False))
        self.store = PracticeStore(resolve(str(pc.get("path", "practice_data"))))
        self.practice: PracticeSession | None = None
        self.show_panel = True
        self.last: dict | None = None             # last result dict
        self.last_truth: dict | None = None       # practice ground truth for the last result
        self.last_practice_id: str | None = None
        self._t_start: float | None = None
        self._last_cue = -2
        self.startup_messages: list[str] = []
        if not self.enabled and bool(mc["enabled"]):
            self.startup_messages.append("Move counting needs a camera")

    # ------------------------------------------------------------------ timer events
    def on_solve_start(self, t: float) -> None:
        self._t_start = t
        self._last_cue = -2
        self.last = None
        self.last_truth = None
        if self.worker is not None:
            self.worker.begin()

    def feed(self, image, t: float, hands, mirrored: bool) -> None:
        """Called by main.py for every frame while SOLVING. O(1), never blocks."""
        if self.worker is not None and self.worker.collecting:
            self.worker.submit(image, t, hands, mirrored)

    def tick(self, t: float) -> None:
        """Per-frame while solving: metronome beep for practice cues."""
        if self.practice is None or self._t_start is None or not self.metronome or self.beep is None:
            return
        idx = cue_index(self.practice.cues, t - self._t_start)
        if idx != self._last_cue and 0 <= idx < len(self.practice.steps):
            if self.practice.steps[idx].kind != "pause":
                self.beep(1000, 40)
        self._last_cue = idx

    def on_solve_end(self, t_start: float, t_end: float, info: dict) -> dict | None:
        """Analyse the finished solve. Returns the result dict to store with the solve (None in practice)."""
        self._t_start = None
        if self.worker is None:
            return None
        data = self.worker.finish(t_end)
        data["f"] = roundtrip(data["f"])        # analyse exactly what is saved, so results can be reproduced
        meta = {"frames": int(len(data["t"])), "submitted": data["meta"]["submitted"],
                "dropped": data["meta"]["dropped"], "worker_ms": _r(data["meta"]["ms_median"]),
                "processed_fps": info.get("processed_fps")}
        res = analyze(data["t"], data["f"], t_start, t_end, self.params, meta)
        out = res.to_dict()
        self.last = out
        feats = {"t": data["t"] - t_start, "f": data["f"], "columns": data["columns"],
                 "meta": {"t_start": 0.0, "t_end": t_end - t_start, **data["meta"]}}
        if self.practice is not None:
            labels = self.practice.labels()
            self.last_truth = labels["truth"]
            run = {"timestamp": datetime.now().isoformat(timespec="seconds"), "time": round(t_end - t_start, 3),
                   **labels, "measured": out, "analysis_params": asdict(self.params),
                   "motion_config": {k: v for k, v in asdict(self.mcfg).items()}, "processed_fps": info.get("processed_fps"),
                   "camera": info.get("camera")}
            try:
                saved = self.store.add(run, feats if self.save_mode in ("practice", "all") else None)
                self.last_practice_id = saved["id"]
            except OSError as exc:
                self.say(f"Could not save practice run: {exc}", 5.0)
            if self.practice.specs[self.practice.index].strip().lower() == "random":
                self.practice.load(self.practice.index)      # a fresh random sequence each time
            return None
        if self.save_mode == "all":
            try:
                self.features_dir.mkdir(parents=True, exist_ok=True)
                name = datetime.now().strftime("%Y%m%d-%H%M%S") + ".npz"
                save_features(self.features_dir / name, feats)
                out["features"] = f"{self.features_dir.name}/{name}"
            except OSError as exc:
                self.say(f"Could not save motion features: {exc}", 5.0)
        return out

    def on_abort(self) -> None:
        self._t_start = None
        if self.worker is not None and self.worker.collecting:
            self.worker.finish()

    # ------------------------------------------------------------------ keys
    @property
    def practicing(self) -> bool:
        return self.practice is not None

    def on_key(self, ch: str, idle: bool) -> bool:
        if ch == "p":
            if not idle:
                self.say("Practice mode can only be switched while the timer is idle")
            elif self.worker is None:
                self.say("Practice mode needs a camera (and motion.enabled: true)")
            elif self.practice is None:
                try:
                    self.practice = PracticeSession(self.practice_specs, self.tempo, self.lead_in,
                                                    rng=random.Random())
                except SequenceError as exc:
                    self.say(f"Bad practice sequence: {exc}", 5.0)
                    return True
                self.last, self.last_truth = None, None
                self.say("PRACTICE mode: perform the sequence shown, it is not added to your solves. "
                         "N next, +/- tempo, P exit.", 6.0)
            else:
                self.practice = None
                self.say("Practice mode off")
            return True
        if ch == "t":
            self.show_panel = not self.show_panel
            return True
        if self.practice is not None and idle:
            if ch == "n":
                self.practice.next()
                self.last, self.last_truth = None, None
                return True
            if ch in ("+", "=", "-", "_"):
                step = 0.25 if ch in ("+", "=") else -0.25
                self.tempo = max(0.0, min(4.0, round(self.practice.tempo + step, 2)))
                self.practice.tempo = self.tempo
                self.say(f"Practice tempo: {'free pace (no cues)' if self.tempo == 0 else f'{self.tempo:g} moves/s'}")
                return True
        return False

    # ------------------------------------------------------------------ view
    def view(self, solves, t: float, solving: bool) -> MotionView:
        v = MotionView(show=self.show_panel)
        for n in (5, 12):
            a = motion_averages(solves, n)
            v.averages.append((n, a.tps, a.turns, a.paused))
        pr = self.practice
        if pr is not None:
            v.practice_tokens = [s.text for s in pr.steps]
            if solving and self._t_start is not None:
                rel = t - self._t_start
                v.practice_index = cue_index(pr.cues, rel) if pr.cues else -2
                if pr.cues and rel < pr.lead_in:
                    v.countdown = f"{pr.lead_in - rel:.1f}"
            else:
                v.practice_index = -2
            tr = pr.truth
            pace = "free pace" if pr.tempo <= 0 else f"tempo {pr.tempo:g}/s"
            v.title = f"PRACTICE {pr.index + 1}/{len(pr.specs)}  ({pace})"
            v.lines = [f"True: {tr.turns} turns ({tr.half_turns} half), {tr.rotations} rotations"
                       + (f", {len(tr.pauses)} pauses" if tr.pauses else ""),
                       "Rest hands to arm, lift, perform it, hands back to stop.",
                       "N next   +/- tempo   P exit practice   T hide panel"]
        if self.last is not None and not solving:
            v.result_title = "LAST PRACTICE RUN" if self.last_truth is not None else "LAST SOLVE"
            v.result_lines = summary_lines(self.last, self.last_truth)
            v.timeline = self.last
        return v

    def status_line(self) -> str | None:
        if self.worker is None:
            return None
        w = self.worker
        ms = f"{sorted(w.ms)[len(w.ms) // 2]:.1f} ms/frame" if w.ms else "-"
        state = "measuring" if w.collecting else "ready"
        extra = ""
        if self.last is not None and self.last.get("meta"):
            m = self.last["meta"]
            extra = f" | last solve: {m.get('frames', 0)} frames analysed, {m.get('dropped', 0)} dropped"
        return f"move counting {state} ({ms}){extra}" + (" | PRACTICE MODE" if self.practice else "")

    def close(self) -> None:
        if self.worker is not None:
            self.worker.close()


def _r(x, n=2):
    return None if x is None else round(float(x), n)
