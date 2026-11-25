"""Practice sequences with known turn / rotation counts (stage 3 ground truth).

Syntax (whitespace separated):
    R U R' U2 Rw r M x y' z2     moves (anything cube.parse_moves accepts)
    (R U R' U')x6  or  (...)*6   repeat a group (the x must touch the bracket)
    P2  P1.5                     a deliberate pause of that many seconds

Counting rules (what the measurement is compared against):
    * every layer move is ONE turn, a half turn (R2) included; slices (M) and
      wide moves (Rw) are one turn each;
    * x / y / z are whole-cube rotations, counted separately (x2 = one rotation).

Paced mode: after a lead-in the app highlights one move per beat (tempo in
beats per second) and holds pauses for their length. The cue times are saved as
labels so turning/pause segmentation can be checked, not just counts.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from cube import normalize_move

ROTATIONS = ("x", "y", "z")

DEFAULT_SEQUENCES = [
    "(R U R' U')x6",
    "(y R U R' y' P1)x3",
    "random",
    "(R2 U2)x4",
    "R U R' U' P2 R U R' U' P2 (U R U' R')x2 P2 R U2 R' U2",
    "(x y x' y')x2",
    "(R U R' U')x3 P3 (L' U' L U)x3",
    "(F R U R' U' F')x3",
    "(r U R' U')x4",
]


class SequenceError(ValueError):
    pass


@dataclass(frozen=True)
class Step:
    kind: str            # "turn" | "rotation" | "pause"
    move: str = ""       # canonical move ("R", "U2", "y'") for turns/rotations
    seconds: float = 0.0 # pause length

    @property
    def half(self) -> bool:
        return self.move.endswith("2")

    @property
    def text(self) -> str:
        return f"P{self.seconds:g}" if self.kind == "pause" else self.move


_TOKEN = re.compile(r"\(|\)(?:[x*×](\d+))?|[^\s()]+")


def parse_sequence(text: str) -> list[Step]:
    """Parse a practice sequence into steps (groups expanded)."""
    stack: list[list[Step]] = [[]]
    for m in _TOKEN.finditer(text.strip()):
        tok = m.group(0)
        if tok == "(":
            stack.append([])
        elif tok.startswith(")"):
            if len(stack) == 1:
                raise SequenceError("unbalanced ')'")
            group = stack.pop()
            stack[-1].extend(group * int(m.group(1) or 1))
        elif re.fullmatch(r"[*×]\d+", tok):
            # "(...) *6" with a space: repeat the previous group/step
            raise SequenceError(f"write repeats right after the bracket, e.g. (R U)x6 (got {tok!r})")
        elif re.fullmatch(r"[Pp]\d+(\.\d+)?", tok):
            stack[-1].append(Step("pause", seconds=float(tok[1:])))
        else:
            try:
                mv = normalize_move(tok)
            except ValueError as exc:
                raise SequenceError(str(exc)) from None
            stack[-1].append(Step("rotation" if mv[0] in ROTATIONS else "turn", mv))
    if len(stack) != 1:
        raise SequenceError("unbalanced '('")
    return stack[0]


@dataclass(frozen=True)
class Truth:
    turns: int
    rotations: int
    half_turns: int
    pauses: list[float]

    def to_dict(self) -> dict:
        return asdict(self)


def truth_of(steps: list[Step]) -> Truth:
    return Truth(
        turns=sum(s.kind == "turn" for s in steps),
        rotations=sum(s.kind == "rotation" for s in steps),
        half_turns=sum(s.kind == "turn" and s.half for s in steps),
        pauses=[s.seconds for s in steps if s.kind == "pause"],
    )


def random_sequence(rng: random.Random | None = None, n_turns: int = 15, n_rotations: int = 3,
                    faces: str = "RULDFB", half_prob: float = 0.25) -> str:
    """Random turns (no face twice in a row) with whole-cube rotations mixed in."""
    rng = rng or random.Random()
    moves: list[str] = []
    last = ""
    for _ in range(n_turns):
        f = rng.choice([c for c in faces if c != last])
        last = f
        r = rng.random()
        moves.append(f + ("2" if r < half_prob else "'" if r < (1 + half_prob) / 2 else ""))
    for _ in range(n_rotations):
        rot = rng.choice(ROTATIONS) + rng.choice(["", "'", "", "'", "2"])
        moves.insert(rng.randint(1, len(moves) - 1), rot)
    return " ".join(moves)


def expand_spec(spec: str, rng: random.Random | None = None) -> str:
    """'random' -> a fresh random sequence; anything else is returned as-is."""
    if spec.strip().lower() == "random":
        return random_sequence(rng)
    return spec


# --------------------------------------------------------------------------- paced schedule
@dataclass(frozen=True)
class Cue:
    index: int           # step index
    t0: float            # seconds after the solve start
    t1: float


def schedule(steps: list[Step], tempo: float, lead_in: float) -> list[Cue]:
    """Cue windows: one beat per move, pauses for their length. tempo <= 0 means free pace (no cues)."""
    if tempo <= 0:
        return []
    beat = 1.0 / tempo
    t = lead_in
    cues = []
    for i, s in enumerate(steps):
        d = s.seconds if s.kind == "pause" else beat
        cues.append(Cue(i, round(t, 4), round(t + d, 4)))
        t += d
    return cues


def cue_index(cues: list[Cue], t_rel: float) -> int:
    """Index of the step cued at t_rel (-1 = lead-in, len = finished)."""
    if not cues or t_rel < cues[0].t0:
        return -1
    for c in cues:
        if t_rel < c.t1:
            return c.index
    return cues[-1].index + 1


# --------------------------------------------------------------------------- session
@dataclass
class PracticeSession:
    specs: list[str] = field(default_factory=lambda: list(DEFAULT_SEQUENCES))
    tempo: float = 1.0
    lead_in: float = 2.0
    index: int = 0
    text: str = ""
    steps: list[Step] = field(default_factory=list)
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self):
        self.load(self.index)

    def load(self, index: int) -> None:
        self.index = index % len(self.specs)
        self.text = expand_spec(self.specs[self.index], self.rng)
        self.steps = parse_sequence(self.text)

    def next(self) -> None:
        self.load(self.index + 1)

    @property
    def truth(self) -> Truth:
        return truth_of(self.steps)

    @property
    def cues(self) -> list[Cue]:
        return schedule(self.steps, self.tempo, self.lead_in)

    def labels(self) -> dict:
        tr = self.truth
        return {
            "sequence": self.text,
            "spec": self.specs[self.index],
            "steps": [s.text for s in self.steps],
            "tempo": self.tempo,
            "lead_in": self.lead_in,
            "cues": [[c.t0, c.t1] for c in self.cues],
            "truth": tr.to_dict(),
        }


# --------------------------------------------------------------------------- storage
class PracticeStore:
    """practice_data/practice.json (runs + labels + measured summary) and features/<id>.npz."""

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        self.index_path = self.root / "practice.json"

    def load(self) -> list[dict]:
        if not self.index_path.exists():
            return []
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            return list(data.get("runs", []))
        except (ValueError, OSError):
            return []

    def add(self, run: dict, features: dict | None) -> dict:
        runs = self.load()
        run = dict(run)
        run["id"] = run.get("id") or datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{len(runs) + 1}"
        if features is not None:
            fdir = self.root / "features"
            fdir.mkdir(parents=True, exist_ok=True)
            fname = f"{run['id']}.npz"
            save_features(fdir / fname, features)
            run["features"] = f"features/{fname}"
        runs.append(run)
        self._write(runs)
        return run

    def _write(self, runs: list[dict]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"version": 1, "runs": runs}, indent=1), encoding="utf-8")
        for attempt in range(5):
            try:
                os.replace(tmp, self.index_path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))


def save_features(path: Path, features: dict) -> None:
    """Compressed per-frame motion numbers (quantized, see vision.motion.quantize). No images."""
    from vision.motion import quantize

    np.savez_compressed(
        path,
        t=np.asarray(features["t"], np.float64),          # exact, so re-analysis reproduces the result
        columns=np.array(features["columns"]),
        meta=np.array(json.dumps(features.get("meta", {}))),
        **quantize(features["f"]),
    )


def load_features(path: Path) -> dict:
    from vision.motion import COLUMNS, dequantize

    with np.load(path, allow_pickle=False) as z:
        columns = [str(c) for c in z["columns"]]
        if columns != list(COLUMNS):
            raise ValueError(f"{path.name}: recorded with a different feature layout")
        t = z["t"].astype(np.float64)
        f = dequantize({k: z[k] for k in ("q_flow", "q_vis", "q_other")})
        return {"t": t, "f": f, "columns": columns, "meta": json.loads(str(z["meta"]))}
