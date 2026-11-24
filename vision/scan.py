"""Scramble verification and guided full-cube scanning (pure logic, no OpenCV).

A *face read* is 9 colour letters (W Y R O G B, or '?' when unknown) in the
face's own reading order, i.e. as you see the face when looking straight at it
(already un-mirrored), plus a confidence per sticker.

Scramble check: the centre colour says which face it is (WCA orientation:
white top, green front, so centres never move during a scramble). The face may
be held at any of 4 rotations, so the read is compared with the expected face
at all 4 and the best fit is used.

Guided scan: the user shows each face with a given colour on top, in the
orientation of the standard net. Reads are put together into a 54-facelet
colour string and validated with cube.validate. If that fails, the 4^6 face
rotations are searched for a unique valid cube (someone held a face sideways).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Sequence

import cube
from cube import COLOR_NAMES, FACES, WCA_SCHEME, rotate_face_cw, to_colors

MIN_CONF = 0.5

# Scan order and how to hold each face: (face, centre colour, colour that must be on top).
SCAN_ORDER = [("F", "G", "W"), ("R", "R", "W"), ("B", "B", "W"), ("L", "O", "W"), ("U", "W", "B"), ("D", "Y", "G")]
CENTRE_TO_FACE = {c: f for f, c in WCA_SCHEME.items()}


@dataclass
class FaceRead:
    colors: list[str]                    # 9 letters or '?'
    confidence: list[float]              # 9 values 0..1
    ambiguous: list[tuple | None] = field(default_factory=lambda: [None] * 9)
    lab: list | None = None              # optional (9,3) LAB samples for joint re-assignment

    @property
    def centre(self) -> str:
        return self.colors[4]

    def confident(self, k: int, min_conf: float = MIN_CONF) -> bool:
        return self.colors[k] != "?" and self.confidence[k] >= min_conf and self.ambiguous[k] is None

    def rotated(self, times: int) -> "FaceRead":
        return FaceRead(rotate_face_cw(list(self.colors), times), rotate_face_cw(list(self.confidence), times),
                        rotate_face_cw(list(self.ambiguous), times),
                        None if self.lab is None else rotate_face_cw(list(self.lab), times))


# --------------------------------------------------------------------------- scramble check
@dataclass
class FaceCheck:
    face: str | None                 # U R F D L B, None if the centre could not be read
    status: str                      # "ok" | "mismatch" | "unsure" | "no_centre"
    rotation: int = 0                # quarter turns cw applied to the expected face to match the view
    sticker_status: list[str] = field(default_factory=lambda: ["unseen"] * 9)  # ok|mismatch|uncertain|unseen
    expected: list[str] = field(default_factory=list)   # expected colours in the read's orientation
    seen: int = 0

    @property
    def mismatches(self) -> int:
        return self.sticker_status.count("mismatch")

    @property
    def uncertain(self) -> int:
        return self.sticker_status.count("uncertain")

    def describe(self) -> str:
        if self.face is None:
            return "centre not readable - show the face straight on"
        name = COLOR_NAMES[WCA_SCHEME[self.face]]
        if self.status == "ok":
            return f"{name} face OK"
        if self.status == "mismatch":
            return f"{name} face: {self.mismatches} sticker(s) differ from the scramble"
        return f"{name} face: {self.uncertain} sticker(s) unclear, {self.seen} seen"


def check_face(read: FaceRead, expected_colors54: str, min_conf: float = MIN_CONF, min_seen: int = 6) -> FaceCheck:
    """Compare one face read against the expected (scrambled) cube in colour letters."""
    if not read.confident(4, min_conf * 0.8):
        return FaceCheck(None, "no_centre")
    face = None
    for f in FACES:
        if expected_colors54[cube.CENTERS[f]] == read.centre:
            face = f
    if face is None:
        return FaceCheck(None, "no_centre")
    exp9 = cube.face(expected_colors54, face)
    best = None
    for rot in range(4):
        e = rotate_face_cw(exp9, rot)
        statuses = []
        for k in range(9):
            got, want = read.colors[k], e[k]
            if got == "?" or read.confidence[k] <= 0.05:
                statuses.append("unseen")
            elif read.confident(k, min_conf):
                statuses.append("ok" if got == want else "mismatch")
            else:
                amb = read.ambiguous[k]
                if got == want or (amb is not None and want in amb):
                    statuses.append("ok" if got == want and amb is None else "uncertain")
                else:
                    statuses.append("uncertain")
        key = (statuses.count("mismatch"), statuses.count("uncertain"), -statuses.count("ok"))
        if best is None or key < best[0]:
            best = (key, rot, statuses, list(e))
    _, rot, statuses, e = best
    seen = sum(s != "unseen" for s in statuses)
    if statuses.count("mismatch"):
        status = "mismatch"
    elif statuses.count("uncertain") or seen < min_seen:
        status = "unsure"
    else:
        status = "ok"
    return FaceCheck(face, status, rot, statuses, e, seen)


class ScrambleCheck:
    """Collects face checks for the current scramble until it changes."""

    def __init__(self):
        self.scramble: str | None = None
        self.expected: str | None = None
        self.faces: dict[str, FaceCheck] = {}
        self.last: FaceCheck | None = None

    def set_scramble(self, scramble: str) -> None:
        if scramble != self.scramble:
            self.scramble = scramble
            self.expected = to_colors(cube.scrambled_state(scramble))
            self.faces.clear()
            self.last = None

    def add(self, read: FaceRead) -> FaceCheck:
        assert self.expected is not None, "set_scramble first"
        chk = check_face(read, self.expected)
        self.last = chk
        if chk.face is not None and chk.status != "no_centre":
            prev = self.faces.get(chk.face)
            # a clean read replaces an unsure one; a mismatch is kept until a clean read replaces it
            if prev is None or chk.status in ("ok", "mismatch") or prev.status == "unsure":
                self.faces[chk.face] = chk
        return chk

    @property
    def verdict(self) -> str:
        if any(c.status == "mismatch" for c in self.faces.values()):
            return "mismatch"
        if not self.faces:
            return "none"
        return "ok" if all(c.status == "ok" for c in self.faces.values()) else "unsure"

    def summary(self) -> str:
        n_ok = sum(c.status == "ok" for c in self.faces.values())
        bad = [COLOR_NAMES[WCA_SCHEME[f]] for f, c in self.faces.items() if c.status == "mismatch"]
        if bad:
            return f"SCRAMBLE MISMATCH on {', '.join(bad)} face" + ("s" if len(bad) > 1 else "")
        if not self.faces:
            return "Scramble check: show a face to the camera"
        return f"Scramble OK on {n_ok}/6 faces" + ("" if n_ok == len(self.faces) else " (some unclear)")


# --------------------------------------------------------------------------- guided scan
@dataclass
class ScanResult:
    ok: bool
    colors54: str | None
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    reassigned: list[int] = field(default_factory=list)   # facelet indices changed by the joint assignment


def instruction(face: str) -> str:
    _, centre, top = next(t for t in SCAN_ORDER if t[0] == face)
    return f"Show the {COLOR_NAMES[centre].upper()}-centre face, {COLOR_NAMES[top].upper()} centre on top"


class ScanSession:
    def __init__(self):
        self.reads: dict[str, FaceRead] = {}
        self.target: str | None = SCAN_ORDER[0][0]
        self.result: ScanResult | None = None

    @property
    def complete(self) -> bool:
        return len(self.reads) == 6

    def next_missing(self) -> str | None:
        return next((f for f, _, _ in SCAN_ORDER if f not in self.reads), None)

    def instruction(self) -> str:
        if self.target is None:
            return "All faces scanned"
        return instruction(self.target)

    def offer(self, read: FaceRead, min_conf: float = MIN_CONF, min_confident: int = 7) -> str | None:
        """Offer a still face read. Returns a message, or None when it was not accepted silently."""
        if self.target is None:
            return None
        want_centre = WCA_SCHEME[self.target]
        if not read.confident(4, min_conf * 0.8):
            return None
        if read.centre != want_centre:
            f = CENTRE_TO_FACE.get(read.centre)
            if f is not None and f not in self.reads:
                return f"That is the {COLOR_NAMES[read.centre]} face - {self.instruction().lower()}"
            return None
        good = sum(read.confident(k, min_conf) or (read.colors[k] != "?" and read.ambiguous[k] is not None)
                   for k in range(9))
        if good < min_confident:
            return f"{COLOR_NAMES[want_centre]} face: only {good}/9 stickers readable - hold still, fingers off"
        self.reads[self.target] = read
        self.result = None
        name = COLOR_NAMES[want_centre]
        self.target = self.next_missing()
        return f"Captured {name} face" + ("" if self.target is None else f". {self.instruction()}")

    def rescan(self, face: str) -> None:
        self.reads.pop(face, None)
        self.result = None
        self.target = face

    def colors54(self) -> list[str]:
        out = []
        for f in FACES:
            r = self.reads.get(f)
            out.extend(r.colors if r else ["?"] * 9)
        return out

    def finalize(self, resolver: Callable[[list, list[int]], list[str]] | None = None) -> ScanResult:
        """Validate the scanned cube. `resolver(labs54, centre_idx)` re-assigns colours jointly (9 each)."""
        if not self.complete:
            missing = [COLOR_NAMES[WCA_SCHEME[f]] for f in FACES if f not in self.reads]
            self.result = ScanResult(False, None, [f"missing faces: {', '.join(missing)}"])
            return self.result
        cols = self.colors54()
        notes: list[str] = []
        reassigned: list[int] = []
        unsure = [i for i, f in enumerate(FACES) for k in range(9)
                  if not self.reads[f].confident(k)]
        counts_bad = any(cols.count(c) != 9 for c in "WYROGB")
        if (counts_bad or unsure) and resolver is not None and all(self.reads[f].lab is not None for f in FACES):
            labs = [lab for f in FACES for lab in self.reads[f].lab]
            new = resolver(labs, [cube.CENTERS[f] for f in FACES])
            reassigned = [i for i in range(54) if new[i] != cols[i]]
            if reassigned:
                notes.append(f"{len(reassigned)} sticker(s) decided by the 9-per-colour rule")
            cols = list(new)
        if "?" in cols:
            self.result = ScanResult(False, "".join(cols), [f"{cols.count('?')} sticker(s) unreadable"], notes)
            return self.result
        s = "".join(cols)
        v = cube.validate(s)
        if v.ok:
            self.result = ScanResult(True, s, [], notes, reassigned)
            return self.result
        fixed = search_rotations(s)
        if fixed is not None:
            s2, rots = fixed
            notes.append("auto-rotated: " + ", ".join(f"{COLOR_NAMES[WCA_SCHEME[f]]} x{r}" for f, r in rots.items()))
            self.result = ScanResult(True, s2, [], notes, reassigned)
            return self.result
        self.result = ScanResult(False, s, v.errors, notes, reassigned)
        return self.result


def search_rotations(colors54: str, max_results: int = 2) -> tuple[str, dict[str, int]] | None:
    """Find the unique set of face rotations that turns an invalid scan into a valid cube."""
    faces = [list(colors54[i * 9:(i + 1) * 9]) for i in range(6)]
    if any(sorted(colors54).count(c) != 9 for c in set(colors54)):
        return None
    found: dict[str, dict[str, int]] = {}
    for rots in itertools.product(range(4), repeat=6):
        if not any(rots):
            continue
        s = "".join("".join(rotate_face_cw(faces[i], rots[i])) for i in range(6))
        if s in found:
            continue
        if _quick_valid(s):
            found[s] = {FACES[i]: r for i, r in enumerate(rots) if r}
            if len(found) >= max_results:
                return None    # ambiguous
    if len(found) == 1:
        s, rots = next(iter(found.items()))
        return s, rots
    return None


def _quick_valid(s: str) -> bool:
    return cube.validate(s).ok


def expected_face_colors(scramble: str, face: str) -> str:
    return cube.face(to_colors(cube.scrambled_state(scramble)), face)


def describe_counts(colors: Sequence[str]) -> str:
    return " ".join(f"{c}{list(colors).count(c)}" for c in "WYROGB")
