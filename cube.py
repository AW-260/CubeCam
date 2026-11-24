"""3x3 cube state model (pure Python, no OpenCV).

State = 54 facelets in the standard Kociemba order U1..U9 R1..R9 F1..F9 D1..D9
L1..L9 B1..B9, each holding a face letter (the face whose *centre* has that
colour in the solved cube). Facelets on each face are numbered row by row as
seen when looking straight at that face in the usual net:

              U1 U2 U3
              U4 U5 U6
              U7 U8 U9
    L1 L2 L3  F1 F2 F3  R1 R2 R3  B1 B2 B3
    L4 L5 L6  F4 F5 F6  R4 R5 R6  B4 B5 B6
    L7 L8 L9  F7 F8 F9  R7 R8 R9  B7 B8 B9
              D1 D2 D3
              D4 D5 D6
              D7 D8 D9

Moves are generated from geometry: every sticker has a 3D position and normal,
and a move rotates the stickers of some layers by 90 degrees about an axis.
That gives face turns, wide turns, slices and whole-cube rotations from one rule.

WCA scrambling orientation: white on top (U), green in front (F).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

FACES = "URFDLB"
SOLVED = "".join(f * 9 for f in FACES)
# WCA colour scheme in scrambling orientation (white top, green front).
WCA_SCHEME = {"U": "W", "R": "R", "F": "G", "D": "Y", "L": "O", "B": "B"}
COLOR_NAMES = {"W": "white", "Y": "yellow", "R": "red", "O": "orange", "G": "green", "B": "blue"}
OPPOSITE = {"U": "D", "D": "U", "R": "L", "L": "R", "F": "B", "B": "F"}
CENTERS = {f: i * 9 + 4 for i, f in enumerate(FACES)}

Vec = tuple[int, int, int]


# --------------------------------------------------------------------------- geometry
def _facelet_geometry() -> list[tuple[Vec, Vec]]:
    """(cubie position, outward normal) for each of the 54 facelets.

    Axes: x -> R, y -> U, z -> F. Row r / column c are 0..2 as in the net above.
    """
    out: list[tuple[Vec, Vec]] = []
    for face in FACES:
        for r in range(3):
            for c in range(3):
                if face == "U":
                    pos, n = (c - 1, 1, r - 1), (0, 1, 0)
                elif face == "D":
                    pos, n = (c - 1, -1, 1 - r), (0, -1, 0)
                elif face == "F":
                    pos, n = (c - 1, 1 - r, 1), (0, 0, 1)
                elif face == "B":
                    pos, n = (1 - c, 1 - r, -1), (0, 0, -1)
                elif face == "R":
                    pos, n = (1, 1 - r, 1 - c), (1, 0, 0)
                else:  # L
                    pos, n = (-1, 1 - r, c - 1), (-1, 0, 0)
                out.append((pos, n))
    return out


GEOMETRY = _facelet_geometry()
_INDEX = {g: i for i, g in enumerate(GEOMETRY)}


def _rot(v: Vec, axis: int, q: int) -> Vec:
    """Rotate v by q quarter turns counter-clockwise (right-hand rule) about axis 0/1/2."""
    x, y, z = v
    for _ in range(q % 4):
        if axis == 0:
            x, y, z = x, -z, y
        elif axis == 1:
            x, y, z = z, y, -x
        else:
            x, y, z = -y, x, z
    return (x, y, z)


# name -> (axis, layers, direction). direction -1 = clockwise seen from +axis.
_MOVE_DEFS: dict[str, tuple[int, frozenset, int]] = {
    "U": (1, frozenset({1}), -1), "D": (1, frozenset({-1}), 1), "E": (1, frozenset({0}), 1),
    "R": (0, frozenset({1}), -1), "L": (0, frozenset({-1}), 1), "M": (0, frozenset({0}), 1),
    "F": (2, frozenset({1}), -1), "B": (2, frozenset({-1}), 1), "S": (2, frozenset({0}), -1),
    "Uw": (1, frozenset({1, 0}), -1), "Dw": (1, frozenset({-1, 0}), 1),
    "Rw": (0, frozenset({1, 0}), -1), "Lw": (0, frozenset({-1, 0}), 1),
    "Fw": (2, frozenset({1, 0}), -1), "Bw": (2, frozenset({-1, 0}), 1),
    "x": (0, frozenset({-1, 0, 1}), -1), "y": (1, frozenset({-1, 0, 1}), -1),
    "z": (2, frozenset({-1, 0, 1}), -1),
}
for _f in "UDRLFB":
    _MOVE_DEFS[_f.lower()] = _MOVE_DEFS[_f + "w"]


def _quarter_perm(axis: int, layers: frozenset, direction: int) -> tuple[int, ...]:
    """perm with new[i] = old[perm[i]] for one quarter turn."""
    perm = list(range(54))
    for i, (pos, n) in enumerate(GEOMETRY):
        if pos[axis] in layers:
            j = _INDEX[(_rot(pos, axis, direction), _rot(n, axis, direction))]
            perm[j] = i
    return tuple(perm)


def _compose(p: tuple[int, ...], q: tuple[int, ...]) -> tuple[int, ...]:
    """Apply p then q."""
    return tuple(p[q[i]] for i in range(54))


_PERMS: dict[str, tuple[int, ...]] = {}
for _name, (_axis, _layers, _dir) in _MOVE_DEFS.items():
    q1 = _quarter_perm(_axis, _layers, _dir)
    q2 = _compose(q1, q1)
    _PERMS[_name] = q1
    _PERMS[_name + "2"] = q2
    _PERMS[_name + "'"] = _compose(q2, q1)

_TOKEN = re.compile(r"^([URFDLB]w|[URFDLBMESxyzurfdlb])(2'|2|'|3)?$")


class MoveError(ValueError):
    pass


# --------------------------------------------------------------------------- moves
def normalize_move(token: str) -> str:
    """'R' / "R'" / 'R2' / "R2'" / 'R3' / 'Rw' / 'r' -> canonical key in _PERMS."""
    m = _TOKEN.match(token.strip().replace("’", "'"))
    if not m:
        raise MoveError(f"unknown move {token!r}")
    base, suf = m.group(1), m.group(2) or ""
    if base[0].islower() and base not in ("x", "y", "z"):
        base = base.upper() + "w"
    suf = {"": "", "'": "'", "2": "2", "2'": "2", "3": "'"}[suf]
    return base + suf


def parse_moves(seq: str | list[str]) -> list[str]:
    """Parse "R U R' U2 Rw x" (or a list of tokens) into canonical moves."""
    tokens = seq.split() if isinstance(seq, str) else list(seq)
    return [normalize_move(t) for t in tokens if t.strip()]


def invert_move(move: str) -> str:
    m = normalize_move(move)
    if m.endswith("2"):
        return m
    if m.endswith("'"):
        return m[:-1]
    return m + "'"


def invert_moves(seq: str | list[str]) -> list[str]:
    return [invert_move(m) for m in reversed(parse_moves(seq))]


def apply_moves(state: str, seq: str | list[str]) -> str:
    s = list(state)
    if len(s) != 54:
        raise ValueError("state must have 54 facelets")
    for m in parse_moves(seq):
        p = _PERMS[m]
        s = [s[p[i]] for i in range(54)]
    return "".join(s)


def scrambled_state(scramble: str | list[str]) -> str:
    """Expected facelet string after applying a scramble to a solved cube held in WCA orientation."""
    return apply_moves(SOLVED, scramble)


def to_colors(state: str, scheme: dict[str, str] | None = None) -> str:
    """Face letters -> colour letters (W Y R O G B) using the WCA scheme by default."""
    scheme = scheme or WCA_SCHEME
    return "".join(scheme[c] for c in state)


def face(state: str, f: str) -> str:
    """The 9 facelets of face f."""
    i = FACES.index(f) * 9
    return state[i:i + 9]


def rotate_face_cw(f9: str | list, times: int = 1):
    """Rotate a 9-element row-major face clockwise `times` quarter turns (same type out)."""
    out = list(f9)
    for _ in range(times % 4):
        out = [out[6], out[3], out[0], out[7], out[4], out[1], out[8], out[5], out[2]]
    return "".join(out) if isinstance(f9, str) else out


# --------------------------------------------------------------------------- validation
def _fi(name: str) -> int:
    """'U9' -> facelet index."""
    return FACES.index(name[0]) * 9 + int(name[1]) - 1


def _fl(*names: str) -> tuple[int, ...]:
    return tuple(_fi(n) for n in names)


# Kociemba corner / edge facelets (URF UFL ULB UBR DFR DLF DBL DRB / UR UF UL UB DR DF DL DB FR FL BL BR)
CORNER_FACELETS = [
    _fl("U9", "R1", "F3"), _fl("U7", "F1", "L3"), _fl("U1", "L1", "B3"), _fl("U3", "B1", "R3"),
    _fl("D3", "F9", "R7"), _fl("D1", "L9", "F7"), _fl("D7", "B9", "L7"), _fl("D9", "R9", "B7"),
]
CORNER_NAMES = ["URF", "UFL", "ULB", "UBR", "DFR", "DLF", "DBL", "DRB"]
EDGE_FACELETS = [
    _fl("U6", "R2"), _fl("U8", "F2"), _fl("U4", "L2"), _fl("U2", "B2"), _fl("D6", "R8"), _fl("D2", "F8"),
    _fl("D4", "L8"), _fl("D8", "B8"), _fl("F6", "R4"), _fl("F4", "L6"), _fl("B6", "L4"), _fl("B4", "R6"),
]
EDGE_NAMES = ["UR", "UF", "UL", "UB", "DR", "DF", "DL", "DB", "FR", "FL", "BL", "BR"]


@dataclass
class Validation:
    ok: bool
    errors: list[str] = field(default_factory=list)
    facelets: str | None = None          # face-letter string (relabelled by centres)

    def __bool__(self) -> bool:
        return self.ok


def colors_to_facelets(colors: str | list[str]) -> str:
    """Map a 54-sticker colour string to face letters using the centre colours."""
    colors = list(colors)
    if len(colors) != 54:
        raise ValueError("need 54 stickers")
    centre_map = {colors[CENTERS[f]]: f for f in FACES}
    if len(centre_map) != 6:
        raise ValueError("centres are not 6 different colours")
    try:
        return "".join(centre_map[c] for c in colors)
    except KeyError as exc:
        raise ValueError(f"sticker colour {exc} does not match any centre") from None


def _parity(perm: list[int]) -> int:
    seen, parity = [False] * len(perm), 0
    for i in range(len(perm)):
        if seen[i]:
            continue
        j, length = i, 0
        while not seen[j]:
            seen[j] = True
            j = perm[j]
            length += 1
        parity ^= (length - 1) & 1
    return parity


def validate(state: str | list[str]) -> Validation:
    """Check that a 54-facelet state (any 6 symbols) is a reachable cube.

    Symbols are relabelled by the centre at each face first, so colour letters
    (W/Y/...) or face letters both work, and rotated centres are fine.
    """
    s = list(state)
    errors: list[str] = []
    if len(s) != 54:
        return Validation(False, [f"need 54 facelets, got {len(s)}"])
    counts = {c: s.count(c) for c in set(s)}
    bad = {c: n for c, n in counts.items() if n != 9}
    if len(counts) != 6 or bad:
        errors.append("colour counts must be 9 each: " + ", ".join(f"{c}={n}" for c, n in sorted(counts.items())))
    centres = [s[CENTERS[f]] for f in FACES]
    if len(set(centres)) != 6:
        errors.append("centres are not 6 different colours: " + "".join(centres))
        return Validation(False, errors)
    relabel = {s[CENTERS[f]]: f for f in FACES}
    try:
        fs = "".join(relabel[c] for c in s)
    except KeyError as exc:
        errors.append(f"colour {exc} is not a centre colour")
        return Validation(False, errors)

    corner_colors = [tuple(FACES[i // 9] for i in fl) for fl in CORNER_FACELETS]
    edge_colors = [tuple(FACES[i // 9] for i in fl) for fl in EDGE_FACELETS]
    cp, co, ep, eo = [], [], [], []
    for pos, fl in enumerate(CORNER_FACELETS):
        cols = [fs[i] for i in fl]
        ud = [k for k, c in enumerate(cols) if c in "UD"]
        if len(ud) != 1:
            errors.append(f"impossible corner at {CORNER_NAMES[pos]}: {''.join(cols)}")
            continue
        ori = ud[0]
        seq = (cols[ori], cols[(ori + 1) % 3], cols[(ori + 2) % 3])
        match = [j for j, cc in enumerate(corner_colors) if cc == seq]
        if not match:
            kind = "mirrored" if any(set(cc) == set(cols) for cc in corner_colors) else "impossible"
            errors.append(f"{kind} corner at {CORNER_NAMES[pos]}: {''.join(cols)}")
            continue
        cp.append(match[0])
        co.append(ori)
    for pos, fl in enumerate(EDGE_FACELETS):
        cols = tuple(fs[i] for i in fl)
        j = next((j for j, ec in enumerate(edge_colors) if ec == cols or ec == cols[::-1]), None)
        if j is None:
            errors.append(f"impossible edge at {EDGE_NAMES[pos]}: {''.join(cols)}")
            continue
        ep.append(j)
        eo.append(0 if edge_colors[j] == cols else 1)
    if errors:
        return Validation(False, errors, fs)
    if len(set(cp)) != 8:
        dup = sorted({CORNER_NAMES[j] for j in cp if cp.count(j) > 1})
        errors.append("duplicate corner pieces: " + ", ".join(dup))
    if len(set(ep)) != 12:
        dup = sorted({EDGE_NAMES[j] for j in ep if ep.count(j) > 1})
        errors.append("duplicate edge pieces: " + ", ".join(dup))
    if errors:
        return Validation(False, errors, fs)
    if sum(co) % 3:
        errors.append("twisted corner (corner orientation sum is not a multiple of 3)")
    if sum(eo) % 2:
        errors.append("flipped edge (edge orientation sum is odd)")
    if _parity(cp) != _parity(ep):
        errors.append("parity error (two pieces swapped)")
    return Validation(not errors, errors, fs)
