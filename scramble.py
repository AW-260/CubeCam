"""Random-move 3x3 scramble generator.

Rules: never turn the same face twice in a row (no R R'), and never three
consecutive moves on one axis (no R L R, which is just R2 L).
"""
from __future__ import annotations

import random

FACES = ("U", "D", "L", "R", "F", "B")
AXIS = {"U": 0, "D": 0, "L": 1, "R": 1, "F": 2, "B": 2}
MODIFIERS = ("", "'", "2")


def generate_scramble(length: int = 20, rng: random.Random | None = None) -> str:
    return " ".join(generate_moves(length, rng))


def generate_moves(length: int = 20, rng: random.Random | None = None) -> list[str]:
    rng = rng or random.Random()
    faces: list[str] = []
    while len(faces) < length:
        face = rng.choice(FACES)
        if faces and face == faces[-1]:
            continue
        if len(faces) >= 2 and AXIS[face] == AXIS[faces[-1]] == AXIS[faces[-2]]:
            continue
        faces.append(face)
    return [f + rng.choice(MODIFIERS) for f in faces]


def is_valid_scramble(moves: list[str]) -> bool:
    faces = []
    for m in moves:
        if not m or m[0] not in FACES or m[1:] not in MODIFIERS:
            return False
        faces.append(m[0])
    for i in range(1, len(faces)):
        if faces[i] == faces[i - 1]:
            return False
        if i >= 2 and AXIS[faces[i]] == AXIS[faces[i - 1]] == AXIS[faces[i - 2]]:
            return False
    return True
