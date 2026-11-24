import random

import pytest

import cube
from cube import (SOLVED, MoveError, apply_moves, colors_to_facelets, face, invert_moves, parse_moves,
                  rotate_face_cw, scrambled_state, to_colors, validate)
from scramble import generate_scramble

ALL_BASES = ["U", "D", "L", "R", "F", "B", "M", "E", "S", "x", "y", "z",
             "Uw", "Dw", "Lw", "Rw", "Fw", "Bw", "u", "d", "l", "r", "f", "b"]


def _swap(s: str, *pairs: tuple[str, str]) -> str:
    out = list(s)
    for a, b in pairs:
        i, j = cube._fi(a), cube._fi(b)
        out[i], out[j] = out[j], out[i]
    return "".join(out)


def _set(s: str, **kw: str) -> str:
    out = list(s)
    for name, v in kw.items():
        out[cube._fi(name)] = v
    return "".join(out)


@pytest.mark.parametrize("base", ALL_BASES)
def test_every_move_four_times_is_identity(base):
    assert apply_moves(SOLVED, [base] * 4) == SOLVED
    assert apply_moves(SOLVED, [base + "2"] * 2) == SOLVED
    assert apply_moves(SOLVED, [base, base + "'"]) == SOLVED
    assert apply_moves(SOLVED, base) != SOLVED


@pytest.mark.parametrize("base", ALL_BASES)
def test_moves_are_permutations_of_any_state(base):
    rng = random.Random(base)
    start = "".join(rng.sample(SOLVED, 54))
    assert apply_moves(start, [base] * 4) == start
    assert sorted(apply_moves(start, base)) == sorted(start)


def test_sexy_move_six_times_is_identity():
    assert apply_moves(SOLVED, "R U R' U' " * 6) == SOLVED
    assert apply_moves(SOLVED, "R U R' U' " * 3) != SOLVED


def test_scramble_plus_inverse_is_solved():
    rng = random.Random(1)
    for _ in range(200):
        s = generate_scramble(rng=rng)
        st = scrambled_state(s)
        assert validate(st).ok
        assert apply_moves(st, invert_moves(s)) == SOLVED


def test_long_random_sequences_with_all_move_types_stay_valid():
    rng = random.Random(2)
    for _ in range(100):
        seq = [rng.choice(ALL_BASES) + rng.choice(["", "'", "2"]) for _ in range(30)]
        st = apply_moves(SOLVED, seq)
        assert validate(st).ok, (seq, validate(st).errors)
        assert apply_moves(st, invert_moves(seq)) == SOLVED


def test_known_facelet_strings():
    assert apply_moves(SOLVED, "R") == "UUFUUFUUFRRRRRRRRRFFDFFDFFDDDBDDBDDBLLLLLLLLLUBBUBBUBB"
    assert apply_moves(SOLVED, "U") == "UUUUUUUUUBBBRRRRRRRRRFFFFFFDDDDDDDDDFFFLLLLLLLLLBBBBBB"
    assert apply_moves(SOLVED, "F") == "UUUUUULLLURRURRURRFFFFFFFFFRRRDDDDDDLLDLLDLLDBBBBBBBBB"
    # D2 and L' (opposite direction / other faces)
    assert apply_moves(SOLVED, "D") == "UUUUUUUUURRRRRRFFFFFFFFFLLLDDDDDDDDDLLLLLLBBBBBBBBBRRR"
    assert apply_moves(SOLVED, "L") == "BUUBUUBUURRRRRRRRRUFFUFFUFFFDDFDDFDDLLLLLLLLLBBDBBDBBD"
    assert apply_moves(SOLVED, "B") == "RRRUUUUUURRDRRDRRDFFFFFFFFFDDDDDDLLLULLULLULLBBBBBBBBB"


def test_known_superflip_and_checkerboard():
    # Checkerboard: M2 E2 S2 = R2 L2 U2 D2 F2 B2 (as a state, centres differ but colours alternate)
    a = apply_moves(SOLVED, "R2 L2 U2 D2 F2 B2")
    for f in "URFDLB":
        fc = face(a, f)
        assert fc[4] == f and fc[0] == fc[2] == fc[6] == fc[8] == f
        assert fc[1] == fc[3] == fc[5] == fc[7] == cube.OPPOSITE[f]
    # Superflip: every edge flipped, everything else solved.
    sf = apply_moves(SOLVED, "U R2 F B R B2 R U2 L B2 R U' D' R2 F R' L B2 U2 F2")
    v = validate(sf)
    assert v.ok
    for fl in cube.EDGE_FACELETS:
        assert sf[fl[0]] == cube.FACES[fl[1] // 9] and sf[fl[1]] == cube.FACES[fl[0] // 9]


def test_slices_wides_and_rotations():
    m = apply_moves(SOLVED, "M")
    assert [m[cube._fi(n)] for n in ("U2", "U5", "U8")] == ["B"] * 3
    assert [m[cube._fi(n)] for n in ("F2", "F5", "F8")] == ["U"] * 3
    x = apply_moves(SOLVED, "x")
    assert face(x, "U") == "F" * 9 and face(x, "F") == "D" * 9 and face(x, "R") == "R" * 9
    y = apply_moves(SOLVED, "y")
    assert face(y, "F") == "R" * 9 and face(y, "U") == "U" * 9
    z = apply_moves(SOLVED, "z")
    assert face(z, "U") == "L" * 9 and face(z, "R") == "U" * 9
    assert apply_moves(SOLVED, "Rw") == apply_moves(SOLVED, "R M'") == apply_moves(SOLVED, "r")
    assert apply_moves(SOLVED, "x") == apply_moves(SOLVED, "R M' L'")
    assert apply_moves(SOLVED, "u") == apply_moves(SOLVED, "U E'")
    assert apply_moves(SOLVED, "f") == apply_moves(SOLVED, "F S")
    assert apply_moves(SOLVED, "y") == apply_moves(SOLVED, "U E' D'")
    assert apply_moves(SOLVED, "z") == apply_moves(SOLVED, "F S B'")


def test_parse_and_invert():
    assert parse_moves("R U2 R' Rw r2 x' M2' R3 U’") == ["R", "U2", "R'", "Rw", "Rw2", "x'", "M2", "R'", "U'"]
    assert invert_moves("R U2 F' Rw") == ["Rw'", "F", "U2", "R'"]
    with pytest.raises(MoveError):
        parse_moves("R Q")
    with pytest.raises(MoveError):
        parse_moves("R4")


def test_rotate_face():
    f = "123456789"
    assert rotate_face_cw(f) == "741852963"
    assert rotate_face_cw(f, 4) == f
    assert rotate_face_cw(list(f), 2) == list("987654321")


def test_wca_orientation_colours():
    cols = to_colors(SOLVED)
    assert face(cols, "U") == "W" * 9 and face(cols, "F") == "G" * 9 and face(cols, "R") == "R" * 9
    assert colors_to_facelets(to_colors(scrambled_state("R U F"))) == scrambled_state("R U F")


# ----------------------------------------------------------------------- validation
def test_validation_accepts_solved_and_colour_letters():
    assert validate(SOLVED).ok
    assert validate(to_colors(scrambled_state("R U R' F2 D L'"))).ok


def test_validation_rejects_bad_counts():
    bad = _set(SOLVED, U1="R")
    v = validate(bad)
    assert not v.ok and any("counts" in e for e in v.errors)


def test_validation_rejects_duplicate_centres():
    bad = _set(SOLVED, R5="U", U1="R")
    v = validate(bad)
    assert not v.ok and any("centres" in e for e in v.errors)


def test_validation_rejects_twisted_corner():
    st = scrambled_state("R U F' D2")
    i, j, k = cube.CORNER_FACELETS[0]
    twisted = list(st)
    twisted[i], twisted[j], twisted[k] = st[k], st[i], st[j]   # cyclic: still a real corner, wrong twist
    v = validate("".join(twisted))
    assert not v.ok and any("twisted" in e for e in v.errors), v.errors


def test_validation_rejects_flipped_edge():
    v = validate(_swap(SOLVED, ("U6", "R2")))
    assert not v.ok and any("flipped" in e for e in v.errors)


def test_validation_rejects_mirrored_corner_and_impossible_pieces():
    v = validate(_swap(SOLVED, ("R1", "F3")))
    assert not v.ok and any("mirrored corner" in e for e in v.errors)
    # An edge with two opposite colours (U and D) can't exist.
    v = validate(_swap(SOLVED, ("R2", "D2")))
    assert not v.ok and any("impossible edge" in e for e in v.errors)


def test_validation_rejects_parity():
    # Swap the UR and UF edges' side stickers -> two edges exchanged, odd permutation.
    v = validate(_swap(SOLVED, ("R2", "F2")))
    assert not v.ok and any("parity" in e for e in v.errors), v.errors


def test_validation_with_rotated_centres():
    assert validate(apply_moves(SOLVED, "M E S x y")).ok
