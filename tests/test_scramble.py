import random

from scramble import AXIS, FACES, MODIFIERS, generate_moves, generate_scramble, is_valid_scramble


def test_length_and_tokens():
    moves = generate_moves(20, random.Random(1))
    assert len(moves) == 20
    for m in moves:
        assert m[0] in FACES and m[1:] in MODIFIERS


def test_no_same_face_or_same_axis_triples_many_seeds():
    for seed in range(2000):
        moves = generate_moves(20, random.Random(seed))
        faces = [m[0] for m in moves]
        for i in range(1, len(faces)):
            assert faces[i] != faces[i - 1], moves
            if i >= 2:
                assert not (AXIS[faces[i]] == AXIS[faces[i - 1]] == AXIS[faces[i - 2]]), moves
        assert is_valid_scramble(moves)


def test_deterministic_with_seed():
    assert generate_scramble(rng=random.Random(42)) == generate_scramble(rng=random.Random(42))


def test_scrambles_vary():
    rng = random.Random(7)
    assert len({generate_scramble(rng=rng) for _ in range(50)}) == 50


def test_uses_all_faces_and_modifiers():
    rng = random.Random(3)
    moves = [m for _ in range(100) for m in generate_moves(20, rng)]
    assert {m[0] for m in moves} == set(FACES)
    assert {m[1:] for m in moves} == set(MODIFIERS)


def test_validator_rejects_bad_scrambles():
    assert not is_valid_scramble(["R", "R'"])
    assert not is_valid_scramble(["R", "L", "R2"])
    assert not is_valid_scramble(["X"])
    assert is_valid_scramble(["R", "L", "U"])


def test_scramble_string_format():
    s = generate_scramble(rng=random.Random(0))
    assert len(s.split(" ")) == 20
