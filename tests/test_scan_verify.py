import random

import numpy as np
import pytest

import cube
from cube import FACES, SOLVED, rotate_face_cw, scrambled_state, to_colors
from scramble import generate_scramble
from vision.colors import ColorModel, balanced_assign, bgr_to_lab, DEFAULT_BGR
from vision.scan import FaceRead, SCAN_ORDER, ScanSession, ScrambleCheck, check_face, search_rotations
from vision.worker import MIRROR


def read_of(colors9, conf=1.0) -> FaceRead:
    return FaceRead(list(colors9), [conf] * 9)


def expected(scramble):
    return to_colors(scrambled_state(scramble))


@pytest.mark.parametrize("rot", [0, 1, 2, 3])
def test_correct_face_at_any_rotation_is_ok(rot):
    s = "R U F' L2 D B' R2 U'"
    exp = expected(s)
    for f in FACES:
        seen = rotate_face_cw(cube.face(exp, f), rot)
        chk = check_face(read_of(seen), exp)
        assert chk.face == f and chk.status == "ok", (f, rot, chk)
        assert chk.mismatches == 0


def test_wrong_sticker_is_reported_at_the_right_position():
    s = "R U F' L2 D B' R2 U'"
    exp = expected(s)
    face9 = list(rotate_face_cw(cube.face(exp, "F"), 1))
    k = 2
    face9[k] = next(c for c in "WYROGB" if c != face9[k] and c != face9[4])
    chk = check_face(read_of(face9), exp)
    assert chk.status == "mismatch"
    assert chk.sticker_status[k] == "mismatch"
    assert chk.mismatches == 1


def test_scrambled_wrong_detects_mismatch_on_most_scrambles():
    rng = random.Random(4)
    caught = 0
    for _ in range(50):
        s = generate_scramble(rng=rng)
        exp = expected(s)
        wrong = expected(s.rsplit(" ", 1)[0])   # forgot the last move
        results = [check_face(read_of(cube.face(wrong, f)), exp) for f in FACES]
        caught += any(r.status == "mismatch" for r in results)
    assert caught == 50


def test_mirrored_view_is_unmirrored_by_worker_permutation():
    exp = expected("R U R' F2")
    true9 = cube.face(exp, "R")
    mirrored_display = [true9[MIRROR[i]] for i in range(9)]   # what a selfie view shows in reading order
    assert mirrored_display != list(true9)
    # A mirror image is not a rotation, so it must not pass as this face (for this asymmetric face).
    assert all(rotate_face_cw(true9, r) != "".join(mirrored_display) for r in range(4))
    assert check_face(read_of(mirrored_display), exp).status == "mismatch"
    unmirrored = [mirrored_display[MIRROR[i]] for i in range(9)]
    assert check_face(read_of(unmirrored), exp).status == "ok"


def test_low_confidence_and_ambiguous_stickers_are_uncertain_not_mismatch():
    exp = expected("R U F")
    f9 = list(cube.face(exp, "U"))
    r = read_of(f9)
    # a sticker read with low confidence and the wrong colour
    k = 0
    r.colors[k] = "O" if f9[k] != "O" else "R"
    r.confidence[k] = 0.2
    chk = check_face(r, exp)
    assert chk.sticker_status[k] == "uncertain" and chk.status == "unsure"
    # occluded stickers are unseen, not wrong
    r2 = read_of(f9)
    r2.colors[1], r2.confidence[1] = "?", 0.0
    chk2 = check_face(r2, exp)
    assert chk2.sticker_status[1] == "unseen" and chk2.status == "ok"


def test_unreadable_centre():
    exp = expected("R")
    r = read_of(cube.face(exp, "F"))
    r.colors[4], r.confidence[4] = "?", 0.0
    assert check_face(r, exp).status == "no_centre"


def test_scramble_check_accumulates_and_resets():
    sc = ScrambleCheck()
    sc.set_scramble("R U")
    exp = expected("R U")
    assert sc.verdict == "none"
    sc.add(read_of(cube.face(exp, "F")))
    sc.add(read_of(rotate_face_cw(cube.face(exp, "R"), 3)))
    assert sc.verdict == "ok" and "2/6" in sc.summary()
    bad = list(cube.face(exp, "U"))
    bad[0] = "Y"
    sc.add(read_of(bad))
    assert sc.verdict == "mismatch" and "white" in sc.summary()
    sc.set_scramble("F D")
    assert sc.verdict == "none"


# ----------------------------------------------------------------------- guided scan
def _scan_reads(colors54, rotate=None):
    rotate = rotate or {}
    out = {}
    for f in FACES:
        f9 = cube.face(colors54, f)
        out[f] = read_of(rotate_face_cw(f9, rotate.get(f, 0)))
    return out


def test_scan_session_assembles_valid_cube():
    target = expected("R U2 F' L D2 B")
    sess = ScanSession()
    reads = _scan_reads(target)
    for f, centre, _ in SCAN_ORDER:
        msg = sess.offer(reads[f])
        assert msg and msg.startswith("Captured"), msg
    assert sess.complete
    res = sess.finalize()
    assert res.ok and res.colors54 == target


def test_scan_rejects_wrong_face_and_allows_rescan():
    target = expected("R U")
    reads = _scan_reads(target)
    sess = ScanSession()
    msg = sess.offer(reads["U"])          # shows white while green is asked for
    assert "white" in msg.lower() and not sess.reads
    sess.offer(reads["F"])
    assert "F" in sess.reads and sess.target == "R"
    sess.rescan("F")
    assert "F" not in sess.reads and sess.target == "F"


def test_scan_auto_corrects_one_face_held_sideways():
    target = expected("R U F' D L2 B R'")
    sess = ScanSession()
    reads = _scan_reads(target, rotate={"U": 1})
    for f, _, _ in SCAN_ORDER:
        sess.offer(reads[f])
    res = sess.finalize()
    assert res.ok and res.colors54 == target
    assert any("auto-rotated" in n for n in res.notes)


def test_scan_reports_impossible_cube():
    bad = list(expected("R U"))
    i, j = cube._fi("U6"), cube._fi("R2")
    bad[i], bad[j] = bad[j], bad[i]          # flipped edge
    sess = ScanSession()
    reads = _scan_reads("".join(bad))
    for f, _, _ in SCAN_ORDER:
        sess.offer(reads[f])
    res = sess.finalize()
    assert not res.ok and res.errors


def test_search_rotations_is_unique_or_none():
    target = expected("R U F' D L2 B R'")
    rot = "".join("".join(rotate_face_cw(target[i * 9:(i + 1) * 9], 2 if f == "D" else 0))
                  for i, f in enumerate(FACES))
    fixed = search_rotations(rot)
    assert fixed is not None and fixed[0] == target and fixed[1] == {"D": 2}


def test_balanced_assignment_fixes_red_orange_confusions():
    """Make red and orange stickers nearly identical; the 9-per-colour rule still separates them."""
    rng = np.random.default_rng(0)
    target = expected("R U F' D L2 B R' U2")
    base = {c: bgr_to_lab(DEFAULT_BGR[c]) for c in "WYROGB"}
    # under this "light" orange drifts towards red
    base["O"] = base["R"] * 0.45 + base["O"] * 0.55
    labs = [base[c] + rng.normal(0, 2.0, 3) for c in target]
    model = ColorModel.default()
    single = [r.label for r in model.classify_many(labs)]
    joint = balanced_assign(labs, model, [cube.CENTERS[f] for f in FACES])
    assert sum(a != b for a, b in zip(joint, target)) <= sum(a != b for a, b in zip(single, target))
    assert "".join(joint) == target
