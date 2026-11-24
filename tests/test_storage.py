from models import Penalty, Solve
from storage import SolveStore


def test_roundtrip(tmp_path):
    store = SolveStore(tmp_path / "solves.json")
    assert store.load() == []
    solves = [
        Solve(12.34, Penalty.OK, "R U F", "2026-01-01T10:00:00", 60.0, None),
        Solve(9.87, Penalty.PLUS2, "L D B", "2026-01-01T10:01:00", 59.5, 15.4),
        Solve(20.0, Penalty.DNF, "U2", "2026-01-01T10:02:00", None, 17.8),
    ]
    store.save(solves)
    assert SolveStore(tmp_path / "solves.json").load() == solves


def test_corrupt_file_is_backed_up(tmp_path):
    p = tmp_path / "solves.json"
    p.write_text("{not json", encoding="utf-8")
    store = SolveStore(p)
    assert store.load() == []
    assert store.warning
    assert not p.exists()
    assert len(list(tmp_path.glob("solves.corrupt-*.json"))) == 1
