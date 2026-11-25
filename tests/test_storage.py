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


def test_extra_fields_from_later_stages_are_ignored(tmp_path):
    import json
    path = tmp_path / "solves.json"
    path.write_text(json.dumps({"version": 1, "solves": [
        {"time": 12.3, "penalty": "+2", "scramble": "R U", "motion": {"turns": 40}, "future": [1, 2]},
    ]}), encoding="utf-8")
    store = SolveStore(path)
    solves = store.load()
    assert store.warning is None
    assert len(solves) == 1 and solves[0].time == 12.3 and solves[0].penalty is Penalty.PLUS2
