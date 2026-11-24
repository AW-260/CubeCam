"""JSON persistence for the solve session."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from models import Solve

FORMAT_VERSION = 1


class SolveStore:
    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.warning: str | None = None

    def load(self) -> list[Solve]:
        """Load solves; a missing file is an empty session, a corrupt one is backed up."""
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            items = data["solves"] if isinstance(data, dict) else data
            return [Solve.from_dict(d) for d in items]
        except (ValueError, KeyError, TypeError) as exc:
            backup = self.path.with_name(f"{self.path.stem}.corrupt-{int(time.time())}{self.path.suffix}")
            try:
                self.path.replace(backup)
                self.warning = f"Solve file was unreadable ({exc}); backed up to {backup.name}"
            except OSError:
                self.warning = f"Solve file was unreadable ({exc}); starting empty"
            return []

    def save(self, solves: list[Solve]) -> None:
        """Atomic write (temp file + replace). Retries briefly if OneDrive/AV holds a lock."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": FORMAT_VERSION, "solves": [s.to_dict() for s in solves]}
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        for attempt in range(5):
            try:
                os.replace(tmp, self.path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
