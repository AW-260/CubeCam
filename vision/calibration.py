"""Colour calibration: learn this cube's six colours under the current light.

The user shows each face (centre colour known from the prompt). The centre is
always used; on a solved face the other stickers that match the centre colour
are used too, which gives more and better-spread samples. Samples from several
still frames are pooled and turned into a ColorModel (saved as calibration.json).
"""
from __future__ import annotations

import numpy as np

from vision.colors import COLORS, ColorModel

ORDER = "WYGBRO"
PROMPTS = {
    "W": "WHITE", "Y": "YELLOW", "G": "GREEN", "B": "BLUE", "R": "RED", "O": "ORANGE",
}


class CalibrationSession:
    def __init__(self, order: str = ORDER, frames_needed: int = 4, same_color_ab: float = 10.0):
        self.order = order
        self.frames_needed = frames_needed
        self.same_color_ab = same_color_ab
        self.samples: dict[str, list[np.ndarray]] = {c: [] for c in COLORS}
        self.frames: dict[str, int] = {c: 0 for c in COLORS}
        self.index = 0

    @property
    def current(self) -> str | None:
        return self.order[self.index] if self.index < len(self.order) else None

    @property
    def done(self) -> bool:
        return self.index >= len(self.order)

    def count(self, color: str) -> int:
        return len(self.samples[color])

    def add(self, color: str, labs, confidence) -> int:
        """Add one still frame of the face whose centre is `color`. Returns samples added."""
        labs = np.asarray(labs, dtype=np.float64).reshape(9, 3)
        conf = np.asarray(confidence, dtype=np.float64).reshape(9)
        if conf[4] < 0.3:
            return 0
        centre = labs[4]
        added = [centre]
        for k in range(9):
            if k == 4 or conf[k] < 0.5:
                continue
            d_ab = float(np.hypot(*(labs[k, 1:] - centre[1:])))
            if d_ab < self.same_color_ab and abs(labs[k, 0] - centre[0]) < 25:
                added.append(labs[k])
        self.samples[color].extend(added)
        self.frames[color] += 1
        return len(added)

    def add_current(self, labs, confidence) -> bool:
        """Add a frame for the prompted colour; advance when enough frames. Returns True if advanced."""
        c = self.current
        if c is None:
            return False
        if self.add(c, labs, confidence) and self.frames[c] >= self.frames_needed:
            self.index += 1
            return True
        return False

    def redo_previous(self) -> None:
        if self.index > 0:
            self.index -= 1
        c = self.current
        if c is not None:
            self.samples[c].clear()
            self.frames[c] = 0

    def build(self, base: ColorModel | None = None) -> ColorModel:
        return ColorModel.from_samples({c: v for c, v in self.samples.items() if v}, base)

    def check_distinct(self, model: ColorModel) -> list[str]:
        """Warnings when two calibrated colours are too close to tell apart."""
        warnings = []
        for i, a in enumerate(COLORS):
            for b in COLORS[i + 1:]:
                d = model.distances(model.means[a].reshape(1, 3))[0, COLORS.index(b)]
                if d < 2.5:
                    warnings.append(f"{PROMPTS[a].lower()} and {PROMPTS[b].lower()} look very similar "
                                    f"in this light (separation {d:.1f})")
        return warnings
