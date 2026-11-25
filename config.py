"""Config loading: config.yaml deep-merged over defaults, so missing keys never crash."""
from __future__ import annotations

import copy
import os
import re
import time
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "camera": {
        "enabled": True,
        "index": 0,
        "backend": "auto",
        "width": 1280,
        "height": 720,
        "fps": 120,
        "fourcc": "MJPG",
        "mirror": True,
    },
    "detection": {
        "backend": "auto",
        "model_path": "models/hand_landmarker.task",
        "hands_required": 2,
        "model_complexity": 0,
        "min_detection_confidence": 0.5,
        "min_tracking_confidence": 0.5,
        "process_width": 640,
    },
    # No default "points": an old config with only x/y/w/h must keep its rectangle.
    "zone": {"mode": "fixed", "x": 0.10, "y": 0.68, "w": 0.80, "h": 0.30},
    "markers": {"dictionary": "DICT_4X4_50", "ids": [0, 1, 2, 3], "interval": 0.25, "process_width": 960},
    "timer": {
        "hold_time": 0.3,
        "debounce": 0.05,
        "inspection": False,
        "inspection_limit": 15.0,
        "dnf_limit": 17.0,
        "warnings": [8, 12],
        "audio": True,
    },
    "storage": {"path": "solves.json"},
    "ui": {"window_name": "CubeCam", "keyboard_canvas": [1280, 720], "show_debug": False},
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path = "config.yaml") -> tuple[dict, str | None]:
    """Return (config, warning_or_None)."""
    p = Path(path)
    if not p.exists():
        return copy.deepcopy(DEFAULTS), f"{p} not found - using defaults"
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError("top level must be a mapping")
    except (yaml.YAMLError, ValueError) as exc:
        return copy.deepcopy(DEFAULTS), f"could not parse {p} ({exc}) - using defaults"
    return _merge(DEFAULTS, data), None


_ZONE_LINE = re.compile(r"^zone:\s*(#.*)?$")


def _points(points) -> list[list[float]]:
    return [[round(float(x), 4), round(float(y), 4)] for x, y in points]


def zone_block(mode: str, points) -> list[str]:
    pts = ", ".join(f"[{x!r}, {y!r}]" for x, y in _points(points))
    return [
        "zone:",
        f"  mode: {mode:<12}# fixed = use these points; markers = follow the 4 ArUco markers",
        f"  points: [{pts}]",
    ]


def replace_zone_block(text: str, mode: str, points) -> str:
    """Swap the top-level `zone:` block in config text, keeping everything else (comments too)."""
    lines = text.splitlines()
    new = zone_block(mode, points)
    start = next((i for i, line in enumerate(lines) if _ZONE_LINE.match(line)), None)
    if start is None:
        body = text.rstrip("\n")
        return (body + "\n\n" if body else "") + "\n".join(new) + "\n"
    end = start + 1
    while end < len(lines) and (not lines[end].strip() or lines[end][0] in " \t"):
        end += 1
    while end > start + 1 and not lines[end - 1].strip():  # keep blank lines before the next section
        end -= 1
    return "\n".join(lines[:start] + new + lines[end:]) + "\n"


def save_zone(path: str | Path, mode: str, points) -> None:
    """Persist the zone to config.yaml (atomic write, checked by re-parsing first)."""
    if mode not in ("fixed", "markers"):
        raise ValueError(f"unknown zone mode {mode!r}")
    p = Path(path)
    text = p.read_text(encoding="utf-8") if p.exists() else ""
    out = replace_zone_block(text, mode, points)
    zone = (yaml.safe_load(out) or {}).get("zone") or {}
    if zone.get("mode") != mode or zone.get("points") != _points(points):
        raise ValueError(f"could not update the zone in {p.name} (unexpected layout)")
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(out, encoding="utf-8")
    for attempt in range(5):
        try:
            os.replace(tmp, p)
            return
        except PermissionError:  # OneDrive / antivirus briefly holding the file
            if attempt == 4:
                raise
            time.sleep(0.05 * (attempt + 1))
