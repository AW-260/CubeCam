"""Config loading: config.yaml deep-merged over defaults, so missing keys never crash."""
from __future__ import annotations

import copy
import json
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
    "markers": {"dictionary": "DICT_4X4_50", "ids": [0, 1, 2, 3], "interval": 0.25, "process_width": 960,
                "size_mm": 60},
    # Table-plane calibration (key T). phone_marker_mm: width of the black square
    # on your phone screen, measured with a ruler; None = typical phone, approximate.
    "table": {"phone_marker_mm": None, "phone_marker_id": 10, "dictionary": "DICT_4X4_50", "camera_hfov_deg": None},
    "mat": {"ar": True, "width_cm": 30.0, "height_cm": 22.0},
    # Written by the app (T / M); None = not calibrated / not placed.
    "table_calibration": None,
    "mat_placement": None,
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


ZONE_MODES = ("fixed", "markers", "mat")


def _points(points) -> list[list[float]]:
    return [[round(float(x), 4), round(float(y), 4)] for x, y in points]


def _key_line(key: str) -> re.Pattern:
    return re.compile(rf"^{re.escape(key)}:(\s.*)?$")


def replace_block(text: str, key: str, lines: list[str]) -> str:
    """Swap the top-level `key:` block (its line plus the indented lines under it) for `lines`.

    Everything else in the file, comments included, is kept. A missing block is appended.
    """
    src = text.splitlines()
    pat = _key_line(key)
    start = next((i for i, line in enumerate(src) if pat.match(line)), None)
    if start is None:
        body = text.rstrip("\n")
        return (body + "\n\n" if body else "") + "\n".join(lines) + "\n"
    end = start + 1
    while end < len(src) and (not src[end].strip() or src[end][0] in " \t"):
        end += 1
    while end > start + 1 and not src[end - 1].strip():  # keep blank lines before the next section
        end -= 1
    return "\n".join(src[:start] + lines + src[end:]) + "\n"


def _flow(value) -> str:
    """One-line YAML for a scalar or (nested) list; JSON is valid YAML flow style."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    return json.dumps(value)


def mapping_block(key: str, value: dict | None, comment: str = "") -> list[str]:
    head = f"{key}:" + (f"  # {comment}" if comment else "")
    if value is None:
        return [f"{key}: null" + (f"  # {comment}" if comment else "")]
    return [head] + [f"  {k}: {_flow(v)}" for k, v in value.items()]


def zone_block(mode: str, points) -> list[str]:
    pts = ", ".join(f"[{x!r}, {y!r}]" for x, y in _points(points))
    return [
        "zone:",
        f"  mode: {mode:<12}# fixed = these points; markers = follow the 4 markers; mat = the AR mat",
        f"  points: [{pts}]",
    ]


def replace_zone_block(text: str, mode: str, points) -> str:
    return replace_block(text, "zone", zone_block(mode, points))


def _write_checked(p: Path, out: str, checks: dict) -> None:
    data = yaml.safe_load(out) or {}
    for key, expected in checks.items():
        if data.get(key) != expected:
            raise ValueError(f"could not update '{key}' in {p.name} (unexpected layout)")
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


def save_blocks(path: str | Path, zone: tuple[str, list] | None = None, **mappings: dict | None) -> None:
    """Persist app-written settings to config.yaml in one atomic write (checked by re-parsing).

    zone=(mode, points) rewrites the `zone:` block; every keyword (e.g.
    table_calibration=..., mat_placement=...) rewrites that top-level block.
    """
    p = Path(path)
    out = p.read_text(encoding="utf-8") if p.exists() else ""
    checks: dict = {}
    if zone is not None:
        mode, points = zone
        if mode not in ZONE_MODES:
            raise ValueError(f"unknown zone mode {mode!r}")
        out = replace_zone_block(out, mode, points)
        checks["zone"] = {"mode": mode, "points": _points(points)}
    for key, value in mappings.items():
        out = replace_block(out, key, mapping_block(key, value, "written by CubeCam - delete to reset"))
        checks[key] = json.loads(json.dumps(value)) if value is not None else None
    _write_checked(p, out, checks)


def save_zone(path: str | Path, mode: str, points) -> None:
    save_blocks(path, zone=(mode, points))
