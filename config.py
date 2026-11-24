"""Config loading: config.yaml deep-merged over defaults, so missing keys never crash."""
from __future__ import annotations

import copy
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
    "zone": {"x": 0.10, "y": 0.68, "w": 0.80, "h": 0.30},
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
    "vision": {
        "enabled": True,
        "process_width": 640,
        "still_time": 0.3,
        "auto_verify": True,
        "calibration_path": "calibration.json",
    },
    "dataset": {
        "path": "datasets",
        "jpeg_quality": 90,
        "max_queue": 120,
        "preroll": 0.5,
        "tail": 0.5,
        "max_width": 0,
    },
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
