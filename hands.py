"""MediaPipe hand detection -> "both hands in start zone" trigger.

Two MediaPipe backends are supported because the API changed between versions:
  * legacy: mp.solutions.hands (mediapipe <= 0.10.21, no model file needed)
  * tasks:  mediapipe.tasks HandLandmarker (all recent versions, needs
            models/hand_landmarker.task - run `python setup_model.py`)
backend "auto" tries legacy first, then tasks.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np

from detection import DetectorUnavailable, TriggerSample, Zone

# Quieten MediaPipe / TFLite logging (must be set before importing mediapipe).
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PALM_IDS = (0, 5, 9, 13, 17)  # wrist + finger MCP joints
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


class _LegacyBackend:
    name = "mediapipe-legacy"

    def __init__(self, cfg: dict):
        import mediapipe as mp

        if not hasattr(mp, "solutions"):
            raise DetectorUnavailable(f"mediapipe {mp.__version__} has no legacy solutions API")
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=int(cfg.get("model_complexity", 0)),
            min_detection_confidence=float(cfg.get("min_detection_confidence", 0.5)),
            min_tracking_confidence=float(cfg.get("min_tracking_confidence", 0.5)),
        )

    def detect(self, rgb: np.ndarray, t: float) -> list[np.ndarray]:
        rgb.flags.writeable = False
        res = self._hands.process(rgb)
        if not res.multi_hand_landmarks:
            return []
        return [np.array([(p.x, p.y) for p in h.landmark], dtype=np.float32) for h in res.multi_hand_landmarks]

    def close(self) -> None:
        self._hands.close()


class _TasksBackend:
    name = "mediapipe-tasks"

    def __init__(self, cfg: dict):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        model = Path(cfg.get("model_path", "models/hand_landmarker.task"))
        if not model.is_file():
            raise DetectorUnavailable(f"model file {model} missing - run: python setup_model.py")
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=float(cfg.get("min_detection_confidence", 0.5)),
            min_tracking_confidence=float(cfg.get("min_tracking_confidence", 0.5)),
        )
        self._mp = mp
        self._landmarker = vision.HandLandmarker.create_from_options(opts)
        self._last_ms = -1

    def detect(self, rgb: np.ndarray, t: float) -> list[np.ndarray]:
        ms = max(int(t * 1000), self._last_ms + 1)  # must strictly increase
        self._last_ms = ms
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self._landmarker.detect_for_video(image, ms)
        return [np.array([(p.x, p.y) for p in h], dtype=np.float32) for h in res.hand_landmarks]

    def close(self) -> None:
        self._landmarker.close()


def _create_backend(cfg: dict):
    choice = cfg.get("backend", "auto")
    order = {"auto": ["legacy", "tasks"], "legacy": ["legacy"], "tasks": ["tasks"]}.get(choice)
    if order is None:
        raise DetectorUnavailable(f"unknown detection backend {choice!r}")
    errors = []
    for name in order:
        try:
            return _LegacyBackend(cfg) if name == "legacy" else _TasksBackend(cfg)
        except ImportError as exc:
            errors.append(f"{name}: mediapipe not installed ({exc})")
        except OSError as exc:  # e.g. Windows Application Control blocking the DLL
            errors.append(f"{name}: native library failed to load ({exc})")
        except DetectorUnavailable as exc:
            errors.append(f"{name}: {exc}")
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise DetectorUnavailable("; ".join(errors))


class HandZoneDetector:
    """TriggerDetector: active when >= hands_required palm centres are in the zone."""

    name = "hands"

    def __init__(self, zone: Zone, cfg: dict | None = None):
        cfg = cfg or {}
        self.zone = zone
        self.hands_required = int(cfg.get("hands_required", 2))
        self.process_width = int(cfg.get("process_width", 640))
        self._backend = _create_backend(cfg)
        self.backend_name = self._backend.name

    def process(self, frame_bgr: np.ndarray, t: float) -> TriggerSample:
        h, w = frame_bgr.shape[:2]
        img = frame_bgr
        if self.process_width and w > self.process_width:
            scale = self.process_width / w
            img = cv2.resize(frame_bgr, (self.process_width, int(h * scale)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        hands = self._backend.detect(rgb, t)
        detect_ms = (time.perf_counter() - t0) * 1000
        # Landmarks are normalised to the (proportionally resized) image, so they
        # map straight back onto the full frame.
        palms = [tuple(lm[list(PALM_IDS)].mean(axis=0)) for lm in hands]
        in_zone = [self.zone.contains(px, py) for px, py in palms]
        active = sum(in_zone) >= self.hands_required
        return TriggerSample(
            t,
            active,
            {
                "hands": hands,
                "palms": palms,
                "palms_in_zone": in_zone,
                "detect_ms": detect_ms,
                "backend": self.backend_name,
            },
        )

    def close(self) -> None:
        self._backend.close()
