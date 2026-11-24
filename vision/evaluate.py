"""Accuracy evaluation of detection + colour classification on synthetic scenes.

Used by tests/test_vision_synth.py and tools/eval_synthetic.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vision.colors import ColorModel, plausible_face, read_stickers
from vision.face_detect import FaceDetector
from vision.synth import Lighting, calibration_views, render_background, render_scene

CONFIDENT = 0.5


@dataclass
class EvalReport:
    scenes: int = 0
    detected: int = 0               # correct face, correct orientation
    wrong_face: int = 0             # a grid was found but not the front face (or rotated)
    missed: int = 0
    stickers: int = 0               # stickers on correctly detected faces
    visible: int = 0                # not occluded
    confident: int = 0              # reported confident
    confident_correct: int = 0
    visible_confident: int = 0
    occluded: int = 0
    occluded_confident: int = 0     # occluded but reported confident (bad)
    all_correct: int = 0            # argmax label correct, regardless of confidence (visible only)
    ro_total: int = 0               # visible red/orange stickers
    ro_confident_wrong: int = 0
    ro_flagged: int = 0             # red/orange reported ambiguous or low confidence
    ms: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def detection_rate(self) -> float:
        return self.detected / max(1, self.scenes)

    @property
    def confident_accuracy(self) -> float:
        return self.confident_correct / max(1, self.confident)

    @property
    def coverage(self) -> float:
        """Share of visible stickers that were reported confidently."""
        return self.visible_confident / max(1, self.visible)

    @property
    def raw_accuracy(self) -> float:
        return self.all_correct / max(1, self.visible)

    @property
    def occluded_leak(self) -> float:
        return self.occluded_confident / max(1, self.occluded)

    def summary(self, title: str = "") -> str:
        ms = np.array(self.ms) if self.ms else np.zeros(1)
        return (
            f"{title}: {self.scenes} scenes | face found {self.detection_rate:.1%} "
            f"(wrong {self.wrong_face}, missed {self.missed}) | stickers: confident accuracy "
            f"{self.confident_accuracy:.2%} on {self.confident}, coverage {self.coverage:.1%} of visible, "
            f"raw argmax accuracy {self.raw_accuracy:.2%} | occluded reported confident {self.occluded_leak:.1%} "
            f"of {self.occluded} | red/orange: {self.ro_confident_wrong} confident errors, "
            f"{self.ro_flagged}/{self.ro_total} flagged | detect {np.median(ms):.1f} ms median, "
            f"{np.percentile(ms, 95):.1f} ms p95"
        )


def evaluate(n: int, seed: int = 0, model: ColorModel | None = None, detector: FaceDetector | None = None,
             lighting_strength: float = 1.0, fixed_lighting: Lighting | None = None, **scene_kw) -> EvalReport:
    rng = np.random.default_rng(seed)
    model = model or ColorModel.default()
    detector = detector or FaceDetector()
    rep = EvalReport()
    for i in range(n):
        light = fixed_lighting or Lighting.random(rng, lighting_strength)
        sc = render_scene(rng, lighting=light, **scene_kw)
        rep.scenes += 1
        det = detector.detect(sc.image)
        if det is not None and not plausible_face(det.lab, det.seen, model):
            det = None
        if det is None:
            rep.missed += 1
            rep.errors.append(f"scene {i}: missed {sc.params}")
            continue
        rep.ms.append(det.ms)
        err = np.linalg.norm(det.centers - sc.centers, axis=1)
        if err.max() > 0.35 * sc.pitch:
            rep.wrong_face += 1
            rep.errors.append(f"scene {i}: wrong grid (max err {err.max() / sc.pitch:.2f} pitch) {sc.params}")
            continue
        rep.detected += 1
        reads = read_stickers(det.lab, det.confidence, model, det.occluded)
        for k in range(9):
            truth = sc.colors[k]
            r = reads[k]
            rep.stickers += 1
            conf = r.confidence >= CONFIDENT and r.label != "?"
            if sc.occluded[k]:
                rep.occluded += 1
                rep.occluded_confident += int(conf)
            else:
                rep.visible += 1
                rep.all_correct += int(r.label == truth)
                rep.visible_confident += int(conf)
                if truth in "RO":
                    rep.ro_total += 1
                    rep.ro_flagged += int(not conf or r.ambiguous is not None)
                    rep.ro_confident_wrong += int(conf and r.label != truth)
            if conf:
                rep.confident += 1
                rep.confident_correct += int(r.label == truth)
                if r.label != truth:
                    rep.errors.append(f"scene {i} sticker {k}: {truth} read as {r.label} "
                                      f"(conf {r.confidence:.2f}, occluded {sc.occluded[k]})")
    return rep


def calibrate_synthetic(rng: np.random.Generator, lighting: Lighting, detector: FaceDetector | None = None,
                        frames: int = 3) -> ColorModel:
    """Simulate the calibration flow: show each solved face (centre colour) a few times."""
    from vision.calibration import CalibrationSession

    detector = detector or FaceDetector()
    session = CalibrationSession()
    views = calibration_views()
    for color in session.order:
        for _ in range(frames * 3):
            sc = render_scene(rng, colors54=views[color], lighting=lighting, fingers=0)
            det = detector.detect(sc.image)
            if det is None or np.linalg.norm(det.centers - sc.centers, axis=1).max() > 0.35 * sc.pitch:
                continue
            session.add(color, det.lab, det.confidence)
            if session.count(color) >= frames * 9:
                break
    return session.build()


def evaluate_false_positives(n: int, seed: int = 0, detector: FaceDetector | None = None,
                             model: ColorModel | None = None) -> tuple[int, int]:
    """(detections that passed the plausibility check, n) on cube-free scenes."""
    rng = np.random.default_rng(seed)
    detector = detector or FaceDetector()
    model = model or ColorModel.default()
    hits = 0
    for _ in range(n):
        det = detector.detect(render_background(rng))
        hits += int(det is not None and plausible_face(det.lab, det.seen, model))
    return hits, n
