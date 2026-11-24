"""Stage-2 app logic: calibration / scan modes, scramble check, dataset recording.

main.py owns the timer; this controller owns everything vision-related and is
told every frame whether vision may run (`allowed` = timer IDLE and nothing in
the start zone). When it may not, nothing is submitted to the vision worker,
so timing frames are never slowed down.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

import cube
from cube import COLOR_NAMES, FACES, WCA_SCHEME
from ui import CYAN, GREEN, RED, WHITE, YELLOW, VisionView
from vision.calibration import CalibrationSession, PROMPTS
from vision.colors import ColorModel, balanced_assign
from vision.dataset import DatasetRecorder
from vision.face_detect import FaceDetector
from vision.scan import SCAN_ORDER, ScanSession, ScrambleCheck
from vision.worker import VisionResult, VisionWorker

FRESH = 0.4   # seconds a vision result stays on screen


class VisionController:
    def __init__(self, cfg: dict, resolve: Callable[[str], Path], say: Callable[[str, float], None],
                 have_camera: bool):
        vc, dc = cfg["vision"], cfg["dataset"]
        self.say = say
        self.enabled = bool(vc["enabled"]) and have_camera
        self.process_width = int(vc["process_width"])
        self.calibration_path = resolve(str(vc["calibration_path"]))
        self.model, warn = ColorModel.load(self.calibration_path)
        self.startup_messages: list[str] = [warn] if warn else []
        if self.model.calibrated:
            self.startup_messages.append(f"Colour calibration loaded ({self.calibration_path.name})")
        self.verify = bool(vc["auto_verify"])
        self.worker: VisionWorker | None = None
        if self.enabled:
            self.worker = VisionWorker(FaceDetector(process_width=self.process_width), self.model,
                                       still_time=float(vc["still_time"]))
        self.recorder: DatasetRecorder | None = None
        if have_camera:
            self.recorder = DatasetRecorder(resolve(str(dc["path"])), int(dc["jpeg_quality"]), int(dc["max_queue"]),
                                            float(dc["preroll"]), float(dc["tail"]), int(dc["max_width"]))
        self.mode: str | None = None               # None | "calibrate" | "scan"
        self.calib: CalibrationSession | None = None
        self.scan: ScanSession | None = None
        self.check = ScrambleCheck()
        self.scramble = ""
        self._active = False
        self._last_seq = -1
        self._need_motion = False
        self._last_hint = ("", 0.0)
        self._result: VisionResult | None = None
        self._now = 0.0

    # ------------------------------------------------------------------ queries
    @property
    def modal(self) -> bool:
        """While calibrating/scanning the hand trigger is ignored (holding the cube up must not arm the timer)."""
        return self.mode is not None

    # ------------------------------------------------------------------ per frame
    def set_scramble(self, scramble: str) -> None:
        if scramble != self.scramble:
            self.scramble = scramble
            self.check.set_scramble(scramble)

    def feed(self, img: np.ndarray, t: float, seq: int, mirrored: bool, allowed: bool, debug: bool) -> None:
        self._now = t
        if self.worker is None:
            return
        if not allowed and self.mode is not None:
            self.cancel_mode("Timer started - left " + self.mode + " mode")
        want = allowed and (self.mode is not None or self.verify or debug)
        if want:
            h, w = img.shape[:2]
            small = img
            if w > self.process_width:
                small = cv2.resize(img, (self.process_width, int(h * self.process_width / w)),
                                   interpolation=cv2.INTER_AREA)
            else:
                small = img.copy()
            self.worker.submit(small, t, seq, mirrored)
            self._active = True
        elif self._active:
            self.worker.reset()
            self._active = False
            self._result = None
            return
        res = self.worker.latest
        if res is None or res.seq == self._last_seq:
            return
        self._last_seq = res.seq
        self._result = res
        if res.face is None:
            if res.det is None or res.stable_progress < 0.5:
                self._need_motion = False
            return
        self._on_stable(res)

    def _hint(self, msg: str | None, seconds: float = 2.5) -> None:
        """say() without repeating the same message every frame."""
        if not msg:
            return
        now = time.perf_counter()
        if msg != self._last_hint[0] or now - self._last_hint[1] > seconds:
            self._last_hint = (msg, now)
            self.say(msg, seconds)

    def _on_stable(self, res: VisionResult) -> None:
        face = res.face
        if self.mode == "calibrate" and self.calib is not None:
            if self._need_motion:
                return
            color = self.calib.current
            centre_lab = np.asarray(face.lab[4])
            # protect against showing a face twice (e.g. white again when yellow is asked for)
            for other in self.calib.order[: self.calib.index]:
                m = np.median(np.asarray(self.calib.samples[other]), axis=0)
                if np.hypot(*(m[1:] - centre_lab[1:])) < 8 and abs(m[0] - centre_lab[0]) < 15:
                    self._hint(f"That looks like the {PROMPTS[other]} face again - show {PROMPTS[color]}")
                    return
            advanced = self.calib.add_current(face.lab, face.confidence)
            if advanced:
                self._need_motion = True
                if self.calib.done:
                    self._finish_calibration()
                else:
                    self._hint(f"{PROMPTS[color]} done. Now show the {PROMPTS[self.calib.current]} centre", 3.0)
        elif self.mode == "scan" and self.scan is not None:
            if self.scan.target is None or self._need_motion:
                return
            msg = self.scan.offer(face)
            if msg and msg.startswith("Captured"):
                self._need_motion = True
                self.say(msg, 3.0)
                if self.scan.complete:
                    self._finish_scan()
            else:
                self._hint(msg)
        elif self.verify:
            self.check.add(face)

    # ------------------------------------------------------------------ modes
    def start_calibration(self) -> None:
        self.mode = "calibrate"
        self.calib = CalibrationSession()
        self._need_motion = False
        self.say(f"Calibration: show each centre to the camera and hold still. "
                 f"Start with the {PROMPTS[self.calib.current]} centre. ESC cancels.", 5.0)

    def _finish_calibration(self) -> None:
        model = self.calib.build()
        warnings = self.calib.check_distinct(model)
        try:
            model.save(self.calibration_path)
            saved = f"saved to {self.calibration_path.name}"
        except OSError as exc:
            saved = f"NOT saved ({exc})"
        self.model = model
        if self.worker is not None:
            self.worker.set_model(model)
        self.mode = None
        self.calib = None
        self.say("Calibration complete, " + saved + (". WARNING: " + "; ".join(warnings) if warnings else ""), 7.0)

    def start_scan(self) -> None:
        self.mode = "scan"
        self.scan = ScanSession()
        self._need_motion = False
        self.say("Scan: hold the cube in WCA orientation (white top, green front) and turn it to show each face "
                 "as asked. ESC cancels.", 5.0)

    def _finish_scan(self) -> None:
        model = self.model
        res = self.scan.finalize(lambda labs, cidx: balanced_assign(labs, model, cidx))
        if res.ok:
            match = res.colors54 == cube.to_colors(cube.scrambled_state(self.scramble)) if self.scramble else None
            extra = "" if match is None else (" - matches the current scramble" if match
                                              else " - does NOT match the current scramble")
            self.say("Scan complete: valid cube" + extra, 6.0)
        else:
            self.say("Scan complete but the cube is not valid: " + "; ".join(res.errors[:2]), 7.0)

    def cancel_mode(self, msg: str = "") -> None:
        self.mode = None
        self.calib = None
        self.scan = None
        if msg:
            self.say(msg, 3.0)

    # ------------------------------------------------------------------ keys
    def on_key(self, key: int, ch: str, idle: bool) -> bool:
        """Handle a key; returns True if consumed."""
        code = key & 0xFF
        if self.mode is not None:
            if code == 27:
                self.cancel_mode(f"Left {self.mode} mode")
                return True
            if self.mode == "calibrate" and code in (8, 127):
                self.calib.redo_previous()
                self._need_motion = False
                self.say(f"Redo: show the {PROMPTS[self.calib.current]} centre", 3.0)
                return True
            if self.mode == "scan":
                if code in (8, 127) and self.scan.reads:
                    last = [f for f, _, _ in SCAN_ORDER if f in self.scan.reads][-1]
                    self.scan.rescan(last)
                    self._need_motion = False
                    self.say("Rescan: " + self.scan.instruction(), 3.0)
                    return True
                if ch and ch in "123456":
                    face = SCAN_ORDER[int(ch) - 1][0]
                    self.scan.rescan(face)
                    self._need_motion = False
                    self.say("Rescan: " + self.scan.instruction(), 3.0)
                    return True
                if code == 13 or ch == "s":
                    self.cancel_mode("Scan closed")
                    return True
            if ch == "c" and self.mode == "calibrate":
                self.cancel_mode("Calibration cancelled")
                return True
            # swallow other timer keys while a mode is open, except quit/debug/check/record
            return ch not in ("q", "h", "v", "r")
        if ch == "v":
            self.verify = not self.verify
            self.check.faces.clear()
            self.say(f"Scramble check {'ON' if self.verify else 'OFF'}" +
                     (" (show faces to the camera while IDLE)" if self.verify else ""))
            return True
        if ch == "r":
            if self.recorder is None:
                self.say("Dataset recording needs a camera")
            else:
                on = self.recorder.toggle()
                self.say(f"Dataset recording {'ON - solves will be saved to ' + str(self.recorder.root.name) if on else 'OFF'}")
            return True
        if ch in ("c", "s"):
            if self.worker is None:
                self.say("Cube vision needs a camera (and vision.enabled: true)")
            elif not idle:
                self.say("Only while the timer is idle")
            elif ch == "c":
                self.start_calibration()
            else:
                self.start_scan()
            return True
        return False

    # ------------------------------------------------------------------ timer events
    def on_solve_begin(self, t: float, meta: dict) -> None:
        if self.recorder is not None:
            self.recorder.begin(t, meta)

    def on_solve_end(self, t: float, result: dict) -> None:
        if self.recorder is not None:
            self.recorder.end(t, result)

    def on_abort(self, t: float) -> None:
        if self.recorder is not None:
            self.recorder.abort(t)

    # ------------------------------------------------------------------ view
    def view(self, debug: bool) -> VisionView:
        v = VisionView()
        rec = self.recorder
        if rec is not None:
            v.rec_on, v.rec_active = rec.enabled, rec.recording
        res = self._result
        fresh = res is not None and res.det is not None and self._active and self._now - res.t < FRESH
        if fresh:
            v.corners, v.points = res.corners, res.points
            v.progress = res.stable_progress
            src = res.face
            if src is not None:
                v.colors, v.confs = src.colors, src.confidence
                v.labels = [f"{a[0]}/{a[1]}?" if a else c for c, a in zip(src.colors, src.ambiguous)]
            else:
                v.colors = [r.label for r in res.live]
                v.confs = [r.confidence for r in res.live]
                v.labels = [r.text for r in res.live]
            v.show_grid = debug or self.mode is not None
        if self.mode is None and self.verify and self.worker is not None:
            last = self.check.last
            if fresh and res.face is not None and last is not None and last.face is not None:
                v.status = last.sticker_status
            verdict = self.check.verdict
            if verdict != "none":
                v.check_text = self.check.summary()
                v.check_color = {"ok": GREEN, "mismatch": RED, "unsure": YELLOW}[verdict]
        if self.mode == "calibrate" and self.calib is not None:
            c = self.calib.current
            v.panel_title = f"CALIBRATE {self.calib.index + 1}/6"
            v.panel_lines = [
                f"Show the {PROMPTS[c]} centre to the camera, face-on, and hold still." if c else "Done",
                f"frames: {self.calib.frames.get(c, 0)}/{self.calib.frames_needed}" if c else "",
                "A solved face gives the best result. Even, diffuse light; avoid glare.",
                "BKSP redo previous colour   ESC cancel",
            ]
        elif self.mode == "scan" and self.scan is not None:
            sc = self.scan
            net = sc.colors54()
            if sc.target is not None and fresh and res.face is not None and res.face.centre == WCA_SCHEME[sc.target]:
                i = FACES.index(sc.target) * 9
                net[i:i + 9] = res.face.colors
            v.net, v.net_target = net, sc.target
            v.panel_title = f"SCAN {len(sc.reads)}/6"
            lines = [sc.instruction()] if sc.target else []
            if sc.result is not None:
                r = sc.result
                lines.append("VALID cube" if r.ok else "NOT VALID: " + "; ".join(r.errors[:3]))
                lines += r.notes
                if r.ok and self.scramble:
                    same = r.colors54 == cube.to_colors(cube.scrambled_state(self.scramble))
                    lines.append("matches current scramble" if same else "differs from current scramble")
                v.net = list(r.colors54) if r.colors54 else net
                v.net_marks = set(r.reassigned)
            order = "  ".join(f"{i + 1}={COLOR_NAMES[c][:3]}" for i, (_, c, _) in enumerate(SCAN_ORDER))
            lines.append(f"rescan: {order}   BKSP last   ENTER/ESC close")
            v.panel_lines = lines
        return v

    def debug_info(self) -> dict:
        d = {}
        if self.worker is not None:
            hist = self.worker.ms_history[-30:]
            d["vision"] = "on" if self._active else "paused (not idle)"
            if hist:
                d["vision ms"] = f"{np.median(hist):.1f}"
            res = self._result
            if res is not None and res.det is not None:
                d["face"] = f"{res.det.method}, {res.det.n_seen}/9 outlines, {res.det.debug.get('candidates', 0)} cands"
            if res is not None and res.error:
                d["vision error"] = res.error[:60]
            d["colours"] = "calibrated" if self.model.calibrated else "defaults (press C)"
        return d

    def status_line(self) -> str | None:
        parts = []
        if self.worker is not None:
            parts.append(f"cube vision {'ON' if self._active else 'paused'}"
                         f" | scramble check {'ON' if self.verify else 'OFF'}"
                         f" | colours {'calibrated' if self.model.calibrated else 'default'}")
        if self.recorder is not None and self.recorder.enabled:
            parts.append(self.recorder.status_text())
        return " | ".join(parts) if parts else None

    def close(self) -> None:
        if self.worker is not None:
            self.worker.close()
        if self.recorder is not None:
            self.recorder.close()
