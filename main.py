"""CubeCam - webcam-triggered Rubik's cube timer (stage 1).

Run:  python main.py [--config config.yaml] [--no-camera]
"""
from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from camera import Camera
from ar_mat import Mat, reanchor
from config import DEFAULTS, ZONE_MODES, load_config, save_blocks
from detection import DetectorUnavailable, NullDetector, TriggerSample
from keyinput import SpaceKey
from models import Penalty, Solve, format_solve, format_time
from scramble import generate_scramble
from stats import compute_stats
from storage import SolveStore
from timer_state import EventType, State, TimerConfig, TimerStateMachine
from plane import TablePlane
from table_calib import CalibMode, TableCalibration
from ui import GREEN, ORANGE, RED, WHITE, YELLOW, ArView, Renderer, ViewModel
from zone import Zone, zone_warnings
from zone_setup import ROTATE_STEP, SetupMode, ZoneSetup

BASE = Path(__file__).resolve().parent


class RateMeter:
    """Samples-per-second over a sliding window of timestamps."""

    def __init__(self, n: int = 60):
        self._t: deque[float] = deque(maxlen=n)

    def tick(self, t: float) -> None:
        self._t.append(t)

    @property
    def rate(self) -> float:
        if len(self._t) < 2:
            return 0.0
        span = self._t[-1] - self._t[0]
        return (len(self._t) - 1) / span if span > 0 else 0.0


def beep(freq: int, ms: int) -> None:
    def _run():
        try:
            import winsound

            winsound.Beep(freq, ms)
        except Exception:
            sys.stdout.write("\a")
            sys.stdout.flush()

    threading.Thread(target=_run, daemon=True).start()


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else BASE / p


class App:
    def __init__(self, config_path: Path, force_no_camera: bool = False):
        self.cfg, cfg_warning = load_config(config_path)
        c = self.cfg
        t = c["timer"]
        self.machine = TimerStateMachine(
            TimerConfig(
                hold_time=float(t["hold_time"]),
                debounce=float(t["debounce"]),
                inspection_enabled=bool(t["inspection"]),
                inspection_limit=float(t["inspection_limit"]),
                dnf_limit=float(t["dnf_limit"]),
                warnings=tuple(float(w) for w in t["warnings"]),
            )
        )
        self.audio = bool(t["audio"])
        self.window = str(c["ui"]["window_name"])
        self.show_debug = bool(c["ui"]["show_debug"])
        self.config_path = Path(config_path)
        zone_msgs: list[str] = []
        try:
            self.zone = Zone.from_config(c["zone"])
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            self.zone = Zone.from_config(DEFAULTS["zone"])
            zone_msgs.append(f"Zone in config is invalid ({exc}) - using the default zone")
        self.zone_mode = str(c["zone"].get("mode", "fixed"))
        if self.zone_mode not in ZONE_MODES:
            zone_msgs.append(f"Unknown zone mode {self.zone_mode!r} - using fixed")
            self.zone_mode = "fixed"
        # AR mat on the table plane (T calibrates the plane, M places the mat)
        mc = c["mat"]
        self.ar_enabled = bool(mc.get("ar", True))
        self.mat_size = (float(mc.get("width_cm", 30.0)), float(mc.get("height_cm", 22.0)))
        self.plane: TablePlane | None = None
        self.mat: Mat | None = None
        try:
            self.plane = TablePlane.from_config(c.get("table_calibration"))
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            zone_msgs.append(f"Table calibration in config is invalid ({exc}) - press T to calibrate again")
        try:
            self.mat = Mat.from_config(c.get("mat_placement"))
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            zone_msgs.append(f"Mat placement in config is invalid ({exc}) - press M to place it again")
        tc = c["table"]
        mm = tc.get("phone_marker_mm")
        hfov = tc.get("camera_hfov_deg")
        self.calib = TableCalibration(float(mm) if mm else None, int(tc.get("phone_marker_id", 10)),
                                      hfov_deg=float(hfov) if hfov else None)
        self.phone = None            # ArucoWorker while calibrating with the phone marker
        self._phone_t: float | None = None
        self._hands_px: list = []
        self._hands_age = 0
        self.markers = None          # MarkerWorker while marker mode (or its setup) is on
        self.setup: ZoneSetup | None = None
        self._canvas = (0, 0)
        self._zone_warn_cache: tuple[Zone | None, list[str]] = (None, [])
        self.store = SolveStore(_resolve(c["storage"]["path"]))
        self.solves: list[Solve] = self.store.load()
        self.scramble = generate_scramble()
        self.renderer = Renderer()
        self.space = SpaceKey(self.window)
        self.loop_rate = RateMeter()
        self.camera: Camera | None = None
        self.detector = NullDetector()
        self.mirror = bool(c["camera"]["mirror"])
        self._message: str | None = None
        self._message_until = 0.0
        self._flash_color = None
        self._flash_until = 0.0
        self._last_sample = TriggerSample(0.0, False)

        startup_msgs = [m for m in (cfg_warning, self.store.warning) if m] + zone_msgs
        startup_msgs += self._setup_camera(force_no_camera)
        startup_msgs += self._setup_zone()
        if self.solves:
            startup_msgs.append(f"Loaded {len(self.solves)} solves from {self.store.path.name}")
        if startup_msgs:
            self.say("  |  ".join(startup_msgs), 8.0)
        for m in startup_msgs:
            print("[CubeCam]", m)

    # ------------------------------------------------------------------ setup
    def _setup_camera(self, force_no_camera: bool) -> list[str]:
        cc = self.cfg["camera"]
        if force_no_camera or not cc["enabled"]:
            return ["Keyboard mode (camera disabled). Hold SPACE, release to start, press to stop."]
        cam = Camera(int(cc["index"]), int(cc["width"]), int(cc["height"]), int(cc["fps"]),
                     str(cc["backend"]), cc.get("fourcc") or None)
        if not cam.open():
            return [f"No camera available ({cam.error}). Keyboard mode: hold SPACE, release to start, press to stop."]
        self.camera = cam
        det_cfg = dict(self.cfg["detection"])
        det_cfg["model_path"] = str(_resolve(det_cfg["model_path"]))
        try:
            from hands import HandZoneDetector

            self.detector = HandZoneDetector(self.zone, det_cfg)
        except DetectorUnavailable as exc:
            return [f"Hand tracking unavailable ({exc}). Camera shown, use SPACE to time."]
        except ImportError as exc:
            return [f"Hand tracking unavailable ({exc}). Camera shown, use SPACE to time."]
        return []

    def _setup_zone(self) -> list[str]:
        from markers import aruco_available

        msgs = []
        self.setup = ZoneSetup(hands_available=self.hand_mode,
                               markers_available=self.camera is not None and aruco_available())
        if self.zone_mode == "markers":
            if self.camera is None:
                msgs.append("Zone mode is 'markers' but there is no camera - using the saved zone")
            elif not aruco_available():
                msgs.append("Zone mode is 'markers' but OpenCV has no aruco module "
                            "(install opencv-contrib-python) - using the saved zone")
            else:
                self._start_markers()
        if self.camera is not None:
            size = (self.camera.width, self.camera.height)
            self.calib.image_size = size
            if self.plane is not None:
                self.plane = self.plane.scaled_to(size)   # camera resolution may have changed
        if self.zone_mode == "mat":
            if self.plane is None or self.mat is None:
                msgs.append("Zone mode is 'mat' but there is no table calibration / mat - using the saved "
                            "outline. Press T to calibrate the table.")
            else:
                z = self.mat.zone(self.plane)
                if z is None:
                    msgs.append("The saved mat isn't on the visible table any more - press M to place it")
                else:
                    self.set_zone(z)
        msgs += [f"{w}. Press M to set up the zone." for w in zone_warnings(self.zone)]
        return msgs

    # ------------------------------------------------------------------- zone
    def set_zone(self, zone: Zone) -> None:
        self.zone = zone
        if hasattr(self.detector, "zone"):
            self.detector.zone = zone

    def _start_markers(self) -> None:
        if self.markers is None:
            from markers import MarkerWorker

            self.markers = MarkerWorker(self.cfg.get("markers"), mirror=self.mirror)

    def _stop_markers(self) -> None:
        if self.markers is not None:
            self.markers.close()
            self.markers = None

    def zone_warnings(self) -> list[str]:
        if self._zone_warn_cache[0] is not self.zone:
            self._zone_warn_cache = (self.zone, zone_warnings(self.zone))
        return self._zone_warn_cache[1]

    def active_plane(self) -> TablePlane | None:
        """The table plane in use: the 4 mat markers' while marker mode runs, else the T calibration."""
        if self.zone_mode == "markers" and self.markers is not None:
            p = self.markers.plane_snapshot()
            if p is not None:
                return p
        return self.plane

    def start_zone_setup(self, mode: SetupMode | None = None) -> None:
        if self.camera is None or not self.camera.alive:
            self.say("Zone setup needs the camera")
            return
        plane = self.active_plane()
        current = self.mat if (self.zone_mode == "mat" and plane is self.plane) else None
        self.setup.start(mode, plane=plane, mat_size=self.mat_size, current_mat=current)

    def _setup_key(self, key: int, ch: str) -> None:
        """Keys while the zone setup is open (everything except Q and H comes here)."""
        setup = self.setup
        code = key & 0xFF
        if code == 27:
            setup.cancel()
            self.say("Zone setup cancelled - zone unchanged")
        elif code in (10, 13):
            plane_used = setup.plane
            result = setup.accept()
            if result is None:
                self.say("Not all 4 markers seen yet" if setup.mode is SetupMode.MARKERS else "No zone yet")
                return
            mode, zone, mat = result
            blocks = {}
            if mode == "mat":
                if plane_used is not self.plane:       # e.g. the plane came from the 4 markers
                    self.plane = plane_used
                    blocks["table_calibration"] = plane_used.to_config()
                self.mat = mat
                blocks["mat_placement"] = mat.to_config()
            self.zone_mode = mode
            self.set_zone(zone)
            saved = self._save(zone=(mode, zone.to_config()), **blocks)
            warn = zone_warnings(zone)
            what = {"markers": "follows the markers", "mat": "AR mat"}.get(mode, mode)
            self.say(f"Zone set ({what}), {saved}" + (f". Warning: {'; '.join(warn)}" if warn else ""),
                     5.0 if warn else 3.0)
        elif ch in ("[", "]"):
            if not setup.rotate(ROTATE_STEP if ch == "]" else -ROTATE_STEP):
                self.say("Nothing to rotate (or it would leave the visible table)")
        elif ch == "m":
            if setup.cycle() is None:
                self.say("Zone setup closed - zone unchanged")
        elif ch == "r":
            setup.redo()
        # The marker detector runs only while marker mode is in use or being set up.
        if setup.active and setup.mode is SetupMode.MARKERS:
            self._start_markers()
        elif self.zone_mode != "markers":
            self._stop_markers()

    def on_mouse(self, event, x, y, flags, param) -> None:
        if self.calib.active:
            self.calib.on_mouse(event, x, y, *self._canvas)
        elif self.setup is not None and self.setup.active:
            self.setup.on_mouse(event, x, y, *self._canvas)

    def _save(self, zone=None, **blocks) -> str:
        try:
            save_blocks(self.config_path, zone=zone, **blocks)
            return f"saved to {self.config_path.name}"
        except (OSError, ValueError) as exc:
            return f"NOT saved ({exc})"

    # ------------------------------------------------------- table calibration
    def start_calibration(self) -> None:
        from markers import aruco_available

        if self.camera is None or not self.camera.alive:
            self.say("Table calibration needs the camera")
            return
        self.calib.start(CalibMode.PHONE, (self.camera.width, self.camera.height))
        if aruco_available():
            self._start_phone()
        else:
            self.calib.cycle()   # no aruco module: straight to the click method
            self.say("OpenCV has no aruco module - click 4 corners of a known rectangle instead")

    def _start_phone(self) -> None:
        if self.phone is None:
            from markers import ArucoWorker

            tc = self.cfg["table"]
            # full resolution + sub-pixel corners: calibration runs rarely, precision matters
            self.phone = ArucoWorker(str(tc.get("dictionary", "DICT_4X4_50")), self.mirror, interval=0.12,
                                     process_width=0, refine=True)
            self._phone_t = None

    def _stop_phone(self) -> None:
        if self.phone is not None:
            self.phone.close()
            self.phone = None

    def _update_phone(self, raw_image, t: float) -> None:
        if self.phone is None:
            return
        self.phone.submit(raw_image, t)
        corners, ct = self.phone.latest()
        if ct is not None and ct != self._phone_t:
            self._phone_t = ct
            self.calib.update_phone(t, corners, self.phone.image_size)
        else:
            self.calib.update_phone(t, None)

    def _calib_key(self, key: int, ch: str) -> None:
        calib = self.calib
        code = key & 0xFF
        if calib.on_key(key):
            pass
        elif code == 27:
            calib.cancel()
            self.say("Table calibration cancelled")
        elif code in (10, 13):
            plane = calib.accept()
            if plane is None:
                self.say("Nothing to accept yet" + (" - show the phone marker" if calib.mode is CalibMode.PHONE
                                                    else " - click 4 corners and give the size"))
            else:
                self._apply_plane(plane)
        elif ch == "t":
            if calib.cycle() is None:
                self.say("Table calibration closed - unchanged")
        elif ch == "x":
            calib.cancel()
            self._forget_plane()
        if not (calib.active and calib.mode is CalibMode.PHONE):
            self._stop_phone()

    def _apply_plane(self, plane: TablePlane) -> None:
        old = self.plane
        self.plane = plane
        blocks: dict = {"table_calibration": plane.to_config()}
        zone = None
        if self.zone_mode == "mat" and self.mat is not None and old is not None:
            moved = reanchor(self.mat, old, plane)   # keep the mat where it was on screen
            if moved is not None:
                self.mat = moved
                z = moved.zone(plane)
                self.set_zone(z)
                blocks["mat_placement"] = moved.to_config()
                zone = ("mat", z.to_config())
        saved = self._save(zone=zone, **blocks)
        note = "" if self.calib.scale_known or plane.source != "phone" else \
            " Scale approximate - set table.phone_marker_mm."
        warn = plane.warnings()
        self.say(f"Table calibrated ({plane.source}, camera ~{plane.elevation_deg():.0f} deg above the table), "
                 f"{saved}.{note}" + (f" Warning: {'; '.join(warn)}" if warn else ""), 6.0)
        if self.zone_mode != "mat":
            self.start_zone_setup()       # next step: place the mat

    def _forget_plane(self) -> None:
        if self.plane is None:
            self.say("No table calibration to forget")
            return
        self.plane = None
        zone = None
        if self.zone_mode == "mat":
            self.zone_mode = "fixed"          # keep the mat's outline as a flat zone
            zone = ("fixed", self.zone.to_config())
        saved = self._save(zone=zone, table_calibration=None)
        self.say(f"Table calibration forgotten ({saved}); the zone stays where it was")

    # --------------------------------------------------------------- AR view
    def _hands_in_px(self, sample: TriggerSample) -> list:
        """Hand landmarks in canvas pixels; the last ones are kept for 3 frames if tracking flickers."""
        hands = sample.debug.get("hands")
        W, H = self._canvas
        if hands:
            self._hands_px = [np.asarray(h, dtype=np.float64)[:, :2] * (W, H) for h in hands]
            self._hands_age = 0
        else:
            self._hands_age += 1
            if self._hands_age > 3:
                self._hands_px = []
        return self._hands_px

    def _marker_mat_size(self, quad: np.ndarray) -> tuple[float, float]:
        plane = self.markers.plane_snapshot() if self.markers is not None else None
        if plane is None:
            return self.mat_size
        c, ok = plane.to_table(quad)
        if not ok.all():
            return self.mat_size
        w = (np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[3])) / 2
        h = (np.linalg.norm(c[3] - c[0]) + np.linalg.norm(c[2] - c[1])) / 2
        return (round(float(w), 1), round(float(h), 1)) if w > 2 and h > 2 else self.mat_size

    def ar_view(self, zone_mode: str, progress: float, sample: TriggerSample) -> ArView | None:
        hands = self._hands_in_px(sample)
        if not self.ar_enabled or zone_mode == "off" or self.calib.active:
            return None
        quad = size = None
        state = zone_mode if zone_mode in ("idle", "holding", "ready", "solving") else "idle"
        if self.setup is not None and self.setup.active:
            m, plane = self.setup.candidate_mat, self.setup.plane
            if m is not None and plane is not None:
                quad, size = m.image_quad(plane), (m.width_cm, m.height_cm)
        elif self.zone_mode == "mat" and self.plane is not None and self.mat is not None:
            quad, size = self.mat.image_quad(self.plane), (self.mat.width_cm, self.mat.height_cm)
        elif self.zone_mode == "markers" and self.markers is not None and len(self.zone.points) == 4:
            quad = np.array(self.zone.points) * self._canvas
            size = self._marker_mat_size(quad)
        if quad is None:
            return None
        return ArView(quad, size, state, progress if state == "holding" else 0.0, hands)

    # ---------------------------------------------------------------- helpers
    def say(self, msg: str, seconds: float = 3.0) -> None:
        self._message = msg
        self._message_until = time.perf_counter() + seconds

    @property
    def hand_mode(self) -> bool:
        return self.camera is not None and self.camera.alive and not isinstance(self.detector, NullDetector)

    def effective_fps(self) -> float:
        """Rate at which the state machine actually receives samples."""
        return self.loop_rate.rate

    # ------------------------------------------------------------------ events
    def handle_events(self, events) -> None:
        for e in events:
            if e.type == EventType.INSPECTION_WARNING:
                sec = int(e.data)
                self._flash_color = ORANGE if sec < 12 else RED
                self._flash_until = time.perf_counter() + 0.7
                self.say(f"{sec} SECONDS!", 1.5)
                if self.audio:
                    beep(880 if sec < 12 else 1320, 150 if sec < 12 else 300)
            elif e.type == EventType.SOLVE_STOP:
                r = e.data
                fps = self.effective_fps()
                solve = Solve(
                    time=round(r.time, 4),
                    penalty=r.penalty,
                    scramble=self.scramble,
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                    fps=round(fps, 1) if fps else None,
                    inspection=None if r.inspection_time is None else round(r.inspection_time, 3),
                )
                self.solves.append(solve)
                self.save()
                self.scramble = generate_scramble()
                if r.penalty != Penalty.OK:
                    self.say(f"Inspection {r.inspection_time:.2f}s -> {r.penalty.value}", 4.0)
            elif e.type == EventType.ABORT:
                self.say("Aborted - solve not recorded", 2.0)

    def save(self) -> None:
        try:
            self.store.save(self.solves)
        except OSError as exc:
            self.say(f"Could not save solves: {exc}", 5.0)

    def handle_key(self, key: int, t: float) -> bool:
        """Returns False to quit."""
        if key == -1:
            return True
        if self.space.on_key(key):
            return True
        ch = chr(key & 0xFF).lower() if 0 <= (key & 0xFF) < 128 else ""
        busy = self.machine.state in (State.SOLVING, State.INSPECTION, State.READY)
        if ch == "q":
            return False
        if self.calib.active and ch != "h":
            self._calib_key(key, ch)
            return True
        if self.setup is not None and self.setup.active and ch != "h":
            self._setup_key(key, ch)
            return True
        if ch in ("m", "t"):
            if busy:
                self.say("Finish or abort (ESC) the current solve first")
            elif ch == "m":
                self.start_zone_setup()
            else:
                self.start_calibration()
            return True
        if ch == "a":
            self.ar_enabled = not self.ar_enabled
            has_mat = self.zone_mode in ("mat", "markers") and (self.plane is not None or self.markers is not None)
            self.say(f"AR mat {'ON' if self.ar_enabled else 'OFF (plain zone outline)'}"
                     + ("" if has_mat or not self.ar_enabled else " - calibrate the table (T) and place the mat (M)"))
            return True
        if key & 0xFF == 27:
            self.handle_events(self.machine.abort(t))
        elif ch in ("2", "d") or key & 0xFF in (8, 127):
            if busy:
                self.say("Finish or abort (ESC) the current solve first")
            elif not self.solves:
                self.say("No solves yet")
            elif ch == "2":
                s = self.solves[-1]
                s.penalty = Penalty.OK if s.penalty == Penalty.PLUS2 else Penalty.PLUS2
                self.save()
                self.say(f"Last solve: {format_solve(s)}")
            elif ch == "d":
                s = self.solves[-1]
                s.penalty = Penalty.OK if s.penalty == Penalty.DNF else Penalty.DNF
                self.save()
                self.say(f"Last solve: {format_solve(s)}")
            else:
                s = self.solves.pop()
                self.save()
                self.say(f"Deleted {format_solve(s)}")
        elif ch == "i":
            if busy:
                self.say("Can't change inspection during a solve")
            else:
                cfg = self.machine.config
                cfg.inspection_enabled = not cfg.inspection_enabled
                self.say(f"Inspection {'ON' if cfg.inspection_enabled else 'OFF'}")
        elif ch == "h":
            self.show_debug = not self.show_debug
        elif ch == "n" and not busy:
            self.scramble = generate_scramble()
        return True

    # ------------------------------------------------------------------- view
    def build_view(self, t: float, sample: TriggerSample, space_down: bool, canvas_w: int) -> ViewModel:
        m = self.machine
        st = m.state
        now = time.perf_counter()
        color, sub = WHITE, ""
        if m.in_inspection:
            el = m.inspection_elapsed(t) or 0.0
            cfg = m.config
            if el <= cfg.inspection_limit:
                text = str(max(1, math.ceil(cfg.inspection_limit - el)))
            elif el <= cfg.dnf_limit:
                text = "+2"
            else:
                text = "DNF"
            color = RED if el >= cfg.warnings[-1] else ORANGE if el >= cfg.warnings[0] else WHITE
            sub = "inspection - rest hands in zone to arm"
            if st == State.READY:
                color, sub = GREEN, "READY - lift to start"
            elif m.hold_progress(t) > 0:
                color = YELLOW
        elif st == State.SOLVING:
            text = format_time(m.solve_elapsed(t))
        else:
            text = format_solve(self.solves[-1]) if self.solves else "0.00"
            if st == State.READY:
                color = GREEN
                sub = "lift to start inspection" if m.config.inspection_enabled else "READY - lift to start"
            elif m.hold_progress(t) > 0:
                color = YELLOW
        if not self.hand_mode:
            zone_mode = "off"
        elif st == State.READY:
            zone_mode = "ready"
        elif st == State.SOLVING:
            zone_mode = "solving"
        elif st in (State.IDLE, State.INSPECTION) and m.zone_active:
            zone_mode = "holding"
        else:
            zone_mode = "idle"

        status = []
        eff = self.effective_fps()
        res = f"+/-{1000 / eff:.0f} ms (1 sample)" if eff > 0 else "-"
        if self.camera is not None and self.camera.alive:
            cam = self.camera
            status.append(
                f"Camera {cam.width}x{cam.height} via {cam.backend_name}: capture {cam.measured_fps:.0f} fps "
                f"(driver says {cam.reported_fps:.0f}) | processed {eff:.0f} fps | timing resolution {res}"
            )
        else:
            status.append(f"Keyboard mode | loop {eff:.0f} Hz | timing resolution {res}")
        zone_txt = {"markers": "markers", "mat": "AR mat" if self.ar_enabled else "mat (AR off)"}.get(
            self.zone_mode, "rectangle" if self.zone.is_rect else "polygon")
        if self.markers is not None and self.zone_mode == "markers":
            zone_txt += f" ({self.markers.snapshot()[1]})"
        status.append(
            f"Zone: {zone_txt} | State {st.value} | inspection {'ON' if m.config.inspection_enabled else 'OFF'} | "
            f"trigger: {'hands' if self.hand_mode else ''}{' + ' if self.hand_mode else ''}SPACE"
            f"{'' if self.space.polled else ' (tap toggles)'}"
        )

        zone_warn = self.zone_warnings()
        if zone_warn:
            status.append("ZONE WARNING: " + "; ".join(zone_warn) + " - press M to set up the zone")
        if self.markers is not None and self.markers.error:
            status.append(f"Marker detection error: {self.markers.error}")
        setup_view = self.setup.view() if self.setup is not None else None
        calib_view = self.calib.view()
        if setup_view is not None or calib_view is not None:
            zone_mode = "setup"
        ar = self.ar_view(zone_mode if setup_view is None else "setup", m.hold_progress(t), sample)

        debug = {
            "state": st.value,
            "raw trigger": m.raw_active,
            "debounced zone": m.zone_active,
            "space down": space_down,
            "detector": getattr(self.detector, "backend_name", self.detector.name),
        }
        if "hands" in sample.debug:
            debug["hands seen"] = len(sample.debug["hands"])
            debug["palms in zone"] = sum(sample.debug["palms_in_zone"])
            debug["detect ms"] = f"{sample.debug['detect_ms']:.1f}"
            debug.update({k: sample.debug[k] for k in ("hands", "palms", "palms_in_zone")})
        if self.markers is not None:
            debug["marker ms"] = f"{self.markers.detect_ms:.1f}"
        if self.plane is not None:
            debug["table"] = f"{self.plane.source}, {self.plane.elevation_deg():.0f} deg"
        if ar is not None:
            debug["AR ms"] = f"{self.renderer.ar.last_ms:.1f}"

        stats = compute_stats(self.solves)
        return ViewModel(
            state=st,
            time_text=text,
            time_color=color,
            sub_text=sub,
            scramble=self.scramble,
            stats=stats,
            recent=[format_solve(s) for s in self.solves[-5:]],
            zone=self.zone,
            zone_mode=zone_mode,
            hold_progress=m.hold_progress(t),
            status_lines=status,
            message=self._message if now < self._message_until else None,
            flash=self._flash_color if now < self._flash_until else None,
            show_debug=self.show_debug,
            debug=debug,
            setup=setup_view,
            zone_warning=bool(zone_warn),
            calib=calib_view,
            ar=ar,
        )

    # ------------------------------------------------------------------- loop
    def run(self) -> None:
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        if self.camera is not None:
            cv2.resizeWindow(self.window, self.camera.width, self.camera.height)
            canvas_size = (self.camera.width, self.camera.height)
        else:
            canvas_size = tuple(int(v) for v in self.cfg["ui"]["keyboard_canvas"])
            cv2.resizeWindow(self.window, *canvas_size)
        cv2.setMouseCallback(self.window, self.on_mouse)
        last_seq = 0
        shown = False
        try:
            while True:
                if self.camera is not None and self.camera.alive:
                    frame = self.camera.wait_new(last_seq, timeout=0.05)
                    if frame is None:
                        if not self.handle_key(cv2.waitKey(1), self.machine._last_t or time.perf_counter()):
                            break
                        continue
                    last_seq = frame.seq
                    t = frame.t
                    img = cv2.flip(frame.image, 1) if self.mirror else frame.image.copy()
                    self._update_markers(frame.image, t)
                    self._update_phone(frame.image, t)
                    try:
                        sample = self.detector.process(img, t)
                    except Exception as exc:  # never let detection kill the timer
                        self.say(f"Hand detection error: {exc}", 3.0)
                        sample = TriggerSample(t, False)
                else:
                    if self.camera is not None and not self.camera.alive:
                        self.say(f"Camera lost ({self.camera.error}). Switched to keyboard mode.", 8.0)
                        self.camera.close()
                        self.camera = None
                    t = time.perf_counter()
                    img = np.full((canvas_size[1], canvas_size[0], 3), 30, np.uint8)
                    sample = TriggerSample(t, False)

                self.loop_rate.tick(t)
                self._canvas = (img.shape[1], img.shape[0])
                space_down = self.space.is_down()
                trigger = sample.active or space_down
                if self.setup is not None and self.setup.active:
                    # Resting your hands to set the zone must not arm the timer.
                    self._update_setup(t, sample)
                    trigger = False
                if self.calib.active:
                    trigger = False
                self.handle_events(self.machine.update(t, trigger))
                vm = self.build_view(t, sample, space_down, img.shape[1])
                cv2.imshow(self.window, self.renderer.render(img, vm))
                shown = True
                if not self.handle_key(cv2.waitKey(1), t):
                    break
                if shown and cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            self.close()

    def _update_markers(self, raw_image, t: float) -> None:
        """Marker mode: hand the raw frame to the worker (O(1)) and pick up its latest zone."""
        if self.markers is None:
            return
        self.markers.submit(raw_image, t)
        if self.zone_mode == "markers":
            zone = self.markers.snapshot()[0]
            if zone is not None and zone != self.zone:
                self.set_zone(zone)

    def _update_setup(self, t: float, sample: TriggerSample) -> None:
        if self.markers is not None:
            zone, status, seen = self.markers.snapshot()
        else:
            zone, status, seen = None, "", {}
        self.setup.update(t, sample.debug.get("hands", []), zone, status, list(seen.values()))

    def close(self) -> None:
        self._stop_markers()
        self._stop_phone()
        self.renderer.close()
        if self.camera is not None:
            self.camera.close()
        self.detector.close()
        cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CubeCam webcam cube timer")
    ap.add_argument("--config", default=str(BASE / "config.yaml"))
    ap.add_argument("--no-camera", action="store_true", help="keyboard-only mode")
    args = ap.parse_args(argv)
    App(Path(args.config), force_no_camera=args.no_camera).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
