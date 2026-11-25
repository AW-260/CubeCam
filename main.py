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
from config import DEFAULTS, load_config, save_zone
from detection import DetectorUnavailable, NullDetector, TriggerSample
from keyinput import SpaceKey
from models import Penalty, Solve, format_solve, format_time
from scramble import generate_scramble
from stats import compute_stats
from storage import SolveStore
from timer_state import EventType, State, TimerConfig, TimerStateMachine
from ui import GREEN, ORANGE, RED, WHITE, YELLOW, Renderer, ViewModel
from zone import Zone, zone_warnings
from zone_setup import SetupMode, ZoneSetup

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
        if self.zone_mode not in ("fixed", "markers"):
            zone_msgs.append(f"Unknown zone mode {self.zone_mode!r} - using fixed")
            self.zone_mode = "fixed"
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

    def start_zone_setup(self) -> None:
        if self.camera is None or not self.camera.alive:
            self.say("Zone setup needs the camera")
            return
        self.setup.start(SetupMode.DRAG)

    def _setup_key(self, key: int, ch: str) -> None:
        """Keys while the zone setup is open (everything except Q and H comes here)."""
        setup = self.setup
        code = key & 0xFF
        if code == 27:
            setup.cancel()
            self.say("Zone setup cancelled - zone unchanged")
        elif code in (10, 13):
            result = setup.accept()
            if result is None:
                self.say("Not all 4 markers seen yet" if setup.mode is SetupMode.MARKERS else "No zone yet")
                return
            mode, zone = result
            self.zone_mode = mode
            self.set_zone(zone)
            try:
                save_zone(self.config_path, mode, zone.to_config())
                saved = f"saved to {self.config_path.name}"
            except (OSError, ValueError) as exc:
                saved = f"NOT saved ({exc})"
            warn = zone_warnings(zone)
            self.say(f"Zone set ({'follows the markers' if mode == 'markers' else mode}), {saved}"
                     + (f". Warning: {'; '.join(warn)}" if warn else ""), 5.0 if warn else 3.0)
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
        if self.setup is not None and self.setup.active:
            self.setup.on_mouse(event, x, y, *self._canvas)

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
        if self.setup is not None and self.setup.active and ch != "h":
            self._setup_key(key, ch)
            return True
        if ch == "m":
            if busy:
                self.say("Finish or abort (ESC) the current solve first")
            else:
                self.start_zone_setup()
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
        zone_txt = "markers" if self.zone_mode == "markers" else ("rectangle" if self.zone.is_rect else "polygon")
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
        if setup_view is not None:
            zone_mode = "setup"

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
