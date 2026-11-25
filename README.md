# CubeCam

A csTimer-style Rubik's cube timer that uses your webcam as a virtual stackmat:
rest both hands in the on-screen start zone, lift them to start, put them back
to stop. Stage 1 of a larger project (cube detection, colour reading and move
counting come later).

## Setup

Requires Python 3.12 (developed and tested on 3.12.10; 3.10.11 also worked
before. mediapipe 0.10.21 supports 3.9–3.12, so 3.13+ won't work).

```powershell
cd CubeCam
py -3.12 -m venv .venv                # macOS/Linux: python3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Optional: `python setup_model.py` downloads `models/hand_landmarker.task`. You only
need it for the MediaPipe Tasks backend (see "MediaPipe version" below).

## Run

```powershell
python main.py                 # camera + hand tracking (falls back to keyboard if no camera)
python main.py --no-camera     # keyboard only, like csTimer
python main.py --config other.yaml
```

## How a solve works

| State | What you do |
|---|---|
| IDLE | Last time + new scramble shown. Scramble the cube. |
| → READY | Rest **both hands in the start zone for 0.3 s**. Zone turns green. |
| → INSPECTION (if on) | Lift your hands: the 15 s countdown starts. Beeps and a coloured border appear at 8 s and 12 s. Put your hands back in the zone for 0.3 s to arm again. |
| → SOLVING | Lift your hands from READY. The timer starts at the timestamp of the **first frame** where your hands were out. |
| → STOPPED | Put both hands back in the zone. The time stops at the first frame where they were back. |
| → IDLE | Lift your hands. |

WCA inspection penalties: starting after 15 s gives +2, after 17 s gives DNF.

**Keyboard fallback:** the spacebar acts like "hands in zone". On Windows it works
exactly like csTimer: hold space for 0.3 s, release to start, press to stop. On
other OSes OpenCV can't see key releases, so each press *toggles* the held state.

### Keys
`SPACE` timer · `2` toggle +2 on last solve · `D` toggle DNF · `Backspace` delete
last solve · `I` toggle inspection · `N` new scramble · `H` debug overlay ·
`M` start-zone setup (see below) · `Esc` abort current solve (not recorded) · `Q` quit

## Setting up the start zone ("mat")

The start zone is where you rest your hands. A hand counts as "in the zone" when
its **palm centre** (the average of the wrist and the four knuckles) is inside the
zone. The zone can be any polygon, not just a rectangle. Hold time and debounce
work exactly as before; only the inside/outside test changed.

### Camera angle first
The camera must **see the desk area where your hands rest**, not just you. Tilt it
down (or put it higher, looking down at the desk) until the mat is in the lower
half of the picture with some room around it. Your hands must stay visible while
they rest, so don't let the timer window or a laptop lid cut them off. Even light
from the front helps the hand tracker. If you move the camera later, set the zone
again, or use marker mode, which follows the mat.

### Three ways to set it
Press `M` while the timer is idle. `M` again moves to the next mode. `Enter`
accepts the zone shown and saves it to `config.yaml` (it persists across
restarts). `R` starts the current mode over. `Esc` cancels and keeps the old zone.
While the setup is open the timer ignores your hands, so resting them there
can't arm it.

1. **Drag** (`M`): click and drag a rectangle over the area where you rest your
   hands, then press `Enter`. Drag again to redo it.
2. **Hands** (`M` `M`): rest both hands where you want the mat, as you would
   before a solve, and hold still. A bar fills over about 2 s; if a hand moves
   more than about 3 % of the frame width, or isn't seen for more than a quarter
   of a second, it starts again. The zone is a rectangle around both whole hands
   plus a margin of about 0.6 x your palm size on each side, so your palms are
   still inside when you put your hands down slightly differently next time.
   Check it and press `Enter`.
3. **Markers** (`M` `M` `M`, optional): use 4 printed markers at the corners of a
   physical mat. The zone is the shape through the marker centres, so it can be
   a trapezoid if the camera looks at the desk at an angle.
   * Run `python tools/make_markers.py`. It writes `markers.pdf` and `markers.png`,
     an A4 page with the 4 markers (60 mm; `--size 40`..`85` to change), cut lines,
     instructions and a 50 mm scale bar.
   * Print at **100 % / actual size**, cut the markers out with their white border,
     and tape them flat at the 4 corners of the mat. Any marker can go at any
     corner, either way up.
   * In marker mode, wait for "4/4 markers" and press `Enter`.
   * The markers are re-detected about 4 times per second in a background thread
     (it never slows the timer). If the camera is bumped, the zone moves with
     the mat. If one marker is hidden under a hand, its position is estimated
     from the other three. If more are hidden, the last known zone is kept. The
     status bar shows `Zone: markers (4/4 markers)` and so on.
   * The saved points are used at startup until the markers are seen again.
     Choosing a drag or hands zone switches marker mode off.

**Warnings.** If a zone is mostly outside the camera picture (less than half of it
visible) or too small (under 2 % of the frame, or narrower than 12 % of its
width), the setup shows it in orange with a warning before you accept. An active
zone with that problem also shows an orange outline and a "ZONE WARNING" line in
the status bar.

## Configuration (`config.yaml`)

The main settings are the zone, camera index, resolution and requested FPS, hold
time, debounce, and inspection on/off.
Missing keys fall back to the defaults in `config.py`.
The zone is stored as

```yaml
zone:
  mode: fixed       # fixed = use these points; markers = follow the 4 ArUco markers
  points: [[0.1, 0.68], [0.9, 0.68], [0.9, 0.98], [0.1, 0.98]]
```

in fractions of the displayed (mirrored) picture, with 0,0 at the top left. `M`
rewrites only this block and keeps the rest of the file and its comments. Older
configs with `x`, `y`, `w`, `h` still load as a rectangle. The `markers:`
section sets the ArUco dictionary and ids (`DICT_4X4_50`, ids 0-3), how often
they are detected, and at what resolution. Solves are saved to
`solves.json` (time, penalty, scramble, timestamp, measured FPS, inspection time)
and reloaded on startup.

## Timing accuracy

* A dedicated capture thread calls `grab()` and takes `time.perf_counter()` right
  after it. Slow hand detection can't delay timestamps. Camera latency is nearly
  constant, so it cancels out between the start and stop frames.
  (`CAP_PROP_POS_MSEC` is unreliable for live webcams on Windows, so it isn't used.)
* Debounce (default 50 ms) filters tracking flicker. Each edge is still
  timestamped at the *first* changed frame, so debounce adds display latency but
  no timing error.
* The status bar shows the capture FPS, the FPS actually processed by the hand
  tracker, and the resulting resolution (±1 processed frame). On a 30 fps webcam
  that's about ±33 ms. A 60/120 fps camera improves it proportionally, as long as
  detection keeps up (see `detect ms` in the debug overlay; lower
  `detection.process_width` if it doesn't).
* With DirectShow, the "driver says" FPS is often wrong. Trust the measured number.

## MediaPipe version

`hands.py` supports both MediaPipe APIs (`detection.backend: auto | legacy | tasks`):

* **legacy** `mp.solutions.hands` – the pinned mediapipe 0.10.21 is the last release with it. It needs no model file.
* **tasks** `HandLandmarker` – mediapipe 1.x only has this API. It needs `python setup_model.py`.

mediapipe 0.10.21 is pinned because on the dev machine Windows Smart App Control /
Application Control **blocked the native DLL of mediapipe 1.2.3** (`WinError 4551`),
while 0.10.21 loads fine. If hand tracking fails to load, the app says why,
keeps showing the camera and stays usable with the spacebar.

## Architecture

| File | Role |
|---|---|
| `main.py` | App loop: wires camera → detector → state machine → UI, keys, saving |
| `camera.py` | Threaded capture with per-frame timestamps, measured FPS |
| `detection.py` | `TriggerDetector` interface, `NullDetector`, `CombinedDetector` |
| `hands.py` | `HandZoneDetector`: MediaPipe → "≥2 palm centres inside the zone polygon" per frame |
| `zone.py` | Zone polygon (inside test, clipping, warnings) and the hands-mode zone builder |
| `zone_setup.py` | `M` setup modes: drag, hands, markers (pure logic, drawn by `ui.py`) |
| `markers.py` | ArUco marker detection in a background thread → zone that follows the mat |
| `tools/make_markers.py` | Printable A4 PDF/PNG of the 4 markers with instructions |
| `timer_state.py` | Pure state machine (no OpenCV, no clock) fed `(timestamp, active)` samples |
| `stats.py` | best, mo3, ao5, ao12, mean (WCA rules: trim best/worst, DNF = worst) |
| `scramble.py` | 20-move random-move scrambles (no `R R'`, no `R L R`) |
| `storage.py` | Atomic JSON save/load, backs up a corrupt file |
| `ui.py` | OpenCV drawing from a `ViewModel` |
| `keyinput.py` | Spacebar held-state (Windows `GetAsyncKeyState`) |
| `config.py` / `config.yaml` | Defaults + user config |

A future cube-motion detector implements `TriggerDetector.process(frame, t) ->
TriggerSample` and can replace the hand detector or be combined with it via
`CombinedDetector([hands, cube], mode="all"|"any")`. The state machine only ever
sees a boolean per timestamp.

## Tests

```powershell
python -m pytest -q
```

They cover the state machine (synthetic 60 fps frame streams with flicker, debounce
bias, inspection +2/DNF boundaries, warnings and abort), stats (ao5/ao12 with DNFs
and +2), scrambles and storage. `tests/test_zone.py` covers the zone:
* **Geometry:** rectangles, concave and slanted polygons, edges, clipping to the
  frame, the "mostly outside" and "too small" warnings, and old vs new config formats.
* **Hands-mode builder:** it needs 2 s of stillness, restarts on movement or slow
  drift, forgives a brief dropout, ignores a third hand, and the margin covers
  both hands.
* **Setup modes and detector:** drag, hands and marker setup flows, and the
  detector using the palm centre inside a trapezoid.
* **Markers:** detected in a synthetic perspective desk scene (mirrored and not),
  the estimated 4th marker and the hold when markers are covered, and the printed
  sheet is detectable with a well-formed PDF.
* **Saving the zone:** comments and other settings in `config.yaml` are kept.
