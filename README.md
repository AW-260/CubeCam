# CubeCam

A csTimer-style Rubik's cube timer that uses your webcam as a virtual stackmat:
rest both hands in the on-screen start zone, lift them to start, put them back
to stop.

Stage 2 adds cube vision while the timer is idle. It reads sticker colours when
you hold a face still, checks your scramble, scans the whole cube and records
solves as training data. Move counting comes in a later stage.

## Setup

Requires Python 3.10 or 3.11 (developed and tested on 3.10.11; mediapipe 0.10.21
supports 3.9–3.12).

```powershell
cd CubeCam
python -m venv .venv
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
`Esc` abort current solve (not recorded) · `Q` quit

Stage 2 (cube vision, only while the timer is IDLE):

| Key | What it does |
|---|---|
| `C` | Guided colour calibration. Show each centre (white, yellow, green, blue, red, orange) and hold it still. Saved to `calibration.json`. `Backspace` redoes the previous colour, `Esc` cancels. |
| `S` | Guided full-cube scan with a 2D net on screen (see below). `1`–`6` rescan a face, `Backspace` rescans the last face, `Enter`/`Esc` closes. |
| `V` | Turn the scramble check on or off (on by default via `vision.auto_verify`). |
| `R` | Turn dataset recording of solves on or off. |
| `H` | The debug overlay also shows the detected grid, each sticker's colour and confidence, and the vision timings. |

While calibrating or scanning, the hand trigger is ignored. Holding the cube up to
the camera can't arm the timer, and the start zone is hidden. If you start the
timer with the spacebar, the mode closes.

## Cube vision (stage 2)

### Scramble check
Scramble in WCA orientation (white top, green front). Then, while the timer is idle,
show any face to the camera and **hold it still for about 0.3 s**. The app works out
which face it is from the centre colour. It compares the face with what the current
scramble should give, at all 4 rotations, so you can hold it any way up. Under the
scramble you'll see one of these:

* **green "Scramble OK on n/6 faces"**: every face shown so far matches.
* **red "SCRAMBLE MISMATCH on … face"**: wrong stickers are circled red and crossed out on the live image.
* **yellow "(some unclear)"**: some stickers were too uncertain to call, for example red vs
  orange in poor light or a finger in the way. Those stickers are circled yellow.
  The app never reports a mismatch on a sticker it isn't sure about.

The results are kept until the scramble changes. Check as many faces as you like before
you put your hands down.

### Full scan (`S`)
Hold the cube in WCA orientation. The panel tells you which face to show and which
colour must be on top, matching the standard net:

1. green, white on top · 2. red, white on top · 3. blue, white on top ·
4. orange, white on top · 5. white, **blue** on top · 6. yellow, **green** on top

Each face is captured automatically once it is still and at least 7 of 9 stickers
are readable, and the net fills in. After 6 faces the scan is joined up and checked
with `cube.py`: 9 of each colour, 6 different centres, no impossible, twisted or
flipped pieces, and no parity errors.

* If colours are uncertain or the counts are off, all 54 stickers are re-decided
  together under the rule "exactly 9 of each colour" (a minimum-cost assignment).
  Stickers changed this way are marked on the net.
* If a face was held sideways, the app tries every face rotation. It accepts the
  result only if exactly one valid cube comes out, and says which faces it rotated.
* It also tells you whether the scanned cube matches the current scramble.

### Calibration tips (`C`)
* Calibrate **with the light you'll actually use**, and again if that changes
  (day/night, a different lamp). Without calibration, built-in typical colours are used.
  They work in neutral light but report many more stickers as unsure in warm, cool
  or dim light.
* Showing a **solved** face for each colour gives 9 samples per frame instead of 1.
  The centre alone also works.
* Use even, diffuse light from in front of or above you. Avoid a bright window behind
  the cube and lamps that cause glare on glossy stickers. The brightest 15% of each
  sample patch is ignored to cope with highlights, but a sticker that is all glare
  can't be read.
* If your webcam software allows it, turn off auto white balance and auto exposure
  after calibrating, so the colours don't drift.
* **Red vs orange** is the classic problem. Without calibration, red and orange
  are told apart by hue angle only, with a wide "unsure" band, because brightness
  depends on exposure. After calibration, lightness helps too. Anything near the
  boundary shows as `R/O?` and is never silently guessed. If the app warns that two
  colours look very similar after calibration, improve the light.
* Fill a good part of the frame with the face: about 1/4 of the frame height or more
  works best. Stickers smaller than about 15 px aren't detected.

### How it works
* `vision/face_detect.py` runs Canny edges on L\*, a\* and b\* (the chroma edges matter
  for stickerless cubes). Sticker-shaped regions are found, and every candidate is
  tried as the seed of a 3x3 lattice. A homography from the lattice to the image
  handles about ±30° of tilt. Every row and column must contain a sticker whose
  outline was actually found, which stops the 3x3 window sliding onto a side face or
  a hand. The face is warped flat and a glare-trimmed median patch is sampled from
  the middle of each sticker. As a fallback for faces without visible borders, a
  large quad with dark seams where the grid lines should be is accepted.
* Finger occlusion: when a sticker's own outline wasn't found and its colour carries
  on unchanged past the edge of the face, it is reported as covered. A real sticker
  is enclosed by the body or seams; a finger isn't. Skin-coloured patches are also
  rejected.
* `vision/stability.py` only reads when the sticker centres moved less than 0.12
  sticker widths between frames (0.25 overall) and the flattened face barely changed
  for `vision.still_time`. Colours are averaged over that window.
* `vision/colors.py` classifies in CIELAB using a spread-normalised distance with a
  low weight on lightness. It has the explicit red/orange test above, a distance
  cut-off so anything far from every colour comes out low-confidence, and the
  balanced 9-per-colour assignment for full scans.
* **The timer is never slowed down.** Detection runs in its own thread
  (`vision/worker.py`), and `main.py` hands it frames **only while the timer is IDLE
  with nothing in the start zone**. From the moment your hands are in the zone, and
  through READY, inspection and the solve, no vision work happens. Handing a frame
  over costs the main loop about 0.2 ms (one resize).

## Dataset recording (`R`)
For a later move-recognition stage. When it's on, every solve is saved to
`datasets/<date-time>/` (git-ignored). Recording runs from inspection start (or solve
start) plus a 0.5 s pre-roll, to the stop plus a 0.5 s tail. Each folder holds:

* `frames/000001.jpg …`: raw camera frames (not mirrored), full resolution unless
  `dataset.max_width` is set
* `frames.csv`: frame index, camera sequence number, capture timestamp `t` (same clock
  as the timer) and `t_rel` (seconds from the start)
* `meta.json`: scramble, final time, penalty, inspection time, solve start/end
  timestamps, camera size and FPS, and frames written/dropped. Aborted solves are marked `aborted: true`.

Frames are handed over by the capture thread without blocking, and a separate
thread does the JPEG encoding and writing. If the disk can't keep up, frames are
**dropped and counted in `meta.json`**; the timer and the camera thread are never
held up. The screen shows a REC indicator, and the status bar shows the number of
solves saved, disk used, free space and the queue length. At 720p and 30 fps this is
about 3–4 MB per second of solve (JPEG quality 90). Later changes to a solve, such
as +2, DNF or deleting it with Backspace, are not written back to `meta.json`.

### Known limitations
* It has only been tested on synthetic images and live camera timing. It hasn't been
  tested on a real cube yet (see "What to test by hand" below).
* Faces with no visible borders between stickers (a solved face on a stickerless
  cube with very faint seams) are only found by the fallback and may not be detected.
* A static 3x3 arrangement of coloured squares in the background (tiles, a keyboard
  block) can occasionally be mistaken for a face: 1–3% of very cluttered synthetic
  frames. With many fingers over the face, the grid is occasionally placed one row or
  column off (1–4% of synthetic scenes). Turn on `H` to see what is being detected.
* Face rotations in the scramble check are free, but a mirrored view (a real mirror)
  isn't supported. The app's own selfie mirroring is undone automatically.

### What to test by hand (real cube)
1. Press `H`. Hold a scrambled face 30–50 cm from the camera: the grid should lock on
   and the stickers should be labelled correctly. Tilt about 30° each way and cover
   1–2 edge stickers with fingers. Covered stickers should show low confidence or
   `?`, not a wrong colour.
2. Press `C` and calibrate in your normal light. Then check that red and orange are
   told apart on a face that has both.
3. Scramble with the shown scramble, then show 2–3 faces: you should get "Scramble OK".
   Make one deliberate wrong move: the mismatch should be flagged on the right stickers.
4. Press `S` and scan the whole cube. It should say VALID and "matches current
   scramble". Hold one face sideways on purpose: it should be auto-rotated.
5. Time a few solves, with and without the scramble check on, with and without `R`.
   The processed FPS in the status bar should not drop while READY/SOLVING, and
   `datasets/` should contain a folder per solve.
6. If you have one, try a stickerless cube and a solved stickerless face.

## Configuration (`config.yaml`)

The main settings are the zone position and size (fractions of the frame), camera
index, resolution and requested FPS, hold time, debounce, and inspection on/off.
Missing keys fall back to the defaults in `config.py`. Solves are saved to
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
| `detection.py` | `TriggerDetector` interface, `Zone`, `NullDetector`, `CombinedDetector` |
| `hands.py` | `HandZoneDetector`: MediaPipe → "≥2 palms in zone" per frame |
| `timer_state.py` | Pure state machine (no OpenCV, no clock) fed `(timestamp, active)` samples |
| `stats.py` | best, mo3, ao5, ao12, mean (WCA rules: trim best/worst, DNF = worst) |
| `scramble.py` | 20-move random-move scrambles (no `R R'`, no `R L R`) |
| `storage.py` | Atomic JSON save/load, backs up a corrupt file |
| `ui.py` | OpenCV drawing from a `ViewModel` (+ `VisionView`: grid overlay, net, panels) |
| `cube.py` | 54-facelet cube model: all moves (faces, wide, slices, rotations), parse/invert, validation |
| `vision/face_detect.py` | 3x3 grid detection, homography, sticker sampling, occlusion test |
| `vision/colors.py` | LAB colour model (defaults/calibrated), red/orange handling, balanced 9-per-colour assignment |
| `vision/calibration.py` | Calibration session (C) |
| `vision/stability.py` | "Cube held still" tracker + colour averaging |
| `vision/scan.py` | Scramble check (any rotation) and guided full scan (pure logic) |
| `vision/worker.py` | Background detection thread (frames only while IDLE) |
| `vision/controller.py` | Modes, keys, timer events and on-screen view for all of stage 2 |
| `vision/dataset.py` | Background solve recorder (R) |
| `vision/synth.py`, `vision/evaluate.py` | Synthetic cube renderer + accuracy evaluation |
| `tools/profile_vision.py` | Live-camera timing of detection vs. hand tracking |
| `tools/eval_synthetic.py` | Synthetic accuracy report |
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

They cover:

* **Stage 1:** the state machine (synthetic 60 fps frame streams with flicker, debounce
  bias, inspection +2/DNF boundaries, warnings and abort), stats (ao5/ao12 with DNFs
  and +2), scrambles and storage.
* **The cube model:** every move ×4 = identity, (R U R' U')×6 = identity, scramble +
  inverse = solved, known facelet strings, and validation rejecting bad counts,
  duplicate centres, twisted corners, flipped edges, mirrored or impossible pieces
  and parity.
* **Scramble check and scan:** every face at all 4 rotations, single wrong stickers, a
  missing last move, the mirrored view, uncertain vs mismatch, auto-rotation of a
  sideways face, and the 9-per-colour assignment fixing red/orange.
* **Colours:** red/orange ambiguity, skin, glare, calibration save/load, and the
  Hungarian assignment against brute force.
* **Synthetic vision:** rendered cubes with random pose (±30° yaw/pitch/roll), black
  and stickerless bodies, 0–3 fingers, random light tint, gain, gamma, vignetting,
  blur, noise and JPEG. The tests cover detection rate, confident-sticker accuracy,
  coverage, finger leakage, false positives on cube-free clutter, calibration vs
  defaults in warm/dim light, end-to-end scramble checks with the face rolled
  0/90/180/-90°, mirror handling, stillness and speed. Run `pytest -s
  tests/test_vision_synth.py` to see the numbers.
* **Dataset and controller:** recordings with pre-roll and tail, frames dropped
  (never blocking) when the queue is full, aborted solves, and vision never being
  fed unless allowed.

```powershell
python tools/eval_synthetic.py --n 300   # larger synthetic accuracy report
python tools/profile_vision.py           # live camera: detection ms vs. hand tracking
```
