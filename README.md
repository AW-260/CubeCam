# CubeCam

A csTimer-style Rubik's cube timer that uses your webcam as a virtual stackmat:
rest both hands in the on-screen start zone, lift them to start, put them back
to stop.

Stage 2 adds cube vision while the timer is idle. It reads sticker colours when
you hold a face still, checks your scramble, scans the whole cube and records
solves as training data.

Stage 3 measures each solve: move count, whole-cube rotations, TPS and a timeline of
turning vs pauses, from the motion of the cube between your hands (no smart cube, no
video saved). A practice mode with known sequences checks how accurate that is for you.

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
`Esc` abort current solve (not recorded) · `Q` quit

Stage 2 (cube vision, only while the timer is IDLE):

| Key | What it does |
|---|---|
| `C` | Guided colour calibration. Show each centre (white, yellow, green, blue, red, orange) and hold it still. Saved to `calibration.json`. `Backspace` redoes the previous colour, `Esc` cancels. |
| `S` | Guided full-cube scan with a 2D net on screen (see below). `1`–`6` rescan a face, `Backspace` rescans the last face, `Enter`/`Esc` closes. |
| `V` | Turn the scramble check on or off (on by default via `vision.auto_verify`). |
| `R` | Turn dataset recording of solves on or off. |
| `H` | The debug overlay also shows the detected grid, each sticker's colour and confidence, and the vision timings. |

Stage 3 (move count / TPS):

| Key | What it does |
|---|---|
| `P` | Practice mode on/off (idle only). The scramble line shows a known sequence instead. |
| `N` | In practice mode: next practice sequence. |
| `+` / `-` | In practice mode: cue tempo (moves per second; 0 = free pace, no cues). |
| `T` | Show/hide the motion panel (last solve's numbers and timeline). |

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

## Move count, TPS and pauses (stage 3)

After every solve the panel on the right shows:

* **Moves**: layer turns. A half turn (R2) counts as **one** turn; slices (M) and wide
  moves (r) are one turn each. **Rotations** (x, y, z) are counted separately.
  Anything the app can't tell apart is counted as **unclassified**, never guessed.
* **TPS** two ways: turns / solve time, and turns / time spent turning.
* **Paused**: total time in pauses, the number of pauses (stops of at least 0.5 s),
  and the 3 longest with when they happened.
* A **timeline**: green = turning (including rotations and the short gaps between
  moves), grey = paused, cyan marks = rotations, white = unclassified, orange
  boxes = the 3 longest pauses, time axis in seconds.

The stats panel adds TPS, moves and paused time averaged over your last 5 and 12
solves that have motion data. Results are stored with each solve in `solves.json`
under `"motion"` (about 1-2 KB). Older solves without it load as before.

### Definitions
* **Turning vs paused.** A pause is any stop of at least 0.5 s with no turning
  and no rotating. Everything else is turning time, including the short gaps
  between moves of one algorithm. Paused + turning = the solve time.
* **Half turns.** A half turn is recognised when one event shows two
  same-direction humps (the same part keeps moving the same way, for example U2
  done as a double flick), is clearly longer or farther than this solve's typical
  turn, or lasts at least 0.36 s. Either way it is **one** turn, and the count of
  half turns is shown separately. Quarter-vs-half is the least reliable label (see
  the accuracy below); the move count does not depend on it.

### How it works
* **Cube region.** The hand tracker already runs on every frame. The cube is held
  between the hands, so the region is a square between the two palm centres,
  smoothed over time and held for a moment when a hand is lost. Arms, the body and
  anything outside that square are never looked at.
* **Per frame** (in a background thread, `vision/motion.py`): the same square is cut
  from the previous and the current frame at 128x128 px and the hands are masked
  out: the palm and finger bones are drawn from the landmarks, plus skin-coloured
  pixels right next to them. Colour alone isn't used, because red and orange stickers
  look like skin in warm light. Then dense optical flow (DIS) is computed, and a
  row of about 90 numbers is kept: how the cube body moves, how the hands move,
  how much of the cube moves relative to the rest, and a 5x5 grid of mean flow.
  The frame itself is dropped straight away; only the previous one is kept, to
  compare against.
* **After the stop** (`vision/motion_analysis.py`, pure numpy):
  * **Layer motion.** Part of the cube moves while the rest stays still, and the
    movement isn't just the hands carrying it. Bursts of activity are split into
    single moves at the dips between them. Back-to-back humps where the same region
    keeps moving the same way are merged into one move (a half turn).
  * **Rotations.** The middle of the region (the middle of the face you see) stays
    still for R/L/U/D/F turns, but moves with the cube in x/y rotations. An event
    is a rotation when the middle carries most of its motion, a turn when it
    carries little, and unclassified in between.
  * **Carrying.** Lifting or putting the cube down (the cube and the hands moving
    together) is not a move.
* **The timer is never slowed down.** `main.py` hands the worker a reference to the
  raw frame plus the landmarks it already has. That costs about 0.01 ms. If the
  worker is still busy, the waiting frame is replaced and counted as dropped. The
  worker needs about 2-3 ms per frame. `python tools/profile_motion.py` measures
  the processed hand-tracking FPS with stage 3 off, on, and off again.

### Practice mode (`P`): checking accuracy without a smart cube
The scramble line shows a known sequence. Rest your hands, lift, then perform it.
With a tempo set, the move to do now is boxed in yellow, one move per beat, after a
2 s lead-in. `P2` means hold still for 2 s. Put your hands back to stop. The run is
saved to `practice_data/practice.json` with its labels: the sequence, the true
turn/rotation/half-turn counts, the cue times, and what was measured. The per-frame
numbers (about 2 KB per second, no images) go to `practice_data/features/`. Practice
runs are **not** added to your solves. The same sequence stays loaded so you can
repeat it; `N` moves on. A "random" item draws a new 15-turn sequence with 3
rotations each time.

```powershell
python tools/evaluate_counts.py                # accuracy on your practice runs
python tools/evaluate_counts.py --set on=0.15 rot_hi=0.65   # re-analyse them with other thresholds
python tools/evaluate_counts.py --synthetic 18 # the same report on rendered clips
python tools/profile_motion.py                 # processed FPS with / without stage 3 (live camera)
```

The report gives measured vs true turns, rotations, unclassified and half turns per
run. For paced runs it also checks whether each deliberate pause was found, and how
well turning/paused periods match the cues after estimating your reaction lag.

### Accuracy so far (synthetic only, not yet on a real cube)
On rendered 30 fps clips (`tools/evaluate_counts.py --synthetic 18`) of a cube seen
from the back with hands on its sides, motion blur, the cube being lifted, and
landmark noise:

| Sequence type | Turns | Rotations |
|---|---|---|
| R / U / L sequences | 18-24 of 20-24 (within 1-2) | no false rotations |
| x / y only | - | 8/8, 8/8 |
| y R U R' y' | 9/9 | 4-6 of 6 (rest unclassified) |
| R2 U2 | 8-9 of 8 (each R2 counted once) | - |
| Deliberate pauses | all 14 found | - |
| F R U R' U' F' | 11-12 of 18 | - |
| r U R' U' (wide) | 11 of 16 | 3-4 false rotations |

Overall turn error is 12% (mostly F and wide moves). Half-turn labels vary a lot
between clips (7/8 in one, 2/8 in another).

### Known limitations
* **F (and B) turns from behind.** The camera sees the face pointing away from you.
  An F turn mostly moves the side of the cube nearest to you, so the camera sees
  little of it and F turns are undercounted. A camera a little higher, looking down
  on the top face, helps.
* **Wide moves** (r, u) move the middle of the visible face, so they can look like
  rotations. **z rotations and B turns** both spin the visible face in place and
  can't be told apart. These usually come out as turns or unclassified.
* It needs both hands tracked. The panel says "Motion data low: hands tracked in
  N% of frames" when that wasn't the case.
* Thresholds come from synthetic data. Phase B tunes them on recorded practice runs.

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
| `vision/synth.py`, `vision/evaluate.py` | Synthetic cube renderer (still faces; stage 3: animated cubies, hands, blur) + accuracy evaluation |
| `tools/profile_vision.py` | Live-camera timing of detection vs. hand tracking |
| `tools/eval_synthetic.py` | Synthetic accuracy report |
| `vision/motion.py` | Stage 3: cube region from the palms, hand mask, optical flow -> one row of numbers per frame; `MotionWorker` thread |
| `vision/motion_analysis.py` | Stage 3: rows -> turns / rotations / unclassified / half turns, pauses, TPS, timeline |
| `vision/motion_controller.py` | Stage 3: worker per solve, practice mode, panel, keys |
| `vision/practice.py` | Practice sequences (parser, true counts, random, paced cues) and `practice_data/` storage |
| `vision/motion_eval.py`, `tools/evaluate_counts.py` | Accuracy against practice labels / synthetic clips |
| `tools/profile_motion.py` | Live camera: processed FPS with and without stage 3 |
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
* **Stage 3:** the practice sequence parser and counts, the paced cue schedule,
  analysis on hand-made rows (turns, rotation, half turn, pauses, carrying the cube,
  a hand moving alone), the worker never blocking and dropping frames instead, and
  practice runs saved with labels and small feature files (no images). Also
  end-to-end tests on rendered animated cubes: turn counts, rotations, R2 counted
  once, deliberate pauses found, a still cube with motion outside the region giving
  nothing, and extractor speed. Run `pytest -s tests/test_motion.py` to see the reports.
* **Dataset and controller:** recordings with pre-roll and tail, frames dropped
  (never blocking) when the queue is full, aborted solves, and vision never being
  fed unless allowed.

```powershell
python tools/eval_synthetic.py --n 300   # larger synthetic accuracy report
python tools/profile_vision.py           # live camera: detection ms vs. hand tracking
```
