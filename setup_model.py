"""Download the MediaPipe HandLandmarker model (only needed for the 'tasks' backend).

The pinned mediapipe 0.10.21 uses the legacy backend, which needs no model file.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
DEST = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"


def main() -> int:
    if DEST.is_file() and DEST.stat().st_size > 1_000_000:
        print(f"Model already present: {DEST}")
        return 0
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}\n  -> {DEST}")
    tmp = DEST.with_suffix(".part")
    urllib.request.urlretrieve(URL, tmp)
    tmp.replace(DEST)
    print(f"Done ({DEST.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
