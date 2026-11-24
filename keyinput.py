"""Spacebar 'held' state for the csTimer-style keyboard fallback.

cv2.waitKey only reports key presses, not releases. On Windows we poll the real
key state with GetAsyncKeyState (only while the CubeCam window has focus). On
other platforms each space press toggles the held state instead (press once =
hands down, press again = hands up).
"""
from __future__ import annotations

import sys

VK_SPACE = 0x20


class SpaceKey:
    def __init__(self, window_name: str):
        self.window_name = window_name
        self.toggled = False
        self._user32 = None
        if sys.platform == "win32":
            try:
                import ctypes

                self._user32 = ctypes.windll.user32
                self._ctypes = ctypes
            except Exception:
                self._user32 = None

    @property
    def polled(self) -> bool:
        """True if the real key state is polled (hold-to-arm works like csTimer)."""
        return self._user32 is not None

    def _window_focused(self) -> bool:
        hwnd = self._user32.GetForegroundWindow()
        buf = self._ctypes.create_unicode_buffer(256)
        self._user32.GetWindowTextW(hwnd, buf, 256)
        return buf.value == self.window_name

    def on_key(self, key: int) -> bool:
        """Feed a cv2.waitKey code. Returns True if it was the spacebar."""
        if key != 32:
            return False
        if not self.polled:
            self.toggled = not self.toggled
        return True

    def is_down(self) -> bool:
        if self.polled:
            if not self._window_focused():
                return False
            state = self._user32.GetAsyncKeyState(VK_SPACE)
            # 0x8000 = currently down; 0x0001 = pressed since last poll (catches quick taps)
            return bool(state & 0x8000) or bool(state & 0x0001)
        return self.toggled
