"""Fast typed-text output.

``pyautogui.write()`` posts one key event per character with a mandatory
sleep between them, so a full sentence visibly "types out" over seconds.
This module picks the fastest backend available on the host:

  * ``pynput`` ``Controller.type()`` (macOS/Linux): posts key events with no
    per-character sleep — a whole sentence lands near-instantly.  On macOS it
    needs the same Accessibility permission pyautogui did.
  * ``keyboard.write()`` (Windows): same idea, no per-character sleeps.
  * ``pyautogui`` with ``interval=0`` as the last-resort fallback.

Construct :class:`TextTyper` on the **main thread**: on macOS the pynput
backend caches the keyboard layout via
:func:`utils.hotkeys.prewarm_macos_key_layout`, whose TIS calls must run on
the main thread (see that function's docstring).  ``type()`` itself is then
safe to call from any worker thread.
"""

from __future__ import annotations

from .hotkeys import prewarm_macos_key_layout


class TextTyper:
    """Types text into the focused window using the fastest available backend."""

    def __init__(self):
        self.backend_name = "none"
        self._type = self._make_backend()

    def _make_backend(self):
        try:
            from pynput.keyboard import Controller

            prewarm_macos_key_layout()  # main-thread TIS cache; no-op off macOS
            controller = Controller()
            self.backend_name = "pynput"
            return controller.type
        except Exception:
            pass

        try:
            import keyboard

            self.backend_name = "keyboard"
            return keyboard.write
        except Exception:
            pass

        import pyautogui

        self.backend_name = "pyautogui"
        return lambda text: pyautogui.write(text, interval=0)

    def type(self, text: str) -> None:
        if text:
            self._type(text)
