"""Short, non-blocking sound cues for record start/stop.

Cross-platform best-effort: macOS uses ``afplay`` with built-in system
sounds, Windows uses ``winsound``, Linux tries ``paplay``/``aplay`` and
otherwise falls back to the terminal bell. All playback is fire-and-forget
so it never blocks the hotkey or audio threads.
"""

import sys
import subprocess

_MACOS_SOUNDS = {
    "start": "/System/Library/Sounds/Tink.aiff",
    "stop": "/System/Library/Sounds/Bottle.aiff",
}

# freedesktop sound theme (common on Linux desktops)
_LINUX_SOUNDS = {
    "start": "/usr/share/sounds/freedesktop/stereo/message.oga",
    "stop": "/usr/share/sounds/freedesktop/stereo/complete.oga",
}


def _spawn(cmd):
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def play_cue(cue: str = "start") -> None:
    """Play a short non-blocking cue. ``cue`` is ``"start"`` or ``"stop"``."""
    cue = cue if cue in ("start", "stop") else "start"
    try:
        if sys.platform == "darwin":
            _spawn(["afplay", _MACOS_SOUNDS[cue]])
        elif sys.platform.startswith("win"):
            import winsound
            # SND_ASYNC = non-blocking; use distinct system aliases.
            alias = "SystemAsterisk" if cue == "start" else "SystemHand"
            winsound.PlaySound(alias, winsound.SND_ALIAS | winsound.SND_ASYNC)
        else:
            if not _spawn(["paplay", _LINUX_SOUNDS[cue]]):
                sys.stdout.write("\a")  # terminal bell fallback
                sys.stdout.flush()
    except Exception:
        pass
