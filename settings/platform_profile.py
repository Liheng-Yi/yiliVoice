"""Platform detection and per-OS profiles.

This module is the single source of truth for everything that differs
between operating systems, so the rest of the app can stay OS-agnostic.

It drives **automatic switching** of:
  * the transcription backend + compute device (NVIDIA CUDA on Windows,
    Apple-GPU MLX on Apple Silicon, CPU everywhere else),
  * the global-hotkey backend (``keyboard`` on Windows, ``pynput`` on
    macOS so no ``sudo`` is required),
  * the virtual-audio-cable keywords + setup instructions
    (VB-CABLE on Windows, BlackHole on macOS).

By default the profile is chosen from the host OS, but the user can pin
an explicit profile (``auto`` / ``windows`` / ``macos`` / ``linux``) from
the UI.  Capability-dependent fields (which accelerator actually exists,
which hotkey backend is importable) always reflect the *real* host, even
when a different profile is pinned, so pinning can never make the app
non-functional.
"""

from __future__ import annotations

import importlib.util
import platform
import sys
from dataclasses import dataclass, field

# Canonical OS keys ---------------------------------------------------------
OS_WINDOWS = "windows"
OS_MACOS = "macos"
OS_LINUX = "linux"

PROFILE_AUTO = "auto"
VALID_OVERRIDES = (PROFILE_AUTO, OS_WINDOWS, OS_MACOS, OS_LINUX)

OS_DISPLAY_NAMES = {
    PROFILE_AUTO: "Auto-detect",
    OS_WINDOWS: "Windows",
    OS_MACOS: "macOS",
    OS_LINUX: "Linux",
}

# Backend keys --------------------------------------------------------------
BACKEND_FASTER_WHISPER = "faster-whisper"
BACKEND_MLX = "mlx"


# --------------------------------------------------------------------------- #
# Low-level host detection                                                     #
# --------------------------------------------------------------------------- #

def detect_os() -> str:
    """Return the canonical key for the *real* host OS."""
    p = sys.platform
    if p.startswith("win"):
        return OS_WINDOWS
    if p == "darwin":
        return OS_MACOS
    return OS_LINUX


def is_apple_silicon() -> bool:
    """True on an Apple-Silicon (arm64) Mac."""
    return detect_os() == OS_MACOS and platform.machine().lower() in ("arm64", "aarch64")


def _mlx_whisper_available() -> bool:
    """Cheap check (no import side-effects) for the mlx_whisper package."""
    if not is_apple_silicon():
        return False
    try:
        return importlib.util.find_spec("mlx_whisper") is not None
    except (ImportError, ValueError):
        return False


def _cuda_available() -> tuple[bool, str]:
    """Return (available, gpu_name).  Safe if torch is missing."""
    try:
        import torch  # local import keeps this module light
        if torch.cuda.is_available():
            try:
                return True, torch.cuda.get_device_name(0)
            except Exception:
                return True, "NVIDIA GPU"
    except Exception:
        pass
    return False, ""


def detect_acceleration() -> dict:
    """Pick the best *available* transcription backend for this host.

    Order of preference: Apple-GPU MLX → NVIDIA CUDA → CPU.  This is based
    on real hardware/packages, never on a pinned profile.
    """
    if _mlx_whisper_available():
        return {
            "backend": BACKEND_MLX,
            "device": "gpu",
            "compute_type": "float16",
            "accelerator_label": "Apple GPU · MLX",
        }

    cuda_ok, gpu_name = _cuda_available()
    if cuda_ok:
        return {
            "backend": BACKEND_FASTER_WHISPER,
            "device": "cuda",
            "compute_type": "float16",
            "accelerator_label": f"NVIDIA CUDA · {gpu_name}",
        }

    # CPU fallback (also used on Apple Silicon when mlx_whisper isn't installed)
    return {
        "backend": BACKEND_FASTER_WHISPER,
        "device": "cpu",
        "compute_type": "int8",
        "accelerator_label": "CPU · int8",
    }


# --------------------------------------------------------------------------- #
# Per-OS cosmetic / config tables                                              #
# --------------------------------------------------------------------------- #

_VB_CABLE_SETUP = (
    "Voice changer routing (Windows):\n"
    "1. Install VB-CABLE  →  https://vb-audio.com/Cable/\n"
    "2. Set your real mic as Input above\n"
    "3. Set 'CABLE Input' as Output above\n"
    "4. In your game/app, choose 'CABLE Output' as the microphone"
)

_BLACKHOLE_SETUP = (
    "Voice changer routing (macOS):\n"
    "1. Install BlackHole  →  brew install blackhole-2ch\n"
    "2. Set your real mic as Input above\n"
    "3. Set 'BlackHole 2ch' as Output above\n"
    "4. In your game/app, choose 'BlackHole 2ch' as the microphone\n"
    "   (To also hear yourself, build a Multi-Output Device in Audio MIDI Setup.)"
)

_PULSE_SETUP = (
    "Voice changer routing (Linux):\n"
    "1. Create a virtual sink:  pactl load-module module-null-sink "
    "sink_name=virtmic\n"
    "2. Set your real mic as Input above\n"
    "3. Set the null-sink as Output above\n"
    "4. Point your app at the sink's monitor source"
)

_OS_TABLE = {
    OS_WINDOWS: {
        "hotkey_backend": "keyboard",
        "hotkeys": {
            "toggle_recording": "pause",
            "toggle_voice_changer": "ctrl+f9",
            "toggle_vc_routing": "ctrl+f10",
        },
        "hotkey_labels": {
            "toggle_recording": "Pause / Break",
            "toggle_voice_changer": "Ctrl+F9",
            "toggle_vc_routing": "Ctrl+F10",
        },
        "virtual_cable_keywords": ["cable", "virtual", "voicemeeter"],
        "virtual_cable_setup": _VB_CABLE_SETUP,
        "permission_note": "",
    },
    OS_MACOS: {
        "hotkey_backend": "pynput",
        "hotkeys": {
            "toggle_recording": "<cmd_r>",
            "toggle_voice_changer": "<ctrl>+<alt>+9",
            "toggle_vc_routing": "<ctrl>+<alt>+0",
        },
        "hotkey_labels": {
            "toggle_recording": "Right ⌘ (Command)",
            "toggle_voice_changer": "Ctrl+Option+9",
            "toggle_vc_routing": "Ctrl+Option+0",
        },
        "virtual_cable_keywords": ["blackhole", "loopback", "cable", "aggregate", "multi-output"],
        "virtual_cable_setup": _BLACKHOLE_SETUP,
        "permission_note": (
            "Grant Terminal/your IDE permission under System Settings → "
            "Privacy & Security:  Microphone, Accessibility, and Input "
            "Monitoring.  No sudo required."
        ),
    },
    OS_LINUX: {
        "hotkey_backend": "pynput",
        "hotkeys": {
            "toggle_recording": "<ctrl>+<alt>+v",
            "toggle_voice_changer": "<ctrl>+<alt>+9",
            "toggle_vc_routing": "<ctrl>+<alt>+0",
        },
        "hotkey_labels": {
            "toggle_recording": "Ctrl+Alt+V",
            "toggle_voice_changer": "Ctrl+Alt+9",
            "toggle_vc_routing": "Ctrl+Alt+0",
        },
        "virtual_cable_keywords": ["cable", "virtual", "loopback", "null", "monitor"],
        "virtual_cable_setup": _PULSE_SETUP,
        "permission_note": "",
    },
}


def _recommended_model(backend: str, device: str) -> str:
    """Pick a sensible default Whisper model size for the accelerator."""
    if device == "cuda":
        return "medium"
    if backend == BACKEND_MLX:
        return "small"   # snappy + accurate on Apple GPU
    return "base"        # CPU: keep it responsive


# --------------------------------------------------------------------------- #
# Public profile object                                                        #
# --------------------------------------------------------------------------- #

@dataclass
class PlatformProfile:
    detected_os: str
    override: str
    effective_os: str
    is_apple_silicon: bool

    # transcription / acceleration (always reflects real hardware)
    backend: str
    device: str
    compute_type: str
    accelerator_label: str
    recommended_model: str

    # OS-flavoured config (follows the effective/pinned OS)
    hotkey_backend: str
    hotkeys: dict
    hotkey_labels: dict
    virtual_cable_keywords: list
    virtual_cable_setup: str
    permission_note: str = ""

    def summary(self) -> str:
        """Short human-readable one-liner for logs."""
        pinned = "" if self.override == PROFILE_AUTO else f" (pinned: {self.override})"
        return (
            f"OS={self.effective_os}{pinned}  "
            f"backend={self.backend}  accel={self.accelerator_label}  "
            f"hotkeys={self.hotkey_backend}"
        )


def build_profile(override: str | None = PROFILE_AUTO) -> PlatformProfile:
    """Construct the active :class:`PlatformProfile`.

    Args:
        override: ``auto`` (default) follows the host OS; ``windows`` /
            ``macos`` / ``linux`` pin the *cosmetic* profile.  Acceleration
            and the hotkey backend always fall back to what the real host
            actually supports.
    """
    override = (override or PROFILE_AUTO).lower()
    if override not in VALID_OVERRIDES:
        override = PROFILE_AUTO

    detected = detect_os()
    effective = detected if override == PROFILE_AUTO else override

    # The OS-flavoured table can follow a pin, but the hotkey backend must be
    # importable on the real host — pynput/keyboard are OS-specific.
    table = _OS_TABLE.get(effective, _OS_TABLE[OS_LINUX])
    real_table = _OS_TABLE.get(detected, _OS_TABLE[OS_LINUX])
    hotkey_backend = real_table["hotkey_backend"]
    # Use the binding/label style matching the backend we'll actually run.
    hk_source = table if table["hotkey_backend"] == hotkey_backend else real_table

    accel = detect_acceleration()

    return PlatformProfile(
        detected_os=detected,
        override=override,
        effective_os=effective,
        is_apple_silicon=is_apple_silicon(),
        backend=accel["backend"],
        device=accel["device"],
        compute_type=accel["compute_type"],
        accelerator_label=accel["accelerator_label"],
        recommended_model=_recommended_model(accel["backend"], accel["device"]),
        hotkey_backend=hotkey_backend,
        hotkeys=dict(hk_source["hotkeys"]),
        hotkey_labels=dict(hk_source["hotkey_labels"]),
        virtual_cable_keywords=list(table["virtual_cable_keywords"]),
        virtual_cable_setup=table["virtual_cable_setup"],
        permission_note=table.get("permission_note", ""),
    )
