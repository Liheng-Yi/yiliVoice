# YiliVoice - Voice-to-Text Tool

A personal development tool for responsive voice recognition and
transcription, typed straight into whatever app is focused. Uses NVIDIA's
**Parakeet** model (via Apple MLX) on Apple Silicon and OpenAI's **Whisper**
everywhere else.

**Cross-platform:** it auto-detects your OS and switches to the right setup —
Apple **GPU (Parakeet/MLX)** acceleration on Apple Silicon Macs, NVIDIA
**CUDA** acceleration on Windows, or CPU anywhere else. No code changes
needed when moving the repo between machines.

## Features

- Real-time voice-to-text transcription powered by a background Whisper worker
- **Automatic per-OS setup** (compute backend, hotkeys, audio routing) with a
  manual override in the UI (System tab)
- **Hardware acceleration everywhere**: CUDA on Windows, Apple-GPU MLX on Macs
- **No sudo on macOS** — global hotkeys use `pynput` (Accessibility permission)
- Minimal overlay indicator to show capture state at a glance
- Smart duplicate and filler filtering with a customizable block list
- Automatic low-volume gating + no-speech filtering to avoid hallucinations
- Auto-shutdown after extended inactivity with quick restart
- Optional real-time voice changer routed through a virtual audio cable

## Requirements

- Python 3.10+ on macOS (for the Parakeet engine; `brew install python@3.12`),
  Python 3.9+ elsewhere
- **macOS:** Apple Silicon recommended (for GPU acceleration) + Homebrew
- **Windows:** a CUDA-capable NVIDIA GPU is optional but recommended

## Installation

The transcription engine and hotkey backend are chosen automatically at
runtime, and `requirements.txt` uses platform markers so each OS only installs
what it needs.

### macOS

```bash
# 1. System dependencies: mic capture + a Python new enough for Parakeet
brew install portaudio python@3.12

# 2. Create a virtual environment and install deps
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # installs parakeet-mlx + mlx-whisper + pynput
```

**Grant permissions once** (System Settings → Privacy & Security) to the app
you launch from (Terminal / iTerm / your IDE):

- **Microphone** – to capture audio
- **Accessibility** – so transcribed text can be typed into other apps
- **Input Monitoring** – so the global hotkeys work

No `sudo` required.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt        # installs torch (CUDA) + keyboard on Windows
```

## Usage

```bash
python yiliVoice.py
```

The first run downloads the model weights (cached afterwards).

### Acceleration

| Platform | Backend | Device | Picked when |
|----------|---------|--------|-------------|
| Apple Silicon Mac | `parakeet-mlx` (parakeet-tdt-0.6b) | Apple GPU | `parakeet-mlx` installed (Python ≥ 3.10) |
| Apple Silicon Mac | `mlx-whisper` | Apple GPU | fallback when parakeet-mlx is missing |
| Windows / Linux + NVIDIA | `faster-whisper` | CUDA (float16) | CUDA available |
| Anything else | `faster-whisper` | CPU (int8) | fallback |

Parakeet is preferred on Macs because it is both faster than Whisper
small.en (~0.2 s vs ~0.34 s per phrase on an M3) and more accurate, and it
emits nothing on silence (no "thank you" hallucinations). English uses
`parakeet-tdt-0.6b-v2`; `--non_english` switches to the 25-language v3
checkpoint (for languages outside those 25, uninstall parakeet-mlx or pin
Whisper by editing the profile).

The active backend and accelerator are shown in the **System** tab of the
control panel (click the floating overlay to open it).

### Controls

Hotkeys differ per OS (and are listed in the System tab):

| Action | macOS | Linux | Windows |
|--------|-------|-------|---------|
| Toggle recording | `Ctrl+5` | `Ctrl+Alt+V` | `Pause / Break` |
| Toggle voice changer | `Ctrl+Option+9` | `Ctrl+Alt+9` | `Ctrl+F9` |
| Toggle voice-changer output | `Ctrl+Option+0` | `Ctrl+Alt+0` | `Ctrl+F10` |

> macOS global hotkeys require **Input Monitoring** + **Accessibility**
> permission for your terminal app (System Settings → Privacy & Security),
> then a full restart of that terminal. Without it the hotkeys do nothing.

The overlay appears at the top-center of the screen (just below the macOS
menu bar). Indicator colors: **Amber = loading model**, Green = recording,
Violet = ready, Red = idle/auto-shutdown.

### Control Panel (click the overlay)

- **Microphone** – pick your input device, tune the volume threshold
- **System** – view detected OS / accelerator / hotkeys, and pin a platform
  profile (Auto / Windows / macOS / Linux)
- **Filters** – view active filler/stop-word filters
- **Voice Changer** – real-time pitch shift to a virtual audio cable
- **Settings** – live view of the current configuration; **Save Settings**
  persists your choices (mic, platform, thresholds) to `settings/`

### Voice changer routing

The voice changer sends pitch-shifted audio to a virtual audio device that
your game/app then uses as its microphone:

- **Windows:** install [VB-CABLE](https://vb-audio.com/Cable/)
- **macOS:** `brew install blackhole-2ch` (then optionally build a
  Multi-Output Device in Audio MIDI Setup to also hear yourself)

The Voice Changer tab shows the exact steps for your OS.

### Command Line Arguments

- `--model`: Whisper size (tiny/base/small/medium/large). Default: per-platform
  recommendation (small on Apple GPU, medium on CUDA, base on CPU)
- `--platform`: Pin the platform profile (auto/windows/macos/linux). Default: auto
- `--non_english`: Enable non-English language detection
- `--energy_threshold`: Microphone detection sensitivity [default: 1000]
- `--record_timeout`: Audio buffer length in seconds [default: 1.5]
- `--phrase_timeout`: Silence duration that ends a phrase in seconds [default: 0.8]
- `--volume_threshold`: Minimum peak volume (0-1) to keep audio [default: 0.008]
- `--trailing_silence`: Final drain window after a phrase ends in seconds [default: 0.25]
- `--threshold_adjustment`: Multiplier that makes repetition filtering stricter [default: 1.0]
- `--no_speech_threshold`: Suppress output above this no-speech probability [default: 0.6]

## Architecture notes

- `settings/platform_profile.py` — single source of truth for per-OS behavior
- `utils/transcription.py` — backend abstraction (faster-whisper / MLX)
- `utils/hotkeys.py` — global-hotkey abstraction (keyboard / pynput)
