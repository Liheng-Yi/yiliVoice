# YiliVoice - Voice-to-Text Tool

A personal development tool that uses OpenAI's Whisper model for responsive voice recognition and transcription.

## Features

- Real-time voice-to-text transcription powered by a background Whisper worker
- Minimal overlay indicator to show capture state at a glance
- GPU acceleration support (CUDA) when available
- Smart duplicate and filler filtering with a customizable block list
- Automatic low-volume gating to avoid transcribing background noise
- Drops high no-speech probability outputs to reduce "thank you" hallucinations
- Auto-shutdown after extended inactivity with quick restart

## Requirements

- Python 3.7+
- CUDA-compatible GPU (optional, for faster processing)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/yiliVoice.git
   cd yiliVoice
   ```

2. Install required packages:
   ```bash
   pip install numpy speech_recognition openai-whisper torch pyautogui keyboard tkinter
   ```

## Usage

Run the script with default settings:

```bash
python yiliVoice.py
```

### Command Line Arguments

- `--model`: Choose Whisper model size (tiny/base/small/medium/large) [default: medium]
- `--non_english`: Enable non-English language detection
- `--energy_threshold`: Microphone detection sensitivity [default: 1000]
- `--record_timeout`: Audio buffer length in seconds [default: 1.5]
- `--phrase_timeout`: Silence duration that ends a phrase in seconds [default: 1.0]
- `--volume_threshold`: Minimum peak volume (0-1 range) to keep audio [default: 0.008]
- `--trailing_silence`: Additional silence captured after speech stops [default: 0.8]
- `--threshold_adjustment`: Multiplier that makes repetition filtering stricter [default: 1.0]
- `--no_speech_threshold`: Suppress output when Whisper predicts high no-speech probability [default: 0.6]

### Controls

- `Ctrl+F8`: Toggle recording on/off
- Indicator colors: Green = recording, Pink = ready, Red = auto-shutdown

### Status Indicator

A small overlay circle appears at the top-center of your screen:
- Green indicates active recording
- Pink shows the system is ready
- Red appears when the system enters auto-shutdown mode after roughly 10 minutes of inactivity

## Features in Detail

### Automatic Filtering
The system automatically filters out:
- Duplicate phrases
- Partial repetitions
- Common filler phrases (customizable in the filter list)
- Short phrases below the minimum word count
- High no-speech probability segments that usually produce "thank you" hallucinations

### Performance
- Dedicated worker threads for audio capture, transcription, and UI updates
- Bytearray-backed audio buffer to minimize copying and memory churn
- GPU acceleration when available (CUDA)
- Ambient noise calibration during start-up

