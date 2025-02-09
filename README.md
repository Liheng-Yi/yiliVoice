# YiliVoice - Voice-to-Text Tool

A personal development tool that uses OpenAI's Whisper model for real-time voice recognition and transcription.

## Features

- Real-time voice-to-text transcription
- Visual status indicator (overlay window)
- GPU acceleration support (CUDA)
- Automatic duplicate/partial phrase filtering
- Auto-shutdown on inactivity
- Customizable phrase filtering

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
bash
python yiliVoice.py

### Command Line Arguments

- `--model`: Choose Whisper model size (tiny/base/small/medium/large) [default: medium]
- `--non_english`: Enable non-English language detection
- `--energy_threshold`: Microphone detection sensitivity [default: 1000]
- `--record_timeout`: Audio buffer length in seconds [default: 2]
- `--phrase_timeout`: Silence duration to end phrase in seconds [default: 3]
- `--volume_threshold`: Minimum volume threshold [default: 0.01]

### Controls

- `Ctrl+Q`: Toggle recording on/off
- The overlay indicator shows:
  - 🟢 Green: Recording active
  - 🟪 Pink: Ready/Standby
  - 🔴 Red: Auto-shutdown mode

### Status Indicator

A small overlay circle appears at the top center of your screen:
- Green indicates active recording
- Pink shows the system is ready
- Red appears when system enters auto-shutdown mode (after 100 seconds of inactivity)

## Features in Detail

### Automatic Filtering
The system automatically filters out:
- Duplicate phrases
- Partial repetitions
- Common filler phrases (customizable in the filterList)
- Short phrases below minimum word count

### Performance
- GPU acceleration when available (CUDA)
- Efficient memory management with automatic CUDA cache clearing
- Ambient noise adjustment

