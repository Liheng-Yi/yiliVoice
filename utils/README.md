# yiliVoice Utils Module

This directory contains utility functions and classes for the yiliVoice application, focused on audio processing and text manipulation.

## Module Structure

### `audio_utils.py` (144 lines)
- **AudioBuffer class**: Optimized audio buffer with threading support and size limits
- **Text processing functions**: Clean, normalize, and filter transcription text
- **Duplicate detection**: Advanced algorithms to prevent repeated text output
- **Phrase processing**: Collapse repeated phrases from Whisper transcription output

## Key Components

### AudioBuffer Class
- Thread-safe audio data management
- Automatic size limiting to prevent memory issues
- Efficient append and retrieval operations

### Text Processing Functions
- `clean_sentence()`: Clean up transcription output (remove periods, capitalize)
- `normalize_filter_text()`: Normalize text for duplicate checking
- `is_duplicate_or_partial()`: Detect repeated or partial transcriptions
- `collapse_repeated_phrases()`: Remove consecutive duplicate sentences

## Usage

```python
from utils import (
    AudioBuffer, 
    clean_sentence,
    normalize_filter_text,
    is_duplicate_or_partial,
    collapse_repeated_phrases
)

# Create audio buffer
buffer = AudioBuffer(max_size=16000 * 30)  # 30 seconds

# Process text
clean_text = clean_sentence("hello world.")  # "Hello world"
normalized = normalize_filter_text("Hello, World!")  # "hello world"

# Check for duplicates
is_dup = is_duplicate_or_partial("hello", "hello world")  # True
```

## Features

- **Memory efficient**: Optimized for real-time audio processing
- **Thread-safe**: All operations are safe for multi-threaded use
- **Robust filtering**: Advanced duplicate detection algorithms
- **Text cleaning**: Consistent output formatting
