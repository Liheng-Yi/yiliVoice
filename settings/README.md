# yiliVoice Settings Module

This directory contains the settings and configuration components of the yiliVoice application, organized for better maintainability and code structure.

## Module Structure

### `config.py` (90 lines)
- **VoiceConfig class**: Centralized configuration management
- **Settings persistence**: Save/load configuration to/from JSON files
- **Filter management**: Audio transcription filters and patterns

### `debug_ui.py` (196 lines)
- **DebugUI class**: Complete debug window functionality
- **Microphone information**: Real-time device detection and status
- **Settings display**: Live configuration and system information
- **Export functionality**: Save current settings to ./settings directory

### `ui.py` (35 lines)
- **Overlay window**: Status indicator dot creation and management
- **Click handling**: Debug window toggle functionality
- **Status updates**: Real-time indicator color changes

### `__init__.py` (11 lines)
- **Package exports**: Clean imports for the main application
- **Module organization**: Centralized access to all settings components

## Benefits of This Structure

1. **Reduced main file size**: From 916 to 480 lines (48% reduction)
2. **Better organization**: Related functionality grouped together
3. **Easier maintenance**: Isolated components for specific features
4. **Cleaner imports**: Simple, organized module access
5. **Scalability**: Easy to add new features without cluttering main file

## Usage

```python
from settings import (
    VoiceConfig, DebugUI,
    create_overlay_window, update_indicator
)
from utils import (
    AudioBuffer, clean_sentence,
    normalize_filter_text, is_duplicate_or_partial
)
```

All settings are automatically saved to `./settings/` directory when using the debug window's "Save Settings" button.

**Note**: Audio processing utilities have been moved to the `../utils/` directory for better organization.
