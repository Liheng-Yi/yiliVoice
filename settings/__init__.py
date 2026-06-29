# Settings package for yiliVoice
from .config import (
    VoiceConfig,
    normalize_filter_text,
    _build_filler_strip_pattern,
)
from .platform_profile import (
    PlatformProfile,
    build_profile,
    detect_os,
    is_apple_silicon,
    VALID_OVERRIDES,
    OS_DISPLAY_NAMES,
    PROFILE_AUTO,
)
from .debug_ui import DebugUI
from .ui import create_overlay_window, update_indicator

__all__ = [
    'VoiceConfig',
    'normalize_filter_text',
    '_build_filler_strip_pattern',
    'PlatformProfile',
    'build_profile',
    'detect_os',
    'is_apple_silicon',
    'VALID_OVERRIDES',
    'OS_DISPLAY_NAMES',
    'PROFILE_AUTO',
    'DebugUI',
    'create_overlay_window',
    'update_indicator',
]
