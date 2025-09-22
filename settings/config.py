import re
import os
import json
from datetime import datetime


def normalize_filter_text(text: str) -> str:
    """Normalize a phrase for duplicate/filler checks."""
    return re.sub(r'[^a-z0-9 ]+', ' ', text.lower()).strip()


class VoiceConfig:
    """Configuration class to centralize settings"""
    
    def __init__(self, args):
        self.model = args.model
        self.non_english = args.non_english
        self.energy_threshold = args.energy_threshold
        self.record_timeout = args.record_timeout
        self.phrase_timeout = args.phrase_timeout
        self.volume_threshold = args.volume_threshold
        self.volume_threshold_raw = args.volume_threshold * 32768.0
        self.no_speech_threshold = args.no_speech_threshold
        self.trailing_silence = args.trailing_silence
        self.threshold_adjustment = args.threshold_adjustment
        self.max_buffer_size = 16000 * 30  # 30 seconds max buffer
        self.inactivity_timeout = 600  # 10 minutes
        self.selected_microphone_index = None  # None means use system default

        # Filter list - externalize this to a config file later
        raw_filters = {
            "i'm sorry",
            "thanks for watching!",
            "i'll see you next time.",
            "i'm not gonna lie.",
            "thank you.",
            "thank you",
            "thank you very much",
            "thanks",
            "thanks very much",
            "thanks everyone",
            "thank you everyone",
            "i'm going to go get some food.",
            "i'm going to do it again.",
            "bye."
        }
        self.filter_list = {normalize_filter_text(item) for item in raw_filters}
        self.filter_patterns = [
            re.compile(r'^(?:thank|thanks)(?: you)?(?: so much| very much)?(?: everyone| all)?$', re.I),
            re.compile(r'^thanks(?: for watching| for tuning in)?$', re.I),
        ]

    def to_dict(self):
        """Convert configuration to dictionary for saving"""
        return {
            'model': self.model,
            'non_english': self.non_english,
            'energy_threshold': self.energy_threshold,
            'record_timeout': self.record_timeout,
            'phrase_timeout': self.phrase_timeout,
            'volume_threshold': self.volume_threshold,
            'no_speech_threshold': self.no_speech_threshold,
            'trailing_silence': self.trailing_silence,
            'threshold_adjustment': self.threshold_adjustment,
            'inactivity_timeout': self.inactivity_timeout,
            'selected_microphone_index': self.selected_microphone_index,
            'timestamp': datetime.now().isoformat()
        }

    def save_to_file(self, settings_dir="./settings"):
        """Save current settings to the settings directory"""
        try:
            # Create settings directory if it doesn't exist
            os.makedirs(settings_dir, exist_ok=True)
            
            # Save to JSON file
            settings_file = os.path.join(settings_dir, "voice_config.json")
            with open(settings_file, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            
            print(f"Settings saved to {settings_dir}")
            return True
            
        except Exception as e:
            print(f"Error saving settings: {e}")
            return False

    @classmethod
    def load_from_file(cls, settings_dir="./settings"):
        """Load settings from file if it exists"""
        settings_file = os.path.join(settings_dir, "voice_config.json")
        if os.path.exists(settings_file):
            try:
                with open(settings_file, 'r') as f:
                    settings = json.load(f)
                print(f"Settings loaded from {settings_file}")
                return settings
            except Exception as e:
                print(f"Error loading settings: {e}")
        return None
