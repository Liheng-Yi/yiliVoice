import re
import os
import json
from datetime import datetime


def normalize_filter_text(text: str) -> str:
    """Normalize a phrase for duplicate/filler checks."""
    return re.sub(r'[^a-z0-9 ]+', ' ', text.lower()).strip()


# ---------------------------------------------------------------------------
# Default filter lists (used when filters.json is absent or incomplete)
# ---------------------------------------------------------------------------
_DEFAULT_FILLER_WORDS = [
    "um", "uh", "uhh", "uhhh",
    "em", "emm", "emmm", "emmmm",
    "hmm", "hmmm", "hmmmm",
    "err", "errr",
    "erm", "ermm",
    "ah", "ahh", "ahhh",
    "oh", "ohh", "ohhh",
    "eh", "ehh",
    "like",
    "you know",
    "i mean",
    "sort of",
    "kind of",
    "basically",
    "literally",
    "actually",
    "so",
    "well",
    "right",
    "okay so",
    "alright so",
    "anyway",
]

def _load_filters(settings_dir: str = "./settings") -> dict:
    """Load filler_words from filters.json.

    Falls back to the built-in defaults if the file is missing or malformed.
    """
    filters_file = os.path.join(settings_dir, "filters.json")
    try:
        with open(filters_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        filler = data.get("filler_words", _DEFAULT_FILLER_WORDS)
        print(f"Filters loaded from {filters_file} ({len(filler)} filler)")
        return {"filler_words": filler}
    except FileNotFoundError:
        print(f"filters.json not found at {filters_file}; using built-in defaults.")
    except Exception as exc:
        print(f"Error loading filters.json: {exc}; using built-in defaults.")
    return {"filler_words": _DEFAULT_FILLER_WORDS}


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
        self.selected_microphone_index = 1  # Device 1 as default

        # ------------------------------------------------------------------ #
        # Load external filter lists (filters.json)                           #
        # ------------------------------------------------------------------ #
        _filters = _load_filters()

        # -- Filler / stopping words ----------------------------------------
        # Stored as a set of normalized strings for O(1) whole-phrase look-ups.
        raw_filler: list = _filters["filler_words"]
        self.filler_words: set[str] = {normalize_filter_text(w) for w in raw_filler}

        # Regex pattern: match the filler word/phrase surrounded by word
        # boundaries (or string edges) so that "um" doesn't strip "umbrella".
        self._filler_strip_pattern: re.Pattern = _build_filler_strip_pattern(raw_filler)

        # ------------------------------------------------------------------ #
        # Legacy hard-coded filter list (whole-transcript suppression)        #
        # ------------------------------------------------------------------ #
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

    # ---------------------------------------------------------------------- #
    # Persistence helpers                                                     #
    # ---------------------------------------------------------------------- #

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
            os.makedirs(settings_dir, exist_ok=True)
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


# --------------------------------------------------------------------------- #
# Helper builders (module-level, re-used by VoiceConfig)                      #
# --------------------------------------------------------------------------- #

def _build_filler_strip_pattern(filler_words: list) -> re.Pattern:
    """Build a regex that matches any filler word/phrase at a word boundary.

    Longer phrases are placed first so they take priority over shorter ones
    (e.g. "you know" before "you").
    """
    sorted_words = sorted(filler_words, key=len, reverse=True)
    escaped = [re.escape(w) for w in sorted_words]
    # \\b works for single-word fillers; for multi-word phrases we wrap with
    # (?<!\w) / (?!\w) look-arounds which work across the whole phrase.
    pattern = r'(?<![a-z0-9])(?:' + '|'.join(escaped) + r')(?![a-z0-9])'
    return re.compile(pattern, re.IGNORECASE)
