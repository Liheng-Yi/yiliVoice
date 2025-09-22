import re
import threading
from collections import deque


def clean_sentence(text):
    """
    Clean up final text output:
    - Remove trailing period
    - Capitalize the first character
    """
    text = text.strip()
    if text.endswith('.'):
        text = text[:-1]
    return text.capitalize()


def normalize_filter_text(text: str) -> str:
    """Normalize a phrase for duplicate/filler checks."""
    return re.sub(r'[^a-z0-9 ]+', ' ', text.lower()).strip()


def is_duplicate_or_partial(new_text, previous_text, min_word_count=3):
    """
    Returns True if new_text is likely a repeated or partial repeat of previous_text.
    More aggressive in filtering out near-duplicates.
    """
    if not previous_text:
        return False

    new_text_clean = new_text.lower().strip()
    prev_text_clean = previous_text.lower().strip()

    # If either is blank or extremely short
    if len(new_text_clean) < min_word_count:
        return True

    # Check direct substring
    if new_text_clean in prev_text_clean or prev_text_clean in new_text_clean:
        return True

    # Token overlap check
    words_new = new_text_clean.split()
    words_prev = prev_text_clean.split()
    overlap_count = 0
    for word in words_new:
        if word in words_prev:
            overlap_count += 1

    # If more than half of new_text's words appear in previous_text in the same order,
    # treat it as repeated or partial. Adjust threshold to taste.
    if overlap_count >= (len(words_new) * 0.7):
        return True

    return False


def collapse_repeated_phrases(text: str, max_occurrences: int = 1) -> str:
    """
    Whisper will occasionally output the *same* short sentence back-to-back in a single
    transcription chunk (e.g. "i'm going to do it again." five times).  This utility
    removes such immediate repetitions so that only the first `max_occurrences` are
    retained.

    The algorithm works by:
      1. Splitting the text into sentence-like chunks on punctuation boundaries.
      2. Walking through those chunks and keeping at most `max_occurrences` adjacent
         duplicates (case-insensitive match).
      3. Re-assembling and returning the cleaned text.
    """
    if not text:
        return text

    # Simple sentence segmentation on ., ?, ! boundaries.
    # Keep the punctuation by using a regex capture group.
    parts = re.split(r'( *[.!?]+ *)', text)
    # Re-combine the sentence bodies with their trailing punctuations
    sentences: list[str] = []
    current = ""
    for seg in parts:
        if re.match(r' *[.!?]+ *', seg):
            current += seg  # punctuation part
            sentences.append(current.strip())
            current = ""
        else:
            current += seg
    if current.strip():
        sentences.append(current.strip())

    cleaned: list[str] = []
    for s in sentences:
        if not cleaned:
            cleaned.append(s)
            continue

        # Compare ignoring case and surrounding whitespace
        if s.strip().lower() == cleaned[-1].strip().lower():
            # Already have this sentence right before – keep only if we have
            # not yet reached the allowed max_occurrences.
            duplicates = 1  # We had at least one occurrence in cleaned[-1]
            # Count trailing identical sentences
            for prev in reversed(cleaned):
                if prev.strip().lower() == s.strip().lower():
                    duplicates += 1
                else:
                    break
            if duplicates <= max_occurrences:
                cleaned.append(s)
            # else skip adding – collapse the repetition.
        else:
            cleaned.append(s)

    return ' '.join(cleaned)


class AudioBuffer:
    """Optimized audio buffer with size limits and efficient memory management."""

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = bytearray()
        self._lock = threading.Lock()

    def append(self, data):
        if not data:
            return
        with self._lock:
            self.buffer.extend(data)
            if len(self.buffer) > self.max_size:
                excess = len(self.buffer) - self.max_size
                del self.buffer[:excess]

    def get_and_clear(self):
        with self._lock:
            if not self.buffer:
                return bytes()
            data = bytes(self.buffer)
            self.buffer.clear()
            return data

    def __len__(self):
        with self._lock:
            return len(self.buffer)
