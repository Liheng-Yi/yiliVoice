"""Transcription backend abstraction.

Hides the difference between three ASR engines behind one interface:

  * :class:`ParakeetMLXBackend` — NVIDIA's parakeet-tdt-0.6b running on the
    Apple-Silicon GPU via MLX.  Fastest AND most accurate option for
    dictation on a Mac (leads the Open ASR leaderboard), and it produces no
    text on silence, so it doesn't need Whisper's hallucination filtering.
  * :class:`MLXWhisperBackend` — Apple's MLX engine, which runs Whisper on
    the Apple-Silicon GPU.  Fallback when parakeet-mlx isn't installed
    (it needs Python >= 3.10) and the multilingual choice for languages
    Parakeet doesn't cover.
  * :class:`FasterWhisperBackend` — CTranslate2 engine used on Windows/Linux.
    Runs on NVIDIA CUDA (float16) or CPU (int8).  CTranslate2 has **no**
    Apple-GPU support, so on a Mac it can only ever use the CPU.

All return a normalized :class:`TranscriptionResult(text, no_speech_prob)`
so the rest of the app doesn't care which engine is active.

Use :func:`create_backend` with a
:class:`settings.platform_profile.PlatformProfile` to get the right one,
with automatic fallback (Parakeet → MLX Whisper → faster-whisper/CPU).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from settings.platform_profile import (
    BACKEND_MLX,
    BACKEND_FASTER_WHISPER,
    BACKEND_PARAKEET,
)


@dataclass
class TranscriptionResult:
    text: str
    no_speech_prob: float = 0.0


# --------------------------------------------------------------------------- #
# Model-name helpers                                                           #
# --------------------------------------------------------------------------- #

def faster_whisper_model_name(size: str, non_english: bool) -> str:
    """e.g. ('medium', False) -> 'medium.en';  ('large', True) -> 'large-v3'."""
    if size == "large":
        return "large-v3"
    return size if non_english else f"{size}.en"


def mlx_repo_name(size: str, non_english: bool) -> str:
    """Map a model size to an mlx-community HF repo.

    English-only ``.en`` checkpoints exist for tiny/base/small/medium.
    'large' only ships as the multilingual large-v3.
    """
    if size == "large":
        return "mlx-community/whisper-large-v3-mlx"
    suffix = "" if non_english else ".en"
    return f"mlx-community/whisper-{size}{suffix}-mlx"


def parakeet_repo_name(non_english: bool) -> str:
    """Pick the Parakeet checkpoint: v2 is English-only (best English WER),
    v3 adds 25 (mostly European) languages at a small English accuracy cost."""
    if non_english:
        return "mlx-community/parakeet-tdt-0.6b-v3"
    return "mlx-community/parakeet-tdt-0.6b-v2"


# --------------------------------------------------------------------------- #
# Backends                                                                     #
# --------------------------------------------------------------------------- #

class TranscriptionBackend:
    """Common interface for transcription engines."""

    name = "base"
    device_label = "unknown"
    model_ref = ""
    supports_streaming = False  # can this backend produce live partial results?

    def transcribe(
        self,
        audio_np: np.ndarray,
        *,
        non_english: bool,
        no_speech_threshold: float,
        compression_ratio_threshold: float,
    ) -> TranscriptionResult:
        raise NotImplementedError

    def open_stream(self):
        """Open a live streaming session (only if ``supports_streaming``).

        Returns an object with ``feed(audio_np)`` / ``text`` / ``close()``.
        """
        raise NotImplementedError("this backend does not support streaming")

    def warm_up(self) -> None:
        """Trigger model download / kernel compilation with a tiny clip."""
        try:
            silence = np.zeros(1600, dtype=np.float32)  # 0.1 s @ 16 kHz
            self.transcribe(
                silence,
                non_english=False,
                no_speech_threshold=1.0,
                compression_ratio_threshold=10.0,
            )
        except Exception as exc:  # warm-up is best-effort
            print(f"[Transcribe] Warm-up skipped: {exc}")


class FasterWhisperBackend(TranscriptionBackend):
    name = BACKEND_FASTER_WHISPER

    def __init__(self, size: str, non_english: bool, device: str, compute_type: str):
        from faster_whisper import WhisperModel

        self.device = device
        self.compute_type = compute_type
        self.device_label = f"faster-whisper · {device}/{compute_type}"
        self.model_ref = faster_whisper_model_name(size, non_english)
        print(f"[Transcribe] Loading faster-whisper '{self.model_ref}' on {device} ({compute_type})...")
        self.model = WhisperModel(self.model_ref, device=device, compute_type=compute_type)
        print(f"[Transcribe] faster-whisper ready on {device}.")

    def transcribe(self, audio_np, *, non_english, no_speech_threshold, compression_ratio_threshold):
        segments, _info = self.model.transcribe(
            audio_np,
            language="en" if not non_english else None,
            temperature=0.0,
            beam_size=1,  # greedy: 2-3x faster than beam=5, ~same for dictation
            compression_ratio_threshold=compression_ratio_threshold,
            condition_on_previous_text=False,
            vad_filter=True,  # Silero VAD (faster-whisper only)
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        segments = list(segments)
        if not segments:
            return TranscriptionResult("", 1.0)
        no_speech_prob = float(segments[0].no_speech_prob)
        text = " ".join(s.text for s in segments).strip()
        return TranscriptionResult(text, no_speech_prob)


class ParakeetStream:
    """A live Parakeet streaming session.

    Wraps parakeet-mlx's ``transcribe_stream`` context manager (which yields a
    ``StreamingParakeet``) so the app can push audio as it arrives and read the
    running transcript.  The transcript's TAIL is not stable — the model
    revises the last few words (and, rarely, earlier ones) as more audio comes
    in — so the caller must only *commit* a held-back prefix, never the whole
    string.  See :class:`utils.text_output.IncrementalTyper`.
    """

    def __init__(self, model, mx, context_size=(256, 256), depth=1):
        self._mx = mx
        # Enter the context manager manually so its lifetime spans the whole
        # recording session (open on record-start, close on record-stop).
        self._st = model.transcribe_stream(context_size=context_size, depth=depth)
        self._st.__enter__()

    def feed(self, audio_np) -> None:
        """Push a 1-D float32 chunk (16 kHz) into the stream."""
        self._st.add_audio(self._mx.array(audio_np))

    @property
    def text(self) -> str:
        return (self._st.result.text or "").strip()

    def close(self) -> None:
        try:
            self._st.__exit__(None, None, None)
        except Exception:
            pass


class ParakeetMLXBackend(TranscriptionBackend):
    """NVIDIA parakeet-tdt-0.6b on the Apple-Silicon GPU via parakeet-mlx.

    The model is a TDT (token-and-duration transducer): greedy, no beam
    search, no 30-second mel padding — which is why it turns around a short
    dictation clip in ~0.2 s where Whisper small.en needs ~0.34 s, with
    better accuracy.  It emits nothing on silence, so ``no_speech_prob`` is
    reported as 0.0 (never filtered).

    It also supports true streaming (``supports_streaming = True``): the app
    feeds audio as you speak and types the stabilized prefix live, so the
    text is on screen almost the instant you stop talking.
    """

    name = BACKEND_PARAKEET
    supports_streaming = True

    def __init__(self, non_english: bool):
        import mlx.core as mx  # raises if unavailable -> caller falls back
        from parakeet_mlx import from_pretrained
        from parakeet_mlx.audio import get_logmel

        self._mx = mx
        self._get_logmel = get_logmel
        self.device_label = "parakeet-mlx · Apple GPU/bf16"
        self.model_ref = parakeet_repo_name(non_english)
        print(f"[Transcribe] Loading Parakeet (Apple GPU) '{self.model_ref}'...")
        print("[Transcribe] First run downloads the model from Hugging Face; please wait...")
        self.model = from_pretrained(self.model_ref)
        print("[Transcribe] Parakeet ready.")

    def transcribe(self, audio_np, *, non_english, no_speech_threshold, compression_ratio_threshold):
        # Whisper-specific thresholds don't apply: Parakeet is a transducer
        # (no compression-ratio loops, no no-speech logits) and stays silent
        # on non-speech input.
        mel = self._get_logmel(self._mx.array(audio_np), self.model.preprocessor_config)
        results = self.model.generate(mel)
        text = results[0].text.strip() if results else ""
        return TranscriptionResult(text, 0.0)

    def open_stream(self) -> ParakeetStream:
        return ParakeetStream(self.model, self._mx)

    def warm_up(self) -> None:
        # Warm both the batch path and the streaming path (the first
        # add_audio compiles kernels and would otherwise cost ~1.8 s mid-speech).
        super().warm_up()
        try:
            stream = self.open_stream()
            stream.feed(np.zeros(1600, dtype=np.float32))
            _ = stream.text
            stream.close()
        except Exception as exc:
            print(f"[Transcribe] Streaming warm-up skipped: {exc}")


class MLXWhisperBackend(TranscriptionBackend):
    name = BACKEND_MLX

    def __init__(self, size: str, non_english: bool):
        import mlx_whisper  # raises if unavailable -> caller falls back

        self._mlx_whisper = mlx_whisper
        self.device_label = "mlx · Apple GPU/float16"
        self.model_ref = mlx_repo_name(size, non_english)
        print(f"[Transcribe] Using MLX (Apple GPU) '{self.model_ref}'.")
        print("[Transcribe] First run downloads the model from Hugging Face; please wait...")

    def transcribe(self, audio_np, *, non_english, no_speech_threshold, compression_ratio_threshold):
        # mlx-whisper caches the loaded model per repo internally.
        result = self._mlx_whisper.transcribe(
            audio_np,
            path_or_hf_repo=self.model_ref,
            language="en" if not non_english else None,
            temperature=0.0,
            condition_on_previous_text=False,
            compression_ratio_threshold=compression_ratio_threshold,
            no_speech_threshold=no_speech_threshold,
            fp16=True,  # run on the Apple GPU
            verbose=None,
        )
        text = (result.get("text") or "").strip()
        segments = result.get("segments") or []
        no_speech_prob = float(segments[0].get("no_speech_prob", 0.0)) if segments else 0.0
        return TranscriptionResult(text, no_speech_prob)


# --------------------------------------------------------------------------- #
# Factory                                                                      #
# --------------------------------------------------------------------------- #

def create_backend(profile, size: str, non_english: bool) -> TranscriptionBackend:
    """Build the backend dictated by *profile*, with safe fallback.

    Fallback chain: Parakeet → MLX Whisper → faster-whisper/CPU, so the app
    still works when the preferred engine can't be imported or loaded.
    """
    backend = profile.backend

    if backend == BACKEND_PARAKEET:
        try:
            return ParakeetMLXBackend(non_english)
        except Exception as exc:
            print(f"[Transcribe] Parakeet unavailable ({exc}); falling back to MLX Whisper.")
            backend = BACKEND_MLX

    if backend == BACKEND_MLX:
        try:
            return MLXWhisperBackend(size, non_english)
        except Exception as exc:
            print(f"[Transcribe] MLX unavailable ({exc}); falling back to faster-whisper/CPU.")
            return FasterWhisperBackend(size, non_english, device="cpu", compute_type="int8")

    return FasterWhisperBackend(
        size, non_english, device=profile.device, compute_type=profile.compute_type
    )
