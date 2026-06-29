"""Transcription backend abstraction.

Hides the difference between two Whisper engines behind one interface:

  * :class:`FasterWhisperBackend` — CTranslate2 engine used on Windows/Linux.
    Runs on NVIDIA CUDA (float16) or CPU (int8).  CTranslate2 has **no**
    Apple-GPU support, so on a Mac it can only ever use the CPU.
  * :class:`MLXWhisperBackend` — Apple's MLX engine, which runs Whisper on
    the Apple-Silicon GPU.  This is the Mac equivalent of CUDA.

Both return a normalized :class:`TranscriptionResult(text, no_speech_prob)`
so the rest of the app doesn't care which engine is active.

Use :func:`create_backend` with a
:class:`settings.platform_profile.PlatformProfile` to get the right one,
with automatic fallback to faster-whisper/CPU if MLX can't be loaded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from settings.platform_profile import BACKEND_MLX, BACKEND_FASTER_WHISPER


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


# --------------------------------------------------------------------------- #
# Backends                                                                     #
# --------------------------------------------------------------------------- #

class TranscriptionBackend:
    """Common interface for transcription engines."""

    name = "base"
    device_label = "unknown"
    model_ref = ""

    def transcribe(
        self,
        audio_np: np.ndarray,
        *,
        non_english: bool,
        no_speech_threshold: float,
        compression_ratio_threshold: float,
    ) -> TranscriptionResult:
        raise NotImplementedError

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
            beam_size=5,
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

    If the profile asks for MLX but it can't be imported/loaded, we fall
    back to faster-whisper on the CPU so the app still works.
    """
    if profile.backend == BACKEND_MLX:
        try:
            return MLXWhisperBackend(size, non_english)
        except Exception as exc:
            print(f"[Transcribe] MLX unavailable ({exc}); falling back to faster-whisper/CPU.")
            return FasterWhisperBackend(size, non_english, device="cpu", compute_type="int8")

    return FasterWhisperBackend(
        size, non_english, device=profile.device, compute_type=profile.compute_type
    )
