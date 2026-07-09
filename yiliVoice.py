import argparse
import os
import signal
import numpy as np
import speech_recognition as sr
import threading
import time
from collections import deque

from datetime import datetime, timedelta, timezone
from queue import Queue, Empty, Full


# Import our modular components
from settings import (
    VoiceConfig, DebugUI,
    create_overlay_window, update_indicator,
)
from utils import (
    AudioBuffer, clean_sentence, normalize_filter_text,
    is_duplicate_or_partial, collapse_repeated_phrases,
    strip_filler_words,
    VoiceConverter, SOUNDDEVICE_AVAILABLE,
    create_backend, create_hotkey_manager, play_cue,
    TextTyper, IncrementalTyper,
    fetch_usage, claude_available,
    fetch_ccusage, bunx_available,
)


def _force_quit(signum, frame):
    """Hard-exit on Ctrl+C / SIGTERM.

    The Tk event loop and the pynput listeners can swallow SIGINT on macOS,
    so a normal KeyboardInterrupt may never reach us. Exiting immediately from
    the signal handler guarantees Ctrl+C always quits; the OS reclaims the mic,
    audio streams and event taps.
    """
    print("\nInterrupt received — exiting.", flush=True)
    os._exit(0)




class VoiceRecognitionApp:
    def __init__(self, config):
        self.config = config
        self.profile = config.profile
        self.backend = None          # transcription backend (faster-whisper / mlx)
        self.hotkeys = None          # global-hotkey manager
        self.ready = False           # True once the model + audio are loaded
        self.device = self.profile.accelerator_label
        self.recorder = None
        self.source = None
        self.background_listener = None
        
        # Threading events for better control
        self.recording_event = threading.Event()
        self.shutdown_event = threading.Event()
        self.activity_event = threading.Event()
        
        # Audio processing
        self.data_queue = Queue()
        self.ui_update_queue = Queue()  # Queue for UI updates
        self.transcription_queue = Queue(maxsize=3)
        self.audio_buffer = AudioBuffer(config.max_buffer_size)
        self.transcription = deque(maxlen=100)  # Limit transcription history
        self.last_chunk_size = 0     # bytes in the most recent mic chunk
        self._validated_mic = None   # (resolved_index,) that last passed preflight

        # Typed-text output. Constructed here (main thread) because the pynput
        # backend pre-warms the macOS key layout, which must happen on the
        # main thread; typing itself then works from any worker thread.
        self.typer = TextTyper()
        # Live streaming output (used only when the backend supports streaming).
        self.incremental_typer = IncrementalTyper(self.typer)
        self.use_streaming = False   # set once the backend is known
        self.stream = None           # active ParakeetStream during recording

        # Meter below the dot: limit bars (needs `claude`) + ccusage spend
        # (needs `bunx`). Each half decided in create_ui by CLI availability.
        self.show_usage = False
        self.show_cost = False
        self.usage_refresh_event = threading.Event()  # set to poll now

        # Timing
        self.phrase_time = datetime.now(timezone.utc)
        self.last_activity_time = datetime.now(timezone.utc)

        # Voice converter (real-time pitch shift → virtual cable).
        # Hotkey debounce is handled centrally by the HotkeyManager.
        self.voice_converter = None

        # UI (Qt). ``root``/``canvas``/``indicator`` all alias the status
        # window to stay compatible with the queue-driven update helpers.
        self.qt_app = None
        self.window = None
        self.root = None
        self.canvas = None
        self.indicator = None
        self.debug_ui = None
        
    @staticmethod
    def _drain_queue(queue_obj):
        """Remove all pending items from a queue without blocking."""
        try:
            while True:
                queue_obj.get_nowait()
        except Empty:
            return
        
    def initialize_model(self):
        """Initialize the transcription backend for the active platform.

        The backend is chosen automatically by the platform profile:
        Apple-GPU MLX on Apple Silicon, NVIDIA CUDA on Windows/Linux with
        a CUDA GPU, or CPU otherwise.
        """
        self.profile = self.config.profile
        print(f"[Platform] {self.profile.summary()}")
        self.device = self.profile.accelerator_label

        self.backend = create_backend(
            self.profile, self.config.model, self.config.non_english
        )
        print("Warming up model (first run may download weights)...")
        self.backend.warm_up()
        print(f"Model ready: {self.backend.model_ref}  [{self.backend.device_label}]")
    
    def initialize_audio(self):
        """Initialize audio recording components"""
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = self.config.energy_threshold
        self.recorder.dynamic_energy_threshold = False
        self.recorder.pause_threshold = 0.5
        # Don't create the source here - we'll create it fresh each time
        self.source = None
        
        # Do initial ambient noise adjustment with a temporary source
        temp_source = sr.Microphone(sample_rate=16000)
        with temp_source:
            self.recorder.adjust_for_ambient_noise(temp_source)
        # temp_source is properly disposed of here
    
    def _resolve_and_preflight(self, requested):
        """Resolve a usable input-device index and verify it opens.

        Returns ``(device_index, None)`` on success (``None`` index = system
        default) or ``(None, error_message)`` on failure.

        A stale/invalid index — e.g. a saved ``0`` that points at an output
        device with no input channels — would make ``sr.Microphone`` fail to
        open. speech_recognition *swallows* that failure (it leaves
        ``source.stream = None`` and returns normally), so the background
        listener later crashes with a cryptic ``NoneType`` error. We avoid
        that by test-opening candidates (requested → default input → any
        input device) until one works.

        Everything runs in ONE PyAudio session, and a device that passed
        before skips the full enumeration — recording used to pay three
        CoreAudio init/teardown cycles per hotkey press, which delayed
        capture enough to clip the first word.
        """
        try:
            import pyaudio
        except Exception:
            return requested, None  # can't test; let the normal path try

        pa = pyaudio.PyAudio()
        try:
            def opens(idx):
                kwargs = dict(format=pyaudio.paInt16, channels=1, rate=16000,
                              input=True, frames_per_buffer=1024)
                if idx is not None:
                    kwargs["input_device_index"] = idx
                try:
                    stream = pa.open(**kwargs)
                    stream.close()
                    return True, None
                except Exception as exc:
                    return False, str(exc)

            # Fast path: the previously validated device — just confirm it
            # still opens (catches unplugging / a revoked permission).
            if self._validated_mic is not None and self._validated_mic[0] == requested:
                ok, _ = opens(requested)
                if ok:
                    return requested, None
                self._validated_mic = None  # device went away; re-resolve

            count = pa.get_device_count()

            def is_input(i):
                try:
                    return pa.get_device_info_by_index(i).get("maxInputChannels", 0) > 0
                except Exception:
                    return False

            candidates = []
            if requested is not None and 0 <= requested < count and is_input(requested):
                candidates.append(requested)
            try:
                default_idx = pa.get_default_input_device_info().get("index")
            except Exception:
                default_idx = None
            if default_idx is not None and is_input(default_idx) and default_idx not in candidates:
                candidates.append(default_idx)
            for i in range(count):
                if is_input(i) and i not in candidates:
                    candidates.append(i)

            last_err = None
            for idx in candidates:
                ok, err = opens(idx)
                if ok:
                    if idx != requested:
                        print(f"Microphone index {requested} unusable; using input [{idx}].")
                    self._validated_mic = (idx,)
                    return idx, None
                last_err = err

            # Last resort: the system default without an explicit index.
            ok, err = opens(None)
            if ok:
                self._validated_mic = (None,)
                return None, None
            return None, last_err or err or "no usable input devices found"
        finally:
            pa.terminate()

    def create_fresh_microphone_source(self):
        """Create a fresh, validated microphone source.

        Resolves a usable input-device index and pre-flights the open so a bad
        device or a missing macOS Microphone permission raises here (caught by
        toggle_recording) instead of crashing the background listener thread.
        """
        requested = self.config.selected_microphone_index
        device_index, err = self._resolve_and_preflight(requested)
        if err is not None:
            raise OSError(
                f"could not open microphone (device {requested}): {err}. "
                "On macOS, grant Microphone access in "
                "System Settings → Privacy & Security → Microphone."
            )
        # Remember the resolved choice so the UI + saved config reflect reality.
        self.config.selected_microphone_index = device_index

        if device_index is not None:
            print(f"Creating microphone with device index: {device_index}")
            return sr.Microphone(device_index=device_index, sample_rate=16000)
        print("Creating microphone with system default device")
        return sr.Microphone(sample_rate=16000)

    def should_filter_transcription(self, text: str) -> bool:
        """Decide whether a transcription should be suppressed.

        A transcription is suppressed when it:
        1. Is empty after normalization.
        2. Exactly matches an entry in the legacy filter_list.
        3. Matches one of the legacy filter_patterns (e.g. thank-you variants).
        4. Is a bare "thank you" variant with <=5 words.
        """
        normalized = normalize_filter_text(text or "")
        if not normalized:
            return True
        if normalized in self.config.filter_list:
            return True
        for pattern in getattr(self.config, 'filter_patterns', []):
            if pattern.match(normalized):
                return True
        words = normalized.split()
        if normalized.startswith(('thank', 'thanks')) and len(words) <= 5:
            return True
        return False

    def record_callback(self, _, audio: sr.AudioData) -> None:
        """Optimized record callback with minimal processing"""
        if self.recording_event.is_set():
            data = audio.get_raw_data()
            self.data_queue.put(data)
            self.activity_event.set()  # Signal activity
    
    def audio_processor_thread(self):
        """Dedicated thread for processing audio data (batch backends only).

        When the backend streams (Parakeet), audio is consumed by
        ``streaming_worker_thread`` instead, so this loop stands down to avoid
        two consumers racing on ``data_queue``.
        """
        while not self.shutdown_event.is_set():
            try:
                if self.use_streaming:
                    time.sleep(0.05)
                    continue
                # Process audio data from queue
                if self.recording_event.is_set():
                    self.process_audio_queue()
                    self.check_phrase_completion()

                # Reduced sleep to prevent busy waiting but increase responsiveness
                time.sleep(0.005)

            except Exception as e:
                print(f"Audio processor error: {e}")

    def streaming_worker_thread(self):
        """Live transcription for streaming backends (Parakeet).

        Opens a stream when recording starts, feeds mic chunks as they arrive,
        and types the stabilized prefix so text appears while you talk. On a
        short pause it flushes the held-back tail; on stop it drains the last
        audio, flushes, adds a separating space and closes the stream.
        """
        was_recording = False
        last_feed = time.monotonic()
        flushed_tail = False

        while not self.shutdown_event.is_set():
            try:
                if not self.use_streaming:
                    time.sleep(0.1)
                    continue

                recording = self.recording_event.is_set()

                # --- start of a recording session -------------------------- #
                if recording and self.stream is None:
                    try:
                        self.stream = self.backend.open_stream()
                    except Exception as exc:
                        print(f"[Streaming] open failed ({exc}); using batch mode.")
                        self.use_streaming = False
                        continue
                    self.incremental_typer.reset()
                    last_feed = time.monotonic()
                    flushed_tail = False

                # --- feed audio + type the stabilized prefix ---------------- #
                if recording and self.stream is not None:
                    fed = self._feed_stream()
                    if fed:
                        last_feed = time.monotonic()
                        flushed_tail = False
                        self.last_activity_time = datetime.now(timezone.utc)
                        self.incremental_typer.update(self.stream.text)
                    elif (not flushed_tail and
                          time.monotonic() - last_feed > self.config.phrase_timeout):
                        # Natural pause: surface the held-back trailing words.
                        self.incremental_typer.flush_tail(self.stream.text)
                        flushed_tail = True
                    was_recording = True

                # --- end of a recording session ----------------------------- #
                elif was_recording and self.stream is not None:
                    self._feed_stream()  # drain whatever arrived last
                    self.incremental_typer.finalize(self.stream.text)
                    self.stream.close()
                    self.stream = None
                    was_recording = False

                time.sleep(0.02)
            except Exception as exc:
                print(f"[Streaming] worker error: {exc}")
                time.sleep(0.1)

    def _feed_stream(self) -> bool:
        """Push all queued mic bytes into the active stream. Returns True if any."""
        fed = False
        try:
            while True:
                data = self.data_queue.get_nowait()
                audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                if audio_np.size:
                    self.stream.feed(audio_np)
                    fed = True
        except Empty:
            pass
        return fed
    
    def process_audio_queue(self):
        """Process all available audio data in queue"""
        audio_added = False
        try:
            while True:
                data = self.data_queue.get_nowait()
                self.audio_buffer.append(data)
                self.last_chunk_size = len(data)
                audio_added = True
        except Empty:
            pass
        
        if audio_added:
            self.phrase_time = datetime.now(timezone.utc)
            self.last_activity_time = self.phrase_time
    
    def check_phrase_completion(self):
        """Check if a phrase is complete and queue audio for transcription.

        Endpointing strategy
        ────────────────────
        speech_recognition delivers a chunk for one of two reasons, and the
        chunk's length tells us which:

        * SHORTER than ``record_timeout`` → the recognizer heard
          ``pause_threshold`` (0.5 s) of silence and endpointed the phrase
          itself.  No more audio can be in flight, so the phrase can close
          after just ``phrase_timeout`` of quiet.
        * FULL length → the chunk was cut mid-speech by the
          ``phrase_time_limit``; the speaker may still be talking and the next
          delivery can take up to ``record_timeout`` more, so we wait that
          long plus a margin before declaring the phrase done.

        This replaces the old stacked timers — ``max(phrase_timeout,
        record_timeout)`` followed by a fixed 1.2 s trailing wait, ~2.7 s of
        latency on every utterance — with a gate that is usually just
        ``phrase_timeout``.  The last-word-cut-off guarantee is preserved: it
        comes from the recognizer's own pause detection, not from waiting.
        """
        if len(self.audio_buffer) == 0:
            return

        silence = (datetime.now(timezone.utc) - self.phrase_time).total_seconds()

        full_chunk_bytes = self.config.record_timeout * 16000 * 2  # 16-bit mono @ 16 kHz
        if self.last_chunk_size >= full_chunk_bytes * 0.9:
            gate = self.config.record_timeout + 0.4  # cut mid-speech; next chunk may be pending
        else:
            gate = self.config.phrase_timeout        # recognizer already saw the pause

        if silence <= gate:
            return

        # Final short drain: catch audio that raced in during this polling
        # cycle before the buffer snapshot.
        wait_until = datetime.now(timezone.utc) + timedelta(seconds=self.config.trailing_silence)
        while datetime.now(timezone.utc) < wait_until:
            time.sleep(0.05)  # poll every 50 ms
            self.process_audio_queue()
            if (datetime.now(timezone.utc) - self.phrase_time).total_seconds() <= gate:
                return  # new audio arrived — the phrase continues

        self.process_audio_queue()
        audio_bytes = self.audio_buffer.get_and_clear()
        if audio_bytes:
            self.enqueue_transcription(audio_bytes)

    def enqueue_transcription(self, audio_bytes):
        """Queue audio bytes for background transcription, dropping the oldest if needed."""
        if not audio_bytes:
            return
        try:
            self.transcription_queue.put_nowait(audio_bytes)
        except Full:
            try:
                self.transcription_queue.get_nowait()
                self.transcription_queue.task_done()
            except Empty:
                pass
            else:
                print("Transcription queue full, dropping the oldest chunk.")
            try:
                self.transcription_queue.put_nowait(audio_bytes)
            except Full:
                print("Unable to queue audio for transcription; skipping chunk.")




    def transcribe_audio(self, audio_bytes):
        """Optimized transcription with better error handling."""
        if not audio_bytes:
            return ""

        try:
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if audio_np.size == 0:
                return ""

            peak_amplitude = float(np.max(np.abs(audio_np)))
            if peak_amplitude < self.config.volume_threshold:
                return ""

            result = self.backend.transcribe(
                audio_np,
                non_english=self.config.non_english,
                no_speech_threshold=self.config.no_speech_threshold,
                compression_ratio_threshold=1.35 * self.config.threshold_adjustment,
            )

            if not result.text:
                return ""

            # Drop segments Whisper flags as likely non-speech (reduces the
            # classic "thank you" hallucination on silence).
            if result.no_speech_prob >= self.config.no_speech_threshold:
                return ""

            text = result.text
            text = clean_sentence(text)
            text = collapse_repeated_phrases(text)
            # Strip inline filler/stopping words (e.g. "emmm", "uh", "you know")
            filler_pattern = getattr(self.config, '_filler_strip_pattern', None)
            stripped = strip_filler_words(text, filler_pattern)
            if stripped and stripped != text:
                print(f"[Filter] Filler stripped: '{text}' → '{stripped}'")
            text = stripped if stripped else text
            return text

        except Exception as e:
            print(f"Transcription error: {e}")
            return ""
    
    def transcription_worker_thread(self):
        """Background worker that consumes queued audio and runs transcription."""
        while not self.shutdown_event.is_set():
            try:
                audio_bytes = self.transcription_queue.get(timeout=0.1)
            except Empty:
                continue

            try:
                text = self.transcribe_audio(audio_bytes)
                if text:
                    self.process_transcription(text)
            except Exception as e:
                print(f"Transcription worker error: {e}")
            finally:
                self.transcription_queue.task_done()


    def process_transcription(self, text):
        """Process and output transcribed text"""
        if self.should_filter_transcription(text):
            return

        # Check for duplicates against recent transcriptions
        if self.transcription and is_duplicate_or_partial(text, self.transcription[-1]):
            return

        print(f"Outputting: '{text}'")
        self.transcription.append(text)
        self.typer.type(text + " ")
    
    def initialize_voice_converter(self):
        """Create the voice converter instance (does not start streaming)."""
        if not SOUNDDEVICE_AVAILABLE:
            print("[VoiceConverter] sounddevice not installed — voice changer disabled.")
            print("[VoiceConverter] Install with:  pip install sounddevice")
            return

        keywords = self.profile.virtual_cable_keywords
        cables = VoiceConverter.find_virtual_cables(keywords)
        output_dev = cables[0][0] if cables else None

        if output_dev is None:
            print("[VoiceConverter] No virtual audio cable detected.")
            print(self.profile.virtual_cable_setup)

        self.voice_converter = VoiceConverter(
            input_device=self.config.selected_microphone_index,
            output_device=output_dev,
            pitch_semitones=7,
            sample_rate=48000,
            block_size=1024,
            virtual_cable_keywords=keywords,
        )

    def toggle_voice_converter(self):
        """Toggle the real-time voice converter on/off."""
        if self.voice_converter is None:
            print("[VoiceConverter] Not available (sounddevice or virtual cable missing)")
            return
        is_on = self.voice_converter.toggle()
        state = "ON" if is_on else "OFF"
        print(f"[VoiceConverter] {state}")

    def toggle_voice_converter_routing(self):
        """Toggle the voice converter between virtual cable and actual speakers."""
        if self.voice_converter is None:
            print("[VoiceConverter] Not available")
            return
        is_speaker = self.voice_converter.toggle_routing()
        
    def setup_hotkeys(self):
        """Register global hotkeys via the platform-appropriate backend.

        On Windows this uses the ``keyboard`` library; on macOS/Linux it
        uses ``pynput`` (no sudo needed).  Bindings come from the platform
        profile, so the key combos differ per OS.
        """
        self.hotkeys = create_hotkey_manager(self.profile)
        self.hotkeys.set_activity_callback(self._on_hotkey_activity)
        self.hotkeys.register('toggle_recording', self.toggle_recording)
        self.hotkeys.register('toggle_voice_changer', self.toggle_voice_converter)
        self.hotkeys.register('toggle_vc_routing', self.toggle_voice_converter_routing)
        self.hotkeys.start()
        self._check_hotkey_permissions()

    def _check_hotkey_permissions(self):
        """On macOS, verify (and prompt for) the TWO permissions hotkeys need.

        Input Monitoring lets the listener *receive* the hotkey; Accessibility
        lets the typing backend *type* the transcribed text. They are independent — a
        missing Input Monitoring grant makes the hotkey silently do nothing
        even when Accessibility is on — so we check both.
        """
        if self.profile.detected_os != "macos":
            return
        try:
            from utils.hotkeys import (
                macos_accessibility_trusted,
                macos_input_monitoring_trusted,
            )
            input_mon = macos_input_monitoring_trusted(prompt=True)
            accessibility = macos_accessibility_trusted(prompt=True)
        except Exception:
            input_mon = accessibility = None

        if input_mon and accessibility:
            print("[Permissions] Input Monitoring + Accessibility: granted.")
            return

        def mark(state):
            return "granted" if state else ("MISSING" if state is False else "unknown")

        rec = self.profile.hotkey_labels.get('toggle_recording', 'the hotkey')
        bar = "=" * 66
        print("\n" + bar)
        print("  ⚠  GLOBAL HOTKEYS NEED PERMISSION (macOS)")
        print(f"     '{rec}' will do nothing until BOTH of these are enabled for")
        print("     your terminal app in System Settings → Privacy & Security:")
        print(f"        •  Input Monitoring   [{mark(input_mon)}]   ← receives the hotkey")
        print(f"        •  Accessibility      [{mark(accessibility)}]   ← types the text out")
        print("     Enable the missing one(s), then FULLY QUIT the terminal,")
        print("     reopen it, and run again.")
        print(bar + "\n")

    def _on_hotkey_activity(self):
        """Any hotkey press counts as activity (wakes the app from idle)."""
        self.last_activity_time = datetime.now(timezone.utc)

    def toggle_recording(self):
        """Toggle recording state with proper resource management"""
        self.last_activity_time = datetime.now(timezone.utc)

        if not self.ready:
            print("Model still loading — please wait a moment…")
            return

        if self.recording_event.is_set():
            # Stop recording
            self.recording_event.clear()
            if self.background_listener:
                try:
                    self.background_listener(wait_for_stop=False)
                except Exception as e:
                    print(f"Warning: Error stopping background listener: {e}")
                finally:
                    self.background_listener = None

            # Clear the source reference to ensure it's fully released
            self.source = None

            # Batch backends flush the buffer here; streaming backends let
            # streaming_worker_thread drain data_queue and finalize on the
            # recording-stopped edge, so skip the batch path when streaming.
            if not self.use_streaming:
                self.process_audio_queue()
                final_audio = self.audio_buffer.get_and_clear()
                if final_audio:
                    self.enqueue_transcription(final_audio)

            print("Recording stopped...")
            self.update_indicator_safe(False)
            play_cue("stop")

        else:
            # Start recording with a completely fresh microphone source
            self.recording_event.set()
            self.transcription = deque(maxlen=100)
            self._drain_queue(self.data_queue)
            self.audio_buffer.get_and_clear()
            self.last_chunk_size = 0
            self.phrase_time = datetime.now(timezone.utc)

            try:
                # Create a fresh microphone source each time
                self.source = self.create_fresh_microphone_source()

                # Streaming wants small, frequent chunks so text appears while
                # you talk; batch wants larger chunks for a clean phrase.
                phrase_time_limit = (
                    self.config.stream_block if self.use_streaming
                    else self.config.record_timeout
                )
                self.background_listener = self.recorder.listen_in_background(
                    self.source,
                    self.record_callback,
                    phrase_time_limit=phrase_time_limit
                )
                print("Recording started...")
                self.update_indicator_safe(True)
                play_cue("start")
            except Exception as e:
                print(f"Error starting recording: {e}")
                self.recording_event.clear()
                self.source = None
                self._validated_mic = None  # force a full re-resolve next time
                self.update_indicator_safe(False)  # back to 'ready', not stuck
                # Try to reinitialize audio if there's an issue
                try:
                    print("Attempting to reinitialize audio...")
                    self.initialize_audio()
                except Exception as init_error:
                    print(f"Failed to reinitialize audio: {init_error}")
    
    def usage_monitor_thread(self):
        """Poll the meter data and push it to the dot.

        Two sources: `claude -p /usage` (session/weekly limit %) and
        `bunx ccusage daily --json` (today/month spend). Both run off the UI
        thread (network + subprocess, a few seconds each). Re-polls every
        ``usage_refresh`` seconds, or immediately when the meter is clicked
        (``usage_refresh_event``).
        """
        if not (self.show_usage or self.show_cost):
            return
        # Small initial delay so the first poll doesn't compete with model load.
        if self.shutdown_event.wait(timeout=2.0):
            return
        while not self.shutdown_event.is_set():
            # Fetch the two sources CONCURRENTLY so the reliable ccusage cost
            # (~13s) is never gated behind the slower/flakier `claude -p /usage`
            # call — each updates the dot independently as soon as it returns.
            workers = []
            if self.show_usage:
                workers.append(threading.Thread(target=self._poll_usage, daemon=True))
            if self.show_cost:
                workers.append(threading.Thread(target=self._poll_cost, daemon=True))
            for w in workers:
                w.start()
            for w in workers:
                w.join(timeout=95)
            # Wait for the refresh interval, but wake early on a manual refresh.
            self.usage_refresh_event.wait(timeout=max(30, self.config.usage_refresh))
            self.usage_refresh_event.clear()

    def _poll_usage(self):
        """Fetch session/weekly limit % + 5h reset via `claude -p /usage`."""
        try:
            session, week, reset = fetch_usage()
            if session is not None or week is not None or reset is not None:
                self.update_usage_safe(session, week, reset)
            else:
                print("[Usage] /usage returned no panel (CLI print mode is flaky); "
                      "limit bars stay blank this cycle.")
        except Exception as exc:
            print(f"[Usage] limit poll error: {exc}")

    def _poll_cost(self):
        """Fetch today / last-30-day spend via `bunx ccusage daily --json`."""
        try:
            today, month = fetch_ccusage()
            if today is not None or month is not None:
                self.update_cost_safe(today, month)
            else:
                print("[Usage] ccusage returned no data (is `bunx` on PATH?).")
        except Exception as exc:
            print(f"[Usage] cost poll error: {exc}")

    def inactivity_monitor_thread(self):
        """Monitor for inactivity and handle auto-shutdown"""
        while not self.shutdown_event.is_set():
            now = datetime.now(timezone.utc)
            inactive_time = (now - self.last_activity_time).total_seconds()
            
            if inactive_time > self.config.inactivity_timeout:
                print("No activity for 10 minutes, entering idle mode...")
                self.update_indicator_safe(False, idle=True)

                # Stop recording if active
                if self.recording_event.is_set():
                    self.toggle_recording()

                # Wait for activity.  Any registered hotkey press updates
                # last_activity_time via the hotkey activity callback, so we
                # simply watch that timestamp instead of polling a key.
                while (not self.shutdown_event.is_set() and
                       (datetime.now(timezone.utc) - self.last_activity_time).total_seconds()
                       > self.config.inactivity_timeout):
                    time.sleep(0.5)

                if not self.shutdown_event.is_set():
                    print("Activity detected, resuming...")
                    self.update_indicator_safe(False)  # back to 'ready'

            time.sleep(1)  # Check every second

    def update_indicator_safe(self, is_recording=False, idle=False, loading=False):
        """Thread-safe indicator update using queue (resolves to a state name)."""
        if loading:
            state = 'loading'
        elif idle:
            state = 'idle'
        elif is_recording:
            state = 'recording'
        else:
            state = 'ready'
        self.ui_update_queue.put(('indicator', state))
    
    def process_ui_updates(self):
        """Process all pending UI updates from the queue"""
        if not (self.root and self.canvas and self.indicator):
            return  # UI not ready yet
            
        try:
            while True:
                update_type, *args = self.ui_update_queue.get_nowait()
                if update_type == 'indicator':
                    update_indicator(self.canvas, self.indicator, args[0])
                elif update_type == 'usage':
                    session, week, reset = args[0]
                    if hasattr(self.window, 'set_usage'):
                        self.window.set_usage(session, week, reset)
                elif update_type == 'cost':
                    today, month = args[0]
                    if hasattr(self.window, 'set_cost'):
                        self.window.set_cost(today, month)
        except Empty:
            pass  # No more updates to process
        except Exception as e:
            print(f"UI update error: {e}")
    
    
    def toggle_debug_window(self):
        """Toggle the debug window visibility"""
        if not self.debug_ui:
            self.debug_ui = DebugUI(self)
        self.debug_ui.toggle_debug_window()
    
    def _request_shutdown(self):
        """Ask the app to exit (bound to the status window's close button).

        Setting the flag stops the worker threads; the Qt window also quits the
        event loop, after which run() falls through to cleanup().
        """
        self.shutdown_event.set()
        self.usage_refresh_event.set()  # wake the usage poller out of its wait

    def _ui_tick(self):
        """Per-tick work run on the GUI thread (driven by the window's timer).

        Drains queued indicator updates and starts the global hotkeys once the
        model is ready (kept on the main thread, matching the previous design).
        """
        self.process_ui_updates()
        if self.ready and self.hotkeys is None:
            self.setup_hotkeys()

    def _on_window_moved(self, x, y):
        """Persist the dot's new position (debounced by the window)."""
        self.config.window_x = x
        self.config.window_y = y
        self.config.save_to_file()

    def _request_usage_refresh(self):
        """Ask the usage monitor to re-poll now (bound to a usage-panel click)."""
        self.usage_refresh_event.set()

    def update_usage_safe(self, session, week, session_reset=None):
        """Thread-safe push of limit % + 5-hour reset time to the dot."""
        self.ui_update_queue.put(('usage', (session, week, session_reset)))

    def update_cost_safe(self, today, month):
        """Thread-safe push of ccusage spend (USD) to the dot."""
        self.ui_update_queue.put(('cost', (today, month)))

    def create_ui(self):
        """Create the QApplication and the main status + log window."""
        hotkey_label = self.profile.hotkey_labels.get('toggle_recording', 'the hotkey')
        # The meter needs CLIs on PATH; each half degrades away if its CLI is
        # missing (or the whole meter is disabled).
        self.show_usage = self.config.usage_enabled and claude_available()
        self.show_cost = self.config.usage_enabled and bunx_available()
        self.qt_app, self.window, _ = create_overlay_window(
            debug_callback=self.toggle_debug_window,
            hotkey_label=hotkey_label,
            on_close=self._request_shutdown,
            initial_pos=(self.config.window_x, self.config.window_y),
            on_move=self._on_window_moved,
            show_usage=self.show_usage,
            show_cost=self.show_cost,
            usage_click_callback=self._request_usage_refresh,
        )
        # Historical aliases used by the queue-driven update helpers.
        self.root = self.window
        self.canvas = self.window
        self.indicator = self.window
        self.window.set_tick_callback(self._ui_tick)
        self.update_indicator_safe(loading=True)  # amber until the model is ready
    
    def cleanup(self):
        """Release resources. Each step is guarded so cleanup never raises
        (a raised cleanup would otherwise trip the restart loop in main())."""
        print("Cleaning up resources...")
        self.shutdown_event.set()

        if self.hotkeys:
            try:
                self.hotkeys.stop()
            except Exception:
                pass

        if self.voice_converter and self.voice_converter.running:
            try:
                self.voice_converter.stop()
            except Exception:
                pass

        if self.background_listener:
            try:
                self.background_listener(wait_for_stop=False)
            except Exception:
                pass

        # Release CUDA memory only when CUDA was actually in use.
        if self.profile and self.profile.device == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

        # Clean up debug UI
        if self.debug_ui:
            try:
                self.debug_ui.cleanup()
            except Exception:
                pass
            self.debug_ui = None

        # Stop the log pump, restore stdout/stderr, close the window and quit.
        if self.window is not None:
            teardown = getattr(self.window, "teardown", None)
            if teardown is not None:
                try:
                    teardown()
                except Exception:
                    pass
            try:
                self.window.close()
            except Exception:
                pass
            self.window = None

        if self.qt_app is not None:
            try:
                self.qt_app.quit()
            except Exception:
                pass

        self.root = self.canvas = self.indicator = None
    
    def _background_init(self):
        """Load model + audio off the UI thread (keeps the window responsive).

        MLX on a background thread is fine; the earlier crash was a pynput/Tk
        text-input race, which we avoid by starting hotkeys from the main loop
        only after it has been pumping (see run()).
        """
        try:
            self.initialize_model()
            self.use_streaming = getattr(self.backend, "supports_streaming", False)
            self.initialize_audio()
            self.initialize_voice_converter()
            self.ready = True
            self.update_indicator_safe()  # 'loading' -> 'ready'

            if self.use_streaming:
                print("[Streaming] Live transcription enabled (types as you speak).")
            labels = self.profile.hotkey_labels
            print("Voice recognition system ready.")
            print(f"  {labels.get('toggle_recording', '?'):<22} Toggle speech-to-text recording")
            print(f"  {labels.get('toggle_voice_changer', '?'):<22} Toggle voice changer (sharp/funny)")
            print(f"  {labels.get('toggle_vc_routing', '?'):<22} Toggle voice changer output (cable / speaker)")
        except Exception as exc:
            print(f"Initialization failed: {exc}")
            self.update_indicator_safe(idle=True)

    def run(self):
        """Build the Qt UI, start the worker threads, run the Qt event loop."""
        threads = []
        try:
            # Show the window first (Qt paints reliably; no warm-up needed).
            self.create_ui()

            # Load the model + audio in the background so the window stays
            # responsive and the log shows progress while it loads.
            print("Loading model in the background…")
            threading.Thread(target=self._background_init, daemon=True).start()

            # Worker threads idle until recording starts. audio_processor +
            # transcription_worker serve batch backends; streaming_worker
            # serves streaming backends. Each stands down when the active
            # backend isn't its kind, so they never both consume data_queue.
            threads = [
                threading.Thread(target=self.audio_processor_thread, daemon=True),
                threading.Thread(target=self.transcription_worker_thread, daemon=True),
                threading.Thread(target=self.streaming_worker_thread, daemon=True),
                threading.Thread(target=self.usage_monitor_thread, daemon=True),
                threading.Thread(target=self.inactivity_monitor_thread, daemon=True)
            ]
            for thread in threads:
                thread.start()

            # Re-assert the hard-exit handler so Ctrl+C always quits. The
            # window's 40ms timer keeps Python servicing signals during exec().
            try:
                signal.signal(signal.SIGINT, _force_quit)
            except (ValueError, OSError):
                pass

            # Qt event loop on the main thread. Indicator updates and the
            # deferred hotkey start run via the window's timer (_ui_tick).
            # Returns when the window is closed (or QApplication.quit()).
            self.qt_app.exec()

            # Window closed → tell the workers to stop and wait briefly.
            self.shutdown_event.set()
            for thread in threads:
                if thread.is_alive():
                    thread.join(timeout=1.0)

        except KeyboardInterrupt:
            print("\nKeyboard interrupt detected. Exiting...")
        except Exception as e:
            print(f"Application error: {e}")
        finally:
            self.cleanup()

def main():
    """Optimized main function"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        help="Whisper model size. Defaults to the per-platform "
                             "recommendation (small on Apple GPU, medium on CUDA, base on CPU).",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--platform", default=None,
                        choices=["auto", "windows", "macos", "linux"],
                        help="Pin the platform profile. Default: auto-detect the OS.")
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the English model (if set, tries to auto-detect).")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=1.5,
                        help="Length of audio buffer in seconds for each chunk.", type=float)
    parser.add_argument("--phrase_timeout", default=0.8,
                        help="Silence gap (seconds) after a pause-ended chunk to consider "
                             "a phrase done.", type=float)
    parser.add_argument("--volume_threshold", default=0.008,
                        help="Min volume level to consider valid audio in the buffer.", type=float)
    parser.add_argument("--trailing_silence", default=0.25,
                        help="Final drain window (seconds) to catch audio that races in "
                             "while a phrase is being finalized.",
                        type=float)
    parser.add_argument("--threshold_adjustment", default=1.0,
                        help="Adjust the model's threshold for detecting repetitions (1.0-2.0). Higher values are more aggressive in preventing loops.", type=float)
    parser.add_argument("--no_speech_threshold", default=0.6, type=float,
                        help="Drop transcriptions when Whisper reports a higher no-speech probability.")
    parser.add_argument("--stream_block", default=0.5, type=float,
                        help="Streaming backends (Parakeet): mic capture block in seconds. "
                             "Smaller feels snappier but costs more per-chunk overhead.")
    parser.add_argument("--no_usage", action='store_true',
                        help="Hide the Claude Code usage meter below the dot.")
    parser.add_argument("--usage_refresh", default=300, type=int,
                        help="Seconds between usage-meter polls (default 300 = 5 min); "
                             "click the meter to refresh on demand.")
    
    args = parser.parse_args()
    config = VoiceConfig(args)

    # Make Ctrl+C / SIGTERM reliably quit (Tk + pynput can swallow signals).
    signal.signal(signal.SIGINT, _force_quit)
    try:
        signal.signal(signal.SIGTERM, _force_quit)
    except (ValueError, OSError):
        pass

    # Simple restart loop for error recovery
    while True:
        try:
            app = VoiceRecognitionApp(config)
            app.run()
            break  # Normal exit
        except Exception as e:
            print(f"Application crashed: {e}")
            print("Restarting in 2 seconds...")
            time.sleep(2)

if __name__ == "__main__":
    main()
