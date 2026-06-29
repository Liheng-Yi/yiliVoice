import argparse
import os
import signal
import numpy as np
import speech_recognition as sr
import pyautogui
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
    
    @staticmethod
    def _resolve_input_device_index(requested):
        """Return a usable input-device index (or None for the system default).

        A stale/invalid index — e.g. a saved ``0`` that points at an output
        device with no input channels — would make ``sr.Microphone`` fail to
        open. speech_recognition *swallows* that failure (it leaves
        ``source.stream = None`` and returns normally), so the background
        listener later crashes with a cryptic ``NoneType`` error. We avoid that
        by falling back to the default input device, then to the first device
        that actually has input channels.
        """
        try:
            import pyaudio
        except Exception:
            return requested
        pa = pyaudio.PyAudio()
        try:
            count = pa.get_device_count()

            def is_input(i):
                try:
                    return pa.get_device_info_by_index(i).get("maxInputChannels", 0) > 0
                except Exception:
                    return False

            if requested is not None and 0 <= requested < count and is_input(requested):
                return requested

            try:
                default_idx = pa.get_default_input_device_info().get("index")
            except Exception:
                default_idx = None
            if default_idx is not None and is_input(default_idx):
                print(f"Microphone index {requested} is not an input device; "
                      f"falling back to default input [{default_idx}].")
                return default_idx

            for i in range(count):
                if is_input(i):
                    print(f"Microphone index {requested} unusable; using input [{i}].")
                    return i

            print("No input devices with capture channels were found.")
            return None
        finally:
            pa.terminate()

    @staticmethod
    def _microphone_opens(device_index):
        """Best-effort test that an input stream can actually be opened.

        Returns ``(ok, error_message)``. Catches the real failure (bad device
        or, on macOS, a missing Microphone permission) that SR would otherwise
        hide."""
        try:
            import pyaudio
        except Exception as exc:
            return True, None  # can't test; let the normal path try
        pa = pyaudio.PyAudio()
        try:
            kwargs = dict(format=pyaudio.paInt16, channels=1, rate=16000,
                          input=True, frames_per_buffer=1024)
            if device_index is not None:
                kwargs["input_device_index"] = device_index
            stream = pa.open(**kwargs)
            stream.close()
            return True, None
        except Exception as exc:
            return False, str(exc)
        finally:
            pa.terminate()

    def create_fresh_microphone_source(self):
        """Create a fresh, validated microphone source.

        Resolves a usable input-device index and pre-flights the open so a bad
        device or a missing macOS Microphone permission raises here (caught by
        toggle_recording) instead of crashing the background listener thread.
        """
        device_index = self._resolve_input_device_index(self.config.selected_microphone_index)
        # Remember the resolved choice so the UI + saved config reflect reality.
        self.config.selected_microphone_index = device_index

        ok, err = self._microphone_opens(device_index)
        if not ok:
            raise OSError(
                f"could not open microphone (device {device_index}): {err}. "
                "On macOS, grant Microphone access in "
                "System Settings → Privacy & Security → Microphone."
            )

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
        """Dedicated thread for processing audio data"""
        while not self.shutdown_event.is_set():
            try:
                # Process audio data from queue
                if self.recording_event.is_set():
                    self.process_audio_queue()
                    self.check_phrase_completion()
                
                # Reduced sleep to prevent busy waiting but increase responsiveness
                time.sleep(0.005)
                
            except Exception as e:
                print(f"Audio processor error: {e}")
    
    def process_audio_queue(self):
        """Process all available audio data in queue"""
        audio_added = False
        try:
            while True:
                data = self.data_queue.get_nowait()
                self.audio_buffer.append(data)
                audio_added = True
        except Empty:
            pass
        
        if audio_added:
            self.phrase_time = datetime.now(timezone.utc)
            self.last_activity_time = self.phrase_time
    
    def check_phrase_completion(self):
        """Check if a phrase is complete and queue audio for transcription.

        Fix: last word/digit cut-off
        ────────────────────────────
        Root causes addressed here:

        F2 – Race condition: the extension condition was `phrase_time > last_audio_time`
             which is False when the last chunk arrives at the exact moment the
             phrase_timeout fires.  Changed to `>=` so any tie also extends the window.

        F3 – phrase_timeout < record_timeout: audio chunks can be up to `record_timeout`
             seconds long.  We use `max(phrase_timeout, record_timeout)` so we never
             declare a phrase done before a full chunk could have arrived.

        F4 – Final drain happens BEFORE the buffer snapshot so any audio that arrived
             during the very last polling cycle is captured.
        """
        now = datetime.now(timezone.utc)
        silent_duration = now - self.phrase_time

        # F3: effective timeout must be at least one full audio-chunk long
        effective_phrase_timeout = max(
            self.config.phrase_timeout,
            self.config.record_timeout
        )

        if (silent_duration > timedelta(seconds=effective_phrase_timeout) and
                len(self.audio_buffer) > 0):

            # Wait for trailing silence, checking for new audio periodically.
            # This ensures we capture late-arriving audio chunks (e.g. the last word).
            last_audio_time = self.phrase_time
            wait_until = datetime.now(timezone.utc) + timedelta(seconds=self.config.trailing_silence)

            while datetime.now(timezone.utc) < wait_until:
                time.sleep(0.05)  # poll every 50 ms
                self.process_audio_queue()

                # F2: use >= so a chunk that arrived exactly at the timeout still
                #     extends the capture window.
                if self.phrase_time >= last_audio_time and self.phrase_time > (
                        now - timedelta(seconds=effective_phrase_timeout)):
                    if self.phrase_time > last_audio_time:  # genuinely new audio
                        last_audio_time = self.phrase_time
                        wait_until = datetime.now(timezone.utc) + timedelta(
                            seconds=self.config.trailing_silence)

            # F4: final unconditional drain BEFORE taking the buffer snapshot
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
        pyautogui.write(text + " ", interval=0.01)
    
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
        lets pyautogui *type* the transcribed text. They are independent — a
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

            # Flush any remaining audio and queue it for transcription
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
            self.phrase_time = datetime.now(timezone.utc)

            try:
                # Create a fresh microphone source each time
                self.source = self.create_fresh_microphone_source()

                self.background_listener = self.recorder.listen_in_background(
                    self.source,
                    self.record_callback,
                    phrase_time_limit=self.config.record_timeout
                )
                print("Recording started...")
                self.update_indicator_safe(True)
                play_cue("start")
            except Exception as e:
                print(f"Error starting recording: {e}")
                self.recording_event.clear()
                self.source = None
                self.update_indicator_safe(False)  # back to 'ready', not stuck
                # Try to reinitialize audio if there's an issue
                try:
                    print("Attempting to reinitialize audio...")
                    self.initialize_audio()
                except Exception as init_error:
                    print(f"Failed to reinitialize audio: {init_error}")
    
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

    def _ui_tick(self):
        """Per-tick work run on the GUI thread (driven by the window's timer).

        Drains queued indicator updates and starts the global hotkeys once the
        model is ready (kept on the main thread, matching the previous design).
        """
        self.process_ui_updates()
        if self.ready and self.hotkeys is None:
            self.setup_hotkeys()

    def create_ui(self):
        """Create the QApplication and the main status + log window."""
        hotkey_label = self.profile.hotkey_labels.get('toggle_recording', 'the hotkey')
        self.qt_app, self.window, _ = create_overlay_window(
            debug_callback=self.toggle_debug_window,
            hotkey_label=hotkey_label,
            on_close=self._request_shutdown,
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
            self.initialize_audio()
            self.initialize_voice_converter()
            self.ready = True
            self.update_indicator_safe()  # 'loading' -> 'ready'

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

            # Worker threads idle until recording starts.
            threads = [
                threading.Thread(target=self.audio_processor_thread, daemon=True),
                threading.Thread(target=self.transcription_worker_thread, daemon=True),
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
    parser.add_argument("--phrase_timeout", default=1.0,
                        help="Silence gap (seconds) to consider a phrase ended.", type=float)
    parser.add_argument("--volume_threshold", default=0.008,
                        help="Min volume level to consider valid audio in the buffer.", type=float)
    parser.add_argument("--trailing_silence", default=1.2,
                        help="Extra silence to capture at the end of phrases (seconds). "
                             "Should be >= record_timeout to avoid cutting off the last word.",
                        type=float)
    parser.add_argument("--threshold_adjustment", default=1.0,
                        help="Adjust the model's threshold for detecting repetitions (1.0-2.0). Higher values are more aggressive in preventing loops.", type=float)
    parser.add_argument("--no_speech_threshold", default=0.6, type=float,
                        help="Drop transcriptions when Whisper reports a higher no-speech probability.")
    
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
