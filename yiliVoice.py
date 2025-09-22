import argparse
import numpy as np
import speech_recognition as sr
import whisper
import torch
import pyautogui
import keyboard
import threading
import time
from collections import deque

from datetime import datetime, timedelta, timezone
from queue import Queue, Empty, Full
from time import sleep
from sys import platform

# Import our modular components
from settings import (
    VoiceConfig, DebugUI,
    create_overlay_window, update_indicator
)
from utils import (
    AudioBuffer, clean_sentence, normalize_filter_text, 
    is_duplicate_or_partial, collapse_repeated_phrases
)




class VoiceRecognitionApp:
    def __init__(self, config):
        self.config = config
        self.audio_model = None
        self.device = None
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
        self.last_shortkey_time = datetime.now(timezone.utc) - timedelta(seconds=1)  # Initialize to allow immediate first press
        
        # UI
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
        """Initialize the Whisper model with progress indication"""
        # Check if GPU is available
        if torch.cuda.is_available():
            self.device = "cuda"
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            print("CUDA not available, using CPU")
        
        print("Loading Whisper model...")
        # Use the model specified in config
        model_name = f"{self.config.model}.en" if not self.config.non_english else self.config.model
        self.audio_model = whisper.load_model(model_name, device=self.device)
        print(f"Model loaded on {self.device}")
    
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
    
    def create_fresh_microphone_source(self):
        """Create a fresh microphone source to avoid context manager conflicts"""
        device_index = self.config.selected_microphone_index
        if device_index is not None:
            print(f"Creating microphone with selected device index: {device_index}")
            return sr.Microphone(device_index=device_index, sample_rate=16000)
        else:
            print("Creating microphone with system default device")
            return sr.Microphone(sample_rate=16000)

    def should_filter_transcription(self, text: str) -> bool:
        """Decide whether a transcription should be suppressed."""
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
        """Check if a phrase is complete and queue audio for transcription"""
        now = datetime.now(timezone.utc)
        silent_duration = now - self.phrase_time

        if (silent_duration > timedelta(seconds=self.config.phrase_timeout) and 
            len(self.audio_buffer) > 0):

            time.sleep(self.config.trailing_silence)

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

            result = self.audio_model.transcribe(
                audio_np,
                fp16=(self.device == "cuda"),
                language="en" if not self.config.non_english else None,
                temperature=0.0,
                best_of=1,
                beam_size=5,
                compression_ratio_threshold=1.35 * self.config.threshold_adjustment,
                condition_on_previous_text=False,
            )

            segments = result.get('segments') or []
            first_segment = segments[0] if segments else None
            no_speech_prob = None
            if first_segment is not None:
                no_speech_prob = first_segment.get('no_speech_prob')
            else:
                no_speech_prob = result.get('no_speech_prob')
            if isinstance(no_speech_prob, (int, float)) and no_speech_prob >= self.config.no_speech_threshold:
                return ""

            text = result['text'].strip()
            text = clean_sentence(text)
            text = collapse_repeated_phrases(text)
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
    
    def keyboard_handler_thread(self):
        """Dedicated thread for handling keyboard input"""
        while not self.shutdown_event.is_set():
            try:
                if keyboard.is_pressed('ctrl+f8'):
                    # Check cooldown period - must wait 0.5 seconds between presses
                    now = datetime.now(timezone.utc)
                    time_since_last_press = (now - self.last_shortkey_time).total_seconds()
                    
                    if time_since_last_press >= 0.5:
                        self.last_shortkey_time = now
                        self.toggle_recording()
                        time.sleep(0.2)  # Wait for key release to prevent multiple detections
                    
                time.sleep(0.02)  # Reduced polling interval for faster response
            except Exception as e:
                print(f"Keyboard handler error: {e}")
    
    def toggle_recording(self):
        """Toggle recording state with proper resource management"""
        self.last_activity_time = datetime.now(timezone.utc)

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
            except Exception as e:
                print(f"Error starting recording: {e}")
                self.recording_event.clear()
                self.source = None
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
                
                # Optional: Unload model to save memory (uncomment if you want "rebooting" behavior)
                # print("Unloading model to save memory...")
                # self.audio_model = None
                # if self.device == "cuda":
                #     torch.cuda.empty_cache()
                
                # Wait for activity
                while inactive_time > self.config.inactivity_timeout:
                    if keyboard.is_pressed('ctrl+f8'):
                        self.last_activity_time = datetime.now(timezone.utc)
                        
                        # Optional: Reload model if it was unloaded (uncomment if using unloading above)
                        # if self.audio_model is None:
                        #     print("Reloading model...")
                        #     self.initialize_model()
                        
                        break
                    time.sleep(0.5)
                    now = datetime.now(timezone.utc)
                    inactive_time = (now - self.last_activity_time).total_seconds()
            
            time.sleep(1)  # Check every second

    def update_indicator_safe(self, is_recording, idle=False):
        """Thread-safe indicator update using queue"""
        self.ui_update_queue.put(('indicator', is_recording, idle))
    
    def process_ui_updates(self):
        """Process all pending UI updates from the queue"""
        if not (self.root and self.canvas and self.indicator):
            return  # UI not ready yet
            
        try:
            while True:
                update_type, *args = self.ui_update_queue.get_nowait()
                if update_type == 'indicator':
                    is_recording, idle = args
                    update_indicator(self.canvas, self.indicator, is_recording, idle)
        except Empty:
            pass  # No more updates to process
        except Exception as e:
            print(f"UI update error: {e}")
    
    
    def toggle_debug_window(self):
        """Toggle the debug window visibility"""
        if not self.debug_ui:
            self.debug_ui = DebugUI(self)
        self.debug_ui.toggle_debug_window()
    
    def create_ui(self):
        """Create the UI overlay"""
        self.root, self.canvas, self.indicator = create_overlay_window(debug_callback=self.toggle_debug_window)
        self.update_indicator_safe(False)
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up resources...")
        self.shutdown_event.set()
        
        if self.background_listener:
            self.background_listener(wait_for_stop=False)
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        # Clean up debug UI
        if self.debug_ui:
            self.debug_ui.cleanup()
            self.debug_ui = None
        
        if self.root:
            self.root.destroy()
    
    def run(self):
        """Main application loop with threading"""
        try:
            # Initialize components
            self.initialize_model()
            self.initialize_audio()
            self.create_ui()
            
            # Start worker threads
            threads = [
                threading.Thread(target=self.audio_processor_thread, daemon=True),
                threading.Thread(target=self.transcription_worker_thread, daemon=True),
                threading.Thread(target=self.keyboard_handler_thread, daemon=True),
                threading.Thread(target=self.inactivity_monitor_thread, daemon=True)
            ]
            
            for thread in threads:
                thread.start()
            
            print("Voice recognition system ready. Press Ctrl+F8 to toggle recording.")
            
            # Main UI loop - much more efficient than before
            while not self.shutdown_event.is_set():
                try:
                    # Process any pending UI updates from worker threads
                    self.process_ui_updates()
                    
                    # Update the tkinter UI
                    self.root.update()
                    time.sleep(0.005)  # Reduced sleep for better UI responsiveness
                except tk.TclError:
                    break  # Window was closed
                except Exception as e:
                    print(f"UI loop error: {e}")
                    break
            
            # Wait for threads to finish
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
    parser.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
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
    parser.add_argument("--trailing_silence", default=0.8, 
                        help="Extra silence to capture at the end of phrases (seconds).", type=float)
    parser.add_argument("--threshold_adjustment", default=1.0,
                        help="Adjust the model's threshold for detecting repetitions (1.0-2.0). Higher values are more aggressive in preventing loops.", type=float)
    parser.add_argument("--no_speech_threshold", default=0.6, type=float,
                        help="Drop transcriptions when Whisper reports a higher no-speech probability.")
    
    args = parser.parse_args()
    config = VoiceConfig(args)
    
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
