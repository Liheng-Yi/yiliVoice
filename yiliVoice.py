import argparse
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch
import pyautogui
import keyboard
import tkinter as tk
from tkinter import ttk
import threading
import time
import re  # NEW: For regex utilities

from datetime import datetime, timedelta, timezone
from queue import Queue
from time import sleep
from sys import platform

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

def create_overlay_window():
    """
    Creates a small top-centered overlay circle:
      - Green when recording is active
      - Pink when ready
      - Red when no activity for a while (auto-shutdown mode)
    """
    root = tk.Tk()
    root.title("Voice Status")
    root.attributes('-topmost', True)
    root.overrideredirect(True)
    
    canvas = tk.Canvas(root, width=20, height=20, bg='black', highlightthickness=0)
    canvas.pack()
    
    # Create the indicator circle
    indicator = canvas.create_oval(5, 5, 15, 15, fill='pink')
    
    # Position the window at the top center of the screen
    screen_width = root.winfo_screenwidth()
    root.geometry(f'20x20+{(screen_width//2)-10}+0')
    
    return root, canvas, indicator

def update_indicator(canvas, indicator, is_recording, idle=False):
    if idle:
        color = 'red'  # Complete shutdown
    else:
        color = 'green' if is_recording else 'pink'  # Green for recording, pink for ready
    canvas.itemconfig(indicator, fill=color)

def loading_animation():
    chars = "|/-\\"
    i = 0
    while not loading_complete:
        print(f"\rLoading model {chars[i]}", end="")
        i = (i + 1) % len(chars)
        time.sleep(0.1)

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

def main():
    global loading_complete
    loading_complete = False

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the English model (if set, tries to auto-detect).")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=1.5,
                        help="Length of audio buffer in seconds for each chunk.", type=float)
    parser.add_argument("--phrase_timeout", default=3.0,
                        help="Silence gap (seconds) to consider a phrase ended.", type=float)
    parser.add_argument("--volume_threshold", default=0.008,
                        help="Min volume level to consider valid audio in the buffer.", type=float)
    parser.add_argument("--trailing_silence", default=0.3, 
                        help="Extra silence to capture at the end of phrases (seconds).", type=float)
    parser.add_argument("--threshold_adjustment", default=1.0,
                        help="Adjust the model's threshold for detecting repetitions (1.0-2.0). Higher values are more aggressive in preventing loops.", type=float)
    args = parser.parse_args()

    while True:
        try:
            # Create overlay window
            root, canvas, indicator = create_overlay_window()
            update_indicator(canvas, indicator, False)  # Start with pink (not recording)
            
            # record_callback gets defined inside main() to access data_queue
            def record_callback(_, audio: sr.AudioData) -> None:
                data = audio.get_raw_data()
                data_queue.put(data)
            
            # Check if GPU is available
            if torch.cuda.is_available():
                device = "cuda"
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = "cpu"
                print("CUDA not available, using CPU")
            
            # Start loading animation in a separate thread
            loading_thread = threading.Thread(target=loading_animation)
            loading_thread.start()
            
            # Load the Whisper model
            print("\nLoading Whisper model...")
            # If you want to use the English model only:
            #   model_path = whisper.load_model("medium.en", device=device, download_root=None)
            # If you want to automatically detect language or not restricted to English:
            #   model_path = whisper.load_model(args.model, device=device, download_root=None)
            # For demonstration, we keep the default "medium.en" below for English:
            model_path = whisper.load_model("medium.en", device=device, download_root=None)
            audio_model = model_path.to(device)
            
            loading_complete = True
            loading_thread.join()
            print(f"\nModel loaded on {device}")
            
            # Set up mic recorder
            recorder = sr.Recognizer()
            recorder.energy_threshold = args.energy_threshold
            recorder.dynamic_energy_threshold = False
            # Reduce pause threshold for faster response
            recorder.pause_threshold = 0.5  
            source = sr.Microphone(sample_rate=16000)

            with source:
                recorder.adjust_for_ambient_noise(source)

            data_queue = Queue()
            recording_active = False
            background_listener = None

            # For tracking time
            phrase_time = datetime.now(timezone.utc)
            last_activity_time = datetime.now(timezone.utc)
            last_transcription_time = datetime.now(timezone.utc)
            
            record_timeout = args.record_timeout
            phrase_timeout = args.phrase_timeout
            volume_threshold = args.volume_threshold * 32768.0
            trailing_silence = args.trailing_silence

            # We'll store entire speech for each phrase into this buffer:
            current_audio_buffer = bytes()

            # Keep a global transcription log
            transcription = []

            # Words/Phrases you want to filter out if recognized
            filterList = [
                "i'm sorry",
                "thanks for watching!",
                "i'll see you next time.",
                "i'm not gonna lie.",
                "thank you.",
                "i'm going to go get some food.",
                "i'm going to do it again.",
                "bye."
            ]
            filterList = list({phrase.lower() for phrase in filterList})

            def transcribe_buffered_audio(audio_bytes):
                """
                Actually call Whisper on the entire buffered audio_bytes.
                Return the cleaned text result (or empty if nothing recognized).
                """
                if not audio_bytes:
                    return ""

                audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Example of extra parameters that may improve final accuracy:
                result = audio_model.transcribe(
                    audio_np,
                    fp16=(device == "cuda"),   # Use half precision on GPU
                    language="en" if not args.non_english else None,
                    temperature=0.0,         # Force deterministic decoding
                    best_of=1,               # Use the highest probability result
                    beam_size=5,             # Consider more transcription paths
                    compression_ratio_threshold=1.35 * args.threshold_adjustment,  # Prevent looping/repetitions
                    condition_on_previous_text=False,  # Don't condition on prior outputs
                )
                
                text = result['text'].strip()
                text = clean_sentence(text)
                text = collapse_repeated_phrases(text)  # NEW: Deduplicate spam within chunk
                return text

            while True:
                try:
                    root.update()
                    now = datetime.now(timezone.utc)
                    
                    # Check for 2-minute (1000-second) inactivity to shut down
                    if (now - last_activity_time) > timedelta(seconds=1000):
                        print("No activity for 100 seconds, shutting down...")
                        update_indicator(canvas, indicator, False, idle=True)  # Set to red
                        root.update()
                        
                        # Stop recording if active
                        if background_listener:
                            background_listener(wait_for_stop=False)
                            background_listener = None
                        
                        # Clear CUDA cache
                        if device == "cuda":
                            torch.cuda.empty_cache()
                        
                        # Wait for Ctrl+F8 to restart
                        while True:
                            root.update()
                            if keyboard.is_pressed('ctrl+f8'):
                                root.destroy()
                                break
                            sleep(0.1)
                        break  # Break inner loop to restart main()
                    
                    # Check for Ctrl+F8 toggle (start/stop recording)
                    if keyboard.is_pressed('ctrl+f8'):
                        recording_active = not recording_active
                        update_indicator(canvas, indicator, recording_active, idle=False)
                        last_activity_time = datetime.now(timezone.utc)
                        
                        if recording_active:
                            print("Recording started...")
                            if audio_model is None:
                                print("Reloading Whisper model...")
                                audio_model = model_path.to(device)
                            
                            transcription = []
                            current_audio_buffer = bytes()
                            data_queue.queue.clear()
                            phrase_time = now
                            last_transcription_time = now
                            background_listener = recorder.listen_in_background(
                                source,
                                record_callback,
                                phrase_time_limit=record_timeout
                            )
                        else:
                            print("Recording stopped...")
                            if background_listener:
                                background_listener(wait_for_stop=False)
                                background_listener = None
                            
                            # Flush whatever is in the buffer as a final chunk
                            final_text = transcribe_buffered_audio(current_audio_buffer)
                            current_audio_buffer = bytes()

                            # Also output any sentence fragments in the buffer when recording stops
                            if final_text:
                                if final_text.lower() not in filterList and (not transcription or not is_duplicate_or_partial(final_text, transcription[-1])):
                                    transcription.append(final_text)
                                    print(f"Final output: {final_text}")
                                    pyautogui.write(final_text, interval=0.01)

                            print("\nSession Transcription:")
                            for line in transcription:
                                print(line)
                            print("\n")
                        sleep(0.2)
                        continue

                    # If we are actively recording, handle incoming audio data
                    if recording_active:
                        # If there is new data in the queue, append it to current buffer
                        if not data_queue.empty():
                            last_activity_time = now
                            while not data_queue.empty():
                                current_audio_buffer += data_queue.get()
                            
                            # Each time new data arrives, we reset phrase_time to 'now'
                            phrase_time = now
                        
                        # Check if user has been silent for longer than phrase_timeout
                        # i.e. "phrase_complete" detection
                        silent_duration = now - phrase_time
                        if silent_duration > timedelta(seconds=phrase_timeout) and len(current_audio_buffer) > 0:
                            # We have a completed speech chunk. Transcribe it.
                            
                            # Reduced wait time for trailing silence
                            sleep(trailing_silence)
                            
                            # Get any remaining audio that might have come in during our wait
                            while not data_queue.empty():
                                current_audio_buffer += data_queue.get()
                                
                            text = transcribe_buffered_audio(current_audio_buffer)
                            current_audio_buffer = bytes()  # Reset buffer

                            # Skip empty or filtered text
                            if not text or text.lower() in filterList:
                                print("Hallucination or filtered text detected:", text.lower() if text else "")
                                continue
                                
                            # Skip duplicates
                            if transcription and is_duplicate_or_partial(text, transcription[-1]):
                                continue
                                
                            # Immediately type out the text
                            print(f"Outputting: '{text}'")
                            transcription.append(text)
                            pyautogui.write(text, interval=0.01)
                            last_transcription_time = now

                except Exception as e:
                    print(f"Error in inner loop: {str(e)}")
                    root.destroy()
                    if background_listener:
                        background_listener(wait_for_stop=False)
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    break  # Break inner loop to restart main()

        except KeyboardInterrupt:
            print("\nKeyboard interrupt detected. Exiting...")
            root.destroy()
            if background_listener:
                background_listener(wait_for_stop=False)
            if device == "cuda":
                torch.cuda.empty_cache()
            return  # Exit the program completely
        
        except Exception as e:
            print(f"Error in outer loop: {str(e)}")
            if 'root' in locals():
                root.destroy()
            if 'background_listener' in locals() and background_listener:
                background_listener(wait_for_stop=False)
            if 'device' in locals() and device == "cuda":
                torch.cuda.empty_cache()
            sleep(1)  # Wait briefly before restarting main()

if __name__ == "__main__":
    main()
