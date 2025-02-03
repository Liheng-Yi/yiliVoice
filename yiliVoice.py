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

from datetime import datetime, timedelta, timezone
from queue import Queue
from time import sleep
from sys import platform

def clean_sentence(text):
    # Remove period at the end of the sentence
    text = text.strip()
    if text.endswith('.'):
        text = text[:-1]
    # Capitalize the first letter
    text = text.capitalize()
    return text

def is_duplicate_or_partial(new_text, previous_text):
    """Check if new_text is a duplicate or partial repeat of previous_text."""
    if not previous_text:
        return False
    new_text = new_text.lower()
    previous_text = previous_text.lower()
    
    # Check if one is contained within the other
    if new_text in previous_text or previous_text in new_text:
        return True
    
    # Check for significant overlap
    words_new = new_text.split()
    words_prev = previous_text.split()
    
    # If the new text is too short, it might be a fragment
    if len(words_new) < 3:
        return True
    
    # Check for overlapping consecutive words
    for i in range(len(words_prev) - 2):
        three_words = ' '.join(words_prev[i:i+3])
        if three_words in new_text:
            return True
            
    return False

def create_overlay_window():
    root = tk.Tk()
    root.title("Voice Status")
    
    # Make window stay on top
    root.attributes('-topmost', True)
    
    # Remove window decorations
    root.overrideredirect(True)
    
    # Create a small circular indicator
    canvas = tk.Canvas(root, width=20, height=20, bg='black', highlightthickness=0)
    canvas.pack()
    
    # Create the indicator circle - initialize as pink (ready state)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="Length of audio buffer in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    parser.add_argument("--volume_threshold", default=0.01,
                        help="Minimum volume level to consider valid audio.", type=float)
    args = parser.parse_args()

    while True:
        try:
            # Create overlay window
            root, canvas, indicator = create_overlay_window()
            update_indicator(canvas, indicator, False)  # Start with pink
            
            # Define record_callback function inside main to access data_queue
            def record_callback(_, audio: sr.AudioData) -> None:
                """
                Callback function that receives audio data when recording
                """
                data = audio.get_raw_data()
                data_queue.put(data)
            
            # Initialize everything else
            if torch.cuda.is_available():
                device = "cuda"
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = "cpu"
                print("CUDA not available, using CPU")
            
            # Initialize audio processing
            recorder = sr.Recognizer()
            recorder.energy_threshold = args.energy_threshold
            recorder.dynamic_energy_threshold = False
            source = sr.Microphone(sample_rate=16000)
            
            # Load model
            print("Loading Whisper model...")
            model_path = whisper.load_model("medium.en", device=device, download_root=None)
            audio_model = model_path.to(device)
            print(f"Model loaded on {device}")
            
            record_timeout = args.record_timeout
            phrase_timeout = args.phrase_timeout
            volume_threshold = args.volume_threshold * 32768.0
            transcription = []
            
            with source:
                recorder.adjust_for_ambient_noise(source)

            data_queue = Queue()
            recording_active = False
            background_listener = None
            phrase_time = datetime.now(timezone.utc)
            last_activity_time = datetime.now(timezone.utc)
            last_transcription_time = datetime.now(timezone.utc)
            
            # Add filterList definition
            filterList = [
                "i'm sorry",
                "thanks for watching!",
                "i'll see you next time.",
                "i'm not gonna lie.",
                "thank you.",
                "i'm going to go get some food.",
                "bye."
            ]
            filterList = list({phrase.lower() for phrase in filterList})

            while True:
                try:
                    root.update()
                    now = datetime.now(timezone.utc)
                    
                    # Check for 2-minute inactivity shutdown
                    if (now - last_activity_time) > timedelta(seconds=100):
                        print("No activity for 100 seconds, shutting down...")
                        update_indicator(canvas, indicator, False, idle=True)  # Set to red
                        root.update()
                        
                        # Stop recording if active
                        if background_listener:
                            background_listener(wait_for_stop=False)
                            background_listener = None
                        
                        # Clear CUDA cache if using GPU
                        if device == "cuda":
                            torch.cuda.empty_cache()
                        
                        # Wait for Ctrl+Q to restart
                        while True:
                            root.update()
                            if keyboard.is_pressed('ctrl+q'):
                                root.destroy()
                                break
                            sleep(0.1)
                        break  # Break inner loop to restart main
                    
                    # Normal Ctrl+Q toggle during active operation
                    if keyboard.is_pressed('ctrl+q'):
                        recording_active = not recording_active
                        update_indicator(canvas, indicator, recording_active, idle=False)
                        last_activity_time = datetime.now(timezone.utc)  # Reset timer on toggle
                        
                        if recording_active:
                            print("Recording started...")
                            if audio_model is None:
                                print("Reloading Whisper model...")
                                audio_model = model_path.to(device)  # Reload the model
                            transcription = []
                            data_queue.queue.clear()
                            phrase_time = datetime.now(timezone.utc)
                            last_activity_time = datetime.now(timezone.utc)
                            last_transcription_time = datetime.now(timezone.utc)  # Reset transcription timer
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
                            print("\nSession Transcription:")
                            for line in transcription:
                                print(line)
                            print("\n")
                        sleep(0.5)
                        continue

                    if not data_queue.empty() and recording_active:
                        last_activity_time = now  # Update activity timer when audio is processed
                        phrase_complete = False
                        if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                            phrase_complete = True
                        phrase_time = now

                        audio_data = bytes()
                        while not data_queue.empty():
                            audio_data += data_queue.get()
                        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                        # Add device specification for transcription
                        result = audio_model.transcribe(
                            audio_np, 
                            fp16=(device == "cuda"),  # Use fp16 only when on GPU
                            language="en" if not args.non_english else None
                        )
                        
                        # Get average probability from segments
                        segments = result.get('segments', [])
                        
                        text = result['text'].strip()
                        text = clean_sentence(text)
                        if text.lower() not in filterList:
                            if phrase_complete:
                                # Check for duplicates before adding
                                if not transcription or not is_duplicate_or_partial(text, transcription[-1]):
                                    transcription.append(text)
                                    pyautogui.write(text, interval=0.01)
                                    last_transcription_time = datetime.now(timezone.utc)  # Update last transcription time
                            else:
                                if transcription:
                                    # Only append if it's not a duplicate
                                    if not is_duplicate_or_partial(text, transcription[-1]):
                                        transcription[-1] += ' ' + text
                                        pyautogui.write(' ' + text, interval=0.01)
                                        last_transcription_time = datetime.now(timezone.utc)  # Update last transcription time
                                else:
                                    transcription.append(text)
                                    pyautogui.write(text, interval=0.01)
                                    last_transcription_time = datetime.now(timezone.utc)  # Update last transcription time
                        else:
                            print("Hallucination detected:", text.lower())

                except Exception as e:
                    print(f"Error in inner loop: {str(e)}")
                    root.destroy()
                    if background_listener:
                        background_listener(wait_for_stop=False)
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    break  # Break inner loop to restart main

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
            sleep(1)  # Wait before restarting

if __name__ == "__main__":
    main()
