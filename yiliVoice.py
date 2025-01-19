import argparse
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch
import pyautogui
import keyboard

from datetime import datetime, timedelta, UTC
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    parser.add_argument("--volume_threshold", default=0.01,
                        help="Minimum volume level to consider valid audio.", type=float)
    args = parser.parse_args()

    # More detailed GPU check
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
    
    # Load model with weights_only=True to avoid the warning
    model_path = whisper.load_model("medium.en", device=device, download_root=None)
    audio_model = model_path.to(device)
    
    print(f"Model loaded on {device}")
    
    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout
    volume_threshold = args.volume_threshold * 32768.0  # Scale to audio range
    transcription = []
    def record_callback(_, audio:sr.AudioData):
        audio_data = audio.get_raw_data()
        volume = np.sqrt(np.mean(np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) ** 2))
        if volume > volume_threshold:
            print(f"Volume: {volume}")
            data_queue.put(audio_data)
        else:
            print(f"Discarded low volume audio: {volume}")

    with source:
        recorder.adjust_for_ambient_noise(source)

    data_queue = Queue()
    recording_active = False
    background_listener = None
    phrase_time = datetime.now(UTC)  # Initialize phrase_time here
    last_activity_time = datetime.now(UTC)  # Add this line to track last activity
    
    # Add filterList definition here
    filterList = [
        "i'm sorry",
        "thanks for watching!",
        "i'll see you next time.",
        "i'll see you next time. bye.",
        "i'm not gonna lie.",
        "thank you.",
        "i'm not gonna lie.",
        "i'm going to go get some food.",
        "thank you.",
        "bye.",
        "thank you..",
        "thank you. bye.",
        "i'm not gonna lie.",
        "i'm not gonna lie."
    ]
    filterList = [phrase.lower() for phrase in filterList]  # Convert all to lowercase
    
    while True:
        try:
            # Check for Ctrl+Q press to toggle recording
            if keyboard.is_pressed('ctrl+q'):
                recording_active = not recording_active
                if recording_active:
                    print("Recording started...")
                    transcription = []
                    data_queue.queue.clear()
                    phrase_time = datetime.now(UTC)  # Reset phrase_time when starting new recording
                    last_activity_time = datetime.now(UTC)  # Reset activity timer
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

            now = datetime.now(UTC)
            
            # Check for idle timeout (4 minutes)
            if recording_active and (now - last_activity_time > timedelta(minutes=4)):
                print("Idle timeout reached. Recording stopped...")
                recording_active = False
                if background_listener:
                    background_listener(wait_for_stop=False)
                    background_listener = None
                print("\nSession Transcription:")
                for line in transcription:
                    print(line)
                print("\n")
                continue

            if not data_queue.empty() and recording_active:
                last_activity_time = now  # Update activity timer when audio is processed
                phrase_complete = False
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True
                phrase_time = now

                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                # Add device specification for transcription
                result = audio_model.transcribe(
                    audio_np, 
                    fp16=(device == "cuda"),  # Use fp16 only when on GPU
                    language="en" if not args.non_english else None
                )
                
                # Get average probability from segments
                segments = result.get('segments', [])
                if segments:
                    avg_probability = sum(s.get('avg_logprob', -1) for s in segments) / len(segments)
                    confidence_threshold = -1  # Adjust this threshold as needed (-1 is medium confidence, higher is better)
                    if avg_probability < confidence_threshold:
                        print(f"Low confidence ({avg_probability:.2f}), discarding: {result['text']}")
                        continue
        
                text = result['text'].strip()
                text = clean_sentence(text)
                if text.lower() not in filterList:
                    if phrase_complete:
                        transcription.append(text)
                    else:
                        if transcription:
                            transcription[-1] += ' ' + text
                        else:
                            transcription.append(text)
                    pyautogui.write(transcription[-1], interval=0.01)
                else:
                    print("Hallucination detected:", text.lower())

        except KeyboardInterrupt:
            break

    print("\n\nTranscription:")
    for line in transcription:
        print(line)

if __name__ == "__main__":
    main()
