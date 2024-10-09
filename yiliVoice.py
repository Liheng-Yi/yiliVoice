import argparse
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch
import pyautogui

from datetime import datetime, timedelta
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

    # Initialize audio processing
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False
    source = sr.Microphone(sample_rate=16000)
    audio_model = whisper.load_model("medium.en")

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout
    volume_threshold = 200
    transcription = ['']
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
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    print("Model loaded.\n")
    filterList = ["i'm sorry","Thanks for watching!", "i'll see you next time.", "I'll see you next time. Bye.","i'm not gonna lie.","Thank you.", 
                  "I'm not gonna lie.", "I'm going to go get some food.", "Thank you. ",
                  "Bye.", "Thank you..", "Thank you. Bye.",
                   " I'm not gonna lie.", " I'm not gonna lie. "]
    filterList = [phrase.lower() for phrase in filterList]
    
    # Initialize phrase_time before the loop
    phrase_time = datetime.utcnow()
    
    while True:
        try:
            now = datetime.utcnow()
            if not data_queue.empty():
                phrase_complete = False
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True
                phrase_time = now

                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
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
