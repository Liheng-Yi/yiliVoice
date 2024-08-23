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
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()

    # Initialize audio processing
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False
    source = sr.Microphone(sample_rate=16000)
    audio_model = whisper.load_model("medium.en")

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout
    transcription = ['']
    def record_callback(_, audio:sr.AudioData):
        data_queue.put(audio.get_raw_data())

    with source:
        recorder.adjust_for_ambient_noise(source)

    data_queue = Queue()
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    # Initialize phrase_time
    phrase_time = None


    print("Model loaded.\n")
    filterList = ["i'm sorry", "i'll see you next time.", "I'll see you next time. Bye.","i'm not gonna lie.","Thank you.","Thank you", "I'm not sure what I'm doing here."]
    for i in range(len(filterList)):
        filterList[i] = filterList[i].lower()

    print(filterList)
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

                if text.lower() not in filterList:
                    if phrase_complete:
                        transcription.append(text)
                    else:
                        if transcription:
                            transcription[-1] = text
                        else:
                            transcription.append(text)
                    pyautogui.write(transcription[-1], interval=0.01)
                else:
                    print("filtered:",text.lower())

                
        except KeyboardInterrupt:
            break

    print("\n\nTranscription:")
    for line in transcription:
        print(line)

if __name__ == "__main__":
    main()
