import pyaudio
import wave
import whisper
import threading
import numpy as np

# Load Whisper model
model = whisper.load_model("base")

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Whisper model uses 16kHz
CHUNK = 1024  # Number of frames per buffer

audio_interface = pyaudio.PyAudio()

# Buffer to hold audio
audio_buffer = np.array([], dtype=np.int16)

def process_audio():
    global audio_buffer
    while True:
        # Check if buffer has enough data to process (~3 seconds)
        if len(audio_buffer) > RATE * 3:
            print("Processing new 3-second audio segment...")
            # Convert buffer to proper format
            audio_to_process = np.copy(audio_buffer)
            audio_buffer = np.array([], dtype=np.int16)

            # Save chunk to a temporary WAV file (or process directly if possible)
            with wave.open("temp.wav", "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(audio_interface.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(audio_to_process.tobytes())

            # Transcribe audio
            result = model.transcribe("temp.wav")
            print(result['text'])

def callback(in_data, frame_count, time_info, status):
    global audio_buffer
    audio_data = np.frombuffer(in_data, dtype=np.int16)
    audio_buffer = np.append(audio_buffer, audio_data)
    return (in_data, pyaudio.paContinue)

# Open stream using callback
stream = audio_interface.open(format=FORMAT, channels=CHANNELS,
                              rate=RATE, input=True,
                              frames_per_buffer=CHUNK,
                              stream_callback=callback)

# Start a separate thread for processing
processing_thread = threading.Thread(target=process_audio)
processing_thread.start()

# Start recording
print("Starting audio recording...")
stream.start_stream()

try:
    while True:
        pass
except KeyboardInterrupt:
    print("Stopping audio recording...")
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio_interface.terminate()
    processing_thread.join()
