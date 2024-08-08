import whisper
import torch

if torch.cuda.is_available():
    print("CUDA is available. Using GPU for processing.")
else:
    print("CUDA is not available. Using CPU for processing.")

model = whisper.load_model("base")
result = model.transcribe("jfk.flac")
print(result["text"])