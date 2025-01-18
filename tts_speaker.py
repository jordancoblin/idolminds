import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List all available models
print("Available models:")
print(TTS().list_models())

# Example voice cloning with XTTS v2 (newer model)
print("\nVoice cloning")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True).to(device)

# Split long text into sentences for better processing
text = ("Ah, Lillia, dancing at The Common—how perfect, for there is nothing more beautiful than finding joy in the ordinary! On that dance floor, you aren’t just moving—you’re celebrating the rhythm of life itself. Let the music guide you, let laughter flow like sunlight on water, and let the world melt away as you lose yourself in the moment. Dance not just to the music, but let the music dance through you. Cheers to the beauty of simply being!")

tts.tts_to_file(
    text=text,
    speaker_wav="data/alan_watts_1.wav",
    language="en",
    file_path="lillia_dance.wav"
)