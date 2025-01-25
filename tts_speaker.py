import tempfile
import torch
import torchaudio
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List all available models
print("Available models:")
print(TTS().list_models())

# Example voice cloning with XTTS v2 (newer model)
print("\nVoice cloning")
# tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True).to(device)
# tts = TTS(model_path="models/me", progress_bar=True).to(device)

# Split long text into sentences for better processing
# text = ("Ah, Lillia, dancing at The Common—how perfect, for there is nothing more beautiful than finding joy in the ordinary! On that dance floor, you aren’t just moving—you’re celebrating the rhythm of life itself. Let the music guide you, let laughter flow like sunlight on water, and let the world melt away as you lose yourself in the moment. Dance not just to the music, but let the music dance through you. Cheers to the beauty of simply being!")

# tts.tts_to_file(
#     text=text,
#     speaker_wav="data/alan_watts_1.wav",
#     language="en",
#     file_path="lillia_dance.wav"
# )

# text = "Holy smokes brother, what a beautiful day! This is your good buddy Vlad, does this really sound like me? This was made using only voice cloning, but no fine-tuning. Let's see if it's any better."
text = "Hey there, this is a deep fake version of Jordan. It's a beautiful day to be alive my brothers. The sun is shining, the birds are chirping, the voice cloning is progressing."
# tts.tts_to_file(
#     text=text,
#     language="en",
#     # speaker_wav="data/alan_watts_1.wav",
#     speaker_wav="data/me/jordan_reference.wav",
#     file_path="output/jordan_out.wav"
# )

# Fine-tuned Jordan model
# xtts_config = "models/me/config.json"
# xtts_checkpoint = "models/me/model.pth"
# xtts_vocab = "models/me/vocab.json"
# xtts_speaker = "models/me/speakers.json"
# speaker_audio_file = "data/me/jordan_reference.wav"

# Base xttsv2 model
xtts_config = "models/me/config.json"
xtts_checkpoint = "models/xttsv2/model.pth"
xtts_vocab = "models/xttsv2/vocab.json"
xtts_speaker = "models/xttsv2/speakers_xtts.json"
speaker_audio_file = "data/me/jordan_reference.wav"

config = XttsConfig()
config.load_json(xtts_config)
XTTS_MODEL = Xtts.init_from_config(config)
print("Loading XTTS model")
XTTS_MODEL.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab,speaker_file_path=xtts_speaker, use_deepspeed=False)

gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(audio_path=speaker_audio_file, gpt_cond_len=XTTS_MODEL.config.gpt_cond_len, max_ref_length=XTTS_MODEL.config.max_ref_len, sound_norm_refs=XTTS_MODEL.config.sound_norm_refs)
    
out = XTTS_MODEL.inference(
    text=text,
    language='en',
    gpt_cond_latent=gpt_cond_latent,
    speaker_embedding=speaker_embedding,
    temperature=XTTS_MODEL.config.temperature, # Add custom parameters here
    length_penalty=XTTS_MODEL.config.length_penalty,
    repetition_penalty=XTTS_MODEL.config.repetition_penalty,
    top_k=XTTS_MODEL.config.top_k,
    top_p=XTTS_MODEL.config.top_p,
    enable_text_splitting = True
)

with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
    out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)
    out_path = "output/jordan_base_xtts_out.wav"
    torchaudio.save(out_path, out["wav"], 24000)