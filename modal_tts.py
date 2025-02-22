import modal
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import os
import time
import torch

app = modal.App("idolminds-tts")

# Create a custom image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.9")
    .pip_install(
        "huggingface_hub",
        "torch",
        "torchaudio",
        "TTS",
        "scipy",
        # force_build=True
    )
    # .add_local_dir("models/alan_watts", remote_path="/alan_watts")               # Mount data directory
    # .add_local_file("models/alan_watts/config.json", remote_path="/alan_watts/config.json")
)

with image.imports():
    from huggingface_hub import hf_hub_download
    import torch
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    import os
    import time
    import io
    import scipy.io.wavfile


volume = modal.Volume.from_name(
    "tts-model-storage", create_if_missing=True
)
# Check if the file already exists
# if os.path.exists(file_path):
#     print(f"File already exists: {file_path}, skipping upload.")
#     return
# with volume.batch_upload() as batch:
#     batch.put_file("models/alan_watts/config.json", "/alan_watts/config.json")

# huggingface_secret = modal.Secret.from_name(
#     "huggingface-secret", required_keys=["HF_TOKEN"]
# )

# image = image.env(
#     {"HF_HUB_ENABLE_HF_TRANSFER": "1"}  # turn on faster downloads from HF
# )


# @app.function(
#     volumes={MODEL_DIR: volume},
#     image=image,
#     # secrets=[huggingface_secret],
#     timeout=600,  # 10 minutes
# )


@app.cls(
    gpu=modal.gpu.T4(),
    image=image,
    container_idle_timeout=120,
    volumes={"/models": volume},
)
class TTSService:    
    @modal.enter()
    def enter(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("device: ", self.device)

        REPO_ID = "jcoblin/idolminds-tts"
        CONFIG_FILE = "config.json"
        CHECKPOINT_FILE = "model.pth"
        VOCAB_FILE = "vocab.json"
        SPEAKERS_FILE = "speakers_xtts.pth"
        REFERENCE_FILE = "reference.wav"

        # Ensure the correct full paths inside the volume
        local_model_dir = f"/models/alan_watts"
        os.makedirs(local_model_dir, exist_ok=True)  # Ensure the directory exists
        
        # Download files from Hugging Face
        for file in [CHECKPOINT_FILE, VOCAB_FILE, SPEAKERS_FILE, REFERENCE_FILE]:
            hf_hub_download(
                repo_id=REPO_ID,
                local_dir=local_model_dir,
                filename=f"alan_watts/{file}"
            )

        # Initialize XTTS model with the downloaded files
        xtts_config = f"{local_model_dir}/{CONFIG_FILE}"
        xtts_checkpoint = f"{local_model_dir}/{CHECKPOINT_FILE}"
        xtts_vocab = f"{local_model_dir}/{VOCAB_FILE}"
        xtts_speaker = f"{local_model_dir}/{SPEAKERS_FILE}"
        self.reference_audio = f"{local_model_dir}/{REFERENCE_FILE}"
        
        self.config = XttsConfig()
        self.config.load_json(xtts_config)
        
        print("Loading XTTS model...")
        self.model = Xtts.init_from_config(self.config)
        self.model.load_checkpoint(
            self.config,
            checkpoint_path=xtts_checkpoint,
            vocab_path=xtts_vocab,
            speaker_file_path=xtts_speaker,
            use_deepspeed=False
        )
        if self.device == "cuda":
            self.model.cuda()
        self.model.to(self.device)
        print("XTTS model loaded successfully")

    @modal.method()
    def generate_speech(self, text: str) -> bytes:
        """Generate speech from text using the XTTS model."""
        try:
            # Get conditioning latents
            gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
                audio_path=self.reference_audio,
                gpt_cond_len=self.config.gpt_cond_len,
                max_ref_length=self.config.max_ref_len,
                sound_norm_refs=self.config.sound_norm_refs,
            )

            # TODO: add explicit attention mask?
            # Generate speech
            out = self.model.inference(
                text=text,
                language='en',
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                temperature=self.config.temperature,
                length_penalty=self.config.length_penalty,
                repetition_penalty=self.config.repetition_penalty,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                enable_text_splitting=True
            )

            # Process the output
            wav = out["wav"]
            if isinstance(wav, torch.Tensor):
                wav = wav.detach().cpu().numpy()

            # Convert to int16
            wav = (wav * 32767).astype('int16')

            # Save to bytes
            wav_bytes = io.BytesIO()
            scipy.io.wavfile.write(wav_bytes, rate=24000, data=wav)
            return wav_bytes.getvalue()

        except Exception as e:
            print(f"Error generating speech: {str(e)}")
            raise
    
    @modal.method()
    def save_to_volume(self, audio_bytes: bytes, filename: str):
        import os
        
        out_dir = "/models/alan_watts/out"
        out_path = f"{out_dir}/{filename}"

        os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(audio_bytes)
        
        return out_path

@app.local_entrypoint()
def main():
    import time

    # Sample text to convert to speech
    text = "Hello, this is a test of the text to speech system."

    service = TTSService()

    # TODO: This first speech generation takes a while - I'm guessing
    # it's running enter() -> how to get this to load on startup?
    print("Generating speech...")
    start = time.time()
    audio_bytes = service.generate_speech.remote(text)
    end = time.time()
    print(f"Speech generation took: {end - start} seconds")

    print("Generating speech...")
    start = time.time()
    audio_bytes = service.generate_speech.remote(text)
    end = time.time()
    print(f"Speech generation # 2 took: {end - start} seconds")
    
    # Save to volume using remote function
    saved_path = service.save_to_volume.remote(audio_bytes, "modal_tts_output.wav")
    print(f"Audio saved to {saved_path}")