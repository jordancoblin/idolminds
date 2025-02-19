import modal
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
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
    )
)

# with image.imports():
#     from huggingface_hub import hf_hub_download
#     import torch

volume = modal.Volume.from_name(
    "tts-model-storage", create_if_missing=True
)
MODEL_DIR = "/alan_watts"

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
    volumes={MODEL_DIR: volume},
)
class TTSService:    
    @modal.enter()
    def enter(self):
        from huggingface_hub import hf_hub_download

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        REPO_ID = "jcoblin/idolminds-tts"
        SUB_DIR = "alan_watts"
        CONFIG_FILE = "config.json"
        CHECKPOINT_FILE = "model.pth"
        VOCAB_FILE = "vocab.json"
        SPEAKERS_FILE = "speakers_xtts.pth"
        REFERENCE_FILE = "reference.wav"
        
        # Download files from Hugging Face
        for file in [CONFIG_FILE, CHECKPOINT_FILE, VOCAB_FILE, SPEAKERS_FILE, REFERENCE_FILE]:
            hf_hub_download(
                repo_id=REPO_ID,
                local_dir=MODEL_DIR,
                filename=f"{SUB_DIR}/{file}"
            )

        # Initialize XTTS model with the downloaded files
        xtts_config = f"{MODEL_DIR}/{CONFIG_FILE}"
        xtts_checkpoint = f"{MODEL_DIR}/{CHECKPOINT_FILE}"
        xtts_vocab = f"{MODEL_DIR}/{VOCAB_FILE}"
        xtts_speaker = f"{MODEL_DIR}/{SPEAKERS_FILE}"
        self.reference_audio = f"{MODEL_DIR}/{REFERENCE_FILE}"
        
        # TODO: this config does not exist in the container
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
        self.model.to(self.device)
        print("XTTS model loaded successfully")

    # def load_tts_model(self):
    #     xtts_config = "/content/models/alan_watts/config.json"
    #     xtts_checkpoint = "/content/models/alan_watts/model.pth"
    #     xtts_vocab = "/content/models/alan_watts/vocab.json"
    #     xtts_speaker = "/content/models/alan_watts/speakers_xtts.pth"

    #     config = XttsConfig()
    #     config.load_json(xtts_config)
    #     xtts_model = Xtts.init_from_config(config)
    #     print("Loading XTTS model")
    #     xtts_model.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, speaker_file_path=xtts_speaker, use_deepspeed=False)
    #     xtts_model.cuda()  # Move model to GPU
    #     return xtts_model, config

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
            import io
            import scipy.io.wavfile
            wav_bytes = io.BytesIO()
            scipy.io.wavfile.write(wav_bytes, rate=24000, data=wav)
            return wav_bytes.getvalue()

        except Exception as e:
            print(f"Error generating speech: {str(e)}")
            raise 

@app.local_entrypoint()
def main():
    # Sample text to convert to speech
    text = "Hello, this is a test of the text to speech system."
    
    # Create stub and get speech
    f = TTSService()
    audio_bytes = f.generate_speech.remote(text)
    
    # Save the output
    output_path = "modal_tts_output.wav"
    with open(output_path, "wb") as f:
        f.write(audio_bytes)
    print(f"Audio saved to {output_path}")