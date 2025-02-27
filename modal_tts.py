import modal

from common import app

# Create a custom image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg")
    .pip_install(
        "fastapi==0.115.5",
        "huggingface_hub==0.24.7",
        "torch",
        "torchaudio",
        "TTS",
        "scipy",
        "jinja2",
        "openai-whisper",
        "openai",
        "python-multipart",
    )
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

@app.cls(
    gpu=modal.gpu.T4(),
    image=image,
    container_idle_timeout=120,
    timeout=600,
    volumes={"/models": volume},
    secrets=[modal.Secret.from_name("openai-secret")],
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

    # @modal.method()
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
    
    # @modal.method()
    def save_to_volume(self, audio_bytes: bytes, filename: str):
        import os
        
        out_dir = "/models/alan_watts/out"
        out_path = f"{out_dir}/{filename}"

        os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(audio_bytes)
        
        return out_path
    
    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, UploadFile, File, HTTPException
        from fastapi.responses import FileResponse
        from fastapi.middleware.cors import CORSMiddleware
        import whisper
        import os
        import time
        from openai import OpenAI
        import tempfile

        web_app = FastAPI()
        
        # Add CORS middleware
        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Allow all origins, or specify ["https://jordancoblin--idolminds-tts-web-dev.modal.run"]
            allow_credentials=True,
            allow_methods=["*"],  # Allow all methods
            allow_headers=["*"],  # Allow all headers
        )

        # Initialize OpenAI client
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        # Load Whisper model (using the smallest model for quick responses)
        whisper_model = whisper.load_model("base")
        
        async def generate_text_response(input_text: str) -> str:
            alan_watts_prompt = "This GPT embodies the persona, insights, and wisdom of Alan Watts, the renowned philosopher known for his deep knowledge of Zen Buddhism, Taoism, and Eastern philosophy. It responds with thoughtful, contemplative, and poetic language, encouraging introspection and offering perspectives that challenge conventional thinking. This GPT provides insightful answers, often weaving in metaphor and humor, much like Alan Watts would in his talks. While philosophical at its core, it can also delve into topics like the nature of reality, mindfulness, and the interconnectedness of life, offering clarity without rigid dogma. It refrains from offering absolute truths or prescriptive advice, instead inspiring exploration and self-discovery. It maintains a tone that is warm, engaging, and full of curiosity, and will often pivot from overly literal interpretations to a broader, more encompassing view of questions. It is approachable and speaks to both the seasoned philosopher and the casual seeker. You are a helpful AI assistant. Keep your responses concise and friendly."
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": alan_watts_prompt},
                        {"role": "user", "content": input_text}
                    ]
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error generating response: {str(e)}")
                raise HTTPException(status_code=500, detail="Error generating response")

        @web_app.post("/process-audio")
        async def process_audio(audio: UploadFile = File(...)):
            if not audio:
                raise HTTPException(status_code=400, detail="No audio file provided")
            
            try:
                # Save the audio file temporarily
                temp_dir = tempfile.gettempdir()
                filepath = os.path.join(temp_dir, audio.filename)
                
                with open(filepath, "wb") as buffer:
                    content = await audio.read()
                    buffer.write(content)
                print("Saved temp audio file to: ", filepath)
                
                # Transcribe the audio using Whisper
                start_time = time.time()
                result = whisper_model.transcribe(filepath)
                print(f"Transcription took {time.time() - start_time:.2f} seconds")
                transcribed_text = result["text"].strip()
                
                # Generate response to transcribed text
                response_text = await generate_text_response(transcribed_text)
                print("Response: ", response_text)
                
                # Generate speech using TTSModel
                start_time = time.time()
                temp_response_path = os.path.join(temp_dir, "response.wav")
                wav_bytes = self.generate_speech(response_text)
                print(f"Speech generation took {time.time() - start_time:.2f} seconds")
                
                # Save the generated audio
                try:
                    # Write bytes directly to file
                    with open(temp_response_path, "wb") as f:
                        f.write(wav_bytes)
                except Exception as e:
                    print("Error writing to file: ", e)
                
                # Clean up the temporary file
                os.remove(filepath)
                
                return FileResponse(
                    temp_response_path,
                    media_type="audio/wav",
                    filename="response.wav"
                )
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
        
        return web_app

# @app.local_entrypoint()
# def main():
#     import time

#     # Sample text to convert to speech
#     text = "Hello, this is a test of the text to speech system."

#     service = TTSService()

#     # TODO: This first speech generation takes a while - I'm guessing
#     # it's running enter() -> how to get this to load on startup?
#     print("Generating speech...")
#     start = time.time()
#     audio_bytes = service.generate_speech.remote(text)
#     end = time.time()
#     print(f"Speech generation took: {end - start} seconds")

#     print("Generating speech...")
#     start = time.time()
#     audio_bytes = service.generate_speech.remote(text)
#     end = time.time()
#     print(f"Speech generation # 2 took: {end - start} seconds")
    
#     # Save to volume using remote function
#     saved_path = service.save_to_volume.remote(audio_bytes, "modal_tts_output.wav")
#     print(f"Audio saved to {saved_path}")