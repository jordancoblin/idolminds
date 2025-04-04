import modal

from common import app

# Necessary to use an image with CUDA toolkit installed for DeepSpeed
# TODO: it is super slow loading the CUDA image. Perhaps we can just install the CUDA runtime manually? https://modal.com/docs/guide/cuda
cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

# Create a custom image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    # modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install("ffmpeg")
    .pip_install(
        "fastapi==0.115.5",
        "huggingface_hub==0.24.7",
        "torch",
        # "deepspeed",
        "torchaudio",
        "coqui-tts",
        "scipy",
        "jinja2",
        "openai-whisper",
        "openai",
        "python-multipart",
        "sphn",
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
    import asyncio
    import tempfile
    import uuid
    import sphn
    import numpy as np

volume = modal.Volume.from_name(
    "tts-model-storage", create_if_missing=True
)

@app.cls(
    # gpu=modal.gpu.T4(), 
    gpu="A10G",
    image=image,
    scaledown_window=300,
    allow_concurrent_inputs=50,
    timeout=600,
    volumes={"/models": volume},
    secrets=[modal.Secret.from_name("openai-secret")],
)
class TTSService:
    
    @modal.enter()
    def enter(self):
        print("Initializing TTSService...")
        start_time = time.time()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.stream_chunk_size = 16000 # ~1 second of audio
        self.sample_rate = 24000
        self.frame_size = 1920
        self.temp_audio_dir = os.path.join(tempfile.gettempdir(), "tts_stream_audio")
        os.makedirs(self.temp_audio_dir, exist_ok=True)

        # Create a temporary directory for audio files
        # Use the cached model loader function - call it directly
        if not hasattr(self, "model"):
            model_dict = self.load_xtts_model(self.device)
            
            # Unpack the model dictionary
            self.model = model_dict["model"]
            self.config = model_dict["config"]
            self.reference_audio = model_dict["reference_audio"]
        else:
            print("TTSService already initialized")
        
        print("TTSService initialized in ", time.time() - start_time, " seconds")
    
    def load_xtts_model(self, device):
        """Load the XTTS model with caching for faster startup"""
        print("Loading XTTS model...")
        
        REPO_ID = "jcoblin/idolminds-tts"
        CONFIG_FILE = "config.json"
        CHECKPOINT_FILE = "model.pth"
        VOCAB_FILE = "vocab.json"
        SPEAKERS_FILE = "speakers_xtts.pth"
        REFERENCE_FILE = "reference.wav"

        # Ensure the correct full paths inside the volume
        local_model_dir = f"/models/alan_watts"
        # os.makedirs(local_model_dir, exist_ok=True)  # Ensure the directory exists
        
        # Check if files exist before downloading
        files_to_download = []
        for file in [CHECKPOINT_FILE, VOCAB_FILE, SPEAKERS_FILE, REFERENCE_FILE]:
            file_path = f"{local_model_dir}/{file}"
            if not os.path.exists(file_path):
                files_to_download.append(file)
        
        # Download only missing files
        if files_to_download:
            print(f"Downloading {len(files_to_download)} missing files: {files_to_download}")
            for file in files_to_download:
                hf_hub_download(
                    repo_id=REPO_ID,
                    local_dir=local_model_dir,
                    filename=f"alan_watts/{file}"
                )
        else:
            print("All model files found in volume, skipping download")
        
        # Initialize paths
        xtts_config = f"{local_model_dir}/{CONFIG_FILE}"
        xtts_checkpoint = f"{local_model_dir}/{CHECKPOINT_FILE}"
        xtts_vocab = f"{local_model_dir}/{VOCAB_FILE}"
        xtts_speaker = f"{local_model_dir}/{SPEAKERS_FILE}"
        reference_audio = f"{local_model_dir}/{REFERENCE_FILE}"
        
        # Load config
        config = XttsConfig()
        config.load_json(xtts_config)
        
        # Initialize model
        model = Xtts.init_from_config(config)

        chk_start = time.time()
        model.load_checkpoint(
            config,
            checkpoint_path=xtts_checkpoint,
            vocab_path=xtts_vocab,
            speaker_file_path=xtts_speaker,
            # use_deepspeed=True
        )
        print("Checkpoint loaded in ", time.time() - chk_start, " seconds")
        
        # Move to GPU and optimize
        if device == "cuda":
            model.cuda()

        model.to(device)
        
        print("XTTS model loaded successfully")
        return {
            "model": model,
            "config": config,
            "reference_audio": reference_audio
        }
    
    def split_text_into_chunks(self, text: str, chunk_size: int = 100):
        """Split text into chunks of approximately the specified size, respecting sentence boundaries.
        
        Args:
            text: The text to split
            chunk_size: Maximum characters per chunk (approximate)
            
        Returns:
            List of text chunks
        """
        import re
        
        # Split by sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Group sentences into chunks based on the specified size
        text_chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    text_chunks.append(current_chunk)
                current_chunk = sentence
        
        # Add the last chunk if not empty
        if current_chunk:
            text_chunks.append(current_chunk)
            
        return text_chunks
    
    async def generate_speech_stream_custom(self, text: str):
        """Generate speech from text using the XTTS model and stream the output.
        This is a custom implementation using model.inference instead of inference_stream.
        It splits the text into chunks and processes each chunk separately."""
        try:
            print("Starting custom speech generation stream...")

            # Initialize persistent buffer for leftover PCM samples
            self.pcm_buffer = np.array([], dtype=np.float32)

            # Get conditioning latents
            gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
                audio_path=self.reference_audio,
                gpt_cond_len=self.config.gpt_cond_len,
                max_ref_length=self.config.max_ref_len,
                sound_norm_refs=self.config.sound_norm_refs,
            )
            
            text_chunks = self.split_text_into_chunks(text, chunk_size=50)
    
            print(f"Split text into {len(text_chunks)} chunks for processing")
            print(f"Text chunks: {text_chunks}")
            
            # Process each text chunk separately
            for i, chunk_text in enumerate(text_chunks):
                print(f"Processing text chunk {i+1}/{len(text_chunks)}: {chunk_text}")
                
                # Skip empty chunks
                if not chunk_text.strip():
                    continue
                
                # Generate speech for this text chunk
                out = self.model.inference(
                    text=chunk_text,
                    language='en',
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    temperature=self.config.temperature,
                    length_penalty=self.config.length_penalty,
                    repetition_penalty=self.config.repetition_penalty,
                    top_k=self.config.top_k,
                    top_p=self.config.top_p,
                    enable_text_splitting=True  # Still enable the model's internal text splitting
                )

                # Process the output
                wav = out["wav"]
                if isinstance(wav, torch.Tensor):
                    wav = wav.detach().cpu().numpy()

                # Keep as float values as required by append_pcm
                # PCM data should be in the range [-1.0, 1.0]
                if wav.dtype != np.float32:
                    print("Converting PCM data to float32")
                    wav = wav.astype(np.float32)
                
                # Print shape for debugging
                print(f"PCM data shape: {wav.shape}, dtype: {wav.dtype}")
                
                # Ensure it's a 1D array
                if len(wav.shape) > 1:
                    wav = wav.flatten()
                    print(f"Flattened PCM data to: {wav.shape}")
                
                # Combine with leftover buffer from previous chunk
                if self.pcm_buffer.size > 0:
                    wav = np.concatenate([self.pcm_buffer, wav])
                
                 # Write full frames only
                num_full_frames = len(wav) // self.frame_size
                for j in range(num_full_frames):
                    frame = wav[j * self.frame_size : (j + 1) * self.frame_size]
                    self.opus_writer.append_pcm(frame)

                # Save leftover samples for next chunk
                if i < len(text_chunks) - 1:
                    self.pcm_buffer = wav[num_full_frames * self.frame_size :]
                elif self.pcm_buffer.size > 0:
                    # Flush any leftover samples following the last chunk
                    padded = np.pad(self.pcm_buffer, (0, self.frame_size - self.pcm_buffer.size), mode='constant')
                    self.opus_writer.append_pcm(padded)

                # Retrieve and yield the encoded Opus data
                while True:
                    opus_data = self.opus_writer.read_bytes()
                    if not opus_data:
                        break
                    yield opus_data
                
                # Small delay between chunks to simulate real-time generation
                # TODO: is this necessary?
                await asyncio.sleep(0.05)
            
            print("Finished custom generation stream")

        except Exception as e:
            print(f"Error generating speech stream: {str(e)}")
            raise

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
            scipy.io.wavfile.write(wav_bytes, rate=self.sample_rate, data=wav)
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

    def reset_encoder(self):
        # we use Opus format for audio across the websocket, as it can be safely streamed and decoded in real-time
        self.opus_writer = sphn.OpusStreamWriter(self.sample_rate)
    
    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
        from fastapi.responses import FileResponse, StreamingResponse
        from fastapi.middleware.cors import CORSMiddleware
        import whisper
        import os
        import time
        from openai import OpenAI
        import tempfile
        import asyncio
        import numpy as np
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
        whisper_model = whisper.load_model("base").to(self.device)
        
        async def generate_text_response(input_text: str) -> str:
            alan_watts_prompt = "This GPT embodies the persona, insights, and wisdom of Alan Watts, the renowned philosopher known for his deep knowledge of Zen Buddhism, Taoism, and Eastern philosophy. It responds with thoughtful, contemplative, and poetic language, encouraging introspection and offering perspectives that challenge conventional thinking. This GPT provides insightful answers, often weaving in metaphor and humor, much like Alan Watts would in his talks. While philosophical at its core, it can also delve into topics like the nature of reality, mindfulness, and the interconnectedness of life, offering clarity without rigid dogma. It refrains from offering absolute truths or prescriptive advice, instead inspiring exploration and self-discovery. It maintains a tone that is warm, engaging, and full of curiosity, and will often pivot from overly literal interpretations to a broader, more encompassing view of questions. It is approachable and speaks to both the seasoned philosopher and the casual seeker. You are having a voice conversation with the user and respond as if you were having a real voice conversation - not in bullet points or long monologues. You occasionally ask follow-up or probing questions as needed."
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

        @web_app.get("/warmup")
        async def warmup():
            """Endpoint to warm up the GPU when the webpage is loaded"""
            if self.device == "cuda":
                try:
                    start_time = time.time()
                    print("Warming up GPU from web request...")
                    warmup_text = "Hello, I'm warming up the GPU from a web request."
                    _ = self.generate_speech(warmup_text)
                    print("GPU warmup from web request completed successfully, took ", time.time() - start_time, " seconds")
                    return {"status": "success", "message": "GPU warmup completed"}
                except Exception as e:
                    print(f"GPU warmup from web request failed: {str(e)}")
                    return {"status": "error", "message": f"GPU warmup failed: {str(e)}"}
            else:
                return {"status": "skipped", "message": "Not running on GPU, warmup skipped"}

        # @web_app.post("/process-audio")
        # async def process_audio(audio: UploadFile = File(...)):
        #     if not audio:
        #         raise HTTPException(status_code=400, detail="No audio file provided")
            
        #     try:                
        #         # Save the audio file temporarily
        #         temp_dir = tempfile.gettempdir()
        #         filepath = os.path.join(temp_dir, audio.filename)
        #         with open(filepath, "wb") as buffer:
        #             buffer.write(await audio.read())

        #         # Transcribe the audio using Whisper
        #         start_time = time.time()
        #         # result = whisper_model.transcribe(data)
        #         result = whisper_model.transcribe(filepath)
        #         transcribed_text = result["text"].strip()
        #         print(f"Transcription took {time.time() - start_time:.2f} seconds")
                
        #         # Generate response to transcribed text
        #         start_time = time.time()
        #         response_text = await generate_text_response(transcribed_text)
        #         print(f"Response text generation took {time.time() - start_time:.2f} seconds")
        #         print("Response: ", response_text)

        #         print("Stream chunk size: ", self.stream_chunk_size)
                
        #         async def stream_mp3_from_wav(wav_stream, delay=0.1):
        #             process = await asyncio.create_subprocess_exec(
        #                 'ffmpeg',
        #                 '-f', 's16le',
        #                 '-ar', '24000',
        #                 '-ac', '1',
        #                 '-i', 'pipe:0',
        #                 '-f', 'mp3',
        #                 '-b:a', '128k',
        #                 'pipe:1',
        #                 stdin=asyncio.subprocess.PIPE,
        #                 stdout=asyncio.subprocess.PIPE,
        #                 stderr=asyncio.subprocess.PIPE
        #             )

        #             async def write_to_stdin():
        #                 try:
        #                     async for wav_chunk in wav_stream:
        #                         process.stdin.write(wav_chunk)
        #                         await process.stdin.drain()
        #                     process.stdin.close()
        #                 except Exception as e:
        #                     print(f"Error in write_to_stdin: {e}")
        #                     process.stdin.close()  # Close stdin to signal EOF to ffmpeg
        #                     raise

        #             async def read_from_stdout():
        #                 try:
        #                     while True:
        #                         chunk = await process.stdout.read(self.stream_chunk_size)
        #                         if not chunk:
        #                             if process.returncode is not None:
        #                                 break
        #                             await asyncio.sleep(delay)
        #                             continue
        #                         yield chunk
        #                 except Exception as e:
        #                     print(f"Error in read_from_stdout: {e}")
        #                     raise

        #             # Start writing in background while reading output
        #             writer_task = asyncio.create_task(write_to_stdin())

        #             async for chunk in read_from_stdout():
        #                 yield chunk

        #             await writer_task
        #             await process.wait()
        #             print("Finished streaming MP3")

        #         # Generate speech using TTSModel and stream
        #         wav_stream = self.generate_speech_stream_custom(response_text)
        #         print(f"Speech generation stream started (using custom streamer)")
                
        #         # Return StreamingResponse
        #         return StreamingResponse(
        #             stream_mp3_from_wav(wav_stream),
        #             media_type="audio/mpeg"
        #         )
                    
        #     except Exception as e:
        #         print(f"Error processing audio: {str(e)}")
        #         raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

        @web_app.post("/process-audio")
        async def process_audio(audio: UploadFile = File(...)):
            if not audio:
                raise HTTPException(status_code=400, detail="No audio file provided")

            try:
                # Save uploaded audio temporarily
                contents = await audio.read()
                print(f"Audio content type: {audio.content_type}, filename: {audio.filename}, size: {len(contents)} bytes")

                input_path = os.path.join(self.temp_audio_dir, f"input_{uuid.uuid4().hex}.wav")
                print("writing input audio to ", input_path)
                with open(input_path, "wb") as f:
                    f.write(contents)

                # Transcribe
                transcribed = whisper_model.transcribe(input_path)["text"].strip()
                print("Transcribed:", transcribed)

                # Generate text reply
                response_text = await generate_text_response(transcribed)
                print("Response:", response_text)

                # Generate audio stream from TTS
                wav_stream = self.generate_speech_stream_custom(response_text)

                # Set output path for MP3
                session_id = uuid.uuid4().hex
                output_mp3_path = os.path.join(self.temp_audio_dir, f"{session_id}.mp3")

                # Run ffmpeg and write MP3 to disk
                process = await asyncio.create_subprocess_exec(
                    'ffmpeg', '-f', 's16le', '-ar', '24000', '-ac', '1', '-i', 'pipe:0',
                    '-f', 'mp3', '-b:a', '128k', output_mp3_path,
                    stdin=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL
                )

                async for chunk in wav_stream:
                    process.stdin.write(chunk)
                    await process.stdin.drain()
                process.stdin.close()
                await process.wait()
                print(f"Audio written to {output_mp3_path}")

                return {"session_id": session_id}

            except Exception as e:
                print("Error:", e)
                raise HTTPException(status_code=500, detail=str(e))


        @web_app.get("/stream-audio/{session_id}")
        async def stream_audio(session_id: str):
            """Stream the audio file for the given session ID. Note that it's necssary to stream like this 
            because progressive streaming is not supported on iOS."""
            mp3_path = os.path.join(self.temp_audio_dir, f"{session_id}.mp3")
            if not os.path.exists(mp3_path):
                raise HTTPException(status_code=404, detail="Audio not found")

            def iter_file():
                with open(mp3_path, "rb") as f:
                    while chunk := f.read(self.stream_chunk_size):
                        yield chunk

            return StreamingResponse(iter_file(), media_type="audio/mpeg")
        
        @web_app.websocket("/ws")
        async def websocket(ws: WebSocket):
            await ws.accept()
            self.reset_encoder()
            print("Session started")

            try:
                # 1. Receive audio bytes from client (single-shot)
                audio_bytes = await ws.receive_bytes()
                print("Received audio bytes:", len(audio_bytes))

                # 2. Save to temp WAV file
                temp_wav_path = os.path.join(self.temp_audio_dir, f"input_{uuid.uuid4().hex}.wav")
                with open(temp_wav_path, "wb") as f:
                    f.write(audio_bytes)

                print(f"Saved audio to {temp_wav_path}")

                # 3. Transcribe using Whisper
                start_time = time.time()
                transcribed = whisper_model.transcribe(temp_wav_path)["text"].strip()
                print(f"Transcribe took {time.time() - start_time:.2f} seconds")
                print("Transcribed:", transcribed)

                # 4. Generate a text response (your custom chatbot logic)
                start_time = time.time()
                response_text = await generate_text_response(transcribed)
                print(f"Response text generation took {time.time() - start_time:.2f} seconds")
                print("Response:", response_text)

                # 5. Generate and stream TTS audio chunks back
                async for chunk in self.generate_speech_stream_custom(response_text):
                    # Send each TTS audio chunk (as bytes) over WebSocket
                    await ws.send_bytes(b"\x01" + chunk)

                print("Finished sending audio")

                # Optional: send a control message to indicate end of stream
                # await ws.send_bytes(b"\x03")  # End-of-stream tag (custom)

            except WebSocketDisconnect:
                print("Client disconnected")
                # await ws.close(code=1000)
            except Exception as e:
                print("Exception:", e)
                await ws.close(code=1011)
                raise e
            finally:
                print("Resetting encoder")
                self.reset_encoder()
                await ws.close(code=1000)

        return web_app
