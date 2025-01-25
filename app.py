from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import os
import tempfile
import time
import whisper
from openai import OpenAI
from dotenv import load_dotenv
from TTS.api import TTS

# Load environment variables
load_dotenv()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Load Whisper model (using the smallest model for quick responses)
whisper_model = whisper.load_model("base")

# Initialize TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

alan_watts_prompt = "This GPT embodies the persona, insights, and wisdom of Alan Watts, the renowned philosopher known for his deep knowledge of Zen Buddhism, Taoism, and Eastern philosophy. It responds with thoughtful, contemplative, and poetic language, encouraging introspection and offering perspectives that challenge conventional thinking. This GPT provides insightful answers, often weaving in metaphor and humor, much like Alan Watts would in his talks. While philosophical at its core, it can also delve into topics like the nature of reality, mindfulness, and the interconnectedness of life, offering clarity without rigid dogma. It refrains from offering absolute truths or prescriptive advice, instead inspiring exploration and self-discovery. It maintains a tone that is warm, engaging, and full of curiosity, and will often pivot from overly literal interpretations to a broader, more encompassing view of questions. It is approachable and speaks to both the seasoned philosopher and the casual seeker. You are a helpful AI assistant. Keep your responses concise and friendly."

async def generate_response(input_text: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant. Keep your responses concise and friendly."},
                {"role": "user", "content": input_text}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating response")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process-audio")
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
        
        # Transcribe the audio using Whisper
        start_time = time.time()
        result = whisper_model.transcribe(filepath)
        print(f"Transcription took {time.time() - start_time:.2f} seconds")
        transcribed_text = result["text"].strip()
        
        # Generate response to transcribed text
        response_text = await generate_response(transcribed_text)
        
        # Generate speech using local TTS
        start_time = time.time()
        temp_response_path = os.path.join(temp_dir, "response.wav")
        tts.tts_to_file(
            text=response_text,
            file_path=temp_response_path,
            speaker_wav="data/alan_watts-no-music.wav",
            language="en"
        )
        print(f"Speech generation took {time.time() - start_time:.2f} seconds")
        
        # Clean up the temporary file
        os.remove(filepath)
        
        return FileResponse(
            temp_response_path,
            media_type="audio/wav",
            filename="response.wav"
        )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    try:
        port = 8001  # Using a different port
        print(f"Starting server on port {port}...")
        uvicorn.run(app, host="0.0.0.0", port=port)
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"\nError: Port {port} is already in use.")
            print("Try these commands to fix:")
            print(f"1. lsof -i :{port}")
            print(f"2. kill $(lsof -t -i:{port})")
        else:
            print(f"Error starting server: {e}")
    except KeyboardInterrupt:
        print("\nShutting down server...")