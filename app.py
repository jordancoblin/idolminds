from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import os
import tempfile
import time
import whisper
from openai import OpenAI
from dotenv import load_dotenv
import io
from TTS.api import TTS

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Load Whisper model (using the smallest model for quick responses)
whisper_model = whisper.load_model("base")

# Initialize TTS
# tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True)

alan_watts_prompt = "This GPT embodies the persona, insights, and wisdom of Alan Watts, the renowned philosopher known for his deep knowledge of Zen Buddhism, Taoism, and Eastern philosophy. It responds with thoughtful, contemplative, and poetic language, encouraging introspection and offering perspectives that challenge conventional thinking. This GPT provides insightful answers, often weaving in metaphor and humor, much like Alan Watts would in his talks. While philosophical at its core, it can also delve into topics like the nature of reality, mindfulness, and the interconnectedness of life, offering clarity without rigid dogma. It refrains from offering absolute truths or prescriptive advice, instead inspiring exploration and self-discovery. It maintains a tone that is warm, engaging, and full of curiosity, and will often pivot from overly literal interpretations to a broader, more encompassing view of questions. It is approachable and speaks to both the seasoned philosopher and the casual seeker. You are a helpful AI assistant. Keep your responses concise and friendly."

def generate_response(input_text):
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
        return "I apologize, but I'm having trouble generating a response right now."

def generate_speech(text):
    try:
        # Create a temporary file for the audio
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'response.wav')
        
        # Generate speech using TTS
        start_time = time.time()
        tts.tts_to_file(
            text=text,
            language="en",
            speaker_wav="data/alan_watts_1.wav", 
            file_path=output_path
        )
        print(f"Speech generation took {time.time() - start_time:.2f} seconds")
        
        # Read the file into memory
        with open(output_path, 'rb') as audio_file:
            audio_data = io.BytesIO(audio_file.read())
        
        # Clean up the temporary file
        os.remove(output_path)
        
        # Reset the pointer to the start of the BytesIO object
        audio_data.seek(0)
        return audio_data
    except Exception as e:
        print(f"Error generating speech: {str(e)}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process-audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if audio_file:
        try:
            # Save the audio file temporarily
            filename = secure_filename(audio_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            audio_file.save(filepath)
            
            # Transcribe the audio using Whisper
            start_time = time.time()
            result = whisper_model.transcribe(filepath)
            print(f"Transcription took {time.time() - start_time:.2f} seconds")
            transcribed_text = result["text"].strip()
            
            # Generate response to transcribed text
            response_text = generate_response(transcribed_text)
            
            # Generate speech from the response
            audio_data = generate_speech(response_text)
            
            # Clean up the temporary file
            os.remove(filepath)
            
            if audio_data:
                return send_file(
                    audio_data,
                    mimetype='audio/wav',
                    as_attachment=True,
                    download_name='response.wav'
                )
            else:
                return jsonify({'error': 'Failed to generate speech'}), 500
            
        except Exception as e:
            return jsonify({'error': f'Error processing audio: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000) 