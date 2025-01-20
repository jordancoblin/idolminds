from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import os
import tempfile
import time
import whisper
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Initialize OpenAI client
print("OPENAI_API_KEY: ", os.getenv('OPENAI_API_KEY'))
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
# List available models
# def list_models():
#     try:
#         models = client.models.list()
#         return [model.id for model in models]
#     except Exception as e:
#         print(f"Error listing models: {str(e)}")
#         return []

# # Print available models on startup
# print("Available OpenAI Models:")
# for model in list_models():
#     print(f"- {model}")


# Load Whisper model (using the smallest model for quick responses)
# model = whisper.load_model("small")
model = whisper.load_model("base")

def generate_response(input_text):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant. Keep your responses concise and friendly."},
                {"role": "user", "content": input_text}
            ],
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return "I apologize, but I'm having trouble generating a response right now."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process-text', methods=['POST'])
def process_text():
    text = request.json.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        # Generate response using OpenAI
        response_text = generate_response(text)
        return jsonify({'text': response_text}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
            result = model.transcribe(filepath)
            print(f"Transcription took {time.time() - start_time:.2f} seconds")
            transcribed_text = result["text"].strip()
            
            # Generate response to transcribed text
            response_text = generate_response(transcribed_text)
            
            # Clean up the temporary file
            os.remove(filepath)
            
            return jsonify({
                'text': response_text,
                'transcription': transcribed_text
            })
            
        except Exception as e:
            return jsonify({'error': f'Error processing audio: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000) 