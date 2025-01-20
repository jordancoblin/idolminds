let mediaRecorder;
let audioChunks = [];
let isRecording = false;

async function setupRecorder() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = (event) => {
        audioChunks.push(event.data);
    };

    mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        processAudio(audioBlob);
        audioChunks = [];
    };
}

async function toggleRecording() {
    if (!mediaRecorder) {
        await setupRecorder();
    }

    const button = document.getElementById('recordButton');
    const status = document.getElementById('recordingStatus');

    if (!isRecording) {
        mediaRecorder.start();
        isRecording = true;
        button.textContent = 'Stop Recording';
        button.classList.add('recording');
        status.textContent = 'Recording...';
    } else {
        mediaRecorder.stop();
        isRecording = false;
        button.textContent = 'Start Recording';
        button.classList.remove('recording');
        status.textContent = 'Processing...';
    }
}

async function processText() {
    const text = document.getElementById('textInput').value;
    if (!text) return;

    try {
        const response = await fetch('/process-text', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text }),
        });

        if (response.ok) {
            const audioBlob = await response.blob();
            const responseText = text; // Store the original text
            displayResponse(responseText, audioBlob);
        } else {
            const error = await response.json();
            alert(error.error);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while processing your request.');
    }
}

async function processAudio(audioBlob) {
    const formData = new FormData();
    formData.append('audio', audioBlob);

    try {
        const response = await fetch('/process-audio', {
            method: 'POST',
            body: formData,
        });

        if (response.ok) {
            const audioBlob = await response.blob();
            displayResponse("AI Response", audioBlob);
            document.getElementById('recordingStatus').textContent = '';
        } else {
            const error = await response.json();
            alert(error.error);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while processing your request.');
    }
}

function displayResponse(text, audioBlob = null) {
    const responseSection = document.getElementById('responseSection');
    const textResponse = document.getElementById('textResponse');
    const audioResponse = document.getElementById('audioResponse');

    responseSection.classList.remove('hidden');
    
    textResponse.innerHTML = `
        <div>
            <div class="font-semibold text-gray-700">Your message:</div>
            <div class="text-gray-600">${text}</div>
        </div>
    `;

    if (audioBlob) {
        const audioUrl = URL.createObjectURL(audioBlob);
        audioResponse.src = audioUrl;
        audioResponse.classList.remove('hidden');
        
        // Automatically play the audio response
        audioResponse.play().catch(e => {
            console.log('Auto-play failed:', e);
        });
    } else {
        audioResponse.classList.add('hidden');
    }
} 