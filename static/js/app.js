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
    const micContainer = button.closest('.mic-container');
    const status = document.getElementById('recordingStatus');

    if (!isRecording) {
        mediaRecorder.start();
        isRecording = true;
        micContainer.classList.add('recording');
        status.textContent = 'Listening...';
        
        // Hide previous response when starting new recording
        const responseSection = document.getElementById('responseSection');
        responseSection.classList.add('hidden');
    } else {
        mediaRecorder.stop();
        isRecording = false;
        micContainer.classList.remove('recording');
        status.textContent = 'Processing...';
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
            displayResponse("", audioBlob);
            document.getElementById('recordingStatus').textContent = '';
        } else {
            const error = await response.json();
            alert(error.error);
            document.getElementById('recordingStatus').textContent = '';
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while processing your request.');
        document.getElementById('recordingStatus').textContent = '';
    }
}

function displayResponse(text, audioBlob = null) {
    const responseSection = document.getElementById('responseSection');
    const textResponse = document.getElementById('textResponse');
    const audioResponse = document.getElementById('audioResponse');

    if (audioBlob) {
        const audioUrl = URL.createObjectURL(audioBlob);
        audioResponse.src = audioUrl;
        audioResponse.classList.remove('hidden');
        
        // Show response section with animation
        responseSection.classList.remove('hidden');
        
        // Auto-play the response
        audioResponse.play().catch(e => {
            console.log('Auto-play failed:', e);
        });
    } else {
        audioResponse.classList.add('hidden');
    }
} 