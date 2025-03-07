let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let isReady = false;

// Get the TTSService URL based on current domain
const getTTSServiceURL = () => {
    // Get the current URL
    const currentURL = new URL(window.location.href);
    let hostname = currentURL.hostname;

    console.log("hostname: ", hostname);
    
    // If we're in a Modal deployment, we can derive the TTSService URL from the current hostname
    // Modify this pattern based on your actual Modal deployment naming conventions
    if (hostname.includes('.modal.run')) {
        // Replace the 'web' app with 'TTSService'
        hostname = hostname.replace('-web', '-ttsservice-web');
        return `${currentURL.protocol}//${hostname}`;
    }
    
    // For local development or if we can't determine the URL, use the same origin
    return '';
};

// Function to warm up the GPU when the page loads
async function warmupGPU() {
    try {
        // Show warming up UI
        document.getElementById('warmingUpOverlay').classList.remove('hidden');
        document.getElementById('micContainer').classList.add('warming-up');
        document.getElementById('recordButton').disabled = true;
        
        const baseURL = getTTSServiceURL();
        const url = `${baseURL}/warmup`;
        
        console.log(`Warming up GPU, sending request to: ${url}`);
        
        const response = await fetch(url);
        const result = await response.json();
        
        console.log('GPU warmup result:', result);
        
        // After warmup completes, transition to ready state
        setTimeout(() => {
            // Hide warming up overlay with fade effect
            document.getElementById('warmingUpOverlay').classList.add('hidden');
            
            // Grow the mic button with animation
            const micContainer = document.getElementById('micContainer');
            micContainer.classList.remove('warming-up');
            micContainer.classList.add('ready');
            
            // Enable the record button
            document.getElementById('recordButton').disabled = false;
            
            isReady = true;
        }, 500); // Small delay for better visual effect
        
    } catch (error) {
        console.error('GPU warmup error:', error);
        // Even if there's an error, still show the UI
        document.getElementById('warmingUpOverlay').classList.add('hidden');
        document.getElementById('micContainer').classList.remove('warming-up');
        document.getElementById('micContainer').classList.add('ready');
        document.getElementById('recordButton').disabled = false;
        isReady = true;
    }
}

// Call warmup when the page loads
document.addEventListener('DOMContentLoaded', () => {
    // Initialize UI
    document.getElementById('warmingUpOverlay').classList.remove('hidden');
    
    // Start warmup
    warmupGPU();
});

async function setupRecorder() {
    console.log("Setting up recorder...");
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
    if (!isReady) return; // Prevent recording if not ready
    
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
    console.log("Processing audio...");
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.wav'); // Add filename to help the server

    try {
        // Get the base URL for the TTSService
        const baseURL = getTTSServiceURL();
        const url = `${baseURL}/process-audio`;
        
        console.log(`Sending request to: ${url}`);
        
        const response = await fetch(url, {
            method: 'POST',
            body: formData,
        });

        if (response.ok) {
            const audioBlob = await response.blob();
            displayResponse("", audioBlob);
            document.getElementById('recordingStatus').textContent = '';
        } else {
            const error = await response.text();
            console.error('Error response:', error);
            alert('Error processing audio: ' + error);
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