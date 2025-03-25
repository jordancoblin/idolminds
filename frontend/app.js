let isRecording = false;
let isReady = false;
let currentRecorder = null;
let recorderStream = null;
let recorderReleaseTimeout = null;

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
        // Show warming up UI with IdolMinds title, spinner and text
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
    // Initialize UI - show IdolMinds title with spinner and text during warm-up
    document.getElementById('warmingUpOverlay').classList.remove('hidden');
    
    // Start warmup
    warmupGPU();
    initRecorderStream(); // Initialize recorder stream when page loads
});

// Initialize recorder stream to be ready for recording
async function initRecorderStream() {
    if (recorderStream) {
        console.log("Recorder stream already initialized");
        return;
    }
    recorderStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    tracks = recorderStream.getTracks();
    console.log("Initialized recorder stream: ", recorderStream);
    console.log("Tracks: ", tracks);
}

async function toggleRecording() {
    if (!isReady) return;

    const button = document.getElementById('recordButton');
    const micContainer = button.closest('.mic-container');
    const status = document.getElementById('recordingStatus');

    if (!isRecording) {
        // This should probably be called with await, but I want to avoid adding UI latency here.
        startRecording();
        isRecording = true;
        micContainer.classList.add('recording');
        status.textContent = 'Listening...';

        // Hide previous response
        const responseSection = document.getElementById('responseSection');
        responseSection.classList.add('hidden');
    } else {
        // Stop recording
        stopRecording();
        isRecording = false;
        micContainer.classList.remove('recording');
        status.textContent = 'Processing...';

        // scheduleReleaseRecorder();
    }
}

async function startRecording() {
    // const { mediaRecorder, stream } = await createRecorder();
    if (!recorderStream || !recorderStream.active) {
        // Would love to call this immediately after the stream is released, so that it's ready for next recording.
        // However it seems this doesn't play well with Safari on iOS. Possibly because the old stream is still being reused.
        // Seems to be safe here, because it's triggered by a user gesture.
        // As a workaround, we'll call startRecording without await, so that UI is unblocked.
        await initRecorderStream();
    }
    const mediaRecorder = new MediaRecorder(recorderStream);
    console.log("MediaRecorder initialized:", mediaRecorder);
    let audioChunks = [];

    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            audioChunks.push(event.data);
        }
    };

    mediaRecorder.onstop = () => {
        const mimeType = mediaRecorder.mimeType;
        const extension = mimeType.split("/")[1];
        const audioBlob = new Blob(audioChunks, { type: mimeType });

        console.log("Audio blob size:", audioBlob.size);

        if (audioBlob.size === 0) {
            alert("Empty audio recorded. Please try again.");
            document.getElementById('recordingStatus').textContent = '';
            return;
        }

        processAudio(audioBlob);

        // Stop the current stream and initialize a new one for next recording
        // This ensures we'll have a stream ready when user clicks record again
        releaseRecorderStream()
    };

    mediaRecorder.onerror = (event) => {
        console.error("MediaRecorder error:", event.error);
    };

    mediaRecorder.start();
    console.log("Recording started.");
    currentRecorder = mediaRecorder;
}

function stopRecording() {
    if (currentRecorder && currentRecorder.state !== "inactive") {
        console.log("Stopping recording...");
        currentRecorder.stop();
    }
}

async function processAudio(audioBlob) {
    console.log("Processing audio...");
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.wav');

    try {
        const baseURL = getTTSServiceURL();
        const processURL = `${baseURL}/process-audio`;

        // Show response section while processing
        const responseSection = document.getElementById('responseSection');
        responseSection.classList.remove('hidden');
        document.getElementById('recordingStatus').textContent = 'Philosophizing...';

        console.log(`Sending audio to: ${processURL}`);
        const response = await fetch(processURL, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const error = await response.text();
            throw new Error('Server error: ' + error);
        }

        const { session_id } = await response.json();
        console.log("Session ID:", session_id);

        const streamURL = `${baseURL}/stream-audio/${session_id}`;
        console.log("Streaming audio from:", streamURL);

        // Set up audio element to stream the result
        const audioResponse = document.getElementById('audioResponse');
        audioResponse.src = streamURL;
        audioResponse.classList.remove('hidden');

        // Attempt autoplay
        try {
            await audioResponse.play();
            document.getElementById('recordingStatus').textContent = '';
            document.getElementById('micContainer').classList.add('audio-playing');

            audioResponse.addEventListener('ended', () => {
                document.getElementById('micContainer').classList.remove('audio-playing');
            });
        } catch (err) {
            console.warn("Autoplay blocked, waiting for user interaction:", err);

            // Show tap-to-play overlay
            const listenButton = document.getElementById('tapToListenButton');
            listenButton.classList.remove('hidden');
            document.getElementById('recordingStatus').textContent = '';

            const resumePlayback = async () => {
                try {
                    await audioResponse.play();
                    listenButton.classList.add('hidden');
                    document.getElementById('micContainer').classList.add('audio-playing');

                    audioResponse.addEventListener('ended', () => {
                        document.getElementById('micContainer').classList.remove('audio-playing');
                    });
                } catch (e) {
                    console.error("Playback still failed:", e);
                    alert("Audio playback failed. Please try again.");
                }
            };

            listenButton.addEventListener('click', resumePlayback, { once: true });
        }

    } catch (error) {
        console.error('Error processing audio:', error);
        alert('An error occurred while processing your request.');
        document.getElementById('recordingStatus').textContent = '';
    }
}

function releaseRecorderStream() {
    if (recorderStream) {
        console.log("Releasing recorder stream...");
        recorderStream.getTracks().forEach(track => track.stop());
        console.log("stream track state: ", recorderStream.getAudioTracks()[0].readyState); // → "ended"
        recorderStream = null;
    }
}

// TODO: call this somewhere
function scheduleReleaseRecorder() {
    if (recorderReleaseTimeout) {
        clearTimeout(recorderReleaseTimeout);
    }
    recorderReleaseTimeout = setTimeout(() => {
        releaseRecorderStream();
    }, 30000); // 30 seconds of inactivity
}