let isRecording = false;
let isReady = false;
let currentRecorder = null;
let recorderStream = null;
let recorderReleaseTimeout = null;
let decoder = null;
let currentQuoteIndex = 0;
let progressInterval = null;
let quoteInterval = null;
const quoteDuration = 10000;

// Quotes from Alan Watts (or your current AI speaker)
let speakerQuotes = [
    {
        text: "The only way to make sense out of change is to plunge into it, move with it, and join the dance.",
        speaker: "Alan Watts"
    },
    {
        text: "We do not 'come into' this world; we come out of it, as leaves from a tree.",
        speaker: "Alan Watts"
    },
    {
        text: "The meaning of life is just to be alive. It is so plain and so obvious and so simple.",
        speaker: "Alan Watts"
    },
    {
        text: "You are a function of what the whole universe is doing in the same way that a wave is a function of what the whole ocean is doing.",
        speaker: "Alan Watts"
    },
    {
        text: "The art of living is neither careless drifting on the one hand nor fearful clinging on the other.",
        speaker: "Alan Watts"
    },
    {
        text: "Muddy water is best cleared by leaving it alone.",
        speaker: "Alan Watts"
    },
    {
        text: "A scholar tries to learn something everyday; a student of Buddhism tries to unlearn something daily.",
        speaker: "Alan Watts"
    },
    {
        text: "It's better to have a short life that is full of what you like doing, than a long life spent in a miserable way.",
        speaker: "Alan Watts"
    },
    {
        text: "For there is never anything but the present, and if one cannot live there, one cannot live anywhere.",
        speaker: "Alan Watts"
    },
];

// Fisher-Yates (Knuth) shuffle algorithm
function shuffleArray(array) {
    // Create a copy of the array to avoid modifying the original
    const shuffled = [...array];
    
    // Start from the last element and swap with a random element
    for (let i = shuffled.length - 1; i > 0; i--) {
        // Pick a random index from 0 to i
        const j = Math.floor(Math.random() * (i + 1));
        
        // Swap elements at i and j
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    
    return shuffled;
}

function isIOS() {
    return /iPad|iPhone|iPod/.test(navigator.userAgent) && !window.MSStream;
}

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
        const baseURL = getTTSServiceURL();
        const url = `${baseURL}/warmup`;
        
        console.log(`Warming up GPU, sending request to: ${url}`);
        
        const response = await fetch(url);
        const result = await response.json();
        
        console.log('GPU warmup result:', result);
        
        // Fill progress bar to 100% when API call completes
        stopProgressBar();
        document.getElementById('progressBar').style.width = '100%';
        
        // Add a delay to ensure users see the completed progress bar
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // After warmup completes, transition to ready state
        // Stop quote rotation and progress bar
        stopQuoteRotation();
        
        // Hide warming up overlay with fade effect
        document.getElementById('warmingUpOverlay').classList.add('hidden');
        document.getElementById('micContainer').classList.remove('hidden');
        
        // Grow the mic button with animation
        const micContainer = document.getElementById('micContainer');
        micContainer.classList.remove('warming-up');
        micContainer.classList.add('ready');
        
        // Enable the record button
        document.getElementById('recordButton').disabled = false;
        
        isReady = true;
    } catch (error) {
        console.error('GPU warmup error:', error);
        // Even if there's an error, still show the UI
        stopQuoteRotation();
        document.getElementById('warmingUpOverlay').classList.add('hidden');
        document.getElementById('micContainer').classList.remove('warming-up');
        document.getElementById('micContainer').classList.add('ready');
        document.getElementById('recordButton').disabled = false;
        isReady = true;
    }
}

// Function to start rotating through quotes
function startQuoteRotation() {
    // Start from the first quote in the shuffled list
    currentQuoteIndex = 0;
    updateQuote();
    
    // Rotate quotes every quoteDuration seconds, going through the shuffled list sequentially
    quoteInterval = setInterval(() => {
        // Move to the next quote in the shuffled list
        currentQuoteIndex = (currentQuoteIndex + 1) % speakerQuotes.length;
        updateQuote();
    }, quoteDuration);
}

// Function to stop quote rotation
function stopQuoteRotation() {
    if (quoteInterval) {
        clearInterval(quoteInterval);
        quoteInterval = null;
    }
}

function stopProgressBar() {
    if (progressInterval) {
        clearInterval(progressInterval);
        progressInterval = null;
    }
}

// Function to update the displayed quote
function updateQuote() {
    const quoteText = document.getElementById('quoteText');
    const quoteSpeaker = document.getElementById('quoteSpeaker');
    
    const currentQuote = speakerQuotes[currentQuoteIndex];
    
    quoteText.textContent = `"${currentQuote.text}"`;
    quoteSpeaker.textContent = `- ${currentQuote.speaker}`;
}

// Function to animate the progress bar
function startProgressBar(targetPercentage = 100) {
    const progressBar = document.getElementById('progressBar');
    let progress = 0;
    const totalTime = 45; // Total estimated time in seconds
    const updateInterval = 200; // Update every 200ms
    
    progressInterval = setInterval(() => {
        progress += (updateInterval / (totalTime * 1000)) * targetPercentage;
        if (progress > targetPercentage) progress = targetPercentage;
        
        progressBar.style.width = `${progress}%`;
    }, updateInterval);
}

// Call warmup when the page loads
document.addEventListener('DOMContentLoaded', () => {
    // Shuffle the quotes array once at page load
    speakerQuotes = shuffleArray(speakerQuotes);
    
    // Show warming up UI with IdolMinds title and quotes
    document.getElementById('warmingUpOverlay').classList.remove('hidden');
    document.getElementById('micContainer').classList.add('hidden');
    document.getElementById('recordButton').disabled = true;
    startQuoteRotation();
    startProgressBar(90);

    // Start warmup
    warmupGPU();
    initRecorderStream(); // Initialize recorder stream when page loads
    initializeDecoder(); // Initialize the Opus decoder
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
}

async function toggleRecording() {
    if (!isReady) return;

    const button = document.getElementById('recordButton');
    const micContainer = button.closest('.mic-container');
    const status = document.getElementById('recordingStatus');

    if (!isRecording) {
        if (isIOS()) {
            // Avoid adding UI latency here, since we need to create a new stream each time.
            console.log("Starting recording on iOS...");
            startRecording();
        } else {
            await startRecording();
        }
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
        // scheduleReleaseRecorder();
    }
}

async function startRecording() {
    if (isIOS() || !recorderStream) {
        // Would love to call this immediately after the stream is released, so that it's ready for next recording.
        // However it seems this doesn't play well with Safari on iOS. Possibly because the old stream is still being reused.
        // Seems to be safe here, because it's triggered by a user gesture.
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

        releaseRecorderStream();

        // On iOS, we can't create a usable stream here in advance, possibly because of the browser reusing the old stream.
        if (!isIOS()) {
            initRecorderStream();
        }
    };

    mediaRecorder.onerror = (event) => {
        console.error("MediaRecorder error:", event.error);
    };

    try {
        mediaRecorder.start();
        console.log("Recording started.");
        currentRecorder = mediaRecorder;
    } catch (err) {
        console.error("Failed to start recording:", err);
        alert("Failed to start recording. Please try again.");
    }
}

function stopRecording() {
    if (currentRecorder && currentRecorder.state !== "inactive") {
        console.log("Stopping recording...");
        currentRecorder.stop();
    }
}

async function initializeDecoder() {
    // Initialize the opus decoder
    try {
        decoder = new window["ogg-opus-decoder"].OggOpusDecoder();
        await decoder.ready;
        console.log("Ogg Opus decoder initialized");
    } catch (err) {
        console.error("Failed to initialize opus decoder:", err);
    }
}

// Decode Opus audio using WebAssembly Opus decoder (opus-decoder library)
// Requires opusDecoder to be initialized elsewhere with OpusDecoder WebAssembly loader
// async function decodeOpus(opusData, audioContext) {
//     if (!window.opusDecoder) {
//         throw new Error("OpusDecoder not initialized. Please load opus-decoder.js and its WASM file.");
//     }

//     const packets = [opusData.buffer];
//     const decoded = await window.opusDecoder.decode(packets);

//     const audioBuffer = audioContext.createBuffer(1, decoded.samples.length, decoded.sampleRate);
//     audioBuffer.getChannelData(0).set(decoded.samples);
//     return audioBuffer;
// }

async function validateOggOpus(arrayBuffer) {
    const view = new DataView(arrayBuffer);
    const oggHeader = new TextDecoder().decode(new Uint8Array(arrayBuffer, 0, 4));
    const opusHead = new TextDecoder().decode(new Uint8Array(arrayBuffer, 28, 8));
  
    return oggHeader === 'OggS' && opusHead === 'OpusHead';
  }

async function processAudio(audioBlob) {
    console.log("Processing audio with WebSocket...");
    const baseURL = getTTSServiceURL().replace(/^http/, "ws");
    const wsURL = `${baseURL}/ws`;
    const ws = new WebSocket(wsURL);

    // Show response section while processing
    const responseSection = document.getElementById('responseSection');
    responseSection.classList.remove('hidden');
    document.getElementById('recordingStatus').textContent = 'Contemplating...';

    // Playback setup
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    let scheduledEndTime = null;
    let audioFinishTimeout = null; // To track when audio finishes playing

    ws.binaryType = "arraybuffer";

    ws.onopen = async () => {
        console.log("WebSocket connected, sending audio blob...");
        try {
            const arrayBuffer = await audioBlob.arrayBuffer();
            ws.send(arrayBuffer);
        } catch (err) {
            console.error("Error sending audio:", err);
            document.getElementById('recordingStatus').textContent = 'Error sending audio';
        }
    };

    ws.onmessage = async (event) => {
        const view = new Uint8Array(event.data);
        const type = view[0];
        const payload = view.slice(1);

        if (type === 1) { // Audio chunk (Opus)
            try {
                // Change status to show we're playing audio
                document.getElementById('recordingStatus').textContent = '';
                document.getElementById('micContainer').classList.add('audio-playing');
                
                const { channelData, samplesDecoded, sampleRate } = await decoder.decode(new Uint8Array(payload));
                let audioData = channelData[0];
                
                if (samplesDecoded <= 0) {
                    console.log("Received empty audio data", audioData, samplesDecoded, sampleRate);
                    return;
                }

                console.log("Received audio data of length:", audioData.length, "samplesDecoded:", samplesDecoded, "sampleRate:", sampleRate);
                
                // Create an AudioBuffer from the decoded data
                const audioBuffer = audioContext.createBuffer(1, samplesDecoded, sampleRate);
                audioBuffer.copyToChannel(audioData, 0);
                
                // Create a source node for playback
                const source = audioContext.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(audioContext.destination);
                
                // Schedule playback based on the current playhead position
                const scheduledStartTime = Math.max(scheduledEndTime, audioContext.currentTime);
                scheduledEndTime = scheduledStartTime + audioBuffer.duration;

                source.start(scheduledStartTime);
                
                // Clear any existing timeout for audio finish
                if (audioFinishTimeout) {
                    clearTimeout(audioFinishTimeout);
                }
                
                // Set timeout to remove audio-playing class after the scheduled end time
                const timeUntilFinished = (scheduledEndTime - audioContext.currentTime) * 1000;
                audioFinishTimeout = setTimeout(() => {
                    document.getElementById('micContainer').classList.remove('audio-playing');
                    audioFinishTimeout = null;
                }, timeUntilFinished + 100); // Add 100ms buffer
            } catch (err) {
                console.error("Error decoding Opus audio:", err);
            }
        } 
    };

    ws.onerror = (event) => {
        console.error("WebSocket error:", event);
        // Only remove audio-playing class if no audio is scheduled to play
        if (!scheduledEndTime || audioContext.currentTime >= scheduledEndTime) {
            document.getElementById('micContainer').classList.remove('audio-playing');
        }
        setTimeout(() => {
            document.getElementById('recordingStatus').textContent = '';
        }, 3000);
    };

    ws.onclose = (event) => {
        console.log("WebSocket closed:", event.code, event.reason);
        // Don't remove audio-playing here - let it be handled by the timeout
        // that's scheduled after the last audio chunk finishes playing
        if (event.code !== 1000) {
            setTimeout(() => {
                document.getElementById('recordingStatus').textContent = '';
            }, 3000);
        }
    };
}

// async function processAudio(audioBlob) {
//     console.log("Processing audio...");
//     const formData = new FormData();
//     formData.append('audio', audioBlob, 'recording.wav');

//     try {
//         const baseURL = getTTSServiceURL();
//         const processURL = `${baseURL}/process-audio`;

//         // Show response section while processing
//         const responseSection = document.getElementById('responseSection');
//         responseSection.classList.remove('hidden');
//         document.getElementById('recordingStatus').textContent = 'Philosophizing...';

//         console.log(`Sending audio to: ${processURL}`);
//         const response = await fetch(processURL, {
//             method: 'POST',
//             body: formData,
//         });

//         if (!response.ok) {
//             const error = await response.text();
//             throw new Error('Server error: ' + error);
//         }

//         const { session_id } = await response.json();
//         console.log("Session ID:", session_id);

//         const streamURL = `${baseURL}/stream-audio/${session_id}`;
//         console.log("Streaming audio from:", streamURL);

//         // Set up audio element to stream the result
//         const audioResponse = document.getElementById('audioResponse');
//         audioResponse.src = streamURL;
//         audioResponse.classList.remove('hidden');

//         // Attempt autoplay
//         try {
//             await audioResponse.play();
//             document.getElementById('recordingStatus').textContent = '';
//             document.getElementById('micContainer').classList.add('audio-playing');

//             audioResponse.addEventListener('ended', () => {
//                 document.getElementById('micContainer').classList.remove('audio-playing');
//             });
//         } catch (err) {
//             console.warn("Autoplay blocked, waiting for user interaction:", err);

//             // Show tap-to-play overlay
//             const listenButton = document.getElementById('tapToListenButton');
//             listenButton.classList.remove('hidden');
//             document.getElementById('recordingStatus').textContent = '';

//             const resumePlayback = async () => {
//                 try {
//                     await audioResponse.play();
//                     listenButton.classList.add('hidden');
//                     document.getElementById('micContainer').classList.add('audio-playing');

//                     audioResponse.addEventListener('ended', () => {
//                         document.getElementById('micContainer').classList.remove('audio-playing');
//                     });
//                 } catch (e) {
//                     console.error("Playback still failed:", e);
//                     alert("Audio playback failed. Please try again.");
//                 }
//             };

//             listenButton.addEventListener('click', resumePlayback, { once: true });
//         }

//     } catch (error) {
//         console.error('Error processing audio:', error);
//         alert('An error occurred while processing your request.');
//         document.getElementById('recordingStatus').textContent = '';
//     }
// }

function releaseRecorderStream() {
    if (recorderStream) {
        console.log("Releasing recorder stream...");
        recorderStream.getTracks().forEach(track => track.stop());
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