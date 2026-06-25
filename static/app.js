// Global States
let selectedFile = null;
let transcriptSegments = [];
let activeEventSource = null;
let playerUpdateInterval = null;

// DOM Elements
const form = document.getElementById('transcribe-form');
const dropzone = document.getElementById('audio-dropzone');
const fileInput = document.getElementById('audio-file');
const fileInfo = document.getElementById('file-info');
const selectedFileName = document.getElementById('selected-file-name');
const selectedFileSize = document.getElementById('selected-file-size');
const removeFileBtn = document.getElementById('remove-file-btn');
const submitBtn = document.getElementById('submit-btn');

const whisperModel = document.getElementById('whisper-model');
const diarizationPipeline = document.getElementsByName('diarization_pipeline');
const hfTokenGroup = document.getElementById('hf-token-group');
const hfToken = document.getElementById('hf-token');
const autoSpeakers = document.getElementById('auto-speakers');
const speakerSliderContainer = document.getElementById('speaker-slider-container');
const numSpeakers = document.getElementById('num-speakers');
const speakerValDisplay = document.getElementById('speaker-val-display');

const idleState = document.getElementById('idle-state');
const progressState = document.getElementById('progress-state');
const resultState = document.getElementById('result-state');

const progressCircle = document.getElementById('progress-circle');
const progressPercent = document.getElementById('progress-percent');
const currentStepTitle = document.getElementById('current-step-title');
const logTerminal = document.getElementById('log-terminal');

const searchInput = document.getElementById('search-transcript');
const exportCsvBtn = document.getElementById('export-csv');
const exportJsonBtn = document.getElementById('export-json');
const resetAppBtn = document.getElementById('reset-app');

const playerFileName = document.getElementById('player-file-name');
const playerTimeDisplay = document.getElementById('player-time-display');
const mainAudioPlayer = document.getElementById('main-audio-player');
const transcriptTimeline = document.getElementById('transcript-timeline');

// Initialization
document.addEventListener('DOMContentLoaded', () => {
    lucide.createIcons();
    initEventListeners();
});

function initEventListeners() {
    // Dropzone Events
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    removeFileBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        clearSelectedFile();
    });

    // Control Panel Panel Bindings
    diarizationPipeline.forEach(radio => {
        radio.addEventListener('change', (e) => {
            document.querySelectorAll('.radio-label').forEach(label => {
                label.classList.remove('active');
            });
            const selectedLabel = document.getElementById(`label-${e.target.value}`);
            if (selectedLabel) selectedLabel.classList.add('active');

            if (e.target.value === 'pyannote') {
                hfTokenGroup.classList.remove('hidden');
                hfToken.required = true;
            } else {
                hfTokenGroup.classList.add('hidden');
                hfToken.required = false;
            }
        });
    });

    autoSpeakers.addEventListener('change', (e) => {
        if (e.target.checked) {
            speakerSliderContainer.classList.add('disabled');
            numSpeakers.disabled = true;
        } else {
            speakerSliderContainer.classList.remove('disabled');
            numSpeakers.disabled = false;
        }
    });

    numSpeakers.addEventListener('input', (e) => {
        speakerValDisplay.textContent = e.target.value;
    });

    // Submit Action
    form.addEventListener('submit', handleFormSubmit);

    // Result Actions
    searchInput.addEventListener('input', handleTranscriptSearch);
    exportCsvBtn.addEventListener('click', exportToCSV);
    exportJsonBtn.addEventListener('click', exportToJSON);
    resetAppBtn.addEventListener('click', resetDashboard);
}

// File Selection Controllers
function handleFileSelect(file) {
    if (!file.type.startsWith('audio/') && !file.name.endsWith('.wav') && !file.name.endsWith('.mp3') && !file.name.endsWith('.flac') && !file.name.endsWith('.ogg')) {
        alert('Invalid file format. Please upload an audio file (WAV, FLAC, OGG, or MP3).');
        return;
    }

    selectedFile = file;
    selectedFileName.textContent = file.name;
    selectedFileSize.textContent = formatBytes(file.size);

    // Toggle dropzone state
    dropzone.querySelector('.dropzone-content').classList.add('hidden');
    fileInfo.classList.remove('hidden');

    submitBtn.disabled = false;
}

function clearSelectedFile() {
    selectedFile = null;
    fileInput.value = '';

    // Toggle dropzone state
    dropzone.querySelector('.dropzone-content').classList.remove('hidden');
    fileInfo.classList.add('hidden');

    submitBtn.disabled = true;
}

// Form Submission & SSE Handler
async function handleFormSubmit(e) {
    e.preventDefault();
    if (!selectedFile) return;

    // Show Progress State
    switchViewState('progress');
    clearTerminal();
    updateProgressRing(0);
    currentStepTitle.textContent = "Uploading audio file...";
    addLogLine("Uploading audio to backend...", "system");

    try {
        // Upload File first
        const uploadData = new FormData();
        uploadData.append('file', selectedFile);

        const uploadResponse = await fetch('/api/upload', {
            method: 'POST',
            body: uploadData
        });

        if (!uploadResponse.ok) {
            const err = await uploadResponse.json();
            throw new Error(err.detail || 'Failed to upload audio file.');
        }

        const uploadResult = await uploadResponse.json();
        const taskId = uploadResult.task_id;
        const extension = uploadResult.extension;

        addLogLine(`Upload completed. Task ID: ${taskId}`, "success");
        addLogLine("Establishing server-sent events (SSE) progress connection...", "system");

        // Parse configurations
        const selectedModel = whisperModel.value;
        const pipeline = Array.from(diarizationPipeline).find(r => r.checked).value;
        const token = hfToken.value;
        const speakers = autoSpeakers.checked ? 0 : numSpeakers.value;

        // Establish SSE
        const sseUrl = `/api/stream-progress/${taskId}?extension=${encodeURIComponent(extension)}&whisper_model=${selectedModel}&diarization_pipeline=${pipeline}&hf_token=${encodeURIComponent(token)}&num_speakers=${speakers}`;

        activeEventSource = new EventSource(sseUrl);

        activeEventSource.addEventListener('progress', (event) => {
            const payload = JSON.parse(event.data);
            updateProgressRing(payload.percentage);
            currentStepTitle.textContent = payload.step;
            addLogLine(payload.step, "system");
        });

        activeEventSource.addEventListener('completed', (event) => {
            const payload = JSON.parse(event.data);
            addLogLine("Processing pipeline finished successfully!", "success");

            transcriptSegments = payload.segments;
            renderTranscript(transcriptSegments);

            // Set up audio player using local object URL for instant, zero-bandwidth streaming
            const localAudioUrl = URL.createObjectURL(selectedFile);
            mainAudioPlayer.src = localAudioUrl;
            playerFileName.textContent = selectedFile.name;

            // Close EventSource
            cleanupSSE();

            // Transition to Results State
            setTimeout(() => {
                switchViewState('result');
                initAudioSync();
            }, 600);
        });

        activeEventSource.addEventListener('error', (event) => {
            let message = "An error occurred during server execution.";
            if (event.data) {
                try {
                    const payload = JSON.parse(event.data);
                    message = payload.message || message;
                } catch (e) { }
            }

            addLogLine(`Error: ${message}`, "error");
            alert(`Execution failed: ${message}`);
            cleanupSSE();
            switchViewState('idle');
        });

    } catch (error) {
        addLogLine(`Failed: ${error.message}`, "error");
        alert(`Failed to execute: ${error.message}`);
        switchViewState('idle');
    }
}

// Audio Player & Transcript Synchronization
function initAudioSync() {
    mainAudioPlayer.addEventListener('loadedmetadata', updateTimeDisplay);
    mainAudioPlayer.addEventListener('timeupdate', updateTimeDisplay);

    // Highlight currently playing segment
    playerUpdateInterval = setInterval(() => {
        if (mainAudioPlayer.paused) return;

        const currentTime = mainAudioPlayer.currentTime;
        let activeIndex = -1;

        for (let i = 0; i < transcriptSegments.length; i++) {
            const seg = transcriptSegments[i];
            if (currentTime >= seg.start && currentTime <= seg.end) {
                activeIndex = i;
                break;
            }
        }

        // Update active class
        const segmentElements = document.querySelectorAll('.transcript-segment');
        segmentElements.forEach((el, index) => {
            if (index === activeIndex) {
                if (!el.classList.contains('active-segment')) {
                    el.classList.add('active-segment');
                    // Scroll active segment into view smoothly if needed
                    el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }
            } else {
                el.classList.remove('active-segment');
            }
        });
    }, 200);
}

function updateTimeDisplay() {
    const current = formatTime(mainAudioPlayer.currentTime);
    const duration = formatTime(mainAudioPlayer.duration || 0);
    playerTimeDisplay.textContent = `${current} / ${duration}`;
}

// Render Transcript segments
function renderTranscript(segments, filterTerm = '') {
    transcriptTimeline.innerHTML = '';

    if (segments.length === 0) {
        transcriptTimeline.innerHTML = '<div class="segment-text" style="text-align: center; padding: 2rem;">No transcribed text found.</div>';
        return;
    }

    segments.forEach(seg => {
        // Apply text filter if any
        if (filterTerm && !seg.text.toLowerCase().includes(filterTerm.toLowerCase())) {
            return;
        }

        const segmentEl = document.createElement('div');
        segmentEl.className = 'transcript-segment';

        // Extract speaker number for coloring badge class
        const speakerNum = parseInt(seg.speaker.replace(/\D/g, '')) || 0;
        const speakerBadgeClass = `spk-${speakerNum % 10}`;

        let textToShow = seg.text;
        if (filterTerm) {
            // Highlight matching search query
            const regex = new RegExp(`(${escapeRegExp(filterTerm)})`, 'gi');
            textToShow = seg.text.replace(regex, '<span class="highlight">$1</span>');
        }

        segmentEl.innerHTML = `
            <div class="segment-meta">
                <span class="speaker-badge ${speakerBadgeClass}">${seg.speaker}</span>
                <span class="segment-time">${formatTime(seg.start)} - ${formatTime(seg.end)}</span>
            </div>
            <p class="segment-text">${textToShow}</p>
        `;

        // Set jump duration on click
        segmentEl.addEventListener('click', () => {
            mainAudioPlayer.currentTime = seg.start;
            if (mainAudioPlayer.paused) {
                mainAudioPlayer.play();
            }
        });

        transcriptTimeline.appendChild(segmentEl);
    });
}

function handleTranscriptSearch(e) {
    renderTranscript(transcriptSegments, e.target.value);
}

// Exports
function exportToCSV() {
    if (transcriptSegments.length === 0) return;

    let csvContent = "data:text/csv;charset=utf-8,StartTime,EndTime,Speaker,Text\n";
    transcriptSegments.forEach(seg => {
        const textEscaped = seg.text.replace(/"/g, '""');
        csvContent += `${seg.start.toFixed(2)},${seg.end.toFixed(2)},"${seg.speaker}","${textEscaped}"\n`;
    });

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    const name = selectedFile ? selectedFile.name.replace(/\.[^/.]+$/, "") : "transcript";
    link.setAttribute("download", `${name}_transcript.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function exportToJSON() {
    if (transcriptSegments.length === 0) return;

    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(transcriptSegments, null, 2));
    const link = document.createElement("a");
    link.setAttribute("href", dataStr);
    const name = selectedFile ? selectedFile.name.replace(/\.[^/.]+$/, "") : "transcript";
    link.setAttribute("download", `${name}_transcript.json`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// UI Switch Views
function switchViewState(state) {
    idleState.classList.add('hidden');
    progressState.classList.add('hidden');
    resultState.classList.add('hidden');

    if (state === 'idle') {
        idleState.classList.remove('hidden');
    } else if (state === 'progress') {
        progressState.classList.remove('hidden');
    } else if (state === 'result') {
        resultState.classList.remove('hidden');
    }
}

function updateProgressRing(percent) {
    progressPercent.textContent = `${percent}%`;
    const circumference = 2 * Math.PI * 45; // ~282.74
    const offset = circumference - (percent / 100) * circumference;
    progressCircle.style.strokeDashoffset = offset;
}

function addLogLine(text, type = 'system') {
    const line = document.createElement('div');
    line.className = `log-line ${type}`;

    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    line.innerHTML = `<span class="log-time" style="color: var(--text-muted); margin-right: 0.5rem;">[${timestamp}]</span> ${text}`;

    logTerminal.appendChild(line);
    logTerminal.scrollTop = logTerminal.scrollHeight;
}

function clearTerminal() {
    logTerminal.innerHTML = '';
}

function cleanupSSE() {
    if (activeEventSource) {
        activeEventSource.close();
        activeEventSource = null;
    }
}

function resetDashboard() {
    cleanupSSE();
    if (playerUpdateInterval) {
        clearInterval(playerUpdateInterval);
        playerUpdateInterval = null;
    }

    mainAudioPlayer.pause();
    mainAudioPlayer.src = '';

    clearSelectedFile();
    searchInput.value = '';
    transcriptSegments = [];

    switchViewState('idle');
}

// Helper functions
function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

function formatTime(seconds) {
    if (isNaN(seconds)) return '00:00';
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);

    let result = '';
    if (hrs > 0) {
        result += `${hrs.toString().padStart(2, '0')}:`;
    }
    result += `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    return result;
}

function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
