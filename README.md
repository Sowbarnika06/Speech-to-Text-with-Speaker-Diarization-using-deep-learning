# Speech-to-Text-with-Speaker-Diarization-using-deep-learning
# VoxScribe AI - Local Speech-to-Text & Speaker Diarization
VoxScribe AI is a high-performance, locally-run web application designed for Speech-to-Text (STT) transcription and Speaker Diarization (SD) using state-of-the-art deep learning models. It runs entirely on your local machine, ensuring complete privacy for your audio recordings.
![VoxScribe AI Interface](screenshot.png)
## Key Features
- **Local Deep Learning Speech-to-Text**: Utilizes OpenAI's Whisper model (supporting sizes from `tiny` to `large`) running locally on CPU.
- **Dual Speaker Diarization Pipelines**:
  - **WeSpeaker Embedding Clustering (No Hugging Face Token Required)**: Extracts speaker characteristics using PyAnnote's non-gated `wespeaker-voxceleb-resnet34-LM` model and clusters segments using Scikit-Learn's KMeans. Works completely offline out-of-the-box.
  - **PyAnnote Pipeline (Hugging Face Token Required)**: Leverages the official gated `pyannote/speaker-diarization-3.1` pipeline.
- **Auto-Detect Speaker Count**: Intelligently estimates the number of unique speakers in the audio, or allows manual override via a slider.
- **Sleek, Premium UI**: Modern glassmorphic dark interface with neon blue-to-pink gradient accents, featuring:
  - Drag-and-drop file upload with a clear overview.
  - Live progress feedback at each step of transcription and diarization.
  - Interactive transcript player with color-coded speaker badges and synchronized timestamps.
  - Transcript search filter to find spoken phrases instantly.
  - Fast exports to CSV and JSON formats.
## Prerequisites
Before running the application, make sure you have:
1. **Python 3.10+** (Python 3.13 is fully supported).
2. **Pip** or **uv** installed.
No command-line `ffmpeg` executable is required; VoxScribe AI uses native Python libraries (`soundfile` and `librosa` with Windows Media Foundation fallbacks) to read and decode a wide variety of audio files including WAV, MP3, FLAC, and OGG.
## Installation
1. Clone or download this project to your local directory.
2. Install the required Python packages:
   ```bash
   pip install fastapi uvicorn openai-whisper pyannote-audio scikit-learn numpy soundfile librosa httpx jinja2
   ```
## Getting Started
1. Start the FastAPI backend server:
   ```bash
   python app.py
   ```
2. Open your web browser and navigate to:
   ```
   http://localhost:8000
   ```
3. Upload an audio file, configure the Whisper model size and diarization settings, and click **Transcribe & Diarize**!
## Advanced: Using PyAnnote Pipeline with Hugging Face
If you want to use the official PyAnnote pipeline:
1. Visit [Hugging Face](https://huggingface.co/) and create an account.
2. Accept the user conditions for:
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
3. Generate an Access Token at [HF Token Settings](https://huggingface.co/settings/tokens).
4. Select the **PyAnnote Pipeline** in the VoxScribe AI configuration panel and paste your Hugging Face Token.
