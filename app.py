import os
import uuid
import json
import shutil
import numpy as np
import soundfile as sf
import librosa
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.concurrency import run_in_threadpool
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Initialize FastAPI
app = FastAPI(title="Speech-to-Text & Speaker Diarization API")

# Configuration
TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# Global models cache to avoid reloading on every request
MODELS_CACHE = {}

def get_whisper_model(model_size: str):
    cache_key = f"whisper_{model_size}"
    if cache_key not in MODELS_CACHE:
        import whisper
        print(f"Loading Whisper model '{model_size}' into memory...")
        MODELS_CACHE[cache_key] = whisper.load_model(model_size)
    return MODELS_CACHE[cache_key]

def get_wespeaker_model():
    cache_key = "wespeaker"
    if cache_key not in MODELS_CACHE:
        from pyannote.audio import Model
        print("Loading WeSpeaker Embedding model into memory...")
        MODELS_CACHE[cache_key] = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
    return MODELS_CACHE[cache_key]


# Helper functions
def load_audio_resampled(file_path: str) -> np.ndarray:
    """Loads audio and resamples it to 16kHz mono using librosa/soundfile."""
    try:
        # librosa.load will automatically resample and convert to mono
        y, sr = librosa.load(file_path, sr=16000, mono=True)
        return y
    except Exception as e:
        print(f"Librosa load failed, falling back to soundfile: {e}")
        # Fallback to soundfile (useful for wav)
        data, sr = sf.read(file_path)
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)  # Mono conversion
        
        if sr != 16000:
            data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        return data

def run_whisper(y: np.ndarray, model_size: str) -> dict:
    """Runs Whisper transcription on 1D numpy array."""
    model = get_whisper_model(model_size)
    # Ensure FP32 since we run on CPU
    result = model.transcribe(y.astype(np.float32), fp16=False)
    return result

def run_wespeaker_clustering(y: np.ndarray, whisper_segments: list, num_speakers: int) -> list:
    """Extracts speaker embeddings from segments and clusters them via KMeans."""
    from pyannote.audio import Inference
    
    model = get_wespeaker_model()
    inference = Inference(model, window="whole")
    
    # Convert numpy array to torch tensor of shape (1, num_samples)
    waveform = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
    sample_rate = 16000
    
    embeddings = []
    valid_indices = []
    
    for idx, seg in enumerate(whisper_segments):
        start = seg["start"]
        end = seg["end"]
        
        # Calculate sample indices
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        
        segment_waveform = waveform[:, start_sample:end_sample]
        
        # WeSpeaker requires at least 0.5 seconds of audio to get clean embeddings
        min_samples = int(sample_rate * 0.5)
        if segment_waveform.shape[1] < min_samples:
            pad_len = min_samples - segment_waveform.shape[1]
            segment_waveform = torch.nn.functional.pad(segment_waveform, (0, pad_len))
            
        try:
            emb = inference({"waveform": segment_waveform, "sample_rate": sample_rate})
            embeddings.append(emb)
            valid_indices.append(idx)
        except Exception as e:
            print(f"Failed to extract speaker embedding for segment {idx} ({start:.2f}s - {end:.2f}s): {e}")
            
    if not embeddings:
        print("No embeddings could be extracted for any segments.")
        for seg in whisper_segments:
            seg["speaker"] = "SPEAKER_00"
        return whisper_segments
        
    embeddings_matrix = np.array(embeddings)
    num_segments = len(embeddings_matrix)
    
    # Auto-detect speakers if num_speakers is None or <= 0
    if num_speakers is None or num_speakers <= 0:
        if num_segments <= 1:
            num_speakers = 1
        else:
            max_speakers = min(10, num_segments - 1)
            if max_speakers < 2:
                num_speakers = 1
            else:
                best_score = -1
                best_k = 2
                for k in range(2, max_speakers + 1):
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
                    labels = kmeans.fit_predict(embeddings_matrix)
                    score = silhouette_score(embeddings_matrix, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
                num_speakers = best_k
                print(f"Auto-detected number of speakers: {num_speakers} (silhouette score: {best_score:.4f})")

    # Fit final clustering
    print(f"Running final KMeans clustering with {num_speakers} speaker(s)...")
    kmeans = KMeans(n_clusters=num_speakers, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(embeddings_matrix)
    
    # Map clustered segments to speakers
    label_map = {idx: f"SPEAKER_{labels[i]:02d}" for i, idx in enumerate(valid_indices)}
    
    for idx, seg in enumerate(whisper_segments):
        if idx in label_map:
            seg["speaker"] = label_map[idx]
        else:
            # Fallback to closest valid segment label
            closest_idx = min(valid_indices, key=lambda x: abs(x - idx)) if valid_indices else 0
            seg["speaker"] = label_map.get(closest_idx, "SPEAKER_00")
            
    return whisper_segments

def run_pyannote_pipeline(y: np.ndarray, hf_token: str, num_speakers: int) -> list:
    """Runs official PyAnnote Speaker Diarization pipeline."""
    from pyannote.audio import Pipeline
    
    print("Loading PyAnnote Speaker Diarization 3.1 pipeline...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )
    
    # Run on CPU
    pipeline.to(torch.device("cpu"))
    
    waveform = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
    sample_rate = 16000
    
    params = {}
    if num_speakers is not None and num_speakers > 0:
        params["num_speakers"] = num_speakers
        
    print("Executing PyAnnote Diarization pipeline...")
    diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, **params)
    
    diarization_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        diarization_segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })
    return diarization_segments

def align_whisper_pyannote(whisper_segments: list, diarization_segments: list) -> list:
    """Aligns Whisper transcription segments with PyAnnote diarization segments."""
    if not diarization_segments:
        print("PyAnnote diarization returned no speaker tracks. Defaulting to SPEAKER_00.")
        for seg in whisper_segments:
            seg["speaker"] = "SPEAKER_00"
        return whisper_segments
        
    for seg in whisper_segments:
        w_start = seg["start"]
        w_end = seg["end"]
        
        best_speaker = None
        max_overlap = 0.0
        
        for diag in diarization_segments:
            # Intersection of two segments
            overlap_start = max(w_start, diag["start"])
            overlap_end = min(w_end, diag["end"])
            overlap = max(0.0, overlap_end - overlap_start)
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = diag["speaker"]
                
        # If no overlap, assign to the closest segment
        if best_speaker is None:
            min_dist = float('inf')
            for diag in diarization_segments:
                dist = min(abs(w_start - diag["end"]), abs(w_end - diag["start"]))
                if dist < min_dist:
                    min_dist = dist
                    best_speaker = diag["speaker"]
                    
        seg["speaker"] = best_speaker or "SPEAKER_00"
        
    return whisper_segments

def sse_event(event: str, data: dict) -> str:
    """Format SSE response stream message."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# API Routes
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Receives and saves audio file temporarily, returning a task ID."""
    task_id = str(uuid.uuid4())
    # Retain file extension for parsing
    ext = os.path.splitext(file.filename)[1]
    # Default to .wav if empty
    if not ext:
        ext = ".wav"
    file_path = os.path.join(TEMP_DIR, f"{task_id}{ext}")
    
    print(f"Uploading file '{file.filename}' -> saved to '{file_path}' (Task: {task_id})")
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"task_id": task_id, "filename": file.filename, "extension": ext}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")


@app.get("/api/stream-progress/{task_id}")
async def stream_progress(
    task_id: str,
    extension: str = ".wav",
    whisper_model: str = "base",
    diarization_pipeline: str = "wespeaker",
    hf_token: str = "",
    num_speakers: int = 0
):
    """Streams transcription & diarization progress and sends final results via Server-Sent Events (SSE)."""
    
    async def event_generator():
        # Match temp file
        audio_path = os.path.join(TEMP_DIR, f"{task_id}{extension}")
        if not os.path.exists(audio_path):
            yield sse_event("error", {"message": "Audio file not found. Please upload again."})
            return
            
        try:
            # 1. Load Audio
            yield sse_event("progress", {"step": "Loading audio and resampling to 16kHz mono...", "percentage": 10})
            y = await run_in_threadpool(load_audio_resampled, audio_path)
            
            # 2. Whisper
            yield sse_event("progress", {
                "step": f"Transcribing audio with Whisper '{whisper_model}' model (FP32, CPU)...",
                "percentage": 30
            })
            whisper_result = await run_in_threadpool(run_whisper, y, whisper_model)
            segments = whisper_result.get("segments", [])
            
            if not segments:
                yield sse_event("progress", {"step": "No speech detected in audio.", "percentage": 50})
                yield sse_event("completed", {"segments": [], "text": ""})
                return
                
            # 3. Diarization
            yield sse_event("progress", {
                "step": f"Diarizing speakers using '{diarization_pipeline}' pipeline...",
                "percentage": 65
            })
            
            if diarization_pipeline == "pyannote":
                if not hf_token.strip():
                    raise ValueError("Hugging Face API Token is empty. It is required to download and use the official PyAnnote model.")
                
                # Run official pipeline
                diarization_segments = await run_in_threadpool(run_pyannote_pipeline, y, hf_token, num_speakers)
                
                # Align
                yield sse_event("progress", {
                    "step": "Aligning transcribing outputs with speaker tracks...",
                    "percentage": 85
                })
                final_segments = align_whisper_pyannote(segments, diarization_segments)
            else:
                # WeSpeaker offline clustering
                final_segments = await run_in_threadpool(run_wespeaker_clustering, y, segments, num_speakers)
                
            # Format output
            formatted_segments = []
            for seg in final_segments:
                formatted_segments.append({
                    "id": seg.get("id", 0),
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"].strip(),
                    "speaker": seg.get("speaker", "SPEAKER_00")
                })
                
            yield sse_event("progress", {"step": "Finalizing transcription results...", "percentage": 95})
            
            # Send completed event with payload
            yield sse_event("completed", {
                "segments": formatted_segments,
                "text": whisper_result.get("text", "").strip()
            })
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield sse_event("error", {"message": str(e)})
        finally:
            # Always delete temp file to avoid disk build-up
            if os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                    print(f"Cleaned up temp file: {audio_path}")
                except Exception as ex:
                    print(f"Failed to remove temp file '{audio_path}': {ex}")

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# Mount static folder
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    # Start the FastAPI server
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
