# vad_api.py

import os
import tempfile
import subprocess
import requests
import urllib.parse
import numpy as np
import soundfile as sf
import librosa
import torch
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="VAD Audio Analysis API", version="1.0")

# ---------- Pydantic Models ----------
class AudioAnalysisResponse(BaseModel):
    success: bool
    duration: float
    talk_time: float
    silence_time: float
    dead_air: float
    longest_silence: float
    speech_segments: List[List[float]]
    error: str = ""

# ---------- VAD Functions ----------
def load_silero_vad_model():
    """Load Silero VAD model using torch.hub with caching."""
    hub_dir = os.path.expanduser("~/.cache/torch/hub")
    try:
        os.makedirs(hub_dir, exist_ok=True)
    except Exception:
        hub_dir = os.path.join(tempfile.gettempdir(), "torch_hub")
        os.makedirs(hub_dir, exist_ok=True)
    
    torch.hub.set_dir(hub_dir)
    
    try:
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',  # Git hub se VAD le rha hai
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        get_speech_timestamps = utils[0]
        return model, get_speech_timestamps
    except Exception as e:
        print(f"Error loading VAD from torch.hub: {e}")
        return None, None

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to prevent clipping."""
    audio = audio.astype(np.float32)
    max_val = np.max(np.abs(audio))
    if max_val > 1e-6:
        audio = audio / max_val * 0.95
    return audio

def process_audio_file(audio_path: str, target_sr: int = 16000):
    """Load and preprocess audio file."""
    try:
        data, sr = sf.read(audio_path)
        
        # Convert to mono if stereo
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        
        # Resample if needed
        if sr != target_sr:
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        
        data = normalize_audio(data)
        return data, sr
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None, None

def compute_speech_energy_based(audio: np.ndarray, sr: int, threshold: float = 0.02, 
                                min_speech_duration: float = 0.1):
    """Fallback VAD using energy-based detection."""
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    audio = audio / (np.max(np.abs(audio)) + 1e-6)
    window_size = int(sr * 0.025)
    hop_size = int(sr * 0.010)
    
    energy = []
    for i in range(0, len(audio) - window_size, hop_size):
        window = audio[i:i+window_size]
        energy.append(np.sqrt(np.mean(window**2)))
    energy = np.array(energy)
    
    is_speech = energy > threshold
    min_frames = int(min_speech_duration * sr / hop_size)
    speech_intervals = []
    start = None
    
    for i, speech in enumerate(is_speech):
        if speech and start is None:
            start = i * hop_size / sr
        elif not speech and start is not None:
            end = i * hop_size / sr
            if end - start >= min_speech_duration:
                speech_intervals.append((start, end))
            start = None
    
    if start is not None:
        end = len(audio) / sr
        if end - start >= min_speech_duration:
            speech_intervals.append((start, end))
    
    return speech_intervals

def calculate_metrics_from_intervals(speech_intervals: List[tuple], total_duration: float, 
                                     dead_air_secs: float = 5.0) -> Dict:
    """Calculate metrics from speech intervals."""
    if not speech_intervals:
        return {
            "talk_time": 0.0,
            "silence_time": round(total_duration, 2),
            "dead_air": round(total_duration, 2) if total_duration > dead_air_secs else 0.0,
            "longest_silence": round(total_duration, 2),
            "duration": round(total_duration, 2),
            "speech_segments": []
        }
    
    # Merge overlapping speech segments
    speech_intervals.sort(key=lambda x: x[0])
    merged = []
    for start, end in speech_intervals:
        if not merged or start > merged[-1][1] + 0.1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    
    # Calculate metrics
    speech_time = 0.0
    longest_silence = 0.0
    dead_air = 0.0
    prev_end = 0.0
    
    for start, end in merged:
        speech_time += (end - start)
        silence = max(0.0, start - prev_end)
        longest_silence = max(longest_silence, silence)
        if silence > dead_air_secs:
            dead_air += silence
        prev_end = end
    
    ending_silence = max(0.0, total_duration - prev_end)
    longest_silence = max(longest_silence, ending_silence)
    if ending_silence > dead_air_secs:
        dead_air += ending_silence
    
    silence_time = max(0.0, total_duration - speech_time)
    
    return {
        "talk_time": round(speech_time, 2),
        "silence_time": round(silence_time, 2),
        "dead_air": round(dead_air, 2),
        "longest_silence": round(longest_silence, 2),
        "duration": round(total_duration, 2),
        "speech_segments": [(round(s, 2), round(e, 2)) for s, e in merged]
    }

def compute_vad_metrics(audio: np.ndarray, sr: int, threshold: float = 0.3, 
                        dead_air_secs: float = 5.0) -> Dict:
    """Compute VAD metrics using Silero VAD with fallback."""
    total_duration = len(audio) / sr
    
    # Try Silero VAD first
    try:
        model, get_speech_timestamps = load_silero_vad_model()
        if model is not None and get_speech_timestamps is not None:
            audio_tensor = torch.from_numpy(audio).float()
            vad_kwargs = {
                'sampling_rate': sr,
                'threshold': threshold,
                'min_speech_duration_ms': 250,
                'min_silence_duration_ms': 200,
                'speech_pad_ms': 400,
                'window_size_samples': 512,
            }
            speech_timestamps = get_speech_timestamps(audio_tensor, model, **vad_kwargs)
            
            speech_intervals = []
            for ts in speech_timestamps:
                start = ts['start'] / sr
                end = ts['end'] / sr
                speech_intervals.append((start, end))
            
            return calculate_metrics_from_intervals(speech_intervals, total_duration, dead_air_secs)
    except Exception as e:
        print(f"Silero VAD failed, using fallback: {e}")
    
    # Fallback to energy-based VAD
    speech_intervals = compute_speech_energy_based(audio, sr, threshold=0.02)
    return calculate_metrics_from_intervals(speech_intervals, total_duration, dead_air_secs)

def convert_audio_to_wav(input_path: str, output_path: str, target_sr: int = 16000) -> bool:
    """Convert audio to WAV format using ffmpeg."""
    try:
        ffmpeg_cmd = "ffmpeg"
        ff = subprocess.run(
            [ffmpeg_cmd, "-y", "-i", input_path, "-acodec", "pcm_s16le", "-ar", str(target_sr), 
             "-ac", "1", output_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        return ff.returncode == 0 and os.path.exists(output_path)
    except Exception as e:
        print(f"FFmpeg conversion error: {e}")
        return False

def is_url(path: str) -> bool:
    """Check if the given path is a URL."""
    try:
        parsed = urllib.parse.urlparse(path)
        return parsed.scheme in ('http', 'https')
    except Exception:
        return False

def download_audio_from_url(url: str, timeout: int = 60) -> Optional[str]:
    """
    Download audio from URL and save to temporary file.
    
    Args:
        url: HTTP/HTTPS URL of the audio file
        timeout: Download timeout in seconds (increased to 60)
        
    Returns:
        Path to downloaded temporary file, or None if download fails
    """
    try:
        # Validate URL
        parsed = urllib.parse.urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid URL: {url}")
        
        # Get filename from URL to preserve extension
        filename = os.path.basename(parsed.path)
        if not filename or '.' not in filename:
            filename = 'audio.mp3'
        
        # Create temporary file with original extension
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"vad_download_{os.getpid()}_{filename}")
        
        # Download with streaming and increased timeout
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        print(f"Downloading audio from: {url}")
        print(f"Timeout set to: {timeout} seconds")
        
        # Using a longer timeout with connection and read timeouts separately
        response = requests.get(
            url, 
            headers=headers, 
            timeout=(10, timeout),  # (connection_timeout, read_timeout)
            stream=True,
            verify=False  # If SSL certificate issues, you can set this to False
        )
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('content-type', '')
        print(f"Content-Type: {content_type}")
        
        # Download with progress tracking
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
        
        # Verify file was downloaded
        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            file_size = os.path.getsize(temp_path)
            print(f"Successfully downloaded {file_size} bytes to {temp_path}")
            return temp_path
        else:
            raise Exception("Downloaded file is empty or not found")
            
    except requests.exceptions.Timeout as e:
        print(f"Download timeout for {url}: {e}")
        return None
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error for {url}: {e}")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error for {url}: {e}")
        return None
    except requests.exceptions.SSLError as e:
        print(f"SSL error for {url}: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request error for {url}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error downloading {url}: {e}")
        return None

def get_audio_file_path(audio_path: str) -> tuple:
    """
    Get audio file path, handling both local files and URLs.
    
    Args:
        audio_path: Local file path or URL
        
    Returns:
        Tuple of (file_path, is_temp) where is_temp indicates if file should be cleaned up
    """
    if is_url(audio_path):
        # Download URL and return temp file path
        temp_path = download_audio_from_url(audio_path, timeout=60)  # Increased timeout
        if temp_path is None:
            return None, False
        return temp_path, True
    else:
        # Local file path
        return audio_path, False

# ---------- FASTAPI ENDPOINT ----------
@app.post("/analyze_audio", response_model=AudioAnalysisResponse)
async def analyze_audio_endpoint(
    audio_path: str,
    vad_threshold: float = 0.3,
    dead_air_secs: float = 5.0,
    sample_rate: int = 16000
):
    """
    Analyze audio file and return VAD metrics.
    
    Input:
        - audio_path: str - Path to audio file on server OR HTTP/HTTPS URL
        - vad_threshold: float - VAD sensitivity (0.1-0.5, default: 0.3)
        - dead_air_secs: float - Silence threshold for dead air (default: 5.0)
        - sample_rate: int - Target sample rate (default: 16000)
    
    Output:
        {
            "success": bool,
            "duration": float,
            "talk_time": float,
            "silence_time": float,
            "dead_air": float,
            "longest_silence": float,
            "speech_segments": [[start, end], ...],
            "error": str
        }
    """
    result = {
        "success": False,
        "duration": 0.0,
        "talk_time": 0.0,
        "silence_time": 0.0,
        "dead_air": 0.0,
        "longest_silence": 0.0,
        "speech_segments": [],
        "error": ""
    }
    
    local_file_path = None
    is_temp_file = False
    
    try:
        # Get audio file path (local or downloaded from URL)
        file_path, is_temp_file = get_audio_file_path(audio_path)
        
        if file_path is None:
            result["error"] = f"Failed to download audio from: {audio_path}. Check network connectivity and server availability."
            return result
        
        # Check if file exists (for local files) or was downloaded
        if not os.path.exists(file_path):
            if is_temp_file:
                result["error"] = f"Downloaded file not found: {file_path}"
            else:
                result["error"] = f"Audio file not found: {audio_path}"
            return result
        
        local_file_path = file_path
        
        # Create temp directory for processing
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, "audio_processed.wav")
            
            # Convert audio to WAV
            if not convert_audio_to_wav(local_file_path, wav_path, sample_rate):
                result["error"] = "Failed to convert audio to WAV format"
                return result
            
            # Load and process audio
            audio_data, sr = process_audio_file(wav_path, sample_rate)
            if audio_data is None:
                result["error"] = "Failed to load audio data"
                return result
            
            # VAD Analysis
            vad_result = compute_vad_metrics(audio_data, sr, vad_threshold, dead_air_secs)
            
            # Update result
            result.update({
                "success": True,
                "duration": vad_result["duration"],
                "talk_time": vad_result["talk_time"],
                "silence_time": vad_result["silence_time"],
                "dead_air": vad_result["dead_air"],
                "longest_silence": vad_result["longest_silence"],
                "speech_segments": vad_result["speech_segments"]
            })
        
        return result
        
    except Exception as e:
        result["error"] = f"Unexpected error: {str(e)}"
        return result
    
    finally:
        # Clean up temporary downloaded file
        if is_temp_file and local_file_path and os.path.exists(local_file_path):
            try:
                os.remove(local_file_path)
                print(f"Cleaned up temporary file: {local_file_path}")
            except Exception as e:
                print(f"Error cleaning up temp file: {e}")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "VAD Audio Analysis"}


@app.post("/analyze_batch")
async def analyze_batch_endpoint(
    audio_paths: List[str],
    vad_threshold: float = 0.3,
    dead_air_secs: float = 5.0,
    sample_rate: int = 16000
):
    """
    Analyze multiple audio files in batch.
    
    Input:
        - audio_paths: List[str] - List of audio file paths or URLs
        - vad_threshold: float - VAD sensitivity
        - dead_air_secs: float - Silence threshold for dead air
        - sample_rate: int - Target sample rate
    
    Output:
        List of VAD analysis results for each file
    """
    results = []
    for path in audio_paths:
        result = await analyze_audio_endpoint(
            audio_path=path,
            vad_threshold=vad_threshold,
            dead_air_secs=dead_air_secs,
            sample_rate=sample_rate
        )
        results.append({
            "file": path,
            "result": result
        })
    return {"results": results}


# ---------- RUN SERVER ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)