# vad_api.py

import os
import tempfile
import subprocess
import requests
import urllib.parse
import shutil
import numpy as np
import soundfile as sf
import librosa
import torch
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import uvicorn
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = FastAPI(title="VAD Audio Analysis API", version="1.3")

DEBUG_AUDIO_DIR = os.path.join(tempfile.gettempdir(), "vad_debug_audio")
os.makedirs(DEBUG_AUDIO_DIR, exist_ok=True)
LAST_WAV_PATH = os.path.join(DEBUG_AUDIO_DIR, "last_converted.wav")


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
    hub_dir = os.path.expanduser("~/.cache/torch/hub")
    try:
        os.makedirs(hub_dir, exist_ok=True)
    except Exception:
        hub_dir = os.path.join(tempfile.gettempdir(), "torch_hub")
        os.makedirs(hub_dir, exist_ok=True)

    torch.hub.set_dir(hub_dir)

    try:
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
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
    audio = audio.astype(np.float32)
    max_val = np.max(np.abs(audio))
    if max_val > 1e-6:
        audio = audio / max_val * 0.95
    return audio


def process_audio_file(audio_path: str, target_sr: int = 16000):
    try:
        data, sr = sf.read(audio_path)
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
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
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    audio = audio / (np.max(np.abs(audio)) + 1e-6)
    window_size = int(sr * 0.025)
    hop_size = int(sr * 0.010)

    energy = []
    for i in range(0, len(audio) - window_size, hop_size):
        window = audio[i:i + window_size]
        energy.append(np.sqrt(np.mean(window ** 2)))
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
    if not speech_intervals:
        return {
            "talk_time": 0.0,
            "silence_time": round(total_duration, 2),
            "dead_air": round(total_duration, 2) if total_duration > dead_air_secs else 0.0,
            "longest_silence": round(total_duration, 2),
            "duration": round(total_duration, 2),
            "speech_segments": []
        }

    speech_intervals.sort(key=lambda x: x[0])
    merged = []
    for start, end in speech_intervals:
        if not merged or start > merged[-1][1] + 0.1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

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
    total_duration = len(audio) / sr

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

    speech_intervals = compute_speech_energy_based(audio, sr, threshold=0.02)
    return calculate_metrics_from_intervals(speech_intervals, total_duration, dead_air_secs)


def convert_audio_to_wav(input_path: str, output_path: str, target_sr: int = 16000) -> tuple:
    try:
        ffmpeg_cmd = "ffmpeg"
        ff = subprocess.run(
            [ffmpeg_cmd, "-y", "-i", input_path, "-acodec", "pcm_s16le", "-ar", str(target_sr),
             "-ac", "1", output_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        stderr_text = ff.stderr.decode(errors="ignore") if ff.stderr else ""
        ok = ff.returncode == 0 and os.path.exists(output_path)
        return ok, stderr_text
    except Exception as e:
        return False, f"FFmpeg conversion error: {e}"


def is_url(path: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(path)
        return parsed.scheme in ('http', 'https')
    except Exception:
        return False


def _is_private_ip(hostname: str) -> bool:
    """Detect LAN / private IPs so we can fail fast with a helpful message."""
    import ipaddress
    try:
        ip = ipaddress.ip_address(hostname)
        return ip.is_private or ip.is_loopback or ip.is_link_local
    except ValueError:
        return False


def download_audio_from_url(url: str, timeout: int = 60) -> tuple:
    try:
        parsed = urllib.parse.urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return None, f"Invalid URL: {url}"

        # Fail FAST and with a clear message if the host is a private LAN IP.
        if _is_private_ip(parsed.hostname or ""):
            return None, (
                f"'{parsed.hostname}' is a private/LAN IP address. A cloud-hosted server "
                f"(Render, AWS, etc.) CANNOT reach private IPs inside your local network. "
                f"You must either:\n"
                f"  1) Upload the file directly to POST /analyze_audio_upload, OR\n"
                f"  2) Expose the file via a public tunnel (Cloudflare Tunnel / ngrok), OR\n"
                f"  3) Deploy this API on a machine inside the same LAN."
            )

        filename = os.path.basename(parsed.path)
        if not filename or '.' not in filename:
            filename = 'audio.mp3'

        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"vad_download_{os.getpid()}_{filename}")

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        print(f"Downloading audio from: {url}")

        response = requests.get(
            url,
            headers=headers,
            timeout=(10, timeout),
            stream=True,
            verify=False
        )
        response.raise_for_status()

        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            print(f"Successfully downloaded {os.path.getsize(temp_path)} bytes to {temp_path}")
            return temp_path, ""
        else:
            return None, "Downloaded file is empty or not found"

    except requests.exceptions.Timeout as e:
        return None, f"Timeout connecting to {url}: {e}"
    except requests.exceptions.ConnectionError as e:
        return None, (
            f"Connection error reaching {url}: {e}. "
            f"If this host is a private/LAN IP (e.g. 192.168.x.x), a cloud-hosted server "
            f"(like Render) cannot reach it directly — it needs a public URL, a tunnel "
            f"(e.g. Cloudflare Tunnel/ngrok), or the file uploaded via /analyze_audio_upload instead."
        )
    except requests.exceptions.HTTPError as e:
        return None, f"HTTP error for {url}: {e}"
    except requests.exceptions.SSLError as e:
        return None, f"SSL error for {url}: {e}"
    except requests.exceptions.RequestException as e:
        return None, f"Request error for {url}: {e}"
    except Exception as e:
        return None, f"Unexpected error downloading {url}: {e}"


def get_audio_file_path(audio_path: str) -> tuple:
    if is_url(audio_path):
        temp_path, err = download_audio_from_url(audio_path, timeout=60)
        if temp_path is None:
            return None, False, err
        return temp_path, True, ""
    else:
        if not os.path.exists(audio_path):
            return None, False, f"Local file not found: {audio_path}"
        return audio_path, False, ""


# ---------- Shared processing logic ----------
def run_vad_pipeline(local_file_path: str, vad_threshold: float, dead_air_secs: float,
                     sample_rate: int) -> Dict:
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

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "audio_processed.wav")

        ok, ffmpeg_err = convert_audio_to_wav(local_file_path, wav_path, sample_rate)
        if not ok:
            result["error"] = f"Failed to convert audio to WAV format. ffmpeg said: {ffmpeg_err}"
            return result

        try:
            shutil.copyfile(wav_path, LAST_WAV_PATH)
        except Exception as copy_err:
            print(f"Could not copy debug wav: {copy_err}")

        audio_data, sr = process_audio_file(wav_path, sample_rate)
        if audio_data is None:
            result["error"] = "Failed to load audio data after conversion"
            return result

        vad_result = compute_vad_metrics(audio_data, sr, vad_threshold, dead_air_secs)

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


# ---------- FASTAPI ENDPOINTS ----------
@app.post("/analyze_audio", response_model=AudioAnalysisResponse)
async def analyze_audio_endpoint(
    audio_path: str,
    vad_threshold: float = 0.3,
    dead_air_secs: float = 5.0,
    sample_rate: int = 16000
):
    """
    Analyze audio file and return VAD metrics.

    NOTE: audio_path must be reachable FROM THIS SERVER. If this API is deployed
    on a cloud host (e.g. Render) and audio_path is a private LAN address
    (e.g. 192.168.x.x), it will NOT be reachable — use /analyze_audio_upload
    instead, or expose the file via a public URL / tunnel.
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
        file_path, is_temp_file, err = get_audio_file_path(audio_path)

        if file_path is None:
            result["error"] = err or f"Failed to resolve audio path: {audio_path}"
            return result

        local_file_path = file_path
        pipeline_result = run_vad_pipeline(local_file_path, vad_threshold, dead_air_secs, sample_rate)
        return pipeline_result

    except Exception as e:
        result["error"] = f"Unexpected error: {str(e)}"
        return result

    finally:
        if is_temp_file and local_file_path and os.path.exists(local_file_path):
            try:
                os.remove(local_file_path)
                print(f"Cleaned up temporary file: {local_file_path}")
            except Exception as e:
                print(f"Error cleaning up temp file: {e}")


@app.post("/analyze_audio_upload", response_model=AudioAnalysisResponse)
async def analyze_audio_upload_endpoint(
    file: UploadFile = File(...),
    vad_threshold: float = Form(0.3),
    dead_air_secs: float = Form(5.0),
    sample_rate: int = Form(16000)
):
    """
    Analyze an uploaded audio file directly (multipart/form-data).

    Use this when the audio lives on a private network the server can't reach
    (e.g. a LAN-only IP like 192.168.x.x while this API runs on Render/cloud).
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

    try:
        suffix = os.path.splitext(file.filename or "")[1] or ".mp3"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            if not content:
                result["error"] = "Uploaded file is empty"
                return result
            tmp.write(content)
            local_file_path = tmp.name

        try:
            pipeline_result = run_vad_pipeline(local_file_path, vad_threshold, dead_air_secs, sample_rate)
            return pipeline_result
        finally:
            if os.path.exists(local_file_path):
                os.remove(local_file_path)

    except Exception as e:
        result["error"] = f"Unexpected error: {str(e)}"
        return result


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "VAD Audio Analysis"}


@app.get("/debug/last_wav")
async def debug_last_wav():
    if not os.path.exists(LAST_WAV_PATH):
        raise HTTPException(
            status_code=404,
            detail="No converted WAV available yet. Run /analyze_audio or /analyze_audio_upload first."
        )
    from fastapi.responses import FileResponse
    return FileResponse(LAST_WAV_PATH, media_type="audio/wav", filename="last_converted.wav")


@app.post("/analyze_batch")
async def analyze_batch_endpoint(
    audio_paths: List[str],
    vad_threshold: float = 0.3,
    dead_air_secs: float = 5.0,
    sample_rate: int = 16000
):
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
