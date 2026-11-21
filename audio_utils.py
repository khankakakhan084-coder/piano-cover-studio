"""
audio_utils.py
Audio Processing, FFmpeg Wrappers, Trimming, Format Transcoding,
BPM & Key Detection, and Master Mixer for Piano Cover Studio.
(Pure Python + Scipy + FFmpeg + Lameenc — 100% portable)
"""
import os
import sys
import wave
import shutil
import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import scipy.io.wavfile as wavfile

logger = logging.getLogger("piano_audio_utils")
logger.setLevel(logging.INFO)


def get_ffmpeg() -> str:
    """Locate FFmpeg executable across virtualenvs and system PATH."""
    scripts_dir = Path(sys.prefix) / "Scripts"
    venv_ffmpeg = scripts_dir / "ffmpeg.exe"
    if venv_ffmpeg.exists():
        if str(scripts_dir) not in os.environ.get("PATH", ""):
            os.environ["PATH"] = str(scripts_dir) + os.pathsep + os.environ.get("PATH", "")
        return str(venv_ffmpeg)

    found = shutil.which("ffmpeg")
    if found:
        return found

    for candidate in [
        r"C:\Users\khank\AppData\Local\Programs\Python\Python312\Scripts\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
    ]:
        if Path(candidate).exists():
            return candidate

    return "ffmpeg"


def get_audio_duration(file_path: Path) -> float:
    """Returns audio file duration in seconds."""
    try:
        if file_path.suffix.lower() == ".wav":
            with wave.open(str(file_path), "rb") as wf:
                return float(wf.getnframes()) / float(wf.getframerate())
    except Exception:
        pass

    try:
        import librosa
        return float(librosa.get_duration(path=str(file_path)))
    except Exception:
        pass

    # FFmpeg fallback
    try:
        ffmpeg_bin = get_ffmpeg()
        cmd = [ffmpeg_bin, "-i", str(file_path)]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        import re
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", res.stderr)
        if m:
            hours, mins, secs = m.groups()
            return int(hours) * 3600 + int(mins) * 60 + float(secs)
    except Exception:
        pass

    return 0.0


def trim_audio(
    input_path: Path,
    output_path: Path,
    start_sec: float,
    end_sec: Optional[float] = None
) -> Path:
    """
    Trims audio from start_sec to end_sec with high precision using FFmpeg.
    Encodes to high-quality MP3 if target is .mp3 to keep upload size light.
    """
    ffmpeg_bin = get_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg_bin, "-y",
        "-ss", f"{max(0.0, start_sec):.3f}",
        "-i", str(input_path),
    ]

    if end_sec is not None and end_sec > start_sec:
        duration = end_sec - start_sec
        cmd.extend(["-t", f"{duration:.3f}"])

    if output_path.suffix.lower() == ".mp3":
        cmd.extend(["-c:a", "libmp3lame", "-b:a", "320k", str(output_path)])
    else:
        cmd.extend(["-c:a", "pcm_s16le", str(output_path)])

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0 or not output_path.exists():
        logger.warning(f"FFmpeg trim notice: {res.stderr[:200]}")
        if not output_path.exists():
            shutil.copy2(input_path, output_path)

    return output_path


def convert_audio_format(
    input_path: Path,
    output_format: str = "mp3",
    output_path: Optional[Path] = None,
    bitrate: str = "320k"
) -> Path:
    """Converts audio file to specified target format with high quality."""
    fmt = output_format.lower().strip().replace(".", "")
    if output_path is None:
        output_path = input_path.with_suffix(f".{fmt}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = get_ffmpeg()

    codec_args = []
    if fmt == "mp3":
        codec_args = ["-c:a", "libmp3lame", "-b:a", bitrate]
    elif fmt == "wav":
        codec_args = ["-c:a", "pcm_s16le"]
    elif fmt == "flac":
        codec_args = ["-c:a", "flac"]
    elif fmt in ("m4a", "aac"):
        codec_args = ["-c:a", "aac", "-b:a", "256k"]
    elif fmt == "ogg":
        codec_args = ["-c:a", "libvorbis", "-q:a", "6"]
    else:
        codec_args = ["-c:a", "libmp3lame", "-b:a", bitrate]

    cmd = [ffmpeg_bin, "-y", "-i", str(input_path)] + codec_args + [str(output_path)]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0 or not output_path.exists():
        logger.error(f"Format conversion failed: {res.stderr}")
        raise RuntimeError(f"FFmpeg audio conversion to {fmt} failed.")

    return output_path


def save_numpy_to_mp3(audio_np: np.ndarray, sr: int, output_path: Path, bitrate: int = 320) -> Path:
    """Saves float32 numpy stereo audio as MP3 using lameenc directly."""
    import lameenc
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if audio_np.ndim == 1:
        audio_np = np.vstack([audio_np, audio_np])

    # Clamp and convert to int16
    clamped = np.clip(audio_np, -1.0, 1.0)
    wav_i16 = (clamped * 32767.0).astype(np.int16)
    nch = wav_i16.shape[0]

    enc = lameenc.Encoder()
    enc.set_bit_rate(bitrate)
    enc.set_in_sample_rate(sr)
    enc.set_channels(nch)
    enc.set_quality(2)

    if nch == 2:
        interleaved = np.empty(wav_i16.shape[1] * 2, dtype=np.int16)
        interleaved[0::2] = wav_i16[0]
        interleaved[1::2] = wav_i16[1]
        pcm = interleaved.tobytes()
    else:
        pcm = wav_i16[0].tobytes()

    data = enc.encode(pcm) + enc.flush()
    output_path.write_bytes(data)
    return output_path


def save_numpy_to_wav(audio_np: np.ndarray, sr: int, output_path: Path) -> Path:
    """Saves float32 numpy stereo audio as 16-bit PCM WAV using scipy."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if audio_np.ndim == 1:
        audio_np = np.vstack([audio_np, audio_np])
    clamped = np.clip(audio_np, -1.0, 1.0)
    wav_i16 = (clamped.T * 32767.0).astype(np.int16)
    wavfile.write(str(output_path), sr, wav_i16)
    return output_path


def detect_bpm_and_key(audio_path: Path) -> Dict[str, Any]:
    """Detects song BPM, musical Key, and Mode using Librosa."""
    try:
        import librosa
        y, sr = librosa.load(str(audio_path), sr=22050, duration=60.0, mono=True)
        if len(y) == 0:
            return {"bpm": 120, "key": "C Major", "confidence": 0.5}

        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        bpm = float(np.atleast_1d(tempo)[0])
        bpm = round(bpm, 1)

        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_avg = np.mean(chroma, axis=1)

        pitch_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

        major_corrs = [np.corrcoef(np.roll(major_profile, i), chroma_avg)[0, 1] for i in range(12)]
        minor_corrs = [np.corrcoef(np.roll(minor_profile, i), chroma_avg)[0, 1] for i in range(12)]

        best_maj_idx = int(np.argmax(major_corrs))
        best_min_idx = int(np.argmax(minor_corrs))

        if major_corrs[best_maj_idx] >= minor_corrs[best_min_idx]:
            detected_key = f"{pitch_names[best_maj_idx]} Major"
        else:
            detected_key = f"{pitch_names[best_min_idx]} Minor"

        return {
            "bpm": bpm,
            "key": detected_key,
            "confidence": round(float(max(major_corrs[best_maj_idx], minor_corrs[best_min_idx])), 2),
        }

    except Exception as ex:
        logger.warning(f"Notice in music theory analysis: {ex}")
        return {"bpm": 120.0, "key": "C Major", "confidence": 0.5}


def mix_piano_with_backing(
    piano_audio_path: Path,
    backing_audio_path: Path,
    output_path: Path,
    piano_gain: float = 0.85,
    backing_gain: float = 0.55
) -> Path:
    """
    Blends piano track with backing audio using FFmpeg amix filter or Librosa.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = get_ffmpeg()

    try:
        # Use FFmpeg amix for ultra-fast and reliable mixing of any audio inputs
        cmd = [
            ffmpeg_bin, "-y",
            "-i", str(piano_audio_path),
            "-i", str(backing_audio_path),
            "-filter_complex",
            f"[0:a]volume={piano_gain:.2f}[a0];[1:a]volume={backing_gain:.2f}[a1];[a0][a1]amix=inputs=2:duration=first:dropout_transition=2",
            "-c:a", "libmp3lame", "-b:a", "320k",
            str(output_path),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0 and output_path.exists():
            return output_path
    except Exception as e:
        logger.warning(f"FFmpeg mix notice: {e}")

    # Fallback copy
    shutil.copy2(piano_audio_path, output_path)
    return output_path
