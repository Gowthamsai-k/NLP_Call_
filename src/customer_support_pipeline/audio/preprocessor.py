"""
Audio preprocessing, conversion, loading, and turn slicing.
"""

import os
import shutil
import subprocess
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf

try:
    import librosa
except (ImportError, ModuleNotFoundError):
    librosa = None

from ..core.config import (
    TARGET_SR,
    MIN_TURN_DURATION_S,
    SILENCE_THRESHOLD,
    SUPPORTED_AUDIO_EXTENSIONS,
)

logger = logging.getLogger(__name__)


def check_ffmpeg() -> bool:
    """Verifies that ffmpeg is installed."""
    return shutil.which("ffmpeg") is not None


def convert_to_clean_wav(input_path: str, output_path: str) -> str:
    """Converts audio file into mono, 16 kHz, 16-bit PCM WAV using ffmpeg."""
    input_p = Path(input_path).resolve()
    output_p = Path(output_path).resolve()
    output_p.parent.mkdir(parents=True, exist_ok=True)

    if not input_p.exists():
        raise FileNotFoundError(f"Input audio file not found: {input_p}")

    cmd = [
        "ffmpeg", "-y", "-i", str(input_p),
        "-vn", "-ac", "1", "-ar", str(TARGET_SR),
        "-c:a", "pcm_s16le", str(output_p),
    ]

    logger.info(f"Converting audio via FFmpeg: {input_p.name} -> {output_p.name}")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        stderr_msg = e.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"FFmpeg audio conversion failed: {stderr_msg}") from e

    return str(output_p)


def load_and_resample_audio(audio_path: str) -> Tuple[np.ndarray, int, float]:
    """Loads an audio file as float32 mono at 16 kHz."""
    data, original_sr = sf.read(audio_path, dtype="float32")

    if data.ndim > 1:
        data = np.mean(data, axis=1)

    if original_sr != TARGET_SR:
        if librosa is not None:
            data = librosa.resample(data, orig_sr=original_sr, target_sr=TARGET_SR)
        else:
            import math
            from scipy.signal import resample_poly
            gcd = math.gcd(int(original_sr), int(TARGET_SR))
            data = resample_poly(data, TARGET_SR // gcd, original_sr // gcd).astype(np.float32)

    audio_data = np.asarray(data, dtype=np.float32)
    duration = len(audio_data) / TARGET_SR
    return audio_data, TARGET_SR, duration


def slice_turn_audio(
    audio_data: np.ndarray,
    start_time: float,
    end_time: float,
    samplerate: int = TARGET_SR,
) -> Optional[np.ndarray]:
    """Slices audio for a diarized turn, validating duration and silence."""
    start_sample = max(0, int(round(start_time * samplerate)))
    end_sample = min(len(audio_data), int(round(end_time * samplerate)))

    if end_sample <= start_sample:
        return None

    turn_audio = np.asarray(audio_data[start_sample:end_sample], dtype=np.float32)

    if len(turn_audio) < int(MIN_TURN_DURATION_S * samplerate):
        return None

    if np.max(np.abs(turn_audio)) < SILENCE_THRESHOLD:
        return None

    return turn_audio


def find_audio_files(input_dir: str, recursive: bool = True) -> List[Path]:
    """Scans a directory for all supported audio files."""
    input_path = Path(input_dir).resolve()
    if not input_path.is_dir():
        raise NotADirectoryError(f"Provided path is not a directory: {input_path}")

    pattern = "**/*" if recursive else "*"
    found = [
        p for p in input_path.glob(pattern)
        if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    ]
    found.sort()
    return found
