"""
Speaker diarization using pyannote/speaker-diarization-community-1.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import soundfile as sf
import torch

from ..core.config import DIARIZATION_MODEL_ID
from ..core.memory import release_memory, log_vram

logger = logging.getLogger(__name__)


def run_diarization(
    audio_path: str,
    output_metadata_path: str,
    hf_token: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Executes speaker diarization on clean WAV audio and writes call_metadata.json."""
    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        try:
            from google.colab import userdata
            token = userdata.get("HF_TOKEN")
        except Exception:
            pass

    if not token:
        raise ValueError(
            "Hugging Face token not found. Please provide hf_token or set the HF_TOKEN environment variable."
        )

    _actual_cuda_version = getattr(torch.version, "cuda", None)
    torch.version.cuda = "13.0"
    try:
        from pyannote.audio import Pipeline
    finally:
        torch.version.cuda = _actual_cuda_version

    try:
        import numpy._core.strings as _np_strings
        _np_strings._override___module__ = lambda: None
    except Exception:
        pass

    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Running Diarization on device: {target_device}")

    original_cudnn_state = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False

    pipeline = None
    result = None
    annotation = None

    try:
        log_vram("Before Diarization")
        pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL_ID, token=token)
        pipeline.to(target_device)

        with torch.inference_mode():
            result = pipeline(audio_path)

        annotation = getattr(result, "speaker_diarization", result)

        segments = []
        if hasattr(annotation, "itertracks"):
            for turn, _, speaker in annotation.itertracks(yield_label=True):
                segments.append({
                    "start": round(float(turn.start), 3),
                    "end": round(float(turn.end), 3),
                    "speaker_id": str(speaker),
                    "text": None,
                })
        else:
            for turn, speaker in annotation:
                segments.append({
                    "start": round(float(turn.start), 3),
                    "end": round(float(turn.end), 3),
                    "speaker_id": str(speaker),
                    "text": None,
                })

        info = sf.info(audio_path)
        speaker_ids = sorted({seg["speaker_id"] for seg in segments})

        metadata = {
            "audio_file": os.path.basename(audio_path),
            "duration_seconds": round(info.duration, 3),
            "sample_rate": info.samplerate,
            "num_channels": info.channels,
            "diarization_model": DIARIZATION_MODEL_ID,
            "speaker_ids": speaker_ids,
            "segments": segments,
        }

        output_p = Path(output_metadata_path).resolve()
        output_p.parent.mkdir(parents=True, exist_ok=True)
        with output_p.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info(
            f"Diarization complete: {len(segments)} turns, "
            f"speakers: {speaker_ids} -> {output_p.name}"
        )
        return metadata

    finally:
        torch.backends.cudnn.enabled = original_cudnn_state
        # Drop this function's final references before clearing the CUDA cache.
        # Passing these objects to release_memory while keeping the local
        # variables alive prevents PyTorch from reclaiming their allocations.
        pipeline = None
        result = None
        annotation = None
        release_memory()
        log_vram("After Diarization Cleanup")
