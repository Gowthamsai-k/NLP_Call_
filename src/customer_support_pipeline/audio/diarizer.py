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

    # Ensure token is registered with Hugging Face Hub environment
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
    except Exception:
        pass

    try:
        from pyannote.audio import Pipeline
    except RuntimeError as e:
        if "different CUDA versions" in str(e) or "compiled with different CUDA" in str(e):
            try:
                import torchaudio._extension.utils as _tau
                _tau._check_cuda_version = lambda: None
                from pyannote.audio import Pipeline
            except Exception:
                raise e
        else:
            raise e

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

        # Explicitly allowlist pyannote task types for PyTorch 2.6+ unpickler
        try:
            from pyannote.audio.core.task import Specifications, Problem, Resolution
            torch.serialization.add_safe_globals([Specifications, Problem, Resolution])
        except Exception:
            pass
        try:
            from torch.torch_version import TorchVersion
            torch.serialization.add_safe_globals([TorchVersion])
        except Exception:
            pass

        # Try supported token passing conventions across pyannote versions
        pipeline = None
        last_err = None
        models_to_try = [DIARIZATION_MODEL_ID, "pyannote/speaker-diarization-3.1"]

        for model_id in models_to_try:
            for kwargs in [{"token": token}, {"use_auth_token": token}, {}]:
                try:
                    pipeline = Pipeline.from_pretrained(model_id, **kwargs)
                    if pipeline is not None:
                        logger.info(f"Loaded diarization pipeline: {model_id}")
                        break
                except TypeError as te:
                    last_err = te
                    if "unexpected keyword argument 'token'" in str(te) or "unexpected keyword argument 'use_auth_token'" in str(te):
                        continue
                    if "plda" in str(te):
                        # PLDA requires pyannote.audio 4.0; try next model if on 3.x
                        break
                    continue
                except Exception as exc:
                    last_err = exc
                    break
            if pipeline is not None:
                break

        if pipeline is None:
            if last_err:
                raise last_err
            raise RuntimeError(f"Failed to load diarization pipeline: {DIARIZATION_MODEL_ID}")

        pipeline.to(target_device)

        # Pre-load audio directly with soundfile to bypass TorchCodec / torchaudio.load entirely
        audio_data, sr = sf.read(audio_path, dtype="float32")
        if audio_data.ndim == 1:
            waveform_tensor = torch.from_numpy(audio_data).unsqueeze(0)
        else:
            waveform_tensor = torch.from_numpy(audio_data.T)

        diarization_input = {
            "waveform": waveform_tensor,
            "sample_rate": sr,
        }

        with torch.inference_mode():
            try:
                result = pipeline(diarization_input)
            except Exception as exc:
                logger.debug(f"Direct waveform dict failed ({exc}), falling back to audio file path...")
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
        release_memory(pipeline, result, annotation)
        log_vram("After Diarization Cleanup")
