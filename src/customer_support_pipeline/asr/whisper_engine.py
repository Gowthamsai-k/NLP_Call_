"""
Speech-to-text transcription using the configured FP8 Whisper large-v3 model on vLLM.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

import torch

from ..core.config import (
    WHISPER_MODEL_ID,
    WHISPER_MAX_MODEL_LEN,
    WHISPER_MAX_BATCHED_TOKENS,
    WHISPER_MAX_TOKENS,
    WHISPER_GPU_MEM,
)
from ..audio.preprocessor import load_and_resample_audio, slice_turn_audio
from ..core.memory import (
    bounded_gpu_memory_utilization,
    release_memory,
    log_vram,
)

logger = logging.getLogger(__name__)


def run_transcription(
    audio_path: str,
    metadata_or_path: Any,
    output_transcript_path: str,
    gpu_memory_utilization: float = WHISPER_GPU_MEM,
) -> List[Dict[str, Any]]:
    """Transcribes diarized audio turns using quantized Whisper large-v3 on vLLM."""
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    # Crucial: Bypass CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH in Colab
    # Whisper's Conv1D frontend uses PyTorch native CUDA kernels when cuDNN is disabled
    original_cudnn_state = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False

    if isinstance(metadata_or_path, (str, Path)):
        with open(metadata_or_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = metadata_or_path

    segments = metadata.get("segments", [])
    if not segments:
        logger.warning("No diarization segments provided for transcription.")
        return []

    audio_data, samplerate, duration = load_and_resample_audio(audio_path)
    logger.info(
        f"Audio loaded for ASR: {duration:.2f}s at {samplerate} Hz across {len(segments)} turns"
    )

    from vllm import LLM, SamplingParams

    _orig_stdout = sys.stdout
    sys.stdout = sys.__stdout__

    llm = None
    sampling_params = None

    try:
        log_vram("Before Whisper vLLM Load")
        llm = LLM(
            model=WHISPER_MODEL_ID,
            dtype="half",
            trust_remote_code=True,
            max_model_len=WHISPER_MAX_MODEL_LEN,
            max_num_batched_tokens=WHISPER_MAX_BATCHED_TOKENS,
            max_num_seqs=1,
            limit_mm_per_prompt={"audio": 1},
            gpu_memory_utilization=bounded_gpu_memory_utilization(gpu_memory_utilization),
            disable_log_stats=True,
        )
        sys.stdout = _orig_stdout

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=WHISPER_MAX_TOKENS,
        )

        transcript_segments = []
        failed_segments = 0

        for seg in segments:
            start_time = float(seg["start"])
            end_time = float(seg["end"])
            speaker = seg.get("speaker_id", "UNKNOWN")

            turn_audio = slice_turn_audio(audio_data, start_time, end_time, samplerate)
            if turn_audio is None:
                continue

            try:
                inputs = {
                    "encoder_prompt": {
                        "prompt": "",
                        "multi_modal_data": {
                            "audio": (turn_audio, samplerate),
                        },
                    },
                    "decoder_prompt": "<|startoftranscript|>",
                }

                outputs = llm.generate([inputs], sampling_params=sampling_params, use_tqdm=False)
                text = outputs[0].outputs[0].text.strip()
            except Exception as exc:
                failed_segments += 1
                logger.error(
                    f"[ASR Error] {start_time:.2f}-{end_time:.2f}: {type(exc).__name__}: {exc}"
                )
                continue

            if not text:
                continue

            item = {
                "start": round(start_time, 3),
                "end": round(end_time, 3),
                "speaker": str(speaker),
                "text": text,
                "raw_text": text,
                "words": [],
            }
            transcript_segments.append(item)

        output_p = Path(output_transcript_path).resolve()
        output_p.parent.mkdir(parents=True, exist_ok=True)
        with output_p.open("w", encoding="utf-8") as f:
            json.dump(transcript_segments, f, ensure_ascii=False, indent=2)

        logger.info(
            f"ASR complete: {len(transcript_segments)} transcribed segments "
            f"({failed_segments} failed) -> {output_p.name}"
        )
        return transcript_segments

    finally:
        sys.stdout = _orig_stdout
        torch.backends.cudnn.enabled = original_cudnn_state
        # Shut vLLM down while the engine reference is available, then drop
        # local references and purge again so cached CUDA allocations return
        # to the device before the next stage or recording starts.
        release_memory(llm, sampling_params, delay=0)
        llm = None
        sampling_params = None
        release_memory()
        log_vram("After Whisper Cleanup")
