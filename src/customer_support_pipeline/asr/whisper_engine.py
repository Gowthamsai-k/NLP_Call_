"""
Speech-to-text transcription using AI4Bharat Indic-Conformer Multilingual (with automatic Whisper fallback).
Model: ai4bharat/indic-conformer-600m-multilingual
Supports 22 official Indian languages + Indian English with CTC and RNNT decoding.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch

from ..core.config import (
    ASR_MODEL_ID,
    ASR_LANG,
    ASR_DECODER,
    WHISPER_MAX_MODEL_LEN,
    WHISPER_MAX_BATCHED_TOKENS,
    WHISPER_MAX_TOKENS,
    WHISPER_GPU_MEM,
)
from ..audio.preprocessor import load_and_resample_audio, slice_turn_audio
from ..core.memory import release_memory, log_vram

logger = logging.getLogger(__name__)


def run_transcription(
    audio_path: str,
    metadata_or_path: Any,
    output_transcript_path: str,
    gpu_memory_utilization: float = WHISPER_GPU_MEM,
    language: Optional[str] = None,
    decoder: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Transcribes diarized audio turns using ai4bharat/indic-conformer-600m-multilingual.
    Falls back gracefully to Whisper if needed.
    """
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

    lang_code = language or ASR_LANG or "ta"
    decoder_type = decoder or ASR_DECODER or "ctc"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    indic_model = None
    whisper_pipe = None
    vllm_llm = None
    sampling_params = None

    _orig_stdout = sys.stdout
    try:
        # 1. Attempt to initialize AI4Bharat Indic-Conformer 600M Multilingual
        is_indic_conformer = "indic-conformer" in ASR_MODEL_ID.lower()
        if is_indic_conformer:
            try:
                log_vram("Before Indic-Conformer ASR Load")
                logger.info(
                    f"Loading Indic-Conformer ASR model ({ASR_MODEL_ID}) on {device} (Lang: {lang_code}, Decoder: {decoder_type})..."
                )
                from transformers import AutoModel
                indic_model = AutoModel.from_pretrained(
                    ASR_MODEL_ID,
                    trust_remote_code=True,
                ).to(device)
                indic_model.eval()
                logger.info("Indic-Conformer ASR loaded successfully.")
            except Exception as indic_err:
                logger.warning(
                    f"Indic-Conformer load failed ({type(indic_err).__name__}: {indic_err}). "
                    "Falling back to Whisper ASR engine..."
                )
                indic_model = None
                release_memory()

        # 2. Fallback to Whisper if not Indic-Conformer or if Indic-Conformer failed
        if indic_model is None:
            try:
                from vllm import LLM, SamplingParams
                log_vram("Before Whisper vLLM Load")
                sys.stdout = sys.__stdout__
                fallback_whisper_id = "openai/whisper-large-v3-turbo"
                vllm_llm = LLM(
                    model=fallback_whisper_id,
                    dtype="half",
                    trust_remote_code=True,
                    max_model_len=WHISPER_MAX_MODEL_LEN,
                    max_num_batched_tokens=WHISPER_MAX_BATCHED_TOKENS,
                    max_num_seqs=1,
                    limit_mm_per_prompt={"audio": 1},
                    gpu_memory_utilization=gpu_memory_utilization,
                    disable_log_stats=True,
                )
                sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=WHISPER_MAX_TOKENS,
                )
                logger.info("vLLM Whisper engine initialized as fallback.")
            except Exception as vllm_err:
                logger.warning(
                    f"vLLM engine unavailable ({type(vllm_err).__name__}: {vllm_err}). "
                    "Switching to native Hugging Face Transformers Whisper pipeline..."
                )
                release_memory()
                from transformers import pipeline as make_pipeline
                target_device = 0 if torch.cuda.is_available() else -1
                fallback_model = "openai/whisper-large-v3-turbo"
                whisper_pipe = make_pipeline(
                    "automatic-speech-recognition",
                    model=fallback_model,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device=target_device,
                )
                logger.info("Hugging Face Whisper ASR pipeline ready.")

        sys.stdout = _orig_stdout

        transcript_segments = []
        failed_segments = 0

        for seg in segments:
            start_time = float(seg["start"])
            end_time = float(seg["end"])
            speaker = seg.get("speaker_id", "UNKNOWN")

            turn_audio = slice_turn_audio(audio_data, start_time, end_time, samplerate)
            if turn_audio is None:
                continue

            text = ""
            try:
                if indic_model is not None:
                    # Convert float32 numpy array to torch tensor (1, samples)
                    wav_tensor = torch.from_numpy(turn_audio).unsqueeze(0).to(device)
                    with torch.no_grad():
                        pred = indic_model(wav_tensor, lang_code, decoder_type)
                    if isinstance(pred, (list, tuple)):
                        text = pred[0].strip() if pred else ""
                    else:
                        text = str(pred or "").strip()

                elif vllm_llm is not None:
                    inputs = {
                        "encoder_prompt": {
                            "prompt": "",
                            "multi_modal_data": {
                                "audio": (turn_audio, samplerate),
                            },
                        },
                        "decoder_prompt": "<|startoftranscript|>",
                    }
                    outputs = vllm_llm.generate([inputs], sampling_params=sampling_params, use_tqdm=False)
                    text = outputs[0].outputs[0].text.strip()
                elif whisper_pipe is not None:
                    hf_res = whisper_pipe({"raw": turn_audio, "sampling_rate": samplerate})
                    text = hf_res.get("text", "").strip()

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
        release_memory(indic_model, vllm_llm, sampling_params, whisper_pipe)
        log_vram("After ASR Cleanup")
