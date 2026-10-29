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
from .language_detector import detect_audio_chunk_language, resolve_language_name

logger = logging.getLogger(__name__)


def load_asr_model(
    gpu_memory_utilization: float = WHISPER_GPU_MEM,
    language: Optional[str] = None,
    decoder: Optional[str] = None,
) -> Dict[str, Any]:
    """Loads and returns the ASR model bundle (Indic-Conformer or fallback Whisper) onto GPU once."""
    lang_code = language or ASR_LANG or "en"
    decoder_type = decoder or ASR_DECODER or "ctc"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bodhan_model = None
    indic_model = None
    whisper_pipe = None
    vllm_llm = None
    sampling_params = None

    _orig_stdout = sys.stdout
    try:
        # 1. Attempt to initialize Bodhan IndicTranscribe if configured
        is_bodhan = "indic-transcribe" in ASR_MODEL_ID.lower() or "bodhan" in ASR_MODEL_ID.lower()
        if is_bodhan:
            try:
                log_vram("Before Bodhan IndicTranscribe ASR Load")
                logger.info(f"Loading Bodhan IndicTranscribe ASR model ({ASR_MODEL_ID}) on {device}...")
                token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
                from huggingface_hub import snapshot_download
                model_dir = snapshot_download(repo_id=ASR_MODEL_ID, token=token)
                if str(model_dir) not in sys.path:
                    sys.path.insert(0, str(model_dir))
                from indic_transcribe import IndicTranscribe
                bodhan_model = IndicTranscribe.from_pretrained(model_dir)
                if hasattr(bodhan_model, "to"):
                    try:
                        bodhan_model = bodhan_model.to(device)
                    except Exception:
                        pass
                logger.info("Bodhan IndicTranscribe ASR loaded successfully.")
            except Exception as bodhan_err:
                logger.warning(
                    f"Bodhan ASR load failed ({type(bodhan_err).__name__}: {bodhan_err}). "
                    "Falling back to Indic-Conformer or Whisper..."
                )
                bodhan_model = None
                release_memory()

        # 2. Attempt to initialize AI4Bharat Indic-Conformer 600M Multilingual
        is_indic_conformer = "indic-conformer" in ASR_MODEL_ID.lower()
        if is_indic_conformer and bodhan_model is None:
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

        # 3. Fallback to Whisper if not Bodhan/Indic-Conformer or if they failed
        if bodhan_model is None and indic_model is None:
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
                    enforce_eager=True,
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
    finally:
        sys.stdout = _orig_stdout

    return {
        "bodhan_model": bodhan_model,
        "indic_model": indic_model,
        "whisper_pipe": whisper_pipe,
        "vllm_llm": vllm_llm,
        "sampling_params": sampling_params,
        "lang_code": lang_code,
        "decoder_type": decoder_type,
    }


def run_transcription(
    audio_path: str,
    metadata_or_path: Any,
    output_transcript_path: str,
    gpu_memory_utilization: float = WHISPER_GPU_MEM,
    language: Optional[str] = None,
    decoder: Optional[str] = None,
    existing_model: Optional[Dict[str, Any]] = None,
    release_after: bool = True,
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

    lang_code = language or ASR_LANG or "en"
    decoder_type = decoder or ASR_DECODER or "ctc"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if existing_model is not None:
        bodhan_model = existing_model.get("bodhan_model")
        indic_model = existing_model.get("indic_model")
        whisper_pipe = existing_model.get("whisper_pipe")
        vllm_llm = existing_model.get("vllm_llm")
        sampling_params = existing_model.get("sampling_params")
    else:
        model_bundle = load_asr_model(
            gpu_memory_utilization=gpu_memory_utilization,
            language=lang_code,
            decoder=decoder_type,
        )
        bodhan_model = model_bundle.get("bodhan_model")
        indic_model = model_bundle.get("indic_model")
        whisper_pipe = model_bundle.get("whisper_pipe")
        vllm_llm = model_bundle.get("vllm_llm")
        sampling_params = model_bundle.get("sampling_params")

    _orig_stdout = sys.stdout
    try:

        transcript_segments = []
        failed_segments = 0

        for seg in segments:
            start_time = float(seg["start"])
            end_time = float(seg["end"])
            speaker = seg.get("speaker_id", "UNKNOWN")

            turn_audio = slice_turn_audio(audio_data, start_time, end_time, samplerate)
            if turn_audio is None:
                continue

            # Dynamically determine language for the turn when auto-detection or multilingual decoding is active
            active_lang = lang_code
            if lang_code in ("auto", "multilingual", None):
                active_lang = detect_audio_chunk_language(turn_audio, samplerate, existing_model or model_bundle)

            text = ""
            try:
                if bodhan_model is not None:
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                        temp_wav_path = tf.name
                    try:
                        sf.write(temp_wav_path, turn_audio, samplerate)
                        target_lang = active_lang if active_lang not in ("auto", "multilingual", None) else None
                        lang_map = {
                            "hindi": "hi", "hin": "hi",
                            "telugu": "te", "tel": "te",
                            "tamil": "ta", "tam": "ta",
                            "english": "en", "eng": "en",
                            "kannada": "kn", "kan": "kn",
                            "malayalam": "ml", "mal": "ml",
                            "marathi": "mr", "mar": "mr",
                            "gujarati": "gu", "guj": "gu",
                            "bengali": "bn", "ben": "bn",
                            "punjabi": "pa", "pan": "pa",
                            "odia": "or", "ori": "or",
                        }
                        if target_lang:
                            target_lang = lang_map.get(target_lang.lower(), target_lang.lower())
                        pred = bodhan_model(temp_wav_path, lang=target_lang) if target_lang else bodhan_model(temp_wav_path)
                        if isinstance(pred, dict):
                            text = pred.get("text") or pred.get("transcription") or str(pred)
                        elif isinstance(pred, (list, tuple)):
                            text = pred[0].strip() if pred else ""
                        else:
                            text = str(pred or "").strip()
                    finally:
                        if os.path.exists(temp_wav_path):
                            try:
                                os.remove(temp_wav_path)
                            except Exception:
                                pass

                elif indic_model is not None:
                    # Convert float32 numpy array to torch tensor (1, samples)
                    wav_tensor = torch.from_numpy(turn_audio).unsqueeze(0).to(device)
                    with torch.no_grad():
                        target_lang = active_lang if active_lang not in ("auto", "multilingual", None) else "en"
                        pred = indic_model(wav_tensor, target_lang, decoder_type)
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
                        "decoder_prompt": f"<|startoftranscript|><|{active_lang}|><|transcribe|>" if active_lang and active_lang != "auto" else "<|startoftranscript|>",
                    }
                    outputs = vllm_llm.generate([inputs], sampling_params=sampling_params, use_tqdm=False)
                    text = outputs[0].outputs[0].text.strip()
                elif whisper_pipe is not None:
                    gen_kwargs = {"task": "transcribe"}
                    if active_lang and active_lang not in ("auto", "multilingual", None):
                        gen_kwargs["language"] = active_lang
                    hf_res = whisper_pipe({"raw": turn_audio, "sampling_rate": samplerate}, generate_kwargs=gen_kwargs)
                    text = hf_res.get("text", "").strip()

            except Exception as exc:
                failed_segments += 1
                logger.error(
                    f"[ASR Error] {start_time:.2f}-{end_time:.2f}: {type(exc).__name__}: {exc}"
                )
                continue

            if not text:
                continue

            turn_lang_name = resolve_language_name(active_lang)

            item = {
                "start": round(start_time, 3),
                "end": round(end_time, 3),
                "speaker": str(speaker),
                "text": text,
                "raw_text": text,
                "language": turn_lang_name,
                "language_code": active_lang if active_lang and active_lang != "auto" else "en",
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
        if release_after:
            release_memory(bodhan_model, indic_model, vllm_llm, sampling_params, whisper_pipe)
            log_vram("After ASR Cleanup")
