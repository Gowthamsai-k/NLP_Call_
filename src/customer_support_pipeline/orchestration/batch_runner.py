"""
Folder processing, batch execution, and archive upload handling.
"""

import os
import json
import time
import zipfile
import tarfile
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..core.config import (
    DEFAULT_STAGE_RETRIES,
    RETRY_BACKOFF_FACTOR,
    WHISPER_GPU_MEM,
    INTENT_GPU_MEM,
    EVAL_GPU_MEM,
)
from ..audio.preprocessor import find_audio_files
from ..core.memory import release_memory, log_vram
from .pipeline import process_single_call

logger = logging.getLogger(__name__)


def unpack_archive(archive_path: str, destination_dir: Optional[str] = None) -> Path:
    """Unpacks an uploaded .zip, .tar, or .tar.gz archive of audio files."""
    arch_p = Path(archive_path).resolve()
    if not arch_p.exists():
        raise FileNotFoundError(f"Archive file not found: {arch_p}")

    out_p = Path(destination_dir) if destination_dir else arch_p.parent / arch_p.stem
    out_p.mkdir(parents=True, exist_ok=True)

    logger.info(f"Unpacking {arch_p.name} -> {out_p}...")
    if zipfile.is_zipfile(arch_p):
        with zipfile.ZipFile(arch_p, "r") as zf:
            zf.extractall(out_p)
    elif tarfile.is_tarfile(arch_p):
        with tarfile.open(arch_p, "r:*") as tf:
            tf.extractall(out_p)
    else:
        raise ValueError(f"Unsupported archive format: {arch_p.suffix}")

    logger.info(f"Extraction complete -> {out_p}")
    return out_p


def process_audio_folder(
    input_folder: str,
    output_base_dir: str,
    hf_token: Optional[str] = None,
    max_retries: int = DEFAULT_STAGE_RETRIES,
    backoff_factor: float = RETRY_BACKOFF_FACTOR,
    resume: bool = True,
    clean_intermediate: bool = True,
    reuse_llm: bool = True,
    recursive: bool = True,
    whisper_gpu_mem: float = WHISPER_GPU_MEM,
    intent_gpu_mem: float = INTENT_GPU_MEM,
    eval_gpu_mem: float = EVAL_GPU_MEM,
    asr_lang: Optional[str] = None,
    asr_decoder: Optional[str] = None,
) -> Dict[str, Any]:
    """Processes all audio files in an input folder with failure isolation."""
    batch_start = time.time()
    in_dir = Path(input_folder).resolve()
    out_base = Path(output_base_dir).resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    audio_files = find_audio_files(str(in_dir), recursive=recursive)
    if not audio_files:
        logger.warning(f"No audio files found in directory: {in_dir}")
        return {
            "total_files": 0,
            "processed": 0,
            "failed": 0,
            "summary_file": None,
        }

    logger.info(f"Found {len(audio_files)} audio file(s) to process in {in_dir}")

    results = []
    failed = []
    skipped = []

    for index, audio_path in enumerate(audio_files, start=1):
        file_stem = audio_path.stem
        call_output_dir = out_base / file_stem
        logger.info(f"\n[{index}/{len(audio_files)}] Processing: {audio_path.name}")

        try:
            call_res = process_single_call(
                input_audio_path=str(audio_path),
                output_dir=str(call_output_dir),
                hf_token=hf_token,
                max_retries=max_retries,
                backoff_factor=backoff_factor,
                resume=resume,
                clean_intermediate=clean_intermediate,
                reuse_llm=reuse_llm,
                whisper_gpu_mem=whisper_gpu_mem,
                intent_gpu_mem=intent_gpu_mem,
                eval_gpu_mem=eval_gpu_mem,
                asr_lang=asr_lang,
                asr_decoder=asr_decoder,
            )
            results.append(call_res)
            if call_res.get("skipped"):
                skipped.append(audio_path.name)
        except Exception as exc:
            logger.error(f"Failed processing {audio_path.name}: {exc}", exc_info=True)
            failed.append({
                "audio_file": audio_path.name,
                "error": str(exc),
            })
        finally:
            release_memory(delay=1.0)
            log_vram(f"Post-Call Cleanup ({index}/{len(audio_files)})")

    batch_summary = {
        "input_directory": str(in_dir),
        "output_directory": str(out_base),
        "total_files": len(audio_files),
        "successful": len(results) - len(skipped),
        "skipped_resumed": len(skipped),
        "failed_count": len(failed),
        "failed_files": failed,
        "total_elapsed_seconds": round(time.time() - batch_start, 2),
    }

    summary_path = out_base / "batch_processing_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(batch_summary, f, ensure_ascii=False, indent=2)

    logger.info(
        f"\n========================================\n"
        f"BATCH PROCESSING COMPLETE\n"
        f"Total: {len(audio_files)} | Success: {batch_summary['successful']} | "
        f"Resumed: {len(skipped)} | Failed: {len(failed)}\n"
        f"Summary: {summary_path}\n"
        f"========================================"
    )
    return batch_summary
