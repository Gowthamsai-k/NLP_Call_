"""
Folder processing, batch execution, and archive upload handling.
Implements Stage-by-Stage Batch Architecture:
Loads each model ONCE per batch (PyAnnote, Indic-Conformer, Horizon, Qwen),
processes all files in the batch, saves outputs immediately,
and unloads the model before transitioning to the next stage.
"""

import os
import json
import time
import zipfile
import tarfile
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable

from ..core.config import (
    DEFAULT_STAGE_RETRIES,
    RETRY_BACKOFF_FACTOR,
    CLEAN_WAV_FILENAME,
    METADATA_FILENAME,
    WHISPER_RAW_FILENAME,
    PUNCTUATED_FILENAME,
    FINAL_TRANSCRIPT_FILENAME,
    FINAL_INTENT_FILENAME,
    FINAL_EVALUATION_FILENAME,
    PIPELINE_STATUS_FILENAME,
    WHISPER_GPU_MEM,
    INTENT_GPU_MEM,
    EVAL_GPU_MEM,
    ASR_LANG,
    ASR_DECODER,
)
from ..core.memory import release_memory, retry_with_memory_cleanup, log_vram
from ..audio.preprocessor import convert_to_clean_wav, find_audio_files
from ..audio.diarizer import load_diarization_pipeline, run_diarization
from ..asr.whisper_engine import load_asr_model, run_transcription
from ..asr.punctuator import load_punctuation_model, run_punctuation_restoration
from ..asr.aligner import align_speakers_with_transcript
from ..nlp.intent_extractor import run_intent_extraction
from ..nlp.evaluator import run_call_evaluation

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


def process_batch_by_stages(
    audio_files: List[Union[str, Path]],
    output_base_dir: str,
    hf_token: Optional[str] = None,
    max_retries: int = DEFAULT_STAGE_RETRIES,
    backoff_factor: float = RETRY_BACKOFF_FACTOR,
    resume: bool = True,
    clean_intermediate: bool = True,
    whisper_gpu_mem: float = WHISPER_GPU_MEM,
    intent_gpu_mem: float = INTENT_GPU_MEM,
    eval_gpu_mem: float = EVAL_GPU_MEM,
    asr_lang: Optional[str] = None,
    asr_decoder: Optional[str] = None,
    stage_callback: Optional[Callable[[str, int, str], None]] = None,
    file_completed_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Executes all audio files through the pipeline using a Stage-by-Stage Batch Architecture.
    Loads each neural model ONCE for the entire batch:
      1. Stage 1: Audio Preprocessing (FFmpeg) across all files -> saved immediately
      2. Stage 2: PyAnnote Diarization loaded ONCE -> diarizes all files -> saved immediately -> unloaded ONCE
      3. Stage 3: Indic-Conformer ASR loaded ONCE -> transcribes all files -> saved immediately -> unloaded ONCE
      4. Stage 4: Horizon Punctuation loaded ONCE -> restores punctuation for all files -> saved immediately -> unloaded ONCE
      5. Stage 5: Speaker Alignment across all files -> saved immediately
      6. Stages 6 & 7: Qwen LLM loaded ONCE -> extracts intents & evaluates all files -> saved immediately -> unloaded ONCE
      7. Stage 8: Score Deliverables packaged & cleaned up
    """
    batch_start = time.time()
    out_base = Path(output_base_dir).resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    if not audio_files:
        logger.warning("No audio files provided for stage-by-stage batch processing.")
        return {
            "total_files": 0,
            "successful": 0,
            "failed_count": 0,
            "failed_files": [],
            "evaluations": [],
            "total_elapsed_seconds": 0.0,
        }

    logger.info(f"\n{'='*70}\nSTAGE-BY-STAGE BATCH PIPELINE START: {len(audio_files)} audio file(s)\n{'='*70}")

    # Build per-file contexts
    file_contexts: List[Dict[str, Any]] = []
    failed_base_dir = out_base / "failed"

    for audio_item in audio_files:
        audio_p = Path(audio_item).resolve()
        call_out = out_base / audio_p.stem
        call_out.mkdir(parents=True, exist_ok=True)
        file_contexts.append({
            "input_p": audio_p,
            "out_dir": call_out,
            "clean_wav_p": call_out / CLEAN_WAV_FILENAME,
            "metadata_p": call_out / METADATA_FILENAME,
            "whisper_raw_p": call_out / WHISPER_RAW_FILENAME,
            "punctuated_p": call_out / PUNCTUATED_FILENAME,
            "final_transcript_p": call_out / FINAL_TRANSCRIPT_FILENAME,
            "final_intent_p": call_out / FINAL_INTENT_FILENAME,
            "final_eval_p": call_out / FINAL_EVALUATION_FILENAME,
            "status_p": call_out / PIPELINE_STATUS_FILENAME,
            "failed": False,
            "skipped": False,
            "error": None,
            "failed_stage": None,
            "failed_dir": None,
        })

    def relocate_to_failed_dir(ctx_item: Dict[str, Any]) -> Path:
        """Moves any artifacts for a failed call into output/failed/<call_stem> and writes error records."""
        import shutil
        stem = ctx_item["input_p"].stem
        dest_dir = failed_base_dir / stem
        dest_dir.mkdir(parents=True, exist_ok=True)

        failure_info = {
            "audio_file": ctx_item["input_p"].name,
            "input_path": str(ctx_item["input_p"]),
            "status": "failed",
            "failed_stage": ctx_item.get("failed_stage"),
            "error": ctx_item.get("error"),
            "failed_directory": str(dest_dir),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with (dest_dir / "failure_reason.json").open("w", encoding="utf-8") as f:
            json.dump(failure_info, f, ensure_ascii=False, indent=2)

        with (dest_dir / "error.txt").open("w", encoding="utf-8") as f:
            f.write(
                f"Audio File: {ctx_item['input_p'].name}\n"
                f"Failed Stage: {ctx_item.get('failed_stage')}\n"
                f"Error: {ctx_item.get('error')}\n"
                f"Time: {failure_info['timestamp']}\n"
            )

        # Move any partial artifacts from ctx_item["out_dir"] into dest_dir
        if ctx_item["out_dir"].resolve() != dest_dir.resolve() and ctx_item["out_dir"].exists():
            for item in list(ctx_item["out_dir"].iterdir()):
                target_dest = dest_dir / item.name
                if target_dest.resolve() in (
                    (dest_dir / "failure_reason.json").resolve(),
                    (dest_dir / "error.txt").resolve(),
                ):
                    continue
                try:
                    if target_dest.exists():
                        if target_dest.is_dir():
                            shutil.rmtree(target_dest, ignore_errors=True)
                        else:
                            target_dest.unlink(missing_ok=True)
                    shutil.move(str(item), str(target_dest))
                except Exception as move_e:
                    logger.warning(f"Could not move {item.name} to failed directory: {move_e}")
            try:
                shutil.rmtree(ctx_item["out_dir"], ignore_errors=True)
            except Exception:
                pass
        return dest_dir

    # Pre-execution VRAM cleanup
    release_memory()

    # =========================================================================
    # STAGE 1: Audio Conversion (FFmpeg, 16kHz mono PCM) - All files
    # =========================================================================
    logger.info(f"\n--- [Stage 1/8] Converting audio files (FFmpeg) for {len(file_contexts)} file(s) ---")
    for ctx in file_contexts:
        fname = ctx["input_p"].name
        if resume and ctx["clean_wav_p"].exists() and ctx["clean_wav_p"].stat().st_size > 0:
            logger.info(f"Stage 1: Clean WAV already exists for {fname}, resuming.")
            continue

        try:
            if stage_callback:
                stage_callback(fname, 1, "FFmpeg Audio Conversion (16kHz PCM)")
            retry_with_memory_cleanup(
                convert_to_clean_wav,
                str(ctx["input_p"]),
                str(ctx["clean_wav_p"]),
                max_retries=max_retries,
                backoff_factor=backoff_factor,
                stage_name=f"Stage 1 ({fname})",
            )
            logger.info(f"Stage 1 complete -> {ctx['clean_wav_p'].name} (saved immediately)")
        except Exception as exc:
            logger.error(f"Stage 1 failed for {fname}: {exc}", exc_info=True)
            ctx["failed"] = True
            ctx["error"] = f"Stage 1: {exc}"
            ctx["failed_stage"] = 1

    # =========================================================================
    # STAGE 2: Speaker Diarization (PyAnnote) - Loaded ONCE for entire batch
    # =========================================================================
    files_to_diarize = [
        ctx for ctx in file_contexts
        if not ctx["failed"] and not (resume and ctx["metadata_p"].exists() and ctx["metadata_p"].stat().st_size > 0)
    ]

    if files_to_diarize:
        logger.info(f"\n--- [Stage 2/8] Loading PyAnnote Diarization Model ONCE onto GPU for {len(files_to_diarize)} file(s) ---")
        diarization_pipeline = None
        try:
            diarization_pipeline = load_diarization_pipeline(hf_token=hf_token)
            for ctx in files_to_diarize:
                fname = ctx["input_p"].name
                try:
                    if stage_callback:
                        stage_callback(fname, 2, "Speaker Diarization (PyAnnote)")
                    run_diarization(
                        str(ctx["clean_wav_p"]),
                        str(ctx["metadata_p"]),
                        existing_pipeline=diarization_pipeline,
                        release_after=False,
                    )
                    logger.info(f"Stage 2 complete -> {ctx['metadata_p'].name} (saved immediately)")
                except Exception as exc:
                    logger.error(f"Stage 2 failed for {fname}: {exc}", exc_info=True)
                    ctx["failed"] = True
                    ctx["error"] = f"Stage 2: {exc}"
                    ctx["failed_stage"] = 2
        finally:
            logger.info("--- [Stage 2/8] Unloading PyAnnote Diarization Model ONCE from GPU ---")
            release_memory(diarization_pipeline, delay=1.0)
            log_vram("Post-Diarization Batch Cleanup")
    else:
        logger.info("\n--- [Stage 2/8] All files already have metadata. Skipping model load. ---")

    # =========================================================================
    # STAGE 3: ASR Transcription (AI4Bharat Indic-Conformer) - Loaded ONCE
    # =========================================================================
    files_to_asr = [
        ctx for ctx in file_contexts
        if not ctx["failed"] and not (resume and ctx["whisper_raw_p"].exists() and ctx["whisper_raw_p"].stat().st_size > 0)
    ]

    if files_to_asr:
        logger.info(f"\n--- [Stage 3/8] Loading Indic-Conformer ASR Model ONCE onto GPU for {len(files_to_asr)} file(s) ---")
        asr_bundle = None
        try:
            asr_bundle = load_asr_model(
                gpu_memory_utilization=whisper_gpu_mem,
                language=asr_lang,
                decoder=asr_decoder,
            )
            for ctx in files_to_asr:
                fname = ctx["input_p"].name
                try:
                    if stage_callback:
                        stage_callback(fname, 3, "ASR Transcription (AI4Bharat Indic-Conformer)")
                    run_transcription(
                        str(ctx["clean_wav_p"]),
                        str(ctx["metadata_p"]),
                        str(ctx["whisper_raw_p"]),
                        language=asr_lang,
                        decoder=asr_decoder,
                        existing_model=asr_bundle,
                        release_after=False,
                    )
                    logger.info(f"Stage 3 complete -> {ctx['whisper_raw_p'].name} (saved immediately)")
                except Exception as exc:
                    logger.error(f"Stage 3 failed for {fname}: {exc}", exc_info=True)
                    ctx["failed"] = True
                    ctx["error"] = f"Stage 3: {exc}"
                    ctx["failed_stage"] = 3
        finally:
            logger.info("--- [Stage 3/8] Unloading Indic-Conformer ASR Model ONCE from GPU ---")
            release_memory(asr_bundle, delay=1.0)
            log_vram("Post-ASR Batch Cleanup")
    else:
        logger.info("\n--- [Stage 3/8] All files already have ASR transcripts. Skipping model load. ---")

    # =========================================================================
    # STAGE 4: Punctuation Restoration (Horizon-Labs) - Loaded ONCE
    # =========================================================================
    files_to_punctuate = [
        ctx for ctx in file_contexts
        if not ctx["failed"] and not (resume and ctx["punctuated_p"].exists() and ctx["punctuated_p"].stat().st_size > 0)
    ]

    if files_to_punctuate:
        logger.info(f"\n--- [Stage 4/8] Loading Horizon Punctuation Model ONCE for {len(files_to_punctuate)} file(s) ---")
        punct_model = None
        try:
            punct_model = load_punctuation_model()
            for ctx in files_to_punctuate:
                fname = ctx["input_p"].name
                try:
                    if stage_callback:
                        stage_callback(fname, 4, "Punctuation Restoration (Horizon-Labs)")
                    run_punctuation_restoration(
                        str(ctx["whisper_raw_p"]),
                        str(ctx["punctuated_p"]),
                        existing_punctuator=punct_model,
                        release_after=False,
                    )
                    logger.info(f"Stage 4 complete -> {ctx['punctuated_p'].name} (saved immediately)")
                except Exception as exc:
                    logger.error(f"Stage 4 failed for {fname}: {exc}", exc_info=True)
                    ctx["failed"] = True
                    ctx["error"] = f"Stage 4: {exc}"
                    ctx["failed_stage"] = 4
        finally:
            logger.info("--- [Stage 4/8] Unloading Horizon Punctuation Model ONCE ---")
            release_memory(punct_model, delay=1.0)
            log_vram("Post-Punctuation Batch Cleanup")
    else:
        logger.info("\n--- [Stage 4/8] All files already have punctuated transcripts. Skipping model load. ---")

    # =========================================================================
    # STAGE 5: Speaker Turn Alignment & Role Attribution - All files
    # =========================================================================
    logger.info(f"\n--- [Stage 5/8] Aligning speaker turns with transcripts for {len(file_contexts)} file(s) ---")
    for ctx in file_contexts:
        if ctx["failed"]:
            continue
        fname = ctx["input_p"].name
        if resume and ctx["final_transcript_p"].exists() and ctx["final_transcript_p"].stat().st_size > 0:
            logger.info(f"Stage 5: Aligned transcript already exists for {fname}, resuming.")
            continue

        try:
            if stage_callback:
                stage_callback(fname, 5, "Speaker Alignment & Role Attribution")
            align_speakers_with_transcript(
                str(ctx["metadata_p"]),
                str(ctx["punctuated_p"]),
                str(ctx["final_transcript_p"]),
            )
            logger.info(f"Stage 5 complete -> {ctx['final_transcript_p'].name} (saved immediately)")
        except Exception as exc:
            logger.error(f"Stage 5 failed for {fname}: {exc}", exc_info=True)
            ctx["failed"] = True
            ctx["error"] = f"Stage 5: {exc}"
            ctx["failed_stage"] = 5

    # =========================================================================
    # STAGES 6 & 7: Qwen LLM (Intent Extraction & Quality Evaluation) - Loaded ONCE
    # =========================================================================
    files_to_llm = [
        ctx for ctx in file_contexts
        if not ctx["failed"] and (
            not (resume and ctx["final_intent_p"].exists() and ctx["final_intent_p"].stat().st_size > 0)
            or not (resume and ctx["final_eval_p"].exists() and ctx["final_eval_p"].stat().st_size > 0)
        )
    ]

    if files_to_llm:
        logger.info(f"\n--- [Stages 6 & 7] Loading Qwen 2.5 3B LLM ONCE on GPU for {len(files_to_llm)} file(s) ---")
        shared_llm = None
        shared_tokenizer = None
        try:
            for ctx in files_to_llm:
                fname = ctx["input_p"].name

                # Stage 6: Intent Extraction
                if not (resume and ctx["final_intent_p"].exists() and ctx["final_intent_p"].stat().st_size > 0):
                    try:
                        if stage_callback:
                            stage_callback(fname, 6, "Qwen Intent Extraction & Client Needs")
                        intent_res, shared_llm, shared_tokenizer = run_intent_extraction(
                            str(ctx["final_transcript_p"]),
                            str(ctx["final_intent_p"]),
                            work_dir=str(ctx["out_dir"] / "intent_batches"),
                            existing_llm=shared_llm,
                            existing_tokenizer=shared_tokenizer,
                            release_llm_after=False,
                            gpu_memory_utilization=intent_gpu_mem,
                        )
                        logger.info(f"Stage 6 complete -> {ctx['final_intent_p'].name} (saved immediately)")
                    except Exception as exc:
                        logger.error(f"Stage 6 failed for {fname}: {exc}", exc_info=True)
                        ctx["failed"] = True
                        ctx["error"] = f"Stage 6: {exc}"
                        ctx["failed_stage"] = 6
                        continue

                # Stage 7: Call Quality Evaluation
                if not (resume and ctx["final_eval_p"].exists() and ctx["final_eval_p"].stat().st_size > 0):
                    try:
                        if stage_callback:
                            stage_callback(fname, 7, "Qwen Quality Scoring & Rubric Evaluation")
                        run_call_evaluation(
                            str(ctx["final_transcript_p"]),
                            str(ctx["final_intent_p"]),
                            str(ctx["final_eval_p"]),
                            work_dir=str(ctx["out_dir"] / "call_evaluation_batches"),
                            existing_llm=shared_llm,
                            existing_tokenizer=shared_tokenizer,
                            release_llm_after=False,
                            gpu_memory_utilization=eval_gpu_mem,
                        )
                        logger.info(f"Stage 7 complete -> {ctx['final_eval_p'].name} (saved immediately)")
                    except Exception as exc:
                        logger.error(f"Stage 7 failed for {fname}: {exc}", exc_info=True)
                        ctx["failed"] = True
                        ctx["error"] = f"Stage 7: {exc}"
                        ctx["failed_stage"] = 7
                        continue
        finally:
            logger.info("--- [Stages 6 & 7] Unloading Qwen 2.5 3B LLM ONCE from GPU ---")
            release_memory(shared_llm, shared_tokenizer, delay=1.0)
            log_vram("Post-LLM Batch Cleanup")
    else:
        logger.info("\n--- [Stages 6 & 7] All files already have evaluations. Skipping LLM load. ---")

    # =========================================================================
    # STAGE 8: Score Deliverables Preservation & Intermediate Cleanup
    # =========================================================================
    logger.info(f"\n--- [Stage 8/8] Packaging final score deliverables for {len(file_contexts)} file(s) ---")
    evaluations_result: List[Dict[str, Any]] = []
    failed_files: List[Dict[str, Any]] = []

    for ctx in file_contexts:
        fname = ctx["input_p"].name
        if ctx["failed"]:
            dest_dir = relocate_to_failed_dir(ctx)
            ctx["failed_dir"] = str(dest_dir)
            failed_files.append({
                "audio_file": fname,
                "error": ctx["error"],
                "failed_stage": ctx.get("failed_stage"),
                "failed_dir": str(dest_dir),
            })
            continue

        try:
            if stage_callback:
                stage_callback(fname, 8, "Score Deliverable Packaging & Cleanup")

            eval_data = {}
            if ctx["final_eval_p"].exists():
                with ctx["final_eval_p"].open("r", encoding="utf-8") as f:
                    eval_data = json.load(f)
                if isinstance(eval_data, dict):
                    eval_data["audio_file"] = fname
                    with ctx["final_eval_p"].open("w", encoding="utf-8") as f:
                        json.dump(eval_data, f, ensure_ascii=False, indent=2)

            if clean_intermediate:
                import shutil
                for item in list(ctx["out_dir"].iterdir()):
                    if item.resolve() != ctx["final_eval_p"].resolve():
                        try:
                            if item.is_dir():
                                shutil.rmtree(item, ignore_errors=True)
                            else:
                                item.unlink(missing_ok=True)
                        except Exception as ce:
                            logger.warning(f"Could not remove {item.name}: {ce}")

            eval_entry = {
                "id": ctx["input_p"].stem,
                "filename": fname,
                "data": eval_data,
                "path": str(ctx["final_eval_p"]),
            }
            evaluations_result.append(eval_entry)

            if file_completed_callback:
                file_completed_callback(fname, eval_data)

            logger.info(f"Stage 8 complete -> {fname} deliverable ready.")

        except Exception as exc:
            logger.error(f"Stage 8 finalization failed for {fname}: {exc}", exc_info=True)
            ctx["failed"] = True
            ctx["error"] = f"Stage 8: {exc}"
            ctx["failed_stage"] = 8
            dest_dir = relocate_to_failed_dir(ctx)
            ctx["failed_dir"] = str(dest_dir)
            failed_files.append({
                "audio_file": fname,
                "error": ctx["error"],
                "failed_stage": 8,
                "failed_dir": str(dest_dir),
            })

    batch_summary = {
        "output_directory": str(out_base),
        "total_files": len(file_contexts),
        "successful": len(evaluations_result),
        "failed_count": len(failed_files),
        "failed_files": failed_files,
        "evaluations": evaluations_result,
        "total_elapsed_seconds": round(time.time() - batch_start, 2),
    }

    summary_path = out_base / "batch_processing_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(batch_summary, f, ensure_ascii=False, indent=2)

    logger.info(
        f"\n========================================\n"
        f"STAGE-BY-STAGE BATCH COMPLETE\n"
        f"Total: {len(file_contexts)} | Success: {len(evaluations_result)} | Failed: {len(failed_files)}\n"
        f"Total Time: {batch_summary['total_elapsed_seconds']}s\n"
        f"Summary: {summary_path}\n"
        f"========================================"
    )
    return batch_summary


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
    """Processes all audio files in an input folder using Stage-by-Stage batch execution."""
    in_dir = Path(input_folder).resolve()
    audio_files = find_audio_files(str(in_dir), recursive=recursive)
    if not audio_files:
        logger.warning(f"No audio files found in directory: {in_dir}")
        return {
            "total_files": 0,
            "processed": 0,
            "failed": 0,
            "summary_file": None,
        }

    return process_batch_by_stages(
        audio_files=audio_files,
        output_base_dir=output_base_dir,
        hf_token=hf_token,
        max_retries=max_retries,
        backoff_factor=backoff_factor,
        resume=resume,
        clean_intermediate=clean_intermediate,
        whisper_gpu_mem=whisper_gpu_mem,
        intent_gpu_mem=intent_gpu_mem,
        eval_gpu_mem=eval_gpu_mem,
        asr_lang=asr_lang,
        asr_decoder=asr_decoder,
    )
