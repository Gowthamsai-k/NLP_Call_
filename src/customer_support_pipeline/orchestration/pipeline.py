"""
Single-call pipeline orchestrator with dynamic memory adaptation.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional

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
from ..audio.preprocessor import convert_to_clean_wav
from ..audio.diarizer import run_diarization
from ..asr.whisper_engine import run_transcription
from ..asr.punctuator import run_punctuation_restoration
from ..asr.aligner import align_speakers_with_transcript
from ..nlp.intent_extractor import run_intent_extraction
from ..nlp.evaluator import run_call_evaluation
from ..nlp.agent_metrics import attach_agent_performance

logger = logging.getLogger(__name__)


def process_single_call(
    input_audio_path: str,
    output_dir: str,
    hf_token: Optional[str] = None,
    max_retries: int = DEFAULT_STAGE_RETRIES,
    backoff_factor: float = RETRY_BACKOFF_FACTOR,
    resume: bool = True,
    clean_intermediate: bool = True,
    reuse_llm: bool = True,
    whisper_gpu_mem: float = WHISPER_GPU_MEM,
    intent_gpu_mem: float = INTENT_GPU_MEM,
    eval_gpu_mem: float = EVAL_GPU_MEM,
    asr_lang: Optional[str] = None,
    asr_decoder: Optional[str] = None,
    stage_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    """Processes a single audio call end-to-end through all seven pipeline stages."""
    start_time = time.time()
    input_p = Path(input_audio_path).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_wav_p = out_dir / CLEAN_WAV_FILENAME
    metadata_p = out_dir / METADATA_FILENAME
    whisper_raw_p = out_dir / WHISPER_RAW_FILENAME
    punctuated_p = out_dir / PUNCTUATED_FILENAME
    final_transcript_p = out_dir / FINAL_TRANSCRIPT_FILENAME
    final_intent_p = out_dir / FINAL_INTENT_FILENAME
    final_eval_p = out_dir / FINAL_EVALUATION_FILENAME
    status_p = out_dir / PIPELINE_STATUS_FILENAME

    if resume and final_eval_p.exists() and final_eval_p.stat().st_size > 0:
        logger.info(f"Final evaluation score deliverable already exists in {out_dir}. Skipping.")
        return {
            "status": "success",
            "skipped": True,
            "audio_file": input_p.name,
            "output_dir": str(out_dir),
            "evaluation_path": str(final_eval_p),
            "elapsed_seconds": 0.0,
        }

    status_data = {
        "audio_file": input_p.name,
        "input_path": str(input_p),
        "output_dir": str(out_dir),
        "stages": {},
        "status": "in_progress",
    }

    shared_llm = None
    shared_tokenizer = None

    try:
        # Pre-execution VRAM cleanup
        release_memory()

        # STAGE 1: Audio Conversion
        if stage_callback:
            stage_callback(1, "FFmpeg Audio Conversion (16kHz PCM)")
        if not (resume and clean_wav_p.exists() and clean_wav_p.stat().st_size > 0):
            logger.info(">>> Stage 1: Converting audio to clean mono WAV...")
            t0 = time.time()
            retry_with_memory_cleanup(
                convert_to_clean_wav,
                str(input_p),
                str(clean_wav_p),
                max_retries=max_retries,
                backoff_factor=backoff_factor,
                stage_name="Stage 1 (FFmpeg Audio Conversion)",
            )
            status_data["stages"]["audio_conversion"] = {"status": "success", "duration_s": round(time.time() - t0, 2)}
        else:
            logger.info(">>> Stage 1: Clean WAV already exists, resuming.")
            status_data["stages"]["audio_conversion"] = {"status": "resumed"}

        # STAGE 2: Speaker Diarization
        if stage_callback:
            stage_callback(2, "Speaker Diarization (PyAnnote)")
        if not (resume and metadata_p.exists() and metadata_p.stat().st_size > 0):
            logger.info(">>> Stage 2: Speaker Diarization...")
            t0 = time.time()
            retry_with_memory_cleanup(
                run_diarization,
                str(clean_wav_p),
                str(metadata_p),
                hf_token=hf_token,
                max_retries=max_retries,
                backoff_factor=backoff_factor,
                stage_name="Stage 2 (Diarization)",
            )
            status_data["stages"]["diarization"] = {"status": "success", "duration_s": round(time.time() - t0, 2)}
        else:
            logger.info(">>> Stage 2: Metadata already exists, resuming.")
            status_data["stages"]["diarization"] = {"status": "resumed"}

        # Force full memory release after diarization
        release_memory(delay=1.0)

        # STAGE 3: Audio Transcription
        if stage_callback:
            stage_callback(3, "ASR Transcription (AI4Bharat Indic-Conformer)")
        if not (resume and whisper_raw_p.exists() and whisper_raw_p.stat().st_size > 0):
            logger.info(">>> Stage 3: ASR Transcription (AI4Bharat Indic-Conformer)...")
            t0 = time.time()
            retry_with_memory_cleanup(
                run_transcription,
                str(clean_wav_p),
                str(metadata_p),
                str(whisper_raw_p),
                gpu_memory_utilization=whisper_gpu_mem,
                language=asr_lang,
                decoder=asr_decoder,
                max_retries=max_retries,
                backoff_factor=backoff_factor,
                stage_name="Stage 3 (Indic-Conformer ASR Transcription)",
            )
            status_data["stages"]["transcription"] = {"status": "success", "duration_s": round(time.time() - t0, 2)}
        else:
            logger.info(">>> Stage 3: Raw transcript already exists, resuming.")
            status_data["stages"]["transcription"] = {"status": "resumed"}

        # Force full memory release after Indic-Conformer ASR
        release_memory(delay=1.0)

        # STAGE 4: Punctuation Restoration
        if stage_callback:
            stage_callback(4, "Punctuation Restoration (Horizon-Labs)")
        if not (resume and punctuated_p.exists() and punctuated_p.stat().st_size > 0):
            logger.info(">>> Stage 4: Punctuation Restoration...")
            t0 = time.time()
            retry_with_memory_cleanup(
                run_punctuation_restoration,
                str(whisper_raw_p),
                str(punctuated_p),
                max_retries=max_retries,
                backoff_factor=backoff_factor,
                stage_name="Stage 4 (Punctuation Restoration)",
            )
            status_data["stages"]["punctuation"] = {"status": "success", "duration_s": round(time.time() - t0, 2)}
        else:
            logger.info(">>> Stage 4: Punctuated transcript already exists, resuming.")
            status_data["stages"]["punctuation"] = {"status": "resumed"}

        release_memory()

        # STAGE 5: Alignment
        if stage_callback:
            stage_callback(5, "Speaker Alignment & Role Attribution")
        if not (resume and final_transcript_p.exists() and final_transcript_p.stat().st_size > 0):
            logger.info(">>> Stage 5: Speaker Alignment...")
            t0 = time.time()
            align_speakers_with_transcript(
                str(metadata_p),
                str(punctuated_p),
                str(final_transcript_p),
            )
            status_data["stages"]["alignment"] = {"status": "success", "duration_s": round(time.time() - t0, 2)}
        else:
            logger.info(">>> Stage 5: Speaker-aligned transcript already exists, resuming.")
            status_data["stages"]["alignment"] = {"status": "resumed"}

        release_memory()

        # STAGES 6 & 7: Unified Single-Pass Qwen Trajectory Evaluation & Intent Extraction
        both_exist = (
            resume
            and final_intent_p.exists()
            and final_intent_p.stat().st_size > 0
            and final_eval_p.exists()
            and final_eval_p.stat().st_size > 0
        )
        if both_exist:
            logger.info(">>> Stages 6 & 7: Unified intent and call evaluation already exist, resuming.")
            status_data["stages"]["intent_extraction"] = {"status": "resumed"}
            status_data["stages"]["evaluation"] = {"status": "resumed"}
            if stage_callback:
                stage_callback(6, "Qwen Intent Extraction & Client Needs (Resumed)")
                stage_callback(7, "Qwen Quality Scoring & Rubric Evaluation (Resumed)")
        else:
            if stage_callback:
                stage_callback(6, "Qwen Unified Trajectory Evaluation (Single Pass)")
            logger.info(">>> Stages 6 & 7: Executing Unified Single-Pass Trajectory Evaluation...")
            t0 = time.time()
            try:
                retry_with_memory_cleanup(
                    run_call_evaluation,
                    str(final_transcript_p),
                    intent_or_path=str(final_intent_p) if final_intent_p.exists() else None,
                    output_path=str(final_eval_p),
                    intent_output_path=str(final_intent_p),
                    work_dir=str(out_dir / "call_evaluation_batches"),
                    existing_llm=None,
                    existing_tokenizer=None,
                    release_llm_after=True,
                    gpu_memory_utilization=eval_gpu_mem,
                    max_retries=max_retries,
                    backoff_factor=backoff_factor,
                    stage_name="Stages 6 & 7 (Unified Single-Pass Evaluation)",
                )
            except Exception as eval_exc:
                logger.error(f"Unified evaluation failed: {eval_exc}", exc_info=True)
                raise eval_exc

            dur = round(time.time() - t0, 2)
            status_data["stages"]["intent_extraction"] = {"status": "success", "duration_s": dur}
            status_data["stages"]["evaluation"] = {"status": "success", "duration_s": dur}
            if stage_callback:
                stage_callback(7, "Qwen Quality Scoring & Rubric Evaluation (Completed)")

        # STAGE 8: Deliverable Cleanup (keep only final score evaluation)
        if stage_callback:
            stage_callback(8, "Score Deliverable Packaging & Cleanup")
        removed_files = []
        # Annotate final_eval_p with audio_file name and diarization_timeline if missing
        try:
            if final_eval_p.exists():
                with final_eval_p.open("r", encoding="utf-8") as f:
                    eval_data = json.load(f)
                if isinstance(eval_data, dict):
                    eval_data["audio_file"] = input_p.name
                    dt = eval_data.get("diarization_timeline")
                    if not dt or not dt.get("turns"):
                        try:
                            segments = []
                            if metadata_p.exists():
                                with metadata_p.open("r", encoding="utf-8") as mf:
                                    segments = json.load(mf).get("segments") or []
                            elif final_transcript_p.exists():
                                with final_transcript_p.open("r", encoding="utf-8") as tf:
                                    segments = json.load(tf) or []
                            agent_id = None
                            client_id = None
                            if final_intent_p.exists():
                                with final_intent_p.open("r", encoding="utf-8") as inf:
                                    roles = json.load(inf).get("speaker_roles", {})
                                    agent_id = roles.get("agent", {}).get("speaker_id")
                                    client_id = roles.get("client", {}).get("speaker_id")
                            if segments:
                                timeline_turns = []
                                a_talk = 0.0
                                c_talk = 0.0
                                t_dur = 0.0
                                for s in segments:
                                    st = round(float(s.get("start", 0.0)), 2)
                                    en = round(float(s.get("end", st)), 2)
                                    dur = max(0.0, round(en - st, 2))
                                    spk = str(s.get("speaker_id") or "UNKNOWN")
                                    role = "agent" if (agent_id and spk == agent_id) else ("client" if (client_id and spk == client_id) else "other")
                                    if role == "agent":
                                        a_talk += dur
                                    elif role == "client":
                                        c_talk += dur
                                    if en > t_dur:
                                        t_dur = en
                                    timeline_turns.append({"start": st, "end": en, "duration": dur, "speaker_id": spk, "role": role})
                                total_sp = a_talk + c_talk
                                eval_data["diarization_timeline"] = {
                                    "total_duration_seconds": round(t_dur, 2),
                                    "agent_speaker_id": agent_id,
                                    "client_speaker_id": client_id,
                                    "agent_talk_time_seconds": round(a_talk, 2),
                                    "client_talk_time_seconds": round(c_talk, 2),
                                    "agent_talk_ratio_pct": round(a_talk / total_sp * 100, 1) if total_sp > 0 else 55.0,
                                    "client_talk_ratio_pct": round(c_talk / total_sp * 100, 1) if total_sp > 0 else 45.0,
                                    "total_turns": len(timeline_turns),
                                    "turns": timeline_turns,
                                }
                        except Exception as dt_err:
                            logger.warning(f"Could not build diarization timeline in Stage 8: {dt_err}")
                    attach_agent_performance(eval_data, str(clean_wav_p) if clean_wav_p.exists() else None)
                    with final_eval_p.open("w", encoding="utf-8") as f:
                        json.dump(eval_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Could not annotate audio_file to evaluation deliverable: {e}")

        if clean_intermediate:
            import shutil
            for item in list(out_dir.iterdir()):
                if item.resolve() != final_eval_p.resolve():
                    try:
                        if item.is_dir():
                            shutil.rmtree(item, ignore_errors=True)
                        else:
                            item.unlink(missing_ok=True)
                        removed_files.append(item.name)
                    except Exception as e:
                        logger.warning(f"Could not remove {item.name}: {e}")

        status_data["status"] = "success"
        status_data["removed_intermediates"] = removed_files
        status_data["elapsed_seconds"] = round(time.time() - start_time, 2)
        status_data["deliverables"] = {
            "evaluation": str(final_eval_p),
        }

        logger.info(
            f"Pipeline complete for {input_p.name} in {status_data['elapsed_seconds']}s -> {out_dir}"
        )
        return status_data

    except Exception as exc:
        status_data["status"] = "failed"
        status_data["error"] = f"{type(exc).__name__}: {str(exc)}"
        status_data["elapsed_seconds"] = round(time.time() - start_time, 2)

        # Place failed record in failed/ directory
        try:
            import shutil
            if out_dir.name == "output":
                failed_dest_dir = out_dir / "failed" / input_p.stem
            else:
                failed_dest_dir = out_dir.parent / "failed" / out_dir.name
            failed_dest_dir.mkdir(parents=True, exist_ok=True)

            failure_info = {
                "audio_file": input_p.name,
                "input_path": str(input_p),
                "status": "failed",
                "error": status_data["error"],
                "elapsed_seconds": status_data["elapsed_seconds"],
                "failed_directory": str(failed_dest_dir),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with (failed_dest_dir / "failure_reason.json").open("w", encoding="utf-8") as f:
                json.dump(failure_info, f, ensure_ascii=False, indent=2)

            with (failed_dest_dir / "error.txt").open("w", encoding="utf-8") as f:
                f.write(
                    f"Audio File: {input_p.name}\n"
                    f"Error: {status_data['error']}\n"
                    f"Time: {failure_info['timestamp']}\n"
                )

            # Move any intermediate files created in out_dir to failed_dest_dir
            if out_dir.resolve() != failed_dest_dir.resolve() and out_dir.exists():
                for item in list(out_dir.iterdir()):
                    target_dest = failed_dest_dir / item.name
                    if target_dest.resolve() in (
                        (failed_dest_dir / "failure_reason.json").resolve(),
                        (failed_dest_dir / "error.txt").resolve(),
                    ):
                        continue
                    if target_dest.exists():
                        if target_dest.is_dir():
                            shutil.rmtree(target_dest, ignore_errors=True)
                        else:
                            target_dest.unlink(missing_ok=True)
                    shutil.move(str(item), str(target_dest))
                shutil.rmtree(out_dir, ignore_errors=True)

            status_data["failed_dir"] = str(failed_dest_dir)
        except Exception as move_err:
            logger.warning(f"Could not relocate failed record to failed directory: {move_err}")
            try:
                with status_p.open("w", encoding="utf-8") as f:
                    json.dump(status_data, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
        raise exc

    finally:
        release_memory(shared_llm, shared_tokenizer)
        log_vram("End of Pipeline Run")
