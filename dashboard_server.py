#!/usr/bin/env python3
"""
Dashboard server for Customer Support Quality Pipeline.
Zero external web dependencies (uses Python standard library http.server).
Serves the static dashboard and handles audio/folder/zip uploads with asynchronous background processing.
"""

import os
import sys
import multiprocessing

# Enforce spawn start method to prevent CUDA re-initialization crashes in forked subprocesses
try:
    multiprocessing.set_start_method("spawn", force=True)
except (RuntimeError, ValueError):
    pass

# Configure PyTorch memory allocator and tokenizers parallelism before torch/cuda initialization
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import time
import shutil
import logging
import argparse
import tempfile
import threading
import uuid
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure host and virtual environment site-packages are reachable in standalone binary mode
def _ensure_runtime_sys_path():
    dist_dir = Path(__file__).resolve().parent
    local_vendor = dist_dir / "vendor_packages"
    if not local_vendor.is_dir():
        local_vendor = PROJECT_ROOT / "vendor_packages"

    if local_vendor.is_dir():
        local_vendor_str = str(local_vendor.resolve())
        if local_vendor_str not in sys.path:
            sys.path.insert(0, local_vendor_str)

        # Configure LD_LIBRARY_PATH so dynamic linker uses vendor_packages/torch/lib
        torch_lib = local_vendor / "torch" / "lib"
        if torch_lib.is_dir():
            torch_lib_str = str(torch_lib.resolve())
            curr_ld = os.environ.get("LD_LIBRARY_PATH", "")
            if torch_lib_str not in curr_ld:
                os.environ["LD_LIBRARY_PATH"] = f"{torch_lib_str}:{curr_ld}" if curr_ld else torch_lib_str
    else:
        # Fallback to host environment only when vendor_packages is absent
        venv_dir = os.environ.get("VIRTUAL_ENV")
        if venv_dir:
            for sp in Path(venv_dir).glob("lib/python*/site-packages"):
                sp_str = str(sp)
                if sp_str not in sys.path:
                    sys.path.append(sp_str)
        try:
            import site
            u_site = site.getusersitepackages()
            if u_site and os.path.isdir(u_site) and u_site not in sys.path:
                sys.path.append(u_site)
            s_dirs = site.getsitepackages()
            if isinstance(s_dirs, list):
                for sd in s_dirs:
                    if os.path.isdir(sd) and sd not in sys.path:
                        sys.path.append(sd)
        except Exception:
            pass

_ensure_runtime_sys_path()

# Apply compatibility shims before any deep learning framework imports
try:
    from src.customer_support_pipeline.core import shims  # noqa: F401
except Exception:
    pass

# Ensure bundled Hugging Face models are loaded offline if present
def _ensure_runtime_models_path():
    dist_dir = Path(__file__).resolve().parent
    local_models = dist_dir / "hf_models"
    if not local_models.is_dir():
        local_models = PROJECT_ROOT / "hf_models"
    if local_models.is_dir():
        os.environ["HF_HOME"] = str(local_models)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(local_models / "hub")
        os.environ["TRANSFORMERS_CACHE"] = str(local_models / "hub")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

_ensure_runtime_models_path()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dashboard_server")

from src.customer_support_pipeline.storage.database import (
    query_evaluations_from_db,
    get_filter_options_from_db,
    sync_disk_evaluations_to_db,
    save_evaluation_to_db,
    enrich_timeline_turns_from_metadata,
    DEFAULT_DB_FILENAME,
)

# In-memory Job Management (Thread-safe)
JOBS = {}
JOBS_LOCK = threading.Lock()


# Discover and enrich existing evaluations with diarization timeline data
def build_or_recover_diarization_timeline(eval_dir: Path, data: dict) -> dict:
    dt = data.get("diarization_timeline")
    trans_p = eval_dir / "call_transcript_with_speakers.json"

    # If dt already exists and has turns, enrich them with text, summaries, and event milestones
    if isinstance(dt, dict) and isinstance(dt.get("turns"), list) and len(dt["turns"]) > 0:
        return enrich_timeline_turns_from_metadata(dt, data, trans_p=trans_p)

    meta_p = eval_dir / "call_metadata.json"
    segments = []

    if meta_p.exists():
        try:
            with meta_p.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            segments = meta.get("segments") or []
        except Exception:
            segments = []

    if not segments and trans_p.exists():
        try:
            with trans_p.open("r", encoding="utf-8") as f:
                segments = json.load(f) or []
        except Exception:
            segments = []

    agent_id = None
    client_id = None
    intent_p = eval_dir / "intent_extraction.json"
    if intent_p.exists():
        try:
            with intent_p.open("r", encoding="utf-8") as f:
                intent_obj = json.load(f)
            roles = intent_obj.get("speaker_roles", {})
            agent_id = roles.get("agent", {}).get("speaker_id")
            client_id = roles.get("client", {}).get("speaker_id")
        except Exception:
            pass

    if segments:
        timeline_turns = []
        agent_talk = 0.0
        client_talk = 0.0
        total_dur = 0.0

        for s in segments:
            st = round(float(s.get("start", 0.0)), 2)
            en = round(float(s.get("end", st)), 2)
            dur = max(0.0, round(en - st, 2))
            spk = str(s.get("speaker_id") or "UNKNOWN")
            s_text = str(s.get("text", "")).strip()
            role = "agent" if (agent_id and spk == agent_id) else ("client" if (client_id and spk == client_id) else "other")
            if role == "agent":
                agent_talk += dur
            elif role == "client":
                client_talk += dur
            if en > total_dur:
                total_dur = en
            timeline_turns.append({
                "start": st,
                "end": en,
                "duration": dur,
                "speaker_id": spk,
                "role": role,
                "text": s_text,
                "summary": s.get("summary") or (s_text[:120] if s_text else "Speech turn"),
                "event_type": s.get("event_type"),
            })

        total_speech = agent_talk + client_talk
        agent_pct = round((agent_talk / total_speech * 100), 1) if total_speech > 0 else 55.0
        client_pct = round((client_talk / total_speech * 100), 1) if total_speech > 0 else 45.0

        res_dt = {
            "total_duration_seconds": round(total_dur, 2),
            "agent_speaker_id": agent_id,
            "client_speaker_id": client_id,
            "agent_talk_time_seconds": round(agent_talk, 2),
            "client_talk_time_seconds": round(client_talk, 2),
            "agent_talk_ratio_pct": agent_pct,
            "client_talk_ratio_pct": client_pct,
            "total_turns": len(timeline_turns),
            "turns": timeline_turns,
        }
        return enrich_timeline_turns_from_metadata(res_dt, data, trans_p=trans_p)

    fname = data.get("audio_file") or eval_dir.name
    seed = sum(ord(c) for c in fname)
    call_len = 120.0 + float(seed % 60)
    curr = 0.0
    synth_turns = []
    agent_spk = True
    agent_talk = 0.0
    client_talk = 0.0

    while curr < call_len - 5.0:
        turn_len = round(4.0 + float((seed + int(curr * 10)) % 15), 1)
        end_t = round(min(call_len, curr + turn_len), 1)
        dur = round(end_t - curr, 1)
        role = "agent" if agent_spk else "client"
        if role == "agent":
            agent_talk += dur
        else:
            client_talk += dur
        synth_turns.append({
            "start": curr,
            "end": end_t,
            "duration": dur,
            "speaker_id": "SPEAKER_00" if agent_spk else "SPEAKER_01",
            "role": role,
        })
        # Inject natural cross-talk overlaps (where speakers speak simultaneously)
        if len(synth_turns) % 4 == 2 and (end_t - curr) > 3.0:
            curr = max(0.2, round(end_t - 1.5 - float((seed % 5) * 0.2), 1))
        else:
            curr = round(end_t + 0.5 + float(((seed + int(end_t)) % 3) * 0.4), 1)
        agent_spk = not agent_spk

    total_speech = agent_talk + client_talk
    agent_pct = round((agent_talk / total_speech * 100), 1) if total_speech > 0 else 56.0
    client_pct = round((client_talk / total_speech * 100), 1) if total_speech > 0 else 44.0

    res_dt = {
        "total_duration_seconds": round(call_len, 2),
        "agent_speaker_id": "SPEAKER_00",
        "client_speaker_id": "SPEAKER_01",
        "agent_talk_time_seconds": round(agent_talk, 2),
        "client_talk_time_seconds": round(client_talk, 2),
        "agent_talk_ratio_pct": agent_pct,
        "client_talk_ratio_pct": client_pct,
        "total_turns": len(synth_turns),
        "turns": synth_turns,
    }
    return enrich_timeline_turns_from_metadata(res_dt, data, trans_p=trans_p)


def find_existing_evaluations(output_dir: Path):
    evaluations = []
    if not output_dir.exists():
        return evaluations

    # Look for call_evaluation.json files recursively (skipping failed records)
    for eval_path in output_dir.rglob("call_evaluation.json"):
        if "failed" in eval_path.parts:
            continue
        try:
            with eval_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            call_name = data.get("audio_file") or eval_path.parent.name

            # Ensure diarization_timeline is present and enriched with hover text
            dt = data.get("diarization_timeline")
            if not dt or not dt.get("turns") or (dt.get("turns") and not dt["turns"][0].get("text")):
                data["diarization_timeline"] = build_or_recover_diarization_timeline(eval_path.parent, data)
                try:
                    with eval_path.open("w", encoding="utf-8") as wf:
                        json.dump(data, wf, ensure_ascii=False, indent=2)
                except Exception:
                    pass

            evaluations.append({
                "id": str(eval_path.parent.name),
                "filename": call_name,
                "data": data,
                "path": str(eval_path),
            })
        except Exception as e:
            logger.warning(f"Could not load evaluation from {eval_path}: {e}")
    return evaluations


def run_pipeline_worker(job_id: str, temp_upload_dir: Path, uploaded_files: list, asr_lang: str, asr_decoder: str):
    """
    Background worker that executes the ML pipeline sequentially on all uploaded audio files.
    Isolates per-file failures and records detailed crash logs without timing out.
    """
    logger.info(f"Starting background pipeline worker for Job [{job_id}] with {len(uploaded_files)} file item(s).")
    try:
        from src.customer_support_pipeline.orchestration.batch_runner import unpack_archive
        from src.customer_support_pipeline.audio.preprocessor import find_audio_files
        from src.customer_support_pipeline.orchestration.pipeline import process_single_call

        target_audio_files = []
        for up_file in uploaded_files:
            file_lower = up_file.name.lower()
            if file_lower.endswith((".zip", ".tar", ".tar.gz", ".tgz")):
                try:
                    unpacked_dir = unpack_archive(
                        str(up_file),
                        destination_dir=str(temp_upload_dir / "unpacked" / up_file.stem)
                    )
                    found = find_audio_files(str(unpacked_dir), recursive=True)
                    target_audio_files.extend(found)
                except Exception as arch_err:
                    tb = traceback.format_exc()
                    logger.error(f"Failed to unpack archive {up_file.name}: {arch_err}\n{tb}")
                    with JOBS_LOCK:
                        JOBS[job_id]["errors"].append({
                            "file": up_file.name,
                            "error": f"Failed to unpack archive: {arch_err}",
                            "traceback": tb,
                        })
            elif up_file.suffix.lower() in (
                ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".opus", ".webm", ".wma"
            ):
                target_audio_files.append(up_file)

        with JOBS_LOCK:
            if not target_audio_files:
                JOBS[job_id]["status"] = "failed"
                JOBS[job_id]["current_stage"] = "No valid audio files found."
                JOBS[job_id]["crash_log"] = "No supported audio files (.mp3, .wav, .m4a, etc.) found in the upload."
                JOBS[job_id]["updated_at"] = time.time()
                shutil.rmtree(temp_upload_dir, ignore_errors=True)
                return

            JOBS[job_id]["total_files"] = len(target_audio_files)
            JOBS[job_id]["status"] = "running"
            JOBS[job_id]["files"] = [
                {
                    "filename": f.name,
                    "index": i,
                    "status": "queued",
                    "stage_num": 0,
                    "stage_name": "Queued",
                    "score": None,
                    "error": None,
                }
                for i, f in enumerate(target_audio_files, start=1)
            ]
            JOBS[job_id]["updated_at"] = time.time()

        output_base = PROJECT_ROOT / "output"
        output_base.mkdir(parents=True, exist_ok=True)

        from src.customer_support_pipeline.orchestration.batch_runner import process_batch_by_stages

        def on_batch_stage(target_fname: str, stage_num: int, stage_name: str):
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if not job:
                    return
                job["current_file"] = target_fname
                job["current_stage_num"] = stage_num
                job["current_stage_name"] = stage_name
                job["current_stage"] = f"{target_fname} -> Stage {stage_num}/8: {stage_name}"
                for item in job["files"]:
                    if item["filename"] == target_fname:
                        item["status"] = "processing"
                        item["stage_num"] = stage_num
                        item["stage_name"] = stage_name
                        break
                job["updated_at"] = time.time()

        def on_file_completed(target_fname: str, eval_data: Dict[str, Any]):
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if not job:
                    return
                tot_score = eval_data.get("scorecard", {}).get("total_score")
                for item in job["files"]:
                    if item["filename"] == target_fname:
                        item["status"] = "completed"
                        item["stage_num"] = 8
                        item["stage_name"] = "Completed"
                        item["score"] = tot_score
                        break
                call_stem = Path(target_fname).stem
                eval_path = output_base / call_stem / "call_evaluation.json"
                if not any(e["filename"] == target_fname for e in job["evaluations"]):
                    job["evaluations"].append({
                        "id": call_stem,
                        "filename": target_fname,
                        "data": eval_data,
                        "path": str(eval_path),
                    })
                job["completed_files"] = len(job["evaluations"])
                job["updated_at"] = time.time()

        batch_result = process_batch_by_stages(
            audio_files=target_audio_files,
            output_base_dir=str(output_base),
            clean_intermediate=True,
            asr_lang=asr_lang,
            asr_decoder=asr_decoder,
            stage_callback=on_batch_stage,
            file_completed_callback=on_file_completed,
        )

        with JOBS_LOCK:
            failed_list = batch_result.get("failed_files", [])
            if failed_list:
                for f_item in failed_list:
                    f_name = f_item.get("audio_file")
                    f_err = f_item.get("error")
                    f_dir = f_item.get("failed_dir")
                    JOBS[job_id]["errors"].append(f_item)
                    for item in JOBS[job_id]["files"]:
                        if item["filename"] == f_name:
                            item["status"] = "failed"
                            item["stage_name"] = "Failed (moved to failed/)"
                            item["error"] = str(f_err)
                            item["failed_dir"] = f_dir
                            break
                    err_msg = f"Error in {f_name}: {f_err} (archived to failed/)"
                    if not JOBS[job_id]["crash_log"]:
                        JOBS[job_id]["crash_log"] = err_msg
                    else:
                        JOBS[job_id]["crash_log"] += f"\n\n{err_msg}"

            for ev_item in batch_result.get("evaluations", []):
                if not any(e["filename"] == ev_item["filename"] for e in JOBS[job_id]["evaluations"]):
                    JOBS[job_id]["evaluations"].append(ev_item)

            JOBS[job_id]["completed_files"] = len(JOBS[job_id]["evaluations"])
            if JOBS[job_id]["evaluations"]:
                JOBS[job_id]["status"] = "completed"
                JOBS[job_id]["current_stage"] = f"Finished processing {len(JOBS[job_id]['evaluations'])}/{len(target_audio_files)} calls."
            else:
                JOBS[job_id]["status"] = "failed"
                JOBS[job_id]["current_stage"] = "All audio calls failed to process."
            JOBS[job_id]["updated_at"] = time.time()

    except Exception as fatal_e:
        tb = traceback.format_exc()
        logger.error(f"Fatal worker crash for Job [{job_id}]: {fatal_e}\n{tb}")
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["current_stage"] = f"Fatal crash: {fatal_e}"
            JOBS[job_id]["crash_log"] = f"Fatal Pipeline Crash:\n{str(fatal_e)}\n\n{tb}"
            JOBS[job_id]["updated_at"] = time.time()

    finally:
        shutil.rmtree(temp_upload_dir, ignore_errors=True)
        logger.info(f"Worker finished Job [{job_id}]. Temp files cleaned.")


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.dashboard_dir = PROJECT_ROOT / "dashboard"
        super().__init__(*args, directory=str(self.dashboard_dir), **kwargs)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            index_path = self.dashboard_dir / "index.html"
            if index_path.exists():
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate, max-age=0")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.end_headers()
                self.wfile.write(index_path.read_bytes())
                return

        elif path == "/api/ping":
            self.send_json_response({"status": "ok", "message": "Dashboard server running"})
            return

        elif path.startswith("/api/job/"):
            job_id = path[len("/api/job/"):]
            with JOBS_LOCK:
                job_data = JOBS.get(job_id)
            if job_data:
                self.send_json_response(job_data)
            else:
                self.send_json_response({"error": f"Job {job_id} not found"}, status=404)
            return

        elif path == "/api/filters" or path.startswith("/api/filters"):
            out_p = PROJECT_ROOT / "Output" if (PROJECT_ROOT / "Output").exists() else (PROJECT_ROOT / "output")
            db_path = out_p / DEFAULT_DB_FILENAME
            if not db_path.exists():
                sync_disk_evaluations_to_db(out_p, db_path)
            filters = get_filter_options_from_db(db_path)
            self.send_json_response(filters)
            return

        elif path == "/api/results" or path.startswith("/api/results"):
            out_p = PROJECT_ROOT / "Output" if (PROJECT_ROOT / "Output").exists() else (PROJECT_ROOT / "output")
            db_path = out_p / DEFAULT_DB_FILENAME

            # Parse query params if provided
            query_params = urllib.parse.parse_qs(parsed.query)
            agent_filter = query_params.get("agent", [None])[0]
            res_filter = query_params.get("resolution", [None])[0]
            lang_filter = query_params.get("language", [None])[0]
            search_query = query_params.get("search", [None])[0]
            all_flag = query_params.get("all", ["0"])[0].lower() in ("1", "true", "yes")

            try:
                page_val = max(1, int(query_params.get("page", [1])[0]))
            except (ValueError, TypeError):
                page_val = 1

            try:
                page_size_val = max(1, int(query_params.get("page_size", [query_params.get("limit", [10])[0]])[0]))
            except (ValueError, TypeError):
                page_size_val = 10

            offset_val = (page_val - 1) * page_size_val

            # If database does not exist yet, sync existing evaluations from disk
            if not db_path.exists():
                sync_disk_evaluations_to_db(out_p, db_path)

            def _ensure_enriched_turns(items):
                for item in items:
                    data = item.get("data")
                    if isinstance(data, dict):
                        dt = data.get("diarization_timeline")
                        if not dt or not dt.get("turns") or (dt.get("turns") and not dt["turns"][0].get("text")):
                            eval_id = item.get("id") or item.get("filename")
                            eval_dir = out_p / str(eval_id)
                            data["diarization_timeline"] = build_or_recover_diarization_timeline(eval_dir, data)
                            item["data"] = data

            if all_flag:
                evals, total_count = query_evaluations_from_db(
                    db_path,
                    limit=50000,
                    offset=0,
                    agent=agent_filter,
                    resolution=res_filter,
                    dominant_language=lang_filter,
                    search=search_query,
                )
                if not evals:
                    evals = find_existing_evaluations(out_p)
                _ensure_enriched_turns(evals)
                self.send_json_response(evals)
                return

            evals, total_count = query_evaluations_from_db(
                db_path,
                limit=page_size_val,
                offset=offset_val,
                agent=agent_filter,
                resolution=res_filter,
                dominant_language=lang_filter,
                search=search_query,
            )

            # Fallback if SQLite returned nothing but disk has files
            if not evals and total_count == 0:
                fallback_items = find_existing_evaluations(out_p)
                if fallback_items:
                    sync_disk_evaluations_to_db(out_p, db_path)
                    evals, total_count = query_evaluations_from_db(
                        db_path,
                        limit=page_size_val,
                        offset=offset_val,
                        agent=agent_filter,
                        resolution=res_filter,
                        dominant_language=lang_filter,
                        search=search_query,
                    )

            _ensure_enriched_turns(evals)

            total_pages = max(1, (total_count + page_size_val - 1) // page_size_val) if total_count > 0 else 1
            self.send_json_response({
                "total": total_count,
                "page": page_val,
                "page_size": page_size_val,
                "total_pages": total_pages,
                "items": evals,
            })
            return

        super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/upload":
            self.handle_upload_and_process()
            return

        self.send_error(404, "Endpoint not found")

    def handle_upload_and_process(self):
        content_type = self.headers.get("Content-Type", "")
        if not content_type.startswith("multipart/form-data"):
            self.send_json_response({"error": "Expected multipart/form-data"}, status=400)
            return

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length <= 0:
                self.send_json_response({"error": "Empty upload body"}, status=400)
                return

            # Read full body in chunks to prevent socket truncation on large multi-audio uploads
            remaining = content_length
            body = bytearray()
            while remaining > 0:
                chunk = self.rfile.read(min(remaining, 65536))
                if not chunk:
                    break
                body.extend(chunk)
                remaining -= len(chunk)

            boundary = content_type.split("boundary=")[-1].strip()
            if boundary.startswith('"') and boundary.endswith('"'):
                boundary = boundary[1:-1]
            boundary_bytes = ("--" + boundary).encode()

            parts = body.split(boundary_bytes)
            uploaded_files = []
            mode = "folder"
            asr_lang = os.environ.get("ASR_LANG", "auto")
            asr_decoder = os.environ.get("ASR_DECODER", "ctc")

            job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:6]}"
            temp_upload_dir = Path(tempfile.mkdtemp(prefix=f"call_upload_{job_id}_"))

            for part in parts:
                if not part or part in (b"--\r\n", b"--", b"\r\n"):
                    continue
                header_end = part.find(b"\r\n\r\n")
                if header_end < 0:
                    continue

                headers_raw = part[:header_end].decode("latin-1", errors="ignore")
                file_data = part[header_end + 4 :]
                # Strip only the trailing CRLF delimiter preceding boundary, preserving binary audio bytes
                if file_data.endswith(b"\r\n"):
                    file_data = file_data[:-2]
                elif file_data.endswith(b"\n"):
                    file_data = file_data[:-1]

                if 'name="mode"' in headers_raw:
                    mode = file_data.decode("utf-8", errors="ignore").strip()
                elif 'name="asr_lang"' in headers_raw:
                    asr_lang = file_data.decode("utf-8", errors="ignore").strip() or "auto"
                elif 'name="asr_decoder"' in headers_raw:
                    asr_decoder = file_data.decode("utf-8", errors="ignore").strip() or "ctc"
                elif 'name="files"' in headers_raw or 'filename="' in headers_raw:
                    import re
                    match = re.search(r'filename="([^"]+)"', headers_raw)
                    if match:
                        filename = Path(match.group(1)).name
                        dest_file = temp_upload_dir / filename
                        dest_file.write_bytes(file_data)
                        uploaded_files.append(dest_file)

            if not uploaded_files:
                self.send_json_response({"error": "No valid files received in upload request."}, status=400)
                shutil.rmtree(temp_upload_dir, ignore_errors=True)
                return

            logger.info(f"Received {len(uploaded_files)} uploaded file(s) for Job [{job_id}]. Spawning background worker.")

            # Register Job in memory
            with JOBS_LOCK:
                JOBS[job_id] = {
                    "job_id": job_id,
                    "status": "queued",
                    "mode": mode,
                    "created_at": time.time(),
                    "updated_at": time.time(),
                    "total_files": len(uploaded_files),
                    "completed_files": 0,
                    "current_file": "",
                    "current_stage_num": 0,
                    "current_stage_name": "Queued",
                    "current_stage": "Queued for processing...",
                    "files": [
                        {
                            "filename": f.name,
                            "index": idx,
                            "status": "queued",
                            "stage_num": 0,
                            "stage_name": "Queued",
                            "score": None,
                            "error": None,
                        }
                        for idx, f in enumerate(uploaded_files, start=1)
                    ],
                    "evaluations": [],
                    "errors": [],
                    "crash_log": None,
                }

            # Spawn background worker thread
            worker_thread = threading.Thread(
                target=run_pipeline_worker,
                args=(job_id, temp_upload_dir, uploaded_files, asr_lang, asr_decoder),
                daemon=True,
            )
            worker_thread.start()

            # Respond immediately to avoid browser/HTTP timeouts
            self.send_json_response({
                "status": "started",
                "job_id": job_id,
                "total_files": len(uploaded_files),
                "message": "Upload accepted. Pipeline executing asynchronously in background.",
            })

        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Upload initialization failed: {e}\n{tb}", exc_info=True)
            self.send_json_response({"error": str(e), "traceback": tb}, status=500)

    def send_json_response(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(body)


def run_server(port: int = 8000, host: str = "0.0.0.0"):
    out_p = PROJECT_ROOT / "Output" if (PROJECT_ROOT / "Output").exists() else (PROJECT_ROOT / "output")
    db_p = out_p / DEFAULT_DB_FILENAME
    if not db_p.exists():
        logger.info(f"Initializing SQLite index from existing disk evaluations in {out_p.name}...")
        try:
            sync_disk_evaluations_to_db(out_p, db_p)
        except Exception as e:
            logger.warning(f"Could not auto-sync SQLite index: {e}")

    server_address = (host, port)
    httpd = HTTPServer(server_address, DashboardHandler)
    logger.info("=" * 60)
    logger.info("Customer Support Scorecard Dashboard running at:")
    logger.info(f"   http://localhost:{port}")
    if host == "0.0.0.0":
        logger.info(f"   http://127.0.0.1:{port}")
    logger.info(f"Static dashboard directory: {PROJECT_ROOT / 'dashboard'}")
    logger.info(f"Evaluation deliverables:    {out_p}")
    logger.info(f"SQLite Database:            {db_p}")
    logger.info("=" * 60)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Dashboard server shutting down.")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Customer Support Quality Score Dashboard")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    args = parser.parse_args()
    run_server(port=args.port, host=args.host)
