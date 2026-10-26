#!/usr/bin/env python3
"""
Dashboard server for Customer Support Quality Pipeline.
Zero external web dependencies (uses Python standard library http.server).
Serves the static dashboard and handles audio/folder/zip uploads with asynchronous background processing.
"""

import os
import sys
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
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dashboard_server")

# In-memory Job Management (Thread-safe)
JOBS = {}
JOBS_LOCK = threading.Lock()


# Discover existing evaluations
def find_existing_evaluations(output_dir: Path):
    evaluations = []
    if not output_dir.exists():
        return evaluations

    # Look for call_evaluation.json files recursively
    for eval_path in output_dir.rglob("call_evaluation.json"):
        try:
            with eval_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            call_name = data.get("audio_file") or eval_path.parent.name
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
                    JOBS[job_id]["errors"].append(f_item)
                    for item in JOBS[job_id]["files"]:
                        if item["filename"] == f_name:
                            item["status"] = "failed"
                            item["stage_name"] = f"Failed: {str(f_err)[:45]}"
                            item["error"] = str(f_err)
                            break
                    if not JOBS[job_id]["crash_log"]:
                        JOBS[job_id]["crash_log"] = f"Error in {f_name}: {f_err}"
                    else:
                        JOBS[job_id]["crash_log"] += f"\n\nError in {f_name}: {f_err}"

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

        elif path == "/api/results":
            output_dir = PROJECT_ROOT / "output"
            evals = find_existing_evaluations(output_dir)
            self.send_json_response(evals)
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
            asr_lang = os.environ.get("ASR_LANG", "en")
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
                    asr_lang = file_data.decode("utf-8", errors="ignore").strip() or "en"
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
    server_address = (host, port)
    httpd = HTTPServer(server_address, DashboardHandler)
    logger.info("=" * 60)
    logger.info(f"🚀 Customer Support Scorecard Dashboard running at:")
    logger.info(f"   👉 http://localhost:{port}")
    if host == "0.0.0.0":
        logger.info(f"   👉 http://127.0.0.1:{port}")
    logger.info(f"Static dashboard directory: {PROJECT_ROOT / 'dashboard'}")
    logger.info(f"Evaluation deliverables:    {PROJECT_ROOT / 'output'}")
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
