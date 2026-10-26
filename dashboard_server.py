#!/usr/bin/env python3
"""
Dashboard server for Customer Support Quality Pipeline.
Zero external web dependencies (uses Python standard library http.server).
Serves the static dashboard and handles audio/folder/zip uploads.
"""

import os
import sys
import json
import shutil
import logging
import argparse
import tempfile
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


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.dashboard_dir = PROJECT_ROOT / "dashboard"
        super().__init__(*args, directory=str(self.dashboard_dir), **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            index_path = self.dashboard_dir / "index.html"
            if index_path.exists():
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(index_path.read_bytes())
                return

        elif path == "/api/ping":
            self.send_json_response({"status": "ok", "message": "Dashboard server running"})
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

        # Simple multipart parser for standard library
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            boundary = content_type.split("boundary=")[-1].strip()
            if boundary.startswith('"') and boundary.endswith('"'):
                boundary = boundary[1:-1]
            boundary_bytes = ("--" + boundary).encode()

            parts = body.split(boundary_bytes)
            uploaded_files = []
            mode = "single"
            asr_lang = os.environ.get("ASR_LANG", "en")
            asr_decoder = os.environ.get("ASR_DECODER", "ctc")

            temp_upload_dir = Path(tempfile.mkdtemp(prefix="call_upload_"))

            for part in parts:
                if not part or part in (b"--\r\n", b"--", b"\r\n"):
                    continue
                header_end = part.find(b"\r\n\r\n")
                if header_end < 0:
                    continue

                headers_raw = part[:header_end].decode("latin-1", errors="ignore")
                file_data = part[header_end + 4 :].rstrip(b"\r\n")

                if 'name="mode"' in headers_raw:
                    mode = file_data.decode("utf-8", errors="ignore").strip()
                elif 'name="asr_lang"' in headers_raw:
                    asr_lang = file_data.decode("utf-8", errors="ignore").strip() or "en"
                elif 'name="asr_decoder"' in headers_raw:
                    asr_decoder = file_data.decode("utf-8", errors="ignore").strip() or "ctc"
                elif 'name="files"' in headers_raw or 'filename="' in headers_raw:
                    # Extract filename
                    import re
                    match = re.search(r'filename="([^"]+)"', headers_raw)
                    if match:
                        filename = Path(match.group(1)).name
                        dest_file = temp_upload_dir / filename
                        dest_file.write_bytes(file_data)
                        uploaded_files.append(dest_file)

            if not uploaded_files:
                self.send_json_response({"error": "No files uploaded"}, status=400)
                shutil.rmtree(temp_upload_dir, ignore_errors=True)
                return

            logger.info(f"Received {len(uploaded_files)} file(s), mode: {mode}")

            # Import pipeline functions
            from src.customer_support_pipeline.orchestration.pipeline import process_single_call
            from src.customer_support_pipeline.orchestration.batch_runner import (
                process_audio_folder,
                unpack_archive,
            )

            output_base = PROJECT_ROOT / "output"
            output_base.mkdir(parents=True, exist_ok=True)
            results = []

            for up_file in uploaded_files:
                file_lower = up_file.name.lower()
                if file_lower.endswith((".zip", ".tar", ".tar.gz", ".tgz")):
                    logger.info(f"Unpacking archive {up_file.name}...")
                    unpacked_dir = unpack_archive(str(up_file), destination_dir=str(temp_upload_dir / "unpacked"))
                    batch_res = process_audio_folder(
                        input_folder=str(unpacked_dir),
                        output_base_dir=str(output_base),
                        clean_intermediate=True,
                        asr_lang=asr_lang,
                        asr_decoder=asr_decoder,
                    )
                    # Gather resulting evaluations
                    evals = find_existing_evaluations(output_base)
                    results.extend(evals)

                elif up_file.suffix.lower() in (".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"):
                    call_stem = up_file.stem
                    call_out = output_base / call_stem
                    logger.info(f"Processing audio {up_file.name} -> {call_out} (ASR Lang: {asr_lang})")

                    call_res = process_single_call(
                        input_audio_path=str(up_file),
                        output_dir=str(call_out),
                        clean_intermediate=True,
                        asr_lang=asr_lang,
                        asr_decoder=asr_decoder,
                    )

                    eval_path = call_out / "call_evaluation.json"
                    if eval_path.exists():
                        with eval_path.open("r", encoding="utf-8") as f:
                            data = json.load(f)
                        results.append({
                            "id": call_stem,
                            "filename": up_file.name,
                            "data": data,
                            "path": str(eval_path),
                        })

            # Cleanup uploaded temp files
            shutil.rmtree(temp_upload_dir, ignore_errors=True)

            self.send_json_response({
                "status": "success",
                "total_processed": len(results),
                "evaluations": results,
            })

        except Exception as e:
            logger.error(f"Upload processing failed: {e}", exc_info=True)
            self.send_json_response({"error": str(e)}, status=500)

    def send_json_response(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
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
