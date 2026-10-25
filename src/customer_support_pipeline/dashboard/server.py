"""Small standard-library HTTP server for the static pipeline dashboard."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import threading
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import unquote, urlsplit

from ..core.config import (
    FINAL_EVALUATION_FILENAME,
    FINAL_INTENT_FILENAME,
    FINAL_TRANSCRIPT_FILENAME,
    SUPPORTED_AUDIO_EXTENSIONS,
)

STATIC_DIR = Path(__file__).with_name("static")
MAX_FILES_PER_JOB = 500
MAX_FILE_SIZE_BYTES = 4 * 1024**3
MAX_JOB_SIZE_BYTES = 20 * 1024**3
UPLOAD_CHUNK_SIZE = 1024 * 1024


def safe_relative_name(raw_name: str) -> str:
    """Normalize a browser-supplied filename and reject path traversal."""
    normalized = str(raw_name).replace("\\", "/")
    if not normalized or normalized.startswith("/") or PureWindowsPath(normalized).is_absolute():
        raise ValueError("File names must be relative paths.")

    path = PurePosixPath(normalized)
    parts = [part for part in path.parts if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts) or ":" in parts[0]:
        raise ValueError("Invalid file path in upload.")
    return str(PurePosixPath(*parts))


def extract_zip_safely(archive_path: Path, destination: Path) -> None:
    """Extract regular ZIP entries while rejecting traversal and symlinks."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        extracted_size = sum(entry.file_size for entry in entries if not entry.is_dir())
        if extracted_size > MAX_JOB_SIZE_BYTES:
            raise ValueError("The expanded ZIP contents exceed the 20 GiB limit.")
        for entry in entries:
            relative = safe_relative_name(entry.filename)
            target = (destination / Path(*PurePosixPath(relative).parts)).resolve()
            if root not in target.parents and target != root:
                raise ValueError("Archive contains a path outside its extraction folder.")

            # ZIP stores Unix mode bits in the high 16 bits of external_attr.
            mode = entry.external_attr >> 16
            if mode & 0o170000 == 0o120000:
                raise ValueError("Symbolic links are not allowed in uploaded archives.")
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=UPLOAD_CHUNK_SIZE)


class DashboardState:
    def __init__(self, output_root: Path):
        self.output_root = output_root.resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.jobs: dict[str, dict[str, Any]] = {}
        self.active_job_id: str | None = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pipeline-job")

    def create_job(self, manifest: list[dict[str, Any]]) -> dict[str, Any]:
        if not isinstance(manifest, list) or not manifest:
            raise ValueError("Choose at least one audio file or one ZIP archive.")
        if len(manifest) > MAX_FILES_PER_JOB:
            raise ValueError(f"A folder upload can contain at most {MAX_FILES_PER_JOB} files.")

        safe_files = []
        seen_names: set[str] = set()
        total_bytes = 0
        for item in manifest:
            if not isinstance(item, dict):
                raise ValueError("Invalid upload manifest.")
            name = safe_relative_name(item.get("name", ""))
            size = item.get("size")
            if not isinstance(size, int) or size <= 0 or size > MAX_FILE_SIZE_BYTES:
                raise ValueError(f"Invalid or oversized upload: {name}")
            if name.casefold() in seen_names:
                raise ValueError(f"Duplicate file path in upload: {name}")
            seen_names.add(name.casefold())
            suffix = Path(name).suffix.lower()
            if suffix != ".zip" and suffix not in SUPPORTED_AUDIO_EXTENSIONS:
                raise ValueError(f"Unsupported file type: {name}")
            safe_files.append({"name": name, "size": size, "uploaded": False})
            total_bytes += size

        if total_bytes > MAX_JOB_SIZE_BYTES:
            raise ValueError("The total upload is too large (maximum 20 GiB).")

        zip_files = [item for item in safe_files if Path(item["name"]).suffix.lower() == ".zip"]
        if zip_files and (len(zip_files) != 1 or len(safe_files) != 1):
            raise ValueError("Upload one ZIP by itself, or choose audio files/folder contents.")

        job_id = uuid.uuid4().hex[:12]
        job_root = self.output_root / job_id
        upload_root = job_root / "uploads"
        upload_root.mkdir(parents=True, exist_ok=True)
        with self.lock:
            self.jobs[job_id] = {
                "id": job_id,
                "status": "uploading",
                "files": safe_files,
                "total_upload_bytes": total_bytes,
                "uploaded_bytes": 0,
                "current_file": None,
                "current_stage": "Waiting for files",
                "current_index": 0,
                "total_calls": 0,
                "progress": 0,
                "error": None,
                "summary": None,
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "root": job_root,
                "upload_root": upload_root,
                "results_root": job_root / "results",
            }
        return self.public_job(job_id)

    def accept_file(self, job_id: str, index: int, content_length: int, stream) -> None:
        job = self._get(job_id)
        with self.lock:
            if job["status"] != "uploading":
                raise ValueError("This upload is no longer accepting files.")
            if index < 0 or index >= len(job["files"]):
                raise ValueError("Invalid file number.")
            file_info = job["files"][index]
            if file_info["uploaded"]:
                raise ValueError("This file was already uploaded.")
            if content_length != file_info["size"]:
                raise ValueError("The uploaded file size does not match the manifest.")

        destination = job["upload_root"] / Path(*PurePosixPath(file_info["name"]).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        try:
            with destination.open("wb") as output:
                while written < content_length:
                    chunk = stream.read(min(UPLOAD_CHUNK_SIZE, content_length - written))
                    if not chunk:
                        raise ValueError("Upload ended before the full file was received.")
                    output.write(chunk)
                    written += len(chunk)
        except Exception:
            destination.unlink(missing_ok=True)
            raise

        with self.lock:
            file_info["uploaded"] = True
            job["uploaded_bytes"] += written
            job["current_stage"] = "Files uploaded"

    def start_job(self, job_id: str) -> None:
        with self.lock:
            job = self._get(job_id)
            if job["status"] != "uploading":
                raise ValueError("This job has already started.")
            if not all(item["uploaded"] for item in job["files"]):
                raise ValueError("Upload all selected files before starting processing.")
            if self.active_job_id is not None:
                active = self.jobs.get(self.active_job_id)
                if active and active["status"] in {"queued", "processing"}:
                    raise RuntimeError("Another pipeline job is already using the GPU.")
            job["status"] = "queued"
            job["current_stage"] = "Queued for GPU processing"
            self.active_job_id = job_id
            self.executor.submit(self._run_job, job_id)

    def _run_job(self, job_id: str) -> None:
        job = self._get(job_id)
        self._update(job_id, status="processing", current_stage="Preparing input")
        progress_handler = _ProgressLogHandler(self, job_id)
        package_logger = logging.getLogger("customer_support_pipeline")
        previous_level = package_logger.level
        package_logger.setLevel(min(previous_level or logging.INFO, logging.INFO))
        package_logger.addHandler(progress_handler)

        try:
            from ..orchestration.batch_runner import process_audio_folder
            from ..orchestration.pipeline import process_single_call

            files = job["files"]
            uploaded_path = job["upload_root"] / Path(*PurePosixPath(files[0]["name"]).parts)
            results_root: Path = job["results_root"]
            results_root.mkdir(parents=True, exist_ok=True)
            token = os.environ.get("HF_TOKEN")

            if Path(files[0]["name"]).suffix.lower() == ".zip":
                extracted = job["root"] / "extracted"
                self._update(job_id, current_stage="Extracting ZIP archive")
                extract_zip_safely(uploaded_path, extracted)
                summary = process_audio_folder(
                    input_folder=str(extracted),
                    output_base_dir=str(results_root),
                    hf_token=token,
                    resume=True,
                    clean_intermediate=True,
                )
                self._update(job_id, summary=summary)
            elif len(files) == 1:
                self._update(
                    job_id,
                    total_calls=1,
                    current_index=1,
                    current_file=Path(files[0]["name"]).name,
                )
                process_single_call(
                    input_audio_path=str(uploaded_path),
                    output_dir=str(results_root / uploaded_path.stem),
                    hf_token=token,
                    resume=True,
                    clean_intermediate=True,
                )
                self._update(job_id, summary={"total_files": 1, "successful": 1, "failed_count": 0})
            else:
                summary = process_audio_folder(
                    input_folder=str(job["upload_root"]),
                    output_base_dir=str(results_root),
                    hf_token=token,
                    resume=True,
                    clean_intermediate=True,
                )
                self._update(job_id, summary=summary)

            calls = self._collect_calls(job)
            summary = self._get(job_id).get("summary") or {}
            failed_count = int(summary.get("failed_count", 0))
            no_audio_found = int(summary.get("total_files", 1)) == 0
            if no_audio_found:
                failed_count = 1
                self._update(
                    job_id,
                    error="No supported audio recordings were found in the uploaded input.",
                )
            elif not failed_count:
                # The inputs are temporary working data. A successful dashboard
                # job keeps only its results and batch summary on disk.
                shutil.rmtree(job["upload_root"], ignore_errors=True)
                extracted = job["root"] / "extracted"
                if extracted.exists():
                    shutil.rmtree(extracted, ignore_errors=True)
            self._update(
                job_id,
                status="completed_with_errors" if failed_count else "completed",
                calls=calls,
                progress=100,
                current_stage="Finished",
                current_file=None,
            )
        except Exception as exc:
            logging.getLogger(__name__).exception("Dashboard pipeline job %s failed", job_id)
            self._update(
                job_id,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                current_stage="Stopped",
                calls=self._collect_calls(job),
            )
        finally:
            package_logger.removeHandler(progress_handler)
            package_logger.setLevel(previous_level)
            with self.lock:
                if self.active_job_id == job_id:
                    self.active_job_id = None

    def _collect_calls(self, job: dict[str, Any]) -> list[dict[str, Any]]:
        results_root: Path = job["results_root"]
        if not results_root.exists():
            return []
        calls = []
        failed_names = {
            item.get("audio_file"): item.get("error", "Processing failed")
            for item in (job.get("summary") or {}).get("failed_files", [])
        }
        for folder in sorted(results_root.iterdir(), key=lambda item: item.name.lower()):
            if not folder.is_dir():
                continue
            has_results = all(
                (folder / name).is_file()
                for name in (
                    FINAL_TRANSCRIPT_FILENAME,
                    FINAL_INTENT_FILENAME,
                    FINAL_EVALUATION_FILENAME,
                )
            )
            if has_results:
                calls.append({"id": folder.name, "name": folder.name, "status": "completed"})
        for audio_name, error in failed_names.items():
            if not any(call["name"] == Path(audio_name).stem for call in calls):
                calls.append({"id": Path(audio_name).stem, "name": Path(audio_name).stem, "status": "failed", "error": error})
        return calls

    def public_job(self, job_id: str) -> dict[str, Any]:
        job = self._get(job_id)
        public_keys = (
            "id", "status", "uploaded_bytes", "total_upload_bytes", "current_file",
            "current_stage", "current_index", "total_calls", "progress", "error",
            "summary", "calls", "created_at",
        )
        with self.lock:
            result = {key: job.get(key) for key in public_keys}
            if job["status"] == "uploading" and job["total_upload_bytes"]:
                result["upload_progress"] = round(100 * job["uploaded_bytes"] / job["total_upload_bytes"])
            else:
                result["upload_progress"] = 100 if job["status"] != "uploading" else 0
            if job["status"] == "processing" and job.get("total_calls"):
                fraction = max(0, job.get("current_index", 1) - 1) / job["total_calls"]
                result["progress"] = max(result.get("progress") or 0, round(100 * fraction))
            return result

    def call_details(self, job_id: str, call_id: str) -> dict[str, Any]:
        job = self._get(job_id)
        safe_id = safe_relative_name(call_id)
        if "/" in safe_id:
            raise ValueError("Invalid call name.")
        call_dir = job["results_root"] / safe_id
        allowed = {
            FINAL_TRANSCRIPT_FILENAME: "transcript",
            FINAL_INTENT_FILENAME: "intent",
            FINAL_EVALUATION_FILENAME: "evaluation",
        }
        result: dict[str, Any] = {"id": safe_id, "name": safe_id}
        for filename, key in allowed.items():
            path = call_dir / filename
            if path.is_file():
                with path.open("r", encoding="utf-8") as source:
                    result[key] = json.load(source)
        return result

    def _get(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError("Unknown dashboard job.")
            return job

    def _update(self, job_id: str, **values: Any) -> None:
        with self.lock:
            self._get(job_id).update(values)


class _ProgressLogHandler(logging.Handler):
    def __init__(self, state: DashboardState, job_id: str):
        super().__init__(logging.INFO)
        self.state = state
        self.job_id = job_id

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        found = re.search(r"\[(\d+)/(\d+)\]\s+Processing:\s+(.+)", message)
        if found:
            index, total, filename = found.groups()
            self.state._update(
                self.job_id,
                current_index=int(index),
                total_calls=int(total),
                current_file=filename.strip(),
                current_stage="Starting call",
            )
            return
        stage = re.search(r">>> Stage \d+:\s*(.+)", message)
        if stage:
            self.state._update(self.job_id, current_stage=stage.group(1).strip())
            return
        found = re.search(r"Found (\d+) audio file\(s\)", message)
        if found:
            self.state._update(self.job_id, total_calls=int(found.group(1)))


def create_handler(state: DashboardState):
    class DashboardHandler(BaseHTTPRequestHandler):
        server_version = "CallIntelligenceDashboard/1.0"

        def log_message(self, fmt: str, *args) -> None:
            logging.getLogger(__name__).info("%s - %s", self.address_string(), fmt % args)

        def do_GET(self) -> None:
            parsed = urlsplit(self.path)
            parts = [unquote(part) for part in parsed.path.split("/") if part]
            if not parts or parts == ["index.html"]:
                return self._static("index.html")
            if parts in (["styles.css"], ["app.js"]):
                return self._static(parts[0])
            if len(parts) == 3 and parts[:2] == ["api", "jobs"]:
                try:
                    self._json(200, state.public_job(parts[2]))
                except KeyError as exc:
                    self._json(404, {"error": str(exc)})
                return
            if len(parts) == 5 and parts[:2] == ["api", "jobs"] and parts[3] == "calls":
                try:
                    self._json(200, state.call_details(parts[2], parts[4]))
                except KeyError as exc:
                    self._json(404, {"error": str(exc)})
                except (ValueError, OSError, json.JSONDecodeError) as exc:
                    self._json(400, {"error": str(exc)})
                return
            self._json(404, {"error": "Not found"})

        def do_POST(self) -> None:
            parts = [unquote(part) for part in urlsplit(self.path).path.split("/") if part]
            if parts == ["api", "jobs"]:
                try:
                    body = self._read_json()
                    result = state.create_job(body.get("files"))
                    self._json(201, result)
                except (ValueError, json.JSONDecodeError) as exc:
                    self._json(400, {"error": str(exc)})
                return
            if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "start":
                try:
                    state.start_job(parts[2])
                    self._json(202, state.public_job(parts[2]))
                except KeyError as exc:
                    self._json(404, {"error": str(exc)})
                except RuntimeError as exc:
                    self._json(409, {"error": str(exc)})
                except ValueError as exc:
                    self._json(400, {"error": str(exc)})
                return
            self._json(404, {"error": "Not found"})

        def do_PUT(self) -> None:
            parts = [unquote(part) for part in urlsplit(self.path).path.split("/") if part]
            if len(parts) == 5 and parts[:2] == ["api", "jobs"] and parts[3] == "files":
                try:
                    length = int(self.headers.get("Content-Length", "-1"))
                    if length < 0:
                        raise ValueError("Content-Length is required.")
                    state.accept_file(parts[2], int(parts[4]), length, self.rfile)
                    self._json(201, {"uploaded": True})
                except KeyError as exc:
                    self._json(404, {"error": str(exc)})
                except (ValueError, OSError) as exc:
                    self._json(400, {"error": str(exc)})
                return
            self._json(404, {"error": "Not found"})

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 1024 * 1024:
                raise ValueError("Request is too large.")
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(encoded)

        def _static(self, filename: str) -> None:
            path = STATIC_DIR / filename
            if not path.is_file():
                return self._json(404, {"error": "Not found"})
            content_type = {
                ".html": "text/html; charset=utf-8",
                ".css": "text/css; charset=utf-8",
                ".js": "text/javascript; charset=utf-8",
            }.get(path.suffix, "application/octet-stream")
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return DashboardHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local call intelligence dashboard.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: localhost only).")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port (default: 8765).")
    parser.add_argument(
        "--output",
        default=os.environ.get("CSP_DASHBOARD_OUTPUT", "./output/dashboard_runs"),
        help="Directory for uploaded files and processed results.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    state = DashboardState(Path(args.output))
    server = ThreadingHTTPServer((args.host, args.port), create_handler(state))
    print(f"Dashboard ready at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard...")
    finally:
        server.shutdown()
        server.server_close()
        state.executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
