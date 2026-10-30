# System Architecture Specification
# Zoho Cliq Status Reporting & Dashboard Frontend State Persistence

Date: October 2026
Branch: fix-runtime-errors
Target: Customer Support Call Evaluation Pipeline

================================================================================
SECTION 1: OVERVIEW & GOALS
================================================================================
This document specifies the technical design, payload schemas, and integration points for two key operational enhancements:

1. Zoho Cliq Webhook Integration:
   Dispatches clean, executive-level batch status cards into a Zoho Cliq channel whenever a batch processing job finishes (success, partial success, or failure). The card provides high-impact metrics (total calls, pass/fail counts, average quality score, dominant language, and resolution rate) without flooding the channel with huge data tables.

2. Frontend State Persistence Across Page Reloads:
   Eliminates UI state loss when users refresh or navigate away from the dashboard during long batch processing runs. Implements a two-tier synchronization mechanism (Browser localStorage + Backend Active Job Discovery endpoint `/api/job/latest`) ensuring progress bars, file cards, and stage indicators re-attach automatically.

Zero External Dependencies Requirement:
All network dispatching and JSON serialization utilize Python standard libraries (`urllib.request`, `json`, `time`, `os`). No new external packages are added to requirements.txt, preserving Nuitka standalone binary compatibility.


================================================================================
SECTION 2: ZOHO CLIQ BRIEF STATUS REPORTING
================================================================================

2.1 Webhook Connectivity & Architecture
Zoho Cliq allows external services to post messages via Incoming Webhooks.
The pipeline connects over HTTPS POST using `urllib.request` with a 10-second timeout.
If network connectivity fails or an invalid webhook is provided, the error is logged as a warning; the core audio processing pipeline never crashes.

Configuration Precedence:
1. Web UI upload field (if supplied by the user per job)
2. Command line flag `--cliq-webhook <URL>`
3. Environment variable `ZOHO_CLIQ_WEBHOOK_URL`

2.2 Payload Schema: Batch Completion (Success / Partial Success)
When a batch completes, a brief executive summary card is constructed.

Payload JSON Structure:
{
  "text": "Status: SUCCESS\nTotal Calls: 25 | Passed: 25 | Failed: 0\nAverage Score: 87.2 / 100\nDominant Language: Hindi\nResolution Rate: 92.0%\nExecution Time: 3m 45s\nDashboard: http://localhost:8000",
  "card": {
    "title": "Call Evaluation Batch Summary",
    "theme": "modern-inline"
  }
}

If partial failures occur:
- "theme" switches to warning amber.
- A concise "Failed Calls" section is appended listing up to 5 failed file names and brief error reasons.

2.3 Payload Schema: Batch Failure (Fatal Crash)
If an unhandled crash interrupts the batch:

Payload JSON Structure:
{
  "text": "Status: CRITICAL FAILED\nJob ID: b8e1f0\nTotal Files Affected: 12\nFailure Stage: Stage 3 (ASR Transcription)\nReason: Out of Memory on CUDA device\nTimestamp: 2026-10-30 14:00:00",
  "card": {
    "title": "Call Evaluation Batch HALTED",
    "theme": "prompt-danger"
  }
}

2.4 Implementation Details (src/customer_support_pipeline/reporting/cliq_notifier.py)
- `send_cliq_notification(webhook_url, payload, timeout=10.0) -> bool`
- `build_brief_completed_card(batch_summary, dashboard_url=None) -> dict`
- `build_brief_failed_card(job_id, error_message, stage_name, total_files) -> dict`
- `send_cliq_batch_completed(webhook_url, batch_summary, dashboard_url=None) -> bool`
- `send_cliq_batch_failed(webhook_url, job_id, error_message, stage_name, total_files) -> bool`


================================================================================
SECTION 3: FRONTEND STATE PERSISTENCE ACROSS PAGE RELOADS
================================================================================

3.1 The Problem
Currently, when a user uploads audio or an archive:
1. `dashboard_server.py` creates a background thread and assigns a unique `job_id`.
2. The browser JavaScript stores `job_id` inside an in-memory variable (`currentJobId`).
3. If the user hits Refresh (F5), closes the tab, or navigates away, `currentJobId` is lost.
4. The background thread keeps processing, but the UI reverts to the blank upload form.

3.2 Two-Tier Persistence Architecture

Tier 1: Client-Side localStorage Binding
1. When `/api/upload` returns `{ status: "started", job_id: "xyz" }`:
   - `localStorage.setItem('active_csp_job_id', data.job_id);`
   - Store optional webhook URL in `localStorage.setItem('saved_cliq_webhook', webhookUrl);`
2. On page initialization (`window.addEventListener('load', ...)`):
   - Check `localStorage.getItem('active_csp_job_id')`.
   - If present, query `GET /api/job/<job_id>`.
   - If the job status is `running` or `queued`:
     * Unhide progress container (`progressBox`).
     * Disable the Start Process button (`processBtn`).
     * Re-attach live polling interval (`startJobPolling(jobId)`).
     * Re-populate file progress cards and stage indicators.
   - If job is `completed` or `failed`:
     * Merge deliverables into the UI.
     * Clean up `localStorage.removeItem('active_csp_job_id');`.

Tier 2: Server-Side Active Job Discovery (`/api/job/latest`)
Even if localStorage is empty (e.g., opened in an incognito window, another browser, or a different computer on the local network):
1. On initial page load, if no `active_csp_job_id` is found in localStorage:
   - Dashboard issues `GET /api/job/latest`.
2. Server checks `JOBS` dictionary for any job with status `running` or `queued`, or the most recent job executed within the last 10 minutes.
3. If an active job is found:
   - Server returns the full job metadata.
   - Client stores `active_csp_job_id` in localStorage and attaches the live tracker.
4. If no active job exists, server returns `{ active: false }`.

3.3 Backend API Endpoint Updates (dashboard_server.py)
Add route in `DashboardHandler.do_GET`:
- Endpoint: `/api/job/latest`
- Logic:
  Finds any job where `status in ("running", "queued")`. If none running, finds the job with the highest `created_at` timestamp.
  Returns HTTP 200 with job payload or `{ "active": false }`.


================================================================================
SECTION 4: INTEGRATION CHECKLIST & FILE TOUCHPOINTS
================================================================================
1. `src/customer_support_pipeline/core/config.py`:
   - Add `ZOHO_CLIQ_WEBHOOK_URL = os.environ.get("ZOHO_CLIQ_WEBHOOK_URL", "")`
2. `src/customer_support_pipeline/reporting/cliq_notifier.py`:
   - Create new module for webhook dispatching and card construction.
3. `src/customer_support_pipeline/orchestration/batch_runner.py`:
   - Add `cliq_webhook` parameter to `process_batch_by_stages`.
   - Invoke `send_cliq_batch_completed` at conclusion of batch run.
4. `dashboard_server.py`:
   - Add `/api/job/latest` endpoint.
   - Pass `cliq_webhook` from upload request body to `run_pipeline_worker`.
   - Trigger `send_cliq_batch_completed` or `send_cliq_batch_failed`.
5. `dashboard/index.html`:
   - Add Zoho Cliq Webhook input field in upload settings.
   - Add `localStorage` persistence and `/api/job/latest` auto-reconnect on page load.
6. `src/customer_support_pipeline/cli/main.py`:
   - Add `--cliq-webhook` argument to the CLI parser.
