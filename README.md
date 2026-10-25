# Customer Support Call Intelligence Pipeline

A modular, production-ready Python project converted directly from the single-model VRAM notebook. It processes customer support audio calls end-to-end: diarization, speech-to-text, punctuation restoration, speaker alignment, intent extraction, and call quality evaluation.

---

## Key Features

1. **Strict Code Logic Preservation**:
   - Preserves all prompt templates, system instructions, Indic language handling, token constraints, and validation schemas verbatim.
   - Retains the exact 7-task intent extraction rules and 8-task call evaluation criteria.
   - Preserves the deterministic fallback merge for call evaluation.

2. **GPU Memory Budgeting and Model Release**:
   - Sequential model execution: only one model resides in GPU VRAM at a time.
   - Explicit teardown of vLLM engines and PyTorch pipelines (`release_memory()`).
   - Runs `gc.collect()`, `torch.cuda.empty_cache()`, `torch.cuda.ipc_collect()`, and stream synchronization between stages.
   - Qwen uses a 4-bit checkpoint; vLLM allocations scale to free device VRAM, reserve 0.5 GiB, and are limited to 80% of total VRAM by default.
   - Set `GPU_VRAM_BUDGET_GB` to a positive value to add a fixed absolute ceiling; leave it unset for dynamic allocation.
   - Model inference requires a CUDA GPU. Audio conversion, file handling, and JSON processing remain CPU work.

3. **Stage-Level & Generation Retries**:
   - Each pipeline stage is wrapped in exponential backoff retries (`retry_with_memory_cleanup`).
   - If an out-of-memory or transient CUDA/API error occurs, GPU cache is immediately purged before retrying.
   - LLM JSON generation includes automatic formatting retry prompts if an invalid JSON response is returned.

4. **Folder Processing & Archive Uploading**:
   - Supports single audio files, directories of audio files, or `.zip`/`.tar` archives.
   - In folder mode, each call is isolated in its own subfolder.
   - Failure resilience: if an individual audio file fails, the error is recorded in `batch_processing_summary.json` and the pipeline continues to subsequent files without interruption.

5. **Resume & Checkpointing**:
   - Automatically detects previously completed deliverables (`call_transcript_with_speakers.json`, `intent_extraction.json`, `call_evaluation.json`).
   - Resumes interrupted jobs without re-running expensive model inference stages.

---

## Pipeline Stages

```
Raw Audio (.mp3, .wav, .m4a, etc.)
  │
  ▼
[Stage 1: FFmpeg Conversion] ──────> 16 kHz Mono WAV
  │
  ▼
[Stage 2: Pyannote Diarization] ───> call_metadata.json (Speaker turns)
  │ (Release Diarization VRAM)
  ▼
[Stage 3: Whisper Large v3 ASR] ───> whisper_transcript.json (Turn-by-turn text)
  │ (Release Whisper VRAM)
  ▼
[Stage 4: Horizon Punctuation] ────> horizon_punctuated_transcript.json
  │ (Release Punctuation VRAM)
  ▼
[Stage 5: Speaker Alignment] ──────> call_transcript_with_speakers.json (Deliverable 1)
  │
  ▼
[Stage 6: Qwen3 Intent Extraction] > intent_extraction.json (Deliverable 2)
  │
  ▼
[Stage 7: Qwen3 Call Evaluation] ──> call_evaluation.json (Deliverable 3)
  │ (Release Qwen VRAM)
  ▼
[Stage 8: Cleanup] ────────────────> Removes temporary WAV and intermediate files
```

---

## Project Structure

```
customer_support_pipeline/
├── __init__.py
├── config.py              # Constants, model IDs, prompts, schemas, limits
├── memory.py              # VRAM purging, garbage collection, retry wrappers
├── audio_utils.py         # FFmpeg conversion, audio loading, resampling, turn slicing
├── diarization.py         # Pyannote pipeline with TorchAudio CUDA bypass
├── transcription.py       # Whisper large-v3 quantized vLLM inference
├── punctuation.py         # Horizon Labs punctuation restoration
├── alignment.py           # Temporal overlap speaker assignment
├── intent_extraction.py   # Batched intent extraction & hierarchical consolidation
├── evaluation.py          # 2-chunk evaluation, LLM merge & deterministic fallback
├── pipeline.py            # Single-call coordinator with retries & stage checkpoints
├── batch_processor.py     # Folder scanning, archive extraction, batch summary
├── main.py                # Command-line interface
├── requirements.txt       # Pinned dependencies
└── README.md
```

---

## Installation

### Ubuntu setup

Run these steps from the Ubuntu terminal, in the root directory of this
project. A CUDA-capable NVIDIA driver is required for GPU inference; verify
that Ubuntu can see the GPU with `nvidia-smi` before installing the pipeline.

1. Install FFmpeg, the GCC build tools required by Triton, and `uv`:

```bash
sudo apt update
sudo apt install -y ffmpeg curl build-essential
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

2. Accept access to
[`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1)
on Hugging Face and create a token that can access it. Set the token in this
terminal:

```bash
export HF_TOKEN="your_huggingface_token"
```

3. Install Python dependencies and download all four model repositories:

```bash
bash scripts/setup_runtime.sh
```

The setup script uses `uv sync --python 3.12`, then downloads the diarization,
Whisper, punctuation, and Qwen model repositories into Hugging Face's standard
cache. Runtime loaders use this same cache. Re-running setup checks the cache
and retrieves missing model files. Model downloads require internet access and
several gigabytes of disk space.

4. Start the dashboard:

```bash
uv run --no-sync python dashboard.py
```

Open `http://127.0.0.1:8765` in a browser. Keep `HF_TOKEN` set in the terminal
while running the dashboard or CLI.

---

## Usage

### 1. Process a Single Audio File
```bash
uv run --no-sync python main.py \
    --input "/path/to/customer_call.mp3" \
    --output "./output" \
    --retries 3
```
The results are written under `./output/customer_call/`.

### 2. Process a Folder of Audio Files (Folder Uploading)
```bash
uv run --no-sync python main.py \
    --input "/path/to/calls_folder" \
    --output "./output/batch_results" \
    --retries 3
```

### 3. Process an Uploaded ZIP Archive
```bash
uv run --no-sync python main.py \
    --input "/path/to/recorded_calls.zip" \
    --output "./output/unpacked_results"
```

### 4. Run with an 8 GB GPU

The pipeline uses FP8 Whisper and a 4-bit Qwen checkpoint, runs model stages sequentially, and dynamically limits each vLLM stage to currently free VRAM while reserving 0.5 GiB and capping use at 80% of device VRAM. The default Qwen fraction is 0.65, so an otherwise idle 8 GiB GPU gives that stage a vLLM budget of about 5.2 GiB. Larger GPUs get a larger stage budget. Set `GPU_VRAM_BUDGET_GB` to a positive value if you want a fixed ceiling. A CUDA GPU is required for model inference. The default Whisper FP8 checkpoint avoids the W8A8/INT8 scaled-matrix kernel that is unsupported on Blackwell SM120 GPUs. Override the checkpoint with `WHISPER_MODEL_ID` only when selecting a compatible alternative.

### 5. Use the local dashboard

Start the dashboard from the project directory:

```bash
uv run --no-sync python dashboard.py
```

Open `http://127.0.0.1:8765`. Choose one audio file, a folder of recordings, or one ZIP archive. For folder uploads, use the Folder tab and select the folder. The dashboard processes calls sequentially, then shows the transcript, intent extraction, and evaluation when you click a recording name. By default, uploaded inputs and results are stored under `./output/dashboard_runs/`; set `CSP_DASHBOARD_OUTPUT` or pass `--output` to choose another location. The server binds to localhost by default.

Set `HF_TOKEN` in the environment before starting the dashboard for Pyannote access. The dashboard uses the same GPU and CUDA requirements as the command-line pipeline.

---

## Python API Usage

You can also import and run the pipeline directly within Python scripts or notebooks:

```python
from pathlib import Path
from customer_support_pipeline import process_single_call, process_audio_folder

# Process single call
call_path = Path("call.mp3")
result = process_single_call(
    input_audio_path=str(call_path),
    output_dir=str(Path("./output") / call_path.stem),
    hf_token="your_hf_token",
    resume=True,
    clean_intermediate=True,
)

# Process entire directory
summary = process_audio_folder(
    input_folder="./calls/",
    output_base_dir="./batch_output/",
    hf_token="your_hf_token",
    max_retries=3,
    resume=True,
)
```

---

## Deliverables Generated

By default, each call's output directory contains only these final deliverables:
1. `call_transcript_with_speakers.json`: Full timestamped transcript with assigned speaker IDs and match confidences.
2. `intent_extraction.json`: Validated speaker roles, primary customer intent, secondary intents, explicit client needs, and key events.
3. `call_evaluation.json`: Complete scorecard (professionalism, respectfulness, politeness, total score), categorized evidence quotes, client needs, agent actions, and resolution status.
Use `--keep-intermediates` to retain intermediate files, model checkpoints, and `pipeline_status.json` for debugging.
