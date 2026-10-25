# Customer Support Call Intelligence Pipeline

A modular, production-ready speech intelligence pipeline that processes customer support audio calls end-to-end: speaker diarization, speech-to-text (ASR), punctuation restoration, speaker-turn alignment, intent extraction, and call quality evaluation.

---

## 🌅 Quickstart: Running Tomorrow Morning in WSL

When you start your laptop, simply open **PowerShell** or **Terminal** and run these 3 commands:

### 1. Open WSL
```bash
wsl
```

### 2. Navigate to project & activate virtual environment
```bash
cd /mnt/c/Users/Harshith/NLP_Call_
source .venv/bin/activate
```

### 3. Run the pipeline on any audio call
```bash
python main.py \
    --input "/mnt/c/Users/Harshith/Downloads/anusha_v_nexteducation_in__Support__112__-1__7042757684__2026-10-22_13-24-52.mp3" \
    --output "./output/anusha_call_2"
```

> [!NOTE]
> The pipeline automatically **checkpoints and resumes**: completed stages are skipped in seconds.

---

## Key Features

1. **Strict Code Logic Preservation**:
   - Preserves all prompt templates, system instructions, Indic language handling, token constraints, and validation schemas verbatim.
   - Retains the exact 7-task intent extraction rules and 8-task call evaluation criteria.
   - Preserves the deterministic fallback merge for call evaluation.

2. **Aggressive Unused Memory Release**:
   - Sequential model execution: only one model resides in GPU VRAM at a time.
   - Explicit teardown of model pipelines (`release_memory()`).
   - Runs `gc.collect()`, `torch.cuda.empty_cache()`, `torch.cuda.ipc_collect()`, and stream synchronization between stages to eliminate runtime OOM crashes.

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
[Stage 6: Qwen Intent Extraction] ─> intent_extraction.json (Deliverable 2)
  │
  ▼
[Stage 7: Qwen Call Evaluation] ───> call_evaluation.json (Deliverable 3)
  │ (Release Qwen VRAM)
  ▼
[Stage 8: Cleanup] ────────────────> Removes temporary WAV and intermediate files
```

---

## Project Structure

```
NLP_Call_/
├── src/
│   └── customer_support_pipeline/
│       ├── __init__.py
│       ├── asr/
│       │   ├── aligner.py
│       │   ├── punctuator.py
│       │   └── whisper_engine.py
│       ├── audio/
│       │   ├── diarizer.py
│       │   └── preprocessor.py
│       ├── cli/
│       │   └── main.py
│       ├── core/
│       │   ├── config.py
│       │   ├── memory.py
│       │   └── shims.py
│       ├── nlp/
│       │   ├── evaluator.py
│       │   └── intent_extractor.py
│       └── orchestration/
│           ├── batch_runner.py
│           └── pipeline.py
├── main.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Deliverables Generated

For each call, the final deliverables preserved in the output directory are:
1. `call_transcript_with_speakers.json`: Full timestamped transcript with assigned speaker IDs and match confidences.
2. `intent_extraction.json`: Validated speaker roles, primary customer intent, secondary intents, explicit client needs, and key events.
3. `call_evaluation.json`: Complete scorecard (professionalism, respectfulness, politeness, total score), categorized evidence quotes, client needs, agent actions, and resolution status.
4. `pipeline_status.json`: Execution metadata, stage runtimes, and status.
