# Customer Support Call Intelligence Pipeline

A modular, production-ready speech intelligence pipeline that processes customer support audio calls end-to-end: speaker diarization, native Indic speech-to-text (ASR), punctuation restoration, speaker-turn alignment, intent extraction, deterministic agent performance metrics, trajectory-based call quality evaluation, and an interactive web dashboard.

---

## Key Features

1. **Native Indic Script ASR (Bodhan AI & AI4Bharat)**:
   - Default primary model: `bodhan-ai/indic-transcribe-core` (1-billion-parameter ASR optimized for true native script transcription in Hindi, Telugu, Tamil, and 24 other Indian languages).
   - Direct output in authentic native script (Devanagari for Hindi, Telugu lipi for Telugu, Tamil script for Tamil).
   - Seamless plug-and-play support for `ai4bharat/indic-conformer-600m-multilingual` and `openai/whisper-large-v3-turbo` via `ASR_MODEL_ID`.

2. **Dynamic Turn-Level Spoken Language Auto-Detection (`ASR_LANG=auto`)**:
   - Evaluates language dynamically per speaker turn rather than assuming a single static language for the entire call.
   - Handles real-world code-switching (Hinglish, Tanglish, Tenglish) and multi-lingual conversations (e.g., Telugu customer with English/Hindi agent).
   - Built-in Language Identification (LID) tags each turn with language code and name.

3. **Stage-by-Stage Batch Architecture & VRAM Efficiency**:
   - Each neural model is loaded onto the GPU **only once** for the entire batch of calls:
     - **PyAnnote Diarization** loaded once -> diarizes all files -> unloaded from VRAM.
     - **Bodhan IndicTranscribe / Conformer ASR** loaded once -> transcribes all files -> unloaded from VRAM.
     - **Horizon Punctuation** loaded once -> restores punctuation for all files -> unloaded.
     - **Qwen 2.5 3B LLM** loaded once -> unified single-pass evaluation -> unloaded.
   - Optimized out-of-the-box for **8GB VRAM** GPUs (e.g. RTX 5050, RTX 2000 Ada/Blackwell, RTX 3070/4060) using eager execution and token-bounded compaction.

4. **Trajectory-Based Scoring & Calibrated Gated Deductions**:
   - Evaluates call progression across opening protocol, exploration, problem resolution, and closing phases.
   - Bounded transcript compaction reduces 20-minute calls by over 90% in token volume (~3,200 tokens), remaining safely within the 8K context window.
   - First Call Resolution (FCR) detection, formal closing check, and language mismatch friction flags.
   - Capped deduction rubric preventing single-category score saturation.

5. **Deterministic Agent Acoustic & Behavioral Metrics**:
   - Computes objective communication metrics without LLM calls:
     - Agent interruptions (minimum overlap threshold).
     - Response latency and dead air / hold durations.
     - Speech rate and rushed turn detection.
     - Monotone pitch standard deviation, acoustic volume escalation, and agent fatigue.

6. **SQLite Storage Backend & Server-Side Pagination**:
   - Persistent SQLite database indexing call metadata, agent names, languages, scores, and resolution statuses.
   - Server-side pagination and indexed filtering capable of handling thousands of historical call evaluations smoothly.

7. **Zero-Dependency Web Dashboard**:
   - Built using Python standard library HTTP server (no Node, npm, or complex web frameworks required).
   - Timeline diarization graph with instant hover popovers displaying turn timing, role, duration, milestone, and overlap badges.
   - Interactive slide-over scorecard drawer with keyboard navigation (`ArrowRight`, `ArrowLeft`, `Escape`).

8. **Automated Standalone Production Packaging (Nuitka / Colab)**:
   - Includes `scripts/build_colab_standalone.sh` and `build_in_colab.ipynb` to compile and bundle all dependencies into a single portable `.tar.gz` archive.
   - Supports 100% air-gapped offline model bundling (`--with-models`) so target production machines require zero internet and zero host Python packages.

---

## Cloning the Repository

```bash
git clone https://github.com/Gowthamsai-k/NLP_Call_.git
cd NLP_Call_
```

---

## Prerequisites & Setup

### 1. System Dependencies (FFmpeg & Libsndfile)
FFmpeg and libsndfile are required for audio preprocessing:

```bash
# Ubuntu / Debian / WSL
sudo apt-get update && sudo apt-get install -y ffmpeg libsndfile1

# macOS (Homebrew)
brew install ffmpeg libsndfile
```

### 2. Python Environment & Dependencies
Python 3.10 to 3.13 is supported:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt sentencepiece
```

### 3. Hugging Face Authentication Token
Accessing `bodhan-ai/indic-transcribe-core` and `pyannote/speaker-diarization-community-1` requires accepting their terms on Hugging Face:
1. Accept terms on [bodhan-ai/indic-transcribe-core](https://huggingface.co/bodhan-ai/indic-transcribe-core).
2. Accept terms on [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1).
3. Set your token as an environment variable:

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

---

## How to Run

### Method 1: Web Dashboard (Interactive UI)

Start the local server:
```bash
python3 dashboard_server.py --port 8000
```

- Open your browser at: `http://localhost:8000`
- Drag and drop call audio files (`.mp3`, `.wav`, `.m4a`, etc.) or a `.zip` archive.
- Click **Run Pipeline Analysis**.
- Watch real-time stage progress across all files.
- Click any call in the table to inspect the detailed scorecard and diarization timeline.

---

### Method 2: Command Line (CLI)

#### Process a Folder of Audio Files:
```bash
python3 main.py --input ./path_to_audio_folder/ --output ./output/
```

#### Process a Compressed Archive (`.zip`, `.tar`, `.tar.gz`):
```bash
python3 main.py --input ./customer_calls.zip --output ./output/
```

#### Process a Single Audio File:
```bash
python3 main.py --input ./sample_call.mp3 --output ./output/call_1/
```

---

## Standalone Production Build (Google Colab)

To package the entire pipeline, all Python dependencies, and optional AI model weights into a single zero-dependency standalone bundle:

1. Open Google Colab and run:
```bash
!git clone https://github.com/Gowthamsai-k/NLP_Call_.git
%cd NLP_Call_
!export HF_TOKEN="your_huggingface_token_here" && bash scripts/build_colab_standalone.sh
```

2. To include all 7.4 GB Hugging Face model weights inside the bundle for complete offline deployment:
```bash
!export HF_TOKEN="your_huggingface_token_here" && bash scripts/build_colab_standalone.sh --with-models
```

3. Download the resulting `customer_support_pipeline_production_full.tar.gz` and extract it on your production machine:
```bash
tar -xzf customer_support_pipeline_production_full.tar.gz
cd dashboard_server.dist
./dashboard_server.bin --port 8000
```

---

## Configuration & Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `HF_TOKEN` | *None* | Hugging Face token for PyAnnote and Bodhan ASR. |
| `ASR_MODEL_ID` | `bodhan-ai/indic-transcribe-core` | Primary ASR model (`bodhan-ai/indic-transcribe-core`, `ai4bharat/indic-conformer-600m-multilingual`, `openai/whisper-large-v3-turbo`). |
| `ASR_LANG` | `auto` | Target spoken language (`auto` for dynamic turn detection, or `hi`, `te`, `ta`, `en`, etc.). |
| `ASR_DECODER` | `ctc` | Decoder type when using Indic-Conformer (`ctc` or `rnnt`). |
| `LLM_MODEL_ID` | `Qwen/Qwen2.5-3B-Instruct` | Large language model used for intent extraction and quality scoring. |
| `PUNCTUATION_MODEL_ID` | `Horizon-Labs/punctuation-restoration-small` | Model used for transcript punctuation restoration. |
| `WHISPER_GPU_MEM` | `0.30` | GPU VRAM fraction allocated for ASR (fits ~2.4GB on 8GB VRAM). |
| `INTENT_GPU_MEM` | `0.85` | GPU VRAM fraction for LLM evaluation (fits ~6.8GB on 8GB VRAM). |
| `EVAL_MAX_MODEL_LEN` | `8192` | Context window limit in tokens for evaluation. |
| `CHUNK_TEXT_TOKENS` | `1200` | Slices long transcripts into bounded chunks to prevent context overflow. |

---

## Output Directory Structure

```
output/
├── call_01/
│   ├── clean_16k.wav                     # Preprocessed 16kHz audio
│   ├── call_metadata.json                # Diarization timestamps & speaker segments
│   ├── whisper_transcript.json           # ASR transcript with detected languages
│   ├── horizon_punctuated_transcript.json# Punctuation-restored transcript
│   ├── final_transcript.json             # Speaker-aligned transcript
│   └── call_evaluation.json              # Final score deliverable
├── failed/                               # Auto-created if any record encounters errors
│   └── broken_call_03/
│       ├── failure_reason.json           # Diagnostic error code and traceback
│       └── error.txt                     # Human-readable failure explanation
└── calls_database.sqlite3                # Persistent SQLite database index
```

---

## Architecture Flow

```mermaid
flowchart TD
    Audio[Batch of Audio Calls] --> Stage1[Stage 1: FFmpeg 16kHz Mono PCM]
    Stage1 --> Stage2[Stage 2: PyAnnote Diarization]
    Stage2 --> Stage3[Stage 3: Bodhan Indic-Transcribe ASR (Auto LID)]
    Stage3 --> Stage4[Stage 4: Horizon Punctuation Restoration]
    Stage4 --> Stage5[Stage 5: Speaker Turn Alignment]
    Stage5 --> Stage6[Stage 6 & 7: Qwen 2.5 3B Unified Trajectory Scoring]
    Stage6 --> Stage7[Stage 8: Packaging & SQLite Database Indexing]
    Stage7 -->|Success| SuccessDir[output/call_name/call_evaluation.json]
    Stage7 -->|Failure| FailedDir[output/failed/call_name/failure_reason.json]
```

---

## License

Proprietary - Internal Enterprise Speech Analytics. Model weights are governed by their respective licenses (Indic Open Model License, MIT, Apache 2.0).
