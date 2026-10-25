"""
Central configuration definitions for customer_support_pipeline.
"""

import os
from pathlib import Path

# ==============================================================================
# MODEL IDENTIFIERS
# ==============================================================================
DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-community-1"
WHISPER_MODEL_ID = "RedHatAI/whisper-large-v3-quantized.w8a8"
PUNCTUATION_MODEL_ID = "Horizon-Labs/punctuation-restoration-small"
LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")

# ==============================================================================
# AUDIO & ASR CONFIGURATION
# ==============================================================================
TARGET_SR = 16000
MIN_TURN_DURATION_S = 0.1
SILENCE_THRESHOLD = 1e-5

WHISPER_MAX_MODEL_LEN = 448
WHISPER_MAX_BATCHED_TOKENS = 2048
WHISPER_MAX_TOKENS = 448
# Safer defaults for T4 GPU (15GB VRAM)
WHISPER_GPU_MEM = float(os.environ.get("WHISPER_GPU_MEM", "0.35"))

# ==============================================================================
# INTENT EXTRACTION CONFIGURATION
# ==============================================================================
INTENT_MAX_MODEL_LEN = 8192
MAX_TRANSCRIPT_TOKENS = 2200
MAX_BATCH_OUTPUT_TOKENS = 700
MAX_MERGE_OUTPUT_TOKENS = 1200
OVERLAP_SEGMENTS = 2
# Safer default for Colab T4 GPU
INTENT_GPU_MEM = float(os.environ.get("INTENT_GPU_MEM", "0.60"))

INTENT_SYSTEM_PROMPT = """
You are an expert customer-support conversation analyst.

Your task is to extract customer intent from a timestamped conversation.

The conversation may contain English, Indian English, Indic languages,
and code-switching between languages.

Reason from the transcript only. Never invent customer details,
actions, promises, events, or outcomes.

Return valid JSON only. Do not return Markdown or explanations outside JSON.
""".strip()

INTENT_MERGE_SYSTEM_PROMPT = """
You consolidate partial analyses of a single customer-support call.

Merge duplicate findings, including duplicates from overlapping batches.
Preserve distinct customer goals and important events.

Infer the overall primary customer intent from the available evidence.
Distinguish requests, actions actually performed, and final outcomes.
Resolve speaker roles using evidence across the supplied analyses.
Do not assume roles from speaker ID ordering.

Use only valid speaker IDs provided in the input.
Do not invent facts, promises, resolutions, quotations, or timestamps.
Preserve uncertainty caused by unclear transcription or code-switching.

Return one valid JSON object matching the requested schema.
Return JSON only.
""".strip()

# ==============================================================================
# CALL EVALUATION CONFIGURATION
# ==============================================================================
EVAL_MAX_MODEL_LEN = 8192
CHUNK_TEXT_TOKENS = 1000
CHUNK_OUTPUT_TOKENS = 1800
MERGE_OUTPUT_TOKENS = 2200
MERGE_GROUP_SIZE = 2
EVAL_MAX_RETRIES = 2
EVAL_GPU_MEM = float(os.environ.get("EVAL_GPU_MEM", "0.60"))

EVAL_SYSTEM_PROMPT = """
You are an independent customer-support call evaluator.
Evaluate only the supplied transcript and extracted findings.
Do not invent events, promises, quotations, or customer details.
The call may contain Indian English, Indic languages, and code-switching.
Use evidence-based reasoning and return exactly one valid JSON object.
Do not use Markdown fences or text outside JSON.
Keep explanations concise and output only relevant evidence.
""".strip()

EVAL_SCHEMA = """
Return one JSON object with exactly this structure:
{
  "intent_review": "string",
  "client_needs": ["string"],
  "agent_actions": ["string"],
  "resolution": {
    "status": "resolved | partially_resolved | unresolved | pending_follow_up | unclear",
    "explanation": "string"
  },
  "scorecard": {
    "professionalism_score": null,
    "respectfulness_score": null,
    "politeness_score": null,
    "total_score": null,
    "professionalism_rationale": "string",
    "respectfulness_rationale": "string",
    "politeness_rationale": "string",
    "evidence": [
      {
        "timestamp": "string",
        "speaker_id": "string",
        "quote": "string",
        "category": "professionalism | respectfulness | politeness",
        "interpretation": "string"
      }
    ]
  },
  "uncertainties": ["string"]
}
Use numeric scores from 0 to 100 when the agent is validated
and evidence is sufficient.
Use JSON null for unavailable scores.
Include at most 5 client needs, 6 agent actions, and 6 evidence items.
Keep each rationale and explanation to 1-2 sentences.
Use exact transcript quotes only.
Do not list unintelligible fragments or routine repeated thank-yous
as agent actions unless relevant.
A promise is not proof of completion.
""".strip()

# ==============================================================================
# PIPELINE & RETRY CONFIGURATION
# ==============================================================================
DEFAULT_STAGE_RETRIES = 3
RETRY_BACKOFF_FACTOR = 2.0
SUPPORTED_AUDIO_EXTENSIONS = (
    ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".opus", ".webm", ".wma"
)

CLEAN_WAV_FILENAME = "call_clean.wav"
METADATA_FILENAME = "call_metadata.json"
WHISPER_RAW_FILENAME = "whisper_transcript.json"
PUNCTUATED_FILENAME = "horizon_punctuated_transcript.json"
FINAL_TRANSCRIPT_FILENAME = "call_transcript_with_speakers.json"
FINAL_INTENT_FILENAME = "intent_extraction.json"
FINAL_EVALUATION_FILENAME = "call_evaluation.json"
PIPELINE_STATUS_FILENAME = "pipeline_status.json"
