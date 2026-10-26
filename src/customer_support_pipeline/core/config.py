"""
Central configuration definitions for customer_support_pipeline.
"""

import os
from pathlib import Path

# ==============================================================================
# MODEL IDENTIFIERS
# ==============================================================================
DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-community-1"
ASR_MODEL_ID = os.environ.get("ASR_MODEL_ID", "ai4bharat/indic-conformer-600m-multilingual")
WHISPER_MODEL_ID = ASR_MODEL_ID  # Backward-compatible alias
PUNCTUATION_MODEL_ID = "Horizon-Labs/punctuation-restoration-small"
LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")

# ==============================================================================
# AUDIO & ASR CONFIGURATION
# ==============================================================================
TARGET_SR = 16000
MIN_TURN_DURATION_S = 0.1
SILENCE_THRESHOLD = 1e-5

# Indic-Conformer ASR Configuration (IN-22 Languages)
ASR_LANG = os.environ.get("ASR_LANG", "en")  # Default to 'en' (English), supports 'ta', 'hi', 'te', 'kn', etc.
ASR_DECODER = os.environ.get("ASR_DECODER", "ctc")  # 'ctc' or 'rnnt' 

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
MAX_BATCH_OUTPUT_TOKENS = 1200
MAX_MERGE_OUTPUT_TOKENS = 1200
OVERLAP_SEGMENTS = 2
# Safer default for Colab T4 GPU
INTENT_GPU_MEM = float(os.environ.get("INTENT_GPU_MEM", "0.60"))

INTENT_SYSTEM_PROMPT = """
You are an expert customer-support conversation analyst.

Your task is to extract customer intent from a timestamped conversation.

The conversation may contain English, Indian English, Indic languages,
and code-switching between languages.

Reason from the transcript only. All output JSON fields (intents, evidence,
summaries, rationales) MUST be written in concise ENGLISH only.
When speech is in Tamil, Hindi, or any Indic language, translate or summarize
the meaning in English. Do not output raw Indic text or repeat phrases.
Never invent customer details, actions, promises, events, or outcomes.

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
You are an expert, independent customer-support quality evaluator.
Evaluate only the supplied transcript and extracted findings.
Every call is unique. You must evaluate the agent critically based on the specific transcript and assign varied, dynamic scores that reflect real differences in agent performance. Do NOT output default static numbers like 85 or 90 for every call.
The call may contain Indian English, Indic languages, and code-switching.
Account for honorifics, tone, customer responsiveness, and issue resolution.
Return exactly one valid JSON object without Markdown fences.
""".strip()

EVAL_SCHEMA = """
Return one JSON object with exactly this structure:
{
  "intent_review": "supported | partially_supported | unclear",
  "client_needs": ["string"],
  "agent_actions": ["string"],
  "resolution": {
    "status": "resolved | partially_resolved | unresolved | pending_follow_up | unclear",
    "explanation": "string"
  },
  "scorecard": {
    "professionalism_score": 0.0,
    "respectfulness_score": 0.0,
    "politeness_score": 0.0,
    "total_score": 0.0,
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

SCORING CRITERIA AND DYNAMIC SCORING RUBRIC (Each subscore must be an independent decimal float between 0.0 and 100.0 reflecting the specific recording):
- Professionalism (0.0 to 100.0):
  * 92.0 - 98.0: Exemplary problem understanding, structured guidance, proactive assistance, accurate information.
  * 82.0 - 91.0: Good competence, answers customer questions correctly with minor hesitation or standard procedures.
  * 70.0 - 81.0: Average support, took long to understand issue, repeated explanations needed, vague answers.
  * 50.0 - 69.0: Poor competence, unable to address customer request, inaccurate info or unhelpful.

- Respectfulness (0.0 to 100.0):
  * 93.0 - 99.0: Actively listened, zero interruptions, acknowledged customer feelings empathetically, appropriate cultural respect.
  * 83.0 - 92.0: Attentive and polite throughout, standard respectful customer dialogue.
  * 70.0 - 82.0: Talked over the customer or sounded impatient, abrupt transitions.
  * 50.0 - 69.0: Dismissive, argumentative, or disrespectful.

- Politeness (0.0 to 100.0):
  * 92.0 - 98.0: Warm, personalized greeting, consistent pleasantries ("thank you", "please", "glad to assist"), warm sign-off.
  * 82.0 - 91.0: Standard courteous greeting, polite tone throughout.
  * 70.0 - 81.0: Rushed, transactional tone, skipped standard pleasantries.
  * 50.0 - 69.0: Blunt, cold, or rude phrasing.

RESOLUTION ALIGNMENT:
- Fully resolved calls should generally receive overall scores in the 86.0 - 96.0 range.
- Partially resolved or pending follow-up calls should generally receive overall scores in the 74.0 - 85.0 range.
- Unresolved issues or calls with agent friction should receive overall scores in the 52.0 - 72.0 range.
Calculate total_score as the arithmetic mean of the three scores rounded to 1 decimal place.
Extract explicit agent actions and client needs from the conversation.
Include at most 5 client needs, 6 agent actions, and 6 evidence items.
Keep each rationale and explanation to 1-2 sentences.
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
