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

# Indic-Conformer / Whisper ASR Configuration (IN-22 Languages)
ASR_LANG = os.environ.get("ASR_LANG", "auto")  # 'auto' enables dynamic multi-lingual detection per turn/batch; supports 'hi', 'te', 'ta', 'en', etc.
ASR_DECODER = os.environ.get("ASR_DECODER", "ctc")  # 'ctc' or 'rnnt' 

WHISPER_MAX_MODEL_LEN = 448
WHISPER_MAX_BATCHED_TOKENS = 2048
WHISPER_MAX_TOKENS = 448
# Optimized default for 8GB VRAM (e.g., RTX 5050 / RTX 2000 Blackwell)
WHISPER_GPU_MEM = float(os.environ.get("WHISPER_GPU_MEM", "0.30"))

# ==============================================================================
# INTENT EXTRACTION CONFIGURATION
# ==============================================================================
INTENT_MAX_MODEL_LEN = int(os.environ.get("INTENT_MAX_MODEL_LEN", "8192"))
MAX_TRANSCRIPT_TOKENS = int(os.environ.get("MAX_TRANSCRIPT_TOKENS", "2000"))
MAX_BATCH_OUTPUT_TOKENS = 1000
MAX_MERGE_OUTPUT_TOKENS = 1200
OVERLAP_SEGMENTS = 2
# Optimized default for 8GB VRAM (allows ~6GB Qwen 2.5 3B fp16 weights + KV cache)
INTENT_GPU_MEM = float(os.environ.get("INTENT_GPU_MEM", "0.85"))

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
EVAL_MAX_MODEL_LEN = int(os.environ.get("EVAL_MAX_MODEL_LEN", "8192"))
SINGLE_PASS_MAX_TOKENS = int(os.environ.get("SINGLE_PASS_MAX_TOKENS", "4500"))
CHUNK_TEXT_TOKENS = int(os.environ.get("CHUNK_TEXT_TOKENS", "1800"))
CHUNK_OUTPUT_TOKENS = int(os.environ.get("CHUNK_OUTPUT_TOKENS", "1600"))
MERGE_OUTPUT_TOKENS = int(os.environ.get("MERGE_OUTPUT_TOKENS", "1800"))
MERGE_GROUP_SIZE = 2
EVAL_MAX_RETRIES = 2
# Optimized default for 8GB VRAM (allows ~6GB Qwen 2.5 3B fp16 weights + KV cache)
EVAL_GPU_MEM = float(os.environ.get("EVAL_GPU_MEM", "0.85"))

EVAL_SYSTEM_PROMPT = """
You are an expert, independent customer-support quality evaluator.
Evaluate only the supplied transcript and conversation context.
Every call is unique. You must evaluate the agent critically based on the specific transcript and assign varied, dynamic scores that reflect real differences in agent performance. Do NOT output default static numbers like 85 or 90 for every call.
The call may contain Indian English, Indic languages, and code-switching.
Account for honorifics, tone, customer responsiveness, emotional trajectory, and issue resolution.
Return exactly one valid JSON object without Markdown fences.
""".strip()

EVAL_SCHEMA = """
Return one JSON object with exactly this structure:
{
  "language_analysis": {
    "agent_language": "string (e.g. English, Hindi, Hinglish, Tamil, Telugu)",
    "client_language": "string (e.g. English, Hindi, Hinglish, Tamil, Telugu)",
    "dominant_language": "string (e.g. English, Hindi, Tamil, Telugu)",
    "explanation": "short string under 15 words"
  },
  "introduction_check": {
    "status": "compliant | partially_compliant | non_compliant",
    "compliant": true,
    "agent_name_stated": "string or null",
    "company_named": true,
    "greeting_quote": "short string under 20 words or null",
    "notes": "short string under 15 words"
  },
  "closing_check": {
    "status": "compliant | partially_compliant | non_compliant",
    "summary_provided": true,
    "next_steps_set": true,
    "additional_help_offered": true,
    "closing_quote": "short string under 25 words or null",
    "notes": "short string under 20 words"
  },
  "sentiment_analysis": {
    "starting_sentiment": "positive | neutral | frustrated | angry",
    "ending_sentiment": "positive | neutral | frustrated | angry",
    "sentiment_shift": "improved | stable | worsened",
    "explanation": "short explanation under 25 words"
  },
  "risk_signals": {
    "churn_risk": false,
    "escalation_requested": false,
    "threat_detected": false,
    "notes": "string"
  },
  "intent_review": "supported | partially_supported | unclear",
  "client_needs": ["string"],
  "agent_actions": ["string"],
  "resolution": {
    "status": "resolved | partially_resolved | unresolved | pending_follow_up | unclear",
    "fcr_likelihood": "high | medium | low",
    "fcr_explanation": "string under 25 words",
    "explanation": "detailed narrative covering problem, actions, verification, and closure"
  },
  "resolution_lifecycle": {
    "problem_definition": "string",
    "diagnostic_action": "string",
    "verification_outcome": "string",
    "closure_state": "string"
  },
  "scorecard": {
    "professionalism_score": 0.0,
    "respectfulness_score": 0.0,
    "politeness_score": 0.0,
    "empathy_score": 0.0,
    "total_score": 0.0,
    "deductions": [
      {
        "category": "professionalism | respectfulness | politeness | empathy",
        "points": 5.0,
        "reason": "string",
        "timestamp": "string",
        "quote": "string"
      }
    ],
    "bonuses": [
      {
        "category": "professionalism | respectfulness | politeness | empathy",
        "points": 5.0,
        "reason": "string",
        "timestamp": "string",
        "quote": "string"
      }
    ],
    "professionalism_rationale": "string",
    "respectfulness_rationale": "string",
    "politeness_rationale": "string",
    "empathy_rationale": "string",
    "evidence": [
      {
        "timestamp": "string",
        "speaker_id": "string",
        "quote": "string",
        "category": "professionalism | respectfulness | politeness | empathy",
        "interpretation": "string"
      }
    ]
  },
  "turn_trajectory": [
    {
      "turn": 1,
      "timestamp": "string",
      "speaker": "string",
      "event_type": "greeting | problem_stated | action_taken | customer_verification | closing | escalation | filler",
      "summary": "string under 15 words"
    }
  ],
  "uncertainties": ["string"]
}

GUIDELINES:
1. GATED DEDUCTION SCORING:
- Each category starts at 100.0 (Professionalism: 100.0, Respectfulness: 100.0, Politeness: 100.0, Empathy: 100.0).
- Penalties are capped per category so realistic slips do not collapse scores:
  * Minor Protocol Slip (e.g. missing company name in greeting, rushed closing): deduct 3 to 5 points.
  * Medium Lapse (e.g. slight interruption, delayed response, lack of active empathy): deduct 5 to 10 points. Maximum total deduction for medium lapses is 15 points per category.
  * Serious Violation (e.g. arguing with customer, dismissive refusal): deduct 15 to 25 points.
  * Fatal Gate (e.g. abusive language, shouting, deliberate disconnect): Respectfulness is set to 0.0.
  * De-escalation Bonus: Calmly handling an angry caller awards +5 bonus points in Empathy or Respectfulness.
- Each subscore is: 100.0 - (sum of deductions) + (sum of bonuses), clamped between 0.0 and 100.0.
- Total score is the weighted average: (0.30 * Professionalism + 0.25 * Respectfulness + 0.25 * Politeness + 0.20 * Empathy) rounded to 1 decimal place.

2. RESOLUTION LIFECYCLE & FIRST CONTACT RESOLUTION (FCR):
- Four phases:
  * Problem Definition: The exact core issue reported by the customer.
  * Diagnostic Action: Concrete steps taken by the agent to solve or troubleshoot the issue.
  * Verification Outcome: Whether the customer tested the solution during the call. If the customer ended politely (e.g. 'okay thanks bye') without testing the fix or without confirmation, status is 'unresolved' or 'pending_follow_up', NOT 'resolved'.
  * Closure State: Final agreement reached on the call.
- FCR Likelihood: 'high' only if fully resolved and verified on call; 'medium' if pending backend follow-up with clear ticket timeline; 'low' if unresolved or customer left dissatisfied.

3. CLOSING PROTOCOL:
- Status is 'compliant' if the agent summarized actions, stated next steps or turnaround time, and asked 'Is there anything else I can help you with?'.
- 'partially_compliant' if one or two were completed; 'non_compliant' if all were omitted.

4. SENTIMENT TRAJECTORY & RISK SIGNALS:
- Compare opening customer sentiment to closing customer sentiment ('improved', 'stable', 'worsened').
- Set churn_risk: true if customer expressed desire to cancel or switch services.
- Set escalation_requested: true if customer demanded to speak to a supervisor or manager.
- Set threat_detected: true if customer mentioned legal action, consumer court, or social media complaints.
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
