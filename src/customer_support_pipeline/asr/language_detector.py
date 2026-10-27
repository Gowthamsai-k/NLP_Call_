"""
Acoustic Spoken Language Identification (LID) for Multilingual Customer Calls.
Detects spoken language directly from audio waveforms before/during ASR decoding,
enabling direct transcription into native Indian scripts (Devanagari, Telugu, Tamil, etc.).
Zero text-based LID dependencies.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ISO-639-1 / ISO-639-3 to readable English language names
LANGUAGE_NAME_MAP = {
    "en": "English",
    "eng": "English",
    "hi": "Hindi",
    "hin": "Hindi",
    "te": "Telugu",
    "tel": "Telugu",
    "ta": "Tamil",
    "tam": "Tamil",
    "mr": "Marathi",
    "mar": "Marathi",
    "kn": "Kannada",
    "kan": "Kannada",
    "bn": "Bengali",
    "ben": "Bengali",
    "gu": "Gujarati",
    "guj": "Gujarati",
    "ml": "Malayalam",
    "mal": "Malayalam",
    "pa": "Punjabi",
    "pan": "Punjabi",
    "or": "Odia",
    "ori": "Odia",
    "ur": "Urdu",
    "urd": "Urdu",
    "as": "Assamese",
    "asm": "Assamese",
}


def resolve_language_name(code_or_name: Optional[str]) -> str:
    """Normalizes language code or name to standard Titlecase name."""
    if not code_or_name:
        return "English"
    cleaned = str(code_or_name).strip().lower().replace("__label__", "").replace("<|", "").replace("|>", "")
    return LANGUAGE_NAME_MAP.get(cleaned, cleaned.capitalize())


def detect_audio_chunk_language(
    audio_slice,
    samplerate: int = 16000,
    model_bundle: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Performs acoustic Spoken Language Identification (LID) directly on audio waveform
    BEFORE ASR decoding. Uses Whisper's acoustic language detection head to identify
    the spoken language and route directly to native script transcription.
    """
    if audio_slice is None or len(audio_slice) < int(0.2 * samplerate):
        return "en"

    if model_bundle is not None:
        whisper_pipe = model_bundle.get("whisper_pipe")
        if whisper_pipe is not None:
            try:
                model = getattr(whisper_pipe, "model", None)
                tokenizer = getattr(whisper_pipe, "tokenizer", None)
                feature_extractor = getattr(whisper_pipe, "feature_extractor", None)

                if model is not None and feature_extractor is not None and tokenizer is not None:
                    import torch
                    features = feature_extractor(
                        audio_slice,
                        sampling_rate=samplerate,
                        return_tensors="pt"
                    )
                    input_features = features.input_features
                    if torch.cuda.is_available() and hasattr(model, "device"):
                        input_features = input_features.to(model.device)
                        if getattr(model, "dtype", None) == torch.float16:
                            input_features = input_features.half()

                    with torch.no_grad():
                        detected_ids = model.detect_language(input_features)
                    if isinstance(detected_ids, list) and detected_ids:
                        lang_token = tokenizer.decode(detected_ids[0])
                        cleaned = lang_token.replace("<|", "").replace("|>", "").strip().lower()
                        if cleaned:
                            return cleaned
            except Exception as e:
                logger.debug(f"Acoustic audio LID exception: {e}")

    return "en"


def aggregate_speaker_languages(
    segments: List[Dict[str, Any]],
    agent_id: Optional[str] = None,
    client_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Aggregates acoustic per-turn language detections across the call to compute:
    1. Agent dominant and secondary spoken languages by speech duration
    2. Customer dominant and secondary spoken languages by speech duration
    3. Call-wide dominant spoken language
    """
    agent_durations: Dict[str, float] = {}
    client_durations: Dict[str, float] = {}
    total_durations: Dict[str, float] = {}

    for s in segments:
        dur = max(0.1, float(s.get("end", 0.0)) - float(s.get("start", 0.0)))
        raw_lang = s.get("language") or s.get("language_code") or "en"
        lang_name = resolve_language_name(raw_lang)

        spk = s.get("speaker_id") or s.get("speaker")
        is_agent = (agent_id and spk == agent_id) or (isinstance(spk, str) and "agent" in spk.lower())

        if is_agent:
            agent_durations[lang_name] = agent_durations.get(lang_name, 0.0) + dur
        else:
            client_durations[lang_name] = client_durations.get(lang_name, 0.0) + dur

        total_durations[lang_name] = total_durations.get(lang_name, 0.0) + dur

    def get_dominant_and_ratio(dur_map: Dict[str, float]) -> Tuple[str, str]:
        if not dur_map:
            return "English", "English"
        total = sum(dur_map.values())
        sorted_langs = sorted(dur_map.items(), key=lambda x: x[1], reverse=True)
        dom = sorted_langs[0][0]
        breakdown_parts = [f"{lang} ({int(round(d / total * 100))}%)" for lang, d in sorted_langs if d / total >= 0.1]
        return dom, ", ".join(breakdown_parts) if breakdown_parts else dom

    agent_dom, agent_breakdown = get_dominant_and_ratio(agent_durations)
    client_dom, client_breakdown = get_dominant_and_ratio(client_durations)
    overall_dom, _ = get_dominant_and_ratio(total_durations)

    return {
        "dominant_language": overall_dom,
        "agent_language": agent_dom,
        "client_language": client_dom,
        "agent_breakdown": agent_breakdown,
        "client_breakdown": client_breakdown,
        "explanation": f"Agent: {agent_breakdown} | Caller: {client_breakdown}",
    }
