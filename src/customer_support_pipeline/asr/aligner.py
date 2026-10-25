"""
Speaker and transcript alignment.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Union

logger = logging.getLogger(__name__)


def overlap_duration(start1: float, end1: float, start2: float, end2: float) -> float:
    """Calculates temporal overlap duration between two time intervals."""
    return max(0.0, min(end1, end2) - max(start1, start2))


def align_speakers_with_transcript(
    metadata_or_path: Union[str, Path, Dict[str, Any]],
    asr_or_path: Union[str, Path, List[Dict[str, Any]]],
    output_path: str,
) -> List[Dict[str, Any]]:
    """Aligns speaker turns with punctuated ASR segments by maximum temporal overlap."""
    if isinstance(metadata_or_path, (str, Path)):
        with open(metadata_or_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = metadata_or_path

    if isinstance(asr_or_path, (str, Path)):
        with open(asr_or_path, "r", encoding="utf-8") as f:
            transcript_segments = json.load(f)
    else:
        transcript_segments = asr_or_path

    combined = []
    diarization_turns = metadata.get("segments", [])

    for seg in transcript_segments:
        overlaps = {}
        for turn in diarization_turns:
            duration = overlap_duration(
                seg["start"], seg["end"],
                turn["start"], turn["end"]
            )
            if duration > 0:
                speaker = turn["speaker_id"]
                overlaps[speaker] = overlaps.get(speaker, 0.0) + duration

        total_overlap = sum(overlaps.values())
        if overlaps and total_overlap > 0:
            speaker_id = max(overlaps, key=overlaps.get)
            confidence = overlaps[speaker_id] / total_overlap
        else:
            speaker_id = None
            confidence = 0.0

        combined.append({
            "start": seg["start"],
            "end": seg["end"],
            "speaker_id": speaker_id,
            "speaker_match_confidence": round(confidence, 3),
            "text": seg["text"],
            "raw_text": seg.get("raw_text", seg["text"]),
            "words": seg.get("words", [])
        })

    output_p = Path(output_path).resolve()
    output_p.parent.mkdir(parents=True, exist_ok=True)
    with output_p.open("w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    logger.info(
        f"Alignment complete: {len(combined)} speaker-tagged segments -> {output_p.name}"
    )
    return combined
