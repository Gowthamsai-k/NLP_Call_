"""
Punctuation restoration using Horizon-Labs/punctuation-restoration-small.
"""

import os
import json
import logging
import importlib.util
import sys
from pathlib import Path
from typing import List, Dict, Any, Union, Optional

from huggingface_hub import hf_hub_download

from ..core.config import PUNCTUATION_MODEL_ID
from ..core.memory import release_memory, log_vram

logger = logging.getLogger(__name__)


def load_punctuation_model() -> Any:
    """Downloads helper and loads Horizon-Labs punctuation model once."""
    try:
        log_vram("Before Punctuation Load")
        helper_path = hf_hub_download(
            repo_id=PUNCTUATION_MODEL_ID,
            filename="punctuate.py",
        )

        spec = importlib.util.spec_from_file_location("horizon_punctuate", helper_path)
        punctuate_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = punctuate_module
        spec.loader.exec_module(punctuate_module)

        punctuator = punctuate_module.Punctuator(PUNCTUATION_MODEL_ID)
        logger.info("Horizon punctuation model loaded successfully.")
        return punctuator
    except Exception as exc:
        logger.warning(
            f"Could not load Horizon punctuation model ({type(exc).__name__}: {exc}). "
            "Proceeding with native ASR punctuation..."
        )
        return None


def run_punctuation_restoration(
    transcript_or_path: Union[str, Path, List[Dict[str, Any]]],
    output_path: str,
    existing_punctuator: Optional[Any] = None,
    release_after: bool = True,
) -> List[Dict[str, Any]]:
    """Restores punctuation for transcript segments with graceful fallback to native ASR punctuation."""
    if isinstance(transcript_or_path, (str, Path)):
        with open(transcript_or_path, "r", encoding="utf-8") as f:
            transcript_segments = json.load(f)
    else:
        transcript_segments = transcript_or_path

    if not transcript_segments:
        logger.warning("No transcript segments provided for punctuation restoration.")
        return []

    punctuator = existing_punctuator

    try:
        if punctuator is None and existing_punctuator is None:
            punctuator = load_punctuation_model()

        for i, segment in enumerate(transcript_segments):
            raw_text = segment.get("raw_text") or segment.get("text", "")
            segment["raw_text"] = raw_text

            if not raw_text.strip():
                segment["text"] = raw_text
                continue

            if punctuator is not None:
                try:
                    punctuated_text = punctuator(raw_text)
                    segment["text"] = (
                        punctuated_text.strip()
                        if isinstance(punctuated_text, str)
                        else raw_text
                    )
                except Exception as e:
                    logger.warning(f"Segment {i} punctuation failed: {e}")
                    segment["text"] = raw_text
            else:
                segment["text"] = raw_text

        output_p = Path(output_path).resolve()
        output_p.parent.mkdir(parents=True, exist_ok=True)
        with output_p.open("w", encoding="utf-8") as f:
            json.dump(transcript_segments, f, ensure_ascii=False, indent=2)

        logger.info(f"Punctuation restored for {len(transcript_segments)} segments -> {output_p.name}")
        return transcript_segments

    finally:
        if release_after:
            release_memory(punctuator)
            log_vram("After Punctuation Cleanup")
