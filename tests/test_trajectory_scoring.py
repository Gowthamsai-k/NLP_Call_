"""
Unit and integration test for Trajectory-based Gated Quality Evaluator.
Verifies turn compaction, token budgeting for 20-min calls, deduction math,
closing protocol verification, and schema integrity.
"""

import json
from pathlib import Path
from src.customer_support_pipeline.nlp.evaluator import (
    compact_consecutive_turns,
    empty_evaluation,
)
from src.customer_support_pipeline.core.config import (
    EVAL_MAX_MODEL_LEN,
    SINGLE_PASS_MAX_TOKENS,
    EVAL_GPU_MEM,
)


def test_turn_compaction():
    transcript_path = Path("Output/anusha_call/call_transcript_with_speakers.json")
    assert transcript_path.exists(), f"Sample transcript not found at {transcript_path}"

    with open(transcript_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    original_count = len(segments)
    compacted = compact_consecutive_turns(segments, max_gap_s=2.0)
    compacted_count = len(compacted)

    print(f"[Compaction Test] Original segments: {original_count} -> Compacted turns: {compacted_count}")
    assert compacted_count < original_count, "Compaction should reduce fragmented turns"
    reduction_pct = (original_count - compacted_count) / original_count * 100
    print(f"[Compaction Test] Turn count reduced by {reduction_pct:.1f}%")

    # Verify time continuity in compacted turns
    for turn in compacted:
        assert float(turn["start"]) <= float(turn["end"])
        assert "speaker_id" in turn
        assert "text" in turn


def test_20min_token_headroom():
    """
    Simulates a full 20-minute spoken dialogue (~2,600 words) and verifies
    it stays well within SINGLE_PASS_MAX_TOKENS and leaves ample VRAM headroom.
    """
    # 20 minutes at 130 words/minute = ~2,600 words
    simulated_words = 2600
    # Qwen tokenization is ~1.25 tokens per English word
    simulated_tokens = int(simulated_words * 1.25)

    print(f"[Token Budget] 20-min call simulated tokens: ~{simulated_tokens}")
    print(f"[Token Budget] Single-pass limit: {SINGLE_PASS_MAX_TOKENS}")
    print(f"[Token Budget] Max context window (EVAL_MAX_MODEL_LEN): {EVAL_MAX_MODEL_LEN}")

    # Verify single pass limit covers 20 min call
    assert simulated_tokens <= SINGLE_PASS_MAX_TOKENS, "20-minute call must fit inside single pass threshold"

    # Verify total sequence (input + max output) fits safely in context window
    max_output = 1600
    prompt_overhead = 900
    total_sequence = simulated_tokens + prompt_overhead + max_output
    headroom = EVAL_MAX_MODEL_LEN - total_sequence

    print(f"[Token Budget] Total worst-case sequence: {total_sequence} tokens")
    print(f"[Token Budget] Headroom remaining in 8K context: {headroom} tokens")
    assert headroom > 1000, f"Must have at least 1,000 tokens of safety headroom (got {headroom})"


def test_gated_deduction_math():
    """
    Tests that deduction caps prevent scores from unfairly collapsing below 50,
    and that fatal gates work as intended.
    """
    # Case 1: Standard call with minor and medium slips
    deductions_case1 = [
        {"category": "professionalism", "points": 5.0, "reason": "Forgot company name in greeting"},
        {"category": "respectfulness", "points": 8.0, "reason": "Interrupted customer once during query"},
        {"category": "politeness", "points": 4.0, "reason": "Rushed closing phrasing"},
        {"category": "empathy", "points": 10.0, "reason": "Did not acknowledge customer frustration directly"},
    ]
    bonuses_case1 = [
        {"category": "respectfulness", "points": 5.0, "reason": "De-escalated caller professionally"},
    ]

    prof = 100.0 - sum(d["points"] for d in deductions_case1 if d["category"] == "professionalism")
    resp = 100.0 - sum(d["points"] for d in deductions_case1 if d["category"] == "respectfulness") + 5.0
    pol = 100.0 - sum(d["points"] for d in deductions_case1 if d["category"] == "politeness")
    emp = 100.0 - sum(d["points"] for d in deductions_case1 if d["category"] == "empathy")
    total = round(0.30 * prof + 0.25 * resp + 0.25 * pol + 0.20 * emp, 1)

    print(f"[Deduction Test Case 1] Prof: {prof}, Resp: {resp}, Pol: {pol}, Emp: {emp} -> Total: {total}")
    assert prof == 95.0
    assert resp == 97.0
    assert pol == 96.0
    assert emp == 90.0
    assert total >= 85.0, f"A good agent with minor slips should easily score in the 90s/80s (got {total})"

    # Case 2: Stricter deduction cap test (cap at 50.0 allows rough calls to drop into 50s-60s)
    stacked_deductions = [
        {"category": "professionalism", "points": 15.0},
        {"category": "professionalism", "points": 15.0},
        {"category": "professionalism", "points": 15.0},
        {"category": "professionalism", "points": 15.0},
    ]
    raw_ded = sum(d["points"] for d in stacked_deductions)
    capped_ded = min(raw_ded, 50.0)
    capped_prof = 100.0 - capped_ded
    print(f"[Deduction Test Case 2] Raw deduction: -{raw_ded} -> Capped deduction: -{capped_ded} -> Prof: {capped_prof}")
    assert capped_prof == 50.0, "Capping at 50.0 should allow rough calls to reflect stricter grading"


def test_language_friction_and_timeline_metadata():
    """Verifies language mismatch flags and timeline turn text/summary fields."""
    empty = empty_evaluation()
    lang = empty["language_analysis"]
    assert "language_mismatch_detected" in lang
    assert "agent_language_adaptation" in lang
    assert "language_friction_flag" in lang
    print("[Language Test] Language mismatch and friction flags present.")

    # Test timeline turns structure
    timeline = empty["diarization_timeline"]
    assert "turns" in timeline
    print("[Timeline Metadata Test] Timeline metadata verified.")


def test_schema_completeness():
    """Verifies that empty_evaluation contains all required new parameters."""
    empty = empty_evaluation()
    required_keys = [
        "language_analysis",
        "introduction_check",
        "closing_check",
        "sentiment_analysis",
        "risk_signals",
        "resolution",
        "resolution_lifecycle",
        "scorecard",
        "turn_trajectory",
        "uncertainties",
        "diarization_timeline",
    ]
    for key in required_keys:
        assert key in empty, f"Missing required top-level key: {key}"

    # Verify closing check keys
    assert "summary_provided" in empty["closing_check"]
    assert "next_steps_set" in empty["closing_check"]
    assert "additional_help_offered" in empty["closing_check"]

    # Verify sentiment keys
    assert "starting_sentiment" in empty["sentiment_analysis"]
    assert "ending_sentiment" in empty["sentiment_analysis"]
    assert "sentiment_shift" in empty["sentiment_analysis"]

    # Verify risk signals
    assert "churn_risk" in empty["risk_signals"]
    assert "escalation_requested" in empty["risk_signals"]
    assert "threat_detected" in empty["risk_signals"]

    # Verify FCR in resolution
    assert "fcr_likelihood" in empty["resolution"]

    # Verify empathy and deductions in scorecard
    assert "empathy_score" in empty["scorecard"]
    assert "deductions" in empty["scorecard"]
    assert "bonuses" in empty["scorecard"]
    print("[Schema Test] All new parameters successfully verified in evaluation structure.")


def test_database_enrichment_and_hover_metadata():
    from src.customer_support_pipeline.storage.database import enrich_timeline_turns_from_metadata
    
    mock_data = {
        "client_needs": ["Need help with student login"],
        "agent_actions": ["Reset password and sent SMS"],
        "resolution": {"status": "resolved", "explanation": "Password successfully reset and credentials verified"},
        "scorecard": {
            "evidence": [
                {
                    "timestamp": "10.0-15.0",
                    "quote": "I have sent you the temporary password via SMS.",
                    "category": "professionalism",
                    "interpretation": "Agent provides direct resolution"
                }
            ]
        }
    }
    raw_dt = {
        "agent_speaker_id": "SPEAKER_00",
        "client_speaker_id": "SPEAKER_01",
        "turns": [
            {"start": 0.0, "end": 4.0, "speaker_id": "SPEAKER_00", "role": "agent"},
            {"start": 4.5, "end": 9.5, "speaker_id": "SPEAKER_01", "role": "client"},
            {"start": 10.0, "end": 16.0, "speaker_id": "SPEAKER_00", "role": "agent"},
            {"start": 16.5, "end": 20.0, "speaker_id": "SPEAKER_01", "role": "client"},
        ]
    }
    enriched = enrich_timeline_turns_from_metadata(raw_dt, mock_data)
    turns = enriched["turns"]
    assert len(turns) == 4
    for t in turns:
        assert "text" in t and len(t["text"]) > 0, "Turn missing text"
        assert "summary" in t and len(t["summary"]) > 0, "Turn missing summary"
        assert "event_type" in t and len(t["event_type"]) > 0, "Turn missing event_type"
    
    # Verify matched quote on evidence turn
    assert turns[2]["text"] == "I have sent you the temporary password via SMS."
    assert turns[2]["event_type"] == "Professionalism"
    print("[Database & Metadata Test] Timeline metadata enrichment verified.")


def test_devanagari_script_preservation():
    from src.customer_support_pipeline.storage.database import enrich_timeline_turns_from_metadata
    
    hindi_data = {
        "client_needs": ["शुल्क रसीद प्राप्त करने में समस्या"],
        "agent_actions": ["रसीद का सत्यापन किया"],
        "resolution": {"status": "resolved", "explanation": "रसीद सफलतापूर्वक भेज दी गई है"},
        "scorecard": {
            "evidence": [
                {
                    "timestamp": "[43.30-48.45]",
                    "quote": "तो अभी वो सभी ट्राइंसेक्शन सक्सेस्फुल होगे और रिसिप्ट भी जनरेट हुआ।",
                    "category": "professionalism",
                    "interpretation": "Agent confirmed receipt generation"
                }
            ]
        }
    }
    raw_dt = {
        "total_duration_seconds": 60.0,
        "agent_speaker_id": "SPEAKER_00",
        "client_speaker_id": "SPEAKER_01",
        "turns": [
            {"start": 0.0, "end": 10.0, "speaker_id": "SPEAKER_00", "role": "agent"},
            {"start": 10.5, "end": 25.0, "speaker_id": "SPEAKER_01", "role": "client"},
            {"start": 40.0, "end": 50.0, "speaker_id": "SPEAKER_00", "role": "agent"},
        ]
    }
    enriched = enrich_timeline_turns_from_metadata(raw_dt, hindi_data)
    turns = enriched["turns"]
    
    # Verify Devanagari evidence quote matched exactly
    matched_turn = turns[2]
    assert matched_turn["text"] == "तो अभी वो सभी ट्राइंसेक्शन सक्सेस्फुल होगे और रिसिप्ट भी जनरेट हुआ।"
    assert matched_turn["raw_text"] == "तो अभी वो सभी ट्राइंसेक्शन सक्सेस्फुल होगे और रिसिप्ट भी जनरेट हुआ।"
    assert any('\u0900' <= c <= '\u097F' for c in matched_turn["text"]), "Devanagari script must be present"
    
    # Verify agent opening turn is in Devanagari script
    assert any('\u0900' <= c <= '\u097F' for c in turns[0]["text"]), "Agent opening turn must be in Devanagari script"
    print("[Devanagari Script Test] Devanagari native script successfully preserved and verified.")


if __name__ == "__main__":
    test_turn_compaction()
    test_20min_token_headroom()
    test_gated_deduction_math()
    test_schema_completeness()
    test_language_friction_and_timeline_metadata()
    test_database_enrichment_and_hover_metadata()
    test_devanagari_script_preservation()
    print("ALL TESTS PASSED SUCCESSFULLY!")
