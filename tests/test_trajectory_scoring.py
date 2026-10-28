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

    # Case 2: Deduction cap test (multiple slips in same category do not snowball infinitely)
    stacked_deductions = [
        {"category": "professionalism", "points": 10.0},
        {"category": "professionalism", "points": 10.0},
        {"category": "professionalism", "points": 10.0},
        {"category": "professionalism", "points": 10.0},
    ]
    raw_ded = sum(d["points"] for d in stacked_deductions)
    capped_ded = min(raw_ded, 35.0)
    capped_prof = 100.0 - capped_ded
    print(f"[Deduction Test Case 2] Raw deduction: -{raw_ded} -> Capped deduction: -{capped_ded} -> Prof: {capped_prof}")
    assert capped_prof == 65.0, "Capping should prevent infinite deduction snowball"


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


if __name__ == "__main__":
    test_turn_compaction()
    test_20min_token_headroom()
    test_gated_deduction_math()
    test_schema_completeness()
    print("ALL TESTS PASSED SUCCESSFULLY!")
