"""
Unit tests for deterministic agent-performance metrics.
Uses synthetic timelines and synthetic tones only; no models, GPU, or real call audio.
"""

import numpy as np
import soundfile as sf

from src.customer_support_pipeline.nlp.agent_metrics import (
    attach_agent_performance,
    compute_agent_performance,
)

AGENT = "SPEAKER_00"
CLIENT = "SPEAKER_01"


def _timeline(turns, total=None):
    if total is not None:
        tot = total
    else:
        ends = [t.get("end") for t in turns if isinstance(t, dict) and isinstance(t.get("end"), (int, float))]
        tot = max(ends) if ends else 0.0
    return {
        "total_duration_seconds": tot,
        "agent_speaker_id": AGENT,
        "client_speaker_id": CLIENT,
        "agent_talk_ratio_pct": 50.0,
        "turns": turns,
    }


def _turn(speaker, start, end, text=""):
    return {"speaker_id": speaker, "start": start, "end": end, "text": text}


def test_agent_interrupting_client_is_counted():
    tl = _timeline([
        _turn(CLIENT, 0.0, 3.0, "I want to cancel my course"),
        _turn(AGENT, 2.0, 4.0, "Let me check that for you"),
    ])
    perf = compute_agent_performance(tl, wav_path=None)
    assert perf["conversation"]["interruptions"]["agent_interrupted_client"] == 1
    assert perf["conversation"]["interruptions"]["client_interrupted_agent"] == 0


def test_short_backchannel_is_not_an_interruption():
    tl = _timeline([
        _turn(CLIENT, 0.0, 3.0, "so the issue is with the portal login"),
        _turn(AGENT, 1.0, 1.5, "yes"),
    ])
    perf = compute_agent_performance(tl, wav_path=None)
    assert perf["conversation"]["interruptions"]["agent_interrupted_client"] == 0


def test_response_latency_from_client_end_to_agent_start():
    tl = _timeline([
        _turn(CLIENT, 0.0, 5.0, "hello I have a problem"),
        _turn(AGENT, 6.5, 9.0, "sure, I can help"),
    ])
    latency = compute_agent_performance(tl, wav_path=None)["conversation"]["response_latency"]
    assert latency["count"] == 1
    assert latency["median_s"] == 1.5


def test_dead_air_and_hold_are_separated_by_threshold():
    tl = _timeline([
        _turn(CLIENT, 0.0, 2.0, "hello"),
        _turn(AGENT, 7.0, 9.0, "one moment"),
        _turn(AGENT, 34.0, 36.0, "thanks for waiting"),
    ])
    conv = compute_agent_performance(tl, wav_path=None)["conversation"]
    assert conv["dead_air"]["count"] == 1
    assert conv["hold_candidates"]["count"] == 1
    assert conv["hold_candidates"]["total_s"] == 25.0


def test_rushed_agent_turn_is_flagged_against_own_baseline():
    slow = "please wait while I look into this account for you today"
    fast = " ".join(["word"] * 20)
    tl = _timeline([
        _turn(AGENT, 0.0, 6.0, slow),
        _turn(CLIENT, 6.0, 7.0, "ok"),
        _turn(AGENT, 7.0, 13.0, slow),
        _turn(CLIENT, 13.0, 14.0, "ok"),
        _turn(AGENT, 14.0, 16.0, fast),
    ])
    rate = compute_agent_performance(tl, wav_path=None)["speaking_rate"]
    assert rate["agent_rushed_turns"] == 1


def test_missing_audio_marks_acoustics_unavailable_without_failing():
    tl = _timeline([_turn(AGENT, 0.0, 3.0, "hello thanks for calling"), _turn(CLIENT, 3.0, 4.0, "hi")])
    perf = compute_agent_performance(tl, wav_path="/nonexistent/call_clean.wav")
    assert perf["acoustics"]["available"] is False
    assert perf["flags"]["agent_monotone"] is None


def test_monotone_agent_voice_is_flagged(tmp_path):
    try:
        import librosa
    except ImportError:
        return
    sr = 16000
    turns, samples, t_cursor = [], [], 0.0
    for i in range(4):
        dur = 2.5
        t = np.arange(int(dur * sr)) / sr
        samples.append(0.3 * np.sin(2 * np.pi * 180.0 * t))
        turns.append(_turn(AGENT, t_cursor, t_cursor + dur, "we will process your request now"))
        t_cursor += dur
        samples.append(np.zeros(int(0.5 * sr)))
        t_cursor += 0.5
    wav = tmp_path / "call_clean.wav"
    sf.write(str(wav), np.concatenate(samples).astype(np.float32), sr)

    perf = compute_agent_performance(_timeline(turns), wav_path=str(wav))
    assert perf["acoustics"]["available"] is True
    assert perf["acoustics"]["agent"]["analysed_turns"] >= 3
    assert perf["flags"]["agent_monotone"] is True


def test_malformed_turns_are_skipped_not_raised():
    tl = _timeline([
        {"speaker_id": AGENT, "start": "bad", "end": 2.0, "text": "x"},
        {"speaker_id": AGENT, "start": 3.0, "end": 2.0, "text": "reversed"},
        "not-a-dict",
        _turn(CLIENT, 4.0, 6.0, "valid turn"),
    ])
    perf = compute_agent_performance(tl, wav_path=None)
    assert perf["conversation"]["total_duration_seconds"] is not None


def test_attach_is_noop_without_timeline():
    eval_data = {"scorecard": {}}
    attach_agent_performance(eval_data, None)
    assert "agent_performance" not in eval_data


def test_attach_adds_block_for_valid_timeline():
    eval_data = {"diarization_timeline": _timeline([_turn(AGENT, 0.0, 2.0, "hello"), _turn(CLIENT, 2.5, 4.0, "hi")])}
    attach_agent_performance(eval_data, None)
    assert eval_data["agent_performance"]["version"] == 1
    assert "conversation" in eval_data["agent_performance"]
