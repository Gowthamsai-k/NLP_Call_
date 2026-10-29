"""
Deterministic agent-performance metrics from diarized turns and the clean call audio.

Every section is computed independently and degrades to None with a note on failure,
so a missing dependency or a malformed turn never fails the pipeline stage.
"""

import logging
import math
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..core.config import (
    AGENT_METRICS_VERSION,
    AM_INTERRUPT_MIN_OVERLAP_S,
    AM_INTERRUPT_MIN_TURN_S,
    AM_RESPONSE_MAX_GAP_S,
    AM_DEAD_AIR_MIN_S,
    AM_HOLD_MIN_S,
    AM_MIN_RATE_TURN_S,
    AM_RUSHED_RATE_RATIO,
    AM_RUSHED_MIN_TURN_S,
    AM_MIN_ACOUSTIC_TURN_S,
    AM_MONOTONE_PITCH_STD_ST,
    AM_ESCALATION_DB_PER_MIN,
    AM_FATIGUE_PITCH_DROP_ST,
    AM_FATIGUE_ENERGY_DROP_DB,
    AM_FATIGUE_RATE_DROP_PCT,
    AM_PITCH_MIN_HZ,
    AM_PITCH_MAX_HZ,
)

logger = logging.getLogger(__name__)

_FRAME_LENGTH = 1024
_HOP_LENGTH = 256
_MIN_VOICED_FRAMES = 10
_MIN_TREND_POINTS = 4


def _round(value: Optional[float], digits: int = 2) -> Optional[float]:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), digits)


def _median(values: List[float]) -> Optional[float]:
    return float(np.median(values)) if values else None


def _mean(values: List[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def _load_turns(timeline: Dict[str, Any]) -> List[Dict[str, Any]]:
    agent = timeline.get("agent_speaker_id")
    client = timeline.get("client_speaker_id")
    turns = []
    for raw in timeline.get("turns") or []:
        if not isinstance(raw, dict):
            continue
        try:
            start = float(raw["start"])
            end = float(raw["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        speaker = str(raw.get("speaker_id") or "")
        role = "agent" if (agent and speaker == agent) else ("client" if (client and speaker == client) else "other")
        turns.append({
            "start": start,
            "end": end,
            "duration": end - start,
            "speaker": speaker,
            "role": role,
            "text": str(raw.get("text") or ""),
        })
    turns.sort(key=lambda t: t["start"])
    return turns


def _word_rate(turn: Dict[str, Any]) -> Optional[float]:
    words = len(turn["text"].split())
    if words == 0 or turn["duration"] < AM_MIN_RATE_TURN_S:
        return None
    return words / turn["duration"]


def _conversation_metrics(turns: List[Dict[str, Any]], timeline: Dict[str, Any]) -> Dict[str, Any]:
    interruptions = {"agent_interrupted_client": 0, "client_interrupted_agent": 0}
    for i, later in enumerate(turns):
        if later["role"] not in ("agent", "client") or later["duration"] < AM_INTERRUPT_MIN_TURN_S:
            continue
        for earlier in reversed(turns[max(0, i - 6):i]):
            if earlier["role"] not in ("agent", "client") or earlier["role"] == later["role"]:
                continue
            overlap = min(earlier["end"], later["end"]) - later["start"]
            if earlier["start"] < later["start"] < earlier["end"] and overlap >= AM_INTERRUPT_MIN_OVERLAP_S:
                key = "agent_interrupted_client" if later["role"] == "agent" else "client_interrupted_agent"
                interruptions[key] += 1
                break

    latencies = []
    speaking = [t for t in turns if t["role"] in ("agent", "client")]
    for prev, cur in zip(speaking, speaking[1:]):
        if prev["role"] == "client" and cur["role"] == "agent":
            gap = cur["start"] - prev["end"]
            if 0.0 <= gap <= AM_RESPONSE_MAX_GAP_S:
                latencies.append(gap)

    dead_air_gaps, hold_gaps = [], []
    max_end = None
    for turn in turns:
        if max_end is not None:
            gap = turn["start"] - max_end
            if gap >= AM_HOLD_MIN_S:
                hold_gaps.append(gap)
            elif gap >= AM_DEAD_AIR_MIN_S:
                dead_air_gaps.append(gap)
        max_end = turn["end"] if max_end is None else max(max_end, turn["end"])

    total_duration = float(timeline.get("total_duration_seconds") or 0.0)
    return {
        "total_duration_seconds": _round(total_duration),
        "agent_talk_ratio_pct": timeline.get("agent_talk_ratio_pct"),
        "interruptions": interruptions,
        "response_latency": {
            "count": len(latencies),
            "mean_s": _round(_mean(latencies)),
            "median_s": _round(_median(latencies)),
            "max_s": _round(max(latencies)) if latencies else None,
        },
        "dead_air": {"count": len(dead_air_gaps), "total_s": _round(sum(dead_air_gaps))},
        "hold_candidates": {"count": len(hold_gaps), "total_s": _round(sum(hold_gaps))},
    }


def _speaking_rate_metrics(turns: List[Dict[str, Any]]) -> Dict[str, Any]:
    agent_turns = [t for t in turns if t["role"] == "agent"]
    client_turns = [t for t in turns if t["role"] == "client"]
    agent_rates = [(t, _word_rate(t)) for t in agent_turns]
    agent_rates = [(t, r) for t, r in agent_rates if r is not None]
    client_rates = [r for r in (_word_rate(t) for t in client_turns) if r is not None]

    baseline = _median([r for _, r in agent_rates])
    rushed = 0
    if baseline:
        for t, r in agent_rates:
            words = len(t["text"].split())
            if (
                t["duration"] >= AM_RUSHED_MIN_TURN_S
                and words >= 6
                and r > AM_RUSHED_RATE_RATIO * baseline
            ):
                rushed += 1

    return {
        "agent_words_per_second_median": _round(baseline),
        "client_words_per_second_median": _round(_median(client_rates)),
        "agent_rushed_turns": rushed,
        "agent_rushed_turn_share_pct": _round(rushed / len(agent_rates) * 100.0, 1) if agent_rates else None,
    }


def _load_audio(wav_path: Optional[str]):
    if not wav_path or not os.path.exists(wav_path):
        return None, None, "clean WAV not available"
    try:
        import soundfile as sf
    except ImportError:
        return None, None, "soundfile not installed"
    audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, int(sr), None


def _turn_acoustics(y: np.ndarray, sr: int, start: float, end: float) -> Optional[Dict[str, float]]:
    import librosa

    seg = y[int(start * sr):int(end * sr)]
    if seg.size < int(0.3 * sr):
        return None
    rms = librosa.feature.rms(y=seg, frame_length=_FRAME_LENGTH, hop_length=_HOP_LENGTH)[0]
    if rms.size == 0 or float(rms.max()) < 1e-4:
        return None
    f0 = librosa.yin(
        seg,
        fmin=AM_PITCH_MIN_HZ,
        fmax=AM_PITCH_MAX_HZ,
        sr=sr,
        frame_length=_FRAME_LENGTH,
        hop_length=_HOP_LENGTH,
    )
    n = min(len(f0), len(rms))
    f0, rms = f0[:n], rms[:n]
    voiced = rms > 0.1 * float(rms.max())
    pitched = voiced & (f0 > AM_PITCH_MIN_HZ) & (f0 < AM_PITCH_MAX_HZ)
    if int(voiced.sum()) < _MIN_VOICED_FRAMES:
        return None

    energy_db = 20.0 * math.log10(float(rms[voiced].mean()) + 1e-9)
    result: Dict[str, float] = {"energy_db": energy_db}
    if int(pitched.sum()) >= _MIN_VOICED_FRAMES:
        f_values = f0[pitched].astype(np.float64)
        f_ref = float(np.median(f_values))
        semitones = 12.0 * np.log2(f_values / f_ref)
        result["pitch_median_hz"] = f_ref
        result["pitch_std_st"] = float(np.std(semitones))
        result["pitch_range_st"] = float(np.percentile(semitones, 90) - np.percentile(semitones, 10))
    return result


def _acoustic_metrics(turns: List[Dict[str, Any]], wav_path: Optional[str]) -> Dict[str, Any]:
    y, sr, reason = _load_audio(wav_path)
    if y is None:
        return {"available": False, "reason": reason}

    per_turn: Dict[str, List[Dict[str, Any]]] = {"agent": [], "client": []}
    for t in turns:
        if t["role"] not in per_turn or t["duration"] < AM_MIN_ACOUSTIC_TURN_S:
            continue
        try:
            feats = _turn_acoustics(y, sr, t["start"], t["end"])
        except Exception as exc:
            logger.debug(f"Acoustic features skipped for turn at {t['start']:.2f}s: {exc}")
            continue
        if feats is None:
            continue
        feats.update({"mid_min": (t["start"] + t["end"]) / 2.0 / 60.0, "words_per_second": _word_rate(t)})
        per_turn[t["role"]].append(feats)

    def summarise(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        pitched = [f for f in items if "pitch_std_st" in f]
        return {
            "analysed_turns": len(items),
            "pitch_median_hz": _round(_median([f["pitch_median_hz"] for f in pitched]), 1),
            "pitch_std_st_median": _round(_median([f["pitch_std_st"] for f in pitched])),
            "pitch_range_st_median": _round(_median([f["pitch_range_st"] for f in pitched])),
            "energy_db_mean": _round(_mean([f["energy_db"] for f in items]), 1),
            "energy_db_std": _round(float(np.std([f["energy_db"] for f in items])) if items else None, 1),
        }

    def trend(items: List[Dict[str, Any]]) -> Optional[float]:
        if len(items) < _MIN_TREND_POINTS:
            return None
        x = np.array([f["mid_min"] for f in items])
        yv = np.array([f["energy_db"] for f in items])
        if float(np.ptp(x)) == 0.0:
            return None
        return _round(float(np.polyfit(x, yv, 1)[0]), 3)

    agent_items = per_turn["agent"]
    client_items = per_turn["client"]
    fatigue = None
    if len(agent_items) >= 6:
        first, last = agent_items[:3], agent_items[-3:]
        pitch_first = _median([f["pitch_std_st"] for f in first if "pitch_std_st" in f])
        pitch_last = _median([f["pitch_std_st"] for f in last if "pitch_std_st" in f])
        energy_first = _mean([f["energy_db"] for f in first])
        energy_last = _mean([f["energy_db"] for f in last])
        rate_first = _median([f["words_per_second"] for f in first if f["words_per_second"]])
        rate_last = _median([f["words_per_second"] for f in last if f["words_per_second"]])
        rate_change_pct = (
            (rate_last - rate_first) / rate_first * 100.0 if rate_first and rate_last else None
        )
        pitch_change = (pitch_last - pitch_first) if pitch_first is not None and pitch_last is not None else None
        energy_change = energy_last - energy_first
        fatigue = {
            "pitch_std_change_st": _round(pitch_change),
            "energy_db_change": _round(energy_change, 1),
            "word_rate_change_pct": _round(rate_change_pct, 1),
        }

    return {
        "available": True,
        "reason": None,
        "agent": summarise(agent_items),
        "client": summarise(client_items),
        "agent_volume_trend_db_per_min": trend(agent_items),
        "client_volume_trend_db_per_min": trend(client_items),
        "agent_fatigue_drift": fatigue,
    }


def _flags(rate: Dict[str, Any], acoustics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    flags: Dict[str, Any] = {
        "agent_rushed": None,
        "agent_monotone": None,
        "agent_fatigue_signal": None,
        "call_volume_escalation": None,
    }
    if rate.get("agent_rushed_turn_share_pct") is not None:
        flags["agent_rushed"] = rate["agent_rushed_turn_share_pct"] >= 25.0

    if acoustics and acoustics.get("available"):
        agent_pitch_std = acoustics["agent"]["pitch_std_st_median"]
        if agent_pitch_std is not None and acoustics["agent"]["analysed_turns"] >= 3:
            flags["agent_monotone"] = agent_pitch_std < AM_MONOTONE_PITCH_STD_ST

        drift = acoustics.get("agent_fatigue_drift")
        if drift:
            indicators = [
                drift["pitch_std_change_st"] is not None and drift["pitch_std_change_st"] <= -AM_FATIGUE_PITCH_DROP_ST,
                drift["energy_db_change"] is not None and drift["energy_db_change"] <= -AM_FATIGUE_ENERGY_DROP_DB,
                drift["word_rate_change_pct"] is not None and drift["word_rate_change_pct"] <= -AM_FATIGUE_RATE_DROP_PCT,
            ]
            flags["agent_fatigue_signal"] = sum(indicators) >= 2

        a_trend = acoustics.get("agent_volume_trend_db_per_min")
        c_trend = acoustics.get("client_volume_trend_db_per_min")
        if a_trend is not None and c_trend is not None:
            flags["call_volume_escalation"] = (
                a_trend >= AM_ESCALATION_DB_PER_MIN and c_trend >= AM_ESCALATION_DB_PER_MIN
            )
    return flags


def _section(notes: List[str], name: str, fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception as exc:
        logger.warning(f"Agent metric '{name}' skipped: {exc}")
        notes.append(f"{name}: {type(exc).__name__}: {exc}")
        return None


def compute_agent_performance(timeline: Dict[str, Any], wav_path: Optional[str]) -> Dict[str, Any]:
    notes: List[str] = []
    turns = _section(notes, "turns", lambda: _load_turns(timeline)) or []
    conversation = _section(notes, "conversation", lambda: _conversation_metrics(turns, timeline))
    rate = _section(notes, "speaking_rate", lambda: _speaking_rate_metrics(turns))
    acoustics = _section(notes, "acoustics", lambda: _acoustic_metrics(turns, wav_path))
    flags = _section(notes, "flags", lambda: _flags(rate or {}, acoustics)) or {}

    return {
        "version": AGENT_METRICS_VERSION,
        "agent_speaker_id": timeline.get("agent_speaker_id"),
        "client_speaker_id": timeline.get("client_speaker_id"),
        "conversation": conversation,
        "speaking_rate": rate,
        "acoustics": acoustics,
        "flags": flags,
        "notes": notes,
    }


def attach_agent_performance(eval_data: Dict[str, Any], wav_path: Optional[str]) -> None:
    timeline = eval_data.get("diarization_timeline")
    if not isinstance(timeline, dict):
        return
    try:
        eval_data["agent_performance"] = compute_agent_performance(timeline, wav_path)
    except Exception as exc:
        logger.warning(f"Agent performance metrics unavailable: {exc}")
        eval_data["agent_performance"] = {
            "version": AGENT_METRICS_VERSION,
            "error": f"{type(exc).__name__}: {exc}",
        }
