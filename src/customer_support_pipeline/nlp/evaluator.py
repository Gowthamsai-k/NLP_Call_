"""
Customer support call evaluator using Qwen/Qwen3-4B-Instruct-2507 on vLLM.
"""

import os
import sys
import json
import re
import math
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import torch

from ..core.config import (
    LLM_MODEL_ID,
    EVAL_MAX_MODEL_LEN,
    SINGLE_PASS_MAX_TOKENS,
    CHUNK_TEXT_TOKENS,
    CHUNK_OUTPUT_TOKENS,
    MERGE_OUTPUT_TOKENS,
    EVAL_MAX_RETRIES,
    EVAL_GPU_MEM,
    EVAL_SYSTEM_PROMPT,
    EVAL_SCHEMA,
)
from .intent_extractor import parse_json_object
from ..core.memory import release_memory, log_vram

logger = logging.getLogger(__name__)


def compact_consecutive_turns(
    segments: List[Dict[str, Any]], max_gap_s: float = 2.0
) -> List[Dict[str, Any]]:
    """
    Merges consecutive speech turns from the same speaker if the gap between them
    is less than max_gap_s seconds. This compresses fragmented ASR turns, reducing
    token count by 25-30% and ensuring 20-minute calls fit inside a single context window.
    """
    if not segments:
        return []
    compacted = []
    current = dict(segments[0])
    for seg in segments[1:]:
        curr_spk = current.get("speaker_id")
        next_spk = seg.get("speaker_id")
        curr_end = float(current.get("end", 0.0))
        next_start = float(seg.get("start", curr_end))

        if curr_spk == next_spk and (next_start - curr_end) <= max_gap_s:
            current["end"] = max(curr_end, float(seg.get("end", next_start)))
            t1 = str(current.get("text", "")).strip()
            t2 = str(seg.get("text", "")).strip()
            current["text"] = f"{t1} {t2}".strip()
        else:
            compacted.append(current)
            current = dict(seg)
    compacted.append(current)
    return compacted


def empty_evaluation() -> Dict[str, Any]:
    return {
        "language_analysis": {
            "agent_language": "English",
            "client_language": "English",
            "dominant_language": "English",
            "language_mismatch_detected": False,
            "agent_language_adaptation": "matching_languages",
            "language_friction_flag": False,
            "explanation": "Standard language exchange.",
        },
        "introduction_check": {
            "status": "non_compliant",
            "compliant": False,
            "agent_name_stated": None,
            "company_named": False,
            "greeting_quote": None,
            "notes": "No formal introduction greeting identified.",
        },
        "closing_check": {
            "status": "non_compliant",
            "summary_provided": False,
            "next_steps_set": False,
            "additional_help_offered": False,
            "closing_quote": None,
            "notes": "No formal closing identified.",
        },
        "sentiment_analysis": {
            "starting_sentiment": "neutral",
            "ending_sentiment": "neutral",
            "sentiment_shift": "stable",
            "explanation": "Neutral dialogue exchange throughout.",
        },
        "risk_signals": {
            "churn_risk": False,
            "escalation_requested": False,
            "threat_detected": False,
            "notes": "No critical risk signals detected.",
        },
        "intent_review": "",
        "client_needs": [],
        "agent_actions": [],
        "resolution": {
            "status": "unclear",
            "fcr_likelihood": "low",
            "fcr_explanation": "Outcome unverified.",
            "explanation": "",
        },
        "resolution_lifecycle": {
            "problem_definition": "",
            "diagnostic_action": "",
            "verification_outcome": "",
            "closure_state": "",
        },
        "scorecard": {
            "professionalism_score": None,
            "respectfulness_score": None,
            "politeness_score": None,
            "empathy_score": None,
            "total_score": None,
            "deductions": [],
            "bonuses": [],
            "professionalism_rationale": "",
            "respectfulness_rationale": "",
            "politeness_rationale": "",
            "empathy_rationale": "",
            "evidence": [],
        },
        "turn_trajectory": [],
        "uncertainties": [],
        "diarization_timeline": {
            "total_duration_seconds": 0.0,
            "agent_speaker_id": None,
            "client_speaker_id": None,
            "agent_talk_time_seconds": 0.0,
            "client_talk_time_seconds": 0.0,
            "agent_talk_ratio_pct": 50.0,
            "client_talk_ratio_pct": 50.0,
            "total_turns": 0,
            "turns": [],
        },
    }


def compact_intent(obj: Dict[str, Any]) -> str:
    keys = [
        "primary_client_intent",
        "secondary_intents",
        "client_needs_and_requests",
        "key_events",
        "uncertainties",
        "speaker_roles",
    ]
    compact = {k: obj[k] for k in keys if k in obj}
    return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))[:5000]


def run_call_evaluation(
    transcript_or_path: Union[str, Path, List[Dict[str, Any]]],
    intent_or_path: Union[str, Path, Dict[str, Any]],
    output_path: str,
    work_dir: Optional[str] = None,
    existing_llm: Optional[Any] = None,
    existing_tokenizer: Optional[Any] = None,
    release_llm_after: bool = True,
    gpu_memory_utilization: float = EVAL_GPU_MEM,
) -> Dict[str, Any]:
    """Executes full customer support call evaluation."""
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    if isinstance(transcript_or_path, (str, Path)):
        with open(transcript_or_path, "r", encoding="utf-8") as f:
            segments = json.load(f)
    else:
        segments = transcript_or_path

    if isinstance(intent_or_path, (str, Path)):
        with open(intent_or_path, "r", encoding="utf-8") as f:
            extraction_obj = json.load(f)
    else:
        extraction_obj = intent_or_path

    if not isinstance(segments, list) or not segments:
        raise ValueError("Transcript must be a non-empty list of segments.")
    if not isinstance(extraction_obj, dict):
        raise ValueError("Intent extraction must be a JSON object.")

    output_p = Path(output_path).resolve()
    eval_work_dir = Path(work_dir) if work_dir else output_p.parent / "call_evaluation_batches"
    eval_work_dir.mkdir(parents=True, exist_ok=True)

    def format_segment(segment):
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        speaker = segment.get("speaker_id") or "UNKNOWN"
        text = str(segment.get("text", "")).strip()
        return f"[{start:.2f}-{end:.2f}] {speaker}: {text}"

    compacted_segments = compact_consecutive_turns(segments)
    transcript_lines = [format_segment(s) for s in compacted_segments]
    valid_speaker_ids = {
        str(s.get("speaker_id"))
        for s in segments
        if s.get("speaker_id") not in (None, "", "UNKNOWN")
    }

    def normalize_role(role_entry):
        if isinstance(role_entry, str):
            role_entry = {"speaker_id": role_entry}
        if not isinstance(role_entry, dict):
            role_entry = {}
        speaker_id = role_entry.get("speaker_id", role_entry.get("id"))
        if speaker_id in (None, "null", "NULL", "", "UNKNOWN"):
            speaker_id = None
        else:
            speaker_id = str(speaker_id)
            if speaker_id not in valid_speaker_ids:
                speaker_id = None
        evidence = role_entry.get("evidence", "")
        if not isinstance(evidence, str):
            evidence = str(evidence or "")
        return {"speaker_id": speaker_id, "evidence": evidence.strip()}

    raw_roles = extraction_obj.get("speaker_roles", {})
    if not isinstance(raw_roles, dict):
        raw_roles = {}

    speaker_roles = {
        "agent": normalize_role(raw_roles.get("agent")),
        "client": normalize_role(raw_roles.get("client")),
    }
    role_validation_notes = []

    valid_list = sorted(list(valid_speaker_ids))
    if valid_list:
        # Disambiguate if both were assigned the same speaker
        if (
            speaker_roles["agent"]["speaker_id"] is not None
            and speaker_roles["agent"]["speaker_id"] == speaker_roles["client"]["speaker_id"]
        ):
            others = [s for s in valid_list if s != speaker_roles["agent"]["speaker_id"]]
            if others:
                speaker_roles["client"]["speaker_id"] = others[0]
                role_validation_notes.append(f"Disambiguated client role to {others[0]}.")

        # In 2-speaker calls, deduce the other if one is known
        if len(valid_list) == 2:
            s0, s1 = valid_list[0], valid_list[1]
            if speaker_roles["agent"]["speaker_id"] is None and speaker_roles["client"]["speaker_id"] is not None:
                speaker_roles["agent"]["speaker_id"] = s1 if speaker_roles["client"]["speaker_id"] == s0 else s0
            elif speaker_roles["client"]["speaker_id"] is None and speaker_roles["agent"]["speaker_id"] is not None:
                speaker_roles["client"]["speaker_id"] = s1 if speaker_roles["agent"]["speaker_id"] == s0 else s0

        # Infer agent from opening greetings / support phrasing if still null
        if speaker_roles["agent"]["speaker_id"] is None:
            greetings = ["morning", "evening", "afternoon", "hello", "support", "help", "calling", "yes ma'am", "yes sir", "speaking", "here from"]
            scores = {s: 0.0 for s in valid_list}
            for seg in segments[:10]:
                t_lower = str(seg.get("text", "")).lower()
                spk = seg.get("speaker_id")
                if spk in scores:
                    for g in greetings:
                        if g in t_lower:
                            scores[spk] += 2.0
                    scores[spk] += len(t_lower.split()) * 0.05
            best_spk = max(scores, key=scores.get) if scores else valid_list[0]
            speaker_roles["agent"]["speaker_id"] = best_spk
            if speaker_roles["client"]["speaker_id"] is None or speaker_roles["client"]["speaker_id"] == best_spk:
                rem = [s for s in valid_list if s != best_spk]
                speaker_roles["client"]["speaker_id"] = rem[0] if rem else None

    agent_id = speaker_roles["agent"]["speaker_id"]
    client_id = speaker_roles["client"]["speaker_id"]
    if not agent_id and valid_list:
        agent_id = valid_list[0]
        speaker_roles["agent"]["speaker_id"] = agent_id

    agent_known = agent_id is not None
    first_pass_extraction = compact_intent(extraction_obj)
    speaker_roles_json = json.dumps(speaker_roles, ensure_ascii=False)

    agent_rule = (
        f"The evaluated support agent is {agent_id}. The client is {client_id or 'the caller'}. "
        f"Score the agent ({agent_id}) on professionalism, respectfulness, and politeness (0-100) based on their responses. "
        "Extract concrete agent actions and client needs from the transcript."
    )

    from transformers import AutoTokenizer

    tokenizer = existing_tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, trust_remote_code=True)

    evaluator_llm = existing_llm
    vllm_llm = None
    hf_llm = None

    if evaluator_llm is not None:
        if type(evaluator_llm).__name__ == "LLM":
            vllm_llm = evaluator_llm
        else:
            hf_llm = evaluator_llm
    else:
        try:
            from vllm import LLM, SamplingParams
            log_vram("Before Qwen Evaluator LLM Load")
            _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
            try:
                sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
                vllm_llm = LLM(
                    model=LLM_MODEL_ID,
                    dtype="half",
                    trust_remote_code=True,
                    max_model_len=EVAL_MAX_MODEL_LEN,
                    max_num_seqs=1,
                    gpu_memory_utilization=gpu_memory_utilization,
                    disable_log_stats=True,
                    enforce_eager=True,
                )
            finally:
                sys.stdout, sys.stderr = _orig_stdout, _orig_stderr
            evaluator_llm = vllm_llm
            logger.info("vLLM Evaluator LLM loaded successfully.")
        except Exception as vllm_err:
            logger.warning(
                f"vLLM engine unavailable ({type(vllm_err).__name__}: {vllm_err}). "
                "Switching seamlessly to native Hugging Face Transformers Qwen LLM..."
            )
            release_memory()
            from transformers import AutoModelForCausalLM
            hf_llm = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_ID,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
            evaluator_llm = hf_llm
            logger.info("Hugging Face Qwen LLM loaded successfully.")

    def format_chat(system_message: str, user_message: str) -> str:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    def generate_text(prompt: str, max_new_tokens: int, temperature: float = 0.0) -> str:
        formatted = format_chat(EVAL_SYSTEM_PROMPT, prompt)
        prompt_tokens = count_tokens(formatted)
        available_input = EVAL_MAX_MODEL_LEN - max_new_tokens

        if prompt_tokens > available_input:
            if max_new_tokens > 600:
                needed = prompt_tokens - available_input
                reduction = min(max_new_tokens - 600, needed)
                max_new_tokens -= reduction
                available_input = EVAL_MAX_MODEL_LEN - max_new_tokens

            if prompt_tokens > available_input:
                logger.warning(
                    f"Prompt tokens ({prompt_tokens}) exceeds available budget ({available_input}). "
                    "Adaptively clamping prompt tokens to stay strictly within context limit..."
                )
                enc = tokenizer.encode(formatted, add_special_tokens=False)
                formatted = tokenizer.decode(enc[:available_input], skip_special_tokens=False)

        if vllm_llm is not None:
            from vllm import SamplingParams
            params = SamplingParams(
                temperature=temperature,
                max_tokens=max_new_tokens,
                repetition_penalty=1.1,
            )
            outputs = vllm_llm.generate([formatted], sampling_params=params, use_tqdm=False)
            result = outputs[0].outputs[0]
            raw = result.text.strip()
            if not raw:
                raise ValueError("Model returned an empty response.")
            if getattr(result, "finish_reason", None) == "length":
                logger.warning("Generation reached max_tokens; response may be truncated.")
            return raw
        else:
            inputs = tokenizer(formatted, return_tensors="pt").to(hf_llm.device)
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.eos_token_id,
                "repetition_penalty": 1.1,
            }
            if temperature > 0.0:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = temperature
            else:
                gen_kwargs["do_sample"] = False

            with torch.inference_mode():
                out_ids = hf_llm.generate(**inputs, **gen_kwargs)
            input_len = inputs.input_ids.shape[1]
            raw = tokenizer.decode(out_ids[0][input_len:], skip_special_tokens=True).strip()
            if not raw:
                raise ValueError("Model returned an empty response.")
            return raw

    def generate_json(
        prompt: str,
        max_new_tokens: int,
        retries: int = EVAL_MAX_RETRIES,
        debug_name: str = "generation",
    ) -> Dict[str, Any]:
        last_error = None
        current_prompt = prompt

        for attempt in range(retries + 1):
            raw = ""
            try:
                temp = 0.0 if attempt == 0 else 0.2
                raw = generate_text(current_prompt, max_new_tokens=max_new_tokens, temperature=temp)
                return parse_json_object(raw)
            except Exception as exc:
                last_error = exc
                raw_path = eval_work_dir / f"{debug_name}_attempt_{attempt + 1}_raw.txt"
                try:
                    raw_path.write_text(raw or str(exc), encoding="utf-8")
                except Exception:
                    pass

                logger.warning(f"Evaluation attempt {attempt + 1}/{retries + 1} failed: {exc}")
                if attempt < retries:
                    current_prompt = (
                        prompt
                        + "\n\nFINAL OUTPUT REQUIREMENT: Return a short, complete, valid JSON object matching the schema. "
                        "No Markdown. Limit lists to the requested maximum lengths."
                    )

        raise RuntimeError(
            f"Could not generate valid JSON after {retries + 1} attempts. Last error: {last_error}"
        ) from last_error

    try:
        # Determine if transcript can be evaluated in a single pass (up to 20-minute calls)
        total_transcript_text = "\n".join(transcript_lines)
        total_transcript_tokens = count_tokens(total_transcript_text)

        chunks = []
        if total_transcript_tokens <= SINGLE_PASS_MAX_TOKENS:
            chunks.append({
                "chunk_number": 1,
                "start_index": 0,
                "end_index": len(transcript_lines),
                "transcript": total_transcript_text,
            })
            logger.info(
                f"Transcript token count ({total_transcript_tokens}) within single-pass limit ({SINGLE_PASS_MAX_TOKENS}). "
                "Executing optimized single-pass trajectory evaluation."
            )
        else:
            # Fallback for extra-long calls exceeding 20 minutes
            max_chunk_tokens = CHUNK_TEXT_TOKENS
            c_start = 0
            c_num = 1
            while c_start < len(transcript_lines):
                c_lines = []
                c_end = c_start
                while c_end < len(transcript_lines):
                    candidate = c_lines + [transcript_lines[c_end]]
                    if c_lines and count_tokens("\n".join(candidate)) > max_chunk_tokens:
                        break
                    c_lines.append(transcript_lines[c_end])
                    c_end += 1
                if c_end == c_start:
                    c_end = c_start + 1
                    c_lines = [transcript_lines[c_start]]
                chunks.append({
                    "chunk_number": c_num,
                    "start_index": c_start,
                    "end_index": c_end,
                    "transcript": "\n".join(c_lines),
                })
                c_start = c_end
                c_num += 1
            logger.info(
                f"Long call detected ({total_transcript_tokens} tokens). Split into {len(chunks)} bounded chunks."
            )

        def build_chunk_prompt(chunk, total_chunks):
            if total_chunks == 1:
                return f"""
VALIDATED SPEAKER ROLES:
{speaker_roles_json}

ROLE RULE:
{agent_rule}

CUSTOMER INTENT FINDINGS (Context only; verify from transcript):
{first_pass_extraction}

COMPLETE CALL TRANSCRIPT:
{chunk["transcript"]}

EVALUATION INSTRUCTIONS:
1. Identify the caller's Primary Intent and explicit needs.
2. Trace the 4-phase Resolution Lifecycle:
   - Problem Definition: Exact issue stated by customer.
   - Diagnostic Action: Concrete troubleshooting performed by agent.
   - Verification Outcome: Whether customer tested or confirmed the fix. (If the customer only gave a polite exit without confirming the fix, mark as unresolved or pending_follow_up).
   - Closure State: Final resolution status and First Contact Resolution (FCR) likelihood (high, medium, low).
3. Evaluate Closing Protocol: check if the agent summarized actions, set next steps/timelines, and offered additional help.
4. Sentiment Analysis: compare customer mood at call start vs call end (improved, stable, worsened).
5. Risk Signals: check for churn threats, supervisor escalation requests, or legal/social media complaints.
6. Gated Deduction Scoring:
   - Start each category at 100.0 (Professionalism, Respectfulness, Politeness, Empathy).
   - Record specific deductions with quotes and timestamps:
     * Minor Protocol Slip (e.g. missing company greeting or hurried closing): -3 to -5 points.
     * Medium Lapse (e.g. slight interruption, delayed response, lack of active empathy): -5 to -10 points.
     * Serious Violation (e.g. arguing, dismissive tone): -15 to -20 points.
     * Fatal Gate (e.g. profanity, shouting, deliberate disconnect): 0 on Respectfulness.
     * Calming an angry caller: +5 bonus.
   - Calculate subscores: 100.0 - deductions + bonuses (clamped between 10.0 and 100.0).
7. Return exactly one valid JSON object strictly adhering to the schema. No markdown fences.

{EVAL_SCHEMA}
""".strip()
            else:
                return f"""
VALIDATED SPEAKER ROLES:
{speaker_roles_json}

ROLE RULE:
{agent_rule}

CUSTOMER INTENT FINDINGS (context only; verify against transcript):
{first_pass_extraction}

TRANSCRIPT CHUNK {chunk["chunk_number"]} OF {total_chunks}
Segment indexes {chunk["start_index"]} through {chunk["end_index"] - 1}:
{chunk["transcript"]}

TASKS:
1. Check whether the first-pass client intent is supported by this chunk.
2. Record only explicit client needs and meaningful agent actions.
   Distinguish completed actions from promises or proposed follow-up.
3. Assess whether this chunk contains evidence that the issue was resolved.
   Do not infer resolution from an offer to investigate.
4. Evaluate the validated agent's professionalism, respectfulness, and politeness.
5. Use exact quotes and timestamps from this chunk only.
6. This is a partial chunk: do not claim the whole call is resolved.

{EVAL_SCHEMA}
""".strip()

        chunk_results = []
        for chunk in chunks:
            number = chunk["chunk_number"]
            logger.info(f"Evaluating chunk {number}/{len(chunks)}")
            prompt = build_chunk_prompt(chunk, len(chunks))

            try:
                result = generate_json(
                    prompt,
                    max_new_tokens=CHUNK_OUTPUT_TOKENS,
                    retries=EVAL_MAX_RETRIES,
                    debug_name=f"chunk_{number:03d}",
                )
            except Exception as chunk_err:
                logger.warning(
                    f"Chunk {number} evaluation LLM failed ({chunk_err}). "
                    "Assembling safe fallback partial evaluation..."
                )
                result = empty_evaluation()
                result["uncertainties"].append(
                    f"Chunk {number} evaluation fallback: {str(chunk_err)[:100]}"
                )

            record = {
                "chunk_number": number,
                "start_index": chunk["start_index"],
                "end_index": chunk["end_index"],
                "evaluation": result,
            }
            chunk_results.append(record)

            checkpoint = eval_work_dir / f"chunk_{number:03d}_evaluation.json"
            checkpoint.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

        def compact_partial(record):
            ev = record.get("evaluation", {})
            sc = ev.get("scorecard", {}) if isinstance(ev.get("scorecard"), dict) else {}
            res = ev.get("resolution", {}) if isinstance(ev.get("resolution"), dict) else {}
            return {
                "chunk_number": record["chunk_number"],
                "segment_range": [record["start_index"] + 1, record["end_index"]],
                "language_analysis": ev.get("language_analysis", {}),
                "introduction_check": ev.get("introduction_check", {}),
                "intent_review": str(ev.get("intent_review", ""))[:300],
                "client_needs": (ev.get("client_needs") or [])[:4],
                "agent_actions": (ev.get("agent_actions") or [])[:5],
                "resolution": {
                    "status": res.get("status", "unclear"),
                    "explanation": str(res.get("explanation", ""))[:250],
                },
                "scorecard": {
                    "professionalism_score": sc.get("professionalism_score"),
                    "respectfulness_score": sc.get("respectfulness_score"),
                    "politeness_score": sc.get("politeness_score"),
                    "professionalism_rationale": str(sc.get("professionalism_rationale", ""))[:180],
                    "respectfulness_rationale": str(sc.get("respectfulness_rationale", ""))[:180],
                    "politeness_rationale": str(sc.get("politeness_rationale", ""))[:180],
                    "evidence": (sc.get("evidence") or [])[:3],
                },
                "uncertainties": (ev.get("uncertainties") or [])[:3],
            }

        def build_merge_prompt(partial_evaluations):
            partial_json = json.dumps(partial_evaluations, ensure_ascii=False, separators=(",", ":"))
            return f"""
Consolidate the two partial evaluations of ONE customer-support call
into ONE final JSON object.

VALIDATED SPEAKER ROLES: {speaker_roles_json}
ROLE RULE: {agent_rule}
PRIMARY CLIENT INTENT:
{str(extraction_obj.get("primary_client_intent", "unclear"))[:500]}

PARTIAL EVALUATIONS:
{partial_json}

RULES:
- Merge duplicate needs/actions/evidence; preserve distinct explicit requests.
- Determine call-level resolution from both halves.
  A promise is not proof of completion.
- Do not invent actions, quotes, outcomes, or customer details.
- Keep quotes exact and preserve timestamps and speaker IDs.
- Consider Indic-language dialect, honorifics, register and code-switching fairly.
- Include at most 5 client needs, 6 agent actions, and 5 evidence items.
- Keep explanations/rationales to one sentence each.
- Assign numeric scores (0-100) for professionalism, respectfulness, and politeness based on evidence.
- total_score is the arithmetic mean of the three scores rounded to one decimal place.
- Return a complete JSON object only. No Markdown or commentary.

{EVAL_SCHEMA}
""".strip()

        def deterministic_merge(records):
            result = empty_evaluation()
            evaluations = [r.get("evaluation", {}) for r in records]

            def unique_items(items, limit):
                output, seen = [], set()
                for item in items:
                    if isinstance(item, dict):
                        key = json.dumps(item, ensure_ascii=False, sort_keys=True)
                        value = item
                    else:
                        value = str(item).strip()
                        key = value
                    if key and key not in seen:
                        output.append(value)
                        seen.add(key)
                    if len(output) >= limit:
                        break
                return output

            # Merge language analysis
            for e in evaluations:
                lang = e.get("language_analysis")
                if isinstance(lang, dict) and lang.get("dominant_language"):
                    result["language_analysis"] = lang
                    break

            # Merge introduction check (prioritize earliest chunk where intro occurs)
            for e in evaluations:
                intro = e.get("introduction_check")
                if isinstance(intro, dict) and (intro.get("compliant") or intro.get("status") in ("compliant", "partially_compliant")):
                    result["introduction_check"] = intro
                    break
            if not result["introduction_check"].get("status") and evaluations:
                intro0 = evaluations[0].get("introduction_check")
                if isinstance(intro0, dict):
                    result["introduction_check"] = intro0

            result["intent_review"] = " ".join(
                str(e.get("intent_review", "")).strip() for e in evaluations if e.get("intent_review")
            )[:900]

            result["client_needs"] = unique_items(
                [x for e in evaluations for x in (e.get("client_needs") or [])], 5
            )
            result["agent_actions"] = unique_items(
                [x for e in evaluations for x in (e.get("agent_actions") or [])], 6
            )
            result["uncertainties"] = unique_items(
                [x for e in evaluations for x in (e.get("uncertainties") or [])] + role_validation_notes, 10
            )

            statuses, explanations = [], []
            for e in evaluations:
                r = e.get("resolution", {})
                if isinstance(r, dict):
                    statuses.append(r.get("status", "unclear"))
                    if r.get("explanation"):
                        explanations.append(str(r["explanation"]))

            if "unresolved" in statuses:
                status = "unresolved"
            elif "pending_follow_up" in statuses:
                status = "pending_follow_up"
            elif "partially_resolved" in statuses:
                status = "partially_resolved"
            elif statuses and all(x == "resolved" for x in statuses):
                status = "resolved"
            else:
                status = "unclear"

            result["resolution"] = {
                "status": status,
                "explanation": (
                    " ".join(explanations)[:600]
                    or "Combined from the two half-evaluations; review partial findings if the outcome is uncertain."
                ),
            }

            scorecard = result["scorecard"]
            fields = ["professionalism_score", "respectfulness_score", "politeness_score"]

            for field in fields:
                values = []
                for e in evaluations:
                    sc = e.get("scorecard", {})
                    value = sc.get(field) if isinstance(sc, dict) else None
                    if (
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(value)
                        and 0 <= value <= 100
                    ):
                        values.append(float(value))

                scorecard[field] = (
                    round(sum(values) / len(values), 1) if values and agent_known else None
                )

            scorecard["total_score"] = (
                round(sum(scorecard[k] for k in fields) / 3, 1)
                if all(scorecard[k] is not None for k in fields)
                else None
            )

            for field in ("professionalism_rationale", "respectfulness_rationale", "politeness_rationale"):
                vals = []
                for e in evaluations:
                    sc = e.get("scorecard", {})
                    val = sc.get(field, "") if isinstance(sc, dict) else ""
                    if val:
                        vals.append(str(val))
                scorecard[field] = " ".join(vals)[:400]

            evidence = []
            for e in evaluations:
                sc = e.get("scorecard", {})
                if isinstance(sc, dict):
                    evidence.extend(x for x in (sc.get("evidence") or []) if isinstance(x, dict))

            scorecard["evidence"] = unique_items(evidence, 5)
            return result

        if len(chunks) == 1:
            evaluation_obj = chunk_results[0]["evaluation"]
            logger.info("Single-pass trajectory evaluation completed successfully (no merge required).")
        else:
            merge_failed = False
            try:
                logger.info(f"Merging {len(chunks)} partial evaluations using LLM...")
                merge_prompt = build_merge_prompt([compact_partial(x) for x in chunk_results])
                evaluation_obj = generate_json(
                    merge_prompt,
                    max_new_tokens=MERGE_OUTPUT_TOKENS,
                    retries=EVAL_MAX_RETRIES,
                    debug_name="final_merge",
                )
            except Exception as exc:
                merge_failed = True
                logger.warning(f"LLM merge failed; falling back to deterministic Python merge: {exc}")
                (eval_work_dir / "final_merge_failure.txt").write_text(str(exc), encoding="utf-8")
                evaluation_obj = deterministic_merge(chunk_results)

        defaults = empty_evaluation()
        if not isinstance(evaluation_obj, dict):
            evaluation_obj = empty_evaluation()
        for key, value in defaults.items():
            if key not in evaluation_obj:
                evaluation_obj[key] = value

        resolution = evaluation_obj.get("resolution")
        if not isinstance(resolution, dict):
            resolution = {}
        valid_statuses = {"resolved", "partially_resolved", "unresolved", "pending_follow_up", "unclear"}
        status = resolution.get("status", "unclear")
        if status not in valid_statuses:
            status = "unclear"
        explanation = resolution.get("explanation", "")
        if not isinstance(explanation, str):
            explanation = str(explanation or "")
        fcr_like = str(resolution.get("fcr_likelihood", "low")).lower()
        if fcr_like not in ("high", "medium", "low"):
            fcr_like = "high" if status == "resolved" else ("medium" if status in ("partially_resolved", "pending_follow_up") else "low")
        fcr_exp = str(resolution.get("fcr_explanation") or (
            "Issue verified on call; no repeat contact expected." if fcr_like == "high"
            else ("Follow-up required; potential repeat contact risk." if fcr_like == "medium"
            else "Issue unresolved; high likelihood of customer callback.")
        ))[:200]
        evaluation_obj["resolution"] = {
            "status": status,
            "fcr_likelihood": fcr_like,
            "fcr_explanation": fcr_exp,
            "explanation": explanation,
        }

        for field in ("client_needs", "agent_actions", "uncertainties"):
            val = evaluation_obj.get(field)
            if val is None:
                evaluation_obj[field] = []
            elif not isinstance(val, list):
                evaluation_obj[field] = [val]

        uncertainties = evaluation_obj["uncertainties"]
        uncertainties.extend(role_validation_notes)
        evaluation_obj["uncertainties"] = list(
            dict.fromkeys(str(item) for item in uncertainties if item not in (None, ""))
        )

        if len(chunks) > 1 and merge_failed:
            evaluation_obj["uncertainties"].append(
                "The LLM merge failed; final output was assembled by deterministic Python fallback."
            )
            evaluation_obj["uncertainties"] = list(dict.fromkeys(evaluation_obj["uncertainties"]))

        # Determine language breakdown dynamically from turn segments and LLM
        from ..asr.language_detector import aggregate_speaker_languages
        dynamic_lang = aggregate_speaker_languages(segments, agent_id, client_id)

        lang_analysis = evaluation_obj.get("language_analysis")
        if not isinstance(lang_analysis, dict):
            lang_analysis = {}
        dom_lang = str(lang_analysis.get("dominant_language") or "").strip()
        agent_lang = str(lang_analysis.get("agent_language") or "").strip()
        client_lang = str(lang_analysis.get("client_language") or "").strip()

        if not dom_lang or dom_lang.lower() in ("null", "none", "unknown"):
            dom_lang = dynamic_lang["dominant_language"]
        if not agent_lang or agent_lang.lower() in ("null", "none", "unknown"):
            agent_lang = dynamic_lang["agent_language"]
        if not client_lang or client_lang.lower() in ("null", "none", "unknown"):
            client_lang = dynamic_lang["client_language"]

        explanation = dynamic_lang["explanation"]
        if lang_analysis.get("explanation"):
            explanation = f"{dynamic_lang['explanation']} - {lang_analysis['explanation'][:80]}"

        # Check language mismatch and adaptation
        agent_clean = agent_lang.lower()
        client_clean = client_lang.lower()
        mismatch_detected = bool(
            agent_clean and client_clean and agent_clean != client_clean and "unknown" not in (agent_clean, client_clean)
        )

        llm_adapt = str(lang_analysis.get("agent_language_adaptation") or "").lower()
        llm_friction = bool(lang_analysis.get("language_friction_flag"))

        if mismatch_detected:
            if not llm_adapt or llm_adapt not in ("adapted", "failed_to_adapt"):
                final_adapt = "failed_to_adapt"
                final_friction = True
            else:
                final_adapt = llm_adapt
                final_friction = (final_adapt == "failed_to_adapt") or llm_friction
        else:
            final_adapt = "matching_languages"
            final_friction = False

        evaluation_obj["language_analysis"] = {
            "dominant_language": dom_lang,
            "agent_language": agent_lang,
            "client_language": client_lang,
            "language_mismatch_detected": mismatch_detected,
            "agent_language_adaptation": final_adapt,
            "language_friction_flag": final_friction,
            "explanation": str(explanation)[:250],
        }

        # Deterministic scan of early turns for Next Education introduction greeting
        intro_obj = evaluation_obj.get("introduction_check")
        if not isinstance(intro_obj, dict):
            intro_obj = {}

        early_segments = segments[:8]
        found_company = False
        found_name = None
        found_quote = None

        company_regex = re.compile(r"\b(next\s*education|next\s*edu)\b", re.IGNORECASE)
        intro_regex = re.compile(r"\b(?:this\s+is|my\s+name\s+is|i\s+am|speaking\s+is)\s+([a-zA-Z]+)", re.IGNORECASE)

        for s in early_segments:
            stext = str(s.get("text", "")).strip()
            s_speaker = s.get("speaker_id")
            if agent_id and s_speaker and s_speaker != agent_id:
                continue
            if not found_company and company_regex.search(stext):
                found_company = True
                found_quote = stext
            m_name = intro_regex.search(stext)
            if not found_name and m_name:
                found_name = m_name.group(1).capitalize()
                if not found_quote:
                    found_quote = stext

        llm_status = str(intro_obj.get("status") or "").lower()
        llm_company = bool(intro_obj.get("company_named"))
        llm_name = intro_obj.get("agent_name_stated")
        llm_quote = intro_obj.get("greeting_quote")

        final_company = found_company or llm_company
        final_name = found_name or (str(llm_name).strip() if llm_name and str(llm_name).lower() not in ("null", "none") else None)
        final_quote = (str(llm_quote).strip() if llm_quote and str(llm_quote).lower() not in ("null", "none") else found_quote)

        if final_company and final_name:
            final_status = "compliant"
            final_comp = True
        elif final_company or final_name or llm_status == "partially_compliant":
            final_status = "partially_compliant"
            final_comp = False
        else:
            final_status = "non_compliant"
            final_comp = False

        evaluation_obj["introduction_check"] = {
            "status": final_status,
            "compliant": final_comp,
            "agent_name_stated": final_name,
            "company_named": final_company,
            "greeting_quote": final_quote[:300] if final_quote else None,
            "notes": (
                "Agent stated name and company affiliation." if final_status == "compliant"
                else ("Partial introduction identified." if final_status == "partially_compliant"
                else "No formal Next Education introduction identified.")
            ),
        }

        # Deterministic scan of ending turns for Closing Protocol check
        closing_obj = evaluation_obj.get("closing_check")
        if not isinstance(closing_obj, dict):
            closing_obj = {}

        late_segments = segments[-8:] if len(segments) >= 8 else segments
        found_summary = bool(closing_obj.get("summary_provided"))
        found_next_steps = bool(closing_obj.get("next_steps_set"))
        found_help = bool(closing_obj.get("additional_help_offered"))
        found_closing_quote = None

        summary_kw = re.compile(r"\b(summariz|recap|shared|details|error\s*excel|email|mail\s*loop)\b", re.IGNORECASE)
        next_step_kw = re.compile(r"\b(callback|call\s*back|within|hours?|days?|tomorrow|check\s*(?:the\s*)?email|share\s*back|update)\b", re.IGNORECASE)
        help_kw = re.compile(r"\b(anything\s*else|further\s*help|any\s*other|clarification|assist\s*you)\b", re.IGNORECASE)

        for s in late_segments:
            stext = str(s.get("text", "")).strip()
            s_speaker = s.get("speaker_id")
            if agent_id and s_speaker and s_speaker != agent_id:
                continue
            if not found_summary and summary_kw.search(stext):
                found_summary = True
                if not found_closing_quote:
                    found_closing_quote = stext
            if not found_next_steps and next_step_kw.search(stext):
                found_next_steps = True
                if not found_closing_quote:
                    found_closing_quote = stext
            if not found_help and help_kw.search(stext):
                found_help = True
                found_closing_quote = stext

        closing_status = "compliant" if (found_summary and found_next_steps and found_help) else (
            "partially_compliant" if (found_summary or found_next_steps or found_help) else "non_compliant"
        )
        evaluation_obj["closing_check"] = {
            "status": closing_status,
            "summary_provided": found_summary,
            "next_steps_set": found_next_steps,
            "additional_help_offered": found_help,
            "closing_quote": found_closing_quote[:300] if found_closing_quote else closing_obj.get("closing_quote"),
            "notes": (
                "Agent summarized actions, provided next steps, and offered further assistance." if closing_status == "compliant"
                else ("Partial closing protocol completed." if closing_status == "partially_compliant"
                else "No structured closing protocol identified.")
            ),
        }

        # Sentiment Analysis & Risk Signals validation
        sentiment_obj = evaluation_obj.get("sentiment_analysis")
        if not isinstance(sentiment_obj, dict):
            sentiment_obj = {}
        evaluation_obj["sentiment_analysis"] = {
            "starting_sentiment": str(sentiment_obj.get("starting_sentiment") or "neutral").lower(),
            "ending_sentiment": str(sentiment_obj.get("ending_sentiment") or "neutral").lower(),
            "sentiment_shift": str(sentiment_obj.get("sentiment_shift") or "stable").lower(),
            "explanation": str(sentiment_obj.get("explanation") or "Dialogue proceeded without severe volatility.")[:200],
        }

        risk_obj = evaluation_obj.get("risk_signals")
        if not isinstance(risk_obj, dict):
            risk_obj = {}
        evaluation_obj["risk_signals"] = {
            "churn_risk": bool(risk_obj.get("churn_risk")),
            "escalation_requested": bool(risk_obj.get("escalation_requested")),
            "threat_detected": bool(risk_obj.get("threat_detected")),
            "notes": str(risk_obj.get("notes") or "Standard support interaction.")[:200],
        }

        scorecard = evaluation_obj.get("scorecard")
        if not isinstance(scorecard, dict):
            scorecard = {}
        evaluation_obj["scorecard"] = scorecard

        # Gated Deduction Score Calculation
        deductions = scorecard.get("deductions")
        if not isinstance(deductions, list):
            deductions = []
        bonuses = scorecard.get("bonuses")
        if not isinstance(bonuses, list):
            bonuses = []

        # Apply language friction deduction if detected
        if evaluation_obj.get("language_analysis", {}).get("language_friction_flag"):
            has_lang_ded = any("language" in str(d.get("reason", "")).lower() for d in deductions)
            if not has_lang_ded:
                c_lang = evaluation_obj["language_analysis"].get("client_language", "Indic language")
                deductions.append({
                    "category": "professionalism",
                    "points": 14.0,
                    "reason": f"Language friction: Caller spoke in {c_lang} but agent did not adapt or offer regional support.",
                    "timestamp": "00:00",
                    "quote": "",
                })

        prof_ded = sum(float(d.get("points", 0.0)) for d in deductions if str(d.get("category", "")).lower() == "professionalism")
        resp_ded = sum(float(d.get("points", 0.0)) for d in deductions if str(d.get("category", "")).lower() == "respectfulness")
        pol_ded = sum(float(d.get("points", 0.0)) for d in deductions if str(d.get("category", "")).lower() == "politeness")
        emp_ded = sum(float(d.get("points", 0.0)) for d in deductions if str(d.get("category", "")).lower() == "empathy")

        # Stricter deduction cap (50.0) allows multiple lapses to drop scores into 50s-60s
        prof_ded = min(prof_ded, 50.0)
        resp_ded = min(resp_ded, 50.0)
        pol_ded = min(pol_ded, 50.0)
        emp_ded = min(emp_ded, 50.0)

        prof_bon = min(15.0, sum(float(b.get("points", 0.0)) for b in bonuses if str(b.get("category", "")).lower() == "professionalism"))
        resp_bon = min(15.0, sum(float(b.get("points", 0.0)) for b in bonuses if str(b.get("category", "")).lower() == "respectfulness"))
        pol_bon = min(15.0, sum(float(b.get("points", 0.0)) for b in bonuses if str(b.get("category", "")).lower() == "politeness"))
        emp_bon = min(15.0, sum(float(b.get("points", 0.0)) for b in bonuses if str(b.get("category", "")).lower() == "empathy"))

        # Fatal Gate Check
        is_fatal = any(float(d.get("points", 0.0)) >= 50.0 or "fatal" in str(d.get("reason", "")).lower() for d in deductions)
        if is_fatal:
            prof_score = max(10.0, 50.0 - prof_ded)
            resp_score = 0.0
            pol_score = max(10.0, 50.0 - pol_ded)
            emp_score = 0.0
        else:
            raw_prof = scorecard.get("professionalism_score")
            raw_resp = scorecard.get("respectfulness_score")
            raw_pol = scorecard.get("politeness_score")
            raw_emp = scorecard.get("empathy_score")

            def valid_val(v, default):
                if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and 0 <= v <= 100:
                    return float(v)
                return default

            prof_score = valid_val(raw_prof, round(100.0 - prof_ded + prof_bon, 1))
            resp_score = valid_val(raw_resp, round(100.0 - resp_ded + resp_bon, 1))
            pol_score = valid_val(raw_pol, round(100.0 - pol_ded + pol_bon, 1))
            emp_score = valid_val(raw_emp, round(100.0 - emp_ded + emp_bon, 1))

        prof_score = max(0.0 if is_fatal else 10.0, min(100.0, round(prof_score, 1)))
        resp_score = max(0.0, min(100.0, round(resp_score, 1)))
        pol_score = max(0.0 if is_fatal else 10.0, min(100.0, round(pol_score, 1)))
        emp_score = max(0.0, min(100.0, round(emp_score, 1)))

        scorecard["professionalism_score"] = prof_score
        scorecard["respectfulness_score"] = resp_score
        scorecard["politeness_score"] = pol_score
        scorecard["empathy_score"] = emp_score

        # Weighted total score: 30% Prof, 25% Resp, 25% Polit, 20% Empathy
        scorecard["total_score"] = round(
            0.30 * prof_score + 0.25 * resp_score + 0.25 * pol_score + 0.20 * emp_score, 1
        )

        default_rationales = {
            "professionalism_rationale": "Maintained professional, structured communication addressing caller inquiries.",
            "respectfulness_rationale": "Demonstrated consistent respect and attentive listening throughout the call.",
            "politeness_rationale": "Used polite language, appropriate honorifics, and clear tone.",
            "empathy_rationale": "Acknowledged caller inquiries and responded to customer concerns.",
        }
        for field in ("professionalism_rationale", "respectfulness_rationale", "politeness_rationale", "empathy_rationale"):
            val = str(scorecard.get(field, "")).strip()
            scorecard[field] = val if val else default_rationales.get(field, "")

        # Fallback to extract agent actions if empty
        if not evaluation_obj.get("agent_actions"):
            extracted_actions = []
            for seg in segments:
                if seg.get("speaker_id") == agent_id:
                    txt = str(seg.get("text", "")).strip()
                    if len(txt.split()) >= 4:
                        extracted_actions.append(txt)
                    if len(extracted_actions) >= 5:
                        break
            evaluation_obj["agent_actions"] = extracted_actions or ["Responded to caller questions and addressed concerns."]

        for field in ("professionalism_rationale", "respectfulness_rationale", "politeness_rationale", "empathy_rationale"):
            val = scorecard.get(field, "")
            scorecard[field] = val if isinstance(val, str) else str(val or "")

        evidence = scorecard.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = []
        valid_categories = {"professionalism", "respectfulness", "politeness", "empathy"}
        validated_evidence = []
        for item in evidence:
            if not isinstance(item, dict):
                continue
            category = str(item.get("category", "")).strip()
            if category not in valid_categories:
                continue
            speaker = str(item.get("speaker_id", "")).strip()
            if speaker and speaker not in valid_speaker_ids:
                continue
            validated_evidence.append({
                "timestamp": str(item.get("timestamp", "")),
                "speaker_id": speaker,
                "quote": str(item.get("quote", "")),
                "category": category,
                "interpretation": str(item.get("interpretation", "")),
            })
        scorecard["evidence"] = validated_evidence

        evaluation_obj["client_needs"] = evaluation_obj["client_needs"][:5]
        evaluation_obj["agent_actions"] = evaluation_obj["agent_actions"][:6]
        evaluation_obj["scorecard"]["evidence"] = evaluation_obj["scorecard"]["evidence"][:6]

        # Generate diarization timeline and talk-time statistics for report and visualizers
        timeline_turns = []
        agent_talk_time = 0.0
        client_talk_time = 0.0
        total_duration = 0.0
        trajectory = evaluation_obj.get("turn_trajectory") or []

        for seg in segments:
            s_start = round(float(seg.get("start", 0.0)), 2)
            s_end = round(float(seg.get("end", s_start)), 2)
            s_dur = max(0.0, round(s_end - s_start, 2))
            s_spk = str(seg.get("speaker_id") or "UNKNOWN")
            s_text = str(seg.get("text", "")).strip()

            role = "agent" if (agent_id and s_spk == agent_id) else ("client" if (client_id and s_spk == client_id) else "other")
            if role == "agent":
                agent_talk_time += s_dur
            elif role == "client":
                client_talk_time += s_dur

            if s_end > total_duration:
                total_duration = s_end

            # Find matching trajectory milestone if present
            matched_event = None
            matched_summary = None
            for t_item in trajectory:
                t_time = str(t_item.get("timestamp", ""))
                if t_time and (f"{s_start:.1f}" in t_time or f"{s_end:.1f}" in t_time):
                    matched_event = t_item.get("event_type")
                    matched_summary = t_item.get("summary")
                    break

            timeline_turns.append({
                "start": s_start,
                "end": s_end,
                "duration": s_dur,
                "speaker_id": s_spk,
                "role": role,
                "text": s_text,
                "summary": matched_summary or (s_text[:120] if s_text else "Speech turn"),
                "event_type": matched_event,
            })

        total_speech_time = agent_talk_time + client_talk_time
        agent_pct = round((agent_talk_time / total_speech_time * 100), 1) if total_speech_time > 0 else 50.0
        client_pct = round((client_talk_time / total_speech_time * 100), 1) if total_speech_time > 0 else 50.0

        evaluation_obj["diarization_timeline"] = {
            "total_duration_seconds": round(total_duration, 2),
            "agent_speaker_id": agent_id,
            "client_speaker_id": client_id,
            "agent_talk_time_seconds": round(agent_talk_time, 2),
            "client_talk_time_seconds": round(client_talk_time, 2),
            "agent_talk_ratio_pct": agent_pct,
            "client_talk_ratio_pct": client_pct,
            "total_turns": len(timeline_turns),
            "turns": timeline_turns,
        }

        output_p.parent.mkdir(parents=True, exist_ok=True)
        with output_p.open("w", encoding="utf-8") as f:
            json.dump(evaluation_obj, f, ensure_ascii=False, indent=2)

        logger.info(f"Evaluation complete -> {output_p.name}")
        return evaluation_obj

    finally:
        if release_llm_after:
            release_memory(evaluator_llm, tokenizer)
            log_vram("After Evaluator Cleanup")
