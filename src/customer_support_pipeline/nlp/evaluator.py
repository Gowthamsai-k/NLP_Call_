"""
Customer support call evaluation using 4-bit Qwen3 on vLLM.
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
    CHUNK_OUTPUT_TOKENS,
    MERGE_OUTPUT_TOKENS,
    EVAL_MAX_RETRIES,
    EVAL_GPU_MEM,
    EVAL_SYSTEM_PROMPT,
    EVAL_SCHEMA,
)
from .intent_extractor import parse_json_object
from ..core.memory import (
    bounded_gpu_memory_utilization,
    release_memory,
    log_vram,
)

logger = logging.getLogger(__name__)


def empty_evaluation() -> Dict[str, Any]:
    return {
        "intent_review": "",
        "client_needs": [],
        "agent_actions": [],
        "resolution": {
            "status": "unclear",
            "explanation": "",
        },
        "scorecard": {
            "professionalism_score": None,
            "respectfulness_score": None,
            "politeness_score": None,
            "total_score": None,
            "professionalism_rationale": "",
            "respectfulness_rationale": "",
            "politeness_rationale": "",
            "evidence": [],
        },
        "uncertainties": [],
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

    transcript_lines = [format_segment(s) for s in segments]
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

    if (
        speaker_roles["agent"]["speaker_id"] is not None
        and speaker_roles["agent"]["speaker_id"] == speaker_roles["client"]["speaker_id"]
    ):
        role_validation_notes.append(
            "Agent and client were assigned the same speaker ID; both assignments were marked uncertain."
        )
        speaker_roles["agent"]["speaker_id"] = None
        speaker_roles["client"]["speaker_id"] = None

    agent_id = speaker_roles["agent"]["speaker_id"]
    client_id = speaker_roles["client"]["speaker_id"]
    agent_known = agent_id is not None

    if not agent_known:
        role_validation_notes.append(
            "Agent speaker ID could not be validated. Agent-specific scores must be null."
        )

    first_pass_extraction = compact_intent(extraction_obj)
    speaker_roles_json = json.dumps(speaker_roles, ensure_ascii=False)

    agent_rule = (
        f"The validated agent speaker ID is {agent_id}. Score only turns with this exact ID."
        if agent_known
        else "The agent ID is unknown/unvalidated. Set all four scores to JSON null and do not attribute behavior to an unidentified speaker."
    )

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = existing_tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, trust_remote_code=True)

    evaluator_llm = existing_llm
    if evaluator_llm is None:
        log_vram("Before Qwen Evaluator LLM Load")
        _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
            evaluator_llm = LLM(
                model=LLM_MODEL_ID,
                dtype="half",
                trust_remote_code=True,
                max_model_len=EVAL_MAX_MODEL_LEN,
                max_num_seqs=1,
                gpu_memory_utilization=bounded_gpu_memory_utilization(gpu_memory_utilization),
                disable_log_stats=True,
            )
        finally:
            sys.stdout, sys.stderr = _orig_stdout, _orig_stderr

    def format_chat(system_message: str, user_message: str) -> str:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    def generate_text(prompt: str, max_new_tokens: int) -> str:
        formatted = format_chat(EVAL_SYSTEM_PROMPT, prompt)
        prompt_tokens = count_tokens(formatted)
        available_input = EVAL_MAX_MODEL_LEN - max_new_tokens

        if prompt_tokens > available_input:
            raise ValueError(
                f"Prompt too long: {prompt_tokens} input tokens + {max_new_tokens} output tokens "
                f"exceeds {EVAL_MAX_MODEL_LEN} context limit."
            )

        params = SamplingParams(temperature=0.1, max_tokens=max_new_tokens)
        outputs = evaluator_llm.generate([formatted], sampling_params=params, use_tqdm=False)
        result = outputs[0].outputs[0]
        raw = result.text.strip()
        if not raw:
            raise ValueError("Model returned an empty response.")
        if getattr(result, "finish_reason", None) == "length":
            logger.warning("Generation reached max_tokens; response may be truncated.")
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
                raw = generate_text(current_prompt, max_new_tokens=max_new_tokens)
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
        midpoint = (len(transcript_lines) + 1) // 2
        chunks = [
            {
                "chunk_number": 1,
                "start_index": 0,
                "end_index": midpoint,
                "transcript": "\n".join(transcript_lines[:midpoint]),
            },
            {
                "chunk_number": 2,
                "start_index": midpoint,
                "end_index": len(transcript_lines),
                "transcript": "\n".join(transcript_lines[midpoint:]),
            },
        ]

        def build_chunk_prompt(chunk, total_chunks):
            return f"""
VALIDATED SPEAKER ROLES:
{speaker_roles_json}

ROLE RULE:
{agent_rule}

FIRST-PASS INTENT FINDINGS (context only; verify against transcript):
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
5. For Telugu, Hindi, Tamil, Kannada, Malayalam, Bengali, Marathi,
   Gujarati, Punjabi, Odia, Assamese, Urdu, other Indic languages and
   code-switching, account for dialect, honorifics, register, relationship
   and context. Do not impose English-only norms or penalize a phrase
   merely for being informal. Consider ASR uncertainty.
6. Do not infer vocal tone, anger, or emotion from text alone.
7. Use exact quotes and timestamps from this chunk only.
   Avoid unintelligible fragments and routine repeated thanks.
8. This is a partial chunk: do not claim the whole call is resolved.
   Keep output concise and follow the schema.

{EVAL_SCHEMA}
""".strip()

        chunk_results = []
        for chunk in chunks:
            number = chunk["chunk_number"]
            logger.info(f"Evaluating chunk {number}/{len(chunks)}")
            prompt = build_chunk_prompt(chunk, len(chunks))

            result = generate_json(
                prompt,
                max_new_tokens=CHUNK_OUTPUT_TOKENS,
                retries=EVAL_MAX_RETRIES,
                debug_name=f"chunk_{number:03d}",
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
- Scores must be 0-100 only if the agent is validated and evidence is
  sufficient; otherwise null.
- total_score is the mean of the three numeric scores rounded to one
  decimal if all three exist; otherwise null.
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

        merge_failed = False
        try:
            logger.info("Merging partial evaluations using LLM...")
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
        evaluation_obj["resolution"] = {"status": status, "explanation": explanation}

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

        if merge_failed:
            evaluation_obj["uncertainties"].append(
                "The LLM merge failed; final output was assembled by deterministic Python fallback. Review half-level checkpoints."
            )
            evaluation_obj["uncertainties"] = list(dict.fromkeys(evaluation_obj["uncertainties"]))

        scorecard = evaluation_obj.get("scorecard")
        if not isinstance(scorecard, dict):
            scorecard = {}
        evaluation_obj["scorecard"] = scorecard

        score_fields = ["professionalism_score", "respectfulness_score", "politeness_score"]
        if not agent_known:
            for field in score_fields + ["total_score"]:
                scorecard[field] = None
        else:
            for field in score_fields:
                value = scorecard.get(field)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or not 0 <= value <= 100
                ):
                    scorecard[field] = None
                else:
                    scorecard[field] = round(float(value), 1)

            if all(scorecard.get(field) is not None for field in score_fields):
                scorecard["total_score"] = round(sum(scorecard[field] for field in score_fields) / 3, 1)
            else:
                scorecard["total_score"] = None

        for field in ("professionalism_rationale", "respectfulness_rationale", "politeness_rationale"):
            val = scorecard.get(field, "")
            scorecard[field] = val if isinstance(val, str) else str(val or "")

        evidence = scorecard.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = []
        valid_categories = {"professionalism", "respectfulness", "politeness"}
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

        output_p.parent.mkdir(parents=True, exist_ok=True)
        with output_p.open("w", encoding="utf-8") as f:
            json.dump(evaluation_obj, f, ensure_ascii=False, indent=2)

        logger.info(f"Evaluation complete -> {output_p.name}")
        return evaluation_obj

    finally:
        if release_llm_after:
            release_memory(evaluator_llm, tokenizer)
            log_vram("After Evaluator Cleanup")
