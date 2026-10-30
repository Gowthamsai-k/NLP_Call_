"""
Customer support intent extraction using Qwen/Qwen3-4B-Instruct-2507 on vLLM.
"""

import os
import sys
import json
import re
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import torch

from ..core.config import (
    LLM_MODEL_ID,
    INTENT_MAX_MODEL_LEN,
    MAX_TRANSCRIPT_TOKENS,
    MAX_BATCH_OUTPUT_TOKENS,
    MAX_MERGE_OUTPUT_TOKENS,
    OVERLAP_SEGMENTS,
    INTENT_GPU_MEM,
    INTENT_SYSTEM_PROMPT,
    INTENT_MERGE_SYSTEM_PROMPT,
)
from ..core.memory import release_memory, log_vram

logger = logging.getLogger(__name__)


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Robust JSON extraction, validation, and auto-recovery for truncated output."""
    text = text.strip()
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        pass

    # Fallback auto-recovery: repair truncated JSON or unclosed string loops
    try:
        cleaned = text[start:]
        # Deduplicate consecutive repeating word/token loops (3+ times)
        cleaned = re.sub(r'(\S+)(\s+\1){3,}', r'\1', cleaned)

        # Strip trailing unclosed keys or dangling colons/commas before closing braces
        cleaned = re.sub(r':\s*$', ': null', cleaned)
        cleaned = re.sub(r',\s*"[^"]*"\s*:\s*$', '', cleaned)
        cleaned = re.sub(r',\s*$', '', cleaned)

        stack = []
        in_str = False
        esc = False
        for c in cleaned:
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c in "{[":
                    stack.append("}" if c == "{" else "]")
                elif c in "}]":
                    if stack and stack[-1] == c:
                        stack.pop()

        repaired = cleaned
        if in_str:
            repaired += '"'
        while stack:
            repaired += stack.pop()

        obj = json.loads(repaired)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    return None


def empty_result() -> Dict[str, Any]:
    return {
        "speaker_roles": {
            "agent": {"speaker_id": None, "evidence": ""},
            "client": {"speaker_id": None, "evidence": ""},
        },
        "primary_client_intent": "unclear",
        "secondary_intents": [],
        "client_needs_and_requests": [],
        "key_events": [],
        "uncertainties": [],
    }


def ensure_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def deterministic_merge_intent_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Deterministically merges partial intent extraction records when LLM generation truncates or exceeds token limits."""
    merged = empty_result()
    all_needs = []
    all_intents = []
    all_events = []
    all_uncertainties = []

    for r in records:
        if not isinstance(r, dict):
            continue
        p_intent = r.get("primary_client_intent")
        if p_intent and p_intent != "unclear" and merged["primary_client_intent"] == "unclear":
            merged["primary_client_intent"] = p_intent

        roles = r.get("speaker_roles", {})
        if isinstance(roles, dict):
            for role in ("agent", "client"):
                r_info = roles.get(role, {})
                if isinstance(r_info, dict) and r_info.get("speaker_id") and not merged["speaker_roles"][role]["speaker_id"]:
                    merged["speaker_roles"][role] = {
                        "speaker_id": r_info.get("speaker_id"),
                        "evidence": r_info.get("evidence", ""),
                    }

        for need in ensure_list(r.get("client_needs_and_requests")):
            if need and need not in all_needs:
                all_needs.append(need)
        for intent in ensure_list(r.get("secondary_intents")):
            if intent and intent not in all_intents and intent != merged["primary_client_intent"]:
                all_intents.append(intent)
        for evt in ensure_list(r.get("key_events")):
            if evt and evt not in all_events:
                all_events.append(evt)
        for unc in ensure_list(r.get("uncertainties")):
            if unc and unc not in all_uncertainties:
                all_uncertainties.append(unc)

    merged["client_needs_and_requests"] = all_needs
    merged["secondary_intents"] = all_intents
    merged["key_events"] = all_events
    all_uncertainties.append("Assembled via deterministic fallback merge due to output token boundaries.")
    merged["uncertainties"] = all_uncertainties
    return merged


def run_intent_extraction(
    transcript_or_path: Union[str, Path, List[Dict[str, Any]]],
    output_path: str,
    work_dir: Optional[str] = None,
    existing_llm: Optional[Any] = None,
    existing_tokenizer: Optional[Any] = None,
    release_llm_after: bool = False,
    gpu_memory_utilization: float = INTENT_GPU_MEM,
) -> Tuple[Dict[str, Any], Any, Any]:
    """Executes batched intent extraction and hierarchical consolidation."""
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    if isinstance(transcript_or_path, (str, Path)):
        with open(transcript_or_path, "r", encoding="utf-8") as f:
            segments = json.load(f)
    else:
        segments = transcript_or_path

    if not isinstance(segments, list) or not segments:
        raise ValueError("Expected a non-empty JSON list of transcript segments.")

    output_p = Path(output_path).resolve()
    batch_dir = Path(work_dir) if work_dir else output_p.parent / "intent_batches"
    batch_dir.mkdir(parents=True, exist_ok=True)

    valid_speaker_ids = sorted({
        str(s["speaker_id"])
        for s in segments
        if s.get("speaker_id") not in (None, "", "UNKNOWN")
    })

    def format_segment(segment):
        speaker = segment.get("speaker_id") or "UNKNOWN"
        return (
            f"[{float(segment['start']):.2f}-"
            f"{float(segment['end']):.2f}] "
            f"{speaker}: {str(segment.get('text', '')).strip()}"
        )

    lines = [format_segment(s) for s in segments]
    logger.info(f"Intent extraction: {len(segments)} segments, speakers: {valid_speaker_ids}")

    from transformers import AutoTokenizer

    tokenizer = existing_tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, trust_remote_code=True)

    llm = existing_llm
    vllm_llm = None
    hf_llm = None

    if llm is not None:
        if type(llm).__name__ == "LLM":
            vllm_llm = llm
        else:
            hf_llm = llm
    else:
        use_vllm = os.environ.get("USE_VLLM", "0").lower() in ("1", "true", "yes")
        if use_vllm:
            try:
                from vllm import LLM, SamplingParams
                log_vram("Before Qwen Intent LLM Load")
                _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
                try:
                    sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
                    vllm_llm = LLM(
                        model=LLM_MODEL_ID,
                        dtype="half",
                        trust_remote_code=True,
                        max_model_len=INTENT_MAX_MODEL_LEN,
                        max_num_seqs=1,
                        gpu_memory_utilization=gpu_memory_utilization,
                        disable_log_stats=True,
                        enforce_eager=True,
                    )
                finally:
                    sys.stdout, sys.stderr = _orig_stdout, _orig_stderr
                llm = vllm_llm
                logger.info("vLLM Qwen LLM loaded successfully.")
            except Exception as vllm_err:
                logger.warning(
                    f"vLLM engine unavailable ({type(vllm_err).__name__}: {vllm_err}). "
                    "Switching seamlessly to native Hugging Face Transformers Qwen LLM..."
                )
                release_memory()
                vllm_llm = None

        if llm is None:
            from transformers import AutoModelForCausalLM
            hf_llm = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_ID,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
            llm = hf_llm
            logger.info("Hugging Face Qwen LLM loaded successfully.")

    def format_chat(system_message: str, user_message: str) -> str:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    def generate_text(system_prompt: str, user_message: str, max_new_tokens: int, temperature: float = 0.0) -> str:
        formatted_prompt = format_chat(system_prompt, user_message)
        prompt_tokens = count_tokens(formatted_prompt)
        max_allowed = INTENT_MAX_MODEL_LEN - max_new_tokens

        if prompt_tokens > max_allowed:
            if max_new_tokens > 600:
                needed = prompt_tokens - max_allowed
                reduction = min(max_new_tokens - 600, needed)
                max_new_tokens -= reduction
                max_allowed = INTENT_MAX_MODEL_LEN - max_new_tokens

            if prompt_tokens > max_allowed:
                logger.warning(
                    f"Prompt tokens ({prompt_tokens}) exceeds available budget ({max_allowed}). "
                    "Adaptively clamping prompt tokens to stay strictly within context limit..."
                )
                enc = tokenizer.encode(formatted_prompt, add_special_tokens=False)
                formatted_prompt = tokenizer.decode(enc[:max_allowed], skip_special_tokens=False)

        if vllm_llm is not None:
            from vllm import SamplingParams
            params = SamplingParams(
                temperature=temperature,
                max_tokens=max_new_tokens,
                repetition_penalty=1.1,
            )
            outputs = vllm_llm.generate([formatted_prompt], sampling_params=params, use_tqdm=False)
            result = outputs[0].outputs[0]
            if getattr(result, "finish_reason", None) == "length":
                logger.warning("Generation reached max_tokens; output may be truncated.")
            return result.text.strip()
        else:
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(hf_llm.device)
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
            return tokenizer.decode(out_ids[0][input_len:], skip_special_tokens=True).strip()

    def generate_json(
        user_message: str,
        system_prompt: str = INTENT_SYSTEM_PROMPT,
        max_new_tokens: int = MAX_BATCH_OUTPUT_TOKENS,
        retries: int = 1,
    ) -> Dict[str, Any]:
        last_error = None
        curr_msg = user_message
        curr_max = max_new_tokens
        for attempt in range(retries + 1):
            try:
                temp = 0.0 if attempt == 0 else 0.2
                if attempt > 0:
                    curr_max = min(int(max_new_tokens * 1.5), 2000)
                raw = generate_text(system_prompt, curr_msg, max_new_tokens=curr_max, temperature=temp)
                parsed = parse_json_object(raw)
                if parsed is not None:
                    return parsed
                last_error = ValueError(f"Output not valid JSON. Raw:\n{raw[:2000]}")
            except Exception as exc:
                last_error = exc

            if attempt < retries:
                curr_msg += (
                    "\n\nIMPORTANT: Your previous response was either truncated or could not be parsed as valid JSON. "
                    "Return exactly one complete, valid JSON object matching the requested schema. "
                    "Keep the output concise, limit bullet points, and do not use Markdown fences."
                )

        raise RuntimeError(f"JSON generation failed: {last_error}") from last_error

    def create_batches(transcript_lines, max_tokens=MAX_TRANSCRIPT_TOKENS, overlap=OVERLAP_SEGMENTS):
        batches = []
        start = 0
        while start < len(transcript_lines):
            end = start
            batch_lines = []
            while end < len(transcript_lines):
                candidate = batch_lines + [transcript_lines[end]]
                candidate_text = "\n".join(candidate)
                if batch_lines and count_tokens(candidate_text) > max_tokens:
                    break
                batch_lines.append(transcript_lines[end])
                end += 1
                if count_tokens("\n".join(batch_lines)) > max_tokens:
                    break

            if end == start:
                end = start + 1
                batch_lines = [transcript_lines[start]]

            batches.append({
                "batch_number": len(batches) + 1,
                "start_index": start,
                "end_index": end,
                "transcript": "\n".join(batch_lines),
            })

            if end >= len(transcript_lines):
                break
            start = max(start + 1, end - overlap)
        return batches

    try:
        batches = create_batches(lines)
        logger.info(f"Batches created: {len(batches)}")

        batch_results = []
        previous_context = "No previous batch has been analyzed."

        for batch in batches:
            number = batch["batch_number"]
            logger.info(
                f"Processing batch {number}/{len(batches)} (segments {batch['start_index'] + 1}-{batch['end_index']})"
            )

            user_message = f"""
Analyze batch {number} of {len(batches)} from one continuous
customer-support call.

VALID SPEAKER IDS:
{json.dumps(valid_speaker_ids, ensure_ascii=False)}

PREVIOUS BATCH CONTEXT:
{json.dumps(previous_context, ensure_ascii=False)}

CURRENT TRANSCRIPT:
{batch["transcript"]}

TASKS

1. Identify the agent and client from conversational evidence.
   Do not assume that a speaker ID identifies a role.
   If evidence is insufficient, use null.
   The agent and client must not have the same speaker ID.
   Keep role evidence to a short English phrase under 10 words (e.g., "greets caller and offers help").

2. Identify the client's primary intent:
   What is the client trying to accomplish, resolve, obtain,
   change, or understand?
   State a specific goal in English.

3. Identify secondary intents and explicit client needs or requests in English.

4. Record important events, including relevant actions, explanations,
   commitments, outcomes, and unresolved requests.

5. All output values (intents, evidence, events) MUST be in concise ENGLISH only.
   If the conversation is in Tamil, Hindi, or an Indic language, summarize in English.
   Do NOT paste raw non-English words into the JSON. Do not repeat words.

6. Use the previous context only as context. The current transcript
   is the source for quotations and timestamps.
   Overlapping segments may repeat information; do not treat repetition
   as a new event.

7. Do not label intent as unclear if the client's goal can reasonably
   be inferred from the dialogue. Use unclear only when evidence
   is genuinely insufficient.

Return exactly one JSON object with this schema:

{{
  "speaker_roles": {{
    "agent": {{
      "speaker_id": null,
      "evidence": ""
    }},
    "client": {{
      "speaker_id": null,
      "evidence": ""
    }}
  }},
  "primary_client_intent": "unclear",
  "secondary_intents": [],
  "client_needs_and_requests": [],
  "key_events": [],
  "uncertainties": []
}}

For secondary_intents, client_needs_and_requests, key_events,
and uncertainties, use arrays. Keep the response concise but informative.
Return JSON only.
"""
            try:
                result = generate_json(user_message, system_prompt=INTENT_SYSTEM_PROMPT, max_new_tokens=MAX_BATCH_OUTPUT_TOKENS)
            except Exception as b_err:
                logger.warning(f"Batch {number} extraction encountered error ({b_err}). Recovering via fallback template...")
                result = empty_result()
                result["uncertainties"].append(f"Batch {number} exceeded token bounds: {str(b_err)[:80]}")

            normalized = empty_result()
            if isinstance(result.get("speaker_roles"), dict):
                normalized["speaker_roles"].update(result["speaker_roles"])
            for field in (
                "primary_client_intent",
                "secondary_intents",
                "client_needs_and_requests",
                "key_events",
                "uncertainties",
            ):
                if field in result:
                    normalized[field] = result[field]
            result = normalized

            record = {
                "batch_number": number,
                "start_index": batch["start_index"],
                "end_index": batch["end_index"],
                "result": result,
            }
            batch_results.append(record)

            batch_path = batch_dir / f"batch_{number:03d}.json"
            with batch_path.open("w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)

            previous_context = {
                "speaker_roles": result["speaker_roles"],
                "primary_client_intent": result["primary_client_intent"],
                "secondary_intents": result["secondary_intents"],
                "client_needs_and_requests": result["client_needs_and_requests"],
                "key_events": result["key_events"],
                "uncertainties": result["uncertainties"],
            }

        def consolidate_group(records, group_number, total_groups):
            user_message = f"""
Consolidation group {group_number} of {total_groups}.

VALID SPEAKER IDS:
{json.dumps(valid_speaker_ids, ensure_ascii=False)}

PARTIAL ANALYSES:
{json.dumps(records, ensure_ascii=False)}

Merge these analyses into one result. Preserve meaningful details
and evidence. Remove duplicates without dropping distinct requests.

Use this exact schema:

{{
  "speaker_roles": {{
    "agent": {{"speaker_id": null, "evidence": ""}},
    "client": {{"speaker_id": null, "evidence": ""}}
  }},
  "primary_client_intent": "unclear",
  "secondary_intents": [],
  "client_needs_and_requests": [],
  "key_events": [],
  "uncertainties": []
}}

Use null for an unknown role. The agent and client must have
different speaker IDs if both are identified.
Return JSON only.
"""
            try:
                return generate_json(
                    user_message,
                    system_prompt=INTENT_MERGE_SYSTEM_PROMPT,
                    max_new_tokens=MAX_MERGE_OUTPUT_TOKENS,
                    retries=1,
                )
            except Exception as merge_err:
                logger.warning(
                    f"Consolidation group {group_number} LLM merge failed ({merge_err}). "
                    "Assembling via deterministic fallback merge..."
                )
                return deterministic_merge_intent_records(records)

        current_level = [record["result"] for record in batch_results]
        level_number = 1
        while len(current_level) > 1:
            logger.info(f"Aggregation level {level_number}: {len(current_level)} partial results")
            next_level = []
            groups = [current_level[i : i + 3] for i in range(0, len(current_level), 3)]
            for group_index, group in enumerate(groups, start=1):
                merged = consolidate_group(group, group_index, len(groups))
                next_level.append(merged)
                checkpoint = batch_dir / f"aggregation_level_{level_number}_group_{group_index}.json"
                with checkpoint.open("w", encoding="utf-8") as f:
                    json.dump(merged, f, ensure_ascii=False, indent=2)

            current_level = next_level
            level_number += 1

        final_result = current_level[0] if current_level else empty_result()

        def normalize_role(entry, role_name):
            if not isinstance(entry, dict):
                entry = {}
            speaker_id = entry.get("speaker_id")
            if speaker_id in (None, "", "null", "UNKNOWN", "NULL"):
                speaker_id = None
            else:
                speaker_id = str(speaker_id)

            evidence = entry.get("evidence", "")
            if not isinstance(evidence, str):
                evidence = str(evidence or "")

            notes = []
            if speaker_id is not None and speaker_id not in valid_speaker_ids:
                notes.append(
                    f"{role_name} speaker ID {speaker_id!r} was not present in the transcript."
                )
                speaker_id = None

            return {"speaker_id": speaker_id, "evidence": evidence.strip()}, notes

        validation_notes = []
        roles = final_result.get("speaker_roles", {})
        if not isinstance(roles, dict):
            roles = {}

        agent, notes = normalize_role(roles.get("agent"), "Agent")
        validation_notes.extend(notes)
        client, notes = normalize_role(roles.get("client"), "Client")
        validation_notes.extend(notes)

        # Smart role disambiguation & recovery
        valid_list = sorted(list(valid_speaker_ids))
        if valid_list:
            # If both assigned the same speaker, assign the other valid speaker to the client
            if agent["speaker_id"] is not None and agent["speaker_id"] == client["speaker_id"]:
                others = [s for s in valid_list if s != agent["speaker_id"]]
                if others:
                    client["speaker_id"] = others[0]
                    client["evidence"] = "Inferred as caller/client from conversation turns"
                    validation_notes.append(f"Disambiguated client role to {client['speaker_id']}.")
            
            # If one role is identified in a 2-speaker call, deduce the other
            if len(valid_list) == 2:
                s0, s1 = valid_list[0], valid_list[1]
                if agent["speaker_id"] is None and client["speaker_id"] is not None:
                    agent["speaker_id"] = s1 if client["speaker_id"] == s0 else s0
                    agent["evidence"] = "Inferred as agent from conversational turns"
                elif client["speaker_id"] is None and agent["speaker_id"] is not None:
                    client["speaker_id"] = s1 if agent["speaker_id"] == s0 else s0
                    client["evidence"] = "Inferred as caller from conversational turns"

            # If agent is still null, inspect early turns for greetings/support keywords
            if agent["speaker_id"] is None:
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
                agent["speaker_id"] = best_spk
                agent["evidence"] = "Inferred from opening greeting and support turns"
                if client["speaker_id"] is None or client["speaker_id"] == agent["speaker_id"]:
                    rem = [s for s in valid_list if s != agent["speaker_id"]]
                    client["speaker_id"] = rem[0] if rem else None

        uncertainties = ensure_list(final_result.get("uncertainties"))
        uncertainties.extend(validation_notes)

        primary_intent = final_result.get("primary_client_intent", "unclear")
        if not isinstance(primary_intent, str):
            primary_intent = "unclear"

        normalized_result = {
            "speaker_roles": {
                "agent": agent,
                "client": client,
            },
            "primary_client_intent": primary_intent,
            "secondary_intents": ensure_list(final_result.get("secondary_intents")),
            "client_needs_and_requests": ensure_list(final_result.get("client_needs_and_requests")),
            "key_events": ensure_list(final_result.get("key_events")),
            "uncertainties": uncertainties,
        }

        output_p.parent.mkdir(parents=True, exist_ok=True)
        with output_p.open("w", encoding="utf-8") as f:
            json.dump(normalized_result, f, ensure_ascii=False, indent=2)

        logger.info(f"Intent extraction complete -> {output_p.name}")
        return normalized_result, llm, tokenizer

    finally:
        if release_llm_after:
            release_memory(llm, tokenizer)
            log_vram("After Intent LLM Cleanup")
