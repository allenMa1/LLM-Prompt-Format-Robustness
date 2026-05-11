from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .config import PROJECT_ROOT, RUNS_DIR
from .io import read_jsonl, write_jsonl


TREC_LABEL_DEFINITIONS = {
    "ABBR": "abbreviation",
    "ENTY": "entity, object, or thing",
    "DESC": "description, definition, or explanation",
    "HUM": "human, person, or group",
    "LOC": "location or place",
    "NUM": "number, quantity, date, or other numeric answer",
}


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env")


def _build_trec_prompt(record: Dict[str, Any]) -> str:
    labels = "\n".join(f"- {label}: {definition}" for label, definition in TREC_LABEL_DEFINITIONS.items())
    return f"""You are auditing a TREC-6 question classification evaluation.

The task is to classify a question by the expected answer type.

Valid labels:
{labels}

Question:
{record["input"]}

Gold label:
{record["gold"]}

Model output:
{record["raw_output"]}

Decide whether the model output is semantically equivalent to the gold label.
Ignore harmless formatting differences such as quotes, punctuation, JSON, XML tags, or phrases like "The answer is".
Do not give credit if the model output corresponds to a different label.

Return JSON only in this exact shape:
{{"equivalent": true}}
"""


def _build_gsm8k_prompt(record: Dict[str, Any]) -> str:
    return f"""You are auditing a GSM8K short-answer math evaluation.

Problem:
{record["input"]}

Gold final answer:
{record["gold"]}

Model output:
{record["raw_output"]}

Decide whether the model output gives the same final numeric answer as the gold answer.
Ignore harmless formatting differences such as commas, currency symbols, trailing .0, JSON, XML tags, or explanatory text.
Do not give credit for a different final number, even if the reasoning seems partially correct.

Return JSON only in this exact shape:
{{"equivalent": true}}
"""


def build_judge_prompt(record: Dict[str, Any]) -> str:
    if record["task"] == "trec6":
        return _build_trec_prompt(record)
    if record["task"] == "gsm8k":
        return _build_gsm8k_prompt(record)
    raise KeyError(f"No judge prompt implemented for task: {record['task']}")


def _dump_openai_obj(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return str(value)


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def parse_judge_equivalent(text: str) -> Optional[bool]:
    parsed = _extract_json_object(text.strip())
    if not parsed or "equivalent" not in parsed:
        return None
    value = parsed["equivalent"]
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None


def call_openai_judge(model: str, prompt: str, max_output_tokens: int) -> Dict[str, Any]:
    _load_dotenv()
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The 'openai' package is required for LLM judge calls.") from exc

    client = OpenAI()
    response = client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=max_output_tokens,
    )
    output_text = response.output_text if getattr(response, "output_text", None) is not None else ""
    return {
        "judge_raw_output": output_text,
        "judge_response_status": getattr(response, "status", None),
        "judge_incomplete_details": _dump_openai_obj(getattr(response, "incomplete_details", None)),
        "judge_usage": _dump_openai_obj(getattr(response, "usage", None)),
    }


def _select_records(records: list[Dict[str, Any]], mode: str, limit: Optional[int]) -> list[Dict[str, Any]]:
    if mode == "strict_false":
        selected = [record for record in records if not bool(record.get("strict_correct"))]
    elif mode == "strict_false_equiv_false":
        selected = [
            record
            for record in records
            if not bool(record.get("strict_correct")) and not bool(record.get("equiv_correct"))
        ]
    elif mode == "strict_false_equiv_true":
        selected = [
            record
            for record in records
            if not bool(record.get("strict_correct")) and bool(record.get("equiv_correct"))
        ]
    elif mode == "all":
        selected = records
    else:
        raise ValueError(f"Unknown selection mode: {mode}")

    return selected[:limit] if limit is not None else selected


def run_judge_audit(
    input_path: Path,
    output_path: Path,
    judge_model: str,
    selection: str,
    limit: Optional[int],
    max_output_tokens: int,
    sleep_seconds: float,
    log_file: Optional[Path],
) -> None:
    if output_path.exists():
        raise FileExistsError(f"Output already exists. Pick a new path or delete it: {output_path}")

    records = list(read_jsonl(input_path))
    selected = _select_records(records, mode=selection, limit=limit)

    log_handle = None

    def log(message: str) -> None:
        print(message, flush=True)
        if log_handle is not None:
            log_handle.write(message + "\n")
            log_handle.flush()

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        if log_file.exists():
            raise FileExistsError(f"Log file already exists. Pick a new path or delete it: {log_file}")
        log_handle = log_file.open("w", encoding="utf-8")

    results = []
    started_at = time.time()
    try:
        log(
            f"[judge-start] input={input_path} selected={len(selected)} "
            f"selection={selection} judge_model={judge_model}"
        )
        for idx, record in enumerate(selected, start=1):
            prompt = build_judge_prompt(record)
            log(
                f"[call-start] n={idx} task={record['task']} model={record['model']} "
                f"example={record['example_id']} prompt={record['prompt_id']}"
            )
            call_start = time.time()
            judge_data = call_openai_judge(
                model=judge_model,
                prompt=prompt,
                max_output_tokens=max_output_tokens,
            )
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            call_elapsed = time.time() - call_start
            judge_equiv = parse_judge_equivalent(judge_data["judge_raw_output"])
            preview = judge_data["judge_raw_output"].replace("\n", "\\n")[:100]
            log(
                f"[call-done] n={idx} status={judge_data['judge_response_status']} "
                f"judge_equiv={judge_equiv} call_s={call_elapsed:.1f} output={preview!r}"
            )

            judged = dict(record)
            judged.update(judge_data)
            judged["judge_model"] = judge_model
            judged["judge_prompt"] = prompt
            judged["judge_equiv_correct"] = judge_equiv
            judged["judge_created_at"] = datetime.now(timezone.utc).isoformat()
            judged["judge_call_elapsed_s"] = call_elapsed
            judged["judge_selection"] = selection
            results.append(judged)

            if len(results) >= 50:
                write_jsonl(output_path, results, append=True)
                log(f"[flush] wrote_batch={len(results)} output={output_path}")
                results.clear()

        if results:
            write_jsonl(output_path, results, append=True)
            log(f"[flush] wrote_batch={len(results)} output={output_path}")
        log(f"[done] selected={len(selected)} elapsed_s={time.time() - started_at:.1f} output={output_path}")
    finally:
        if log_handle is not None:
            log_handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an LLM-judge equivalence audit over scored outputs.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=RUNS_DIR / "scores" / "main_clean_judge_audit_gpt54.jsonl")
    parser.add_argument("--judge-model", default="gpt-5.4")
    parser.add_argument(
        "--selection",
        choices=["strict_false", "strict_false_equiv_false", "strict_false_equiv_true", "all"],
        default="strict_false",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--log-file", type=Path, default=RUNS_DIR / "logs" / "main_clean_judge_audit_gpt54.log")
    args = parser.parse_args()

    run_judge_audit(
        input_path=args.input,
        output_path=args.output,
        judge_model=args.judge_model,
        selection=args.selection,
        limit=args.limit,
        max_output_tokens=args.max_output_tokens,
        sleep_seconds=args.sleep_seconds,
        log_file=args.log_file,
    )


if __name__ == "__main__":
    main()
