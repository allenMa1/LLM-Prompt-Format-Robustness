from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .config import RUNS_DIR, enabled_models
from .inference import call_model
from .io import read_jsonl, write_jsonl


def _model_lookup() -> Dict[str, Dict[str, Any]]:
    return {model["id"]: model for model in enabled_models()}


def _is_failed(record: Dict[str, Any]) -> bool:
    return record.get("response_status") != "completed" or not str(record.get("raw_output", "")).strip()


def _filter_records(
    records: Iterable[Dict[str, Any]],
    limit: Optional[int],
    tasks: Optional[set[str]],
    models: Optional[set[str]],
    prompts: Optional[set[str]]
) -> list[Dict[str, Any]]:
    selected = []
    seen = set()
    for record in records:
        if not _is_failed(record):
            continue
        if tasks is not None and record["task"] not in tasks:
            continue
        if models is not None and record["model"] not in models:
            continue
        if prompts is not None and record["prompt_id"] not in prompts:
            continue
        key = (record["example_id"], record["task"], record["model"], record["prompt_id"])
        if key in seen:
            continue
        seen.add(key)
        selected.append(record)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def retry_failed(
    input_path: Path,
    output_path: Path,
    limit: Optional[int],
    tasks: Optional[set[str]],
    models: Optional[set[str]],
    prompts: Optional[set[str]],
    sleep_seconds: float,
    log_file: Optional[Path]
) -> None:
    if output_path.exists():
        raise FileExistsError(f"Output already exists. Pick a new path or delete it: {output_path}")

    model_by_id = _model_lookup()
    failed_records = _filter_records(read_jsonl(input_path), limit=limit, tasks=tasks, models=models, prompts=prompts)
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

    log(f"[retry-start] selected={len(failed_records)} input={input_path} output={output_path}")

    output_records = []
    started_at = time.time()
    try:
        for idx, record in enumerate(failed_records, start=1):
            model_id = record["model"]
            if model_id not in model_by_id:
                raise KeyError(f"Model is not enabled in configs/models.json: {model_id}")
            model = model_by_id[model_id]
            log(
                f"[call-start] n={idx} task={record['task']} example={record['example_id']} "
                f"prompt={record['prompt_id']} model={model_id}"
            )
            call_start = time.time()
            response_data = call_model(model, record["raw_prompt"], dry_run=False)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            call_elapsed = time.time() - call_start
            elapsed = time.time() - started_at
            preview = response_data["raw_output"].replace("\n", "\\n")[:80]
            log(
                f"[call-done] n={idx} status={response_data['response_status']} "
                f"call_s={call_elapsed:.1f} elapsed_s={elapsed:.1f} output={preview!r}"
            )

            retried = dict(record)
            retried["raw_output"] = response_data["raw_output"]
            retried["response_status"] = response_data["response_status"]
            retried["incomplete_details"] = response_data["incomplete_details"]
            retried["usage"] = response_data["usage"]
            retried["created_at"] = datetime.now(timezone.utc).isoformat()
            retried["call_elapsed_s"] = call_elapsed
            retried["retry_of"] = {
                "input_path": str(input_path),
                "previous_created_at": record.get("created_at"),
                "previous_response_status": record.get("response_status"),
                "previous_incomplete_details": record.get("incomplete_details")
            }
            retried["model_params"] = {
                "temperature": model.get("temperature"),
                "max_output_tokens": model.get("max_output_tokens"),
                "reasoning_effort": model.get("reasoning_effort")
            }
            output_records.append(retried)

        write_jsonl(output_path, output_records)
        log(f"[done] retried={len(output_records)} elapsed_s={time.time() - started_at:.1f}")
    finally:
        if log_handle is not None:
            log_handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Retry failed or empty inference records from a raw JSONL file.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=RUNS_DIR / "raw_outputs" / "retry_failed.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--prompts", nargs="*", default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args()

    retry_failed(
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
        tasks=set(args.tasks) if args.tasks else None,
        models=set(args.models) if args.models else None,
        prompts=set(args.prompts) if args.prompts else None,
        sleep_seconds=args.sleep_seconds,
        log_file=args.log_file
    )


if __name__ == "__main__":
    main()
