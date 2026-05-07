from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .config import PROJECT_ROOT, RUNS_DIR, enabled_models, load_json_config
from .data import configured_task_ids, load_examples
from .io import write_jsonl
from .prompts import get_prompt_variants, render_prompt


def _dump_openai_obj(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return str(value)


def call_model(model: Dict[str, Any], prompt: str, dry_run: bool = False) -> Dict[str, Any]:
    if dry_run:
        return {
            "raw_output": "",
            "response_status": "dry_run",
            "incomplete_details": None,
            "usage": None
        }

    if model["provider"] == "openai":
        return _call_openai(model, prompt)

    raise KeyError(f"Unsupported provider: {model['provider']}")


def _call_openai(model: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None
    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The 'openai' package is required for inference. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    client = OpenAI()
    kwargs: Dict[str, Any] = {
        "model": model["id"],
        "input": prompt,
        "max_output_tokens": model.get("max_output_tokens", 256)
    }
    if model.get("temperature") is not None:
        kwargs["temperature"] = model["temperature"]
    if model.get("reasoning_effort") is not None:
        kwargs["reasoning"] = {"effort": model["reasoning_effort"]}

    response = client.responses.create(**kwargs)
    output_text = response.output_text if getattr(response, "output_text", None) is not None else ""
    metadata = {
        "response_status": getattr(response, "status", None),
        "incomplete_details": _dump_openai_obj(getattr(response, "incomplete_details", None)),
        "usage": _dump_openai_obj(getattr(response, "usage", None))
    }
    if hasattr(response, "output_text") and response.output_text is not None:
        metadata["raw_output"] = output_text
    else:
        metadata["raw_output"] = str(response)
    return metadata


def _model_by_ids(model_ids: Optional[Iterable[str]]) -> list[Dict[str, Any]]:
    models = enabled_models()
    if model_ids is None:
        return models
    wanted = set(model_ids)
    selected = [model for model in models if model["id"] in wanted]
    missing = wanted.difference({model["id"] for model in selected})
    if missing:
        raise KeyError(f"Unknown or disabled model ids: {sorted(missing)}")
    return selected


def run_inference(
    task_ids: Iterable[str],
    model_ids: Optional[Iterable[str]],
    prompt_ids: Optional[Iterable[str]],
    limit: Optional[int],
    output_path: Path,
    dry_run: bool = False,
    frozen_dir: Optional[Path] = None,
    sleep_seconds: float = 0.0,
    log_file: Optional[Path] = None
) -> None:
    models = _model_by_ids(model_ids)
    variants = get_prompt_variants(prompt_ids)
    records = []
    completed = 0
    started_at = time.time()
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

    try:
        for task_id in task_ids:
            examples = load_examples(task_id, limit=limit, use_fixtures=dry_run, frozen_dir=frozen_dir)
            total_for_task = len(examples) * len(variants) * len(models)
            log(
                f"[task-start] task={task_id} examples={len(examples)} "
                f"prompts={len(variants)} models={len(models)} calls={total_for_task}"
            )
            for example in examples:
                for variant in variants:
                    prompt = render_prompt(task_id, example, variant)
                    for model in models:
                        completed += 1
                        log(
                            f"[call-start] n={completed} task={task_id} example={example['id']} "
                            f"prompt={variant['id']} model={model['id']}"
                        )
                        started = datetime.now(timezone.utc).isoformat()
                        call_started_at = time.time()
                        response_data = call_model(model, prompt, dry_run=dry_run)
                        if not dry_run and sleep_seconds > 0:
                            time.sleep(sleep_seconds)
                        call_elapsed = time.time() - call_started_at
                        elapsed = time.time() - started_at
                        output_preview = response_data["raw_output"].replace("\n", "\\n")[:80]
                        log(
                            f"[call-done] n={completed} status={response_data['response_status']} "
                            f"call_s={call_elapsed:.1f} elapsed_s={elapsed:.1f} output={output_preview!r}"
                        )

                        records.append(
                            {
                                "example_id": example["id"],
                                "task": task_id,
                                "model": model["id"],
                                "prompt_id": variant["id"],
                                "wording": variant["wording"],
                                "output_format": variant["output_format"],
                                "input": example["input"],
                                "gold": example["gold"],
                                "raw_prompt": prompt,
                                "raw_output": response_data["raw_output"],
                                "response_status": response_data["response_status"],
                                "incomplete_details": response_data["incomplete_details"],
                                "usage": response_data["usage"],
                                "model_params": {
                                    "temperature": model.get("temperature"),
                                    "max_output_tokens": model.get("max_output_tokens"),
                                    "reasoning_effort": model.get("reasoning_effort")
                                },
                                "created_at": started,
                                "call_elapsed_s": call_elapsed
                            }
                        )

                        if len(records) >= 50:
                            write_jsonl(output_path, records, append=True)
                            log(f"[flush] wrote_batch=50 output={output_path}")
                            records.clear()

        if records:
            write_jsonl(output_path, records, append=True)
            log(f"[flush] wrote_batch={len(records)} output={output_path}")
        log(f"[done] calls={completed} elapsed_s={time.time() - started_at:.1f} output={output_path}")
    finally:
        if log_handle is not None:
            log_handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM inference for prompt robustness experiments.")
    parser.add_argument("--tasks", nargs="*", default=list(configured_task_ids()))
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--prompts", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=RUNS_DIR / "raw_outputs" / "outputs.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--frozen-dir", type=Path, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Output already exists. Pick a new path or delete it: {args.output}")

    _ = load_json_config("tasks.json")
    run_inference(
        task_ids=args.tasks,
        model_ids=args.models,
        prompt_ids=args.prompts,
        limit=args.limit,
        output_path=args.output,
        dry_run=args.dry_run,
        frozen_dir=args.frozen_dir,
        sleep_seconds=args.sleep_seconds,
        log_file=args.log_file
    )


if __name__ == "__main__":
    main()
