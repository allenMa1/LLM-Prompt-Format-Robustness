from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .config import RUNS_DIR, enabled_models, load_json_config
from .data import configured_task_ids, load_examples
from .io import write_jsonl
from .prompts import get_prompt_variants, render_prompt


def _call_openai(model: Dict[str, Any], prompt: str) -> str:
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

    response = client.responses.create(**kwargs)
    if hasattr(response, "output_text") and response.output_text is not None:
        return response.output_text
    return str(response)


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
    sleep_seconds: float = 0.0
) -> None:
    models = _model_by_ids(model_ids)
    variants = get_prompt_variants(prompt_ids)
    records = []

    for task_id in task_ids:
        examples = load_examples(task_id, limit=limit, use_fixtures=dry_run, frozen_dir=frozen_dir)
        for example in examples:
            for variant in variants:
                prompt = render_prompt(task_id, example, variant)
                for model in models:
                    started = datetime.now(timezone.utc).isoformat()
                    if dry_run:
                        output = ""
                    else:
                        output = _call_openai(model, prompt)
                        if sleep_seconds > 0:
                            time.sleep(sleep_seconds)

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
                            "raw_output": output,
                            "model_params": {
                                "temperature": model.get("temperature"),
                                "max_output_tokens": model.get("max_output_tokens")
                            },
                            "created_at": started
                        }
                    )

                    if len(records) >= 50:
                        write_jsonl(output_path, records, append=True)
                        records.clear()

    if records:
        write_jsonl(output_path, records, append=True)


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
        sleep_seconds=args.sleep_seconds
    )


if __name__ == "__main__":
    main()
