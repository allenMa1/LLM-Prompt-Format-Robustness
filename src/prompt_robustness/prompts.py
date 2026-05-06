from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .config import load_json_config


def get_prompt_variants(prompt_ids: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
    config = load_json_config("prompts.json")
    variants = config["variants"]
    if prompt_ids is None:
        return variants
    wanted = set(prompt_ids)
    selected = [variant for variant in variants if variant["id"] in wanted]
    missing = wanted.difference({variant["id"] for variant in selected})
    if missing:
        raise KeyError(f"Unknown prompt ids: {sorted(missing)}")
    return selected


def render_prompt(task_id: str, example: Dict[str, Any], variant: Dict[str, Any]) -> str:
    config = load_json_config("prompts.json")
    wording = config["wordings"][variant["wording"]][task_id]
    output_instruction = config["formats"][variant["output_format"]][task_id]

    if task_id == "trec6":
        body_label = "Question"
    elif task_id == "gsm8k":
        body_label = "Problem"
    else:
        raise KeyError(f"No prompt renderer implemented for task: {task_id}")

    return (
        f"{wording}\n\n"
        f"{body_label}: {example['input']}\n\n"
        f"{output_instruction}"
    )
