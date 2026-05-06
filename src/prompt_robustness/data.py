from __future__ import annotations

import random
import re
import json
from typing import Any, Dict, Iterable, List, Optional

from .config import PROJECT_ROOT, get_task_config, load_json_config


def _require_datasets():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required for loading benchmark data. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return load_dataset


def _sample_rows(rows: List[Dict[str, Any]], sample_size: Optional[int], seed: int) -> List[Dict[str, Any]]:
    if sample_size is None or sample_size >= len(rows):
        return rows
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    return [rows[i] for i in indices[:sample_size]]


def _parse_gsm8k_gold(answer: str) -> str:
    match = re.search(r"####\s*(.+?)\s*$", answer)
    if not match:
        raise ValueError(f"Could not parse GSM8K final answer from: {answer!r}")
    return match.group(1).strip()


def load_fixture_examples(task_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    path = PROJECT_ROOT / "data" / "fixtures" / f"{task_id}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No fixture file for task {task_id}: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows[:limit] if limit is not None else rows


def load_frozen_examples(task_id: str, frozen_dir: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    path = frozen_dir / f"{task_id}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No frozen examples for task {task_id}: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows[:limit] if limit is not None else rows


def load_examples(
    task_id: str,
    limit: Optional[int] = None,
    use_fixtures: bool = False,
    frozen_dir: Optional[Path] = None
) -> List[Dict[str, Any]]:
    if use_fixtures:
        return load_fixture_examples(task_id, limit=limit)
    if frozen_dir is not None:
        return load_frozen_examples(task_id, frozen_dir=frozen_dir, limit=limit)

    task = get_task_config(task_id)
    load_dataset = _require_datasets()
    subset = task.get("dataset_subset")
    if subset:
        dataset = load_dataset(task["dataset_name"], subset, split=task["split"])
    else:
        dataset = load_dataset(task["dataset_name"], split=task["split"])

    rows = list(dataset)
    sample_size = limit if limit is not None else task.get("sample_size")
    rows = _sample_rows(rows, sample_size, int(task.get("seed", 17)))

    if task_id == "trec6":
        examples = []
        for idx, row in enumerate(rows):
            label_value = row[task["label_field"]]
            gold = (
                str(label_value)
                if isinstance(label_value, str)
                else dataset.features[task["label_field"]].names[int(label_value)]
            )
            gold_label_id = task["labels"].index(gold) if isinstance(label_value, str) else int(label_value)
            examples.append(
                {
                    "id": f"trec6_{idx:04d}",
                    "task": task_id,
                    "input": row[task["input_field"]],
                    "gold": gold,
                    "metadata": {"gold_label_id": gold_label_id}
                }
            )
        return examples

    if task_id == "gsm8k":
        examples = []
        for idx, row in enumerate(rows):
            examples.append(
                {
                    "id": f"gsm8k_{idx:04d}",
                    "task": task_id,
                    "input": row[task["input_field"]],
                    "gold": _parse_gsm8k_gold(row[task["answer_field"]]),
                    "metadata": {"full_solution": row[task["answer_field"]]}
                }
            )
        return examples

    raise KeyError(f"No loader implemented for task: {task_id}")


def configured_task_ids() -> Iterable[str]:
    return [task["id"] for task in load_json_config("tasks.json")["tasks"]]
