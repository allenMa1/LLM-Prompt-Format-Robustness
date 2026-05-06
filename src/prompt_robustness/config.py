from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
RUNS_DIR = PROJECT_ROOT / "runs"


def load_json_config(name: str) -> Dict[str, Any]:
    path = CONFIG_DIR / name
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def get_task_config(task_id: str) -> Dict[str, Any]:
    for task in load_json_config("tasks.json")["tasks"]:
        if task["id"] == task_id:
            return task
    raise KeyError(f"Unknown task id: {task_id}")


def enabled_models() -> list[Dict[str, Any]]:
    return [m for m in load_json_config("models.json")["models"] if m.get("enabled", True)]
