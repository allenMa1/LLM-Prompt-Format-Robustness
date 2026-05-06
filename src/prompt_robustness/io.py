from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator

from .config import ensure_parent


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]], append: bool = False) -> None:
    ensure_parent(path)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
