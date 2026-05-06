from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

from .config import PROJECT_ROOT
from .data import configured_task_ids, load_examples
from .io import write_jsonl


def freeze_examples(task_ids: Iterable[str], limit: Optional[int], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_records = []

    for task_id in task_ids:
        output_path = out_dir / f"{task_id}.jsonl"
        if output_path.exists():
            raise FileExistsError(f"Frozen file already exists: {output_path}")

        examples = load_examples(task_id, limit=limit)
        write_jsonl(output_path, examples)
        manifest_records.append(
            {
                "task": task_id,
                "path": str(output_path),
                "num_examples": len(examples),
                "limit": limit
            }
        )

    manifest_path = out_dir / "manifest.jsonl"
    if manifest_path.exists():
        raise FileExistsError(f"Manifest already exists: {manifest_path}")
    write_jsonl(manifest_path, manifest_records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze sampled benchmark examples to JSONL files.")
    parser.add_argument("--tasks", nargs="*", default=list(configured_task_ids()))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data" / "frozen" / "main")
    args = parser.parse_args()

    freeze_examples(task_ids=args.tasks, limit=args.limit, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
